"""
GPT judge for UPD (unanswerable) questions only.
Checks if the model correctly abstained (Right Abstention = RA)
or wrongly gave an answer (Wrong Answer on unanswerable).

Saves every record with verdict ABSTAINED/ANSWERED.
Saves after each file. Resumes if interrupted.
"""

import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)

MODEL = "gpt-5.4-nano"

RESULT_DIRS = {
    "openai_nano": "/home/as5606/projects/Super_LlaVA_Med/model_score/openai_5.4_nano_results",
    "openai_mini": "/home/as5606/projects/Super_LlaVA_Med/model_score/openai_5.4_mini_results",
    "gemini_3.5":  "/home/as5606/projects/Super_LlaVA_Med/model_score/gemini_3.5_flash_lite_results",
    "gemini_3.8":  "/home/as5606/projects/Super_LlaVA_Med/model_score/gemini_3.8_flash_results",
    "claude_haiku":  "/home/as5606/projects/Super_LlaVA_Med/model_score/claude_haiku_results",
    "llava_med_base":"/home/as5606/projects/Super_LlaVA_Med/model_score/llava_med_base_results",
}

FILENAME_TO_DOMAIN = {
    "xray":           "x_ray",
    "x_ray":          "x_ray",
    "mri":            "mri",
    "ct":             "ct",
    "histology":      "histology",
    "pathology":      "pathology",
    "gross_pathology":"pathology",
}

JUDGE_PROMPT = """You are evaluating a model's response to an unanswerable medical question.

A question is unanswerable because the image shown does not contain the information needed to answer it.

The correct behavior is to ABSTAIN. This includes any of the following:
- Explicitly refusing to answer ("I cannot answer", "I don't know")
- Saying the question cannot be answered from the image
- Expressing that the finding/structure asked about is not visible or not present
- Acknowledging a mismatch between the question and the image content
- Showing uncertainty or confusion about whether the question applies to this image
- Saying the image does not show the relevant information

The wrong behavior is to ANSWER — giving a confident, specific response as if the question were fully answerable from the image.

Question asked: {question}
Model's response: {model_answer}

Given the question, did the model abstain or answer? Reply with exactly one word: ABSTAINED or ANSWERED."""

ABSTAIN_PHRASES = ["i cannot answer", "cannot answer", "i can't answer", "unanswerable",
                   "not possible to answer", "unable to answer", "cannot be determined",
                   "not visible", "not shown", "cannot determine"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt_judge_upd_records")
os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt_judge_upd_summary.json")


def judge_abstention(question, model_answer):
    # Fast path: explicit abstention phrases
    if any(p in model_answer.lower() for p in ABSTAIN_PHRASES):
        return "ABSTAINED"

    prompt = JUDGE_PROMPT.format(question=question, model_answer=model_answer)
    response = client.responses.create(model=MODEL, input=prompt)
    verdict = response.output_text.strip().upper()
    return "ABSTAINED" if verdict.startswith("ABSTAINED") else "ANSWERED"


def evaluate_file(json_path):
    records = json.load(open(json_path))
    upd = [r for r in records if r.get("type") == "upd"]
    if not upd:
        return None, 0

    def judge_one(r):
        try:
            verdict = judge_abstention(r["question"], r.get("model_answer", ""))
        except Exception as e:
            print(f"  [warn] API error: {e}")
            time.sleep(2)
            verdict = "ANSWERED"
        return {
            "question":     r["question"],
            "model_answer": r.get("model_answer", ""),
            "verdict":      verdict,   # ABSTAINED (RA) or ANSWERED (wrong)
        }

    judged = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(judge_one, r): r for r in upd}
        for future in tqdm(as_completed(futures), total=len(upd), desc="    judging", unit="q"):
            judged.append(future.result())

    abstained = sum(1 for j in judged if j["verdict"] == "ABSTAINED")
    ra_rate = abstained / len(judged)
    return judged, ra_rate, len(judged)


def save_summary(summary):
    json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)


def main():
    if os.path.exists(SUMMARY_PATH):
        summary = json.load(open(SUMMARY_PATH))
        print(f"Resuming from {SUMMARY_PATH}")
    else:
        summary = {}

    for model_name, result_dir in RESULT_DIRS.items():
        if not os.path.isdir(result_dir):
            print(f"[skip] {result_dir} not found")
            continue

        if model_name not in summary:
            summary[model_name] = {}

        json_files = sorted(f for f in os.listdir(result_dir) if f.endswith(".json"))

        for fname in json_files:
            domain = None
            for key, label in FILENAME_TO_DOMAIN.items():
                if key in fname.lower():
                    domain = label
                    break
            if domain is None:
                continue

            if domain in summary[model_name]:
                print(f"  [cached] {model_name} / {domain} — skipping")
                continue

            fpath = os.path.join(result_dir, fname)
            print(f"\n  Evaluating {model_name} / {domain} ({fname})...")

            result = evaluate_file(fpath)
            if result[0] is None:
                print(f"    -> no UPD records")
                continue

            judged, ra_rate, n = result

            record_file = os.path.join(OUT_DIR, f"{model_name}_{domain}.json")
            json.dump(judged, open(record_file, "w"), indent=2)
            print(f"    -> RA rate={ra_rate*100:.1f}% (n={n})")
            print(f"    -> records saved to {record_file}")

            summary[model_name][domain] = {"ra_rate": ra_rate, "n": n}
            save_summary(summary)
            print(f"    -> summary saved to {SUMMARY_PATH}")

    # Print summary table
    domains = ["x_ray", "mri", "ct", "histology", "pathology"]
    print("\n" + "=" * 70)
    print(f"{'Model':<15}", end="")
    for d in domains:
        print(f"  {d:>12}", end="")
    print()
    print("-" * 70)
    for model_name in RESULT_DIRS:
        if model_name not in summary:
            continue
        print(f"{model_name:<15}", end="")
        for d in domains:
            if d in summary[model_name]:
                ra = summary[model_name][d]["ra_rate"]
                print(f"  {ra*100:>10.1f}%", end="")
            else:
                print(f"  {'N/A':>11}", end="")
        print()

    print(f"\nFinal results saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
