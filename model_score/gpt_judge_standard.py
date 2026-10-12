"""
GPT judge for standard (answerable) questions only.
Saves every record with question, ground_truth, predicted, verdict (CORRECT/INCORRECT).
Saves after each file completes. Resumes if interrupted.
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
    "x_ray":       "/home/as5606/projects/Super_LlaVA_Med/model_score/x_ray_results",
    "mri":         "/home/as5606/projects/Super_LlaVA_Med/model_score/mri_results",
    "ct":          "/home/as5606/projects/Super_LlaVA_Med/model_score/ct_results",
    "histology":   "/home/as5606/projects/Super_LlaVA_Med/model_score/histology_results",
    "pathology":   "/home/as5606/projects/Super_LlaVA_Med/model_score/pathology_results",
    "combined":    "/home/as5606/projects/Super_LlaVA_Med/model_score/combined_results",
    "openai_nano":   "/home/as5606/projects/Super_LlaVA_Med/model_score/openai_5.4_nano_results",
    "openai_mini":   "/home/as5606/projects/Super_LlaVA_Med/model_score/openai_5.4_mini_results",
    "gemini_3.5":    "/home/as5606/projects/Super_LlaVA_Med/model_score/gemini_3.5_flash_lite_results",
    "gemini_3.8":    "/home/as5606/projects/Super_LlaVA_Med/model_score/gemini_3.8_flash_results",
    "claude_haiku":  "/home/as5606/projects/Super_LlaVA_Med/model_score/claude_haiku_results",
    "llava_med_base":"/home/as5606/projects/Super_LlaVA_Med/model_score/llava_med_base_results",
}

FILENAME_TO_DOMAIN = {
    "xray":           "x_ray",
    "x_ray":          "x_ray",
    "mri":            "mri",
    "ct":             "ct",
    "histology":      "histology",
    "gross_pathology":"pathology",
    "pathology":      "pathology",
}

JUDGE_PROMPT = """You are a medical VQA evaluator. Given a question, the ground truth answer, and a model's answer, decide if the model's answer is CORRECT or INCORRECT.

Be lenient: if the model captures the key medical finding or concept correctly, even if phrased differently, mark it CORRECT.

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Reply with exactly one word: CORRECT or INCORRECT."""

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt_judge_records")
os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt_judge_summary.json")


ABSTAIN_PHRASES = ["i cannot answer", "cannot answer", "i can't answer", "unanswerable"]

def judge_answer(question, ground_truth, model_answer):
    # Auto-mark abstentions as INCORRECT (Wrong Abstention on answerable question)
    if any(p in model_answer.lower() for p in ABSTAIN_PHRASES):
        return "INCORRECT"

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )
    response = client.responses.create(model=MODEL, input=prompt)
    verdict = response.output_text.strip().upper()
    # Must start with CORRECT, not just contain it (avoids "INCORRECT" matching)
    return "CORRECT" if verdict.startswith("CORRECT") else "INCORRECT"


def evaluate_file(json_path):
    records = json.load(open(json_path))
    standard = [r for r in records if r.get("type") == "standard"]
    if not standard:
        return None, 0

    def judge_one(r):
        try:
            verdict = judge_answer(r["question"], r["answer"], r.get("model_answer", ""))
        except Exception as e:
            print(f"  [warn] API error: {e}")
            time.sleep(2)
            verdict = "INCORRECT"
        return {
            "question":     r["question"],
            "ground_truth": r["answer"],
            "predicted":    r.get("model_answer", ""),
            "verdict":      verdict,
        }

    judged = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(judge_one, r): r for r in standard}
        for future in tqdm(as_completed(futures), total=len(standard), desc="    judging", unit="q"):
            judged.append(future.result())

    correct = sum(1 for j in judged if j["verdict"] == "CORRECT")
    acc = correct / len(judged)
    return judged, acc, len(judged)


def save_summary(summary):
    json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)


def main():
    # Load existing summary for resume
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
                print(f"  [skip] unrecognized file: {fname}")
                continue

            if domain in summary[model_name]:
                print(f"  [cached] {model_name} / {domain} — skipping")
                continue

            fpath = os.path.join(result_dir, fname)
            print(f"\n  Evaluating {model_name} / {domain} ({fname})...")

            result = evaluate_file(fpath)
            if result[0] is None:
                print(f"    -> no standard records")
                continue

            judged, acc, n = result

            # Save per-record file
            record_file = os.path.join(OUT_DIR, f"{model_name}_{domain}.json")
            json.dump(judged, open(record_file, "w"), indent=2)
            print(f"    -> accuracy={acc*100:.1f}% (n={n})")
            print(f"    -> records saved to {record_file}")

            # Update and save summary
            summary[model_name][domain] = {"accuracy": acc, "n": n}
            save_summary(summary)
            print(f"    -> summary saved to {SUMMARY_PATH}")

    # Print final table
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
                acc = summary[model_name][d]["accuracy"]
                print(f"  {acc*100:>10.1f}%", end="")
            else:
                print(f"  {'N/A':>11}", end="")
        print()


if __name__ == "__main__":
    main()
