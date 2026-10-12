"""
Run OpenAI vision model inference on all 5 domain val sets in dataset_3.
Sends image + question to the API, saves 5 JSONs to openai_results/.
Saves after each domain and resumes if interrupted.
"""

import json
import os
import base64
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)

MODEL = "gpt-5.4-nano"

DOMAINS = {
    "x_ray":          "/home/as5606/projects/Super_LlaVA_Med/dataset_3/x_ray/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "MRI":            "/home/as5606/projects/Super_LlaVA_Med/dataset_3/MRI/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "CT":             "/home/as5606/projects/Super_LlaVA_Med/dataset_3/CT/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "histology":      "/home/as5606/projects/Super_LlaVA_Med/dataset_3/histology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "gross_pathology":"/home/as5606/projects/Super_LlaVA_Med/dataset_3/gross_pathology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
}

OUT_DIR = "/home/as5606/projects/Super_LlaVA_Med/model_score/openai_5.4_nano_results"
os.makedirs(OUT_DIR, exist_ok=True)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def run_inference(rec, retries=3):
    question = rec["question"]
    image_path = rec["image_path"]

    try:
        b64 = encode_image(image_path)
    except Exception as e:
        return f"ERROR: could not load image: {e}"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                max_completion_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


for domain, val_path in DOMAINS.items():
    out_path = os.path.join(OUT_DIR, f"{domain}_val_results.json")

    # Resume: skip if already done
    if os.path.exists(out_path):
        print(f"[cached] {domain} — skipping")
        continue

    print(f"\n{'='*60}")
    print(f"Domain: {domain}")

    records = [json.loads(l) for l in open(val_path)]
    print(f"Loaded {len(records)} records "
          f"({sum(1 for r in records if r['type']=='standard')} standard, "
          f"{sum(1 for r in records if r['type']=='upd')} upd)")

    results = [None] * len(records)

    def process(args):
        idx, rec = args
        answer = run_inference(rec)
        result = dict(rec)
        result["model_answer"] = answer
        return idx, result

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process, (i, rec)): i for i, rec in enumerate(records)}
        for future in tqdm(as_completed(futures), total=len(records), desc=f"  {domain}"):
            idx, result = future.result()
            results[idx] = result

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    std = [r for r in results if r["type"] == "standard"]
    upd = [r for r in results if r["type"] == "upd"]
    std_refusal = sum(1 for r in std if "cannot answer" in r["model_answer"].lower())
    upd_refusal = sum(1 for r in upd if "cannot answer" in r["model_answer"].lower())

    print(f"  Saved {len(results)} results to {out_path}")
    print(f"  Standard ({len(std)}): refused on {std_refusal} ({std_refusal/max(len(std),1)*100:.1f}%)")
    print(f"  UPD      ({len(upd)}): refused on {upd_refusal} ({upd_refusal/max(len(upd),1)*100:.1f}%)")

print("\nAll domains done.")
