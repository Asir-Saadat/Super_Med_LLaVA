"""
Evaluate med-LLaVA on the Super_LlaVA_Med val sets.

Usage:
  python llava_eval.py --domain x_ray --type both
  python llava_eval.py --domain all --type standard
  python llava_eval.py --domain mri --type upd

Args:
  --domain   : x_ray | mri | histology | gross_pathology | ct | all
  --type     : standard (answerable) | upd (unanswerable) | both
  --max       : max samples per domain/type (default: all)
  --output   : path to save per-sample results JSON (optional)
"""

import argparse
import json
import os
import sys
import string
from collections import Counter

import torch
from PIL import Image

# ── Resolve paths ──────────────────────────────────────────────────────────────
LLAVA_MED_ROOT  = "/home/as5606/projects/LLaVA-Med"
MODEL_PATH      = f"{LLAVA_MED_ROOT}/models/llava-med-v1.5-mistral-7b"
DATA_ROOT       = "/home/as5606/projects/Super_LlaVA_Med/dataset"

sys.path.insert(0, LLAVA_MED_ROOT)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

VAL_FILES = {
    "x_ray":           f"{DATA_ROOT}/x_ray/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "mri":             f"{DATA_ROOT}/MRI/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "histology":       f"{DATA_ROOT}/histology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "gross_pathology": f"{DATA_ROOT}/gross_pathology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "ct":              f"{DATA_ROOT}/CT/open_ended_train_and_val/val_open_ended_annotations.jsonl",
}

CONV_MODE = "llava_v1"


# ── Scoring helpers ────────────────────────────────────────────────────────────
def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def token_f1(pred_tokens, gold_tokens):
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_standard(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and ground-truth answer."""
    return token_f1(normalize(pred), normalize(gold))


def score_upd(pred: str) -> bool:
    """True if the model correctly refuses to answer."""
    pred_l = pred.lower()
    return "cannot answer" in pred_l or "can't answer" in pred_l or "i cannot" in pred_l


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, tokenizer, image_processor, question: str, image_path: str) -> str:
    prompt_q = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], prompt_q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    image = Image.open(image_path).convert("RGB")
    images_tensor = process_images([image], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# ── Main ───────────────────────────────────────────────────────────────────────
def load_records(domain: str, qtype: str, max_samples: int | None):
    path = VAL_FILES[domain]
    records = []
    for line in open(path):
        rec = json.loads(line)
        if qtype != "both" and rec["type"] != qtype:
            continue
        records.append({**rec, "domain": domain})
        if max_samples and len(records) >= max_samples:
            break
    return records


def evaluate(args):
    domains = list(VAL_FILES.keys()) if args.domain == "all" else [args.domain]

    print(f"Loading model from {MODEL_PATH} ...")
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name, load_4bit=False, load_8bit=False, device_map="auto"
    )
    model.eval()
    print("Model loaded.\n")

    all_results = []

    for domain in domains:
        records = load_records(domain, args.type, args.max)
        print(f"[{domain}] {len(records)} records (type={args.type})")

        std_scores, upd_correct, upd_total = [], 0, 0

        for rec in records:
            img_path = rec["image_path"]
            if not os.path.exists(img_path):
                print(f"  SKIP (missing image): {img_path}")
                continue

            pred = run_inference(model, tokenizer, image_processor, rec["question"], img_path)

            if rec["type"] == "standard":
                score = score_standard(pred, rec["answer"])
                std_scores.append(score)
                result = {"correct": score >= 0.3, "f1": score}
            else:  # upd
                correct = score_upd(pred)
                upd_correct += int(correct)
                upd_total   += 1
                result = {"correct": correct}

            all_results.append({
                "domain":    domain,
                "type":      rec["type"],
                "index":     rec.get("index"),
                "question":  rec["question"],
                "gold":      rec["answer"],
                "pred":      pred,
                **result,
            })

        # Per-domain summary
        print(f"\n{'─'*50}")
        print(f"Domain: {domain}")
        if std_scores:
            avg_f1 = sum(std_scores) / len(std_scores)
            acc_at30 = sum(s >= 0.3 for s in std_scores) / len(std_scores)
            print(f"  Standard  | n={len(std_scores):4d}  avg-F1={avg_f1:.3f}  acc@F1≥0.3={acc_at30:.3f}")
        if upd_total:
            upd_acc = upd_correct / upd_total
            print(f"  UPD       | n={upd_total:4d}  acc={upd_acc:.3f}  ({upd_correct}/{upd_total})")
        print()

    # Overall summary
    std_all = [r for r in all_results if r["type"] == "standard"]
    upd_all = [r for r in all_results if r["type"] == "upd"]

    print("="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    if std_all:
        avg_f1 = sum(r["f1"] for r in std_all) / len(std_all)
        acc    = sum(r["correct"] for r in std_all) / len(std_all)
        print(f"Standard  : n={len(std_all):5d}  avg-F1={avg_f1:.3f}  acc@F1≥0.3={acc:.3f}")
    if upd_all:
        acc = sum(r["correct"] for r in upd_all) / len(upd_all)
        print(f"UPD       : n={len(upd_all):5d}  acc={acc:.3f}")
    if std_all and upd_all:
        # Dual accuracy: both standard AND upd correct for same index within same domain
        std_idx  = {(r["domain"], r["index"]) for r in std_all  if r["correct"]}
        upd_idx  = {(r["domain"], r["index"]) for r in upd_all  if r["correct"]}
        dual_correct = len(std_idx & upd_idx)
        dual_total   = len({(r["domain"], r["index"]) for r in all_results})
        print(f"Dual acc  : {dual_correct}/{dual_total} = {dual_correct/max(dual_total,1):.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nPer-sample results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="x_ray",
                        choices=list(VAL_FILES.keys()) + ["all"])
    parser.add_argument("--type",   default="both",
                        choices=["standard", "upd", "both"])
    parser.add_argument("--max",    type=int, default=None,
                        help="Max samples per domain (for quick testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save per-sample results to this JSON path")
    args = parser.parse_args()
    evaluate(args)
