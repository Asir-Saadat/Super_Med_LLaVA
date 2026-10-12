"""
For each domain, take all REMOVE questions and run the combined classifier.
Report and save the ones the classifier incorrectly predicts as KEEP (false positives).
"""

import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

MODEL_DIR = "/home/as5606/projects/Super_LlaVA_Med/guardrails/guardrail_1_combined"
THRESHOLD = json.load(open(f"{MODEL_DIR}/config.json"))["threshold"]

DOMAINS = {
    "x_ray":           "/home/as5606/projects/Super_LlaVA_Med/dataset_3/x_ray/filter_decisions.jsonl",
    "mri":             "/home/as5606/projects/Super_LlaVA_Med/dataset_3/MRI/filter_decisions.jsonl",
    "ct":              "/home/as5606/projects/Super_LlaVA_Med/dataset_3/CT/filter_decisions.jsonl",
    "histology":       "/home/as5606/projects/Super_LlaVA_Med/dataset_3/histology/filter_decisions.jsonl",
    "gross_pathology": "/home/as5606/projects/Super_LlaVA_Med/dataset_3/gross_pathology/filter_decisions.jsonl",
}

class QDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=64)
    def __len__(self):
        return len(self.enc["input_ids"])
    def __getitem__(self, i):
        return {
            "input_ids":      torch.tensor(self.enc["input_ids"][i]),
            "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
        }


print(f"Loading combined classifier (threshold={THRESHOLD})...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).cuda().eval()

summary = {}

for domain, filter_file in DOMAINS.items():
    records = [json.loads(l) for l in open(filter_file)]
    remove_records = [r for r in records if r["decision"] == "REMOVE"]

    texts = [r["question"].replace("<image>", "").strip() for r in remove_records]

    ds     = QDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=64)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            out   = model(input_ids=batch["input_ids"].cuda(),
                          attention_mask=batch["attention_mask"].cuda())
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)

    # False positives: REMOVE questions predicted as KEEP
    false_positives = []
    for idx, (rec, prob) in enumerate(zip(remove_records, all_probs)):
        if prob >= THRESHOLD:
            false_positives.append({
                "question":          rec["question"].replace("<image>", "").strip(),
                "statement":         None,
                "answer":            rec.get("answer", ""),
                "image_path":        rec.get("image_path", ""),
                "type":              "standard",
                "index":             idx,
                "upd_challenge_type": "open_ended",
                "hint":              None,
                "masked_answer":     None,
                "prob_keep":         round(prob, 4),
            })

    false_positives.sort(key=lambda x: x["prob_keep"], reverse=True)

    summary[domain] = {
        "total_remove": len(remove_records),
        "classifier_kept": len(false_positives),
        "rate": round(len(false_positives) / len(remove_records) * 100, 2),
        "questions": false_positives,
    }

    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"  REMOVE questions : {len(remove_records)}")
    print(f"  Classifier KEEP  : {len(false_positives)} ({summary[domain]['rate']}%)")
    print(f"  Sample:")
    for fp in false_positives[:5]:
        print(f"    [{fp['prob_keep']:.3f}] {fp['question']}")

out_path = "/home/as5606/projects/Super_LlaVA_Med/guardrails/combined_classifier_false_positives.json"
json.dump(summary, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")
