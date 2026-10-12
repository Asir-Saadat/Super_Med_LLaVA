import json
import os
import torch
import numpy as np
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda"

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading BioMedCLIP...")
model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    device=DEVICE
)
model.eval()

# ── Domain annotation files ───────────────────────────────────────────────────
DOMAINS = {
    "x_ray":           "/home/as5606/projects/Super_LlaVA_Med/dataset/x_ray/open_ended_train_and_val/train_open_ended_annotations.jsonl",
    "mri":             "/home/as5606/projects/Super_LlaVA_Med/dataset/MRI/open_ended_train_and_val/train_open_ended_annotations.jsonl",
    "histology":       "/home/as5606/projects/Super_LlaVA_Med/dataset/histology/open_ended_train_and_val/train_open_ended_annotations.jsonl",
    "gross_pathology": "/home/as5606/projects/Super_LlaVA_Med/dataset/gross_pathology/open_ended_train_and_val/train_open_ended_annotations.jsonl",
    "ct":              "/home/as5606/projects/Super_LlaVA_Med/dataset/CT/open_ended_train_and_val/train_open_ended_annotations.jsonl",
}

VAL_DOMAINS = {
    "x_ray":           "/home/as5606/projects/Super_LlaVA_Med/dataset/x_ray/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "mri":             "/home/as5606/projects/Super_LlaVA_Med/dataset/MRI/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "histology":       "/home/as5606/projects/Super_LlaVA_Med/dataset/histology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "gross_pathology": "/home/as5606/projects/Super_LlaVA_Med/dataset/gross_pathology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "ct":              "/home/as5606/projects/Super_LlaVA_Med/dataset/CT/open_ended_train_and_val/val_open_ended_annotations.jsonl",
}

# ── Helper: encode a list of image paths ──────────────────────────────────────
def encode_images(image_paths):
    feats = []
    missing = 0
    with torch.no_grad():
        for path in tqdm(image_paths, leave=False):
            if not os.path.exists(path):
                missing += 1
                continue
            try:
                img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
                f = model.encode_image(img)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
            except Exception:
                missing += 1
    if missing:
        print(f"  (skipped {missing} missing/corrupt images)")
    return torch.cat(feats) if feats else torch.zeros(0, 512)

# ── Load unique image paths per domain (standard entries only) ────────────────
def load_image_paths(domain_files):
    domain_images = {}
    for domain, path in domain_files.items():
        seen = set()
        paths = []
        for line in open(path):
            rec = json.loads(line)
            if rec["type"] == "standard" and rec["image_path"] not in seen:
                seen.add(rec["image_path"])
                paths.append(rec["image_path"])
        domain_images[domain] = paths
        print(f"  {domain:20s}: {len(paths)} unique images")
    return domain_images

# ── Build prototypes from training images ─────────────────────────────────────
print("\nLoading train image paths...")
train_images = load_image_paths(DOMAINS)

print("\nEncoding training images and building prototypes...")
prototypes = {}
for domain, paths in train_images.items():
    print(f"  {domain}...")
    embs = encode_images(paths)
    proto = embs.mean(dim=0)
    proto = proto / proto.norm()
    prototypes[domain] = proto

domain_names = list(prototypes.keys())
proto_matrix = torch.stack([prototypes[d] for d in domain_names])  # [5, 512]
print(f"\nPrototype matrix: {proto_matrix.shape}")

# ── Evaluate on val images ────────────────────────────────────────────────────
print("\nLoading val image paths...")
val_images = load_image_paths(VAL_DOMAINS)

print("\nEvaluating on val images...")
all_true, all_pred = [], []

for true_domain, paths in val_images.items():
    print(f"  {true_domain}...")
    embs = encode_images(paths)
    if embs.shape[0] == 0:
        continue
    sims = embs @ proto_matrix.T        # [N, 5]
    pred_indices = sims.argmax(dim=1)   # [N]
    for idx in pred_indices:
        all_true.append(true_domain)
        all_pred.append(domain_names[idx.item()])

print("\n" + "="*60)
print("Image Prototype Classifier — Val Results")
print("="*60)
print(classification_report(all_true, all_pred, target_names=domain_names))
print("Confusion matrix (rows=true, cols=pred):")
print("Labels:", domain_names)
print(confusion_matrix(all_true, all_pred, labels=domain_names))

# ── Show similarity scores for a few examples ─────────────────────────────────
print("\nSample predictions (first 5 val images per domain):")
for true_domain, paths in val_images.items():
    for path in paths[:3]:
        if not os.path.exists(path):
            continue
        with torch.no_grad():
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            f = model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
        sims = (f.cpu() @ proto_matrix.T).squeeze(0)
        pred = domain_names[sims.argmax().item()]
        scores = {domain_names[i]: f"{sims[i].item():.3f}" for i in range(len(domain_names))}
        status = "✓" if pred == true_domain else "✗"
        print(f"  {status} true={true_domain:20s} pred={pred:20s} scores={scores}")
    break
