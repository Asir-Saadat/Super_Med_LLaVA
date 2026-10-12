import os
os.chdir("/home/as5606/projects/CLIP-UP")
import sys
sys.path.insert(0, "/home/as5606/projects/CLIP-UP")

import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
DOMAINS = {
    "X-ray":      "/home/as5606/projects/Super_LlaVA_Med/dataset_3/x_ray/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "MRI":        "/home/as5606/projects/Super_LlaVA_Med/dataset_3/MRI/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "CT":         "/home/as5606/projects/Super_LlaVA_Med/dataset_3/CT/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "Histology":  "/home/as5606/projects/Super_LlaVA_Med/dataset_3/histology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "Pathology":  "/home/as5606/projects/Super_LlaVA_Med/dataset_3/gross_pathology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
}
OUT_DIR = "/home/as5606/projects/Super_LlaVA_Med/paper_graphs"
device  = "cuda"

# ── Load BioMedCLIP ───────────────────────────────────────────────────────────
print("Loading BioMedCLIP...")
model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device=device
)
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.eval()
model.requires_grad_(False)

# ── Extract CLIP signals ──────────────────────────────────────────────────────
all_signals, all_labels = [], []

for domain_name, val_path in DOMAINS.items():
    records = [json.loads(l) for l in open(val_path) if json.loads(l)["type"] == "upd"]
    print(f"\n{domain_name}: {len(records)} UPD records")

    for rec in tqdm(records, desc=domain_name):
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
            img_tensor  = preprocess(image).unsqueeze(0).to(device)
            text_tokens = tokenizer([rec["statement"]]).to(device)

            with torch.inference_mode():
                img_feat  = model.encode_image(img_tensor)
                img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat  = model.encode_text(text_tokens)
                txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                signal    = (img_feat * txt_feat).squeeze(0).cpu().float().numpy()

            all_signals.append(signal)
            all_labels.append(domain_name)
        except Exception as e:
            pass

all_signals = np.array(all_signals)
all_labels  = np.array(all_labels)
print(f"\nTotal signals: {all_signals.shape}")

# ── Save embeddings ───────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "clip_signals.npy"), all_signals)
np.save(os.path.join(OUT_DIR, "clip_labels.npy"),  all_labels)
print("Saved embeddings.")

# ── UMAP + t-SNE ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLORS = {
    "X-ray":     "#e41a1c",
    "MRI":       "#377eb8",
    "CT":        "#4daf4a",
    "Histology": "#ff7f00",
    "Pathology": "#984ea3",
}

def plot_2d(coords, labels, title, ax):
    for domain in DOMAINS.keys():
        mask = labels == domain
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[domain], s=4, alpha=0.5, label=domain)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# UMAP
print("\nRunning UMAP...")
import umap
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
umap_coords = reducer.fit_transform(all_signals)
plot_2d(umap_coords, all_labels, "UMAP — BioMedCLIP CLIP Signal (UPD)", axes[0])

# t-SNE
print("Running t-SNE...")
from sklearn.manifold import TSNE
tsne_coords = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000).fit_transform(all_signals)
plot_2d(tsne_coords, all_labels, "t-SNE — BioMedCLIP CLIP Signal (UPD)", axes[1])

handles = [mpatches.Patch(color=COLORS[d], label=d) for d in DOMAINS.keys()]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])

out_path = os.path.join(OUT_DIR, "clip_signal_upd_visualization.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.show()
