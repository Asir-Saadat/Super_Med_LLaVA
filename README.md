# Super Med LLaVA

A multi-domain medical VQA pipeline with unanswerable question detection (CLIP-UP). Built on top of [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), this project covers dataset construction, guardrail training, CLIP-UP signal injection, and evaluation across five medical imaging domains.

---

## Repository Structure

```
Super_Med_LLaVA/
├── CLIP-UP/                  # CLIP-UP model — linear projection for abstention detection
├── dataset_3/                # Main dataset (not tracked in git — see below)
├── dataset_3_matched/        # Size-matched combined dataset (not tracked)
├── guardrails/               # DistilBERT question-quality classifier
├── model_score/              # Inference scripts and GPT-judge evaluation
└── paper_graphs/             # Figures generated for the paper
```

---

## Dataset: `dataset_3`

`dataset_3` is the core dataset used for training and evaluation. It is organized by imaging domain:

```
dataset_3/
├── x_ray/
├── MRI/
├── CT/
├── histology/
├── gross_pathology/
└── combined/
```

Each domain folder contains:
```
open_ended_train_and_val/
├── train_open_ended_annotations.jsonl
└── val_open_ended_annotations.jsonl
```

Each record in these files has the following fields:

| Field | Description |
|---|---|
| `question` | The visual question |
| `statement` | Declarative version of the question (used as BioMedCLIP text input) |
| `answer` | Ground truth answer (`"I cannot answer."` for UPD questions) |
| `image_path` | Absolute path to the image |
| `type` | `"standard"` (answerable) or `"upd"` (unanswerable) |
| `index` | Record index |
| `upd_challenge_type` | Always `"open_ended"` |
| `hint` | null |
| `masked_answer` | null |
| `domain` | Domain label (in combined split only) |

### `dataset_3_matched`

A size-matched version of the combined dataset with approximately equal standard and UPD samples per domain (5,600 total: 2,800 standard + 2,800 UPD). Used to train the combined CLIP-UP model with balanced class representation across domains.

---

## Guardrails (`guardrails/`)

**Guardrail 1** — DistilBERT-based question-quality classifier. Trained to filter out non-visual, generic, binary, and general medical knowledge questions from the LLaVA-Med Instruct 60K dataset before inclusion in `dataset_3`.

- `guardrail_1_accuracy.ipynb` — training and evaluation notebook
- Domain-specific models: `guardrail_1_xray/`, `guardrail_1_mri/`, etc.
- Combined model: `guardrail_1_combined/` (threshold: 0.619)
- Eval script: `eval_combined_classifier.py`

---

## CLIP-UP (`CLIP-UP/`)

A lightweight linear projection layer (`nn.Linear(512, 4096, bias=False)`, ~2M params) trained on top of frozen BioMedCLIP embeddings to inject an image–text similarity signal into LLaVA-Med's hidden space for abstention detection.

- `train.py` — main training script
- `train_domain.sh` — SLURM training launcher (supports per-domain and combined training)
- `clip_up/` — model code (wrapper, signal extractor, VLM configs)

**To train the combined CLIP-UP model:**
```bash
python train.py \
    --vlm med-llava \
    --injection_type clip_up_emb \
    --vqa_type open_ended \
    --data_root /path/to/dataset_3_matched/combined \
    --output_root_dir /path/to/model_checkpoints \
    --run_name dataset3_matched_combined_run1
```

---

## Model Scoring (`model_score/`)

Inference scripts for all evaluated models:

| Script | Model |
|---|---|
| `score_llava_med_base.py` | LLaVA-Med base (no CLIP-UP) |
| `score_combined_model.py` | CLIP-UP combined model |
| `score_lora_finetuned.py` | LoRA-finetuned LLaVA-Med |
| `score_openai.py` | GPT-5.4-nano |
| `score_openai_mini.py` | GPT-5.4-mini |
| `score_gemini.py` | Gemini 3.5 Flash Lite |
| `score_gemini_38.py` | Gemini 3.8 Flash |
| `score_claude.py` | Claude Haiku |

**GPT-judge evaluation:**

| Script | Metric |
|---|---|
| `gpt_judge_standard.py` | CA — Correct Answer rate |
| `gpt_judge_upd.py` | RA — Right Abstention rate |
| `gpt_judge_wa.py` | WA — Wrong Abstention rate |

---

## Environment Variables

API keys are read from environment variables — never hardcoded:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```
