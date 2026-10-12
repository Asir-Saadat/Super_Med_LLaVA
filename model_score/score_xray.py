import os
os.chdir("/home/as5606/projects/CLIP-UP")
import sys
sys.path.insert(0, "/home/as5606/projects/CLIP-UP")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from PIL import Image
from tqdm import tqdm

from clip_up.vlms.clip_up_vlms import init_vlm, get_vlm_path_and_config, run_vlm_inference_for_single_item
from clip_up.clip_up_wrapper import ClipUpWrapper, StructureClipSignalExtractor
from clip_up.clip_up_utils import InjectionType
from clip_up.clip_up_general_utils import seed_everything, disable_torch_init

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--val_path", type=str, required=True, help="Path to val_open_ended_annotations.jsonl")
parser.add_argument("--out_path", type=str, required=True, help="Path to save results JSON")
args = parser.parse_args()

CHECKPOINT = "/home/as5606/projects/Super_LlaVA_Med/model_checkpoints/med-llava/open_ended_train_results/dataset3_xray_run1/checkpoint_epoch_2.pt"
VAL_PATH   = args.val_path
OUT_PATH   = args.out_path

device = "cuda"
disable_torch_init()
seed_everything(42, with_accelerate=False)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model_path, model_config = get_vlm_path_and_config("med-llava")
model, image_processor, tokenizer, conv_mode = init_vlm(
    model_name="med-llava", model_path=model_path, model_config=model_config,
    is_clip_up_emb_lora_injection=False, is_for_open_ended_questions=True, device=device
)

# ── Load CLIP-UP checkpoint ───────────────────────────────────────────────────
print("Loading checkpoint...")
wrapper = ClipUpWrapper(model=model, model_name="med-llava", model_config=model_config,
                        conv_mode=conv_mode, tokenizer=tokenizer,
                        image_processor=image_processor, device=device,
                        output_dir="/tmp/inference_out", accelerator=None,
                        is_for_open_ended_questions=True,
                        injection_type=InjectionType.CLIP_UP_EMB)
wrapper.load_checkpoint(CHECKPOINT)
model.requires_grad_(False)
model.eval()

# ── BioMedCLIP extractor ──────────────────────────────────────────────────────
extractor = StructureClipSignalExtractor(device=device, torch_dtype=torch.float16,
                                          structure_clip_checkpoint_path=model_config.structure_clip_checkpoint_path)

# ── Load val records ──────────────────────────────────────────────────────────
val_records = [json.loads(l) for l in open(VAL_PATH)]
print(f"Loaded {len(val_records)} val records "
      f"({sum(1 for r in val_records if r['type']=='standard')} standard, "
      f"{sum(1 for r in val_records if r['type']=='upd')} upd)")

# ── Run inference ─────────────────────────────────────────────────────────────
results = []
for rec in tqdm(val_records, desc="Running inference"):
    try:
        clip_signal = extractor.extract_clip_signal(
            [rec["statement"]], Image.open(rec["image_path"])
        ).flatten().to(torch.bfloat16)

        output, _ = run_vlm_inference_for_single_item(
            model=model, model_name="med-llava", conv_mode=conv_mode,
            tokenizer=tokenizer, image_processor=image_processor,
            upd_setting=wrapper.upd_setting, upd_type="open_ended",
            question=rec["question"], hint=None, options=[],
            image=Image.open(rec["image_path"]),
            clip_signal=clip_signal
        )
    except Exception as e:
        output = f"ERROR: {e}"

    result = dict(rec)          # keep all original fields
    result["model_answer"] = output
    results.append(result)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {OUT_PATH}")

# ── Quick summary ─────────────────────────────────────────────────────────────
std = [r for r in results if r["type"] == "standard"]
upd = [r for r in results if r["type"] == "upd"]

std_refusal = sum(1 for r in std if r["model_answer"] == "I cannot answer.")
upd_refusal = sum(1 for r in upd if r["model_answer"] == "I cannot answer.")

print(f"\nStandard ({len(std)}): model said 'I cannot answer.' on {std_refusal} ({std_refusal/len(std)*100:.1f}%)")
print(f"UPD      ({len(upd)}): model said 'I cannot answer.' on {upd_refusal} ({upd_refusal/len(upd)*100:.1f}%)")
