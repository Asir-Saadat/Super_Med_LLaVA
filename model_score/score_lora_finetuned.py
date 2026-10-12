"""
Run LoRA-finetuned LLaVA-Med on all 5 domain val sets — UPD questions only.
Saves 5 JSONs to lora_finetuned_results/.
"""

import os
import sys
import argparse
sys.path.insert(0, "/home/as5606/projects/LLaVA")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

LORA_PATH  = "/shared/rc/vlm-idk/LLaVA/checkpoints/llava-med-lora-combined/checkpoint-2297"
MODEL_BASE = "/home/as5606/projects/LLaVA-Med/models/llava-med-v1.5-mistral-7b"

DOMAINS = {
    "x_ray":           "/home/as5606/projects/Super_LlaVA_Med/dataset_3/x_ray/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "MRI":             "/home/as5606/projects/Super_LlaVA_Med/dataset_3/MRI/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "CT":              "/home/as5606/projects/Super_LlaVA_Med/dataset_3/CT/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "histology":       "/home/as5606/projects/Super_LlaVA_Med/dataset_3/histology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "gross_pathology": "/home/as5606/projects/Super_LlaVA_Med/dataset_3/gross_pathology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
}

OUT_DIR = "/home/as5606/projects/Super_LlaVA_Med/model_score/lora_finetuned_results"
os.makedirs(OUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--reverse", action="store_true", help="Run domains in reverse order")
args = parser.parse_args()

device = "cuda"

print("Loading LoRA-finetuned LLaVA-Med...")
disable_torch_init()
model_name = get_model_name_from_path(MODEL_BASE)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    LORA_PATH, model_base=MODEL_BASE, model_name="llava-med-lora"
)
model.eval()
print(f"Model loaded: {model_name}")

conv_mode = "mistral_instruct"
print(f"Conv mode: {conv_mode}")


def run_inference(question, image_path):
    image = Image.open(image_path).convert("RGB")

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_tensor = process_images([image], image_processor, model.config).to(
        device, dtype=torch.float16
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


domain_items = list(DOMAINS.items())
if args.reverse:
    domain_items = list(reversed(domain_items))

for domain, val_path in domain_items:
    out_path = os.path.join(OUT_DIR, f"{domain}_val_results.json")
    if os.path.exists(out_path):
        print(f"\n[skip] {domain} — already exists")
        continue

    print(f"\n{'='*60}")
    print(f"Domain: {domain}")

    records = [json.loads(l) for l in open(val_path)]
    upd = [r for r in records if r["type"] == "upd"]
    print(f"UPD questions: {len(upd)} (skipping {len(records)-len(upd)} standard)")

    results = []
    for rec in tqdm(upd, desc=f"  {domain}"):
        try:
            answer = run_inference(rec["question"], rec["image_path"])
        except Exception as e:
            answer = f"ERROR: {e}"

        result = dict(rec)
        result["model_answer"] = answer
        results.append(result)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    refusal = sum(1 for r in results if "cannot answer" in r["model_answer"].lower())
    print(f"  Saved {len(results)} results to {out_path}")
    print(f"  Abstained on {refusal}/{len(results)} ({refusal/len(results)*100:.1f}%)")

print("\nAll domains done.")
