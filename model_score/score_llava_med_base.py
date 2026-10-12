"""
Run base LLaVA-Med (no CLIP-UP) on all 5 domain val sets in dataset_3.
Saves 5 JSONs to llava_med_base_results/.
"""

import os
import sys
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

MODEL_PATH = "/shared/rc/vlm-idk/LLaVA/checkpoints/llava-med-v1.5-mistral-7b"

DOMAINS = {
    "x_ray":          "/home/as5606/projects/Super_LlaVA_Med/dataset_3/x_ray/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "MRI":            "/home/as5606/projects/Super_LlaVA_Med/dataset_3/MRI/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "CT":             "/home/as5606/projects/Super_LlaVA_Med/dataset_3/CT/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "histology":      "/home/as5606/projects/Super_LlaVA_Med/dataset_3/histology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
    "gross_pathology":"/home/as5606/projects/Super_LlaVA_Med/dataset_3/gross_pathology/open_ended_train_and_val/val_open_ended_annotations.jsonl",
}

OUT_DIR = "/home/as5606/projects/Super_LlaVA_Med/model_score/llava_med_base_results"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda"

print("Loading LLaVA-Med...")
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, model_base=None, model_name=model_name
)
model.eval()
print(f"Model loaded: {model_name}")

# Determine conv mode
if "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
else:
    conv_mode = "llava_v0"
print(f"Conv mode: {conv_mode}")


def run_inference(question, image_path):
    image = Image.open(image_path).convert("RGB")

    # Build prompt with image token
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

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output


for domain, val_path in DOMAINS.items():
    print(f"\n{'='*60}")
    print(f"Domain: {domain}")

    records = [json.loads(l) for l in open(val_path)]
    print(f"Loaded {len(records)} records "
          f"({sum(1 for r in records if r['type']=='standard')} standard, "
          f"{sum(1 for r in records if r['type']=='upd')} upd)")

    results = []
    for rec in tqdm(records, desc=f"  {domain}"):
        try:
            answer = run_inference(rec["question"], rec["image_path"])
        except Exception as e:
            answer = f"ERROR: {e}"

        result = dict(rec)
        result["model_answer"] = answer
        results.append(result)

    out_path = os.path.join(OUT_DIR, f"{domain}_val_results.json")
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
