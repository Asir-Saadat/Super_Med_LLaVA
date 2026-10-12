"""
Convert false-positive questions to statements using Qwen2.5-1.5B-Instruct.
Reads:  combined_classifier_false_positives.json
Writes: combined_classifier_false_positives_with_statements.json  (same structure)
        + per-domain .jsonl files ready for CLIP-UP inference
"""

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

FP_PATH  = "/home/as5606/projects/Super_LlaVA_Med/guardrails/combined_classifier_false_positives.json"
OUT_JSON = "/home/as5606/projects/Super_LlaVA_Med/guardrails/combined_classifier_false_positives_with_statements.json"
OUT_DIR  = "/home/as5606/projects/Super_LlaVA_Med/guardrails/false_positive_annotations"

import os
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPTS = {
    "x_ray": """You are converting chest X-ray visual-question-answering questions into short
declarative statements, for use as text queries against a biomedical CLIP model.

Convert the question into ONE short, factual, declarative sentence stating the topic
or finding being asked about, phrased like a radiology caption.

Rules:
1. Output exactly ONE sentence. No preamble, no explanation.
2. Do NOT invent a specific answer value unless already in the question.
3. Remove any "<image>" placeholder text.
4. Write in third person, declarative tone — like a radiology report.
5. Do not use hedging language ("might", "could", "possibly").
6. Preserve clinical/anatomical terminology from the question.
7. Prefer "The chest X-ray shows/demonstrates ..." as the base template.

Examples:
Question: "Where is the tip of the central line located?"
Statement: "The chest X-ray shows the tip of a central line."

Question: "What is the size of the pneumothorax?"
Statement: "The chest X-ray shows a pneumothorax."

Question: "What can you say about the tracheal position?"
Statement: "The chest X-ray shows the position of the trachea."

Question: "What do the arrows indicate?"
Statement: "The chest X-ray contains arrows indicating a specific finding.\"""",

    "mri": """You are converting brain/body MRI visual-question-answering questions into short
declarative statements, for use as text queries against a biomedical CLIP model.

Convert the question into ONE short, factual, declarative sentence stating the topic
or finding being asked about, phrased like a radiology caption.

Rules:
1. Output exactly ONE sentence. No preamble, no explanation.
2. Do NOT invent a specific answer value unless already in the question.
3. Remove any "<image>" placeholder text.
4. Write in third person, declarative tone — like a radiology report.
5. Do not use hedging language ("might", "could", "possibly").
6. Preserve clinical/anatomical terminology from the question.
7. Prefer "The MRI shows/demonstrates ..." as the base template.

Examples:
Question: "What is the relationship between the mass and the brainstem?"
Statement: "The MRI shows a mass in relation to the brainstem."

Question: "What does the MR venography show about the dural sinuses?"
Statement: "The MRI shows the dural sinuses on MR venography."

Question: "In which regions are the iron-related hypointensities found?"
Statement: "The MRI demonstrates iron-related hypointensities in specific brain regions."

Question: "What is the issue with the foramen magnum in this image?"
Statement: "The MRI shows an abnormality at the foramen magnum.\"""",

    "ct": """You are converting CT scan visual-question-answering questions into short
declarative statements, for use as text queries against a biomedical CLIP model.

Convert the question into ONE short, factual, declarative sentence stating the topic
or finding being asked about, phrased like a radiology caption.

Rules:
1. Output exactly ONE sentence. No preamble, no explanation.
2. Do NOT invent a specific answer value unless already in the question.
3. Remove any "<image>" placeholder text.
4. Write in third person, declarative tone — like a radiology report.
5. Do not use hedging language ("might", "could", "possibly").
6. Preserve clinical/anatomical terminology from the question.
7. Prefer "The CT shows/demonstrates ..." as the base template.

Examples:
Question: "What is the size of the mass in the left kidney?"
Statement: "The CT shows a mass in the left kidney."

Question: "What abnormalities are seen in the right upper lobe?"
Statement: "The CT demonstrates abnormalities in the right upper lobe."

Question: "What is the relationship between the mass and the scapula?"
Statement: "The CT shows a mass in relation to the scapula."

Question: "What structures are displaced by the lesion?"
Statement: "The CT shows a lesion displacing adjacent structures.\"""",

    "histology": """You are converting histology/histopathology visual-question-answering questions into short
declarative statements, for use as text queries against a biomedical CLIP model.

Convert the question into ONE short, factual, declarative sentence stating the topic
or finding being asked about, phrased like a pathology caption.

Rules:
1. Output exactly ONE sentence. No preamble, no explanation.
2. Do NOT invent a specific answer value unless already in the question.
3. Remove any "<image>" placeholder text.
4. Write in third person, declarative tone — like a pathology report.
5. Do not use hedging language ("might", "could", "possibly").
6. Preserve clinical/anatomical/histological terminology from the question.
7. Prefer "The histology slide shows/demonstrates ..." as the base template.

Examples:
Question: "What type of hyphae is shown in the image?"
Statement: "The histology slide shows fungal hyphae."

Question: "What type of carcinoma is present in the tissue?"
Statement: "The histology slide demonstrates carcinoma in the tissue."

Question: "What are the different layers of the cornea visible in the image?"
Statement: "The histology slide shows the distinct layers of the cornea."

Question: "What is the histopathological appearance of the posterior concavity?"
Statement: "The histology slide shows the histopathological features of the posterior concavity.\"""",

    "gross_pathology": """You are converting gross pathology visual-question-answering questions into short
declarative statements, for use as text queries against a biomedical CLIP model.

Convert the question into ONE short, factual, declarative sentence stating the topic
or finding being asked about, phrased like a pathology caption.

Rules:
1. Output exactly ONE sentence. No preamble, no explanation.
2. Do NOT invent a specific answer value unless already in the question.
3. Remove any "<image>" placeholder text.
4. Write in third person, declarative tone — like a pathology report.
5. Do not use hedging language ("might", "could", "possibly").
6. Preserve clinical/anatomical terminology from the question.
7. Prefer "The gross specimen shows/demonstrates ..." as the base template.

Examples:
Question: "Where is the mass located relative to the capsule?"
Statement: "The gross specimen shows a mass in relation to the capsule."

Question: "What features of the cut surface suggest malignancy?"
Statement: "The gross specimen shows cut surface features suggestive of malignancy."

Question: "What is the appearance of the gallbladder wall?"
Statement: "The gross specimen shows the appearance of the gallbladder wall."

Question: "What structures are visible adjacent to the mass?"
Statement: "The gross specimen shows structures adjacent to the mass.\"""",
}

print("Loading Qwen2.5-1.5B-Instruct...")
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto")
qwen_tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


def make_statement(question, system_prompt):
    question = question.replace("<image>", "").strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f'Question: "{question}"'}
    ]
    text   = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tok([text], return_tensors="pt").to(qwen_model.device)
    out    = qwen_model.generate(**inputs, max_new_tokens=128)
    resp   = qwen_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return resp.strip().strip('"').strip("'").split("\n")[0].strip()


data = json.load(open(FP_PATH))

for domain, sys_prompt in SYSTEM_PROMPTS.items():
    if domain not in data:
        print(f"[skip] {domain} not in false positives")
        continue

    questions = data[domain]["questions"]
    print(f"\n{'='*60}")
    print(f"Domain: {domain}  ({len(questions)} questions)")

    for rec in tqdm(questions, desc=f"  {domain}"):
        rec["statement"] = make_statement(rec["question"], sys_prompt)

    # Save per-domain .jsonl for CLIP-UP inference
    jsonl_path = os.path.join(OUT_DIR, f"{domain}.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in questions:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(questions)} records -> {jsonl_path}")

# Save updated full JSON
json.dump(data, open(OUT_JSON, "w"), indent=2)
print(f"\nFull JSON saved -> {OUT_JSON}")
