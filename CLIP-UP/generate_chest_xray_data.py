"""
Generate chest X-ray training data for CLIP-UP in paired format.

Each answerable example is paired with its unanswerable counterpart.
Pairs are split together so both types always share the same index value
in the same split — this satisfies evaluate_dual_accuracy's assertion.
"""

import json
import os
import random

random.seed(42)

IMAGE_BASE = '/shared/rc/vlm-idk/LLaVA/data/images'
ANSWERABLE_PATH = '/home/as5606/projects/Super_LlaVA_Med/medium_answerable.jsonl'
NEGATIVES_PATH = '/home/as5606/projects/Super_LlaVA_Med/medium_negatives.jsonl'
STATEMENTS_PATH = '/home/as5606/projects/Super_LlaVA_Med/chest_xray_statements.jsonl'
OUT_DIR = '/home/as5606/projects/CLIP-UP/data/open_ended_train_and_val'

TRAIN_RATIO = 0.8


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_statements_by_id(path):
    by_id = {}
    for e in load_jsonl(path):
        if e['id'] not in by_id:
            by_id[e['id']] = e
    return by_id


def image_path(img_id):
    return os.path.join(IMAGE_BASE, img_id + '.jpg')


def make_standard(stmt_entry, index):
    return {
        'question': stmt_entry['question'],
        'statement': stmt_entry['statement'],
        'hint': None,
        'answer': stmt_entry['answer'],
        'image_path': image_path(stmt_entry['id']),
        'type': 'standard',
        'index': index,
        'upd_challenge_type': 'open_ended',
        'masked_answer': None,
    }


def make_upd(neg_entry, stmt_entry, index):
    # Image comes from the answerable example (neg_entry['id']),
    # question/statement from the negative counterpart (medium_negative_id)
    return {
        'question': stmt_entry['question'],
        'statement': stmt_entry['statement'],
        'hint': None,
        'answer': 'I cannot answer.',
        'image_path': image_path(neg_entry['id']),
        'type': 'upd',
        'index': index,
        'upd_challenge_type': 'open_ended',
        'masked_answer': None,
    }


def main():
    answerable = load_jsonl(ANSWERABLE_PATH)
    negatives = {e['id']: e for e in load_jsonl(NEGATIVES_PATH)}
    statements = load_statements_by_id(STATEMENTS_PATH)

    # Build valid pairs: answerable id must have a negative and both ids must be in statements
    pairs = []
    for ans in answerable:
        aid = ans['id']
        if aid not in negatives:
            continue
        neg = negatives[aid]
        nid = neg['medium_negative_id']
        if aid not in statements or nid not in statements:
            continue
        # Check images exist
        if not os.path.exists(image_path(aid)) or not os.path.exists(image_path(nid)):
            continue
        pairs.append((ans, neg))

    print(f'Valid pairs: {len(pairs)}')

    random.shuffle(pairs)
    split = int(TRAIN_RATIO * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    print(f'Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}')

    train_data = []
    for i, (ans, neg) in enumerate(train_pairs):
        std = make_standard(statements[ans['id']], index=i)
        upd = make_upd(neg, statements[neg['medium_negative_id']], index=i)
        train_data.append(std)
        train_data.append(upd)

    val_data = []
    for i, (ans, neg) in enumerate(val_pairs):
        std = make_standard(statements[ans['id']], index=i)
        upd = make_upd(neg, statements[neg['medium_negative_id']], index=i)
        val_data.append(std)
        val_data.append(upd)

    os.makedirs(OUT_DIR, exist_ok=True)

    train_path = os.path.join(OUT_DIR, 'train_open_ended_annotations.jsonl')
    val_path = os.path.join(OUT_DIR, 'val_open_ended_annotations.jsonl')

    with open(train_path, 'w') as f:
        for e in train_data:
            f.write(json.dumps(e) + '\n')

    with open(val_path, 'w') as f:
        for e in val_data:
            f.write(json.dumps(e) + '\n')

    print(f'Train: {len(train_data)} entries → {train_path}')
    print(f'Val: {len(val_data)} entries → {val_path}')

    # Verify pairing: every standard index should appear in upd and vice versa
    train_std_idx = {e['index'] for e in train_data if e['type'] == 'standard'}
    train_upd_idx = {e['index'] for e in train_data if e['type'] == 'upd'}
    assert train_std_idx == train_upd_idx, 'Train index mismatch!'

    val_std_idx = {e['index'] for e in val_data if e['type'] == 'standard'}
    val_upd_idx = {e['index'] for e in val_data if e['type'] == 'upd'}
    assert val_std_idx == val_upd_idx, 'Val index mismatch!'

    print('Index pairing verified OK.')


if __name__ == '__main__':
    main()
