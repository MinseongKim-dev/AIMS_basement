import hashlib
import json
from pathlib import Path

DATASETS = ('vqa_rad', 'slake', 'pathvqa')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path, split=None):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError('Empty dataset')
    seen = set()
    for r in rows:
        for key in ('id', 'dataset', 'split', 'image', 'question', 'answer', 'answer_type'):
            if key not in r:
                raise ValueError(f'Missing {key}')
        key = (r['dataset'], str(r['id']))
        if key in seen:
            raise ValueError(f'Duplicate id: {key}')
        seen.add(key)
        if r['dataset'] not in DATASETS or r['answer_type'] not in ('open', 'closed'):
            raise ValueError('Invalid dataset or answer_type')
        if split and r['split'] != split:
            raise ValueError(f'Expected {split}, got {r["split"]}')
        if not str(r['answer']).strip():
            raise ValueError('Empty answer')
    return rows


def audit_splits(train, test):
    if {(r['dataset'], str(r['id'])) for r in train} & {(r['dataset'], str(r['id'])) for r in test}:
        raise ValueError('Train/test question ID overlap')
    # Image overlap is reported, not rejected: some MedVQA splits share images.
    images = {(r['dataset'], r['image']) for r in train} & {(r['dataset'], r['image']) for r in test}
    return {'shared_images': len(images), 'train': len(train), 'test': len(test)}
