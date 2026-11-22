"""Replace the rule-based VQA-RAD answer_type with the original dataset labels.

The HF mirror (flaviagiammarino/vqa-rad) has no answer_type column, so download_data falls back
to "yes/no answer -> closed" (test: closed 251 / open 200). The original VQA-RAD labels also mark
choice questions such as "left or right?" as CLOSED (test: closed 272 / open 179), which is the
convention the paper's Open/Closed numbers use.

Label source: the original VQA-RAD questions (all 3,515, with answer_type) as published in the
team repo MinseongKim-dev/Expert-CFG, pinned to one commit and checked by sha256. Rows are
matched on normalized (question, answer); the test split must match completely and give 272/179.

  python -m phi3cfg.vqarad_labels --dataset-dir DATA/vqa_rad [--label-dir DIR]

Idempotent: provenance.json records answer_type_source and the run is skipped afterwards.
Only answer_type changes (old value kept as answer_type_rule), so training is unaffected.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from .data import digest, load_data

COMMIT = '6912f3b440ad9643d6c8d034332f65c498ffba9a'
URL = f'https://raw.githubusercontent.com/MinseongKim-dev/Expert-CFG/{COMMIT}/data_common/vqarad_img/{{}}'
FILES = {  # original train (3,064 = trainset + valset) and test (451)
    'trainset.json': ('train', '97b7758929bc3c17b101c3be8eadaea3a82cba2844d6c7274763ba3149c65108'),
    'valset.json': ('train', 'b053fb3996e09c0f34e2321805a406f5359f317f2eae0993c30fb55d3c55a353'),
    'testset.json': ('test', '3d32a047592cb680e18054030b38b1a44d6b17105fc62d3088c8655d56a00ced'),
}
EXPECTED_TEST = {'closed': 272, 'open': 179}
SOURCE = 'original_vqarad'


def norm(text):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).split())


def fetch_labels(label_dir):
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    for name, (_, sha) in FILES.items():
        path = label_dir / name
        if not path.exists():
            print(f'Downloading {name} (Expert-CFG@{COMMIT[:7]})', flush=True)
            with urllib.request.urlopen(URL.format(name), timeout=60) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError(f'{name}: sha256 mismatch for downloaded label file')
            path.write_bytes(data)
        elif digest(path) != sha:
            raise ValueError(f'{path}: sha256 mismatch; delete it to re-download')
    return label_dir


def build_index(label_dir):
    index = defaultdict(set)  # (part|'any', question, answer) -> {answer_type}
    for name, (part, _) in FILES.items():
        for r in json.loads((Path(label_dir) / name).read_text(encoding='utf-8')):
            kind = str(r['answer_type']).strip().lower()  # source has values like 'CLOSED '
            if kind not in ('open', 'closed'):
                raise ValueError(f'{name}: unexpected answer_type {r["answer_type"]!r}')
            key = (norm(r['question']), norm(r['answer']))
            index[(part, *key)].add(kind)
            index[('any', *key)].add(kind)
    return index


def relabel(dataset_dir, label_dir=None):
    d = Path(dataset_dir)
    prov_path = d / 'provenance.json'
    prov = json.loads(prov_path.read_text(encoding='utf-8'))
    if prov.get('dataset') != 'vqa_rad':
        raise ValueError(f'{d} is not the vqa_rad dataset')
    if prov.get('answer_type_source') == SOURCE:
        print(f'{d}: answer_type already uses original VQA-RAD labels', flush=True)
        return prov['answer_type_relabel']
    index = build_index(fetch_labels(label_dir or d.parent / 'vqarad_original_labels'))

    updated, report = {}, {}
    for split in prov['splits']:
        rows = load_data(d / f'{split}.jsonl', split)
        part = 'test' if split == 'test' else 'train'
        unmatched, ambiguous, changed = [], [], 0
        for r in rows:
            key = (norm(r['question']), norm(r['answer']))
            kinds = index.get((part, *key)) or index.get(('any', *key))
            if not kinds:
                unmatched.append(r['id'])
                continue
            if len(kinds) > 1:
                ambiguous.append(r['id'])
                continue
            kind = next(iter(kinds))
            if kind != r['answer_type']:
                r['answer_type_rule'] = r['answer_type']
                r['answer_type'] = kind
                changed += 1
        closed = sum(r['answer_type'] == 'closed' for r in rows)
        updated[split] = rows
        report[split] = {'n': len(rows), 'closed': closed, 'open': len(rows) - closed, 'changed': changed,
                         'unmatched': len(unmatched), 'ambiguous': len(ambiguous),
                         'unmatched_ids': unmatched[:20], 'ambiguous_ids': ambiguous[:20]}
        print(f'vqa_rad/{split}: closed {closed} / open {len(rows) - closed} '
              f'(changed {changed}, unmatched {len(unmatched)}, ambiguous {len(ambiguous)})', flush=True)

    t = report['test']
    if t['unmatched'] or t['ambiguous'] or {k: t[k] for k in EXPECTED_TEST} != EXPECTED_TEST:
        raise ValueError(f'Test labels incomplete or counts differ from {EXPECTED_TEST}: {t}. Nothing was written.')

    temps = []
    for split, rows in updated.items():
        tmp = d / f'{split}.jsonl.tmp'
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        temps.append((tmp, d / f'{split}.jsonl'))
    for tmp, path in temps:
        tmp.replace(path)
        prov['files'][path.name] = digest(path)
        s = report[path.name[:-len('.jsonl')]]
        prov['splits'][path.name[:-len('.jsonl')]].update(open=s['open'], closed=s['closed'])
    prov['answer_type_source'] = SOURCE
    prov['answer_type_label_files'] = {'repo': 'MinseongKim-dev/Expert-CFG', 'commit': COMMIT,
                                       'sha256': {n: sha for n, (_, sha) in FILES.items()}}
    prov['answer_type_relabel'] = report
    tmp = prov_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(prov_path)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-dir', required=True, help='prepared DATA/vqa_rad (download_data output)')
    p.add_argument('--label-dir', help='where the original label JSONs are cached (default: DATA/vqarad_original_labels)')
    args = p.parse_args()
    relabel(args.dataset_dir, args.label_dir)


if __name__ == '__main__':
    main()
