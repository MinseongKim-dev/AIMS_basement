"""Build the team's common VQA-RAD split (same as the Phi-3.5V baseline in MinseongKim-dev/Expert-CFG).

Split: original public train (3,064 questions incl. framed ones) with an image-level validation
split carved out (seed 42): train 2,749 / validation 315 / test 451 (original test, closed 272 /
open 179). Questions, answers and answer_type come from the team repo's JSONs (pinned commit,
sha256-checked, see vqarad_labels.py); answers are kept verbatim like the team code.

Images: the original VQA_RAD Image Folder (--image-dir) if you have it, otherwise the original
JPEG bytes recovered from the HF mirror flaviagiammarino/vqa-rad (pinned revision): every mirror
row is matched to the original questions on (question, answer) and only unambiguous matches name
an image. All 314 images used by the split are recovered and each name maps to one pixel content.

  python -m phi3cfg.vqarad_common --output DATA [--image-dir ".../VQA_RAD Image Folder"]

Writes DATA/vqa_rad/{train,validation,test}.jsonl, images/, provenance.json (skips if complete).
"""
import argparse
import hashlib
import io
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from .data import digest, load_data
from .vqarad_labels import COMMIT, FILES, fetch_labels, norm

HF_REPO = 'flaviagiammarino/vqa-rad'
HF_REVISION = 'bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9'
HF_FILES = ('data/train-00000-of-00001-eb8844602202be60.parquet',
            'data/test-00000-of-00001-e5bc3d208bb4deeb.parquet')
SPLITS = {'train': 'trainset.json', 'validation': 'valset.json', 'test': 'testset.json'}
EXPECTED = {'train': 2749, 'validation': 315, 'test': 451}
EXPECTED_TEST_TYPES = {'closed': 272, 'open': 179}
FORMAT = 'team_common_v1'


def pixel_hash(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return hashlib.sha256(str(im.size).encode() + im.convert('RGB').tobytes()).hexdigest()


def images_from_hf(originals, cache_dir=None):
    """image_name -> original JPEG bytes, recovered from the HF mirror."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    names_by_key = defaultdict(set)
    for r in originals:
        names_by_key[(norm(r['question']), norm(r['answer']))].add(r['image_name'])
    found, hashes = {}, defaultdict(set)
    for f in HF_FILES:
        path = hf_hub_download(HF_REPO, f, repo_type='dataset', revision=HF_REVISION, cache_dir=cache_dir)
        df = pd.read_parquet(path, columns=['question', 'answer', 'image'])
        for q, a, im in zip(df['question'], df['answer'], df['image']):
            names = names_by_key.get((norm(q), norm(a)), set())
            if len(names) != 1:
                continue  # generic question/answer shared by several images: cannot name the image
            name = next(iter(names))
            hashes[name].add(pixel_hash(im['bytes']))
            found.setdefault(name, im['bytes'])
    conflicts = sorted(n for n, h in hashes.items() if len(h) > 1)
    if conflicts:
        raise ValueError(f'HF mirror gives different pixels for the same image name: {conflicts[:5]}')
    return found


def build(output, image_dir=None, label_dir=None, cache_dir=None):
    target = Path(output).resolve() / 'vqa_rad'
    prov_path = target / 'provenance.json'
    if prov_path.exists():
        prov = json.loads(prov_path.read_text(encoding='utf-8'))
        if prov.get('format') != FORMAT:
            raise ValueError(f'{target} holds another data format; use a different --output')
        bad = [f for f, h in prov['files'].items() if not (target / f).is_file() or digest(target / f) != h]
        bad += [n for n, h in prov['images'].items() if not (target / 'images' / n).is_file()]
        if bad:
            raise ValueError(f'{target}: files missing/modified {bad[:5]}; delete the folder to rebuild')
        print(f'vqa_rad (team common split): reusing {target}', flush=True)
        return prov
    if target.exists():
        raise ValueError(f'{target} exists without provenance.json; delete it or use another --output')

    label_dir = fetch_labels(label_dir or Path(output) / 'vqarad_original_labels')
    data = {s: json.loads((label_dir / f).read_text(encoding='utf-8')) for s, f in SPLITS.items()}
    originals = [r for rows in data.values() for r in rows]
    needed = sorted({r['image_name'] for r in originals})
    if image_dir:
        image_bytes = {n: (Path(image_dir) / n).read_bytes() for n in needed if (Path(image_dir) / n).is_file()}
        image_source = {'type': 'local', 'path': str(image_dir)}
    else:
        image_bytes = images_from_hf(originals, cache_dir)
        image_source = {'type': 'hf_mirror', 'repo': HF_REPO, 'revision': HF_REVISION}
    missing = [n for n in needed if n not in image_bytes]
    if missing:
        raise ValueError(f'{len(missing)} images unavailable (e.g. {missing[:3]})')

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.vqa_rad-', dir=target.parent))
    try:
        (staging / 'images').mkdir()
        prov = {'format': FORMAT, 'dataset': 'vqa_rad', 'answer_type_source': 'original_vqarad',
                'split_policy': 'team common split: original train minus image-level val (seed 42); original test',
                'label_files': {'repo': 'MinseongKim-dev/Expert-CFG', 'commit': COMMIT,
                                'sha256': {n: sha for n, (_, sha) in FILES.items()}},
                'image_source': image_source, 'splits': {}, 'files': {}, 'images': {}}
        for name in needed:
            (staging / 'images' / name).write_bytes(image_bytes[name])
            prov['images'][name] = digest(staging / 'images' / name)
        for split, src in data.items():
            rows = []
            for r in src:
                kind = str(r['answer_type']).strip().lower()
                rows.append({'id': f'{split}:{r["qid"]}', 'source_id': str(r['qid']), 'dataset': 'vqa_rad',
                             'split': split, 'image': r['image_name'], 'question': str(r['question']).strip(),
                             'answer': str(r['answer']).strip(), 'answer_type': kind,
                             'question_type': r.get('question_type'), 'phrase_type': r.get('phrase_type'),
                             'image_organ': r.get('image_organ')})
            path = staging / f'{split}.jsonl'
            path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
            load_data(path, split)
            closed = sum(x['answer_type'] == 'closed' for x in rows)
            prov['splits'][split] = {'kept': len(rows), 'open': len(rows) - closed, 'closed': closed,
                                     'images': len({x['image'] for x in rows})}
            prov['files'][path.name] = digest(path)
        counts = {s: v['kept'] for s, v in prov['splits'].items()}
        test_types = {k: prov['splits']['test'][k] for k in EXPECTED_TEST_TYPES}
        if counts != EXPECTED or test_types != EXPECTED_TEST_TYPES:
            raise ValueError(f'split sizes {counts} / test types {test_types} differ from the team split')
        ids = [str(x['qid']) for rows in data.values() for x in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate qid across splits')
        tr = {r['image_name'] for r in data['train']}
        va = {r['image_name'] for r in data['validation']}
        if tr & va:
            raise ValueError('train and validation share images')
        prov['test_images_also_in_train'] = len({r['image_name'] for r in data['test']} & tr)
        (staging / 'provenance.json').write_text(json.dumps(prov, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    for s, v in prov['splits'].items():
        print(f'vqa_rad/{s}: {v["kept"]} questions (closed {v["closed"]} / open {v["open"]}), {v["images"]} images', flush=True)
    return prov


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, help='data root; writes OUTPUT/vqa_rad')
    p.add_argument('--image-dir', help='original "VQA_RAD Image Folder" (optional; default: recover from HF mirror)')
    p.add_argument('--label-dir', help='cache for the original label JSONs (default OUTPUT/vqarad_original_labels)')
    p.add_argument('--cache-dir', help='HF download cache')
    args = p.parse_args()
    build(args.output, args.image_dir, args.label_dir, args.cache_dir)


if __name__ == '__main__':
    main()
