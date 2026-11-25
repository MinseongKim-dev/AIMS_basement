"""Dataset splits, prediction files, question pairing and resumable run files.

A row is {'idx', 'image', 'question', 'answer', 'answer_type'}; idx is the position in the split.
"""
import hashlib
import json
import zipfile
from pathlib import Path

from . import REPO_ROOT

DATASETS = ('vqarad', 'slake')
SPLITS = ('train', 'val', 'test')
VQARAD_FILES = {'train': 'trainset.json', 'val': 'valset.json', 'test': 'testset.json'}
VQARAD_COUNTS = {'train': 2749, 'val': 315, 'test': 451}
SLAKE_REPO = 'BoKelvin/SLAKE'
SLAKE_FILES = {'train': 'train.json', 'val': 'validation.json', 'test': 'test.json'}


def to_rows(raw):
    rows = []
    for i, r in enumerate(raw):
        image = r.get('image_name') or r.get('img_name') or r.get('img_id')
        rows.append({'idx': i, 'image': image, 'question': str(r['question']).strip(),
                     'answer': str(r['answer']).strip(), 'answer_type': str(r['answer_type']).strip().upper()})
    return rows


def _type_counts(rows):
    return (sum(r['answer_type'] == 'CLOSED' for r in rows), sum(r['answer_type'] == 'OPEN' for r in rows))


def _slake(split, data_root):
    """Official English SLAKE split from the Hugging Face dataset."""
    from huggingface_hub import hf_hub_download
    root = Path(data_root)
    if not (root / 'imgs').is_dir():
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(hf_hub_download(SLAKE_REPO, 'imgs.zip', repo_type='dataset')) as z:
            z.extractall(root)
    path = hf_hub_download(SLAKE_REPO, SLAKE_FILES[split], repo_type='dataset')
    raw = [r for r in json.loads(Path(path).read_text(encoding='utf-8')) if r.get('q_lang') == 'en']
    return to_rows(raw), root / 'imgs'


def load_split(dataset, split, image_root=None, data_root=None, check_images=True):
    """Return (rows, image folder). Question counts of the team split are asserted."""
    if dataset not in DATASETS or split not in SPLITS:
        raise ValueError(f'unknown dataset/split: {dataset}/{split}')
    if dataset == 'vqarad':
        if image_root is None and check_images:
            raise ValueError('vqarad needs image_root (the VQA-RAD image folder)')
        path = REPO_ROOT / 'data_common' / 'vqarad_img' / VQARAD_FILES[split]
        rows = to_rows(json.loads(path.read_text(encoding='utf-8')))
        if len(rows) != VQARAD_COUNTS[split]:
            raise ValueError(f'vqarad {split}: {len(rows)} questions, expected {VQARAD_COUNTS[split]}')
        if split == 'test' and _type_counts(rows) != (272, 179):
            raise ValueError(f'vqarad test (closed, open) = {_type_counts(rows)}, expected (272, 179)')
        images = Path(image_root) if image_root is not None else None
    else:
        if data_root is None:
            raise ValueError('slake needs data_root (download folder)')
        rows, images = _slake(split, data_root)
        if split == 'test' and _type_counts(rows) != (416, 645):
            raise ValueError(f'slake test (closed, open) = {_type_counts(rows)}, expected (416, 645)')
    if check_images:
        missing = sorted({r['image'] for r in rows if not (images / r['image']).exists()})
        if missing:
            raise FileNotFoundError(f'{len(missing)} images missing under {images}, e.g. {missing[:3]}')
    return rows, images


def read_predictions(path):
    """{idx: record} from a released JSON list or a JSONL run file."""
    text = Path(path).read_text(encoding='utf-8')
    if text.lstrip().startswith('['):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = {}
    for r in records:
        i = int(r['idx'])
        if i in out:
            raise ValueError(f'{path}: duplicate idx {i}')
        out[i] = r
    return out


def _key(r):
    return str(r['question']).strip().lower(), str(r['answer']).strip().lower()


def check_records(rows, records, name):
    """Every record in {idx: record} must carry the question and answer of rows[idx]."""
    for i, rec in records.items():
        if not 0 <= i < len(rows):
            raise ValueError(f'{name}: idx {i} is outside this split ({len(rows)} questions)')
        if _key(rec) != _key(rows[i]):
            raise ValueError(f'{name}: idx {i} holds another question/answer — wrong dataset, split or order')


def align(rows, records, name):
    """Records in row order (by idx when every record has one, else by file order), verified for every row."""
    records = list(records)
    if len(records) != len(rows):
        raise ValueError(f'{name}: {len(records)} records for {len(rows)} questions')
    if all('idx' in r for r in records):
        by = {int(r['idx']): r for r in records}
        if sorted(by) != list(range(len(rows))):
            raise ValueError(f'{name}: idx values are not exactly 0..{len(rows) - 1}')
        records = [by[i] for i in range(len(rows))]
    check_records(rows, dict(enumerate(records)), name)
    return records


def file_sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def open_run(output, config):
    """Prepare a JSONL run file and return the idx already done.

    The configuration is stored next to it; resuming with a different one is refused
    ('runtime' is informational and ignored). A half-written last line is dropped.
    """
    output = Path(output)
    meta = output.with_name(output.name + '.meta.json')
    config = json.loads(json.dumps(config, default=str))
    has_records = output.exists() and output.stat().st_size > 0
    if meta.exists():
        stored = json.loads(meta.read_text(encoding='utf-8'))
        old, new = ({k: v for k, v in c.items() if k != 'runtime'} for c in (stored, config))
        if old != new:
            changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f'{output} was written with another configuration (differs: {changed}); use a new --output')
    elif has_records:
        raise ValueError(f'{output} has predictions but no {meta.name}; use a new --output')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    if not has_records:
        return set()
    lines = [line for line in output.read_text(encoding='utf-8').split('\n') if line.strip()]
    try:
        json.loads(lines[-1])
    except json.JSONDecodeError:
        lines.pop()
    done = {int(json.loads(line)['idx']) for line in lines}
    output.write_text(''.join(line + '\n' for line in lines), encoding='utf-8', newline='\n')
    return done


def append_record(output, record):
    with open(output, 'a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
