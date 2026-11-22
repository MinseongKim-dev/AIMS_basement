"""Download HF MedVQA mirrors and export the existing baseline JSONL/image interface."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import zipfile

from PIL import Image
from .data import digest, load_data

SOURCES = {
    'vqa_rad': 'flaviagiammarino/vqa-rad',
    'slake': 'BoKelvin/SLAKE',
    'pathvqa': 'flaviagiammarino/path-vqa',
}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
FORMAT_VERSION = 1


def get_field(row, keys, default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


class InvalidAnnotation(ValueError):
    """A source row with unusable content, not a missing/unsupported schema."""


def normalize(row, dataset, split, index):
    language = get_field(row, ('q_lang', 'language', 'lang'))
    if dataset == 'slake' and language is not None:
        if str(language).strip().lower() not in ('en', 'eng', 'english'):
            return None
    question = get_field(row, ('question', 'query'))
    answer = get_field(row, ('answer', 'label'))
    if not any(k in row for k in ('question', 'query')) or not any(k in row for k in ('answer', 'label')):
        raise ValueError(f'{dataset}/{split} row {index}: question/answer columns missing; columns={list(row)}')
    if not isinstance(question, str) or not question.strip():
        raise InvalidAnnotation('empty_or_nontext_question')
    if answer is None or not str(answer).strip():
        raise InvalidAnnotation('empty_answer')
    answer = str(answer).strip()
    if answer.lower() in ('yes', 'no'):
        answer = answer.lower()
    kind = get_field(row, ('answer_type', 'question_answer_type'))
    if kind is None:
        kind = 'closed' if answer.lower() in ('yes', 'no') else 'open'
    else:
        kind = str(kind).strip().lower()
        kind = {'close': 'closed', 'closed-ended': 'closed', 'open-ended': 'open'}.get(kind, kind)
        if kind not in ('closed', 'open'):
            raise ValueError(f'Unsupported answer_type={kind!r}; do not guess numeric category encodings')
    source_id = str(get_field(row, ('qid', 'question_id', 'q_id', 'id'), index))
    # Mirror IDs can be local to a split; preserve the original separately.
    return {'id': f'{split}:{source_id}', 'source_id': source_id, 'source_row': index,
            'dataset': dataset, 'split': split, 'question': question,
            'answer': answer, 'answer_type': kind}


def open_image(value, resolver=None):
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, dict):
        if value.get('bytes') is not None:
            with Image.open(io.BytesIO(value['bytes'])) as im:
                return im.convert('RGB')
        value = value.get('path')
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file() and resolver:
            path = Path(resolver(str(value)))
        if path.is_file():
            with Image.open(path) as im:
                return im.convert('RGB')
    raise ValueError(f'Image not embedded or not found: {str(value)[:160]}')


def cached_paths(target, source, revision=None, config=None):
    path = target/'provenance.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    expected = (source, config, FORMAT_VERSION)
    actual = (manifest['source'], manifest.get('config'), manifest.get('format_version'))
    if actual != expected or (revision and revision not in (manifest['revision'], manifest.get('requested_revision'))):
        raise ValueError(f'{target}: source/config/revision changed; use a different --output directory')
    for file, checksum in manifest['files'].items():
        p = target/file
        if not p.is_file() or digest(p) != checksum:
            raise ValueError(f'Prepared data missing/modified: {p}; use a new output directory to rebuild')
    for filename in manifest['images']:
        image = target/'images'/filename
        if not image.is_file() or digest(image) != manifest['images'][filename]:
            raise ValueError(f'Prepared image missing/modified: {filename}; use a new output directory to rebuild')
    return {'images': str(target/'images'), **{s: str(target/f'{s}.jsonl') for s in manifest['splits']}}


def export(dataset, splits, target, source, revision, config=None,
           requested_revision=None, resolver=None):
    """Publish only a fully exported directory; retain HF's train/test/validation splits."""
    target = Path(target).resolve()
    if 'train' not in splits or 'test' not in splits:
        raise ValueError(f'{source} must supply train AND test; available splits={list(splits)}. No random split is created.')
    if target.exists():
        raise ValueError(f'Refusing to overwrite existing directory: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{dataset}-', dir=target.parent))
    try:
        (staging/'images').mkdir()
        image_hashes = {}
        manifest = {'format_version': FORMAT_VERSION, 'dataset': dataset,
                    'source': source, 'revision': revision, 'requested_revision': requested_revision,
                    'config': config, 'english_only': dataset == 'slake',
                    'split_policy': 'Preserve mirror splits; no random resplit; not verified identical to paper',
                    'annotation_policy': 'Quarantine empty/null answers and empty/nontext questions; never impute ground truth',
                    'splits': {}, 'files': {}, 'images': image_hashes}
        for source_split, records in splits.items():
            split = 'validation' if source_split in ('val', 'valid', 'dev') else source_split
            if split not in ('train', 'validation', 'test'):
                continue
            if split in manifest['splits']:
                raise ValueError(f'Multiple source splits map to {split}')
            raw_count = count = invalid_count = language_count = 0
            closed = 0
            columns = None
            rejected_path = staging/f'{split}.rejected.jsonl'
            with (staging/f'{split}.jsonl').open('w', encoding='utf-8') as f, rejected_path.open('w', encoding='utf-8') as rejected:
                for index, row in enumerate(records):
                    raw_count += 1
                    columns = columns or sorted(row)
                    try:
                        out = normalize(row, dataset, split, index)
                    except InvalidAnnotation as error:
                        invalid_count += 1
                        audit = {'dataset': dataset, 'source_split': source_split, 'source_row': index,
                                 'source_id': str(get_field(row, ('qid','question_id','q_id','id'), index)),
                                 'reason': str(error), 'question': get_field(row, ('question','query')),
                                 'answer': get_field(row, ('answer','label')),
                                 'image_reference': str(get_field(row, ('img_name','image_name','image_path','img_id'), 'embedded')),
                                 'language': get_field(row, ('q_lang','language','lang'))}
                        rejected.write(json.dumps(audit, ensure_ascii=False, default=str)+'\n')
                        continue
                    if out is None:
                        language_count += 1
                        continue
                    value = get_field(row, ('image', 'img', 'img_name', 'image_name', 'image_path', 'img_id'))
                    image = open_image(value, resolver)
                    # Same pixels share a lossless image, including across train/test.
                    identity = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                    filename = identity+'.png'
                    if filename not in image_hashes:
                        image.save(staging/'images'/filename, format='PNG')
                        image_hashes[filename] = digest(staging/'images'/filename)
                    out['image'] = filename
                    out['source_split'] = source_split
                    f.write(json.dumps(out, ensure_ascii=False)+'\n')
                    count += 1
                    closed += out['answer_type'] == 'closed'
            if count == 0:
                raise ValueError(f'{source}/{source_split}: no usable questions after filtering')
            load_data(staging/f'{split}.jsonl', split)
            manifest['splits'][split] = {'source_split': source_split, 'raw': raw_count,
                                        'kept': count, 'open': count-closed, 'closed': closed,
                                        'columns': columns, 'filtered_language': language_count,
                                        'invalid_annotations': invalid_count,
                                        'rejected_file': rejected_path.name}
            manifest['files'][f'{split}.jsonl'] = digest(staging/f'{split}.jsonl')
            manifest['files'][rejected_path.name] = digest(rejected_path)
            print(f'{dataset}/{split}: kept {count}/{raw_count}; language excluded={language_count}; '
                  f'invalid annotations={invalid_count} (audit: {rejected_path.name})', flush=True)
        # Sharing images is allowed; record exact duplicate questions without changing the source split.
        train = load_data(staging/'train.jsonl', 'train')
        test = load_data(staging/'test.jsonl', 'test')
        keys = lambda rows: {(r['image'], r['question'].strip().lower()) for r in rows}
        manifest['train_test_duplicate_questions'] = len(keys(train) & keys(test))
        manifest['train_test_shared_images'] = len({r['image'] for r in train} & {r['image'] for r in test})
        (staging/'provenance.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return cached_paths(target, source, revision, config)


def unpack_assets(archives, destination):
    """Only regular image/JSON assets; never extract links or archive traversal paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    def copy_member(name, stream):
        part = PurePosixPath(name.replace('\\', '/'))
        if part.is_absolute() or '..' in part.parts:
            raise ValueError(f'Unsafe archive member: {name}')
        if part.suffix.lower() not in IMAGE_EXT | {'.json'}:
            return
        out = destination.joinpath(*part.parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('wb') as f:
            shutil.copyfileobj(stream, f)
    for archive in archives:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Archive symlinks are unsupported')
                    with z.open(member) as stream:
                        copy_member(member.filename, stream)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                for member in t:
                    if member.isfile():
                        with t.extractfile(member) as stream:
                            copy_member(member.name, stream)
        else:
            raise ValueError(f'Unsupported image archive: {archive}')


def asset_resolver(repo, revision, cache_dir):
    """Lazy fallback for mirrors with img_name instead of embedded HF Image columns."""
    roots = []
    by_basename = None
    def resolve(name):
        nonlocal by_basename
        if not roots:
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(repo, repo_type='dataset', revision=revision, cache_dir=cache_dir,
                         allow_patterns=['*.png','*.jpg','*.jpeg','*.PNG','*.JPG','*.json','*.zip','*.tar','*.tar.gz','*.tgz']))
            roots.append(root)
            archives = [p for p in root.rglob('*') if p.is_file() and
                        (p.name.lower().endswith(('.zip','.tar','.tar.gz','.tgz')))]
            if archives:
                extracted = Path(cache_dir or tempfile.gettempdir())/'medvqa_assets'/revision
                unpack_assets(archives, extracted)
                roots.append(extracted)
        relative = PurePosixPath(name.replace('\\','/'))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError(f'{repo}: invalid relative image path: {name!r}')
        for root in roots:
            direct = root.joinpath(*relative.parts)
            if direct.is_file():
                return str(direct)
        if by_basename is None:
            # Index once, rather than traversing all assets for every question.
            by_basename = {}
            for root in roots:
                for path in root.rglob('*'):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                        by_basename.setdefault(path.name, []).append(path)
        candidates = by_basename.get(relative.name, [])
        # SLAKE uses xmlab1/source.jpg, xmlab2/source.jpg, ... . Archive
        # prefixes may vary, but the complete annotated relative path must match.
        # Compare before resolve(): HF cache symlinks resolve to extensionless blobs.
        matches = [p for p in candidates if p.parts[-len(relative.parts):] == relative.parts]
        unique = sorted({p.resolve() for p in matches})
        if len(unique) != 1:
            raise ValueError(f'{repo}: image {name!r} has {len(unique)} full-path matches '
                             f'({len(candidates)} basename candidates). '
                             'Source may not include the required images or has ambiguous archive paths.')
        return str(unique[0])
    return resolve


def load_raw_splits(repo, revision, cache_dir):
    """Fallback for JSON/image archives without a load_dataset-compatible layout."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo, repo_type='dataset', revision=revision, cache_dir=cache_dir,
                                 allow_patterns=['*.json','*.zip','*.tar','*.tar.gz','*.tgz']))
    roots = [root]
    archives = [p for p in root.rglob('*') if p.is_file() and
                p.name.lower().endswith(('.zip','.tar','.tar.gz','.tgz'))]
    if archives:
        extracted = Path(cache_dir or tempfile.gettempdir())/'medvqa_raw'/revision
        unpack_assets(archives, extracted)
        roots.append(extracted)
    splits = {}
    for split, filenames in {'train':('train.json',), 'test':('test.json',),
                             'validation':('validation.json','val.json','valid.json','dev.json')}.items():
        candidates = sorted({p.resolve() for root in roots for name in filenames
                             for p in root.rglob(name) if p.is_file()})
        if len(candidates) > 1:
            raise ValueError(f'{repo}: ambiguous {split} JSON annotations: {len(candidates)} files')
        if candidates:
            records = json.loads(candidates[0].read_text(encoding='utf-8'))
            if isinstance(records, dict):
                records = records.get('data', records.get('annotations'))
            if not isinstance(records, list):
                raise ValueError(f'{repo}: {split} annotation must be a JSON array')
            splits[split] = records
    if not {'train','test'} <= splits.keys():
        raise ValueError(f'{repo}: no unambiguous train.json/test.json fallback found')
    return splits


def download_one(dataset, output, source=None, revision=None, config=None, cache_dir=None):
    source = source or SOURCES[dataset]
    target = Path(output).resolve()/dataset
    existing = cached_paths(target, source, revision, config)
    if existing:
        print(f'{dataset}: reusing prepared Drive data ({target})', flush=True)
        return existing
    if target.exists():
        raise ValueError(f'{target} exists without a complete provenance manifest; use a different output root')
    try:
        from datasets import load_dataset, Image as HFImage
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError('Install requirements-colab.txt first (datasets==2.21.0 required)') from error
    try:
        sha = HfApi().dataset_info(source, revision=revision or 'main').sha
        print(f'Downloading {dataset}: {source}@{sha}', flush=True)
        try:
            splits = load_dataset(source, name=config, revision=sha, cache_dir=cache_dir, trust_remote_code=False)
        except (ValueError, RuntimeError, FileNotFoundError) as load_error:
            if config:
                raise
            print(f'Parquet/standard loader unavailable: {load_error}. Trying raw train/test JSON.', flush=True)
            splits = load_raw_splits(source, sha, cache_dir)
        print(f'{source}: splits={list(splits)}', flush=True)
        # Decode with Pillow ourselves; support path and byte payloads consistently.
        for key in list(splits):
            for column, feature in getattr(splits[key], 'features', {}).items():
                if isinstance(feature, HFImage):
                    splits[key] = splits[key].cast_column(column, HFImage(decode=False))
        return export(dataset, splits, target, source, sha, config, revision,
                      resolver=asset_resolver(source, sha, cache_dir))
    except Exception as error:
        raise RuntimeError(f'{dataset}: HF source {source} could not be prepared: {error}. '
                           'No alternative split or dataset was silently substituted. '
                           'Check the dataset card/access or use --source with a compatible mirror.') from error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, help='Persistent prepared dataset root (e.g. Drive/urp_data/hf)')
    p.add_argument('--datasets', nargs='+', choices=tuple(SOURCES), default=list(SOURCES))
    p.add_argument('--cache-dir', help='Optional HF download/cache directory')
    p.add_argument('--source', help='Override HF repo ID; use with exactly one dataset')
    p.add_argument('--revision', help='Pin a HF source commit/ref; exactly one dataset')
    p.add_argument('--config', help='HF dataset config; exactly one dataset')
    args = p.parse_args()
    if len(args.datasets) != 1 and any((args.source, args.revision, args.config)):
        p.error('--source/--revision/--config require exactly one --datasets entry')
    paths = {}
    for dataset in args.datasets:
        paths[dataset] = download_one(dataset, args.output, args.source, args.revision, args.config, args.cache_dir)
    path = Path(args.output)/'data_paths.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(paths, indent=2)+'\n')
    os.replace(temp, path)
    print(f'Ready: {path}', flush=True)


if __name__ == '__main__':
    main()
