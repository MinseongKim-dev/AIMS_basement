"""체크포인트 · 큰 파일을 Hugging Face Hub(비공개)와 주고받고 manifests/ 에 위치와 해시를 기록한다.

토큰은 환경변수 HF_TOKEN 또는 `huggingface-cli login` 세션에서만 읽는다. 코드나 인자로 받지 않는다.

  # 올리기: 폴더를 비공개 레포에 올리고, 커밋 해시와 파일별 sha256 을 매니페스트에 기록
  python tools/hf_sync.py upload --name phi35-vqarad-e10-s42 --repo-id <조직>/<이름> --path <로컬 폴더> \
      --protocol compliant --note "train만 학습, val 선택 ep6" [--repo-type model|dataset] [--dry-run]

  # 받기: 매니페스트의 revision 으로 받고 sha256 을 검증
  python tools/hf_sync.py download --name phi35-vqarad-e10-s42 --out <받을 폴더>

  # 매니페스트 확인
  python tools/hf_sync.py list
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / 'manifests' / 'checkpoints.json'
SKIP = {'.git', '__pycache__', '.ipynb_checkpoints'}


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(bs), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_folder(folder):
    folder = Path(folder)
    files = {}
    for p in sorted(folder.rglob('*')):
        if p.is_file() and not (set(p.relative_to(folder).parts) & SKIP):
            files[p.relative_to(folder).as_posix()] = sha256(p)
    return files


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def cmd_upload(a):
    files = hash_folder(a.path)
    if not files:
        sys.exit(f'올릴 파일이 없습니다: {a.path}')
    print(f'{len(files)}개 파일, 총 {sum(Path(a.path, k).stat().st_size for k in files) / 1e6:.1f} MB')
    for k, v in files.items():
        print(f'  {v[:12]}  {k}')
    if a.dry_run:
        print('(dry-run: 업로드와 매니페스트 기록을 하지 않았습니다)')
        return
    from huggingface_hub import HfApi
    api = HfApi()                                  # 토큰은 HF_TOKEN 또는 로그인 세션에서만
    api.create_repo(a.repo_id, repo_type=a.repo_type, private=True, exist_ok=True)
    info = api.upload_folder(repo_id=a.repo_id, repo_type=a.repo_type, folder_path=a.path,
                             commit_message=f'upload {a.name}', ignore_patterns=['__pycache__/*', '.git/*'])
    m = load_manifest()
    m[a.name] = {'repo_id': a.repo_id, 'repo_type': a.repo_type, 'revision': info.oid, 'files': files,
                 'protocol': a.protocol, 'note': a.note}
    save_manifest(m)
    print('올림. revision =', info.oid, '→ manifests/checkpoints.json 에 기록했습니다 (커밋해서 공유하세요)')


def cmd_download(a):
    m = load_manifest()
    if a.name not in m:
        sys.exit(f'매니페스트에 없습니다: {a.name} (python tools/hf_sync.py list)')
    e = m[a.name]
    from huggingface_hub import snapshot_download
    out = snapshot_download(repo_id=e['repo_id'], repo_type=e['repo_type'], revision=e['revision'], local_dir=a.out)
    bad = [k for k, h in e['files'].items() if not Path(out, k).exists() or sha256(Path(out, k)) != h]
    if bad:
        sys.exit(f'sha256 불일치 또는 누락: {bad[:5]}')
    print(f'받고 검증했습니다: {out} ({len(e["files"])}개 파일, revision {e["revision"][:10]})')


def cmd_list(a):
    for k, e in load_manifest().items():
        print(f'{k}: {e["repo_id"]} @ {e["revision"][:10]} | {e.get("protocol", "?")} | {len(e["files"])}개 파일 | {e.get("note", "")}')
    if not load_manifest():
        print('(비어 있음)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('upload')
    u.add_argument('--name', required=True)
    u.add_argument('--repo-id', required=True)
    u.add_argument('--path', required=True)
    u.add_argument('--repo-type', choices=('model', 'dataset'), default='model')
    u.add_argument('--protocol', choices=('compliant', 'relaxed'), required=True)
    u.add_argument('--note', default='')
    u.add_argument('--dry-run', action='store_true')
    u.set_defaults(fn=cmd_upload)
    d = sub.add_parser('download')
    d.add_argument('--name', required=True)
    d.add_argument('--out', required=True)
    d.set_defaults(fn=cmd_download)
    l = sub.add_parser('list')
    l.set_defaults(fn=cmd_list)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
