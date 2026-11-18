"""공용으로 쓰는 체크포인트 · 큰 파일(릴리스 폴더)의 sha256 을 manifests/checkpoints.json 에 기록하고 검증한다.
저장 위치(공유 Drive, 랩 서버 공용 폴더 등)와 무관하게 쓴다. 위치는 --where 로 라벨만 적는다.

버전 고정이 없는 저장소에서도 같은 파일임을 보장하려고 아래 규칙을 쓴다.
  1) 릴리스 폴더(예: AIMS_shared/checkpoints/phi35/<이름>-v1)는 한 번 올리면 덮어쓰지 않는다. 바뀌면 이름에 -v2 를 붙여 새로 올린다.
  2) 올린 폴더의 파일별 sha256 을 이 도구로 기록해 레포에 커밋한다.
  3) 받은 쪽은 복사해 온 폴더를 이 도구로 검증한 뒤 쓴다.

  # 기록: 릴리스 폴더(로컬 또는 마운트된 Drive 경로)를 읽어 해시를 매니페스트에 쓴다
  python tools/release_manifest.py make --name phi35-vqarad-e10-s42-v1 --path <폴더> \
      --where "AIMS_shared/checkpoints/phi35/phi35-vqarad-e10-s42-v1" --protocol compliant --note "train만 학습, val 선택 ep6"

  # 검증: 받은(복사한) 폴더가 매니페스트와 같은지
  python tools/release_manifest.py verify --name phi35-vqarad-e10-s42-v1 --path <받은 폴더>

  python tools/release_manifest.py list

매니페스트에는 개인 PC 경로를 적지 않고 --where 로 준 라벨(공유 위치 안의 상대 경로)만 적는다.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / 'manifests' / 'checkpoints.json'
SKIP_PARTS = {'.git', '__pycache__', '.ipynb_checkpoints'}


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(bs), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_folder(folder):
    folder = Path(folder)
    out = {}
    for p in sorted(folder.rglob('*')):
        rel = p.relative_to(folder)
        if p.is_file() and not (set(rel.parts) & SKIP_PARTS):
            out[rel.as_posix()] = sha256(p)
    return out


def load():
    return json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}


def save(m):
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def cmd_make(a):
    files = hash_folder(a.path)
    if not files:
        sys.exit(f'파일이 없습니다: {a.path}')
    m = load()
    if a.name in m and m[a.name]['files'] != files:
        sys.exit(f'{a.name} 은 이미 다른 내용으로 기록되어 있습니다. 릴리스는 덮어쓰지 않습니다 — 이름에 -v2 처럼 새 버전을 붙이세요.')
    total = sum(Path(a.path, k).stat().st_size for k in files) / 1e6
    m[a.name] = {'where': a.where, 'files': files,
                 'protocol': a.protocol, 'note': a.note, 'recorded': datetime.date.today().isoformat()}
    save(m)
    print(f'{a.name}: {len(files)}개 파일, {total:.1f} MB 기록했습니다 → manifests/checkpoints.json (커밋해서 공유하세요)')
    for k, v in files.items():
        print(f'  {v[:12]}  {k}')


def cmd_verify(a):
    m = load()
    if a.name not in m:
        sys.exit(f'매니페스트에 없습니다: {a.name} (python tools/release_manifest.py list)')
    want = m[a.name]['files']
    got = hash_folder(a.path)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    bad = sorted(k for k in set(want) & set(got) if want[k] != got[k])
    if missing or bad:
        print('검증 실패')
        for k in missing: print('  없음      ', k)
        for k in bad: print('  해시 다름 ', k)
        sys.exit(1)
    print(f'검증 통과: {a.name} ({len(want)}개 파일)' + (f' | 매니페스트에 없는 파일 {len(extra)}개: {extra[:5]}' if extra else ''))


def cmd_list(a):
    m = load()
    for k, e in m.items():
        print(f'{k}: {e.get("where", "?")} | {e.get("protocol", "?")} | {len(e["files"])}개 파일 | {e.get("recorded", "")} | {e.get("note", "")}')
    if not m:
        print('(비어 있음)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    mk = sub.add_parser('make')
    mk.add_argument('--name', required=True)
    mk.add_argument('--path', required=True)
    mk.add_argument('--where', required=True, help='공유 위치 안의 상대 경로 라벨 (개인 PC 경로 금지). 예: AIMS_shared/checkpoints/phi35/<이름>-v1')
    mk.add_argument('--protocol', choices=('compliant', 'relaxed'), required=True)
    mk.add_argument('--note', default='')
    mk.set_defaults(fn=cmd_make)
    v = sub.add_parser('verify')
    v.add_argument('--name', required=True)
    v.add_argument('--path', required=True)
    v.set_defaults(fn=cmd_verify)
    sub.add_parser('list').set_defaults(fn=cmd_list)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
