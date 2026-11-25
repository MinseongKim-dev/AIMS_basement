"""저자 레포(ecoxial2007/Expert-CFG)의 processor 파일을 고정 커밋에서 받는다.

저자 레포에는 라이선스가 없어서(GitHub API license: null) 파일을 이 레포에 복사해 올리지 않는다.
받은 파일은 phi3/assets, phi35/assets 에 저장되며 .gitignore 로 커밋에서 제외된다.

  python tools/fetch_author_assets.py            # 받기
  python tools/fetch_author_assets.py --check    # 이미 받은 파일이 있는지만 확인
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = 'ecoxial2007/Expert-CFG'
REVISION = 'edcb03c1ad406cc87171319d59a628ccefa9fbdc'   # 2026-11-17 에 확인한 main 의 커밋. 바꾸려면 결과 영향을 확인한다
FILES = {
    'phi35/assets': ['Phi-3.5-vision-instruct/processing_phi3_v_cfg.py', 'Phi-3.5-vision-instruct/processor_config.json'],
    'phi3/assets': ['Phi-3-vision-128k-instruct/processing_phi3_v_cfg.py', 'Phi-3-vision-128k-instruct/preprocessor_config.json'],
}
# phi35 의 processor 는 `from src.highlighter_modules.utils import txt_highlight_mask` 를 한다 → 같은 커밋에서 받아 경로 구조를 유지한다
SOURCES = {'phi35/assets/author_src/src/highlighter_modules/utils.py': 'src/highlighter_modules/utils.py'}
EMPTY = ['phi35/assets/author_src/src/__init__.py']
ROOT = Path(__file__).resolve().parent.parent


def targets():
    """{저장 경로(레포 기준): 저자 레포 경로 또는 None(빈 파일)}"""
    out = {f'{dest}/{Path(p).name}': p for dest, paths in FILES.items() for p in paths}
    out.update(SOURCES)
    out.update({dest: None for dest in EMPTY})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    missing = 0
    for dest, src in targets().items():
        out = ROOT / dest
        if a.check:
            ok = out.exists()
            print(('있음 ' if ok else '없음 ') + dest)
            missing += (not ok)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        data = b'' if src is None else urllib.request.urlopen(
            f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{src}', timeout=60).read()
        out.write_bytes(data)
        print(f'받음 {dest}  {len(data)} bytes  sha256={hashlib.sha256(data).hexdigest()[:16]}')
    if a.check and missing:
        sys.exit(1)


if __name__ == '__main__':
    main()
