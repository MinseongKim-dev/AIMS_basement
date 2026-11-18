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
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    missing = 0
    for dest, paths in FILES.items():
        for p in paths:
            out = ROOT / dest / Path(p).name
            if a.check:
                ok = out.exists()
                print(('있음 ' if ok else '없음 ') + str(out.relative_to(ROOT)))
                missing += (not ok)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{p}'
            data = urllib.request.urlopen(url, timeout=60).read()
            out.write_bytes(data)
            print(f'받음 {out.relative_to(ROOT)}  {len(data)} bytes  sha256={hashlib.sha256(data).hexdigest()[:16]}')
    if a.check and missing:
        sys.exit(1)


if __name__ == '__main__':
    main()
