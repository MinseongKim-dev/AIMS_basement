"""Colab 은 비공개 레포를 clone 할 수 없으므로, phi35 실행에 필요한 코드와 분할 JSON 만 압축한다.

  python tools/pack_phi35_code.py --out <폴더>       # <폴더>/phi35-code.zip 생성, sha256 출력

만든 폴더를 AIMS_shared/code/phi35-v1/ 처럼 새 이름으로 올리고(덮어쓰지 않는다)
python tools/release_manifest.py make --name phi35-code-v1 --path <폴더> --where AIMS_shared/code/phi35-v1 --protocol compliant
로 기록한다. 실행 노트북의 CODE_SHA256 에는 여기서 출력한 값을 넣는다.
저자 파일(phi35/assets)은 넣지 않는다 — 노트북에서 tools/fetch_author_assets.py 로 받는다.
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ('common', 'phi35/phi35cfg', 'phi35/tests', 'data_common/vqarad_img', 'manifests')
FILES = ('tools/fetch_author_assets.py', 'tools/release_manifest.py', 'requirements.txt', 'phi35/requirements.txt')


def members():
    out = [ROOT / f for f in FILES]
    for folder in FOLDERS:
        out += [p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='압축 파일을 만들 폴더')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'phi35-code.zip'
    if target.exists():
        ap.error(f'{target} 이 이미 있습니다. 릴리스는 덮어쓰지 않습니다 — 새 폴더를 쓰세요')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in members():
            z.write(p, p.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f'{target.name}: {len(members())} files\nsha256 {digest}')


if __name__ == '__main__':
    main()
