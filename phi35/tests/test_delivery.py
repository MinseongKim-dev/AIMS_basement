"""Code zip, run notebook, GPU-check verdict and the no-5.x-patch rule (CPU). Run: python phi35/tests/test_delivery.py"""
import contextlib
import importlib
import importlib.util
import io
import json
import re
import shlex
import tempfile
import zipfile
from pathlib import Path

from _util import PHI35, REPO, run

NOTEBOOK = PHI35 / 'Phi35V_run.ipynb'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zip_contents():
    pack = _load('pack_phi35_code', REPO / 'tools' / 'pack_phi35_code.py')
    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            pack.main(['--out', d])
        with zipfile.ZipFile(Path(d) / 'phi35-code.zip') as z:
            names = set(z.namelist())
        assert re.search(r'sha256 [0-9a-f]{64}', printed.getvalue())
    required = {'common/vqa_scoring.py', 'common/gating.py', 'common/highlights.py', 'common/build_rag_testjson.py',
                'phi35/phi35cfg/infer.py', 'phi35/phi35cfg/train.py', 'phi35/tests/gpu_check.py',
                'data_common/vqarad_img/testset.json', 'tools/fetch_author_assets.py', 'tools/release_manifest.py',
                'manifests/checkpoints.json', 'requirements.txt', 'phi35/requirements.txt'}
    assert required <= names, required - names
    assert not [n for n in names if '__pycache__' in n or 'assets/' in n or n.startswith('phi3/')
                or n.endswith('.ipynb')]


def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return nb['cells']


def test_notebook_clean():
    cells = _cells()
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in cells if c['cell_type'] == 'code')
    text = ''.join(''.join(c['source']) for c in cells)
    assert 'Expert-CFG.git' not in text and 'release_manifest.py verify' in text
    # the repository is public: the notebook clones it (no token, no zip upload) and prints the commit it runs
    assert 'https://github.com/MinseongKim-dev/AIMS_basement.git' in text and 'git clone' in text
    assert "REPO_REF = 'phi35cfg-modularization'" in text and 'rev-parse HEAD' in text
    assert 'CODE_SHA256' not in text and 'ghp_' not in text and 'GITHUB_TOKEN' not in text


def test_notebook_rag_default_is_existing_location():
    settings = next(''.join(c['source']) for c in _cells() if c['cell_type'] == 'code')
    assert "/content/drive/MyDrive/ExpertCFG_replication/rag_{DATASET}_peir_roco_trainqa_v2.json" in settings


def _commands():
    text = '\n'.join(''.join(c['source']) for c in _cells() if c['cell_type'] == 'code')
    text = re.sub(r'\\\n\s*', ' ', text)
    for line in text.split('\n'):
        m = re.search(r'!python (-m phi35cfg\.(\w+)|tests/gpu_check\.py) (.*)$', line.strip())
        if m:
            args = m.group(3).replace('{DATASET}', 'vqarad').replace('{RUNS}', '1=x').replace('{RESUME}', '')
            yield m.group(2) or 'gpu_check', shlex.split(re.sub(r'\{[^}]*\}', 'x', args))


def test_notebook_commands_parse():
    seen = []
    for name, argv in _commands():
        module = (_load('gpu_check', PHI35 / 'tests' / 'gpu_check.py') if name == 'gpu_check'
                  else importlib.import_module(f'phi35cfg.{name}'))
        parsed = module.build_parser().parse_args(argv)
        if name == 'infer':
            module._check(module.build_parser(), parsed)
            seen.append(f'infer:{parsed.mode}:{"gated" if parsed.top_percent else "all"}')
        else:
            seen.append(f'{name}:{getattr(parsed, "command", "")}')
    assert {'gpu_check:', 'train:', 'evaluate:select', 'evaluate:report', 'infer:entropy:all', 'infer:baseline:all',
            'infer:rag:all', 'infer:rag:gated', 'infer:expert_cfg:gated'} <= set(seen), sorted(set(seen))


def test_gpu_check_verdict():
    check = _load('gpu_check', PHI35 / 'tests' / 'gpu_check.py')
    good = {'neutral': [True] * 5, 'logits': [(0.0, 3.0)] * 5, 'cache': [True] * 3, 'repro': {'n': 50, 'same': 48}}
    assert check.verdict(good, full=False) == []
    assert check.verdict({**good, 'repro': {'n': 451, 'same': 444}}, full=True) == []
    assert check.verdict({**good, 'repro': {'n': 451, 'same': 430}}, full=True)
    for bad in ({'neutral': [True, False]}, {'neutral': []}, {'logits': [(0.5, 0.5)]}, {'cache': [False]},
                {'cache': []}):
        assert check.verdict({**good, **bad}, full=False), bad


def test_no_version_branches():
    sources = sorted((PHI35 / 'phi35cfg').glob('*.py')) + [PHI35 / 'tests' / 'gpu_check.py']
    assert len(sources) >= 8
    for path in sources:
        text = path.read_text(encoding='utf-8')
        for word in ('__version__', 'DynamicCache', 'rope_scaling', 'phi3cfg.'):
            assert word not in text.replace('phi3/phi3cfg', ''), f'{word} in {path.name}'


if __name__ == '__main__':
    run(globals())
