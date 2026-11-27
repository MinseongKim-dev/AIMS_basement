"""Inference CLI, entropy files and model-folder preparation with fakes (CPU). Run: python phi35/tests/test_infer.py"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

import torch

import _fakes as F
from _util import REPO, raises, run

from phi35cfg import data, infer, model as model_lib, yesno  # noqa: E402

N = 451


@contextlib.contextmanager
def patched(obj, **values):
    old = {k: getattr(obj, k) for k in values}
    for k, v in values.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _fixture(d):
    """An image folder, a retrieval file and an entropy file for the real VQA-RAD test split."""
    from PIL import Image
    d = Path(d)
    rows, _ = data.load_split('vqarad', 'test', check_images=False)
    images = d / 'images'
    images.mkdir()
    for name in {r['image'] for r in rows}:
        Image.new('RGB', (2, 2)).save(images / name, format='JPEG')
    rag = [{'img_id': r['image'], 'question': r['question'], 'answer': r['answer'], 'answer_type': r['answer_type'],
            'top_k_captions': [f"caption {r['idx']}."], 'highlights': ['caption']} for r in rows]
    (d / 'rag.json').write_text(json.dumps(rag), encoding='utf-8')
    entropy = {'dataset': 'vqarad', 'predictions': 'base-v1', 'adapter': 'ad-v1',
               'entropy': {str(i): i / 1000 for i in range(N)}}
    (d / 'entropy.json').write_text(json.dumps(entropy), encoding='utf-8')
    return rows, images


@contextlib.contextmanager
def fake_model(calls):
    def generate(model, processor, image, question, reference=None, highlights=None, cfg=None,
                 max_new_tokens=64, use_cache=True):
        calls.append({'question': question, 'reference': reference, 'highlights': highlights, 'cfg': cfg})
        return {'prediction': 'p', 'cfg_applied': cfg is not None, 'highlight_tokens': 0}

    with patched(model_lib, load=lambda *a, **k: ('model', 'processor', {'gpu': 'fake'})):
        with patched(infer.generation, generate=generate, answer_entropy=lambda *a: 0.5):
            yield


def _args(d, mode, out, *extra):
    d = Path(d)
    return ['--mode', mode, '--dataset', 'vqarad', '--image-root', str(d / 'images'), '--base-dir', str(d / 'base'),
            '--adapter', 'adapter-name-only', '--output', str(d / out), *extra]


def _gated(d):
    return ['--retrieval', str(Path(d) / 'rag.json'), '--entropy', str(Path(d) / 'entropy.json'), '--top-percent', '5']


def _quiet(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        infer.main(argv)


def test_select_ids():
    assert len(infer.select_ids({i: .1 for i in range(451)}, 451)) == 23
    assert len(infer.select_ids({i: .1 for i in range(1061)}, 1061)) == 53
    assert infer.select_ids({0: .1, 1: .9, 2: .9, 3: None}, 4, percent=50) == [1, 2]
    assert infer.select_ids({0: None, 1: .2}, 2, percent=100) == [1]


def test_load_entropy():
    good = {'dataset': 'slake', 'predictions': 'p-v1', 'entropy': {'0': 0.1, '1': None}}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'e.json'
        p.write_text(json.dumps(good), encoding='utf-8')
        assert infer.load_entropy(p, 'slake', 2, 'p-v1') == {0: 0.1, 1: None}
        assert infer.load_entropy(p, 'slake', 2, None) == {0: 0.1, 1: None}
        for dataset, n, preds in (('vqarad', 2, None), ('slake', 3, None), ('slake', 1, None), ('slake', 2, 'p-v2')):
            with raises(ValueError):
                infer.load_entropy(p, dataset, n, preds)


def test_cli_requires():
    with tempfile.TemporaryDirectory() as d:
        bad = [_args(d, 'rag', 'o.jsonl'),
               _args(d, 'expert_cfg', 'o.jsonl', '--retrieval', 'r.json'),
               _args(d, 'rag', 'o.jsonl', '--retrieval', 'r.json', '--top-percent', '5'),
               _args(d, 'entropy', 'o.json', '--predictions', 'p.json')]
        for argv in bad:
            with contextlib.redirect_stderr(io.StringIO()), raises(SystemExit):
                infer.main(argv)


def test_expert_cfg_run():
    with tempfile.TemporaryDirectory() as d:
        rows, _ = _fixture(d)
        calls = []
        with fake_model(calls):
            _quiet(_args(d, 'expert_cfg', 'cfg.jsonl', *_gated(d)))
            got = data.read_predictions(Path(d) / 'cfg.jsonl')
            assert sorted(got) == list(range(N - 23, N)) and len(calls) == 23
            first = calls[0]
            assert first['cfg'].gamma == 1.5 and first['reference'] == f'caption {N - 23}.'
            expected = infer.compliant_highlights(rows[N - 23]['question'], rows[N - 23]['answer_type'],
                                                  first['reference'], ['caption'])
            assert first['highlights'] == expected
            rec = got[N - 1]
            assert rec['mode'] == 'expert_cfg' and rec['cfg_applied'] is True and rec['entropy'] is None
            assert (rec['question'], rec['answer']) == (rows[N - 1]['question'], rows[N - 1]['answer'])
            _quiet(_args(d, 'expert_cfg', 'cfg.jsonl', *_gated(d)))
            assert len(calls) == 23                                       # resumed: nothing left
            with raises(ValueError):
                _quiet(_args(d, 'expert_cfg', 'cfg.jsonl', *_gated(d), '--gamma', '1.3'))
        meta = json.loads((Path(d) / 'cfg.jsonl.meta.json').read_text(encoding='utf-8'))
        assert meta['cfg']['gamma'] == 1.5 and meta['runtime'] == {'gpu': 'fake'}
        assert len(meta['retrieval_sha256']) == 64 and len(meta['entropy_sha256']) == 64


def test_rag_and_baseline_runs():
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        calls = []
        with fake_model(calls):
            _quiet(_args(d, 'rag', 'rag.jsonl', '--retrieval', str(Path(d) / 'rag.json'), '--limit', '3'))
            assert [c['reference'] for c in calls] == ['caption 0.', 'caption 1.', 'caption 2.']
            assert all(c['cfg'] is None and c['highlights'] is None for c in calls)
            calls.clear()
            _quiet(_args(d, 'rag', 'rag.jsonl', '--retrieval', str(Path(d) / 'rag.json'), '--limit', '4'))
            assert len(calls) == 1                                        # --limit is not part of the configuration
            calls.clear()
            _quiet(_args(d, 'baseline', 'base.jsonl', '--limit', '2'))
            assert all(c['reference'] is None and c['cfg'] is None for c in calls)
            assert data.read_predictions(Path(d) / 'base.jsonl')[0]['entropy'] == 0.5


def test_entropy_mode():
    with tempfile.TemporaryDirectory() as d:
        rows, _ = _fixture(d)
        preds = [{**r, 'prediction': 'x'} for r in rows]
        (Path(d) / 'preds.json').write_text(json.dumps(preds), encoding='utf-8')
        extra = ['--predictions', str(Path(d) / 'preds.json'), '--predictions-name', 'base-v1',
                 '--adapter-name', 'ad-v1']
        seen = []
        with fake_model([]), patched(infer.generation, answer_entropy=lambda *a: seen.append(a) or 0.25):
            _quiet(_args(d, 'entropy', 'h.json', *extra, '--limit', '2'))
            with raises(ValueError):                                      # partial file is not usable for gating
                infer.load_entropy(Path(d) / 'h.json', 'vqarad', N, 'base-v1')
            _quiet(_args(d, 'entropy', 'h.json', *extra))
            assert len(seen) == N
            with raises(ValueError):
                _quiet(_args(d, 'entropy', 'h.json', *extra[:3], 'base-v2', *extra[4:]))
        got = json.loads((Path(d) / 'h.json').read_text(encoding='utf-8'))
        assert got['model'] == 'phi35' and got['mode'] == 'teacher_forcing_on_baseline_predictions'
        assert got['n'] == N and got['env']['gpu'] == 'fake' and got['definition'].startswith('eq4')
        assert infer.load_entropy(Path(d) / 'h.json', 'vqarad', N, 'base-v1')[7] == 0.25


def test_yesno_mode_runs_only_yes_no_questions_and_resumes():
    rows, _ = data.load_split('vqarad', 'val', check_images=False)
    with tempfile.TemporaryDirectory() as d:
        from PIL import Image
        images = Path(d) / 'images'
        images.mkdir()
        for name in {r['image'] for r in rows}:
            Image.new('RGB', (2, 2)).save(images / name, format='JPEG')
        calls = []

        class Proc:
            tokenizer = 'tokenizer'

        def first_token_logp(model, processor, image, question):
            calls.append({'question': question})
            return torch.tensor([.4, .3, .2, .1]).log()

        out = Path(d) / 'yn.jsonl'
        with patched(model_lib, load=lambda *a, **k: ('model', Proc(), {'gpu': 'fake'})), \
                patched(infer.generation, first_token_logp=first_token_logp), \
                patched(infer.yesno, label_token_ids=lambda tok: {'yes': [0, 1], 'no': [2]}):
            _quiet(_args(d, 'yesno', 'yn.jsonl', '--split', 'val', '--limit', '3'))
            first = data.read_predictions(out)
            _quiet(_args(d, 'yesno', 'yn.jsonl', '--split', 'val'))
            with contextlib.redirect_stderr(io.StringIO()), raises(SystemExit):
                _quiet(_args(d, 'yesno', 'test.jsonl', '--split', 'test'))
        lines = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
        assert len(first) == 3 and len(lines) == 184 and len({x['idx'] for x in lines}) == 184
        assert len(calls) == 184 and all(set(c) == {'question'} for c in calls)    # the answer is never passed
        assert all(yesno.label_of(x['answer']) for x in lines)
        assert set(lines[0]) == {'idx', 'question', 'answer', 'answer_type', 'mode', 'entropy_first', 'label_probs'}
        assert lines[0]['mode'] == 'yesno' and abs(lines[0]['label_probs']['yes']['1'] - .3) < 1e-6
        assert json.loads(Path(str(out) + '.meta.json').read_text(encoding='utf-8'))['split'] == 'val'
        assert not (Path(d) / 'test.jsonl').exists()


def test_first_token_logp():
    model, processor = F.Model(), F.Processor()
    logp = infer.generation.first_token_logp(model, processor, 'image', 'finding?')
    assert logp.shape == (F.V,) and abs(float(logp.exp().sum()) - 1) < 1e-5 and int(logp.argmax()) == 2
    assert len(model.calls) == 1 and model.calls[0]['n'] == len(F.PROMPT_IDS)
    assert processor.calls[0][0] == '<user><|image_1|>\nfinding?<end><assistant>'


def _assets(d):
    assets = Path(d) / 'assets'
    (assets / 'author_src' / 'src' / 'highlighter_modules').mkdir(parents=True)
    (assets / 'author_src' / 'src' / 'highlighter_modules' / 'utils.py').write_text('x = 1', encoding='utf-8')
    for name in ('processing_phi3_v_cfg.py', 'processor_config.json'):
        (assets / name).write_text(f'author {name}', encoding='utf-8')
    return assets


def test_prepare_base_dir_overwrites_processor():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / 'base'
        base.mkdir()
        (base / 'config.json').write_text('{}', encoding='utf-8')
        (base / 'processor_config.json').write_text('hub version', encoding='utf-8')
        assets = _assets(d)
        before = list(sys.path)
        with patched(model_lib, ASSETS=assets):
            assert model_lib.prepare_base_dir(base) == base
        assert (base / 'processor_config.json').read_text(encoding='utf-8') == 'author processor_config.json'
        assert (base / 'processing_phi3_v_cfg.py').exists()
        assert str(assets / 'author_src') in sys.path
        sys.path[:] = before


def test_prepare_base_dir_missing_assets():
    with tempfile.TemporaryDirectory() as d, patched(model_lib, ASSETS=Path(d) / 'none'):
        with raises(FileNotFoundError, match='fetch_author_assets'):
            model_lib.prepare_base_dir(Path(d) / 'base')


def test_enable_sdpa_idempotent():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'modeling_phi3_v.py'
        f.write_text('class A:\n    _supports_sdpa = False\n', encoding='utf-8')
        model_lib.enable_sdpa(d)
        model_lib.enable_sdpa(d)
        assert f.read_text(encoding='utf-8') == 'class A:\n    _supports_sdpa = True\n'
        f.write_text('no flag', encoding='utf-8')
        with raises(RuntimeError):
            model_lib.enable_sdpa(d)


def test_resolve_adapter_local():
    with tempfile.TemporaryDirectory() as d:
        nested = Path(d) / 'a' / 'b'
        nested.mkdir(parents=True)
        (nested / 'adapter_config.json').write_text('{}', encoding='utf-8')
        assert Path(model_lib.resolve_adapter(str(nested))) == nested
        assert Path(model_lib.resolve_adapter(d)) == nested


def test_sdpa_is_training_only():
    with raises(ValueError):
        model_lib.load('base', 'adapter', attn='sdpa')


def test_fetch_tool_covers_highlight_module():
    spec = importlib.util.spec_from_file_location('fetch_author_assets', REPO / 'tools' / 'fetch_author_assets.py')
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    assert 'phi35/assets/author_src/src/highlighter_modules/utils.py' in fetch.targets()
    assert 'phi35/assets/author_src/src/__init__.py' in fetch.targets()
    assert 'phi35/assets/processor_config.json' in fetch.targets()


if __name__ == '__main__':
    run(globals())
