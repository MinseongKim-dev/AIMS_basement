"""Checkpoint selection and the report table (CPU). Run: python phi35/tests/test_evaluate.py
Set PHI35_BASELINES to the folder holding the two released baseline folders to re-score them."""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

from _util import raises, run

from phi35cfg import data, evaluate  # noqa: E402
from common import vqa_scoring as vs  # noqa: E402

ROWS = data.to_rows([
    {'img_id': '0', 'question': 'Q0?', 'answer': 'yes', 'answer_type': 'CLOSED'},
    {'img_id': '1', 'question': 'Q1?', 'answer': 'no', 'answer_type': 'CLOSED'},
    {'img_id': '2', 'question': 'Q2?', 'answer': 'left lung', 'answer_type': 'OPEN'},
    {'img_id': '3', 'question': 'Q3?', 'answer': 'liver', 'answer_type': 'OPEN'}])
RUNTIME = {'gpu': 'g', 'versions': {'transformers': '4.44.2', 'peft': '0.12.0', 'torch': '2.11'}}


def _records(preds, mode=None, idx=None):
    idx = range(len(preds)) if idx is None else idx
    out = []
    for i, p in zip(idx, preds):
        r = {**{k: ROWS[i][k] for k in ('idx', 'question', 'answer', 'answer_type')}, 'prediction': p}
        if mode:
            r.update(mode=mode, cfg_applied=mode == 'expert_cfg')
        out.append(r)
    return out


def _write(path, records, meta=None):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    if meta is not None:
        path.with_name(path.name + '.meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return str(path)


@contextlib.contextmanager
def four_rows():
    old = evaluate.load_split
    evaluate.load_split = lambda *a, **k: (ROWS, None)
    try:
        yield
    finally:
        evaluate.load_split = old


def _main(argv):
    out = io.StringIO()
    with four_rows(), contextlib.redirect_stdout(out):
        evaluate.main(argv)
    return out.getvalue()


def _fixture(d):
    d = Path(d)
    base = d / 'base.json'                                    # released predictions: a JSON list without meta
    base.write_text(json.dumps(_records(['yes', 'yes', 'lung', 'spleen'])), encoding='utf-8')
    entropy = d / 'entropy.json'
    entropy.write_text(json.dumps({'dataset': 'vqarad', 'predictions': 'base-v1',
                                   'entropy': {'0': .1, '1': .2, '2': .3, '3': .9}}), encoding='utf-8')
    meta = {'runtime': RUNTIME, 'retrieval': 'x/rag.json', 'retrieval_sha256': 'f' * 64}
    return {
        'base': str(base), 'entropy': str(entropy),
        'rag100': _write(d / 'rag100.jsonl', _records(['yes', 'no', 'left lung', 'liver'], 'rag'), meta),
        'rag5': _write(d / 'rag5.jsonl', _records(['liver'], 'rag', [3]), meta),
        'cfg5': _write(d / 'cfg5.jsonl', _records(['liver'], 'expert_cfg', [3]),
                       {**meta, 'cfg': {'gamma': 1.5, 'alpha': .01, 'beta': 3.0, 'delta': 3.1}}),
        'check': _write(d / 'check.jsonl', _records(['yes', 'no', 'lung', 'spleen'], 'baseline'), meta)}


def _report_args(d, f, **over):
    f = {**f, **over}
    return ['report', '--dataset', 'vqarad', '--baseline', f['base'], '--entropy', f['entropy'],
            '--predictions-name', 'base-v1', '--rag100', f['rag100'], '--rag5', f['rag5'], '--cfg5', f['cfg5'],
            '--check-baseline', f['check'], '--out-dir', str(Path(d) / 'results'), '--checkpoint-name', 'ckpt-v1']


def test_select_checkpoint():
    assert evaluate.select_checkpoint({100: 70.0, 200: 71.0, 300: 71.0}) == 200
    with raises(ValueError):
        evaluate.select_checkpoint({})


def test_merge_requires_exact_targets():
    base = {0: 'a', 1: 'b', 2: 'c'}
    assert evaluate.merge(base, {1: 'B'}, {1}) == {0: 'a', 1: 'B', 2: 'c'}
    for run_, targets in (({}, {1}), ({1: 'B'}, {1, 2}), ({1: 'B', 2: 'C'}, {1})):
        with raises(ValueError):
            evaluate.merge(base, run_, targets)


def test_paired_ci():
    assert evaluate.paired_ci([1, 1, 1], [0, 0, 0]) == (100.0, 100.0)
    lo, hi = evaluate.paired_ci([1, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 1])
    assert lo < hi and (lo, hi) == evaluate.paired_ci([1, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 1])


def test_select_cli():
    with tempfile.TemporaryDirectory() as d:
        a = _write(Path(d) / 'a.jsonl', _records(['yes', 'yes', 'x', 'x'], 'baseline'))
        b = _write(Path(d) / 'b.jsonl', _records(['yes', 'no', 'x', 'liver'], 'baseline'))
        out = Path(d) / 'selected.json'
        _main(['select', '--dataset', 'vqarad', '--run', f'100={a}', f'200={b}', '--out', str(out)])
        got = json.loads(out.read_text(encoding='utf-8'))
        assert got['step'] == 200 and got['val_overall'] == {'100': 25.0, '200': 75.0}
        with contextlib.redirect_stderr(io.StringIO()), raises(SystemExit):
            _main(['select', '--dataset', 'vqarad', '--split', 'test', '--run', f'100={a}'])
        partial = _write(Path(d) / 'p.jsonl', _records(['yes'], 'baseline'))
        with raises(ValueError):
            _main(['select', '--dataset', 'vqarad', '--run', f'100={partial}', '--out', str(out)])


def test_report_rows():
    with tempfile.TemporaryDirectory() as d:
        text = _main(_report_args(d, _fixture(d)))
        files = {json.loads(p.read_text(encoding='utf-8'))['row']: json.loads(p.read_text(encoding='utf-8'))
                 for p in (Path(d) / 'results').glob('*.json')}
        assert set(files) == {'baseline', 'RAG-100', 'RAG-5', 'ExpertCFG-5', 'baseline-4.44.2-check'}
        for row in files.values():
            assert {'model', 'dataset', 'row', 'rule', 'scoring', 'scores', 'checkpoint', 'env'} <= set(row)
            assert (row['model'], row['rule'], row['scoring']) == ('phi35', 'compliant', 'phi')
            assert row['checkpoint'] == {'manifest': 'ckpt-v1'}
        s = files['baseline']['scores']            # closed 1/2; open recall: 'lung' 0.5, 'spleen' 0
        assert (s['closed'], s['open'], s['overall'], s['n_open'], s['n_closed']) == (50.0, 25.0, 37.5, 2, 2)
        assert files['baseline']['env'] == {'note': 'not recorded'}
        assert files['RAG-100']['scores']['overall'] == 100.0
        assert files['RAG-5']['scores']['overall'] == 62.5                       # only question 3 replaced
        assert files['RAG-5']['baseline_delta']['overall'] == 25.0
        assert files['RAG-5']['gate'] == {'entropy': 'eq4', 'top_percent': 5.0, 'n_selected': 1}
        assert files['ExpertCFG-5']['cfg']['gamma'] == 1.5 and 'cfg_applied 1/1' in files['ExpertCFG-5']['note']
        assert files['ExpertCFG-5']['env']['transformers'] == '4.44.2'
        assert files['baseline-4.44.2-check']['scores']['overall'] == 62.5
        assert '82.04' in text and 'AUC' in text and 'ExpertCFG-5' in text


def test_report_rejects_partial_or_wrong_runs():
    with tempfile.TemporaryDirectory() as d:
        f = _fixture(d)
        short = _write(Path(d) / 'short.jsonl', _records(['yes', 'no'], 'rag'), {'runtime': RUNTIME})
        wrong_q = _write(Path(d) / 'wrong.jsonl', _records(['liver'], 'rag', [2]), {'runtime': RUNTIME})
        no_meta = _write(Path(d) / 'nometa.jsonl', _records(['liver'], 'rag', [3]))
        for over, message in (({'rag100': short}, 'missing'), ({'rag5': wrong_q}, 'missing'),
                              ({'rag5': f['cfg5']}, 'mode'), ({'rag5': no_meta}, 'meta')):
            with raises(ValueError, match=message):
                _main(_report_args(d, f, **over))
        other = Path(d) / 'e2.json'
        other.write_text(Path(f['entropy']).read_text(encoding='utf-8').replace('base-v1', 'base-v2'), encoding='utf-8')
        with raises(ValueError):
            _main(_report_args(d, f, entropy=str(other)))
        assert not (Path(d) / 'results').exists()          # nothing is written when any input is rejected


def test_no_relaxed_rows():
    source = Path(evaluate.__file__).read_text(encoding='utf-8').lower()
    assert 's1s2' not in source and 'inject' not in source


def test_released_baselines():
    root = os.environ.get('PHI35_BASELINES')
    if not root:
        print('skipped: PHI35_BASELINES not set (released predictions not re-scored)')
        return
    for folder, expected in (('phi35-vqarad-baseline-ckpt257-v1', 79.59), ('phi35-slake-baseline-ckpt380-v1', 85.63)):
        records = data.read_predictions(Path(root) / folder / 'test_predictions.json').values()
        assert round(vs.summarize(list(records), vs.score_phi)['overall'], 2) == expected


if __name__ == '__main__':
    run(globals())
