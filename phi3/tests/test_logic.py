"""Pure-logic tests (no GPU/model): scoring, top-% selection, simulated expert annotations."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phi3cfg.evaluate import score  # noqa: E402
from phi3cfg.infer import select_top  # noqa: E402
from phi3cfg.retrieval import annotate, keyword_highlights  # noqa: E402


def test_score():
    assert score('yes', 'Yes, there is a mass.', 'closed') == 1
    assert score('no', 'Yes, no doubt.', 'closed') == 1   # team rule: answer word contained in prediction
    assert score('no', 'Yes.', 'closed') == 0
    assert score('Left', 'left side', 'closed') == 1        # choice question labelled CLOSED
    assert score('right side', 'right side', 'closed') == 0  # multi-word closed answers never match
    assert score('left lung', 'It is in the left upper lobe', 'open') == 0.5
    assert score('frontal and occipital', 'frontal and occipital lobes', 'open') == 1


def test_select_top():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'b.jsonl'
        p.write_text(''.join(json.dumps({'id': str(i), 'entropy': i / 100}) + '\n' for i in range(100)))
        assert select_top(p, percent=5) == {'99', '98', '97', '96', '95'}
        assert select_top(p, threshold=0.98) == {'99', '98'}


def test_annotate():
    hits = [{'caption': 'Is there fat stranding? yes. Is this a CT? yes.',
             'sentences': ['Is there fat stranding? yes.', 'Is this a CT? yes.']}]
    a = annotate({'id': '1', 'question': 'Where is the lesion?', 'answer': 'Left lobe',
                  'answer_type': 'open'}, hits)
    assert a['highlights'] == ['Left lobe'] and a['expert_reference'].endswith('Left lobe.')
    c = annotate({'id': '2', 'question': 'Is fat stranding present?', 'answer': 'no',
                  'answer_type': 'closed'}, hits)
    assert c['highlights'] == ['Is there fat stranding? yes.']
    assert keyword_highlights('unrelated words', hits[0]['sentences']) == []


def _write_split(d, split, items):
    rows = [{'id': f'{split}:{i}', 'dataset': 'vqa_rad', 'split': split, 'image': 'x.jpg',
             'question': 'q', 'answer': a, 'answer_type': t} for i, (a, t) in enumerate(items)]
    (Path(d) / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


def test_load_train():
    from phi3cfg.train import load_train
    with tempfile.TemporaryDirectory() as d:
        _write_split(d, 'train', [('yes', 'closed')] * 3)
        _write_split(d, 'validation', [('no', 'closed')] * 2)
        assert len(load_train(d)) == 3
        rows = load_train(d, ['train', 'validation'])
        assert len(rows) == 5 and {r['split'] for r in rows} == {'train', 'validation'}


def test_evaluate_report():
    """Report carries paper diff (test only), rule label and the runtime from infer's .meta.json."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
        gt = _write_split(d, 'test', [('yes', 'closed'), ('no', 'closed'), ('left lung', 'open')])
        preds = ['Yes.', 'yes', 'left upper lobe']  # closed 1/2, open 0.5 -> overall 50
        base = Path(d) / 'baseline.jsonl'
        base.write_text(''.join(json.dumps({'id': r['id'], 'prediction': p, 'entropy': h}) + '\n'
                                for r, p, h in zip(gt, preds, (0.1, 0.9, 0.5))))
        base.with_suffix('.meta.json').write_text(json.dumps(
            {'mode': 'baseline', 'runtime': {'attn': 'eager', 'versions': {'torch': 'x'}}}))
        out = Path(d) / 'report.json'
        subprocess.run([sys.executable, '-m', 'phi3cfg.evaluate', '--dataset-dir', d, '--baseline', str(base),
                        '--output', str(out), '--rule', 'compliant', '--rule-note', 'n'],
                       cwd=root, check=True, capture_output=True)
        rep = json.loads(out.read_text())
        m = rep['results']['baseline']
        assert (m['open'], m['closed'], m['overall']) == (50.0, 50.0, 50.0)
        assert abs(m['paper_diff']['overall'] - (50.0 - 79.38)) < 1e-9
        assert rep['rule'] == 'compliant' and rep['runtime']['attn'] == 'eager'
        assert rep['uncertainty_auc'] == 1.0  # the wrong answers have the highest entropy


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
