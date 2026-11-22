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


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
