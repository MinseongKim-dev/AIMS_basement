"""Retrieval files and compliant highlights (CPU). Run: python phi35/tests/test_retrieval.py"""
import inspect
import json
import re
import tempfile
from pathlib import Path

from _util import PHI35, REPO, raises, run

from phi35cfg.data import to_rows  # noqa: E402
from phi35cfg.retrieval import compliant_highlights, load_retrieval  # noqa: E402

TESTSET = json.loads((REPO / 'data_common' / 'vqarad_img' / 'testset.json').read_text(encoding='utf-8'))


def _notebook_rule():
    """compliant_highlights exactly as the evaluation notebook (cell 18) defines it."""
    nb = json.loads((PHI35 / '02_Expert_CFG_RAG_comparison.ipynb').read_text(encoding='utf-8'))
    source = ''.join(nb['cells'][18]['source'])
    namespace = {'re': re}
    exec(source[source.index('_STOP = set'):source.index('HL = {')], namespace)
    return namespace['compliant_highlights']


def test_highlights_have_no_answer_arg():
    assert list(inspect.signature(compliant_highlights).parameters) == ['question', 'answer_type', 'reference',
                                                                        'file_highlights']


def test_closed_sentence_and_open_file_rule():
    ref = 'Is there fat stranding? yes. The liver is normal.'
    assert compliant_highlights('Is fat stranding present?', 'CLOSED', ref, ['x']) == ['Is there fat stranding?']
    assert compliant_highlights('Is fat stranding present?', 'OPEN', ref, ['x']) == ['x']
    assert compliant_highlights('Unrelated words', 'CLOSED', ref, ['x']) == []


def test_matches_notebook():
    reference_rule = _notebook_rule()
    nonempty = 0
    for r in TESTSET:
        q, t = r['question'], r['answer_type'].strip().upper()
        cap = f"{q} yes. Another finding in the {r['image_organ']}. {q}"
        expected = reference_rule({'question': q, 'answer_type': t, 'top_k_captions': [cap], 'highlights': ['x']})
        assert compliant_highlights(q, t, cap, ['x']) == expected
        nonempty += bool(expected)
    assert len(TESTSET) == 451 and nonempty > 400


def _file(d, records):
    p = Path(d) / 'rag.json'
    p.write_text(json.dumps(records), encoding='utf-8')
    return p


ROWS = to_rows([{'img_id': f'{i}.jpg', 'question': f'Q{i}?', 'answer': f'A{i}', 'answer_type': 'OPEN'} for i in range(3)])
RECORDS = [{'img_id': f'{i}.jpg', 'question': f'Q{i}? ', 'answer': f'A{i}', 'answer_type': 'OPEN',
            'top_k_captions': [f'first {i}', 'second'], 'highlights': [f'h{i}']} for i in range(3)]


def test_load_retrieval():
    with tempfile.TemporaryDirectory() as d:
        got = load_retrieval(_file(d, RECORDS), ROWS)
        assert got[1] == {'reference': 'first 1', 'file_highlights': ['h1']}
        assert all('answer' not in g for g in got)


def test_load_rejects_wrong_dataset():
    wrong_image = [{**RECORDS[0], 'img_id': 'other.jpg'}] + RECORDS[1:]
    with tempfile.TemporaryDirectory() as d:
        for bad in (wrong_image, RECORDS[::-1], RECORDS[:2]):
            with raises(ValueError, match='rag.json'):
                load_retrieval(_file(d, bad), ROWS)


if __name__ == '__main__':
    run(globals())
