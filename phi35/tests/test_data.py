"""Splits, prediction files, pairing and resumable run files (CPU only). Run: python phi35/tests/test_data.py"""
import json
import tempfile
from pathlib import Path

from _util import REPO, raises, run

from phi35cfg.data import (align, append_record, check_records, file_sha256, load_split, open_run,  # noqa: E402
                           read_predictions, to_rows)

SPLITS = {'train': 'trainset.json', 'val': 'valset.json', 'test': 'testset.json'}


def _image_root(tmp):
    for fname in SPLITS.values():
        for r in json.loads((REPO / 'data_common' / 'vqarad_img' / fname).read_text(encoding='utf-8')):
            (Path(tmp) / r['image_name']).touch()
    return tmp


def test_vqarad_counts():
    with tempfile.TemporaryDirectory() as tmp:
        root = _image_root(tmp)
        for split, n in (('train', 2749), ('val', 315), ('test', 451)):
            rows, images = load_split('vqarad', split, image_root=root)
            assert len(rows) == n and images == Path(root)
        assert sum(r['answer_type'] == 'CLOSED' for r in rows) == 272
        assert [r['idx'] for r in rows[:3]] == [0, 1, 2]
        assert set(rows[0]) == {'idx', 'image', 'question', 'answer', 'answer_type'}


def test_vqarad_missing_image_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with raises(FileNotFoundError):
            load_split('vqarad', 'test', image_root=tmp)
        assert len(load_split('vqarad', 'test', image_root=tmp, check_images=False)[0]) == 451
    with raises(ValueError):
        load_split('vqarad', 'test')


def test_to_rows_accepts_image_keys_and_strips():
    raw = [{'image_name': 'a', 'question': ' q ', 'answer': 2, 'answer_type': 'open '},
           {'img_name': 'b', 'question': 'q', 'answer': 'yes', 'answer_type': 'CLOSED'},
           {'img_id': 'c', 'question': 'q', 'answer': 'no', 'answer_type': 'closed'}]
    rows = to_rows(raw)
    assert [r['image'] for r in rows] == ['a', 'b', 'c']
    assert rows[0] == {'idx': 0, 'image': 'a', 'question': 'q', 'answer': '2', 'answer_type': 'OPEN'}


ROWS = to_rows([{'img_id': f'{i}.jpg', 'question': f'Q{i}?', 'answer': f'A{i}', 'answer_type': 'OPEN'} for i in range(3)])


def test_align_uses_idx():
    recs = [{'idx': r['idx'], 'question': r['question'].upper(), 'answer': r['answer'], 'p': r['idx']} for r in ROWS]
    assert [r['p'] for r in align(ROWS, recs[::-1], 'x')] == [0, 1, 2]


def test_align_without_idx_keeps_file_order():
    recs = [{'question': r['question'], 'answer': r['answer']} for r in ROWS]
    assert align(ROWS, recs, 'x') == recs


def test_align_rejects_mismatch():
    recs = [{'question': r['question'], 'answer': r['answer']} for r in ROWS]
    for bad in (recs[::-1], recs[:2], recs[:2] + [{'question': 'Q2?', 'answer': 'other'}]):
        with raises(ValueError, match='myfile'):
            align(ROWS, bad, 'myfile')
    with raises(ValueError, match='myfile'):          # duplicated idx
        align(ROWS, [{**recs[0], 'idx': 0}] * 3, 'myfile')


def test_check_records_partial():
    check_records(ROWS, {2: {'question': 'q2?', 'answer': 'a2'}}, 'run')
    for bad in ({2: {'question': 'Q1?', 'answer': 'A1'}}, {7: {'question': 'Q0?', 'answer': 'A0'}}):
        with raises(ValueError, match='run'):
            check_records(ROWS, bad, 'run')


def test_read_predictions_json_and_jsonl():
    with tempfile.TemporaryDirectory() as d:
        recs = [{'idx': 1, 'prediction': 'b'}, {'idx': 0, 'prediction': 'a'}]
        a, b, c = Path(d, 'a.json'), Path(d, 'b.jsonl'), Path(d, 'c.jsonl')
        a.write_text(json.dumps(recs, indent=2), encoding='utf-8')
        b.write_text(''.join(json.dumps(r) + '\n' for r in recs), encoding='utf-8')
        c.write_text(''.join(json.dumps(r) + '\n' for r in recs + recs[:1]), encoding='utf-8')
        assert read_predictions(a) == read_predictions(b) == {0: recs[1], 1: recs[0]}
        with raises(ValueError):
            read_predictions(c)


def test_open_run_resume():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'sub' / 'r.jsonl'
        assert open_run(out, {'mode': 'rag', 'runtime': {'gpu': 'a'}}) == set()
        append_record(out, {'idx': 0, 'prediction': '한글'})
        with out.open('a', encoding='utf-8') as f:
            f.write('{"idx": 1, "pre')
        assert open_run(out, {'mode': 'rag', 'runtime': {'gpu': 'b'}}) == {0}
        assert out.read_text(encoding='utf-8').endswith('}\n')
        append_record(out, {'idx': 1})
        assert set(read_predictions(out)) == {0, 1}
        with raises(ValueError):
            open_run(out, {'mode': 'baseline'})


def test_open_run_refuses_unrecorded_output():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'r.jsonl'
        append_record(out, {'idx': 0})
        with raises(ValueError):                      # predictions without a recorded configuration
            open_run(out, {'mode': 'rag'})


def test_file_sha256():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'f'
        p.write_bytes(b'abc')
        assert file_sha256(p) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


if __name__ == '__main__':
    run(globals())
