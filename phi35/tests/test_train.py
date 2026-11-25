"""Downstream training CLI: pure parts (CPU). Run: python phi35/tests/test_train.py"""
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from _util import raises, run
import _fakes as F

from phi35cfg.train import build_parser, keep_checkpoint, make_collator, prepare_run, resolve_epochs  # noqa: E402

REQUIRED = ['--dataset', 'vqarad', '--base-dir', 'b', '--output', 'o']


def test_resolve_epochs():
    assert resolve_epochs('vqarad', None, None) == (10, {2, 4, 6, 8, 10})
    assert resolve_epochs('slake', None, None) == (5, {1, 2, 3, 4, 5})
    assert resolve_epochs('vqarad', 3, None) == (3, {1, 2, 3})
    assert resolve_epochs('slake', None, '2,5') == (5, {2, 5})
    for bad in ('0', '11', '2,x'):
        with raises(ValueError):
            resolve_epochs('vqarad', None, bad)


def test_defaults():
    a = build_parser().parse_args(REQUIRED)
    assert (a.lr, a.batch, a.weight_decay, a.attn) == (1e-4, 64, 0.0, 'eager')
    assert (a.warmup_ratio, a.max_grad_norm, a.seed) == (0.03, 1.0, 42)
    assert a.adapter == 'ecoxial2007/Phi-3.5V-Med' and not a.grad_ckpt and not a.resume and a.max_steps is None


class Tokenizer(F.Tokenizer):
    def __call__(self, text, add_special_tokens=False, return_tensors='pt'):
        self.text = text
        return {'input_ids': torch.tensor([[2, 3]])}


def test_collator():
    processor = F.Processor()
    processor.tokenizer = Tokenizer()
    with tempfile.TemporaryDirectory() as d:
        Image.new('RGB', (2, 2)).save(Path(d) / 'a.png')
        collate = make_collator(processor, d)
        row = {'idx': 0, 'image': 'a.png', 'question': 'q', 'answer': 'yes', 'answer_type': 'CLOSED'}
        batch = collate([row])
        assert processor.tokenizer.text == 'yes<|end|>\n<|endoftext|>'
        assert batch['input_ids'].tolist() == [F.PROMPT_IDS + [2, 3]]
        assert batch['labels'].tolist() == [[-100] * 4 + [2, 3]]
        assert set(batch) == {'input_ids', 'labels', 'pixel_values', 'image_sizes'}
        assert processor.calls[0] == ('<user><|image_1|>\nq<end><assistant>', [])
        with raises(ValueError):
            collate([row, row])


def test_keep_checkpoint():
    assert keep_checkpoint(2.0, {2, 4}, smoke=False) and not keep_checkpoint(2.98, {2, 4}, smoke=False)
    assert keep_checkpoint(0.02, {2, 4}, smoke=True)


CONFIG = {'lr': 1e-4, 'max_steps': None, 'resume': False, 'runtime': {'gpu': 'a'}}


def test_prepare_run_fresh_and_resume():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'run'
        assert prepare_run(out, CONFIG, resume=False) is None
        assert json.loads((out / 'run_config.json').read_text(encoding='utf-8'))['lr'] == 1e-4
        with raises(ValueError, match='checkpoint'):           # nothing to resume yet
            prepare_run(out, {**CONFIG, 'resume': True}, resume=True)
        for step in (9, 10, 2):
            (out / f'checkpoint-{step}').mkdir()
        got = prepare_run(out, {**CONFIG, 'resume': True, 'runtime': {'gpu': 'b'}}, resume=True)
        assert got == str(out / 'checkpoint-10')
        with raises(ValueError, match='--resume'):             # would silently start over on top of old checkpoints
            prepare_run(out, CONFIG, resume=False)
        with raises(ValueError, match='lr'):
            prepare_run(out, {**CONFIG, 'lr': 2e-4, 'resume': True}, resume=True)
        with raises(ValueError, match='smoke'):                # a short smoke run cannot turn into the real one
            prepare_run(out, {**CONFIG, 'max_steps': 4, 'resume': True}, resume=True)


if __name__ == '__main__':
    run(globals())
