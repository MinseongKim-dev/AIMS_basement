"""Prompt, greedy loop, CFG hooks and eq. 4 entropy on a fake model (CPU). Run: python phi35/tests/test_generation.py"""
import inspect
import math

import torch
from PIL import Image

from _util import raises, run
import _fakes as F

from phi35cfg.generation import (CFGConfig, CFGHooks, answer_entropy, build_prompt, generate,  # noqa: E402
                                 get_hooks, guided_logp)

IMG = Image.new('RGB', (4, 4))


def test_cfg_config_delta():
    assert abs(CFGConfig().delta - (math.log(3) + 2)) < 1e-12
    assert (CFGConfig().gamma, CFGConfig().alpha, CFGConfig().beta) == (1.5, 0.01, 3.0)
    assert CFGConfig(delta=0.0).delta == 0.0
    for bad in (dict(gamma=0.5), dict(beta=0), dict(alpha=1.5)):
        with raises(ValueError):
            CFGConfig(**bad)


def test_build_prompt():
    tok = F.Tokenizer()
    assert build_prompt(tok, 'q?') == '<user><|image_1|>\nq?<end><assistant>'
    assert build_prompt(tok, 'q?', 'ref.') == '<user>ref. <|image_1|>\nq?<end><assistant>'


def test_guided_logp():
    lp = torch.log_softmax(torch.tensor([[1.0, 2.0, 0.5], [3.0, 0.0, 0.5]]), -1)
    assert torch.allclose(guided_logp(lp, 1.0), lp[0])
    g = guided_logp(lp, 1.5)
    assert abs(float(g.exp().sum()) - 1) < 1e-6
    assert int(g.argmax()) == int((1.5 * (lp[0] - lp[1]) + lp[1]).argmax())


def _probe(positional):
    model = F.Model()
    hooks = CFGHooks(model)
    bias = torch.tensor([[0.0, 1.0], [0.0, -2.0]])
    scale = torch.tensor([[1.0, 1.0], [1.0, 0.25]])
    hooks.set(scale, bias)
    hidden = model.vision_embed_tokens(torch.zeros(2, 3, dtype=torch.long))
    seen_embed = model.after_embed(hidden)
    attn = model.layers[0].self_attn
    mask = torch.zeros(2, 1, 3, 3)
    attn(hidden, mask) if positional else attn(hidden, attention_mask=mask)
    return model, hooks, attn, hidden, seen_embed


def test_hooks_bias_keyword_and_positional():
    for positional in (False, True):
        _, _, attn, _, _ = _probe(positional)
        assert attn.seen[0, 0, 0].tolist() == [0.0, 1.0, 0.0]       # padded with 0 beyond the prefix
        assert attn.seen[1, 0, 2].tolist() == [0.0, -2.0, 0.0]


def test_hooks_scale_prefill_only():
    model, hooks, _, hidden, _ = _probe(False)
    out = model.vision_embed_tokens(torch.zeros(2, 3, dtype=torch.long))
    assert out[1, 1].tolist() == [0.25, 0.25] and out[0, 1].tolist() == [1.0, 1.0] and out[1, 2].tolist() == [1.0, 1.0]
    step = model.vision_embed_tokens(torch.zeros(2, 1, dtype=torch.long))
    assert torch.equal(step, torch.ones(2, 1, 2))                    # one-token decoding step is untouched
    hooks.clear()
    assert torch.equal(model.vision_embed_tokens(torch.zeros(2, 3, dtype=torch.long)), torch.ones(2, 3, 2))


def test_hooks_reject_non_4d_mask():
    model, _, attn, hidden, _ = _probe(False)
    for bad in (None, torch.zeros(2, 3)):
        with raises(RuntimeError, match='4D'):
            attn(hidden, attention_mask=bad)


def test_hooks_need_expected_layout():
    with raises(RuntimeError):
        CFGHooks(torch.nn.Linear(1, 1))


def test_hooks_one_per_model():
    model = F.Model()
    assert get_hooks(model) is get_hooks(model)
    assert get_hooks(F.Model()) is not get_hooks(model)


def test_greedy_prediction():
    r = generate(F.Model(), F.Processor(), IMG, 'q')
    assert r == {'prediction': 'a b', 'cfg_applied': False, 'highlight_tokens': 0}


def test_max_new_tokens_and_eos():
    assert generate(F.Model(), F.Processor(), IMG, 'q', max_new_tokens=1)['prediction'] == 'a'
    assert generate(F.Model(script=(2, F.EOS, 3)), F.Processor(), IMG, 'q')['prediction'] == 'a'


def test_cfg_two_branches():
    model, proc = F.Model(), F.Processor()
    r = generate(model, proc, IMG, 'q', 'ref', ['finding'], CFGConfig())
    assert r['cfg_applied'] is True and r['highlight_tokens'] == 1
    assert all(c['B'] == 2 for c in model.calls) and proc.calls[0][1] == ['finding']
    first = model.masks[0]
    assert abs(float(first[0, 0, 0, 2]) - math.log(3)) < 1e-6
    assert abs(float(first[1, 0, 0, 2]) + math.log(3) + 2) < 1e-6
    assert float(model.embeds[0][0, 2, 0]) == 1.0 and abs(float(model.embeds[0][1, 2, 0]) - 0.01) < 1e-6
    assert get_hooks(model).attn_bias is None and get_hooks(model).embed_scale is None


def test_no_highlight_is_single_branch():
    for highlights in (['absent'], [], None):
        model = F.Model()
        r = generate(model, F.Processor(), IMG, 'q', 'ref', highlights, CFGConfig())
        assert r['cfg_applied'] is False and r['highlight_tokens'] == 0
        assert all(c['B'] == 1 for c in model.calls) and float(model.masks[0].abs().sum()) == 0


def test_blank_highlights_never_reach_processor():
    # the authors' txt_highlight_mask raises IndexError on an empty string
    proc = F.Processor()
    r = generate(F.Model(), proc, IMG, 'q', 'ref', ['', '  ', 'finding', None], CFGConfig())
    assert proc.calls[0][1] == ['finding'] and r['cfg_applied'] is True
    proc = F.Processor()
    assert generate(F.Model(), proc, IMG, 'q', 'ref', [''], CFGConfig())['cfg_applied'] is False
    assert proc.calls[0][1] == []


def test_hooks_cleared_on_error():
    model = F.Model(fail=True)
    with raises(RuntimeError, match='forward failed'):
        generate(model, F.Processor(), IMG, 'q', 'ref', ['finding'], CFGConfig())
    assert get_hooks(model).embed_scale is None and get_hooks(model).attn_bias is None


def test_cached_steps_and_nocache_same():
    cached, full = F.Model(), F.Model()
    a = generate(cached, F.Processor(), IMG, 'q', 'ref', ['finding'], CFGConfig(), use_cache=True)
    b = generate(full, F.Processor(), IMG, 'q', 'ref', ['finding'], CFGConfig(), use_cache=False)
    assert a == b
    step = cached.calls[1]
    assert step['n'] == 1 and not step['pixels'] and step['position_ids'].tolist() == [[4], [4]]
    assert step['attention_mask'].shape == (2, 5)
    assert full.calls[1]['n'] == 5 and full.calls[1]['pixels'] and full.calls[1]['past'] == 0


def test_generate_takes_no_answer():
    assert 'answer' not in inspect.signature(generate).parameters


def test_entropy_excludes_special():
    sharp = torch.zeros(F.V)
    sharp[2] = 100.0
    model = F.Model(table={3: torch.zeros(F.V), 4: sharp, 5: torch.zeros(F.V)})
    h = answer_entropy(model, F.Processor(), IMG, 'q', 'a b <s>')
    assert abs(h - math.log(F.V) / 2) < 1e-5
    assert model.calls[0]['n'] == 7 and len(model.calls) == 1


def test_entropy_empty_answer_is_none():
    assert answer_entropy(F.Model(), F.Processor(), IMG, 'q', '') is None
    assert answer_entropy(F.Model(), F.Processor(), IMG, 'q', '<s>') is None


if __name__ == '__main__':
    run(globals())
