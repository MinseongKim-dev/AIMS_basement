"""Greedy decoding for Phi-3.5-Vision with Expert-CFG (Eq. 7-10) and predictive entropy (Eq. 4).

Two-branch batch for Expert-CFG (index 0 = normal context c, 1 = unconditional c-bar):
  Eq. 7  highlighted token embeddings x alpha in branch 1
  Eq. 9  attention-score bias +log(beta) on highlighted keys in branch 0, -delta in branch 1
  Eq. 8  gamma log P(x|c) - (gamma-1) log P(x|c-bar)  -> greedy argmax

Eager attention is required: the bias is added to the additive 4D attention mask.
The highlight token mask comes from the authors' processor (qs_highlighted_parts).
"""
import math
import re
from dataclasses import dataclass

import torch

from . import REPO_ROOT  # noqa: F401  (puts `common` on sys.path)
from common.gating import aggregate_entropy

END_TOKEN = '<|end|>'
DECODER_ATTN = re.compile(r'layers\.\d+\.self_attn$')


@dataclass
class CFGConfig:
    gamma: float = 1.5      # main text of the paper
    alpha: float = 0.01
    beta: float = 3.0
    delta: float = None     # paper: log(beta) + 2

    def __post_init__(self):
        if self.gamma < 1 or self.beta <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError('need gamma >= 1, beta > 0, 0 <= alpha <= 1')
        if self.delta is None:
            self.delta = math.log(self.beta) + 2.0


def build_prompt(tokenizer, question, reference=None):
    content = f'{reference} <|image_1|>\n{question}' if reference else f'<|image_1|>\n{question}'
    return tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)


def guided_logp(logp, gamma):
    """Eq. 8 on the two rows of (2, vocabulary) log-probabilities."""
    return (gamma * logp[0] - (gamma - 1) * logp[1]).log_softmax(-1)


class CFGHooks:
    """Attention-bias and embedding-scale hooks; inactive until set()."""

    def __init__(self, model):
        self.embed_scale = None     # (B, P)
        self.attn_bias = None       # (B, P)
        embedders = [m for n, m in model.named_modules() if n.endswith('vision_embed_tokens')]
        attns = [m for n, m in model.named_modules() if DECODER_ATTN.search(n) and 'img_processor' not in n]
        if len(embedders) != 1 or not attns:
            raise RuntimeError(f'unexpected Phi-3-V module layout: {len(embedders)} embedders, {len(attns)} attentions')
        self.handles = [embedders[0].register_forward_hook(self._embed)]
        self.handles += [m.register_forward_pre_hook(self._attn, with_kwargs=True) for m in attns]

    def set(self, embed_scale, attn_bias):
        self.embed_scale, self.attn_bias = embed_scale, attn_bias

    def clear(self):
        self.embed_scale = self.attn_bias = None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _embed(self, module, args, output):
        scale = self.embed_scale
        if scale is None or output.shape[1] < scale.shape[1]:     # one-token decoding steps
            return output
        output = output.clone()
        P = scale.shape[1]
        output[:, :P] = output[:, :P] * scale[:, :, None].to(device=output.device, dtype=output.dtype)
        return output

    def _attn(self, module, args, kwargs):
        bias = self.attn_bias
        if bias is None:
            return None
        positional = 'attention_mask' not in kwargs
        mask = (args[1] if len(args) > 1 else None) if positional else kwargs['attention_mask']
        if mask is None or mask.dim() != 4:
            raise RuntimeError('expected a 4D additive attention mask: load the model with eager attention')
        kv = mask.shape[-1]
        if kv > bias.shape[1]:
            bias = torch.nn.functional.pad(bias, (0, kv - bias.shape[1]))
        mask = mask + bias[:, None, None, :kv].to(device=mask.device, dtype=mask.dtype)
        if positional:
            args = (args[0], mask, *args[2:])
        else:
            kwargs['attention_mask'] = mask
        return args, kwargs


def get_hooks(model):
    """One CFGHooks per model, installed on first use."""
    hooks = model.__dict__.get('_cfg_hooks')
    if hooks is None:
        hooks = model.__dict__['_cfg_hooks'] = CFGHooks(model)
    return hooks


def _inputs(processor, image, prompt, highlights, device):
    batch = dict(processor(prompt, [image], return_tensors='pt', qs_highlighted_parts=list(highlights or [])))
    mask = batch.pop('highlight_attention_mask', None)
    return {k: v.to(device) for k, v in batch.items()}, mask


def _stop_ids(tokenizer):
    return {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(END_TOKEN)}


@torch.no_grad()
def generate(model, processor, image, question, reference=None, highlights=None,
             cfg: CFGConfig = None, max_new_tokens=64, use_cache=True):
    """Greedy decode. Guidance runs only when cfg is given and a highlight token is found in the prompt;
    otherwise this is plain decoding (with the reference in the prompt, if any)."""
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    prompt = build_prompt(tok, question, reference)
    # blank strings are dropped: the authors' processor raises IndexError on them
    highlights = [h for h in highlights or [] if h and h.strip()] if cfg is not None else []
    batch, mask = _inputs(processor, image, prompt, highlights, device)
    m = None
    if highlights:
        if mask is None:
            raise RuntimeError('the processor returned no highlight mask: use the authors\' processor')
        m = (mask[0] == 1).float().to(device)
    n_highlight = int(m.sum()) if m is not None else 0
    guided = n_highlight > 0
    B = 2 if guided else 1
    seq = batch['input_ids'].repeat(B, 1)
    pixels = batch['pixel_values'].repeat(B, *[1] * (batch['pixel_values'].dim() - 1))
    sizes = batch['image_sizes'].repeat(B, 1)
    stops = _stop_ids(tok)
    hooks = get_hooks(model) if guided else None
    new, past = [], None
    try:
        if guided:
            hooks.set(torch.stack([torch.ones_like(m), 1 - (1 - cfg.alpha) * m]),
                      torch.stack([math.log(cfg.beta) * m, -cfg.delta * m]))
        for _ in range(max_new_tokens):
            if past is None:
                # clone: Phi-3-V edits image placeholder ids in place
                out = model(input_ids=seq.clone(), attention_mask=torch.ones_like(seq), pixel_values=pixels,
                            image_sizes=sizes, use_cache=use_cache)
            else:
                out = model(input_ids=seq[:, -1:], attention_mask=torch.ones_like(seq),
                            position_ids=torch.full((B, 1), seq.shape[1] - 1, dtype=torch.long, device=device),
                            past_key_values=past, use_cache=True)
            past = out.past_key_values if use_cache else None
            logp = out.logits[:, -1].float().log_softmax(-1)
            nxt = int((guided_logp(logp, cfg.gamma) if guided else logp[0]).argmax())
            if nxt in stops:
                break
            new.append(nxt)
            seq = torch.cat([seq, torch.full((B, 1), nxt, dtype=seq.dtype, device=device)], 1)
    finally:
        if guided:
            hooks.clear()
    text = processor.batch_decode(torch.tensor([new], dtype=torch.long), skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
    return {'prediction': text.strip().strip('.'), 'cfg_applied': guided, 'highlight_tokens': n_highlight}


@torch.no_grad()
def first_token_logp(model, processor, image, question):
    """Log-probabilities (V,) of the first answer token: the question alone, one forward pass."""
    device = model.get_input_embeddings().weight.device
    batch, _ = _inputs(processor, image, build_prompt(processor.tokenizer, question), None, device)
    logits = model(input_ids=batch['input_ids'], attention_mask=torch.ones_like(batch['input_ids']),
                   pixel_values=batch['pixel_values'], image_sizes=batch['image_sizes'], use_cache=False).logits
    return logits[0, -1].float().log_softmax(-1)


@torch.no_grad()
def answer_entropy(model, processor, image, question, prediction):
    """Eq. 4 by teacher forcing `prediction`: mean vocabulary entropy over its non-special tokens
    (docs/UNCERTAINTY_SPEC.md). None when no token remains."""
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    batch, _ = _inputs(processor, image, build_prompt(tok, question), None, device)
    answer = tok(prediction, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
    n = answer.shape[1]
    if n == 0:
        return None
    L = batch['input_ids'].shape[1]
    ids = torch.cat([batch['input_ids'], answer.long()], 1)
    token_ids = ids[0, L:].tolist()
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=batch['pixel_values'],
                   image_sizes=batch['image_sizes'], use_cache=False).logits[0, L - 1:L - 1 + n]
    logp = logits.float().log_softmax(-1)
    p = logp.exp()
    entropies = (-torch.where(p > 0, p * logp, torch.zeros_like(p)).sum(-1)).tolist()
    return aggregate_entropy(entropies, token_ids, set(tok.all_special_ids))
