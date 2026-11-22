"""Greedy decoding for Phi-3-Vision with predictive entropy (Eq. 4) and Expert-CFG (Eq. 7-10).

Two-branch batch for Expert-CFG (batch index 0 = normal context c, 1 = unconditional c-bar):
  Eq. 7  c-bar_i = (alpha-1) m_i f(x_i) + f(x_i)      -> highlighted token embeddings x alpha in branch 1
  Eq. 9  h_i = log(beta) m_i + e_i                     -> attention-score bias on highlighted keys in branch 0
         P~(x | c-bar, -delta m)                       -> attention-score bias -delta on highlighted keys in branch 1
  Eq. 8  log P^ = gamma log P(x|c) - (gamma-1) log P(x|c-bar)   -> greedy argmax

The bias is added to the additive 4D attention mask of every Phi-3 decoder self-attention
layer (eager attention), i.e. before softmax, exactly as Eq. 9/10. The CLIP encoder is untouched.
"""
import math
import re
from dataclasses import dataclass

import torch

from .model import model_dtype

IMAGE_TAG = re.compile(r'<\|image_\d+\|>')


@dataclass
class CFGConfig:
    gamma: float = 1.5   # main text; appendix Table 7 best = 1.3
    alpha: float = 0.01
    beta: float = 3.0
    delta: float = None  # paper: "delta satisfying log(beta) + 2"

    def __post_init__(self):
        if self.delta is None:
            self.delta = math.log(self.beta) + 2.0
        if self.gamma < 1 or self.beta <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError('need gamma >= 1, beta > 0, 0 <= alpha <= 1')


def build_prompt(tokenizer, question, reference=None):
    # Same layout as the author demo: f'{caption} <|image_1|>\n{question}'
    content = f'{reference} <|image_1|>\n{question}' if reference else f'<|image_1|>\n{question}'
    return tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)


def _char_ranges(text, parts):
    low = text.lower()
    ranges = []
    for part in parts or []:
        part = part.strip()
        if not part:
            continue
        start = low.find(part.lower())
        while start != -1:
            ranges.append((start, start + len(part)))
            start = low.find(part.lower(), start + len(part))
    return ranges


def highlight_mask(tokenizer, prompt, input_ids, highlights):
    """Token mask m aligned to the processor's input_ids (image placeholder ids are negative).

    Each text chunk between image tags is re-tokenized with character offsets; the processor's
    text segments must equal a suffix of that tokenization (the processor strips BOS), which is
    asserted so a tokenizer/processor mismatch fails loudly instead of misplacing highlights.
    """
    ids = input_ids[0].tolist()
    segments, i = [], 0
    while i < len(ids):
        if ids[i] < 0:
            i += 1
            continue
        j = i
        while j < len(ids) and ids[j] >= 0:
            j += 1
        segments.append((i, j))
        i = j
    chunks = IMAGE_TAG.split(prompt)
    if len(segments) != len(chunks):
        raise ValueError(f'{len(chunks)} text chunks but {len(segments)} token segments')
    mask = torch.zeros(len(ids))
    for (s, e), chunk in zip(segments, chunks):
        enc = tokenizer(chunk, return_offsets_mapping=True)
        tok, offs = enc['input_ids'], enc['offset_mapping']
        n = e - s
        if n > len(tok) or tok[len(tok) - n:] != ids[s:e]:
            raise ValueError('Highlight alignment failed: processor tokenization differs')
        ranges = _char_ranges(chunk, highlights)
        for k, (a, b) in enumerate(offs[len(tok) - n:]):
            if a < b and any(a < r1 and b > r0 for r0, r1 in ranges):
                mask[s + k] = 1.0
    return mask


class _Hooks:
    def __init__(self, model):
        self.embed_scale = None  # (B, P)
        self.attn_bias = None    # (B, P)
        self.handles = []
        embedders = [m for n, m in model.named_modules() if n.endswith('vision_embed_tokens')]
        attns = [m for n, m in model.named_modules()
                 if re.search(r'layers\.\d+\.self_attn$', n) and 'img_processor' not in n]
        if len(embedders) != 1 or not attns:
            raise RuntimeError('Unexpected Phi-3-V module layout')
        self.handles.append(embedders[0].register_forward_hook(self._embed))
        for m in attns:
            self.handles.append(m.register_forward_pre_hook(self._attn, with_kwargs=True))

    def _embed(self, module, args, output):
        if self.embed_scale is None:
            return output
        P = self.embed_scale.shape[1]
        if output.shape[1] < P:
            return output
        output = output.clone()
        output[:, :P] = output[:, :P] * self.embed_scale[:, :, None].to(output.dtype)
        return output

    def _attn(self, module, args, kwargs):
        if self.attn_bias is None:
            return None
        positional = 'attention_mask' not in kwargs
        mask = args[1] if positional and len(args) > 1 else kwargs.get('attention_mask')
        if mask is None or mask.dim() != 4:
            raise RuntimeError('Expected a 4D additive mask: load the model with eager attention')
        kv = mask.shape[-1]
        bias = self.attn_bias
        if kv > bias.shape[1]:
            bias = torch.nn.functional.pad(bias, (0, kv - bias.shape[1]))
        mask = mask + bias[:, None, None, :kv].to(mask.dtype)
        if positional:
            args = (args[0], mask, *args[2:])
        else:
            kwargs['attention_mask'] = mask
        return args, kwargs

    def clear(self):
        self.embed_scale = self.attn_bias = None


_HOOKS = {}


def _hooks(model):
    if id(model) not in _HOOKS:
        _HOOKS[id(model)] = _Hooks(model)
    return _HOOKS[id(model)]


def stop_ids(model, tokenizer):
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|end|>')}
    eos = getattr(model.generation_config, 'eos_token_id', None)
    ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    return {i for i in ids if isinstance(i, int) and i >= 0}


def _entropy(logp):
    p = logp.exp()
    return float(-torch.where(p > 0, p * logp, torch.zeros_like(p)).sum())


@torch.inference_mode()
def generate(model, processor, image, question, reference=None, highlights=None,
             cfg: CFGConfig = None, max_new_tokens=64, use_cache=True):
    """Greedy decode. cfg=None -> plain MedVLM (optionally with a reference in the prompt)."""
    tok = processor.tokenizer
    prompt = build_prompt(tok, question, reference)
    batch = processor(prompt, [image], return_tensors='pt')
    device = model.get_input_embeddings().weight.device
    input_ids = batch['input_ids'].to(device)
    pixel_values = batch['pixel_values'].to(device, model_dtype(model))
    image_sizes = batch['image_sizes'].to(device)
    P = input_ids.shape[1]
    hooks = _hooks(model)
    n_highlight = 0
    if cfg is not None:
        if not highlights:
            raise ValueError('Expert-CFG needs at least one highlight')
        m = highlight_mask(tok, prompt, input_ids, highlights).to(device)
        n_highlight = int(m.sum())
        if n_highlight == 0:
            raise ValueError(f'No highlight found in prompt: {highlights}')
        hooks.embed_scale = torch.stack([torch.ones_like(m), 1 - (1 - cfg.alpha) * m])
        hooks.attn_bias = torch.stack([math.log(cfg.beta) * m, -cfg.delta * m])
        input_ids = input_ids.repeat(2, 1)
        pixel_values = pixel_values.repeat(2, *[1] * (pixel_values.dim() - 1))
        image_sizes = image_sizes.repeat(2, 1)
    B = input_ids.shape[0]
    stops = stop_ids(model, tok)
    special = set(tok.all_special_ids)
    out_ids, ent, ent_cond, probs = [], [], [], []
    past, seq = None, input_ids
    try:
        for _ in range(max_new_tokens):
            attn = torch.ones(B, seq.shape[1], dtype=torch.long, device=device)
            if past is None:
                # clone: Phi-3-V clamps image placeholder ids in-place
                out = model(input_ids=seq.clone(), pixel_values=pixel_values, image_sizes=image_sizes,
                            attention_mask=attn, use_cache=use_cache)
            else:
                out = model(input_ids=seq[:, -1:], attention_mask=attn,
                            past_key_values=past, use_cache=True)
            if use_cache:
                past = out.past_key_values
            logp = out.logits[:, -1].float().log_softmax(-1)
            if cfg is not None:
                final = (cfg.gamma * logp[0] - (cfg.gamma - 1) * logp[1]).log_softmax(-1)
            else:
                final = logp[0]
            nxt = int(final.argmax())
            out_ids.append(nxt)
            if nxt not in special:
                ent.append(_entropy(final))
                ent_cond.append(_entropy(logp[0]))
                probs.append(float(final[nxt].exp()))
            if nxt in stops:
                break
            seq = torch.cat([seq, torch.full((B, 1), nxt, device=device, dtype=seq.dtype)], 1)
    finally:
        hooks.clear()
    mean = lambda v: sum(v) / len(v) if v else None
    return {'prediction': tok.decode(out_ids, skip_special_tokens=True).strip(),
            'entropy': mean(ent),            # Eq. 4 on the decoding distribution
            'entropy_normal_branch': mean(ent_cond),
            'token_probs': probs,
            'tokens': [tok.decode([i]) for i in out_ids if i not in special],
            'truncated': bool(out_ids) and out_ids[-1] not in stops,
            'prompt_tokens': P, 'highlight_tokens': n_highlight}
