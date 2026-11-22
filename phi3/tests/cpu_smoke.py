"""CPU smoke test against the REAL Phi-3-vision processor/modeling code with a tiny random LM.

Needs only the code/tokenizer files (no weights):
  python -c "from huggingface_hub import snapshot_download as s; s('microsoft/Phi-3-vision-128k-instruct', allow_patterns=['*.json','*.py','tokenizer*'], local_dir='phi3v_small')"
  python tests/cpu_smoke.py phi3v_small
Checks: highlight mask alignment, Eq. 7/9 hooks fire, KV-cache == no-cache decoding,
neutral CFG (gamma=1, alpha=1, beta=1, delta=0) == plain decoding, real CFG changes logits.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phi3cfg import generation as G  # noqa: E402


def main(path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, num_crops=1)
    tok = processor.tokenizer

    # --- highlight mask on the real processor
    prompt = G.build_prompt(tok, 'Is there fat stranding?', 'No fat stranding. Appendix normal.')
    image = Image.new('RGB', (64, 48), (120, 30, 200))
    batch = processor(prompt, [image], return_tensors='pt')
    m = G.highlight_mask(tok, prompt, batch['input_ids'], ['no fat stranding'])
    marked = tok.decode([i for i, v in zip(batch['input_ids'][0].tolist(), m.tolist()) if v])
    print('prompt tokens', batch['input_ids'].shape[1], '| highlighted ->', repr(marked))
    assert marked.strip().lower() == 'no fat stranding', marked
    assert (batch['input_ids'][0][m.bool()] >= 0).all()

    # --- tiny random Phi-3-V (real modeling code)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 384, 768, 2
    cfg.num_attention_heads = cfg.num_key_value_heads = 4
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32,
                                             attn_implementation='eager').eval()
    q, ref, hl = 'Is there fat stranding?', 'No fat stranding. Appendix normal.', ['No fat stranding']
    kw = dict(max_new_tokens=6)
    base = G.generate(model, processor, image, q, ref, **kw)
    nocache = G.generate(model, processor, image, q, ref, use_cache=False, **kw)
    assert base['tokens'] == nocache['tokens'], (base['tokens'], nocache['tokens'])
    neutral = G.generate(model, processor, image, q, ref, hl, G.CFGConfig(1.0, 1.0, 1.0, 0.0), **kw)
    assert neutral['tokens'] == base['tokens'] and abs(neutral['entropy'] - base['entropy']) < 1e-3

    # hooks must actually change the forward pass
    hooks = G._hooks(model)
    strong = G.CFGConfig(gamma=3.0, alpha=0.01, beta=3.0)
    guided = G.generate(model, processor, image, q, ref, hl, strong, **kw)
    guided_nc = G.generate(model, processor, image, q, ref, hl, strong, use_cache=False, **kw)
    assert guided['tokens'] == guided_nc['tokens'], 'CFG cache/no-cache mismatch'
    print('baseline:', base['tokens'], round(base['entropy'], 3))
    print('guided  :', guided['tokens'], round(guided['entropy'], 3), 'highlight tokens', guided['highlight_tokens'])
    assert guided['highlight_tokens'] == len(tok.tokenize('No fat stranding'))
    assert abs(guided['entropy'] - base['entropy']) > 1e-4, 'guidance had no effect'
    assert hooks.embed_scale is None and hooks.attn_bias is None
    print(json.dumps({'ok': True}))


if __name__ == '__main__':
    main(sys.argv[1])
