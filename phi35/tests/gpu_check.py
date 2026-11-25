"""GPU verification on transformers 4.44.2 (not run by the CPU tests).

  1) neutral setting (gamma=1, alpha=1, beta=1, delta=0) gives the same text as model.generate
  2) the paper setting changes the two branches' last logits
  3) decoding with and without the KV cache agrees
  4) the released baseline predictions are reproduced (more than 2% mismatch fails with --full)

  cd phi35 && python tests/gpu_check.py --dataset vqarad --image-root IMAGES --base-dir BASE --adapter ADAPTER \\
      --retrieval rag.json --predictions test_predictions.json --out gpu_check.json [--full]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NEUTRAL = dict(gamma=1.0, alpha=1.0, beta=1.0, delta=0.0)
MAX_MISMATCH = 0.02


def verdict(results, full):
    """List of failed checks (empty = passed)."""
    failed = []
    if not results['neutral'] or not all(results['neutral']):
        failed.append('neutral setting differs from model.generate (or no item had a highlight)')
    if not results['logits'] or not all(paper > neutral + 1e-3 for neutral, paper in results['logits']):
        failed.append('paper setting does not change the logits')
    if not results['cache'] or not all(results['cache']):
        failed.append('KV cache and full recomputation disagree')
    repro = results['repro']
    if full and 1 - repro['same'] / repro['n'] > MAX_MISMATCH:
        failed.append(f'baseline reproduced on only {repro["same"]}/{repro["n"]}')
    return failed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=('vqarad', 'slake'))
    p.add_argument('--base-dir', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--image-root')
    p.add_argument('--data-root')
    p.add_argument('--retrieval', required=True)
    p.add_argument('--predictions', required=True, help='released baseline predictions')
    p.add_argument('--out', required=True)
    p.add_argument('--full', action='store_true', help='all questions for reproduction, 20 for the cache check')
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    import math

    import torch
    from PIL import Image

    from phi35cfg import generation as G
    from phi35cfg import model as model_lib
    from phi35cfg.data import align, load_split, read_predictions
    from phi35cfg.retrieval import compliant_highlights, load_retrieval

    rows, images = load_split(a.dataset, 'test', a.image_root, a.data_root)
    retrieval = load_retrieval(a.retrieval, rows)
    preds = align(rows, read_predictions(a.predictions).values(), Path(a.predictions).name)
    model, processor, meta = model_lib.load(a.base_dir, a.adapter)
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    stops = [tok.eos_token_id, tok.convert_tokens_to_ids(G.END_TOKEN)]

    def image(row):
        with Image.open(images / row['image']) as im:
            return im.convert('RGB')

    def inputs(row, reference, highlights):
        prompt = G.build_prompt(tok, row['question'], reference)
        batch = dict(processor(prompt, [image(row)], return_tensors='pt', qs_highlighted_parts=highlights))
        mask = batch.pop('highlight_attention_mask')
        return {k: v.to(device) for k, v in batch.items()}, (mask[0] == 1).float().to(device)

    def library_generate(row, reference):
        batch, _ = inputs(row, reference, [])
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=64, do_sample=False, num_beams=1, eos_token_id=stops,
                                 use_cache=True)
        text = processor.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
        return text.strip().strip('.')

    def last_logit_gap(row, reference, highlights, cfg):
        batch, m = inputs(row, reference, highlights)
        batch = {k: v.repeat(2, *[1] * (v.dim() - 1)) for k, v in batch.items()}
        hooks = G.get_hooks(model)
        hooks.set(torch.stack([torch.ones_like(m), 1 - (1 - cfg.alpha) * m]),
                  torch.stack([math.log(cfg.beta) * m, -cfg.delta * m]))
        try:
            with torch.no_grad():
                logits = model(**batch, use_cache=False).logits[:, -1].float()
        finally:
            hooks.clear()
        return float((logits[0] - logits[1]).abs().max())

    wanted = 20 if a.full else 3
    picked = []                                     # questions where a highlight token exists
    for row in rows:
        ref = retrieval[row['idx']]['reference']
        hl = compliant_highlights(row['question'], row['answer_type'], ref, retrieval[row['idx']]['file_highlights'])
        if hl and inputs(row, ref, hl)[1].sum() > 0:
            picked.append((row, ref, hl))
        if len(picked) >= max(5, wanted):
            break

    results = {'neutral': [], 'logits': [], 'cache': []}
    for row, ref, hl in picked[:5]:
        mine = G.generate(model, processor, image(row), row['question'], ref, hl, G.CFGConfig(**NEUTRAL))
        results['neutral'].append(mine['cfg_applied'] and
                                  mine['prediction'].lower() == library_generate(row, ref).lower())
        results['logits'].append((last_logit_gap(row, ref, hl, G.CFGConfig(**NEUTRAL)),
                                  last_logit_gap(row, ref, hl, G.CFGConfig())))
    for row, ref, hl in picked[:wanted]:
        cached, recomputed = (G.generate(model, processor, image(row), row['question'], ref, hl, G.CFGConfig(),
                                         use_cache=c) for c in (True, False))
        results['cache'].append(cached['prediction'] == recomputed['prediction'])
    subset = rows if a.full else rows[:50]
    mismatch = [r['idx'] for r in subset
                if G.generate(model, processor, image(r), r['question'])['prediction'].lower()
                != str(preds[r['idx']]['prediction']).strip().lower()]
    results['repro'] = {'n': len(subset), 'same': len(subset) - len(mismatch), 'mismatch_idx': mismatch}

    failed = verdict(results, a.full)
    report = {'dataset': a.dataset, 'full': a.full, 'env': meta, **results, 'failed': failed}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('PASSED' if not failed else f'FAILED: {failed}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
