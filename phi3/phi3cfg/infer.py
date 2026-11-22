"""Run Phi3V-Med on a MedVQA split in one of the Table 1 modes (resumable JSONL output).

  baseline    plain Phi3V-Med; records Eq. 4 entropy (needed to pick the top-x% samples)
  rag         top-1 retrieved caption in the prompt
  expert_rag  expert reference (retrieved caption + expert text) in the prompt, no guidance
  expert_cfg  expert reference + highlights + classifier-free guidance (Eq. 7-10)

With --select-from BASELINE.jsonl --top-percent 5 only the top-5% highest-entropy questions
are run (the rest keep their baseline answers, merged by phi3cfg.evaluate).
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from .data import load_data, read_jsonl
from .model import MED_ADAPTER


def select_top(baseline_path, percent=None, threshold=None):
    rows = [r for r in read_jsonl(baseline_path) if r.get('entropy') is not None]
    rows.sort(key=lambda r: r['entropy'], reverse=True)
    if threshold is not None:
        return {r['id'] for r in rows if r['entropy'] >= threshold}
    k = max(1, round(len(rows) * percent / 100))
    return {r['id'] for r in rows[:k]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-dir', required=True, help='DATA/<vqa_rad|slake|pathvqa>')
    p.add_argument('--split', default='test')
    p.add_argument('--mode', choices=('baseline', 'rag', 'expert_rag', 'expert_cfg'), default='baseline')
    p.add_argument('--annotations', help='output of phi3cfg.retrieval search (non-baseline modes)')
    p.add_argument('--select-from', help='baseline predictions; restrict to high-entropy questions')
    p.add_argument('--top-percent', type=float)
    p.add_argument('--entropy-threshold', type=float)
    p.add_argument('--output', required=True)
    p.add_argument('--adapter', default=MED_ADAPTER,
                   help='HF id or local downstream LoRA dir (phi3cfg.train output); "none" = raw Phi-3V')
    p.add_argument('--precision', choices=('fp16', 'bf16', '4bit'), default='fp16')
    p.add_argument('--crops', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--no-cache', action='store_true', help='recompute full context each step')
    p.add_argument('--gamma', type=float, default=1.5)
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--beta', type=float, default=3.0)
    p.add_argument('--delta', type=float, help='default log(beta)+2')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.mode != 'baseline' and not args.annotations:
        p.error('--annotations is required for rag/expert modes')
    if args.select_from and args.top_percent is None and args.entropy_threshold is None:
        p.error('--select-from needs --top-percent or --entropy-threshold')

    ds_dir = Path(args.dataset_dir)
    rows = load_data(ds_dir / f'{args.split}.jsonl', args.split)
    if args.select_from:
        keep = select_top(args.select_from, args.top_percent, args.entropy_threshold)
        rows = [r for r in rows if r['id'] in keep]
        print(f'Selected {len(rows)} high-entropy questions', flush=True)
    if args.limit:
        rows = rows[:args.limit]
    ann = {a['id']: a for a in read_jsonl(args.annotations)} if args.annotations else {}

    from .generation import CFGConfig, generate
    from .model import load
    adapter = None if args.adapter.lower() == 'none' else args.adapter
    model, processor, meta = load(adapter, args.precision, args.crops)
    cfg = CFGConfig(args.gamma, args.alpha, args.beta, args.delta)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_suffix('.meta.json')
    config = {**vars(args), 'runtime': meta, 'cfg': vars(cfg)}
    if meta_path.exists() and json.loads(meta_path.read_text())['mode'] != args.mode:
        raise ValueError('Output belongs to another mode; choose a new --output')
    meta_path.write_text(json.dumps(config, indent=2))
    done = {r['id'] for r in read_jsonl(out)} if out.exists() else set()

    with out.open('a', encoding='utf-8') as f:
        for n, r in enumerate(rows, 1):
            if r['id'] in done:
                continue
            a = ann.get(r['id'], {})
            if args.mode != 'baseline' and not a:
                raise ValueError(f'No annotation for {r["id"]}')
            with Image.open(ds_dir / 'images' / r['image']) as im:
                image = im.convert('RGB')
            kw = dict(max_new_tokens=args.max_new_tokens, use_cache=not args.no_cache)
            cfg_applied = False
            if args.mode == 'baseline':
                res = generate(model, processor, image, r['question'], **kw)
            elif args.mode == 'rag':
                res = generate(model, processor, image, r['question'], a['rag_reference'], **kw)
            elif args.mode == 'expert_rag':
                res = generate(model, processor, image, r['question'], a['expert_reference'], **kw)
            else:
                hl = a.get('highlights') or []
                cfg_applied = bool(hl)
                try:
                    res = generate(model, processor, image, r['question'], a['expert_reference'],
                                   hl, cfg if hl else None, **kw)
                except ValueError as e:  # highlight not tokenizable in prompt -> no guidance
                    print(f'  {r["id"]}: {e}; falling back to expert_rag', flush=True)
                    cfg_applied = False
                    res = generate(model, processor, image, r['question'], a['expert_reference'], **kw)
            rec = {**r, **res, 'mode': args.mode, 'cfg_applied': cfg_applied}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            print(f'[{n}/{len(rows)}] H={res["entropy"] if res["entropy"] is None else round(res["entropy"], 3)} '
                  f'Q: {r["question"][:60]} | GT: {r["answer"]} | P: {res["prediction"][:80]}', flush=True)
    print('Predictions:', out)


if __name__ == '__main__':
    main()
