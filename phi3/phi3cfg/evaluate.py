"""Score runs and print a Table-1-style comparison for Phi3V-Med (paper values included).

Metrics (Sec. 4.3, LLaVA-Med convention): closed = accuracy (answer word contained in the
prediction), open = recall of ground-truth tokens in the generated answer, overall = mean over
all questions. Open/closed come from the dataset labels (VQA-RAD: run phi3cfg.vqarad_labels).
Partial runs (top-x% modes) are merged over the baseline: questions not re-run keep the baseline answer.

  python -m phi3cfg.evaluate --dataset-dir DATA/vqa_rad --baseline out/base.jsonl \
      --runs rag=out/rag.jsonl rag5=out/rag5.jsonl expert_rag5=out/erag5.jsonl expert_cfg5=out/ecfg5.jsonl \
      --output out/report.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .data import load_data, read_jsonl

# Table 1, Phi3V-Med rows: (open, closed, overall)
PAPER = {
    'vqa_rad': {'baseline': (70.39, 85.29, 79.38), 'rag': (66.48, 90.07, 80.71),
                'rag5': (68.72, 85.29, 78.71), 'expert_rag5': (71.50, 86.03, 80.27),
                'expert_cfg5': (75.42, 86.03, 81.82)},
    'slake': {'baseline': (82.02, 87.02, 83.98), 'rag': (83.41, 87.74, 85.11),
              'rag5': (82.33, 87.50, 84.35), 'expert_rag5': (83.25, 88.46, 85.29),
              'expert_cfg5': (84.96, 88.46, 86.33)},
    'pathvqa': {'baseline': (36.88, 91.80, 64.43), 'rag': (49.05, 95.28, 72.24),
                'rag5': (38.81, 91.80, 65.39), 'expert_rag5': (42.37, 91.80, 67.16),
                'expert_cfg5': (45.88, 91.80, 68.91)},
}


def tokens(text):
    return re.findall(r'[a-z0-9]+', str(text).lower())


def score(answer, prediction, answer_type):
    """Same rule as the team's Phi-3.5V baseline (MinseongKim-dev/Expert-CFG, 01 notebook):
    closed = the whole lower-cased answer is one of the prediction's tokens (so a multi-word
    closed answer such as "right side" never scores); open = recall of answer tokens."""
    pred = tokens(prediction)
    if answer_type == 'closed':
        return float(str(answer).strip().lower() in pred)
    counts = Counter(tokens(answer))
    return sum((counts & Counter(pred)).values()) / sum(counts.values()) if counts else 0.0


def summarize(rows, preds):
    s = {'open': [], 'closed': []}
    for r in rows:
        s[r['answer_type']].append(score(r['answer'], preds[r['id']]['prediction'], r['answer_type']))
    pct = lambda v: 100 * sum(v) / len(v) if v else float('nan')
    return {'open': pct(s['open']), 'closed': pct(s['closed']),
            'overall': pct(s['open'] + s['closed']),
            'n_open': len(s['open']), 'n_closed': len(s['closed'])}


def uncertainty_auc(rows, preds):
    from sklearn.metrics import roc_auc_score
    y, h = [], []
    for r in rows:
        p = preds[r['id']]
        if p.get('entropy') is None:
            continue
        y.append(int(score(r['answer'], p['prediction'], r['answer_type']) < 1.0))  # 1 = wrong
        h.append(p['entropy'])
    return float(roc_auc_score(y, h)) if len(set(y)) == 2 else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--baseline', required=True)
    p.add_argument('--runs', nargs='*', default=[], help='name=path (names: rag, rag5, expert_rag5, expert_cfg5, ...)')
    p.add_argument('--output')
    p.add_argument('--partial', action='store_true', help='score only questions in the baseline file')
    args = p.parse_args()
    rows = load_data(Path(args.dataset_dir) / f'{args.split}.jsonl', args.split)
    dataset = rows[0]['dataset']
    base = {r['id']: r for r in read_jsonl(args.baseline)}
    missing = [r['id'] for r in rows if r['id'] not in base]
    if missing and args.partial:
        print(f'WARNING: scoring only {len(rows) - len(missing)}/{len(rows)} questions (smoke test)')
        rows = [r for r in rows if r['id'] in base]
    elif missing:
        raise SystemExit(f'Baseline is missing {len(missing)} questions (e.g. {missing[:3]}); finish it first')

    results = {'baseline': {**summarize(rows, base), 'replaced': 0}}
    for spec in args.runs:
        name, path = spec.split('=', 1)
        run = {r['id']: r for r in read_jsonl(path)}
        merged = {i: run.get(i, b) for i, b in base.items()}
        results[name] = {**summarize(rows, merged), 'replaced': sum(i in run for i in base)}
        if any('cfg_applied' in r for r in run.values()):
            results[name]['cfg_applied'] = sum(bool(r.get('cfg_applied')) for r in run.values())
    auc = uncertainty_auc(rows, base)

    paper = PAPER.get(dataset, {}) if args.split == 'test' else {}  # paper numbers are test-set numbers
    print(f'\n{dataset} ({len(rows)} questions)   uncertainty AUC (baseline entropy) = {auc}')
    print(f'{"run":<14}{"open":>8}{"closed":>8}{"overall":>9}   | paper open/closed/overall | re-run')
    for name, m in results.items():
        ref = paper.get(name)
        ref_s = '/'.join(f'{v:.2f}' for v in ref) if ref else '-'
        print(f'{name:<14}{m["open"]:8.2f}{m["closed"]:8.2f}{m["overall"]:9.2f}   | {ref_s:<25} | {m["replaced"]}')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({'dataset': dataset, 'uncertainty_auc': auc,
                                                 'results': results, 'paper': paper}, indent=2))


if __name__ == '__main__':
    main()
