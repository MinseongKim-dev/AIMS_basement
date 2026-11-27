"""Choose a checkpoint on val, and score the compliant rows against the paper (original Phi scoring).

  python -m phi35cfg.evaluate select --dataset vqarad --run 129=val129.jsonl 257=val257.jsonl
  python -m phi35cfg.evaluate report --dataset vqarad --baseline test_predictions.json --entropy entropy.json \\
      --rag100 rag.jsonl --rag5 rag5.jsonl --cfg5 cfg5.jsonl --out-dir results
  python -m phi35cfg.evaluate yesno --vqarad val_pred.jsonl val_probs.jsonl --slake val_pred.jsonl val_probs.jsonl \
      --out yesno_report.json

Top-x% rows keep the baseline answer for every question that was not selected; a run file must cover
exactly its target questions.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .data import DATASETS, align, check_records, load_split, read_predictions
from .infer import load_entropy, select_ids
from . import yesno
from common import vqa_scoring as vs
from common.gating import auc_wrong_vs_entropy

# Table 1, Phi3.5V-Med (open, closed, overall), arXiv:2507.09209v1 HTML
PAPER = {
    'vqarad': {'baseline': (69.27, 90.44, 82.04), 'RAG-100': (70.39, 88.24, 81.15),
               'RAG-5': (70.95, 89.34, 82.04), 'ExpertCFG-5': (74.86, 91.18, 84.70)},
    'slake': {'baseline': (82.95, 88.70, 85.20), 'RAG-100': (82.79, 89.90, 85.58),
              'RAG-5': (82.64, 89.42, 85.30), 'ExpertCFG-5': (85.27, 91.59, 87.75)},
}
CHECK_ROW = 'baseline-4.44.2-check'
SEED = 42


def select_checkpoint(scores):
    """Step with the highest val overall; ties go to the earlier checkpoint."""
    if not scores:
        raise ValueError('no checkpoint scores')
    return max(scores, key=lambda step: (scores[step], -step))


def merge(base, run, targets):
    """Baseline records with the target questions replaced by the run's."""
    missing, extra = sorted(set(targets) - set(run)), sorted(set(run) - set(targets))
    if missing or extra:
        raise ValueError(f'run does not match its target questions: missing {len(missing)} {missing[:5]}, '
                         f'unexpected {len(extra)} {extra[:5]}')
    return {**base, **run}


def paired_ci(new, ref, n=10000, seed=0):
    """95% bootstrap interval of the paired mean difference, in percentage points."""
    d = np.asarray(new, dtype=float) - np.asarray(ref, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1) * 100
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _read_meta(path, required):
    meta = Path(str(path) + '.meta.json')
    if meta.exists():
        return json.loads(meta.read_text(encoding='utf-8'))
    if required:
        raise ValueError(f'{Path(path).name}: no .meta.json next to the run (environment unknown)')
    return None


def _env(meta):
    runtime = (meta or {}).get('runtime')
    if not runtime:
        return {'note': 'not recorded'}
    versions = runtime.get('versions', {})
    return {**{k: versions.get(k, '') for k in ('transformers', 'peft', 'torch')}, 'gpu': runtime.get('gpu', '')}


def _read_run(path, rows, targets, mode):
    records = read_predictions(path)
    name = Path(path).name
    wrong = sorted({str(r.get('mode')) for r in records.values() if r.get('mode') != mode})
    if wrong:
        raise ValueError(f'{name}: expected mode {mode!r}, found {wrong}')
    check_records(rows, records, name)
    return records, merge({}, records, targets), _read_meta(path, required=True)


def _ordered(records):
    return [records[i] for i in sorted(records)]


def _row(dataset, name, merged, base_scores, checkpoint, env, note=''):
    records = _ordered(merged)
    result = {'model': 'phi35', 'dataset': dataset, 'row': name, 'rule': 'compliant', 'scoring': 'phi',
              'scores': vs.summarize(records, vs.score_phi), 'seed': SEED,
              'checkpoint': {'manifest': checkpoint}, 'env': env, 'note': note}
    if base_scores is not None:
        scores = np.array([vs.score_phi(r) for r in records])
        result['baseline_delta'] = {'overall': float((scores - base_scores).mean() * 100),
                                    'ci95': list(paired_ci(scores, base_scores))}
    return result


def _report(a):
    rows, _ = load_split(a.dataset, 'test', data_root=a.data_root, check_images=False)
    n = len(rows)
    base = dict(enumerate(align(rows, read_predictions(a.baseline).values(), Path(a.baseline).name)))
    base_scores = np.array([vs.score_phi(r) for r in _ordered(base)])
    entropy = load_entropy(a.entropy, a.dataset, n, a.predictions_name) if a.entropy else None
    results = [_row(a.dataset, 'baseline', base, None, a.checkpoint_name, _env(_read_meta(a.baseline, False)))]

    selected = set(select_ids(entropy, n, a.top_percent)) if entropy else set()
    for name, path, targets, mode in (('RAG-100', a.rag100, set(range(n)), 'rag'),
                                      ('RAG-5', a.rag5, selected, 'rag'),
                                      ('ExpertCFG-5', a.cfg5, selected, 'expert_cfg')):
        if not path:
            continue
        run, _, meta = _read_run(path, rows, targets, mode)
        row = _row(a.dataset, name, {**base, **run}, base_scores, a.checkpoint_name, _env(meta))
        if meta.get('retrieval'):
            row['kb'] = {'file': Path(meta['retrieval']).name, 'sha256': meta.get('retrieval_sha256', '')}
        if name != 'RAG-100':
            row['gate'] = {'entropy': 'eq4', 'top_percent': a.top_percent, 'n_selected': len(selected)}
        if name == 'ExpertCFG-5':
            row['cfg'] = meta.get('cfg', {})
            row['note'] = f'cfg_applied {sum(bool(r.get("cfg_applied")) for r in run.values())}/{len(run)}'
        results.append(row)

    if a.check_baseline:        # a separate row: never merged into the others
        run, _, meta = _read_run(a.check_baseline, rows, set(range(n)), 'baseline')
        same = sum(run[i]['prediction'].strip().lower() == str(base[i]['prediction']).strip().lower() for i in run)
        results.append(_row(a.dataset, CHECK_ROW, run, base_scores, a.checkpoint_name, _env(meta),
                            f'same prediction as the released baseline: {same}/{n}'))
    return rows, base_scores, entropy, results


def _print_report(a, rows, base_scores, entropy, results):
    print(f'[{a.dataset}] N={len(rows)}  scoring: original (Phi)  rule: compliant')
    print(f'{"row":<24}{"open":>8}{"closed":>8}{"overall":>9}  {"delta [95% CI]":<24} paper open/closed/overall')
    for r in results:
        s, d = r['scores'], r.get('baseline_delta')
        delta = f'{d["overall"]:+.2f} [{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}]' if d else '-'
        paper = PAPER[a.dataset].get(r['row'])
        print(f'{r["row"]:<24}{s["open"]:8.2f}{s["closed"]:8.2f}{s["overall"]:9.2f}  {delta:<24} '
              + ('/'.join(f'{v:.2f}' for v in paper) if paper else '-')
              + (f'  ({r["note"]})' if r['note'] else ''))
    print('Paper values: arXiv:2507.09209v1 Table 1 (HTML). Its expert rows may rely on expert annotations '
          'that are not used here.')
    if entropy:
        wrong = [int(s < 1.0) for s in base_scores]
        for label, kind in (('all', None), ('closed', 'CLOSED'), ('open', 'OPEN')):
            ids = [r['idx'] for r in rows if kind in (None, r['answer_type'])]
            auc = auc_wrong_vs_entropy([wrong[i] for i in ids], [entropy[i] for i in ids])
            print(f'uncertainty AUC {label:<7} {"-" if auc is None else format(auc, ".3f")}')


def _yesno_items(dataset, pred_path, probs_path, data_root):
    """One item per yes/no val question (wrong answer, D0 / D1 / D2), after checking both run files."""
    rows, _ = load_split(dataset, 'val', data_root=data_root, check_images=False)
    names = Path(pred_path).name, Path(probs_path).name
    preds = align(rows, read_predictions(pred_path).values(), names[0])
    probs = read_predictions(probs_path)
    check_records(rows, probs, names[1])
    metas = [_read_meta(path, required=True) for path in (pred_path, probs_path)]
    for name, records, mode, meta in zip(names, (preds, probs.values()), ('baseline', 'yesno'), metas):
        wrong = sorted({str(r.get('mode')) for r in records if r.get('mode') != mode})
        if wrong:
            raise ValueError(f'{name}: expected mode {mode!r}, found {wrong}')
        if meta.get('split') != 'val':
            raise ValueError(f'{name}: split {meta.get("split")!r}, the comparison runs on val only')
    sha = [meta.get('adapter_sha256') for meta in metas]
    if not sha[0] or sha[0] != sha[1]:
        raise ValueError(f'{names[0]} / {names[1]}: adapter_sha256 is missing or differs')
    targets = {r['idx'] for r in rows if yesno.label_of(r['answer'])}
    if set(probs) != targets:
        raise ValueError(f'{names[1]}: missing {len(targets - set(probs))} yes/no questions, '
                         f'unexpected {len(set(probs) - targets)}')
    items = [{'wrong': int(vs.score_phi(preds[i]) < 1.0), 'D0': preds[i]['entropy'],
              'D1': yesno.merged_entropy(probs[i]['entropy_first'], probs[i]['label_probs']),
              'D2': yesno.binary_entropy(probs[i]['label_probs'])} for i in sorted(targets)]
    return items, sha[0]


def _yesno(a):
    inputs = {name: _yesno_items(name, *getattr(a, name), a.data_root) for name in DATASETS}   # check all first
    datasets = {name: {**yesno.compare(items), 'adapter_sha256': sha} for name, (items, sha) in inputs.items()}
    print('yes/no questions on val. official: D0 (eq. 4); D1 / D2 are checks, not eq. 4')
    for name, c in datasets.items():
        print(f'[{name}] n={c["n"]} wrong={c["n_wrong"]} excluded={c["n_excluded"]}')
        for d in yesno.DEFINITIONS:
            auc, diff = c['auc'][d], c['diff'].get(d)
            line = f'  {d}  AUC {"-" if auc is None else format(auc, ".3f")}'
            if diff and diff['ci95']:
                line += f'  vs D0 {diff["value"]:+.3f} [{diff["ci95"][0]:+.3f}, {diff["ci95"][1]:+.3f}]'
            print(line)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'split': 'val', 'official': 'D0', 'definitions': yesno.DESCRIPTIONS,
                               'datasets': datasets}, indent=2), encoding='utf-8')
    print('Saved:', out)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select', help='choose the checkpoint on val')
    s.add_argument('--dataset', required=True, choices=DATASETS)
    s.add_argument('--split', default='val', choices=('val',), help='selection on test is not allowed')
    s.add_argument('--data-root')
    s.add_argument('--run', required=True, nargs='+', metavar='STEP=PATH')
    s.add_argument('--out', default='selected.json')
    r = sub.add_parser('report', help='score the test rows')
    r.add_argument('--dataset', required=True, choices=DATASETS)
    r.add_argument('--data-root')
    r.add_argument('--baseline', required=True)
    r.add_argument('--entropy')
    r.add_argument('--predictions-name', help='release name the entropy file must have been computed on')
    r.add_argument('--top-percent', type=float, default=5.0)
    r.add_argument('--rag100')
    r.add_argument('--rag5')
    r.add_argument('--cfg5')
    r.add_argument('--check-baseline', help='baseline regenerated in this environment')
    r.add_argument('--out-dir', required=True)
    r.add_argument('--checkpoint-name', default='unrecorded', help='name in manifests/checkpoints.json')
    y = sub.add_parser('yesno', help='compare yes/no entropy definitions on val')
    for name in DATASETS:
        y.add_argument(f'--{name}', required=True, nargs=2, metavar=('PREDS', 'PROBS'),
                       help=f'{name}: val baseline run and val yesno run')
    y.add_argument('--data-root')
    y.add_argument('--out', required=True)
    return p


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.command == 'select':
        rows, _ = load_split(a.dataset, 'val', data_root=a.data_root, check_images=False)
        scores = {}
        for spec in a.run:
            step, _, path = spec.partition('=')
            if not step.isdigit() or not path:
                parser.error(f'--run expects STEP=PATH, got {spec!r}')
            records = align(rows, read_predictions(path).values(), Path(path).name)
            scores[int(step)] = vs.summarize(records, vs.score_phi)['overall']
            print(f'[val] checkpoint-{step}: overall {scores[int(step)]:.2f}')
        step = select_checkpoint(scores)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({'dataset': a.dataset, 'step': step, 'val_overall': scores}, indent=2),
                               encoding='utf-8')
        print(f'Selected checkpoint-{step} (written to {a.out})')
        return
    if a.command == 'yesno':
        return _yesno(a)
    if (a.rag5 or a.cfg5) and not a.entropy:
        parser.error('--rag5 / --cfg5 need --entropy')
    rows, base_scores, entropy, results = _report(a)       # every input is checked before anything is written
    _print_report(a, rows, base_scores, entropy, results)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    for r in results:
        (out / f'{today}_phi35_{a.dataset}_{r["row"]}.json').write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', out)


if __name__ == '__main__':
    main()
