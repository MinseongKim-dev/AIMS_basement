"""Table 6 of arXiv:2507.09209 (fine-tuned rows) with the metric definitions of its reference [44]
(Tian et al., arXiv:2305.14975). Every choice is fixed in docs/superpowers/specs/2026-11-28-table6-design.md;
none of them may be tuned towards the paper's numbers.

AUC is the area under the selective accuracy - coverage curve, not ROC-AUC. Its value for a random
ordering equals the accuracy, so always read it next to the accuracy.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar

from . import REPO_ROOT  # noqa: F401  (puts `common` on sys.path)
from . import yesno
from common import vqa_scoring as vs
from common.gating import auc_wrong_vs_entropy

N_SAMPLES = 10      # Label prob.: samples at temperature 1, top-p 1
N_BINS = 10         # equal-width ECE bins (neither paper states a number)
N_FOLDS = 5         # temperature: fit on one fold, evaluate on the others
SEED = 0
EPS = 1e-6


def accuracy_coverage_auc(confidence, correct):
    """Mean accuracy of the k most confident questions over k = 1..n; ties take their expected value."""
    conf, ok = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    order = np.argsort(-conf, kind='stable')
    conf, ok = conf[order], ok[order]
    _, group = np.unique(conf, return_inverse=True)
    ok = (np.bincount(group, weights=ok) / np.bincount(group))[group]
    return float((np.cumsum(ok) / np.arange(1, len(ok) + 1)).mean())


def ece(confidence, correct, bins=N_BINS):
    conf, ok = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    which = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(ok[which == b].sum() - conf[which == b].sum()) for b in range(bins)) / len(conf))


def brier(confidence, correct):
    return float(np.mean((np.asarray(confidence, dtype=float) - np.asarray(correct, dtype=float)) ** 2))


def temperature_scale(confidence, beta):
    """p' = p^beta / (p^beta + (1 - p)^beta)."""
    p = np.clip(np.asarray(confidence, dtype=float), EPS, 1 - EPS)
    logit = beta * (np.log(p) - np.log1p(-p))
    return 1 / (1 + np.exp(-logit))


def fit_temperature(confidence, correct):
    """beta minimising the negative log likelihood of the correctness labels, ln(beta) in [-5, 5]."""
    ok = np.asarray(correct, dtype=float)

    def nll(log_beta):
        p = np.clip(temperature_scale(confidence, math.exp(log_beta)), 1e-12, 1 - 1e-12)
        return -float(np.mean(ok * np.log(p) + (1 - ok) * np.log(1 - p)))

    return math.exp(minimize_scalar(nll, bounds=(-5, 5), method='bounded').x)


def scaled_metrics(confidence, correct, folds=N_FOLDS, seed=SEED):
    conf, ok = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    if len(conf) < folds:
        raise ValueError(f'{len(conf)} questions cannot be split into {folds} folds')
    parts = np.array_split(np.random.default_rng(seed).permutation(len(conf)), folds)
    eces, briers = [], []
    for k, fit in enumerate(parts):
        rest = np.concatenate([p for j, p in enumerate(parts) if j != k])
        scaled = temperature_scale(conf[rest], fit_temperature(conf[fit], ok[fit]))
        eces.append(ece(scaled, ok[rest]))
        briers.append(brier(scaled, ok[rest]))
    return {'ece_t': float(np.mean(eces)), 'bs_t': float(np.mean(briers))}


def majority(samples):
    """(the most common answer as first written, its share). Answers are grouped after normalisation."""
    if not samples:
        raise ValueError('no samples')
    groups = {}
    for text in samples:
        groups.setdefault(tuple(vs.normalize(text)), []).append(text)
    best = max(groups.values(), key=len)          # max keeps the first group on a tie
    return best[0], len(best) / len(samples)


def _correct(record, prediction):
    return int(vs.score_phi({**record, 'prediction': prediction}) >= 1.0)


def _row(order, confidence, correct):
    """order: higher = more confident."""
    ok = np.asarray(correct)
    auroc = auc_wrong_vs_entropy(1 - ok, -np.asarray(order, dtype=float))
    return {'n': len(ok), 'accuracy': float(ok.mean()), 'ece': ece(confidence, ok),
            **scaled_metrics(confidence, ok), 'auc': accuracy_coverage_auc(order, ok), 'auroc': auroc}


def table(records):
    """{'closed' | 'yes/no': {'n_excluded', 'rows': {'Label prob.', 'Entropy'}}} from table6 run records.

    A question without a greedy answer (no entropy) is left out of both rows."""
    out = {}
    subsets = {'closed': [r for r in records if r['answer_type'] == 'CLOSED']}
    subsets['yes/no'] = [r for r in subsets['closed'] if yesno.label_of(r['answer'])]
    for name, subset in subsets.items():
        kept = [r for r in subset if r['entropy'] is not None]
        label = [majority(r['samples']) for r in kept]
        share = [s for _, s in label]
        token_prob = [math.exp(sum(r['token_logp']) / len(r['token_logp'])) for r in kept]
        out[name] = {'n_excluded': len(subset) - len(kept), 'rows': {
            'Label prob.': _row(share, share, [_correct(r, a) for r, (a, _) in zip(kept, label)]),
            'Entropy': _row([-r['entropy'] for r in kept], token_prob,
                            [_correct(r, r['prediction']) for r in kept])}}
    return out
