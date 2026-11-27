"""Uncertainty checks on yes/no questions, computed from the first answer-token distribution.

  D0  eq. 4 of the spec (the official definition; recorded by the baseline run)
  D1  vocabulary entropy after merging the case / spacing variants of "yes" and of "no"
  D2  binary entropy of P(yes) against P(no)

D1 and D2 are checks only: they are not eq. 4 and nothing selects questions with them.
"""
import math

import numpy as np

from . import REPO_ROOT  # noqa: F401  (puts `common` on sys.path)
from common.gating import auc_wrong_vs_entropy

LABELS = ('yes', 'no')
DEFINITIONS = ('D0', 'D1', 'D2')
DESCRIPTIONS = {   # written into the saved report
    'D0': 'eq. 4 of the spec (official): answer-token vocabulary entropy, special tokens excluded',
    'D1': 'check only, not eq. 4: first-token vocabulary entropy with the yes / no variants merged',
    'D2': 'check only, not eq. 4: first-token binary entropy of P(yes) against P(no)',
}


def label_of(answer):
    text = str(answer).strip().lower()
    return text if text in LABELS else None


def label_token_ids(tokenizer):
    """{label: ids of every token that decodes to the label, ignoring case and surrounding spaces}."""
    ids = {label: [] for label in LABELS}
    for i in range(len(tokenizer)):
        label = label_of(tokenizer.decode([i]))
        if label:
            ids[label].append(i)
    empty = [label for label in LABELS if not ids[label]]
    if empty:
        raise ValueError(f'the tokenizer has no single token for {empty}')
    return ids


def probe_record(logp, ids):
    """What is stored per question, from the log-probabilities (V,) of the first answer token."""
    p = logp.exp()
    entropy = float(-(p * logp.masked_fill(p == 0, 0.0)).sum())
    return {'entropy_first': entropy,
            'label_probs': {label: {str(i): float(p[i]) for i in ids[label]} for label in LABELS}}


def _plogp(p):
    return p * math.log(p) if p > 0 else 0.0


def merged_entropy(entropy_first, label_probs):
    """D1: replace each label's token terms in the entropy by one term for their total."""
    h = entropy_first
    for probs in label_probs.values():
        h += sum(_plogp(p) for p in probs.values()) - _plogp(sum(probs.values()))
    return h


def binary_entropy(label_probs):
    """D2: None when neither label has any probability."""
    totals = [sum(label_probs[label].values()) for label in LABELS]
    mass = sum(totals)
    if mass <= 0:
        return None
    return -sum(_plogp(t / mass) for t in totals)


def compare(items, n_boot=10000, seed=0):
    """AUC (wrong answer vs entropy) per definition and the paired bootstrap difference from D0.

    Questions lacking any definition are left out of all of them."""
    kept = [x for x in items if all(x[d] is not None for d in DEFINITIONS)]
    wrong = np.array([x['wrong'] for x in kept])
    values = {d: np.array([x[d] for x in kept]) for d in DEFINITIONS}
    auc = {d: auc_wrong_vs_entropy(wrong, values[d]) for d in DEFINITIONS}
    others = DEFINITIONS[1:]
    diff = {d: {'value': None, 'ci95': None} for d in others}
    if auc['D0'] is not None:
        rng = np.random.default_rng(seed)
        samples = {d: [] for d in others}
        for _ in range(n_boot):
            s = rng.integers(0, len(kept), len(kept))
            if wrong[s].min() == wrong[s].max():
                continue
            ref = auc_wrong_vs_entropy(wrong[s], values['D0'][s])
            for d in others:
                samples[d].append(auc_wrong_vs_entropy(wrong[s], values[d][s]) - ref)
        for d in others:
            ci = [float(v) for v in np.percentile(samples[d], [2.5, 97.5])] if samples[d] else None
            diff[d] = {'value': auc[d] - auc['D0'], 'ci95': ci}
    return {'n': len(kept), 'n_wrong': int(wrong.sum()), 'n_excluded': len(items) - len(kept),
            'auc': auc, 'diff': diff}
