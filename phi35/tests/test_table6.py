"""Table 6 metrics (CPU). Run: python phi35/tests/test_table6.py"""
import math

import numpy as np

from _util import raises, run

from phi35cfg import table6  # noqa: E402


def test_accuracy_coverage_auc():
    # most confident first: right, right, wrong, wrong -> accuracies 1, 1, 2/3, 1/2
    auc = table6.accuracy_coverage_auc([.9, .8, .7, .6], [1, 1, 0, 0])
    assert abs(auc - (1 + 1 + 2 / 3 + .5) / 4) < 1e-12
    assert abs(table6.accuracy_coverage_auc([.6, .9, .7, .8], [0, 1, 0, 1]) - auc) < 1e-12
    # a tie takes its expected value over both orders: (1, .5) and (0, .5)
    tied = table6.accuracy_coverage_auc([.5, .5], [1, 0])
    assert abs(tied - .5) < 1e-12 and abs(table6.accuracy_coverage_auc([.5, .5], [0, 1]) - tied) < 1e-12


def test_ece_and_brier():
    # bins [0.9, 1.0]: conf .95 and 1.0, accuracy .5; bin [0.1, 0.2): conf .15, accuracy 0
    conf, ok = [.95, 1.0, .15], [1, 0, 0]
    assert abs(table6.ece(conf, ok) - (2 / 3 * abs(.5 - .975) + 1 / 3 * .15)) < 1e-12
    assert abs(table6.brier(conf, ok) - (.05 ** 2 + 1 + .15 ** 2) / 3) < 1e-12


def test_temperature():
    p = np.array([.2, .5, .9])
    assert np.allclose(table6.temperature_scale(p, 1.0), p)
    assert np.allclose(table6.temperature_scale(p, 2.0), p ** 2 / (p ** 2 + (1 - p) ** 2))
    assert np.isfinite(table6.temperature_scale(np.array([0.0, 1.0]), 3.0)).all()
    # confidence .99 but only 70% right: the fitted beta softens it towards .7
    conf, ok = np.full(100, .99), np.array([1] * 70 + [0] * 30)
    beta = table6.fit_temperature(conf, ok)
    assert beta < 1 and abs(table6.temperature_scale(conf, beta)[0] - .7) < 1e-3


def test_scaled_metrics():
    conf, ok = np.full(100, .99), np.array([1] * 70 + [0] * 30)
    m = table6.scaled_metrics(conf, ok)
    assert m['ece_t'] < table6.ece(conf, ok) and m['bs_t'] < table6.brier(conf, ok)
    assert m == table6.scaled_metrics(conf, ok)                       # fixed split seed
    with raises(ValueError):
        table6.scaled_metrics([.5] * 4, [1] * 4)


def test_majority():
    assert table6.majority(['No', 'Yes', 'yes.', 'No', ' YES']) == ('Yes', 3 / 5)
    assert table6.majority(['no', 'yes']) == ('no', .5)               # tie: the first sampled
    with raises(ValueError):
        table6.majority([])


def _record(idx, answer, prediction, logp, entropy, samples):
    return {'idx': idx, 'answer': answer, 'answer_type': 'CLOSED', 'prediction': prediction,
            'token_logp': logp, 'entropy': entropy, 'samples': samples}


def test_table():
    records = []
    for i in range(20):
        wrong = i % 5 == 0
        answer = 'left' if i >= 16 else 'yes'
        text = 'no' if wrong else answer
        records.append(_record(i, answer, text, [math.log(.6 if wrong else .9)] * 2, .6 if wrong else .1,
                               [text] * (6 if wrong else 10) + [answer] * (4 if wrong else 0)))
    records.append(_record(20, 'yes', '', [], None, ['yes'] * 10))      # empty greedy answer
    t = table6.table(records)
    assert set(t) == {'closed', 'yes/no'} and t['closed']['n_excluded'] == 1
    e = t['closed']['rows']['Entropy']
    assert (e['n'], e['accuracy'], e['auroc']) == (20, .8, 1.0)
    assert abs(e['ece'] - (.2 * .6 + .8 * .1)) < 1e-9
    assert e['auc'] > e['accuracy'] and set(e) == {'n', 'accuracy', 'ece', 'ece_t', 'bs_t', 'auc', 'auroc'}
    label = t['closed']['rows']['Label prob.']
    assert label['accuracy'] == .8 and label['auroc'] == 1.0
    assert t['yes/no']['rows']['Entropy']['n'] == 16


if __name__ == '__main__':
    run(globals())
