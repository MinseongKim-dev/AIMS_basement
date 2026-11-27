"""yes/no entropy definitions and their comparison (CPU). Run: python phi35/tests/test_yesno.py"""
import math

import torch

from _util import raises, run

from phi35cfg import yesno  # noqa: E402


class Tok:
    words = [' Yes', 'yes', 'No', 'maybe', '<s>']

    def __len__(self):
        return len(self.words)

    def decode(self, ids):
        return ''.join(self.words[i] for i in ids)


class TokNoNo(Tok):
    words = ['Yes', 'maybe']


def test_definitions():
    lp = {'yes': {'1': .5, '2': .5}, 'no': {'3': 0.0}}
    assert abs(yesno.merged_entropy(math.log(2), lp)) < 1e-9
    two = {'yes': {'1': .3, '2': .15}, 'no': {'3': .05}}
    assert abs(yesno.binary_entropy(two) - 0.325083) < 1e-5
    assert yesno.binary_entropy({'yes': {'1': 0.0}, 'no': {'3': 0.0}}) is None
    assert [yesno.label_of(a) for a in (' Yes', 'NO', 'yes it is', 'left')] == ['yes', 'no', None, None]


def test_token_ids():
    assert yesno.label_token_ids(Tok()) == {'yes': [0, 1], 'no': [2]}
    with raises(ValueError):
        yesno.label_token_ids(TokNoNo())


def test_probe_record():
    p = torch.tensor([.4, .3, .2, .1])
    r = yesno.probe_record(p.log(), {'yes': [0, 1], 'no': [2]})
    assert abs(r['entropy_first'] + float((p * p.log()).sum())) < 1e-6
    assert abs(r['label_probs']['yes']['1'] - .3) < 1e-6
    certain = yesno.probe_record(torch.tensor([0.0, float('-inf'), float('-inf')]), {'yes': [0], 'no': [1]})
    assert certain['entropy_first'] == 0.0 and certain['label_probs']['no']['1'] == 0.0


def test_compare_and_missing():
    d0, d1 = [.1, .5, .3, .2, .4, .6], [.1, .2, .3, .4, .5, .6]
    items = [{'wrong': int(i >= 3), 'D0': d0[i], 'D1': d1[i], 'D2': d0[i]} for i in range(6)]
    items.append({'wrong': 1, 'D0': None, 'D1': .0, 'D2': .0})
    c = yesno.compare(items, n_boot=500)
    assert (c['n'], c['n_wrong'], c['n_excluded']) == (6, 3, 1)
    assert abs(c['auc']['D0'] - 6 / 9) < 1e-9 and c['auc']['D1'] == 1.0
    assert c['diff']['D2'] == {'value': 0.0, 'ci95': [0.0, 0.0]}
    one = yesno.compare([{'wrong': 0, 'D0': .1, 'D1': .1, 'D2': .1}], n_boot=10)
    assert one['auc']['D0'] is None
    assert one['diff']['D1']['value'] is None


if __name__ == '__main__':
    run(globals())
