"""common/ 순수 로직 단위 테스트 (CPU, 모델 불필요). 실행: python tests/test_common.py  (pytest 로도 실행됨)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import gating as G          # noqa: E402
from common import highlights as H      # noqa: E402


def test_aggregate_entropy():
    # 답 "yes" + <|end|>: 토큰 위치 엔트로피 [0.8, 0.1], 두 번째가 특수 토큰(id 9)
    assert abs(G.aggregate_entropy([0.8, 0.1], [5, 9], special_ids={9}) - 0.8) < 1e-12          # eq4: 특수 토큰 제외
    assert abs(G.aggregate_entropy([0.8, 0.1], [5, 9], special_ids={9}, include_special=True) - 0.45) < 1e-12
    assert G.aggregate_entropy([0.3], [9], special_ids={9}) is None                              # 평균에 쓸 위치가 없음
    assert abs(G.aggregate_entropy([0.2, 0.4]) - 0.3) < 1e-12                                    # special_ids 가 없으면 모두 평균


def test_select_top_percent():
    assert len(G.select_top_percent([0.1] * 451, 5)) == 23          # VQA-RAD
    assert len(G.select_top_percent([0.1] * 1061, 5)) == 53         # SLAKE
    assert G.select_top_percent([0.5, 0.9, 0.9, 0.1], 50) == [1, 2]  # 동점이면 작은 인덱스 먼저
    assert G.select_top_percent([0.5, None, 0.7], 34) == [2]         # None 은 선택하지 않음
    assert G.select_top_percent([0.2, 0.1], 1) == [0]                # 최소 1개


def test_auc():
    assert G.auc_wrong_vs_entropy([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert G.auc_wrong_vs_entropy([0, 0, 0], [0.1, 0.2, 0.3]) is None   # 한 클래스뿐


def test_closed_highlights():
    cap = 'Chest x-ray shows a mass in the right lower lobe. The heart is normal. Both lungs are clear.'
    assert H.closed_sentence_highlights('Is there a mass in the right lung?', cap) == ['Chest x-ray shows a mass in the right lower lobe.']
    assert H.closed_sentence_highlights('Is this an MRI?', 'The heart is normal.') == []          # 겹치는 문장 없음 → 하이라이트 없음
    assert len(H.closed_sentence_highlights('heart lungs clear normal', cap, max_n=2)) <= 2


def test_inject():
    cap = 'A mass in the liver.'
    assert H.inject_caption(cap, 'liver', 'OPEN') == cap                    # 정답이 이미 캡션에 있으면 그대로
    assert H.inject_caption(cap, 'Blood vessel', 'OPEN') == 'A mass in the liver. Blood vessel.'
    assert H.inject_caption(cap, 'yes', 'CLOSED') == cap                    # closed 는 그대로
    assert H.inject_highlights('Blood vessel', 'OPEN') == ['Blood vessel']
    assert H.inject_highlights('yes', 'CLOSED', fallback=['s1']) == ['s1']


if __name__ == '__main__':
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); n += 1; print('통과', k)
    print(f'{n}개 테스트 통과')
