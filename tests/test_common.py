"""common/ 순수 로직 단위 테스트 (CPU, 모델 불필요). 실행: python tests/test_common.py  (pytest 로도 실행됨)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import gating as G          # noqa: E402
from common import highlights as H      # noqa: E402
from common import build_rag_testjson as B   # noqa: E402


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


def test_trainqa_groups():
    items = [('a.jpg', 'pa', 'Is it CT?', 'yes'), ('a.jpg', 'pa', 'what organ?', 'lung'),
             ('b.jpg', 'pb', 'Is there a mass?', 'no'),                     # 답이 no 인 QA 는 뺀다 → b 이미지는 항목 자체가 사라진다
             ('c.jpg', 'pc', 'Which plane', 'axial')]
    g = B.paper_trainqa_groups(items)
    assert g == [('a.jpg', 'pa', 'Is it CT? yes. what organ? lung.'), ('c.jpg', 'pc', 'Which plane? axial.')], g


def test_pathvqa_items():
    import tempfile
    # PathVQA 는 행마다 파일 이름이 다르므로 같은 사진은 내용(바이트)으로 묶는다
    rows = [(b'img-one', 'what is shown?', 'liver'), (b'img-two', 'is it red?', 'no'),
            (b'img-one', 'is it enlarged?', 'yes')]
    with tempfile.TemporaryDirectory() as d:
        items = B.pathvqa_items(iter(rows), d)
        names = [n for n, _, _, _ in items]
        assert names[0] == names[2] != names[1] and all(n.startswith('pathvqa_') for n in names)
        assert [(q, a) for _, _, q, a in items] == [(r[1], r[2]) for r in rows]
        assert [open(path, 'rb').read() for _, path, _, _ in items] == [r[0] for r in rows]
        g = B.paper_trainqa_groups(items)
        assert len(g) == 1 and g[0][2] == 'what is shown? liver. is it enlarged? yes.'


def test_kb_tag():
    assert B.kb_tag(['peir', 'roco', 'trainqa'], False) == 'peir_roco_trainqa'
    assert B.kb_tag(['peir', 'roco', 'trainqa'], True) == 'peir_roco_trainqa_pathvqa'     # 기존 파일을 덮지 않는다
    try:
        B.kb_tag(['peir'], True)
        raise AssertionError('pathvqa without trainqa accepted')
    except ValueError:
        pass


def test_retrieval_modes():
    import numpy as np
    kb = B.KB()
    # 항목0: train QA(이미지 특징이 쿼리와 같음, 텍스트는 직교) / 항목1: PEIR(텍스트가 쿼리와 가까움)
    kb.add(['qa text'], 'trainqa', ['img0.jpg'], np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]]))
    kb.add(['peir text'], 'peir', ['1'], np.array([[0.8, 0.6]]))
    kb.finalize()
    q = np.array([1.0, 0.0]); qt = np.array([0.0, 1.0])
    assert kb.retrieve(q, k=2, mode='cross')[0] == 1                      # 이미지→텍스트만: PEIR 가 1순위
    assert kb.retrieve(q, k=2, qtxt=qt, mode='union')[0] == 0             # union: train QA 의 이미지↔이미지 유사도(1.0)가 이긴다


if __name__ == '__main__':
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); n += 1; print('통과', k)
    print(f'{n}개 테스트 통과')
