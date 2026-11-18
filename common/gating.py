"""불확실성 게이팅: 식 4 엔트로피 집계, 상위 N% 선택, AUC. 모델과 무관한 순수 로직 (GPU 불필요).

엔트로피 정의가 둘이라 둘 다 지원한다.
  - eq4 (팀원 phi3_urp 정의): 답의 각 토큰 위치에서 어휘 전체의 엔트로피를 구하고 **특수 토큰(<|end|> 등)을 뺀** 토큰들로 평균
  - with_end (이전 우리 정의): <|end|> 위치까지 평균에 포함
VQA-RAD 베이스라인에서 전체 AUC는 with_end 0.846, eq4 0.817 (open 0.850 → 0.823, closed 0.802 → 0.798)이었다.
논문은 특수 토큰 처리를 말하지 않아서 어느 쪽이 맞는지 확정하지 못했다.
"""


def aggregate_entropy(token_entropies, token_ids=None, special_ids=(), include_special=False):
    """token_entropies: 답 토큰 위치별 엔트로피 목록 (마지막 위치가 <|end|> 예측일 수 있음).
    token_ids: 각 위치에서 실제 고른 토큰 id (special_ids 와 비교). include_special=True 면 모두 평균.
    평균에 쓸 위치가 없으면 None."""
    if include_special or not special_ids or token_ids is None:
        keep = list(token_entropies)
    else:
        assert len(token_entropies) == len(token_ids), (len(token_entropies), len(token_ids))
        special = set(special_ids)
        keep = [h for h, t in zip(token_entropies, token_ids) if t not in special]
    return sum(keep) / len(keep) if keep else None


def select_top_percent(entropies, percent=5.0):
    """엔트로피가 높은 순으로 상위 percent% 문항의 인덱스를 돌려준다.
    개수 = max(1, round(N * percent / 100)) (VQA-RAD 451 → 23, SLAKE 1061 → 53). None 은 선택하지 않는다.
    동점이면 인덱스가 작은 문항이 먼저 (안정 정렬) — 팀원 구현(infer.select_top)과 같은 규칙."""
    valid = [(i, e) for i, e in enumerate(entropies) if e is not None]
    k = max(1, round(len(entropies) * percent / 100))
    order = sorted(valid, key=lambda t: (-t[1], t[0]))
    return [i for i, _ in order[:k]]


def auc_wrong_vs_entropy(wrong, entropies):
    """엔트로피가 오답을 가려내는 정도(ROC-AUC). wrong: 1=오답. 한 클래스뿐이면 None."""
    from sklearn.metrics import roc_auc_score
    pairs = [(w, e) for w, e in zip(wrong, entropies) if e is not None]
    ys = [w for w, _ in pairs]
    if len(set(ys)) < 2:
        return None
    return float(roc_auc_score(ys, [e for _, e in pairs]))
