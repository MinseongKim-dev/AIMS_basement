"""Uncertainty metric scaffold module."""

"""불확실성 측정 모듈.

파이프라인에서 사용하는 순서:
    1. 모델 logits → compute_entropy() → 불확실성 수치
    2. 수치 → 임계값 비교 → direct / rag / expert 라우팅
    3. (선택) mc_dropout_entropy() → 더 정확한 불확실성 추정
    4. (실험 후) compute_ece() → 모델 캘리브레이션 평가
"""

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# 1. 기본 Entropy                                                      #
# ------------------------------------------------------------------ #

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Softmax entropy를 계산합니다.

    H(p) = -Σ p_i * log(p_i)

    정규화: H / log(num_classes) → [0, 1] 범위
        - 0: 완전히 확신 (한 클래스에 확률 1)
        - 1: 완전히 불확실 (모든 클래스 균등)

    정규화하는 이유:
        yes/no 이진 분류면 최대 entropy = log(2) ≈ 0.693
        나중에 다중 클래스로 확장하면 최대값이 바뀜
        → 임계값 τ를 클래스 수에 무관하게 고정할 수 있음

    Args:
        logits: (N, num_classes) 또는 (num_classes,) 텐서
                모델 출력 raw값 (softmax 전)

    Returns:
        (N,) 또는 scalar, 정규화된 entropy [0, 1]
    """
    # (num_classes,) 입력이면 배치 차원 추가
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    probs = F.softmax(logits, dim=-1)           # (N, C)
    num_classes = probs.shape[-1]

    # log(0) 방지를 위해 clamp
    log_probs = torch.log(probs.clamp(min=1e-9))
    raw_entropy = -(probs * log_probs).sum(dim=-1)  # (N,)

    # 정규화: [0, 1] 범위로
    max_entropy = torch.log(
        torch.tensor(float(num_classes), device=logits.device)
    )
    normalized = raw_entropy / max_entropy

    return normalized.squeeze()


def compute_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    최대 softmax 확률을 confidence로 반환합니다.

    confidence = max(softmax(logits))
    entropy와 역관계: confidence 높으면 entropy 낮음

    Args:
        logits: (N, num_classes) 또는 (num_classes,)

    Returns:
        (N,) 또는 scalar [0, 1]
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    probs = F.softmax(logits, dim=-1)
    confidence, _ = probs.max(dim=-1)
    return confidence.squeeze()


# ------------------------------------------------------------------ #
# 2. MC Dropout entropy                                                #
# ------------------------------------------------------------------ #

def mc_dropout_entropy(
    model: torch.nn.Module,
    image: torch.Tensor,
    n_samples: int = 10,
    question: str = None,
) -> float:
    """
    MC Dropout으로 불확실성을 추정합니다.

    BiomedCLIP처럼 text_encoder를 가진 모델은 question을 함께 전달합니다.
    Dropout 레이어가 있는 head 부분에만 dropout이 적용됩니다.

    Args:
        model:     Dropout 레이어가 있는 모델 (SimpleMedCNN, BiomedCLIP 등)
        image:     (1, 3, H, W) 단일 이미지 텐서
        n_samples: dropout 샘플 수
        question:  VLM용 질문 문자열 (BiomedCLIP 등에서 사용)

    Returns:
        scalar float, 정규화된 entropy [0, 1]
    """
    was_training = model.training
    model.train()

    with torch.no_grad():
        if hasattr(model, 'text_encoder') and question is not None:
            logits_list = [model(image, [question]) for _ in range(n_samples)]
        else:
            logits_list = [model(image) for _ in range(n_samples)]
        logits_stack = torch.stack(logits_list, dim=0).squeeze(1)  # (N, C)

    probs      = F.softmax(logits_stack, dim=-1)
    mean_probs = probs.mean(dim=0)

    num_classes = mean_probs.shape[0]
    log_probs   = torch.log(mean_probs.clamp(min=1e-9))
    raw_entropy = -(mean_probs * log_probs).sum()
    max_entropy = torch.log(torch.tensor(float(num_classes), device=image.device))

    model.train(was_training)
    return (raw_entropy / max_entropy).item()


# ------------------------------------------------------------------ #
# 3. ECE (Expected Calibration Error)                                  #
# ------------------------------------------------------------------ #

def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    ECE(Expected Calibration Error)를 계산합니다.

    캘리브레이션이란:
        모델이 "70% 확신"이라고 할 때 실제로 70% 맞아야 잘 캘리브레이션됨
        ECE = 0: 완벽한 캘리브레이션
        ECE 높음: 과신(overconfident) 또는 과소신(underconfident)

    AIMS 프로젝트에서의 역할:
        entropy 임계값으로 라우팅하는데,
        모델이 잘못 캘리브레이션되어 있으면 entropy가 실제 오류율과 안 맞음
        → ECE로 캘리브레이션 상태 확인 후 임계값 조정 근거로 사용

    Args:
        logits: (N, num_classes) 전체 test set 예측값
        labels: (N,) 정답 라벨
        n_bins: confidence 구간 수 (기본 10 → 0~0.1, 0.1~0.2, ...)

    Returns:
        ECE 값 (float), 낮을수록 잘 캘리브레이션됨
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # 이 구간에 해당하는 샘플
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_size = in_bin.sum().item()

        if bin_size == 0:
            continue

        # 구간 내 평균 confidence와 평균 accuracy의 차이
        bin_confidence = confidences[in_bin].mean().item()
        bin_accuracy = accuracies[in_bin].float().mean().item()

        ece += (bin_size / len(logits)) * abs(bin_confidence - bin_accuracy)

    return ece


if __name__ == "__main__":
    # ---- compute_entropy / compute_confidence 테스트 ----
    print("=== Entropy / Confidence 테스트 ===")

    # 완전히 확신하는 경우
    certain_logits = torch.tensor([[10.0, -10.0]])   # yes에 몰림
    print(f"확신 logits: entropy={compute_entropy(certain_logits):.4f}, "f"confidence={compute_confidence(certain_logits):.4f}")
    # entropy ≈ 0, confidence ≈ 1

    # 완전히 불확실한 경우
    uncertain_logits = torch.tensor([[0.0, 0.0]])    # 균등 분포
    print(f"불확실 logits: entropy={compute_entropy(uncertain_logits):.4f}, "f"confidence={compute_confidence(uncertain_logits):.4f}")
    # entropy ≈ 1, confidence ≈ 0.5

    # 배치 입력
    batch_logits = torch.randn(8, 2)
    entropies = compute_entropy(batch_logits)
    print(f"\n배치(8개) entropy shape: {entropies.shape}")  # (8,)
    print(f"entropy 범위: [{entropies.min():.3f}, {entropies.max():.3f}]")

    # ---- ECE 테스트 ----
    print("\n=== ECE 테스트 ===")
    N = 100
    logits = torch.randn(N, 2)
    labels = torch.randint(0, 2, (N,))
    ece = compute_ece(logits, labels)
    print(f"랜덤 예측 ECE: {ece:.4f}")  # 랜덤이면 ECE 높게 나옴