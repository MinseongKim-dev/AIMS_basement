"""AIMS pipeline module.

4가지 설정을 하나의 클래스로 처리합니다.
    - baseline:   1차 추론만, RAG 없음
    - always_rag: 항상 RAG 후 2차 추론
    - adaptive:   entropy 보고 direct / RAG 결정
    - expert:     adaptive + entropy 매우 높으면 전문가 이관 (B방식: 정답 라벨 사용)

TODO (BiomedCLIP/ViT 교체 시 수정 필요):
    - second_inference(): logits 보정 방식 → 텍스트 컨텍스트 입력 방식으로 교체
    - AIMSPipeline.__init__(): model 인자를 VLM으로 교체
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from aims.uncertainty.metrics import compute_entropy, compute_confidence
from aims.rag.retriever import HybridRetriever
from aims.rag.embed import Document, EmbedIndexer

RouteType = Literal["direct", "rag", "expert"]
ModeType  = Literal["baseline", "always_rag", "adaptive", "expert"]

# 버그 수정 목록:
# 1. _adaptive() 호출 시 인자 순서 오류 (mode가 label 위치로 전달됨) → 수정
# 2. IDX_TO_ANSWER 오타 (IDX_To_ANSWER) → 클래스 변수로 통일
# 3. RAG 후 uncertainty 미갱신 → adjusted_logits 기준으로 재계산
# 4. second_inference()의 logits * bias → logits + bias (additive로 변경)
# 5. __init__() 인자로 IDX_TO_ANSWER를 받던 구조 → 클래스 변수로 고정
# 6. retrieve_with_scores() zip iterator 반환 → list로 변경 (retriever.py)
# 7. compute_ece() device mismatch → metrics.py에서 수정
# 8. mc_dropout_entropy() 모델 상태 미보존 → metrics.py에서 수정


@dataclass
class PipelineOutput:
    """파이프라인 단일 샘플 출력.

    Attributes:
        prediction:     예측 답변 ("yes" / "no" / "expert")
        confidence:     최대 softmax 확률 [0, 1]
        uncertainty:    정규화 entropy [0, 1]
        route:          라우팅 결과 ("direct" / "rag" / "expert")
        retrieved_docs: RAG 검색 문서 (direct면 빈 리스트)
        mode:           실행 설정 이름
    """
    prediction:     str
    confidence:     float
    uncertainty:    float
    route:          RouteType
    retrieved_docs: List[Document] = field(default_factory=list)
    mode:           ModeType = "baseline"


class AIMSPipeline:
    """불확실성 기반 RAG 파이프라인.

    Args:
        model:            추론 모델 (SimpleMedCNN 등)
        indexer:          구축된 EmbedIndexer
        k:                RAG 검색 문서 수
        device:           "cpu" / "cuda"
        tau_low:          direct 판단 임계값 (entropy < tau_low → direct)
        tau_high:         expert 이관 임계값 (entropy > tau_high → expert)

    Usage:
        pipeline = AIMSPipeline(model, indexer)
        for mode in ["baseline", "always_rag", "adaptive", "expert"]:
            out = pipeline(image, question, mode=mode, label=label)
    """

    # 버그 2 수정: 클래스 변수로 고정 (오타 IDX_To_ANSWER 제거)
    IDX_TO_ANSWER = {0: "no", 1: "yes"}

    def __init__(
        self,
        model:     torch.nn.Module,
        indexer:   EmbedIndexer,
        k:         int   = 3,
        device:    str   = "cpu",
        tau_low:   float = 0.5,   # 버그 5 수정: threshold → tau_low
        tau_high:  float = 0.8,   # 버그 5 수정: expert_threshold → tau_high
    ):
        self.model      = model.to(device).eval()
        self.retriever  = HybridRetriever(indexer)
        self.k          = k
        self.device     = device
        self.tau_low    = tau_low
        self.tau_high   = tau_high

    def __call__(
        self,
        image:    torch.Tensor,
        question: str,
        mode:     ModeType       = "baseline",
        label:    Optional[int]  = None,
    ) -> PipelineOutput:
        """단일 샘플 파이프라인 실행.

        Args:
            image:    (1, 3, H, W) 이미지 텐서
            question: yes/no 질문
            mode:     baseline / always_rag / adaptive / expert
            label:    정답 라벨 (expert 시뮬레이션용, 0=no / 1=yes)

        Returns:
            PipelineOutput
        """
        image = image.to(self.device)

        logits      = self.first_inference(image)
        uncertainty = compute_entropy(logits).item()
        confidence  = compute_confidence(logits).item()

        if mode == "baseline":
            return self._baseline(logits, uncertainty, confidence, mode)

        elif mode == "always_rag":
            return self._always_rag(logits, question, uncertainty, confidence, mode)

        elif mode in ("adaptive", "expert"):
            # 버그 1 수정: 인자 순서 (logits, uncertainty, confidence, question, mode, label)
            return self._adaptive(logits, uncertainty, confidence, question, mode, label)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------ #
    # 추론 메서드                                                          #
    # ------------------------------------------------------------------ #

    def first_inference(self, image: torch.Tensor) -> torch.Tensor:
        """1차 추론: 이미지 → (1, num_classes) logits."""
        with torch.no_grad():
            logits = self.model(image)
        return logits

    def second_inference(
        self,
        logits: torch.Tensor,
        docs:   List[Document],
    ) -> torch.Tensor:
        """RAG 문서를 반영한 2차 추론.

        TODO (BiomedCLIP/ViT 교체 시 수정):
            현재: 검색 문서의 답변 분포로 logits를 보정 (additive bias)
            교체: 검색 문서를 텍스트 컨텍스트로 모델에 직접 입력
                  예) prompt = f"Context: {doc.text}\nQuestion: {question}"
                      model(image, prompt) → logits

        버그 4 수정:
            logits * bias → logits + bias
            곱셈 방식은 음수 logits에서 방향이 반전되는 문제 있음
            덧셈 방식이 "약하게 밀어주는" 의도에 더 자연스러움
        """
        alpha = 0.3

        if not docs:
            return logits

        answer_counts = {"yes": 0, "no": 0}
        for doc in docs:
            ans = doc.answer.lower().strip()
            if ans in answer_counts:
                answer_counts[ans] += 1

        total = len(docs)
        no_ratio  = answer_counts["no"]  / total
        yes_ratio = answer_counts["yes"] / total

        # (1, 2) 형태로 맞춰서 logits에 더함
        bias = torch.tensor(
            [[no_ratio, yes_ratio]],
            device=logits.device
        ) * alpha

        return logits + bias   # 버그 4 수정: * → +

    # ------------------------------------------------------------------ #
    # 라우팅 메서드                                                        #
    # ------------------------------------------------------------------ #

    def _baseline(
        self,
        logits:      torch.Tensor,
        uncertainty: float,
        confidence:  float,
        mode:        ModeType,
    ) -> PipelineOutput:
        """baseline: 1차 추론 결과를 바로 반환."""
        pred_idx = logits.argmax(dim=-1).item()
        return PipelineOutput(
            prediction=self.IDX_TO_ANSWER[pred_idx],
            confidence=confidence,
            uncertainty=uncertainty,
            route="direct",
            retrieved_docs=[],
            mode=mode,
        )

    def _always_rag(
        self,
        logits:      torch.Tensor,
        question:    str,
        uncertainty: float,
        confidence:  float,
        mode:        ModeType,
    ) -> PipelineOutput:
        """always_rag: 무조건 RAG 후 2차 추론."""
        docs            = self.retriever.retrieve(question, k=self.k)
        adjusted_logits = self.second_inference(logits, docs)
        pred_idx        = adjusted_logits.argmax(dim=-1).item()

        # 버그 3 수정: adjusted_logits 기준으로 confidence 재계산
        adjusted_confidence = F.softmax(adjusted_logits, dim=-1).max().item()

        return PipelineOutput(
            prediction=self.IDX_TO_ANSWER[pred_idx],
            confidence=adjusted_confidence,
            uncertainty=uncertainty,   # uncertainty는 1차 기준 유지 (라우팅 판단 기준)
            route="rag",
            retrieved_docs=docs,
            mode=mode,
        )

    def _adaptive(
        self,
        logits:      torch.Tensor,
        uncertainty: float,
        confidence:  float,
        question:    str,
        mode:        ModeType,          # 버그 1 수정: mode와 label 순서 명확화
        label:       Optional[int] = None,
    ) -> PipelineOutput:
        """adaptive / expert: entropy 기반 라우팅.

            uncertainty < tau_low  → direct
            tau_low ≤ uncertainty  → RAG
            uncertainty > tau_high → expert (expert mode만, B방식)
        """
        # expert 이관 먼저 확인
        if mode == "expert" and uncertainty > self.tau_high:
            if label is not None:
                expert_answer = self.IDX_TO_ANSWER[label]  # 버그 2 수정: 오타 제거
            else:
                expert_answer = "expert"

            return PipelineOutput(
                prediction=expert_answer,
                confidence=1.0,        # 전문가는 100% 확신으로 가정
                uncertainty=uncertainty,
                route="expert",
                retrieved_docs=[],
                mode=mode,
            )

        # 확신 → direct
        if uncertainty < self.tau_low:
            return self._baseline(logits, uncertainty, confidence, mode)

        # 불확실 → RAG
        return self._always_rag(logits, question, uncertainty, confidence, mode)


if __name__ == "__main__":
    from aims.data.dataset import VQARadDataset, load_vqarad
    from aims.models.medcnn import SimpleMedCNN
    from aims.rag.embed import EmbedIndexer
    from torch.utils.data import DataLoader

    train_data, test_data = load_vqarad(only_yes_no=True)

    try:
        indexer = EmbedIndexer.load("data/faiss_index")
    except Exception:
        print("인덱스 없음 → 새로 구축")
        indexer = EmbedIndexer()
        indexer.build_from_hf(train_data)

    model    = SimpleMedCNN()
    pipeline = AIMSPipeline(model, indexer)

    dataset = VQARadDataset(test_data)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    image, question, label = next(iter(loader))

    print(f"Question: {question[0]}")
    print(f"Label:    {'yes' if label.item() == 1 else 'no'}")
    print()

    for mode in ["baseline", "always_rag", "adaptive", "expert"]:
        out = pipeline(image, question[0], mode=mode, label=label.item())
        print(f"[{mode:12s}] pred={out.prediction:6s} | "
              f"uncertainty={out.uncertainty:.3f} | "
              f"confidence={out.confidence:.3f} | "
              f"route={out.route:6s} | "
              f"docs={len(out.retrieved_docs)}")