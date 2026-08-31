"""4가지 설정 비교 실험.

출력:
    설정별 정확도 / RAG 호출률 / expert 이관율 비교 테이블

TODO:
    - W&B 로깅 추가 (wandb.init, wandb.log)
    - ViT/BiomedCLIP 교체 후 재실험
"""

import os
import torch
import wandb
from dataclasses import dataclass, field
from typing import Dict, List
from torch.utils.data import DataLoader

from aims.data.dataset import load_vqarad,VQARadDataset
from aims.models.biomedclip import BiomedCLIPModel
from aims.rag.embed import EmbedIndexer
from aims.rag.pipeline import AIMSPipeline, ModeType


#experiment result total
@dataclass
class ExperimentResult:
    """단일 설정의 실험 결과.

    Attributes:
        mode:           실행 설정 이름
        accuracy:       전체 정확도
        rag_rate:       RAG 호출 비율 (always_rag / adaptive / expert)
        expert_rate:    expert 이관 비율 (expert mode만)
        n_total:        전체 샘플 수
        n_correct:      정답 수
        n_rag:          RAG 호출 수
        n_expert:       expert 이관 수
    
    """
    mode: ModeType
    n_total: int = 0
    n_correct: int = 0
    n_rag: int = 0
    n_expert: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total > 0 else 0.0

    @property
    def rag_rate(self) -> float:
        return self.n_rag / self.n_total if self.n_total > 0 else 0.0

    @property
    def expert_rate(self) -> float:
        return self.n_expert / self.n_total if self.n_total > 0 else 0.0


def run_experiment(
        pipeline: AIMSPipeline,
        dataloader: DataLoader,
        mode: ModeType,
        device: str,
) -> ExperimentResult:
    """단일 설정의 실험 실행.

    Args:
        pipeline: AIMSPipeline 인스턴스
        dataloader: 테스트 데이터 로더
        mode: 실행 설정 (always_rag / adaptive / expert)
        device: 실행 장치 (cpu / cuda)

    Returns:
        ExperimentResult: 실험 결과
    """
    result = ExperimentResult(mode=mode)
    for images, questions, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 배치 내 샘플별 처리 (pipeline은 단일 샘플 기준)
        for i in range(len(images)):
            image    = images[i].unsqueeze(0)   # (1, 3, H, W)
            question = questions[i]
            label    = labels[i].item()

            out = pipeline(image, question, mode=mode, label=label)

            result.n_total += 1

            # expert 이관은 정답으로 간주 (B방식)
            if out.route == "expert":
                result.n_expert += 1
                result.n_correct += 1   # 전문가는 항상 맞는다고 가정
            else:
                pred_label = 1 if out.prediction == "yes" else 0
                if pred_label == label:
                    result.n_correct += 1

            if out.route == "rag":
                result.n_rag += 1

    return result


def print_results(results: Dict[str, ExperimentResult]) -> None:
    """결과 테이블 출력."""
    print("\n" + "=" * 62)
    print(f"{'설정':<14} {'정확도':>8} {'RAG 호출률':>10} {'Expert 이관율':>13}")
    print("-" * 62)
    for mode, r in results.items():
        print(
            f"{mode:<14} "
            f"{r.accuracy:>7.1%}  "
            f"{r.rag_rate:>9.1%}  "
            f"{r.expert_rate:>12.1%}"
        )
    print("=" * 62)


def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    BIOMEDCLIP_CKPT = "data/checkpoints/biomedclip.pt"

    # W&B 초기화
    wandb.init(
        project="AIMS",
        name="pipeline-comparison-biomedclip",
        config={
            "model":      "BiomedCLIP",
            "dataset":    "VQA-RAD (yes/no)",
            "tau_low":    0.5,
            "tau_high":   0.8,
            "k":          3,
            "checkpoint": BIOMEDCLIP_CKPT,
        }
    )

    # 1. 데이터 로드
    train_data, test_data = load_vqarad(only_yes_no=True)
    test_dataset = VQARadDataset(test_data)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    print(f"test 샘플 수: {len(test_dataset)}")

    # 2. 인덱스 로드 (없으면 새로 구축)
    try:
        indexer = EmbedIndexer.load("data/faiss_index")
    except Exception:
        print("인덱스 없음 → 새로 구축")
        indexer = EmbedIndexer()
        indexer.build_from_hf(train_data)
        indexer.save("data/faiss_index")

    # 3. 파이프라인 생성 (학습된 BiomedCLIP 사용)
    model = BiomedCLIPModel(freeze_backbone=True)
    if os.path.exists(BIOMEDCLIP_CKPT):
        print(f"BiomedCLIP 체크포인트 로드: {BIOMEDCLIP_CKPT}")
        model.load_state_dict(torch.load(BIOMEDCLIP_CKPT, map_location="cpu"))
    else:
        print("경고: BiomedCLIP 체크포인트 없음 → 사전학습 가중치만 사용 (train_ViT.py 먼저 실행)")
    pipeline = AIMSPipeline(model, indexer, device=device)

    # 4. 4개 설정 실험
    modes   = ["baseline", "always_rag", "adaptive", "expert"]
    results = {}

    for mode in modes:
        print(f"\n[{mode}] 실험 중...")
        results[mode] = run_experiment(pipeline, test_loader, mode, device)
        r = results[mode]
        print(f"  정확도: {r.accuracy:.1%} | "
              f"RAG: {r.rag_rate:.1%} | "
              f"Expert: {r.expert_rate:.1%}")
        
        # W&B 로깅 - 설정별 결과

        wandb.log({
            f"{mode}/accuracy": r.accuracy,
            f"{mode}/rag_rate": r.rag_rate,
            f"{mode}/expert_rate": r.expert_rate,
            f"{mode}/n_total": r.n_total,
            f"{mode}/n_correct": r.n_correct,
        })


    # 5. 최종 결과 출력
    print_results(results)

    # W&B 요약 테이블

    table = wandb.Table(columns=["설정","정확도", "RAG 호출률", "Expert 이관율"])

    for mode, r in results.items():
        table.add_data(
            mode,
            r.accuracy,
            r.rag_rate,
            r.expert_rate
        )

    wandb.log({"results_table": table})
    wandb.finish()


    return results


if __name__ == "__main__":
    run_comparison()