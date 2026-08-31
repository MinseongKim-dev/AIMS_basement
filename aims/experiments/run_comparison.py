"""4가지 설정 비교 실험.

출력:
    설정별 정확도 / RAG 호출률 / expert 이관율 비교 테이블
    results/comparison_results.json 에 저장 (visualize.py 입력)
"""

import json
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


def _check_prediction_diversity(
    model: "BiomedCLIPModel",
    loader: DataLoader,
    device: str,
    n_samples: int = 30,
) -> float:
    """처음 n_samples개 샘플에 대해 예측 다양성(다른 클래스 예측 비율)을 반환.

    반환값이 0.0이면 모든 예측이 동일 클래스 → 학습 필요.
    """
    model.eval()
    preds = []
    count = 0
    with torch.no_grad():
        for images, questions, _ in loader:
            images = images.to(device)
            for i in range(len(images)):
                if count >= n_samples:
                    break
                img = images[i].unsqueeze(0)
                q   = questions[i]
                logits = model(img, [q])
                preds.append(logits.argmax(dim=-1).item())
                count += 1
            if count >= n_samples:
                break

    if not preds:
        return 0.0
    unique = len(set(preds))
    diversity = (unique - 1) / max(len(set([0, 1])) - 1, 1)  # 0.0 = 단일 클래스, 1.0 = 완전 분산
    return diversity


def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    BIOMEDCLIP_CKPT = "data/checkpoints/biomedclip.pt"

    # ── pre-flight: 체크포인트 확인 ──────────────────────────────────
    if not os.path.exists(BIOMEDCLIP_CKPT):
        print()
        print("=" * 62)
        print("  [오류] BiomedCLIP 체크포인트 없음!")
        print(f"  경로: {BIOMEDCLIP_CKPT}")
        print()
        print("  체크포인트가 없으면 linear head가 랜덤 초기화 상태이므로")
        print("  baseline = always_rag = adaptive ≈ 53% (다수 클래스)가 됩니다.")
        print()
        print("  먼저 학습을 실행하세요:")
        print("    python -m aims.models.train_ViT")
        print("=" * 62)
        print()
        print("  체크포인트 없이 계속하려면 Ctrl+C로 중단하세요.")
        print("  10초 후 사전학습 가중치만으로 계속 진행합니다...")
        import time
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n  중단됨.")
            return {}
        print()

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
            "has_checkpoint": os.path.exists(BIOMEDCLIP_CKPT),
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
        print("경고: 사전학습 가중치만 사용 중 (결과가 무의미할 수 있음)")

    # ── pre-flight: 예측 다양성 확인 ────────────────────────────────
    print("\n[진단] baseline 예측 다양성 확인 중 (30샘플)...")
    diversity = _check_prediction_diversity(model.to(device), test_loader, device)
    if diversity == 0.0:
        print("  [경고] 모델이 모든 샘플을 동일 클래스로 예측합니다!")
        print("  → 학습된 체크포인트 없이 실행 중일 가능성이 높습니다.")
        print("  → baseline = always_rag = adaptive 가 동일하게 나올 것입니다.")
        print("  → train_ViT.py 실행 후 다시 시도하세요.\n")
    else:
        print(f"  예측 다양성 OK (두 클래스 모두 예측함)\n")

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

    # JSON 저장 (visualize.py 입력용)
    os.makedirs("results", exist_ok=True)
    save_data = {
        mode: {
            "accuracy":    r.accuracy,
            "rag_rate":    r.rag_rate,
            "expert_rate": r.expert_rate,
            "n_total":     r.n_total,
            "n_correct":   r.n_correct,
            "n_rag":       r.n_rag,
            "n_expert":    r.n_expert,
        }
        for mode, r in results.items()
    }
    out_path = "results/comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    return results


if __name__ == "__main__":
    run_comparison()