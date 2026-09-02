"""tau_low / tau_high 임계값 튜닝.

validation set에서 adaptive 모드의 정확도를 최대화하는
tau_low, tau_high를 그리드 서치로 찾습니다.

사용:
    python -m aims.experiments.tune_tau
"""

import os
import torch
import wandb
import numpy as np
from itertools import product
from torch.utils.data import DataLoader, Subset

from aims.data.dataset import load_vqarad, VQARadDataset
from aims.experiments import wandb_init
from aims.models.biomedclip import BiomedCLIPModel
from aims.rag.embed import EmbedIndexer
from aims.rag.pipeline import AIMSPipeline


BIOMEDCLIP_CKPT = "data/checkpoints/biomedclip.pt"
VAL_RATIO       = 0.15
BATCH_SIZE      = 1  # pipeline은 단일 샘플 처리

# 탐색 범위: 0.1 단위 그리드
TAU_CANDIDATES = [round(v, 1) for v in np.arange(0.1, 1.0, 0.1)]


def build_val_loader(device: str):
    train_data, _ = load_vqarad(only_yes_no=True)
    dataset = VQARadDataset(train_data)

    n      = len(dataset)
    val_n  = int(n * VAL_RATIO)
    val_indices = list(range(n - val_n, n))
    val_set     = Subset(dataset, val_indices)

    return DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def evaluate_tau(
    pipeline: AIMSPipeline,
    val_loader: DataLoader,
    tau_low: float,
    tau_high: float,
    device: str,
) -> float:
    """주어진 tau로 adaptive 모드 정확도 측정."""
    pipeline.tau_low  = tau_low
    pipeline.tau_high = tau_high

    correct = 0
    total   = 0

    for images, questions, labels in val_loader:
        images = images.to(device)
        for i in range(len(images)):
            image    = images[i].unsqueeze(0)
            question = questions[i]
            label    = labels[i].item()

            out = pipeline(image, question, mode="adaptive", label=label)

            pred = 1 if out.prediction == "yes" else 0
            if out.route == "expert":
                correct += 1  # expert는 정답으로 가정
            elif pred == label:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def tune_tau():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. 데이터
    val_loader = build_val_loader(device)
    print(f"val 샘플 수: {len(val_loader.dataset)}")

    # 2. 인덱스
    train_data, _ = load_vqarad(only_yes_no=True)
    try:
        indexer = EmbedIndexer.load("data/faiss_index")
    except Exception:
        print("인덱스 없음 → 새로 구축")
        indexer = EmbedIndexer()
        indexer.build_from_hf(train_data)
        indexer.save("data/faiss_index")

    # 3. 모델 로드
    model = BiomedCLIPModel(freeze_backbone=True)
    if os.path.exists(BIOMEDCLIP_CKPT):
        model.load_state_dict(torch.load(BIOMEDCLIP_CKPT, map_location="cpu"))
        print(f"체크포인트 로드: {BIOMEDCLIP_CKPT}")
    else:
        print("경고: 체크포인트 없음 → 사전학습 가중치만 사용")

    pipeline = AIMSPipeline(model, indexer, device=device)

    # 4. W&B
    wandb_init(
        project="AIMS",
        name="tau-tuning",
        config={
            "model":          "BiomedCLIP",
            "tau_candidates": TAU_CANDIDATES,
            "val_ratio":      VAL_RATIO,
        }
    )

    # 5. 그리드 서치
    best_acc      = 0.0
    best_tau_low  = 0.5
    best_tau_high = 0.8

    print(f"\n탐색 조합 수: {len(TAU_CANDIDATES)**2}")

    for tau_low, tau_high in product(TAU_CANDIDATES, TAU_CANDIDATES):
        if tau_low >= tau_high:
            continue  # tau_low < tau_high 조건 필수

        acc = evaluate_tau(pipeline, val_loader, tau_low, tau_high, device)

        wandb.log({
            "tau_low":  tau_low,
            "tau_high": tau_high,
            "val_acc":  acc,
        })

        if acc > best_acc:
            best_acc      = acc
            best_tau_low  = tau_low
            best_tau_high = tau_high
            print(f"  새 최적: tau_low={tau_low:.1f} tau_high={tau_high:.1f} acc={acc:.1%}")

    print(f"\n최적 tau_low={best_tau_low:.1f}  tau_high={best_tau_high:.1f}  val_acc={best_acc:.1%}")

    wandb.summary["best_tau_low"]  = best_tau_low
    wandb.summary["best_tau_high"] = best_tau_high
    wandb.summary["best_val_acc"]  = best_acc
    wandb.finish()

    return best_tau_low, best_tau_high


if __name__ == "__main__":
    tune_tau()
