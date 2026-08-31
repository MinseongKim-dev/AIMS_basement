"""MC Dropout vs Entropy 불확실성 비교 실험.

두 가지 불확실성 추정 방식이 라우팅 정확도에 미치는 영향을 비교합니다.
    - entropy:    단일 forward pass softmax entropy
    - mc_dropout: N번 dropout forward의 평균 확률 entropy

adaptive 모드에서 각 방식으로 라우팅한 결과를 W&B에 기록합니다.

사용:
    python -m aims.experiments.run_mcdropout
"""

import os
import torch
import wandb
from dataclasses import dataclass
from typing import Dict
from torch.utils.data import DataLoader

from aims.data.dataset import load_vqarad, VQARadDataset
from aims.models.biomedclip import BiomedCLIPModel
from aims.rag.embed import EmbedIndexer
from aims.rag.pipeline import AIMSPipeline
from aims.uncertainty.metrics import (
    compute_entropy,
    compute_confidence,
    mc_dropout_entropy,
)

BIOMEDCLIP_CKPT = "data/checkpoints/biomedclip.pt"
MC_SAMPLES      = 10
TAU_LOW         = 0.5   # tune_tau.py 결과로 교체 가능
TAU_HIGH        = 0.8


@dataclass
class MCResult:
    n_total:   int = 0
    n_correct: int = 0
    n_rag:     int = 0

    @property
    def accuracy(self):
        return self.n_correct / self.n_total if self.n_total else 0.0

    @property
    def rag_rate(self):
        return self.n_rag / self.n_total if self.n_total else 0.0


def run_mcdropout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. 데이터
    _, test_data = load_vqarad(only_yes_no=True)
    test_dataset = VQARadDataset(test_data)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"test 샘플 수: {len(test_dataset)}")

    # 2. 인덱스
    train_data, _ = load_vqarad(only_yes_no=True)
    try:
        indexer = EmbedIndexer.load("data/faiss_index")
    except Exception:
        print("인덱스 없음 → 새로 구축")
        indexer = EmbedIndexer()
        indexer.build_from_hf(train_data)
        indexer.save("data/faiss_index")

    # 3. 모델
    model = BiomedCLIPModel(freeze_backbone=True)
    if os.path.exists(BIOMEDCLIP_CKPT):
        model.load_state_dict(torch.load(BIOMEDCLIP_CKPT, map_location="cpu"))
        print(f"체크포인트 로드: {BIOMEDCLIP_CKPT}")
    else:
        print("경고: 체크포인트 없음 → 사전학습 가중치만 사용")

    pipeline = AIMSPipeline(model, indexer, device=device, tau_low=TAU_LOW, tau_high=TAU_HIGH)

    # 4. W&B
    wandb.init(
        project="AIMS",
        name="mcdropout-vs-entropy",
        config={
            "model":      "BiomedCLIP",
            "mc_samples": MC_SAMPLES,
            "tau_low":    TAU_LOW,
            "tau_high":   TAU_HIGH,
        }
    )

    # 5. 비교 실험
    results: Dict[str, MCResult] = {
        "entropy":    MCResult(),
        "mc_dropout": MCResult(),
    }

    model_device = next(model.parameters()).device

    for images, questions, labels in test_loader:
        images = images.to(device)

        for i in range(len(images)):
            image    = images[i].unsqueeze(0)
            question = questions[i]
            label    = labels[i].item()

            # --- entropy 기반 ---
            out_ent = pipeline(image, question, mode="adaptive", label=label)

            r = results["entropy"]
            r.n_total += 1
            if out_ent.route == "rag":
                r.n_rag += 1
            pred = 1 if out_ent.prediction == "yes" else 0
            if out_ent.route == "expert" or pred == label:
                r.n_correct += 1

            # --- MC Dropout 기반: uncertainty를 교체해서 라우팅 재결정 ---
            mc_unc = mc_dropout_entropy(
                model.to(model_device), image, n_samples=MC_SAMPLES, question=question
            )
            pipeline.tau_low  = TAU_LOW
            pipeline.tau_high = TAU_HIGH

            r2 = results["mc_dropout"]
            r2.n_total += 1

            if mc_unc > TAU_HIGH:
                # expert 이관
                r2.n_correct += 1
            elif mc_unc < TAU_LOW:
                # direct (entropy 기반 첫 추론 결과 재사용)
                pred2 = 1 if out_ent.prediction == "yes" else 0
                if pred2 == label:
                    r2.n_correct += 1
            else:
                # RAG
                out_rag = pipeline(image, question, mode="always_rag", label=label)
                r2.n_rag += 1
                pred2 = 1 if out_rag.prediction == "yes" else 0
                if pred2 == label:
                    r2.n_correct += 1

    # 6. 결과 출력 및 W&B 기록
    print("\n" + "=" * 52)
    print(f"{'방식':<14} {'정확도':>8} {'RAG 호출률':>10}")
    print("-" * 52)
    for name, r in results.items():
        print(f"{name:<14} {r.accuracy:>7.1%}  {r.rag_rate:>9.1%}")
    print("=" * 52)

    for name, r in results.items():
        wandb.log({
            f"{name}/accuracy": r.accuracy,
            f"{name}/rag_rate": r.rag_rate,
        })

    table = wandb.Table(columns=["방식", "정확도", "RAG 호출률"])
    for name, r in results.items():
        table.add_data(name, r.accuracy, r.rag_rate)
    wandb.log({"mcdropout_comparison": table})
    wandb.finish()

    return results


if __name__ == "__main__":
    run_mcdropout()
