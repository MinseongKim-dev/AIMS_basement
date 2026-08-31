"""ECE (Expected Calibration Error) 캘리브레이션 평가.

test set 전체 logits를 수집하고 ECE를 계산합니다.
결과는 W&B에 기록됩니다.

사용:
    python -m aims.experiments.run_ece

채인영님 파트 연결 포인트:
    - logits_all, labels_all을 넘기면 ECE 재계산 가능
    - compute_ece() 함수를 직접 임포트해서 사용 가능:
        from aims.uncertainty.metrics import compute_ece
"""

import json
import os
import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aims.data.dataset import load_vqarad, VQARadDataset
from aims.models.biomedclip import BiomedCLIPModel
from aims.models.medcnn import SimpleMedCNN
from aims.models.ViT import VisualTransformerModel
from aims.uncertainty.metrics import compute_ece

CHECKPOINTS = {
    "SimpleMedCNN": ("data/checkpoints/simplecnn.pt",   False),
    "ViT-frozen":   ("data/checkpoints/vit_frozen.pt",  False),
    "ViT-full":     ("data/checkpoints/vit_full.pt",    False),
    "BiomedCLIP":   ("data/checkpoints/biomedclip.pt",  True),  # (ckpt, is_vlm)
}


def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    is_vlm: bool = False,
) -> tuple:
    """test set 전체 logits와 labels 수집."""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, questions, labels in loader:
            images = images.to(device)
            if is_vlm:
                logits = model(images, list(questions))
            else:
                logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def run_ece():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    _, test_data = load_vqarad(only_yes_no=True)
    test_dataset = VQARadDataset(test_data)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"test 샘플 수: {len(test_dataset)}")

    wandb.init(
        project="AIMS",
        name="ece-calibration",
        config={"dataset": "VQA-RAD (yes/no)", "n_bins": 10}
    )

    model_builders = {
        "SimpleMedCNN": lambda: SimpleMedCNN(),
        "ViT-frozen":   lambda: VisualTransformerModel(freeze=True),
        "ViT-full":     lambda: VisualTransformerModel(freeze=False),
        "BiomedCLIP":   lambda: BiomedCLIPModel(freeze_backbone=True),
    }

    results = {}

    print("\n" + "=" * 54)
    print(f"{'모델':<16} {'ECE':>8} {'정확도':>8} {'체크포인트':>10}")
    print("-" * 54)

    for name, (ckpt_path, is_vlm) in CHECKPOINTS.items():
        model = model_builders[name]()

        has_ckpt = os.path.exists(ckpt_path)
        if has_ckpt:
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        else:
            print(f"  [{name}] 체크포인트 없음 → 사전학습/랜덤 가중치")

        model = model.to(device)

        logits_all, labels_all = collect_logits(model, test_loader, device, is_vlm)

        ece = compute_ece(logits_all, labels_all, n_bins=10)

        probs = F.softmax(logits_all, dim=-1)
        preds = probs.argmax(dim=-1)
        acc   = preds.eq(labels_all).float().mean().item()

        results[name] = {"ece": ece, "accuracy": acc}

        ckpt_mark = "O" if has_ckpt else "X"
        print(f"{name:<16} {ece:>7.4f}  {acc:>7.1%}  {ckpt_mark:>10}")

        wandb.log({
            f"{name}/ece":      ece,
            f"{name}/accuracy": acc,
        })

    print("=" * 54)

    # W&B 테이블
    table = wandb.Table(columns=["모델", "ECE", "정확도"])
    for name, r in results.items():
        table.add_data(name, r["ece"], r["accuracy"])
    wandb.log({"ece_table": table})
    wandb.finish()

    os.makedirs("results", exist_ok=True)
    with open("results/ece_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("결과 저장: results/ece_results.json")

    return results


if __name__ == "__main__":
    run_ece()
