"""ViT fine-tuning 학습 루프.

실험 설정 3가지를 순서대로 학습하고 비교합니다.
    1. SimpleMedCNN (baseline)
    2. ViT frozen  (linear probe)
    3. ViT full    (full fine-tuning)

RTX 3050 4GB 기준:
    - batch_size: 8 (full fine-tuning), 16 (frozen)
    - mixed precision (fp16) 사용
    - gradient accumulation: steps=4 (effective batch=32)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from torch.utils.data import Subset

from aims.models.ViT import VisualTransformerModel
from aims.models.medcnn import SimpleMedCNN
from aims.data.dataset import VQARadDataset, load_vqarad
from aims.models.biomedclip import BiomedCLIPModel

#-------------------------------------------------------------------- #
# 학습 설정                                                             #
#-------------------------------------------------------------------- #

CONFIG = {
    "epochs" : 20,
    "lr" : 2e-4, # frozen 설정에서 classifier head만 학습할 때
    "lr_full" : 2e-5, # full fine-tuning 설정에서 backbone까지 학습할 때
    "batch_size" : 8,
    "accumulation_steps" : 4, # gradient accumulation steps
    "ViT_ratio" : 0.15, #train의 15%를 Validation Set으로 지정
    "warmup_epochs" : 3, # warmup epochs for learning rate scheduler
    "weight_decay" : 1e-2, # weight decay for optimizer
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
}

# ------------------------------------------------------------------ #
# Augmentation                                                         #
# ------------------------------------------------------------------ #

def get_transform(train: bool = False):
    """train=True면 augmentation 적용, False면 기본 전처리."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])



# ------------------------------------------------------------------ #
# 학습 / 평가 루프                                                     #
# ------------------------------------------------------------------ #

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: GradScaler,
    accumulation_steps: int,
) -> float:
    """한 epoch 학습 루프."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (images, questions, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type="cuda"):
            # BiomedCLIP은 이미지와 질문을 함께 입력받아 logits를 반환, 나머지는 무시
            if hasattr(model, 'text_encoder'):
                logits=model(images,list(questions))
            else:
                logits = model(images)

            loss = criterion(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()
        
        if (step+1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() #gradient accumulation 후 optimizer 초기화 

        running_loss += loss.item() * accumulation_steps

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    """모델 평가 루프."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, questions, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if hasattr(model, 'text_encoder'):
                logits = model(images, list(questions))
            else:
                logits = model(images)

            preds   = logits.argmax(dim=-1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)

    return correct / total

#-------------------------------------------------------------------- #
# 단일 모델 학습 및 평가 루프                                               #
#-------------------------------------------------------------------- #
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    config: dict,
    save_path: str = None,
) -> float:
    """단일 모델 학습 및 평가 루프."""
    """
    모델 학습 후 최종 val accuracy 반환.

    Args:
        model:        학습할 모델
        train_loader: 학습 데이터로더
        val_loader:   검증 데이터로더
        lr:           학습률
        config:       CONFIG dict

    Returns:
        best_val_accuracy
    """
    if save_path and os.path.exists(save_path):
        print(f"  저장된 가중치 로드: {save_path}")
        model.load_state_dict(torch.load(save_path, map_location="cpu"))
        val_acc = evaluate(model.to(config["device"]), val_loader, config["device"])
        print(f"  로드된 모델 val_acc: {val_acc:.1%}")
        return val_acc

    device    = config["device"]
    epochs    = config["epochs"]
    accum     = config["accumulation_steps"]
    warmup_epochs = config["warmup_epochs"]

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config["weight_decay"],
    )

    warmup=LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine=CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs,)
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup, cosine], 
        milestones=[warmup_epochs]
    )
    scaler    = GradScaler(device="cuda")

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, accum
        )
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), save_path)

        print(f"  epoch {epoch:2d}/{epochs} | "
              f"loss={train_loss:.4f} | "
              f"val_acc={val_acc:.1%} | "
              f"best={best_acc:.1%}")

        scheduler.step()

    return best_acc

# ------------------------------------------------------------------ #
# 비교 실험                                                            #
# ------------------------------------------------------------------ #

def run_comparison():
    """SimpleMedCNN vs ViT frozen vs ViT full 비교 실험."""
    device = CONFIG["device"]
    print(f"device: {device}\n")

    # 1. 데이터 준비
    train_data, test_data = load_vqarad(only_yes_no=True)

    train_dataset_aug=VQARadDataset(train_data, transform=get_transform(train=True))
    train_dataset_val=VQARadDataset(train_data, transform=get_transform(train=False))

    # train → train / val 분리
    indices = list(range(len(train_dataset_aug)))
    val_size   = int(len(indices) * CONFIG["ViT_ratio"])
    train_size = len(indices) - val_size

    train_indices=indices[:train_size]
    val_indices  = indices[train_size:]

    # 각자 다른 transform dataset에서 Subset으로 train/val 분리
    train_set = Subset(train_dataset_aug, train_indices)
    val_set   = Subset(train_dataset_val, val_indices)

    test_set = VQARadDataset(test_data, transform=get_transform(train=False))

    train_loader = DataLoader(
        train_set, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0
    )

    print(f"train: {train_size} | val: {val_size} | test: {len(test_set)}\n")

    os.makedirs("data/checkpoints", exist_ok=True)

    # 2. 3가지 설정 실험
    experiments = [
        ("SimpleMedCNN",  SimpleMedCNN(),                        CONFIG["lr"],     "data/checkpoints/simplecnn.pt"),
        ("ViT-frozen",    VisualTransformerModel(freeze=True),   CONFIG["lr"],     "data/checkpoints/vit_frozen.pt"),
        ("ViT-full",      VisualTransformerModel(freeze=False),  CONFIG["lr_full"],"data/checkpoints/vit_full.pt"),
        ("BiomedCLIP",    BiomedCLIPModel(freeze_backbone=True), CONFIG["lr"],     "data/checkpoints/biomedclip.pt"),
    ]

    results = {}
    for name, model, lr, ckpt in experiments:
        print(f"[{name}] 학습 시작")
        print(f"  학습 파라미터: {model.trainable_params() if hasattr(model, 'trainable_params') else '전체'}")

        best_val = train_model(model, train_loader, val_loader, lr, CONFIG, save_path=ckpt)
        test_acc = evaluate(model, test_loader, device)

        results[name] = {"val": best_val, "test": test_acc}
        print(f"  → test accuracy: {test_acc:.1%}\n")

    # 3. 결과 테이블
    print("=" * 50)
    print(f"{'모델':<16} {'Val Acc':>10} {'Test Acc':>10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<16} {r['val']:>9.1%}  {r['test']:>9.1%}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_comparison()