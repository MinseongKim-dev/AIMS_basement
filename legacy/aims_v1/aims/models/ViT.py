"""ViT fine-tuning 모듈.

HuggingFace transformers의 ViTForImageClassification을 사용합니다.
SimpleMedCNN과 동일한 인터페이스(forward: image → logits)를 유지합니다.

실험 설정:
    A) frozen:    backbone 고정, classifier head만 학습
    B) full:      전체 fine-tuning

TODO (BiomedCLIP 교체 시):
    - VisionTransformerModel → BiomedCLIPModel로 교체
    - processor를 BiomedCLIP processor로 교체
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

class VisualTransformerModel(nn.Module):
    """
    ViT fine-tuning 래퍼 클래스.

    SimpleMedCNN과 동일한 인터페이스를 유지합니다.
    → AIMSPipeline 교체 시 model만 바꾸면 됩니다.

    Args:
        model_name:  HuggingFace 모델 이름
        num_classes: 분류 클래스 수 (yes/no → 2)
        freeze:      True면 backbone 고정 (frozen 설정)

    Usage:
        # frozen 설정
        model = VisionTransformerModel(freeze=True)

        # full fine-tuning 설정
        model = VisionTransformerModel(freeze=False)

        # SimpleMedCNN과 동일한 방식으로 사용
        logits = model(image)   # (B, 2)
    """
    MODEL_NAME="google/vit-base-patch16-224"

    def __init__(
            self,
            num_classes: int = 2,
            freeze: bool = False,
    ):
        super().__init__()

        self.vit = ViTForImageClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True, #classifier head만 학습할 때 필요
        )

        if freeze:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 이미지 텐서

        Returns:
            logits: (B, num_classes) 분류 logits
        """
        return self.vit(x).logits        

    def freeze_backbone(self):
        """backbone(ViT) 고정"""
        for param in self.vit.vit.parameters():
            param.requires_grad = False

        for param in self.vit.classifier.parameters():
            param.requires_grad = True

    def trainable_parameters(self):
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self):
        """전체 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters())
