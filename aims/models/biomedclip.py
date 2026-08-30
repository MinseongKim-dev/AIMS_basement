"""BiomedCLIP 기반 VQA 모듈.

CLIP 계열 모델이라 image_features + text_features를 결합해
Linear classifier로 yes/no를 분류합니다.

TODO (second_inference 교체 시):
    - retrieved_docs를 question 컨텍스트로 추가
    - "Context: {doc.text} Question: {question} Answer: yes/no"
"""

import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer


class BiomedCLIPModel(nn.Module):
    """
    BiomedCLIP + Linear probe VQA 모델.

    동작 방식:
        1. 이미지 → image_features (512차원, L2 정규화)
        2. 질문   → text_features  (512차원, L2 정규화)
        3. concat → (1024차원) → Linear(1024, 2) → logits

    Args:
        freeze_backbone: True면 CLIP backbone 고정, head만 학습
    """

    MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    FEATURE_DIM = 512

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # BiomedCLIP 로드
        clip_model, _ = create_model_from_pretrained(self.MODEL_NAME)
        self.tokenizer = get_tokenizer(self.MODEL_NAME)

        self.image_encoder = clip_model.visual
        self.text_encoder  = clip_model.text

        # Linear probe head
        # image(512) + text(512) → concat(1024) → 2
        self.classifier = nn.Sequential(
            nn.Linear(self.FEATURE_DIM * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def forward(
        self,
        image:    torch.Tensor,
        question: list = None, #문자열 하나 또는 리스트
    ) -> torch.Tensor:
        """
        Args:
            image:    (B, 3, 224, 224)
            question: 질문 문자열

        Returns:
            (B, 2) logits
        """
        device = image.device
        B = image.shape[0]

        # 1. 이미지 인코딩
        image_features = self.image_encoder(image)              # (B, 512)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        # 2. 질문 텍스트 인코딩
        if question is None:
            prompt = ["Medical image."] * B
        else:
            if isinstance(question, str):
                prompt = [f"Medical image. {question}"] * B
            else:
                prompt = [f"Medical image. {q}" for q in question]

        text_tokens   = self.tokenizer(prompt).to(device)
        text_features = self.text_encoder(text_tokens)          # (B, 512)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        # 3. concat → classifier
        combined = torch.cat([image_features, text_features], dim=-1)  # (B, 1024)
        logits   = self.classifier(combined)                            # (B, 2)

        return logits

    def _freeze_backbone(self) -> None:
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    from aims.data.dataset import load_vqarad, VQARadDataset
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("BiomedCLIP 로드 중...")
    model = BiomedCLIPModel(freeze_backbone=True).to(device)
    model.eval()

    n_total     = model.total_params()
    n_trainable = model.trainable_params()
    print(f"전체 파라미터:      {n_total:,}")
    print(f"학습 가능 파라미터: {n_trainable:,}")

    _, test_data = load_vqarad(only_yes_no=True)
    dataset = VQARadDataset(test_data)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    image, question, label = next(iter(loader))
    image = image.to(device)

    with torch.no_grad():
        logits = model(image, question[0])

    pred = logits.argmax().item()
    print(f"\nQuestion: {question[0]}")
    print(f"Label:    {'yes' if label.item() == 1 else 'no'}")
    print(f"Logits:   {logits}")
    print(f"예측:     {'yes' if pred == 1 else 'no'}")