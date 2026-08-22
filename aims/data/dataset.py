"""Dataset module scaffold."""
"""
yes/no 필터
이미지 변환
DataLoader 한 배치 shape 확인
완료 기준:
배치 shape (8, 3, 224, 224)
라벨 텐서 long 타입
Windows num_workers=0 유지
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
import torch


def get_transform(train=False):
    """Get the transformation for the dataset."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def _is_yes_no(answer):
    if answer is None:
        return False
    return str(answer).strip().lower() in {"yes", "no"}


def load_vqarad(only_yes_no=True):
    """Load the VQA-RAD dataset. train, test splits"""
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    train_data = dataset["train"]
    test_data = dataset["test"]

    if only_yes_no:
        train_data = train_data.filter(lambda s: _is_yes_no(s["answer"]))
        test_data = test_data.filter(lambda s: _is_yes_no(s["answer"]))

    return train_data, test_data

class VQARadDataset(Dataset):
    """Placeholder dataset class for VQA-RAD."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform if transform is not None else get_transform()
        self.answer_to_index = {"no": 0, "yes": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_obj = sample["image"]
        if isinstance(image_obj, Image.Image):
            image = image_obj.convert("RGB")
        else:
            image = Image.open(image_obj).convert("RGB")
        if self.transform:
            image = self.transform(image)

        question = sample["question"]
        answer_text = str(sample["answer"]).lower().strip()
        answer = torch.tensor(self.answer_to_index[answer_text], dtype=torch.long)

        return image, question, answer


if __name__ == "__main__":
    train_data, test_data = load_vqarad(only_yes_no=True)
    print(f"Train: {len(train_data)}개, Test: {len(test_data)}개")

    train_dataset = VQARadDataset(train_data)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    images, questions, answers = next(iter(loader))
    print(f"Image shape: {images.shape}")   # (8, 3, 224, 224)
    print(f"Answer dtype: {answers.dtype}") # torch.int64
    print(f"Sample Q: {questions[0]}")