"""Medical CNN module scaffold."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMedCNN(nn.Module):
    """Placeholder SimpleMedCNN model."""

    def __init__(self):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2), # yes or no classification
        )

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.classifier(x)
        return x

    

if __name__ == "__main__":
    from aims.uncertainty.metrics import compute_entropy, compute_confidence

    # Test the model with a dummy input
    model = SimpleMedCNN()
    dummy_input = torch.randn(8, 3, 224, 224)  # Batch of 8 images
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (8, 2)
    confidence = compute_confidence(output)
    entropy = compute_entropy(output)
    print("Confidence shape:", confidence.shape)  # Should be (8,)
    print("Entropy shape:", entropy.shape)  # Should be (8,)
    print("Confidence:", confidence)
    print("Entropy:", entropy)