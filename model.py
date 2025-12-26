"""
Visual Model - CNN Architecture
ResNet-based classifier for 8 anomaly types
"""
import torch
import torch.nn as nn
import torchvision.models as models
from visual_mapping import ANOMALY_CLASSES


class AnomalyClassifierCNN(nn.Module):
    """
    CNN-based anomaly classifier using pretrained ResNet
    """

    def __init__(self, num_classes=8, pretrained=True, backbone='resnet50'):
        """
        Args:
            num_classes: Number of anomaly classes (default 8)
            pretrained: Use pretrained ImageNet weights
            backbone: Backbone architecture ('resnet50', 'resnet34', 'efficientnet_b0')
        """
        super(AnomalyClassifierCNN, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer

        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            logits: Class logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_probabilities(self, x):
        """Get class probabilities (softmax)"""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs

    def predict(self, x):
        """Get predicted class indices"""
        probs = self.get_probabilities(x)
        predictions = torch.argmax(probs, dim=1)
        return predictions


def create_model(num_classes=8, pretrained=True, backbone='resnet50', device='cuda'):
    """
    Create and initialize anomaly classifier model

    Args:
        num_classes: Number of anomaly classes
        pretrained: Use pretrained weights
        backbone: Backbone architecture
        device: Device to place model on

    Returns:
        model: Initialized model on specified device
    """
    model = AnomalyClassifierCNN(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone=backbone
    )

    model = model.to(device)

    print("\n" + "=" * 70)
    print(f"  MODEL CREATED: {backbone}")
    print("=" * 70)
    print(f"  Backbone: {backbone}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Device: {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    # Test model creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n  Using device: {device}")
    print(f"  Target classes: {len(ANOMALY_CLASSES)}")
    print(f"  Classes: {ANOMALY_CLASSES}")

    model = create_model(
        num_classes=len(ANOMALY_CLASSES),
        pretrained=True,
        backbone='resnet50',
        device=device
    )

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"\n  Test forward pass:")
    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Output (logits): {output[0].detach().cpu().numpy()}")
