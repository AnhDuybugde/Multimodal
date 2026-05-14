from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class AudioCNN(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).flatten(1)
        return self.proj(x)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class AudioVisualFusionNet(nn.Module):
    def __init__(self, num_classes: int = 4, embed_dim: int = 256, pretrained_image: bool = False) -> None:
        super().__init__()
        self.audio_encoder = AudioCNN(out_dim=embed_dim)
        self.image_encoder = ImageEncoder(out_dim=embed_dim, pretrained=pretrained_image)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, audio: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        audio_emb = self.audio_encoder(audio)
        image_emb = self.image_encoder(image)
        fused = torch.cat([audio_emb, image_emb], dim=1)
        return self.classifier(fused)
