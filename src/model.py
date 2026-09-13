from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert an image into a token sequence with a class token."""

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        patch_count = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.projection(images).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(images.size(0), -1, -1)
        tokens = torch.cat((class_token, patches), dim=1)
        return tokens + self.position


class TinyViT(nn.Module):
    """Small ViT classifier suitable for experiments and unit tests."""

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        num_classes: int = 2,
        embed_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = PatchEmbedding(image_size, patch_size, 1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.embedding(images))
        return self.classifier(encoded[:, 0])
