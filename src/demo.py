from __future__ import annotations

import torch
from torch import nn

from .model import TinyViT


def make_synthetic_batch(batch_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a tiny reproducible batch; real projects should load inspection images."""
    images = torch.rand(batch_size, 1, 64, 64)
    labels = (images.mean(dim=(1, 2, 3)) > 0.5).long()
    return images, labels


def main() -> None:
    torch.manual_seed(7)
    model = TinyViT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    for step in range(5):
        images, labels = make_synthetic_batch()
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'step={step + 1} loss={loss.item():.4f}')

    print('logits shape:', tuple(model(make_synthetic_batch(4)[0]).shape))


if __name__ == '__main__':
    main()
