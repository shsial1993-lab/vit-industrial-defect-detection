import torch

from src.model import TinyViT


def test_vit_output_shape_and_gradients() -> None:
    model = TinyViT(image_size=32, patch_size=8)
    images = torch.randn(3, 1, 32, 32)
    logits = model(images)
    assert logits.shape == (3, 2)
    logits.sum().backward()
    assert model.embedding.class_token.grad is not None
