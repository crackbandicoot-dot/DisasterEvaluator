from __future__ import annotations

import torch
import torch.nn.functional as F

from models.model.contracts import validate_training_batch
from models.model.siamese_unet import SiameseModel


def test_forward_shapes() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # None avoids downloading ImageNet weights during an independent shape test.
    model = SiameseModel(encoder_weights=None).to(device)
    model.eval()

    for n in (64, 128):  # Exact test sizes specified in implementation_plan.es.md
        images = torch.randn(2, 2, 3, n, n, device=device)
        with torch.inference_mode():
            logits = model(images)

        expected = (2, 5, n, n)
        assert tuple(logits.shape) == expected, (
            f"N={n}: se esperaba {expected}, se obtuvo {tuple(logits.shape)}"
        )
        assert torch.isfinite(logits).all(), "Output contains NAN or infinite."
        print(f"OK: {tuple(images.shape)} -> {tuple(logits.shape)}")


def test_mask_contract_and_backward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseModel(encoder_weights=None).to(device)
    model.train()

    images = torch.randn(2, 2, 3, 64, 64, device=device)
    masks = torch.randint(0, 5, (2, 64, 64), device=device, dtype=torch.long)
    masks[:, :4, :4] = 255
    validate_training_batch(images, masks)

    logits = model(images)
    assert tuple(logits.shape) == (2, 5, 64, 64)

    # The training role owns the loss; this is only a contract/backward smoke test.
    loss = F.cross_entropy(logits, masks, ignore_index=255)
    loss.backward()

    encoder_has_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )
    decoder_has_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.decoder.parameters()
        if parameter.requires_grad
    )
    assert encoder_has_gradient, "No gradients detected in encoder."
    assert decoder_has_gradient, "No gradients detected in  decoder."
    print(f"OK: backward and ignore_index=255; loss={loss.item():.4f}")


def main() -> None:
    test_forward_shapes()
    test_mask_contract_and_backward()
    print("All tests passed.")


if __name__ == "__main__":
    main()
