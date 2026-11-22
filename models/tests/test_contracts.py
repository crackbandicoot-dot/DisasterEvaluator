import pytest
import torch

from models.model.contracts import validate_images, validate_training_batch


def test_valid_contract() -> None:
    images = torch.randn(2, 2, 3, 64, 64)
    masks = torch.randint(0, 5, (2, 64, 64), dtype=torch.long)
    masks[:, 0, 0] = 255
    assert validate_images(images) == 64
    validate_training_batch(images, masks)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 64, 64),
        (2, 2, 1, 64, 64),
        (2, 2, 3, 64, 128),
        (2, 2, 3, 96, 96),
    ],
)
def test_invalid_image_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_images(torch.randn(*shape))


def test_invalid_mask_values() -> None:
    images = torch.randn(2, 2, 3, 64, 64)
    masks = torch.zeros(2, 64, 64, dtype=torch.long)
    masks[0, 0, 0] = 5
    with pytest.raises(ValueError, match="0, 1, 2, 3, 4 y 255"):
        validate_training_batch(images, masks)
