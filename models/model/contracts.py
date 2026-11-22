from __future__ import annotations

import torch

NUM_CLASSES = 5
IGNORE_INDEX = 255
VALID_MASK_VALUES = (0, 1, 2, 3, 4, IGNORE_INDEX)


def validate_images(images: torch.Tensor) -> int:
    """Validate the model input contract and return its dynamic spatial size N.

    Required shape: [B, 2, 3, N, N], where N is a power of two and N >= 32.
    The model itself receives images only; segmentation masks are training targets.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images debe ser torch.Tensor, no {type(images).__name__}.")
    if images.ndim != 5:
        raise ValueError(
            "images debe tener forma [B, 2, 3, N, N]; "
            f"se recibió {tuple(images.shape)}."
        )

    batch_size, pair_size, channels, height, width = images.shape
    if batch_size < 1:
        raise ValueError("Batch B size must be positive.")
    if pair_size != 2 or channels != 3:
        raise ValueError(
            "It is required two ordered images: before and after. "
            f"Obtained: [B, {pair_size}, {channels}, H, W]."
        )
    if height != width:
        raise ValueError(
            f"Images must be squared [N, N]; obtained {height}x{width}."
        )
    if height < 32 or (height & (height - 1)) != 0:
        raise ValueError(
            f"N must be a power of 2 and N >= 32; obtained N={height}."
        )
    if not images.is_floating_point():
        raise TypeError(
            "images must be float tensors."
        )

    return int(height)


def validate_masks(masks: torch.Tensor, batch_size: int, image_size: int) -> None:
    """Validate target masks: [B, N, N] with values 0..4 or 255."""
    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"masks must be torch.Tensor, not {type(masks).__name__}.")
    if masks.ndim != 3:
        raise ValueError(
            "masks must have the shape: [B, N, N]; "
            f"obtained {tuple(masks.shape)}."
        )
    expected = (batch_size, image_size, image_size)
    if tuple(masks.shape) != expected:
        raise ValueError(
            f"Mask must have the shape {expected}; obtained {tuple(masks.shape)}."
        )
    if masks.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("masks must have integer value.")

    allowed = torch.tensor(VALID_MASK_VALUES, device=masks.device, dtype=masks.dtype)
    invalid = ~torch.isin(masks, allowed)
    if torch.any(invalid).item():
        invalid_values = torch.unique(masks[invalid]).detach().cpu().tolist()
        raise ValueError(
            "Mask may only contain 0, 1, 2, 3, 4 y 255. "
            f"Invalid values found: {invalid_values}."
        )


def validate_training_batch(
    images: torch.Tensor, masks: torch.Tensor
) -> None:
    """Validate a complete (images, masks) batch returned by the DataLoader."""
    image_size = validate_images(images)
    validate_masks(masks, batch_size=images.shape[0], image_size=image_size)
