from __future__ import annotations

from collections.abc import Sequence

import torch

try:
    from torchgeo.models import FCSiamConc
except ImportError as exc:
    raise ImportError(
        "Missing dependency: TorchGeo with FCSiamConc. Install it with: "
        "`pip install torchgeo`."
    ) from exc

from models.model.contracts import NUM_CLASSES, validate_images


class SiameseModel(FCSiamConc):
    """
    Input:
        images: float tensor [B, 2, 3, N, N], ordered as before/after.

    Output:
        logits: float tensor [B, NUM_CLASSES, N, N].

    The model returns raw logits.

    FCSiamConc performs temporal feature concatenation. Unlike the former
    custom model. It doesn't concatenate abs(after_features-before_features)
    """

    ENCODER_DEPTH = 5
    DEFAULT_DECODER_CHANNELS = (256, 128, 64, 32, 16)

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        decoder_channels: Sequence[int] = DEFAULT_DECODER_CHANNELS,
    ) -> None:
        decoder_channels = tuple(int(ch) for ch in decoder_channels)

        if len(decoder_channels) != self.ENCODER_DEPTH:
            raise ValueError(
                f"decoder_channels must contain {self.ENCODER_DEPTH} elements, "
                f"one per decoder layer; received: {decoder_channels}."
            )
        if any(ch <= 0 for ch in decoder_channels):
            raise ValueError("Every decoder_channels value must be positive.")

        # TorchGeo's FCSiamConc creates one shared encoder and applies it to each timestamp.
        super().__init__(
            encoder_name=encoder_name,
            encoder_depth=self.ENCODER_DEPTH,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,  # Return logits only.
        )

        self.num_classes = NUM_CLASSES
        self.encoder_name = encoder_name
        self.decoder_channels = decoder_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return five class logits for every input pixel.

        Args:
            images: Float tensor [B, 2, 3, N, N].
        Returns:
            Float tensor [B, 5, N, N].
        """
        image_size = validate_images(images)
        batch_size = images.shape[0]

        # FCSiamConc expects [B, time, channels, height, width] directly.
        logits = super().forward(images)

        expected_shape = (batch_size, NUM_CLASSES, image_size, image_size)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Output contract not satisfied: "
                f"expected {expected_shape}, obtained {tuple(logits.shape)}."
            )

        return logits
