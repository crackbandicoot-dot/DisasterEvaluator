from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
except ImportError as exc:
    raise ImportError(
        "Missing dependency: segmentation-models-pytorch. Try: "
        "`pip install -r requirements.txt`."
    ) from exc

from models.model.contracts import NUM_CLASSES, validate_images


class SiameseModel(nn.Module):
    """Siamese U-Net with a shared SMP encoder and an SMP U-Net decoder.

    One ResNet34 encoder instance processes the before/after images by joining
    them along the batch dimension. At each encoder scale, its features are
    fused as [before, after, abs(after - before)]. The fused object is passed
    to the SMP U-Net decoder, whose skip connections combine matching scales.
    """

    ENCODER_DEPTH = 5
    DEFAULT_DECODER_CHANNELS = (256, 128, 64, 32, 16)

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        decoder_channels: Sequence[int] = DEFAULT_DECODER_CHANNELS,
    ) -> None:
        super().__init__()

        decoder_channels = tuple(int(ch) for ch in decoder_channels)
        if len(decoder_channels) != self.ENCODER_DEPTH:
            raise ValueError(
                f"decoder_channels must be {self.ENCODER_DEPTH} elements, one for each"
                f"decoder layer; received: {decoder_channels}."
            )
        if any(ch <= 0 for ch in decoder_channels):
            raise ValueError("Every decoder_channels value must be positive.")

        # A single encoder object is shared by both dates (shared weights).
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=self.ENCODER_DEPTH,
            weights=encoder_weights,
        )

        encoder_channels = tuple(int(ch) for ch in self.encoder.out_channels)
        if len(encoder_channels) != self.ENCODER_DEPTH + 1:
            raise RuntimeError(
                f"Encoder must return input map plus {self.ENCODER_DEPTH} features stages;"
                f" out_channels={encoder_channels}."
            )

        # Each learned feature level is tripled by temporal feature fusion.
        # UnetDecoder ignores encoder_channels[0]/features[0] as its first skip.
        fused_encoder_channels = [6] + [3 * ch for ch in encoder_channels[1:]]
        self.decoder = UnetDecoder(
            encoder_channels=fused_encoder_channels,
            decoder_channels=list(decoder_channels),
            n_blocks=self.ENCODER_DEPTH,
        )

        # Raw logits
        self.segmentation_head = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=NUM_CLASSES,
            kernel_size=3,
            padding=1,
        )
        self.num_classes = NUM_CLASSES
        self.encoder_name = encoder_name
        self.decoder_channels = decoder_channels


    @staticmethod
    def _fuse_features(
        before_features: torch.Tensor,
        after_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse one pair of feature maps along the channel dimension."""
        return torch.cat(
            (
                before_features,
                after_features,
                torch.abs(after_features - before_features),
            ),
            dim=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return five class logits for every input pixel.

        Args:
            images: Float tensor [B, 2, 3, N, N]. Index 0 is before; index 1
                is after. Images should already be normalized by the data pipeline.

        Returns:
            Float tensor [B, 5, N, N].
        """
        image_size = validate_images(images)
        batch_size = images.shape[0]

        before = images[:, 0]  # [B, 3, N, N]
        after = images[:, 1]   # [B, 3, N, N]

        # The batch dimension is doubled; the very same encoder weights process
        # both dates in one call, so the two branches are Siamese/shared-weight.
        both_dates = torch.cat((before, after), dim=0)  # [2B, 3, N, N]
        encoded_features = self.encoder(both_dates)

        if len(encoded_features) != self.ENCODER_DEPTH + 1:
            raise RuntimeError(
                "Unexpected maps number returned by the encoder: "
                f"{len(encoded_features)}. Expected {self.ENCODER_DEPTH + 1}."
            )

        fused_features = []
        for feature_map in encoded_features[1:]:
            before_map = feature_map[:batch_size]
            after_map = feature_map[batch_size:]
            fused_features.append(self._fuse_features(before_map, after_map))

        # SMP's U-Net decoder conventionally drops its first feature, then
        # decodes from the deepest scale while consuming the remaining skips.
        paired_rgb = torch.cat((before, after), dim=1)  # [B, 6, N, N]
        decoder_features = [paired_rgb, *fused_features]
        decoded = self.decoder(decoder_features)

        logits = self.segmentation_head(decoded)
        expected_shape = (batch_size, NUM_CLASSES, image_size, image_size)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Output contract not satisfied: "
                f"Expected {expected_shape}, obtained {tuple(logits.shape)}."
            )
        return logits
