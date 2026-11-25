# Siamese U-Net Model Documentation

## 1. Overview

The current model is a Siamese U-Net designed for **building damage classification using bi-temporal satellite imagery** from the xBD dataset.

It processes pre-disaster and post-disaster images through a shared encoder, combines their feature representations, and uses a U-Net decoder to generate a pixel-level damage classification map.

The model predicts five classes:

| Class ID | Label |
|---:|---|
| 0 | Background |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |

The model returns raw logits for each class. It does not apply softmax or argmax internally.

## 2. Architecture

The implementation is based on TorchGeo's `FCSiamConc` architecture.

Its main components are:

1. Shared feature encoder.
2. Temporal feature concatenation.
3. U-Net decoder with skip connections.
4. Segmentation head.

### 2.1. Input

The model receives a five-dimensional tensor containing the pre-disaster and post-disaster images:

```python
images.shape = [B, 2, 3, N, N]
```

Where:

- `B`: Batch size.
- `2`: Number of temporal images (before and after).
- `3`: RGB channels.
- `N`: Image height and width.

Both images must have the same spatial dimensions.

### 2.2. Shared Encoder

The encoder extracts spatial features from both temporal images.

The current configuration uses:

- Architecture: ResNet34.
- Pretrained weights: ImageNet.
- Encoder depth: 5.
- Input channels: 3.

The encoder weights are shared between the two images. This means that the same encoder processes both the pre-disaster and post-disaster inputs.

Sharing the encoder ensures that both images are represented in the same feature space, making their temporal comparison more consistent.

### 2.3. Temporal Feature Fusion

After encoding both images, the model combines their feature maps at each encoder level.

The current `FCSiamConc` implementation uses concatenation:

```python
fused_features = torch.cat(
    [before_features, after_features],
    dim=1
)
```

This operation combines the feature representations from both dates along the channel dimension.

**Important:** The current implementation does not explicitly concatenate the absolute difference between the two feature maps. Its temporal fusion uses the before and after features directly.

### 2.4. U-Net Decoder

The model uses a U-Net decoder provided through the `FCSiamConc` implementation.

The decoder progressively upsamples the fused feature maps to recover spatial resolution and generate detailed segmentation predictions.

The current decoder configuration is:

```python
decoder_channels = (256, 128, 64, 32, 16)
```

The decoder depth is 5.

### 2.5. Skip Connections

The decoder uses skip connections to combine high-level semantic information with lower-level spatial information from the encoder.

At each decoder stage, the upsampled decoder feature map is combined with the corresponding encoder feature map.

These connections help preserve spatial details that may otherwise be lost during downsampling.

The skip connections are part of the U-Net decoder implemented by `FCSiamConc`; they are not separately implemented in the `SiameseModel` class.

### 2.6. Segmentation Head

The final segmentation head maps the decoder features to five output channels.

```python
classes = 5
activation = None
```

The output tensor has the following shape:

```python
logits.shape = [B, 5, N, N]
```

Each channel represents the predicted score for one of the five classes.

The model returns raw logits without applying softmax.

## 3. Model Configuration

The current implementation uses the following configuration:

| Parameter | Value |
|---|---|
| Base architecture | TorchGeo `FCSiamConc` |
| Encoder | ResNet34 |
| Encoder weights | ImageNet |
| Encoder depth | 5 |
| Input channels | 3 per temporal image |
| Number of classes | 5 |
| Decoder channels | `(256, 128, 64, 32, 16)` |
| Output activation | None |
| Decoder type | U-Net |

The model class is defined as:

```python
class SiameseModel(FCSiamConc):
    ...
```

It inherits the architecture and forward implementation from TorchGeo's `FCSiamConc`, while defining the project-specific configuration and input/output validation.

## 4. Input and Output Contracts

### 4.1. Image Input

The input must satisfy the following requirements:

- Must be a PyTorch tensor.
- Must have five dimensions.
- Must follow the shape `[B, 2, 3, N, N]`.
- Batch size must be at least 1.
- Images must be square.
- Spatial dimensions must be at least 32.
- Spatial dimensions must be powers of two.
- Tensor values must use a floating-point data type.

### 4.2. Model Output

The expected output shape is:

```python
[B, 5, N, N]
```

The output contains raw class logits for every pixel.

To obtain the predicted class map:

```python
predictions = logits.argmax(dim=1)
```

The resulting tensor has shape:

```python
[B, N, N]
```

### 4.3. Target Masks

The target masks are not passed directly to the model. They are used by the training and evaluation pipeline.

Their expected shape is:

```python
[B, N, N]
```

Valid mask values are:

```python
(0, 1, 2, 3, 4, 255)
```

The value `255` is reserved as the ignore index.

For the damage-classification task, the target mask should represent the post-disaster building damage categories, rather than a binary change-detection mask.

## 5. Training Configuration

The model produces raw logits, making it compatible with PyTorch's `CrossEntropyLoss`.

A typical loss configuration is:

```python
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=255
)
```

The training loop should follow this structure:

```python
logits = model(images)

loss = criterion(
    logits,
    masks
)
```

The target masks must contain integer class IDs and must not be converted to one-hot encoding when using this loss function.

Softmax should not be applied before `CrossEntropyLoss`, because the loss function performs the required normalization internally.

## 6. Image Preprocessing

When using ImageNet-pretrained encoder weights, the input images should use the corresponding normalization parameters.

First, convert RGB values from the `[0, 255]` range to `[0, 1]`.

Then apply:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

The same normalization must be applied to both temporal images.

Geometric data augmentation must also be synchronized between the pre-disaster image, post-disaster image, and target mask.

For segmentation masks, use nearest-neighbor interpolation to avoid introducing invalid class values.

## 7. Current Implementation Characteristics

The current model has the following characteristics:

- Shared ResNet34 encoder for both temporal images.
- ImageNet-pretrained encoder weights by default.
- Feature concatenation for temporal fusion.
- U-Net decoder with skip connections.
- Five-class pixel-level segmentation output.
- Raw logits without an output activation.
- Input shape and output shape validation.

The architecture is based on a change-detection model, but its output is configured for five-class segmentation.

The number of output classes alone does not determine the meaning of the predictions. The target masks and their label definitions determine the task being learned.

For this project, the training targets must preserve the intended building damage categories.

## 8. Current Limitations and Possible Improvements

The current implementation provides a baseline Siamese segmentation architecture. Several modifications can be explored in future experiments.

### 8.1. Alternative Temporal Fusion

The current implementation concatenates the before and after feature maps.

An alternative would be to explicitly include their absolute difference:

```python
fused_features = torch.cat(
    [
        before_features,
        after_features,
        torch.abs(after_features - before_features)
    ],
    dim=1
)
```

This would provide the decoder with explicit information about feature-level changes between the two dates.

This modification requires a custom architecture or an appropriate extension of the existing implementation. It is not part of the current `FCSiamConc` configuration.
