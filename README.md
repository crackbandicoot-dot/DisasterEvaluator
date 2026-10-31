# Satellite Imagery Disaster Damage Assessment

Machine Learning project for assessing building damage from pre- and post-disaster satellite image pairs. The goal is to classify damage levels (no damage, minor, major, destroyed), aggregate results by grid cell, and produce prioritized affected-area maps.

## Documentation

The state-of-the-art report for the first project phase can be found at:

[`docs/state_of_art.md`](docs/state_of_art.md)

## Planned Scope

- Dataset: xBD / xView2
- Baseline: building-crop damage classification with a pretrained CNN
- Main model: pre/post image segmentation (U-Net or similar)
- Metrics: per-class F1 and IoU
- Output: damage map by grid cell, GeoJSON/CSV summary, and a simple demo
