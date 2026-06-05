# EfficientNet baseline

Minimal EfficientNet baseline on the shared project data pipeline.

## B0 or B1

EfficientNet-B1 is larger than B0: it has a higher input resolution (240) in the original
model family, plus more parameters and computation. In practice, B1 can give better
quality, but it trains and runs inference more slowly. Both variants are worth trying.

Input size and model type can be changed:

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet --variant b1 --image-size 240
```

## Run

```bash
just install-efficientnet
just prepare-data
just train-efficientnet
```

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet
```

By default, training reads `data/processed/train_df.csv` and
`data/processed/val_df.csv`. These files are created by `just prepare-data`:
class `18` is removed, old class `19` becomes the new class `18`, so
the model is trained on 19 classes.

Class imbalance is handled during training through class weights in

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet --class-balance none
```

DataLoader-level balancing can be enabled with `WeightedRandomSampler`:

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet --use-weighted-sampling --class-balance none
```

## Results

The script saves:

```text
outputs/models/efficientnet/efficientnet_b0_best.pt
reports/metrics/efficientnet/efficientnet_b0_metrics.json
reports/metrics/efficientnet/model_comparison.csv
```

Primary task metric: `best_macro_f1`.
At the end of training, per-class F1 is stored as `best_per_class_f1` inside the metrics JSON.

`model_comparison.csv` can be used as a simple comparison table for
runs and models by appending rows with their results.

## Grad-CAM

Build Grad-CAM for the first available validation example:

```bash
just install-interpretability
just grad-cam-efficientnet
```

For a specific image:

```bash
just run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py --checkpoint outputs/models/efficientnet/efficientnet_b1_best.pt --image data/raw/val_images/14333332896.jpg
```

Results are saved to:

```text
outputs/grad_cam/efficientnet/
```
