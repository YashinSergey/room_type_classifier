# Room Type Classifier

Image classification project for room-type recognition.
Case: Avito, predicting one of 19 room or property photo categories from an image.
Primary quality metric: Macro F1.
Best single model: ConvNeXt Nano.
Final submission solution: an ensemble of ConvNeXt Nano, ResNet50, and ResNet18.

## What is included

- preprocessing for train, validation, and test splits
- removal of an extra train-only class that is not part of the task or validation set
- class ids normalized to the 0-18 range
- several CNN models trained and compared
- experiments logged to MLflow through DagsHub
- final three-model ensemble assembled
- test submission generated
- Streamlit prototype for checking images interactively

## Data

Raw data is expected in this layout:

```text
data/raw/
  train_df.csv
  val_df.csv
  test_df.csv
  train_images/
  val_images/
  test_images/
```

Preprocessing:

```bash
just prepare-data-with-heuristics
```

Processed files are saved to:

```text
data/processed/
```

## Setup

The project expects Python 3.12, `uv`, and `just`.

```bash
pip install uv
uv tool install rust-just
```

Install dependencies:

```bash
just install-all
```

Install Streamlit dependencies:

```bash
just install-streamlit
```

## Training

Example commands:

```bash
just train-resnet18 30
just train-resnet50 30 32
just train-efficientnet-b0 30 32
just train-efficientnet-b1 30 32
just train-densenet121 2 8 5 32
just train-convnext-nano 25 32
just train-convnext-tiny
```

Compare models through MLflow:

```bash
just compare-models
```

Evaluate the final ensemble:

```bash
just eval-ensemble
```

## Submission

Generate the submission:

```bash
just make-submission
```

The file is saved locally:

```text
data/submissions/submission_ensemble.csv
```

## Streamlit

Run the Streamlit interface:

```bash
just run-streamlit
```

The interface lets you upload an image and compare predictions from selected models. It includes the final ensemble as well as individual models.

## Experiments

Experiments are logged to MLflow through DagsHub.

Link:

```text
https://dagshub.com/YashinSergey/room_type_classifier/experiments
```

Main metrics:

- best_macro_f1
- best_accuracy
- best_train_loss
- best_val_loss
- best_epoch

## Project structure

```text
src/
  dataset.py
  dataloaders.py
  preprocess_data.py
  transforms.py
  infer_ensemble.py
  evaluate_ensemble.py

models/
  convnext_nano/
  convnext_tiny/
  densenet121/
  efficientNet/
  resnet18/
  resnet50/
  yolo/

streamlit/
  app.py

reports/metrics/
outputs/models/
data/
```

Model checkpoints are saved to `outputs/models/`.

Metrics are saved to `reports/metrics/`.

Final report: https://docs.google.com/document/d/1LT4T90vRei1lcjus16heISa5xYsl4cUo/edit?usp=sharing&ouid=103046931125072858111&rtpof=true&sd=true

Presentation: https://docs.google.com/presentation/d/1Wa8_ZrLCPGx9kds4LiGtT-41lT_W1HoH/edit?usp=sharing&ouid=103046931125072858111&rtpof=true&sd=true
