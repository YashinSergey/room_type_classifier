# Streamlit service

The service uploads one or more images, lets you choose models, and shows the predicted room type with probability for each selected model.

## Local run

```bash
just install-streamlit
just run-streamlit
```

## Docker

The Dockerfile is meant to be built from the repository root:

```bash
just docker-build-streamlit
just docker-run-streamlit
```

To add Streamlit to `docker-compose.yml`, use a section like this:

```yaml
services:
  streamlit:
    build:
      context: .
      dockerfile: streamlit/Dockerfile
    ports:
      - "8501:8501"
    environment:
      STREAMLIT_ALLOW_MODEL_DOWNLOAD: "1"
```

The Docker image installs `streamlit` and all model groups required for inference.

Project-level Streamlit configuration lives in `.streamlit/config.toml`: file watcher, run-on-save, email prompt, and usage stats are disabled. The port can be overridden with the `STREAMLIT_SERVER_PORT` environment variable.

## Models

The service shows only models that are actually available. If a checkpoint or weight file is missing, that model is disabled in the sidebar.

The comparison includes the final ensemble, YOLO, EfficientNet B0/B1, ResNet18, ResNet50, DenseNet121, ConvNeXt Nano, and ConvNeXt Tiny.

The `YOLO scene classifier` uses an external pretrained weight:

```text
models/yolo/downloads/keremberke/yolov8m-scene-classification/best.pt
```

To allow automatic YOLO download from Hugging Face at inference startup:

```bash
STREAMLIT_ALLOW_MODEL_DOWNLOAD=1 just run-streamlit
```

EfficientNet uses these default checkpoints:

```text
outputs/models/efficientnet/efficientnet_b0_best.pt
outputs/models/efficientnet/efficientnet_b1_best.pt
```

Paths can be overridden with `EFFICIENTNET_B0_CHECKPOINT_PATH` and
`EFFICIENTNET_B1_CHECKPOINT_PATH`.

ResNet uses these default checkpoints:

```text
outputs/models/resnet18/resnet18_best.pt
outputs/models/resnet50/resnet50_best.pt
```

The final ensemble uses these checkpoints:

```text
outputs/models/convnext_nano/convnext_nano_best.pt
outputs/models/resnet50/resnet50_best.pt
outputs/models/resnet18/resnet18_best.pt
```

ConvNeXt Nano uses this default checkpoint:

```text
outputs/models/convnext_nano/convnext_nano_best.pt
```

ConvNeXt Tiny uses this default checkpoint:

```text
outputs/models/convnext_tiny/convnext_tiny_best.pt
```

DenseNet121 uses this default checkpoint:

```text
outputs/models/densenet121/densenet121_best.pt
```
