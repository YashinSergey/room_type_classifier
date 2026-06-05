set dotenv-load := true

set windows-shell := ["bash.exe", "-uc"]

PYTHON_VERSION := "3.12.8"
PYTORCH_PIP := "uv pip"

# Show the list of available commands
default:
    @just --list

# Install the required Python version with uv
setup:
    uv python install {{PYTHON_VERSION}}

# Recreate the local virtual environment
recreate-venv: setup
    uv venv --python {{PYTHON_VERSION}} --clear

# Base project setup
install: install-data

# Dependencies for preprocessing and shared dataloaders
install-data: setup
    uv sync --group data

# Dependencies for MLflow and DagsHub
install-tracking: setup
    uv sync --group tracking

# Log in to DagsHub for remote MLflow
dagshub-login:
    uv run --group tracking dagshub login

# Open the shared DagsHub experiments page
dagshub-experiments:
    @echo "https://dagshub.com/YashinSergey/room_type_classifier/experiments"

# Dependencies for Streamlit
install-streamlit: setup
    uv sync --group streamlit --group yolo --group efficientnet --group resnet18 --group resnet50 --group densenet121 --group convnext_nano --group convnext_tiny

# Dependencies for EfficientNet training
install-efficientnet: setup
    uv sync --group efficientnet

# Dependencies for ResNet18 training
install-resnet18: setup
    uv sync --group resnet18

# Dependencies for ResNet50 training
install-resnet50: setup
    uv sync --group resnet50

# Dependencies for DenseNet121 training
install-densenet121: setup
    uv sync --group densenet121

# EfficientNet plus Grad-CAM libraries
install-interpretability: setup
    uv sync --group efficientnet --group interpretability

# Dependencies for the YOLO script
install-yolo: setup
    uv sync --group yolo

# Dependencies for ConvNeXt Nano
install-convnext-nano: setup
    uv sync --group convnext_nano

# Legacy command name for ConvNeXt Nano
install-convnext_nano: install-convnext-nano

# Dependencies for ConvNeXt Tiny
install-convnext-tiny: setup
    uv sync --group convnext_tiny

# Install all dependency groups
install-all: setup
    uv sync --all-groups

# Prepare processed CSV files from raw data
prepare-data:
    uv run --group data python -m src.preprocess_data

# Prepare data with recommended heuristics
prepare-data-with-heuristics:
    uv run --group data python -m src.preprocess_data --include-heuristics recommended

# Prepare data with selected heuristics
prepare-data-heuristics HEURISTICS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}}

# Prepare data with a per-heuristic row limit
prepare-data-heuristics-limited HEURISTICS MAX_ROWS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}} --max-heuristics-per-source {{MAX_ROWS}}

# Update uv.lock after pyproject.toml changes
lock:
    uv lock

# Validate metric and checkpoint formats
check-training-outputs:
    uv run --group data python -m src.validate_training_outputs --allow-empty-checkpoints

# Build a model comparison table from MLflow
compare-models:
    uv run --group tracking python -m src.compare_mlflow_models

# Download the best MLflow checkpoints from DagsHub to outputs/models for Streamlit
pull-checkpoints *ARGS:
    uv run --group tracking python -m src.pull_mlflow_checkpoints {{ARGS}}

# Reinstall torch/torchvision from regular PyPI
pytorch-pypi:
    {{PYTORCH_PIP}} install --upgrade --reinstall torch torchvision

# Reinstall CPU torch/torchvision
pytorch-cpu:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cpu" torch torchvision

# Reinstall CUDA 13.0 torch/torchvision
pytorch-cu130:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cu130" torch torchvision

# Run YOLO demo/inference
run-yolo:
    uv run --group yolo --group tracking python -m models.yolo.main_yolo

# Train EfficientNet B0
train-efficientnet-b0 EPOCHS="80" BATCH="32" EARLY_STOPPING_PATIENCE="8" LR_SCHEDULER="plateau":
    uv run --group efficientnet --group tracking python -m models.efficientNet.train_efficientnet \
      --variant b0 --epochs {{EPOCHS}} --batch-size {{BATCH}} --early-stopping-patience {{EARLY_STOPPING_PATIENCE}} --lr-scheduler {{LR_SCHEDULER}}

# Train EfficientNet B1
train-efficientnet-b1 EPOCHS="80" BATCH="32" IMAGE_SIZE="240" EARLY_STOPPING_PATIENCE="8" LR_SCHEDULER="plateau":
    uv run --group efficientnet --group tracking python -m models.efficientNet.train_efficientnet \
      --variant b1 --epochs {{EPOCHS}} --batch-size {{BATCH}} --image-size {{IMAGE_SIZE}} --early-stopping-patience {{EARLY_STOPPING_PATIENCE}} --lr-scheduler {{LR_SCHEDULER}}

# Legacy short name for EfficientNet B0
train-efficientnet EPOCHS="30" BATCH="32":
    just train-efficientnet-b0 {{EPOCHS}} {{BATCH}}

# Train ResNet50
train-resnet50 EPOCHS="15" BATCH="32":
    uv run --group resnet50 --group tracking python -m models.resnet50.resnet50 \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Train ResNet18
train-resnet18 EPOCHS="30":
    uv run --group resnet18 --group tracking python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}}

# Train ResNet18 without weighted sampler
train-resnet18-best EPOCHS="30" SEED="42":
    uv run --group resnet18 --group tracking python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}} --seed {{SEED}} --no-weighted-sampling

# Train DenseNet121 in three stages
train-densenet121 STAGE1="2" STAGE2="8" STAGE3="5" BATCH="32":
    uv run --group densenet121 --group tracking python -m models.densenet121.train_densenet121 \
      --epochs-stage1 {{STAGE1}} --epochs-stage2 {{STAGE2}} --epochs-stage3 {{STAGE3}} --batch-size {{BATCH}}

# Train ConvNeXt Nano
train-convnext-nano EPOCHS="30" BATCH="32":
    uv run --group convnext_nano --group tracking python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Legacy short name for ConvNeXt Nano training
train-convnext EPOCHS="30" BATCH="32":
    uv run --group convnext_nano --group tracking python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Legacy underscored name for ConvNeXt Nano
train-convnext_nano EPOCHS="30" BATCH="32":
    uv run --group convnext_nano --group tracking python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Train ConvNeXt Tiny from JSON config
train-convnext-tiny CONFIG="models/convnext_tiny/train_config.json":
    uv run --group convnext_tiny --group tracking python -m models.convnext_tiny.train_convnext_tiny --config {{CONFIG}}

# Evaluate the compact ensemble on validation and log the experiment to DagsHub MLflow
eval-ensemble:
    uv run --group convnext_nano --group resnet50 --group resnet18 --group tracking python -m src.evaluate_ensemble \
      --weighting val_f1 --run-name ensemble_convnext_nano_resnet50_resnet18 --log-mlflow --no-mlflow-local

# Create a ConvNeXt Nano + ResNet50 + ResNet18 ensemble submission in data/submissions
make-submission:
    uv run --group convnext_nano --group resnet50 --group resnet18 python -m src.infer_ensemble

# Open local MLflow UI in fallback mode
mlflow-ui:
    uv run --group tracking mlflow ui --backend-store-uri sqlite:///mlflow.db

# Build Grad-CAM for EfficientNet
grad-cam-efficientnet:
    uv run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py --sample-index 0

# Run the Streamlit app
run-streamlit:
    uv run --group streamlit --group yolo --group efficientnet --group resnet18 --group resnet50 --group densenet121 --group convnext_nano --group convnext_tiny streamlit run streamlit/app.py

# Run an arbitrary command through uv
run *ARGS:
    @if [ "{{ARGS}}" = "yolo" ]; then \
        just run-yolo; \
    else \
        uv run {{ARGS}}; \
    fi

# Docker

# Build Docker images
docker-build:
    docker compose build

# Build the Streamlit Docker image
docker-build-streamlit:
    docker build -f streamlit/Dockerfile -t room-type-classifier-streamlit .

# Run Streamlit in Docker
docker-run-streamlit:
    docker run --rm -p 8501:8501 room-type-classifier-streamlit

# Check GPU availability inside Docker
docker-check-gpu:
    @if docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm base python -c "exec('import torch\nprint(\"torch:\", torch.__version__)\nprint(\"CUDA:\", torch.cuda.is_available())\nprint(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')" 2>/dev/null; then \
        true; \
    else \
        docker compose run --rm base python -c "exec('import torch\nprint(\"torch:\", torch.__version__)\nprint(\"CUDA:\", torch.cuda.is_available())\nprint(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')"; \
    fi

# Run the Docker service with GPU if CUDA is available inside the container
_docker-compose-run SERVICE *ARGS:
    @if docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm base python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then \
        echo "Docker CUDA is available, starting with GPU"; \
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm {{ARGS}} {{SERVICE}}; \
    else \
        echo "Docker CUDA is unavailable, starting on CPU"; \
        docker compose run --rm {{ARGS}} {{SERVICE}}; \
    fi

# Train DenseNet121 in Docker
docker-train-densenet121 STAGE1="2" STAGE2="8" STAGE3="5" BATCH="32":
    just _docker-compose-run train-densenet121 -e STAGE1={{STAGE1}} -e STAGE2={{STAGE2}} -e STAGE3={{STAGE3}} -e BATCH={{BATCH}}

# Train ResNet18 in Docker with the best parameters
docker-train-resnet18 EPOCHS="30" BATCH="32" SEED="42":
    just _docker-compose-run train-resnet18 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} -e SEED={{SEED}}

# Train ResNet50 in Docker
docker-train-resnet50 EPOCHS="15" BATCH="32":
    just _docker-compose-run train-resnet50 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Train EfficientNet in Docker
docker-train-efficientnet EPOCHS="30" BATCH="32":
    just _docker-compose-run train-efficientnet -e VARIANT=b0 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Train EfficientNet B0 in Docker
docker-train-efficientnet-b0 EPOCHS="30" BATCH="32":
    just _docker-compose-run train-efficientnet -e VARIANT=b0 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Train EfficientNet B1 in Docker
docker-train-efficientnet-b1 EPOCHS="30" BATCH="32":
    just _docker-compose-run train-efficientnet -e VARIANT=b1 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Train ConvNeXt Nano in Docker
docker-train-convnext EPOCHS="30" BATCH="32":
    just _docker-compose-run train-convnext -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Train ConvNeXt Nano in Docker
docker-train-convnext-nano EPOCHS="30" BATCH="32":
    just docker-train-convnext {{EPOCHS}} {{BATCH}}

# Train ConvNeXt Tiny in Docker
docker-train-convnext-tiny CONFIG="models/convnext_tiny/train_config.json":
    just _docker-compose-run train-convnext-tiny -e CONFIG={{CONFIG}}

# Run YOLO in Docker
docker-run-yolo:
    just _docker-compose-run yolo

# Run YOLO in Docker and log the inference run
docker-train-yolo:
    just docker-run-yolo

# Build Grad-CAM in Docker
docker-gradcam:
    docker compose run --rm gradcam
