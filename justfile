set dotenv-load := true

PYTHON_VERSION := "3.12.8"
PYTORCH_PIP := "uv pip"

default:
    @just --list

# Install uv-managed Python + create venv
setup:
    uv python install {{PYTHON_VERSION}}
    uv venv --python {{PYTHON_VERSION}}

# Install default data pipeline deps
install: install-data

# Install data pipeline deps
install-data: setup
    uv sync --group data

# Install Streamlit service deps without torch/YOLO deps
install-streamlit: setup
    uv sync --only-group streamlit

# Install YOLO demo deps
install-yolo: setup
    uv sync --group yolo

# Install all project dependency groups
install-all: setup
    uv sync --all-groups

# Update lockfile (optional)
lock:
    uv lock

# PyTorch: optional install/override of torch wheels (choose ONE variant)
# Baseline is pinned in `pyproject.toml` + `uv.lock` (installed by `just install` / `uv sync --group data`).
#
# - pypi: PyPI wheels for your platform
# - cpu: https://download.pytorch.org/whl/cpu
# - cu130: https://download.pytorch.org/whl/cu130
pytorch-pypi:
    {{PYTORCH_PIP}} install --upgrade --reinstall torch torchvision

pytorch-cpu:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cpu" torch torchvision

pytorch-cu130:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cu130" torch torchvision

# Run YOLO demo script
run-yolo:
    uv run --group yolo python models/yolo/main_yolo.py

# Run Streamlit service with mock models
run-streamlit:
    uv run --only-group streamlit streamlit run streamlit/app.py

# Run Streamlit service with YOLO dependencies enabled
run-streamlit-yolo:
    uv run --group streamlit --group yolo streamlit run streamlit/app.py

# Run arbitrary command inside the uv env:
#   just run "python -V"
run *ARGS:
    uv run {{ARGS}}
