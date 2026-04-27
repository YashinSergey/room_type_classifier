set dotenv-load := false

PYTHON_VERSION := "3.12.8"
PYTORCH_PIP := "uv pip"

default:
    @just --list

# Install uv-managed Python + create venv + sync deps
install:
    uv python install {{PYTHON_VERSION}}
    uv venv --python {{PYTHON_VERSION}}
    uv sync

# Update lockfile (optional)
lock:
    uv lock

# PyTorch: optional install/override of torch wheels (choose ONE variant)
# Baseline is pinned in `pyproject.toml` + `uv.lock` (installed by `just install` / `uv sync`).
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
    uv run python models/yolo/main_yolo.py

# Run arbitrary command inside the uv env:
#   just run "python -V"
run *ARGS:
    uv run {{ARGS}}
