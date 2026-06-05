FROM python:3.12-slim

# Install system dependencies for OpenCV, PIL, and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for Docker cache
COPY pyproject.toml uv.lock ./

# Install groups required by docker-compose services
RUN uv sync \
    --group data \
    --group tracking \
    --group densenet121 \
    --group efficientnet \
    --group resnet18 \
    --group resnet50 \
    --group interpretability \
    --group convnext_nano \
    --group convnext_tiny \
    --group yolo \
    --no-install-project \
    --frozen

# Add .venv/bin to PATH so Python and packages come from the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the remaining source code
COPY . .

# Default command can be overridden by docker run arguments
CMD ["python", "-m", "models.densenet121.train_densenet121"]
