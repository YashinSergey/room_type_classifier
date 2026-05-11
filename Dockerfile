FROM python:3.12-slim

# Устанавливаем системные зависимости для OpenCV, PIL и ML-библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Копируем файлы зависимостей первыми — для кэширования слоёв
COPY pyproject.toml uv.lock ./

# data покрывает torch/torchvision/numpy/pandas/pillow/sklearn — общие для всех моделей
# interpretability добавляет grad-cam, matplotlib, opencv (только уникальные зависимости)
# yolo исключён — тяжёлый, монтируется отдельно при необходимости
RUN uv sync \
    --group data \
    --group interpretability \
    --no-install-project

# Добавляем .venv/bin в PATH — python и все пакеты берутся из виртуального окружения
ENV PATH="/app/.venv/bin:$PATH"

# Копируем остальной исходный код
COPY . .

# Команда по умолчанию — переопределяется аргументами docker run
CMD ["python", "-m", "models.densenet121.train_densenet121"]
