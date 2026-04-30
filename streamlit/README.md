# Streamlit service

Сервис загружает одно или несколько изображений, позволяет выбрать модели и выводит предсказанный тип комнаты с вероятностью для каждой выбранной модели.

## Локальный запуск

```bash
uv sync --group streamlit
uv run streamlit run streamlit/app.py
```

## Docker

Dockerfile рассчитан на сборку из корня репозитория (в дальнейшем будет частью docker compose):

```bash
docker build -f streamlit/Dockerfile -t room-type-classifier-streamlit .
docker run --rm -p 8501:8501 room-type-classifier-streamlit
```

Пример секции для будущего `docker-compose.yml`:

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

Docker-образ ставит только dependency group `streamlit`, без лишних зависимостей.

Проектная конфигурация Streamlit лежит в `.streamlit/config.toml`: отключены file watcher, run-on-save, prompt email и usage stats. Порт можно переопределить через переменную окружения `STREAMLIT_SERVER_PORT`.

## Модели

Сервис показывает только реальные доступные модели. Если checkpoint или вес не найден, соответствующая модель будет отключена в сайдбаре.

Для `YOLO scene classifier` используется внешний pretrained вес:

```text
models/yolo/downloads/keremberke/yolov8m-scene-classification/best.pt
```

Чтобы разрешить автозагрузку YOLO из Hugging Face при старте inference:

```bash
STREAMLIT_ALLOW_MODEL_DOWNLOAD=1 uv run streamlit run streamlit/app.py
```

Для EfficientNet по умолчанию используются checkpoints:

```text
models/efficientNet/artifacts/efficientnet_b0_best.pt
models/efficientNet/artifacts/efficientnet_b1_best.pt
```

Пути можно переопределить через `EFFICIENTNET_B0_CHECKPOINT_PATH` и
`EFFICIENTNET_B1_CHECKPOINT_PATH`.
