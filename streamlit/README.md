# Streamlit service

Сервис загружает одно или несколько изображений, позволяет выбрать модели и выводит
предсказанный тип комнаты с вероятностью для каждой выбранной модели.

## Локальный запуск

```bash
uv sync --group streamlit
uv run streamlit run streamlit/app.py
```

## Docker

Dockerfile рассчитан на build context из корня репозитория:

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
      STREAMLIT_ALLOW_MODEL_DOWNLOAD: "0"
```

Docker-образ ставит только dependency group `streamlit`, без `torch` и YOLO
зависимостей.

Проектная конфигурация Streamlit лежит в `.streamlit/config.toml`: отключены
file watcher, run-on-save, prompt email и usage stats. Порт можно переопределить
через переменную окружения `STREAMLIT_SERVER_PORT`.

## YOLO

Сейчас сервис работает на моках. Для `YOLO scene classifier` он пытается
использовать локальную модель:

```text
models/yolo/keremberke/yolov8m-scene-classification/best.pt
```

Если файла нет, эта модель тоже возвращает моковые ответы. Чтобы разрешить
автозагрузку из Hugging Face при старте inference:

```bash
STREAMLIT_ALLOW_MODEL_DOWNLOAD=1 uv run streamlit run streamlit/app.py
```

Если нужно запустить Streamlit с реальным YOLO inference:

```bash
uv sync --group streamlit --group yolo
STREAMLIT_ALLOW_MODEL_DOWNLOAD=1 uv run --group streamlit --group yolo streamlit run streamlit/app.py
```

Для приватного или rate-limited доступа можно дополнительно задать `HF_TOKEN`
или `HUGGINGFACE_HUB_TOKEN`.
