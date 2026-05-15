# Room Type Classifier

Классификация типа комнаты по изображению. В репозитории лежат несколько baseline-моделей: EfficientNet, ResNet, DenseNet, ConvNeXt и небольшой YOLO inference-run.

## Установка

Нужны Python 3.12, `uv` и желательно `just`.

```bash
pip install uv
uv tool install rust-just
```

Базовые зависимости:

```bash
just install
```

Все группы сразу:

```bash
just install-all
```

Отдельные группы:

```bash
just install-efficientnet
just install-resnet18
just install-densenet121
just install-convnext-tiny
just install-streamlit
```

## Данные

Raw-данные ожидаются примерно так:

```text
data/raw/
  train_df.csv
  val_df.csv
  test_df.csv
  train_images/
  val_images/
  test_images/
```

Подготовка processed CSV:

```bash
just prepare-data
```

С эвристическими добавками:

```bash
just prepare-data-with-heuristics
just prepare-data-heuristics cabinet,dressing_room
```

После этого появляются файлы в `data/processed/`.

## Запуск

Обучение:

```bash
just train-resnet18 30
just train-efficientnet-b0 30 32
just train-densenet121 2 8 5 32
just train-convnext-tiny
```

YOLO demo:

```bash
just run-yolo
```

Streamlit:

```bash
just run-streamlit
```

MLflow локально:

```bash
RTC_MLFLOW_LOCAL=1 just mlflow-ui
```

Проверка сохранённых метрик и чекпоинтов:

```bash
just check-training-outputs
```

## Основные команды

```bash
just --list
just prepare-data
just compare-models
just grad-cam-efficientnet
just docker-build
just docker-run-streamlit
```

## Структура

```text
src/
  dataset.py
  dataloaders.py
  transforms.py
  metrics.py
  preprocess_data.py
  mlflow_utils.py

models/
  efficientNet/
  resnet18/
  resnet50/
  densenet121/
  convnext_nano/
  convnext_tiny/
  yolo/

streamlit/
  app.py

reports/metrics/
outputs/models/
data/raw/
data/processed/
```

Чекпоинты обычно сохраняются в `outputs/models/<model>/`, метрики - в `reports/metrics/<model>/`.
