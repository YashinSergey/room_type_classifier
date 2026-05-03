# EfficientNet baseline

Минимальный baseline для обучения EfficientNet на общем data pipeline проекта.

## B0 или B1

EfficientNet-B1 больше, чем B0: у нее выше входное разрешение (240) в оригинальной
семье моделей, больше параметров и вычислений. На практике B1 может дать лучшее
качество, но обучается и инференсится медленнее. Попробуем оба варианта.

Размер входа и тип модели можно менять:

```bash
uv run --group efficientnet python models/efficientNet/train_efficientnet.py --variant b1 --image-size 240
```

## Запуск

```bash
just install-efficientnet
just prepare-data
just train-efficientnet
```

```bash
uv run --group efficientnet python models/efficientNet/train_efficientnet.py
```

По умолчанию обучение читает `data/processed/train_df.csv` и
`data/processed/val_df.csv`. Эти файлы создаются командой `just prepare-data`:
класс `18` удаляется, старый класс `19` становится новым классом `18`, поэтому
модель обучается на 19 классах.

Дисбаланс классов учитывается внутри обучения через веса классов в

```bash
uv run --group efficientnet python models/efficientNet/train_efficientnet.py --class-balance none
```

## Результаты

Скрипт сохраняет:

```text
models/efficientNet/artifacts/efficientnet_b0_best.pt
models/efficientNet/artifacts/efficientnet_b0_metrics.json
models/efficientNet/artifacts/model_comparison.csv
```

Основная метрика для ТЗ: `best_macro_f1`.

`model_comparison.csv` можно использовать как простую таблицу сравнения с
запусков и моделей, добавляя туда строки с их результатами.

## Grad-CAM

Построить Grad-CAM для первого доступного примера из validation:

```bash
just install-interpretability
just grad-cam-efficientnet
```

Для конкретного изображения:

```bash
uv run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py --checkpoint models/efficientNet/artifacts/efficientnet_b1_best.pt --image data/raw/val_images/14333332896.jpg
```

Результаты сохраняются в:

```text
models/efficientNet/artifacts/grad_cam/
```
