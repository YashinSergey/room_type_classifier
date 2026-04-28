# EfficientNet baseline

Минимальный baseline для обучения EfficientNet на общем data pipeline проекта.

## B0 или B1

EfficientNet-B1 больше, чем B0: у нее выше входное разрешение (240) в оригинальной
семье моделей, больше параметров и вычислений. На практике B1 может дать лучшее
качество, но обучается и инференсится медленнее. Попробуем оба варианта.

Размер входа и тип модели можно менять:

```bash
uv run --group efficientnet python models/efficientNet/train.py --variant b1 --image-size 240
```

## Запуск

```bash
just install-efficientnet
just train-efficientnet
```

```bash
uv run --group efficientnet python models/efficientNet/train.py --num-classes 19
```

Дисбаланс классов учитывается внутри обучения через веса классов в
`CrossEntropyLoss`. Split и состав train/val не меняются. Отключить это можно так:

```bash
uv run --group efficientnet python models/efficientNet/train.py --class-balance none
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
