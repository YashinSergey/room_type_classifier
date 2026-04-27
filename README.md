## Установка и запуск проекта

### 0. Клонировать репозиторий

```bash
git clone https://github.com/your_username/room_type_classifier.git
cd room_type_classifier
```

### Требования

- Python: **3.12.x** (фиксируется в `.python-version`)
- Менеджер окружений: **uv**
- Утилита команд: **just** (используется `justfile` в корне проекта)

### 1. Установить `just` (если не установлен)

**macOS (Homebrew):**

```bash
brew install just
```

**Windows (Scoop):**

```powershell
scoop install just
```

**Windows (Chocolatey):**

```powershell
choco install just
```

### 2. Установить `uv` (если не установлен)

**macOS / Linux:**

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Если после установки `uv` не находится в терминале, добавьте в `PATH` (один раз на сессию):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Установка зависимостей (только через `uv`)

Проект использует единый `pyproject.toml`. Зависимости разбиты на группы (могут пересекаться):

- `data` — датасеты, dataloader, transforms, метрики, `torch`/`torchvision`
- `streamlit` — UI-сервис без модельных зависимостей
- `yolo` — YOLO demo/inference зависимости

```bash
just install
```

### 3.1. PyTorch (`torch` / `torchvision`): опционально переустановить (CPU / CUDA / PyPI)

`torch` и `torchvision` **закреплены** в группе `data` в `pyproject.toml` / `uv.lock` — обычно достаточно `just install`.

Если вам нужен CPU-репозиторий PyTorch или конкретная CUDA-ветка, после `just install` можно переустановить `torch`/`torchvision` одной из команд ниже.

- **По умолчанию (PyPI):**

```bash
just pytorch-pypi
```

- **CPU wheels из репозитория PyTorch:**

```bash
just pytorch-cpu
```

- **CUDA 13.0 wheels из репозитория PyTorch:**

```bash
just pytorch-cu130
```

Если команда с CUDA-репозиторием не находит wheel для вашей связки **OS + Python + CUDA**, значит **для вас вариант `cu130` несовместим** — в этом случае ориентируйтесь на “pip install” комбинацию с [официального конфигуратора PyTorch](https://pytorch.org/get-started/locally/) и/или переключитесь на `just pytorch-pypi` / `just pytorch-cpu`.

Проверка:

```bash
just run "python -c \"import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())\""
```

### 4. Запуск примера (YOLO)

```bash
just run-yolo
```

---

## Работа с Jupyter (если не виден kernel)

Если после создания нового окружения (.venv) Jupyter / PyCharm не видит kernel проекта, выполните:

```bash
uv run python -m pip install ipykernel

uv run python -m ipykernel install --user \
  --name room_type_classifier \
  --display-name "room_type_classifier"
```

После этого выберите kernel:

room_type_classifier

---

## Структура проекта

```
data/
  raw/           # сырые данные
  processed/     # обработанные данные

notebooks/       # эксперименты и EDA

outputs/
  models/        # сохранённые модели

src/             # основной код проекта(Classes.py)
```

## Data pipeline

Для работы проекта в папке `data/` должны лежать:

```
data/
  raw/
    train_df.csv
    val_df.csv
    test_df.csv

    train_images/
    val_images/
    test_images/
```

`train_df.csv` и `val_df.csv` содержат разметку классов в признаке `result`

`test_df.csv` пока не трогаем

Изображения берутся не по URL из признака `image`, а из локальных папок `train_images/`, `val_images/`, `test_images/`

Связь между CSV и файлом изображения идёт через признак:

```
image_id_ext -> {image_id_ext}.jpg
```

---

## Dataset

Для чтения данных используется класс:

```python
RoomTypeDataset
```

Он делает следующее:

1. читает CSV-файл
2. берёт `image_id_ext`
3. формирует имя файла вида `{image_id_ext}.jpg`
4. открывает изображение из локальной папки
5. приводит изображение к RGB
6. берёт числовой класс из колонки `result`
7. применяет transforms
8. возвращает:

```python
image, target
```

где:

```
image  — tensor изображения
target — номер класса от 0 до 19
```

---

## Transforms

Для train и validation используются разные transforms

### Train transforms

```
Resize(224, 224)
RandomHorizontalFlip
RandomRotation(10)
ColorJitter(brightness=0.2, contrast=0.2)
ToTensor
Normalize(ImageNet mean/std)
```

### Validation transforms

```
Resize(224, 224)
ToTensor
Normalize(ImageNet mean/std)
```

---

## DataLoaders

Для создания загрузчиков используется функция:

```python
create_dataloaders(...)
```

Она создаёт:

```python
train_loader, val_loader
```

`train_loader` использует:

```
shuffle=True
```

`val_loader` использует:

```
shuffle=False
```

### Пример использования:

```python
train_loader, val_loader = create_dataloaders(
    train_csv_path="../data/train_df.csv",
    val_csv_path="../data/val_df.csv",
    train_image_root="../data/train_images",
    val_image_root="../data/val_images",
    batch_size=32,
    num_workers=2
)
```
---

```python
# В реальном обучении используется цикл по DataLoader:
# for images, targets in train_loader:
#     ...

# Здесь берём один batch для проверки
images, targets = next(iter(train_loader))

print(images.shape)
print(targets.shape)
```

Ожидаемый результат:

```
torch.Size([32, 3, 224, 224])
torch.Size([32])
```

---

## Контракт для моделей

Любая модель должна работать с:

```python
images, targets = batch
```

Формат:

```
images  — tensor [batch_size, 3, 224, 224]
targets — tensor [batch_size]
```

Модель должна возвращать:

```
outputs — tensor [batch_size, 20]
```

---

## Пример проверки на ResNet18

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

num_classes = 20
in_features = model.fc.in_features

model.fc = nn.Linear(in_features, num_classes)

outputs = model(images)

print(outputs.shape)
```

Ожидаемый результат:

```
torch.Size([32, 20])
```

---

## Metric

Используется метрика:

```
Macro F1
```

Функция:

```python
calculate_macro_f1(y_true, y_pred)
```

---

## Важно

- Не менять train/val split  
- Не использовать test_df.csv в обучении  
- Все модели используют общий data pipeline  
- Для YOLOv8 будет отдельная адаптация  
