## Установка и запуск проекта

### 1. Клонировать репозиторий

```bash
git clone https://github.com/your_username/room_type_classifier.git
cd room_type_classifier
```

---

### 2. Установить uv (если не установлен)

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Добавить в PATH (один раз):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

### 3. Создать виртуальное окружение (в корневой директории проекта)

```bash
uv venv
```

---

### 4. Установить зависимости проекта

```bash
uv pip install -r requirements.txt
```

Все новые библиотеки должны добавляться в requirements.txt, чтобы у всех участников было одинаковое окружение

---

### 5. Установить PyTorch отдельно

```bash
uv pip install torch torchvision torchaudio
```

PyTorch устанавливается отдельно, так как его сборка зависит от операционной системы и оборудования (CPU / GPU / MPS).

---

### 6. Проверка установки

```bash
python -c "import torch; print(torch.__version__)"
```
---

### 7. Активация окружения (при необходимости)

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

### 8. Деактивация окружения (при необходимости)

```bash
deactivate
```

---

## Работа с Jupyter (если не виден kernel)

Если после создания нового окружения (.venv) Jupyter / PyCharm не видит kernel проекта, выполните:

```bash
uv pip install ipykernel

python -m ipykernel install --user \
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