## Установка и запуск проекта

### 1. Клонировать репозиторий

```bash
git clone https://github.com/your_username/room_type_classifier.git
cd room_type_classifier
```

---

### 2. Создать виртуальное окружение

```bash
python3 -m venv .venv
```

---

### 3. Активировать виртуальное окружение

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

---

### 4. Обновить pip

```bash
pip install --upgrade pip
```

---

### 5. Установить зависимости проекта

```bash
pip install -r requirements.txt
```
> Все новые библиотеки должны добавляться в requirements.txt, чтобы у всех участников было одинаковое окружение
---

### 6. Установить PyTorch отдельно

```bash
pip install torch torchvision torchaudio
```

> PyTorch устанавливается отдельно, так как его сборка зависит от операционной системы и оборудования (CPU / GPU / MPS).

---

### 7. Проверка установки

```bash
python -c "import torch; print(torch.__version__)"
```

Если версия вывелась без ошибок — всё установлено корректно.

---

### 8. Деактивация окружения (при необходимости)

```bash
deactivate
```
