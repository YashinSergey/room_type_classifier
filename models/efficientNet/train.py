from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    efficientnet_b0,
    efficientnet_b1,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.metrics import calculate_macro_f1


MODEL_BUILDERS = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
}


def parse_args() -> argparse.Namespace:
    """Читаем параметры запуска из командной строки.

    Нужно, чтобы легко менять гиперпараметры (epochs, batch_size, пути к данным)
    без правок кода.
    """
    parser = argparse.ArgumentParser(description="Train EfficientNet baseline")
    parser.add_argument("--variant", choices=MODEL_BUILDERS.keys(), default="b0")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--train-csv", type=Path, default=ROOT_DIR / "data" / "raw" / "train_df.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "data" / "raw" / "val_df.csv")
    parser.add_argument("--train-images", type=Path, default=ROOT_DIR / "data" / "raw" / "train_images")
    parser.add_argument("--val-images", type=Path, default=ROOT_DIR / "data" / "raw" / "val_images")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "models" / "efficientNet" / "artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-col", default="result")
    parser.add_argument("--class-balance", choices=["loss", "none"], default="loss")
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Печатать прогресс обучения каждые N батчей (0 = выключено).",
    )
    return parser.parse_args()


def build_model(variant: str, num_classes: int) -> nn.Module:
    """Собираем EfficientNet и заменяем последний слой под число классов датасета."""
    builder, weights = MODEL_BUILDERS[variant]
    # Берём предобученные веса ImageNet (transfer learning).
    model = builder(weights=weights)
    # В EfficientNet голова-классификатор — это Linear слой в конце.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_class_weights(csv_path: Path, target_col: str, num_classes: int, device: torch.device) -> torch.Tensor:
    """Считаем веса классов для CrossEntropyLoss. Редкие классы получат больший вес."""
    targets = pd.read_csv(csv_path)[target_col].astype(int)
    max_target = int(targets.max())
    if max_target >= num_classes:
        raise ValueError(
            f"Found target={max_target}, but num_classes={num_classes}. "
            "Increase --num-classes or check class indexing."
        )

    counts = torch.bincount(torch.tensor(targets.to_list()), minlength=num_classes).float()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    existing_classes = counts > 0
    weights[existing_classes] = counts.sum() / (existing_classes.sum() * counts[existing_classes])
    return weights.to(device)


def validate_paths(args: argparse.Namespace) -> None:
    """Проверяем, что входные файлы/папки действительно существуют."""
    paths = {
        "--train-csv": args.train_csv,
        "--val-csv": args.val_csv,
        "--train-images": args.train_images,
        "--val-images": args.val_images,
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        joined_paths = "\n".join(missing)
        raise FileNotFoundError(f"Missing input paths:\n{joined_paths}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    log_every: int = 0,
) -> float:
    """Одна эпоха обучения: прямой проход: loss, backprop, шаг оптимизатора."""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        # Переносим батч на нужное устройство (CPU/CUDA/MPS).
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Усредняем loss по всем картинкам.
        total_loss += loss.item() * images.size(0)

        if log_every and batch_idx % log_every == 0:
            print(f"epoch={epoch} batch={batch_idx}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Оценка на валидации: считаем loss и macro-F1."""
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        # Самый вероятный класс.
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        # Сохраняем предсказания и истинные ответы для метрики.
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    # Для вычисления macro-F1 используем общую функцию из src/metrics.py
    macro_f1 = calculate_macro_f1(y_true, y_pred)
    return total_loss / len(loader.dataset), macro_f1


def save_comparison_row(metrics_path: Path, row: dict[str, object]) -> None:
    """Добавляем строку в общий CSV, чтобы потом сравнить разные запуски/модели."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metrics_path.exists()
    fieldnames = ["model", "variant", "num_classes", "best_epoch", "best_macro_f1", "checkpoint"]

    with metrics_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    # Проверим, что входные файлы/папки существуют.
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Автовыбор устройства: CUDA -> MPS -> CPU
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Готовим общий даталоадер (читает CSV и берёт картинки из папок).
    train_loader, val_loader = create_dataloaders(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        train_image_root=args.train_images,
        val_image_root=args.val_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Собираем модель и переносим её на нужное устройство.
    model = build_model(args.variant, args.num_classes).to(device)
    class_weights = None
    if args.class_balance == "loss":
        # Веса классов учитываются прямо в функции потерь.
        class_weights = get_class_weights(args.train_csv, args.target_col, args.num_classes, device)
        print(f"class_weights={class_weights.cpu().tolist()}")

    # CrossEntropyLoss — стандарт для многоклассовой классификации.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # AdamW — популярный оптимизатор для нейросетей (Adam + weight decay).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Будем хранить лучший результат по macro-F1 и сохранять лучший чекпоинт.
    best_macro_f1 = -1.0
    best_epoch = 0
    checkpoint_path = args.output_dir / f"efficientnet_{args.variant}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        # Обучение и затем проверка на валидации.
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            log_every=args.log_every,
        )
        val_loss, macro_f1 = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "macro_f1": macro_f1,
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            # Если стало лучше — обновляем best и сохраняем веса модели.
            best_macro_f1 = macro_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": args.variant,
                    "num_classes": args.num_classes,
                    "image_size": args.image_size,
                    "macro_f1": best_macro_f1,
                    "epoch": best_epoch,
                },
                checkpoint_path,
            )

    # Сохраняем метрики и историю обучения в JSON (чтобы потом можно было анализировать).
    metrics = {
        "model": "efficientnet",
        "variant": args.variant,
        "num_classes": args.num_classes,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "class_balance": args.class_balance,
        },
    }
    metrics_path = args.output_dir / f"efficientnet_{args.variant}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Также добавляем короткую строку в общий CSV для сравнения экспериментов.
    comparison_path = args.output_dir / "model_comparison.csv"
    save_comparison_row(
        comparison_path,
        {
            "model": "efficientnet",
            "variant": args.variant,
            "num_classes": args.num_classes,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "checkpoint": checkpoint_path,
        },
    )

    print(f"best_macro_f1={best_macro_f1:.4f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
