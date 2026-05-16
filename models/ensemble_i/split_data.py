"""Стратифицированное разбиение train для обучения meta-learner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_for_meta(
    train_csv: Path,
    output_dir: Path,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
    label_col: str = "result",
) -> tuple[Path, Path, dict[str, int]]:
    """
    train → meta_new_train + meta_new_val (стратифицированно).
    Устарело для stacking: meta теперь учится на полном train, val — project val.
    """
    df = pd.read_csv(train_csv)
    if label_col not in df.columns:
        raise ValueError(f"В {train_csv} нет колонки {label_col}")

    new_train, new_val = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=df[label_col],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path_train = output_dir / "meta_new_train.csv"
    path_val = output_dir / "meta_new_val.csv"
    new_train.to_csv(path_train, index=False)
    new_val.to_csv(path_val, index=False)

    stats = {
        "train_total": len(df),
        "meta_new_train": len(new_train),
        "meta_new_val": len(new_val),
    }
    return path_train, path_val, stats
