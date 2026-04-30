from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_label_mapping(csv_paths: list[Path | str]) -> dict[int, str]:
    """Загружаем лейблы классов из CSV-файлов.

    CSV-файлы должны содержать колонки:
    - result - числовой класс
    - label - строковое название класса
    """
    frames = [
        pd.read_csv(path, usecols=["result", "label"])
        for path in csv_paths
        if Path(path).exists()
    ]
    if not frames:
        return {}

    labels = pd.concat(frames, ignore_index=True)
    labels = labels.dropna(subset=["result", "label"])
    labels["result"] = labels["result"].astype(int)
    return labels.groupby("result")["label"].agg(lambda values: values.mode().iat[0]).to_dict()
