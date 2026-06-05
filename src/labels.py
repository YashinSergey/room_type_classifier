from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_CSV_PATHS = [
    ROOT_DIR / "data" / "raw" / "train_df.csv",
    ROOT_DIR / "data" / "raw" / "val_df.csv",
]
DEFAULT_CLASS_MAPPING_PATH = ROOT_DIR / "data" / "processed" / "class_mapping.json"
DEFAULT_ROOM_TYPE_LABELS = {
    0: "kitchen / dining room",
    1: "kitchen-living room",
    2: "multi-purpose room",
    3: "living room",
    4: "bedroom",
    5: "study",
    6: "children's room",
    7: "bathroom",
    8: "toilet",
    9: "combined bathroom",
    10: "hallway / entryway",
    11: "walk-in closet / pantry / laundry",
    12: "balcony / loggia",
    13: "view from window / balcony",
    14: "building exterior / yard",
    15: "entrance / stair landing",
    16: "other",
    17: "interior items / home appliances",
    18: "unfurnished room",
}


def load_label_mapping(
    csv_paths: list[Path | str] | None = None,
    class_mapping_path: Path | str | None = DEFAULT_CLASS_MAPPING_PATH,
) -> dict[int, str]:
    """result -> label."""
    csv_paths = csv_paths or DEFAULT_LABEL_CSV_PATHS

    old_to_new = None
    if class_mapping_path and Path(class_mapping_path).exists():
        with Path(class_mapping_path).open(encoding="utf-8") as file:
            class_mapping = json.load(file)
        old_to_new = {
            int(old_class): int(new_class)
            for old_class, new_class in class_mapping.get("old_to_new", {}).items()
        }

    frames = []
    for path in csv_paths:
        if not Path(path).exists():
            continue
        try:
            frames.append(pd.read_csv(path, usecols=["result", "label"]))
        except ValueError:
            continue
    if not frames:
        return DEFAULT_ROOM_TYPE_LABELS.copy()

    labels = pd.concat(frames, ignore_index=True)
    labels = labels.dropna(subset=["result", "label"])
    labels["result"] = labels["result"].astype(int)
    if old_to_new is not None:
        labels = labels[labels["result"].isin(old_to_new)].copy()
        labels["result"] = labels["result"].map(old_to_new).astype(int)
    result_ids = labels["result"].unique()
    return {
        int(result_id): DEFAULT_ROOM_TYPE_LABELS.get(int(result_id), f"class_{int(result_id)}")
        for result_id in result_ids
    }
