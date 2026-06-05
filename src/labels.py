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
RUSSIAN_TO_ENGLISH_LABELS = {
    "кухня / столовая": "kitchen / dining room",
    "кухня-гостиная": "kitchen-living room",
    "универсальная комната": "multi-purpose room",
    "гостиная": "living room",
    "спальня": "bedroom",
    "кабинет": "study",
    "детская": "children's room",
    "ванная комната": "bathroom",
    "туалет": "toilet",
    "совмещенный санузел": "combined bathroom",
    "коридор / прихожая": "hallway / entryway",
    "гардеробная / кладовая / постирочная": "walk-in closet / pantry / laundry",
    "балкон / лоджия": "balcony / loggia",
    "вид из окна / с балкона": "view from window / balcony",
    "дом снаружи / двор": "building exterior / yard",
    "подъезд / лестничная площадка": "entrance / stair landing",
    "другое": "other",
    "предметы интерьера / быт.техника": "interior items / home appliances",
    "комната без мебели": "unfurnished room",
}
DEFAULT_RUSSIAN_ROOM_TYPE_LABELS = dict(enumerate(RUSSIAN_TO_ENGLISH_LABELS))


def translate_label_to_english(label: str) -> str:
    """Translate a dataset label to the English display label"""
    return RUSSIAN_TO_ENGLISH_LABELS.get(str(label).strip(), str(label))


DEFAULT_ENGLISH_ROOM_TYPE_LABELS = {
    class_id: translate_label_to_english(label)
    for class_id, label in DEFAULT_RUSSIAN_ROOM_TYPE_LABELS.items()
}


def load_label_mapping(
    csv_paths: list[Path | str] | None = None,
    class_mapping_path: Path | str | None = DEFAULT_CLASS_MAPPING_PATH,
) -> dict[int, str]:
    """result -> raw dataset label"""
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
        return DEFAULT_RUSSIAN_ROOM_TYPE_LABELS.copy()

    labels = pd.concat(frames, ignore_index=True)
    labels = labels.dropna(subset=["result", "label"])
    labels["result"] = labels["result"].astype(int)
    if old_to_new is not None:
        labels = labels[labels["result"].isin(old_to_new)].copy()
        labels["result"] = labels["result"].map(old_to_new).astype(int)
    return labels.groupby("result")["label"].agg(lambda values: values.mode().iat[0]).to_dict()


def to_english_label_mapping(label_mapping: dict[int, str]) -> dict[int, str]:
    """Translate a result -> dataset label mapping for display/reporting"""
    return {
        int(class_id): translate_label_to_english(label)
        for class_id, label in label_mapping.items()
    }


def load_english_label_mapping(
    csv_paths: list[Path | str] | None = None,
    class_mapping_path: Path | str | None = DEFAULT_CLASS_MAPPING_PATH,
) -> dict[int, str]:
    """result -> English display label"""
    label_mapping = load_label_mapping(csv_paths=csv_paths, class_mapping_path=class_mapping_path)
    if not label_mapping:
        return DEFAULT_ENGLISH_ROOM_TYPE_LABELS.copy()
    return to_english_label_mapping(label_mapping)
