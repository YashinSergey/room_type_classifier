import argparse
import json
import os

import pandas as pd
from PIL import Image, UnidentifiedImageError


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

RESULT_COLUMN = "result"
IMAGE_ID_COLUMN = "image_id_ext"
IMAGE_URL_COLUMN = "image"
ITEM_ID_COLUMN = "item_id"

REMOVED_CLASS = 18
OLD_TO_NEW_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    19: 18,
}

# Оставляем только признаки, выбранные после EDA
TRAIN_VAL_COLUMNS = [IMAGE_ID_COLUMN, IMAGE_URL_COLUMN, RESULT_COLUMN]
TEST_COLUMNS = [IMAGE_ID_COLUMN, IMAGE_URL_COLUMN, ITEM_ID_COLUMN]


def parse_args():
    """Читает аргументы командной строки для подготовки данных

    Returns:
        Разобранные аргументы командной строки
    """
    parser = argparse.ArgumentParser(description="Подготовить raw CSV-файлы для обучения модели")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Проверять только наличие файлов, не открывая изображения через PIL",
    )
    return parser.parse_args()


def is_valid_image(image_path):
    """Проверяет, что файл изображения открывается через PIL
    Args:
        image_path: Путь к файлу изображения
    Returns:
        True, если PIL успешно проверил изображение, иначе False
    """
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def save_class_mapping(processed_dir):
    """Сохраняет mapping классов после удаления старого класса 18

    Args:
        processed_dir: Папка для обработанных файлов

    Returns:
        Путь к сохранённому JSON-файлу
    """
    class_mapping = {
        "target_column": RESULT_COLUMN,
        "removed_old_classes": [REMOVED_CLASS],
        "old_to_new": {str(old_id): new_id for old_id, new_id in OLD_TO_NEW_CLASS.items()},
        "new_to_old": {str(new_id): old_id for old_id, new_id in OLD_TO_NEW_CLASS.items()},
    }
    class_mapping_path = os.path.join(processed_dir, "class_mapping.json")

    with open(class_mapping_path, "w", encoding="utf-8") as file:
        json.dump(class_mapping, file, indent=2, ensure_ascii=False)

    return class_mapping_path


def check_images(df, image_root, image_ext, verify_images):
    """Проверяет наличие и валидность локальных изображений

    Args:
        df: DataFrame с колонкой `image_id_ext`
        image_root: Папка с изображениями split
        image_ext: Расширение файла изображения
        verify_images: Нужно ли открывать изображения через PIL

    Returns:
        Тот же DataFrame с колонками `image_exists`, `image_is_valid` и `can_predict`
    """
    image_exists = []
    image_is_valid = []

    for image_id in df[IMAGE_ID_COLUMN]:
        if pd.isna(image_id):
            image_exists.append(False)
            image_is_valid.append(False)
            continue

        image_path = os.path.join(image_root, f"{image_id}{image_ext}")
        exists = os.path.exists(image_path)
        image_exists.append(exists)

        if not exists:
            image_is_valid.append(False)
        elif verify_images:
            image_is_valid.append(is_valid_image(image_path))
        else:
            image_is_valid.append(True)

    df["image_exists"] = image_exists
    df["image_is_valid"] = image_is_valid
    df["can_predict"] = df["image_exists"] & df["image_is_valid"]
    return df


def preprocess_train_val(split, raw_dir, processed_dir, image_ext, verify_images):
    """Очищает train или val и сохраняет CSV для обучения

    Args:
        split: Название split, `train` или `val`
        raw_dir: Папка с raw-данными
        processed_dir: Папка для обработанных CSV
        image_ext: Расширение файла изображения
        verify_images: Нужно ли открывать изображения через PIL

    Returns:
        Количество строк до и после обработки
    """
    csv_path = os.path.join(raw_dir, f"{split}_df.csv")
    image_root = os.path.join(raw_dir, f"{split}_images")
    df = pd.read_csv(csv_path)

    rows_before = len(df)

    df = df[TRAIN_VAL_COLUMNS].copy()

    df[IMAGE_ID_COLUMN] = df[IMAGE_ID_COLUMN].astype(str).str.strip()
    df[IMAGE_ID_COLUMN] = df[IMAGE_ID_COLUMN].replace({"": None, "nan": None})
    df = df.dropna(subset=[IMAGE_ID_COLUMN])

    df[RESULT_COLUMN] = pd.to_numeric(df[RESULT_COLUMN], errors="coerce")
    df = df.dropna(subset=[RESULT_COLUMN])
    df[RESULT_COLUMN] = df[RESULT_COLUMN].astype(int)

    df = df[df[RESULT_COLUMN] != REMOVED_CLASS]

    df = df[df[RESULT_COLUMN].isin(OLD_TO_NEW_CLASS)]

    df = check_images(df, image_root, image_ext, verify_images)
    df = df[df["can_predict"]].copy()
    df = df.drop(columns=["image_exists", "image_is_valid", "can_predict"])

    df = df.drop_duplicates()

    df = df.drop_duplicates(subset=[IMAGE_ID_COLUMN], keep="first")

    # После удаления класса 18 переносим старый класс 19 в новый id 18
    df[RESULT_COLUMN] = df[RESULT_COLUMN].replace(OLD_TO_NEW_CLASS)
    df = df.reset_index(drop=True)

    output_csv = os.path.join(processed_dir, f"{split}_df.csv")
    df.to_csv(output_csv, index=False)

    return rows_before, len(df)


def preprocess_test(raw_dir, processed_dir, image_ext, verify_images):
    """Готовит test без удаления строк

    Args:
        raw_dir: Папка с raw-данными
        processed_dir: Папка для обработанных CSV
        image_ext: Расширение файла изображения
        verify_images: Нужно ли открывать изображения через PIL

    Returns:
        Количество строк до обработки, после обработки и количество строк для предсказания
    """
    csv_path = os.path.join(raw_dir, "test_df.csv")
    image_root = os.path.join(raw_dir, "test_images")
    df = pd.read_csv(csv_path)

    rows_before = len(df)

    # В test не удаляем строки, чтобы не сломать порядок и количество объектов для проверки
    df = df[TEST_COLUMNS].copy()
    df[IMAGE_ID_COLUMN] = df[IMAGE_ID_COLUMN].astype(str).str.strip()
    df[IMAGE_ID_COLUMN] = df[IMAGE_ID_COLUMN].replace({"": None, "nan": None})
    df = check_images(df, image_root, image_ext, verify_images)
    df = df.reset_index(drop=True)

    output_csv = os.path.join(processed_dir, "test_df.csv")
    df.to_csv(output_csv, index=False)

    return rows_before, len(df), int(df["can_predict"].sum())


def main():
    """Запускает подготовку train, val и test split

    Returns:
        None
    """
    args = parse_args()
    os.makedirs(args.processed_dir, exist_ok=True)

    train_before, train_after = preprocess_train_val(
        "train",
        args.raw_dir,
        args.processed_dir,
        args.image_ext,
        not args.skip_image_verify,
    )
    val_before, val_after = preprocess_train_val(
        "val",
        args.raw_dir,
        args.processed_dir,
        args.image_ext,
        not args.skip_image_verify,
    )
    test_before, test_after, test_can_predict = preprocess_test(
        args.raw_dir,
        args.processed_dir,
        args.image_ext,
        not args.skip_image_verify,
    )

    class_mapping_path = save_class_mapping(args.processed_dir)

    print(f"Папка с обработанными данными: {args.processed_dir}")
    print(f"Mapping классов: {class_mapping_path}")
    print(f"train: строк до={train_before}, строк после={train_after}")
    print(f"val: строк до={val_before}, строк после={val_after}")
    print(f"test: строк до={test_before}, строк после={test_after}")
    print(f"test: строк для предсказания={test_can_predict} из {test_after}")


if __name__ == "__main__":
    main()
