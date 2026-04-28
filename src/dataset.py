import os
import warnings

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RoomTypeDataset(Dataset):
    def __init__(
            self,
            csv_path,
            image_root,
            transform=None,
            target_col='result',
            filter_missing=True
    ):
        # путь к папке с изображениями
        self.image_root = image_root
        # transforms, которые будут применяться к каждому изображению
        self.transform = transform
        # название колонки, где лежит класс изображения
        self.target_col = target_col
        # Загружаем CSV-файл с описанием датасета
        self.df = pd.read_csv(csv_path)
        # Фильтруем строки, где изображение не найдено
        if filter_missing:
            image_paths = self.df["image_id_ext"].astype(str).map(
                lambda image_id: os.path.join(self.image_root, f"{image_id}.jpg")
            )
            exists_mask = image_paths.map(os.path.exists)
            missing_count = int((~exists_mask).sum())
            if missing_count:
                warnings.warn(
                    f"Skipped {missing_count} rows from {csv_path}: images not found in {image_root}",
                    stacklevel=2,
                )
                self.df = self.df.loc[exists_mask].reset_index(drop=True)

    # количество строк в таблице(объектов в датасете)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Берём одну строку из таблицы по индексу
        row = self.df.iloc[idx]

        # Берём id изображения и формируем имя файла
        image_id = row["image_id_ext"]
        image_name = f"{image_id}.jpg"

        # Собираем полный путь к изображению
        image_path = os.path.join(self.image_root, image_name)

        # Открываем изображение
        image = Image.open(image_path).convert("RGB")

        # Получаем числовой класс изображения
        target = int(row[self.target_col])

        # Если transforms переданы, применяем их к изображению
        if self.transform is not None:
            image = self.transform(image)

        # Возвращаем изображение и его класс
        return image, target
