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
            target_col="result",
            filter_missing=True,
            filter_can_predict=False
    ):
        # путь к папке с изображениями
        self.image_root = image_root
        # transforms, которые будут применяться к каждому изображению
        self.transform = transform
        # название колонки, где лежит класс изображения
        self.target_col = target_col
        # Загружаем CSV-файл с описанием датасета
        self.df = pd.read_csv(csv_path)
        # Если в CSV есть image_path, считаем его относительно общей raw-папки
        self.raw_root = os.path.dirname(image_root)
        if filter_can_predict and "can_predict" in self.df.columns:
            self.df = self.df[self.df["can_predict"]].reset_index(drop=True)
        # Фильтруем строки, где изображение не найдено
        if filter_missing:
            image_paths = self.df.apply(self.get_image_path, axis=1)
            exists_mask = image_paths.map(os.path.exists)
            missing_count = int((~exists_mask).sum())
            if missing_count:
                warnings.warn(
                    f"Пропущено строк без локального изображения: {missing_count}",
                    stacklevel=2,
                )
                self.df = self.df.loc[exists_mask].reset_index(drop=True)

    def get_image_path(self, row):
        """Собирает путь к изображению из image_path или по схеме image_id_ext.jpg"""
        if "image_path" in row and pd.notna(row["image_path"]):
            return os.path.join(self.raw_root, str(row["image_path"]))

        image_id = str(row["image_id_ext"])
        return os.path.join(self.image_root, f"{image_id}.jpg")

    # количество строк в таблице(объектов в датасете)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Берём одну строку из таблицы по индексу
        row = self.df.iloc[idx]

        # Берём id изображения
        image_id = str(row["image_id_ext"])

        # Собираем полный путь к изображению
        image_path = self.get_image_path(row)

        # Открываем изображение
        image = Image.open(image_path).convert("RGB")

        # Получаем числовой класс изображения
        if self.transform is not None:
            image = self.transform(image)

        if self.target_col is None:
            item_id = row["item_id"] if "item_id" in row else image_id
            return image, image_id, item_id

        target = int(row[self.target_col])

        # Возвращаем изображение и его класс
        return image, target
