import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RoomTypeDataset(Dataset):
    def __init__(
            self,
            csv_path,
            image_root,
            transform=None,
            image_transform=None,
            image_col='image',
            target_col='result'
    ):
        # путь к папке с изображениями
        self.image_root = image_root
        # transforms, которые будут применяться к каждому изображению
        self.transform = transform
        # название колонки, где лежит путь или имя изображения
        self.image_col = image_col
        # название колонки, где лежит класс изображения
        self.target_col = target_col
        # Загружаем CSV-файл с описанием датасета
        self.df = pd.read_csv(csv_path)

    # количество строк в таблице(объектов в датасете)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Берём одну строку из таблицы по индексу
        row = self.df.iloc[idx]
        # Получаем путь или имя файла изображения из нужной колонки
        image_name = row[self.image_col]
        # Собираем полный путь к изображению
        image_path = os.path.join(self.image_root, image_name)
        # Открываем изображение.
        # convert("RGB") нужен, чтобы все изображения были в одном формате: 3 канала
        image = Image.open(image_path).convert("RGB")
        # Получаем числовой класс изображения
        target = int(row[self.target_col])
        # Если transforms переданы, применяем их к изображению
        if self.transform is not None:
            image = self.transform(image)
        # Возвращаем изображение и его класс
        return image, target