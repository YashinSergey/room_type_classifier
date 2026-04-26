from torch.utils.data import DataLoader

from src.dataset import RoomTypeDataset
from src.transforms import get_train_transforms, get_val_transforms

def create_dataloaders(
        train_csv_path,   # путь к train CSV (разметка)
        val_csv_path,     # путь к validation CSV
        train_image_root, # папка с изображениями (train)
        val_image_root,   # папка с изображениями (val)
        batch_size=32,    # размер батча (сколько изображений за один шаг обучения)
        num_workers=2     # число процессов для параллельной загрузки данных
):
    # train dataset
    # Для train используем аугментации
    train_dataset = RoomTypeDataset(
        csv_path=train_csv_path,
        image_root=train_image_root,
        transform=get_train_transforms()
    )

    # validation dataset
    # Для validation используем только resize + normalize (без случайных аугментаций)
    val_dataset = RoomTypeDataset(
        csv_path=val_csv_path,
        image_root=val_image_root,
        transform=get_val_transforms()
    )

    # DataLoader для train
    # shuffle=True нужен, чтобы модель не видела данные всегда в одном порядке
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # DataLoader для validation
    # shuffle=False, потому что на валидации порядок не важен и лучше держать его стабильным
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader