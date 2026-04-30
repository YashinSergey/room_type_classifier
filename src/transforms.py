from torchvision import transforms

def get_train_transforms(image_size=224):
    return transforms.Compose([
        # Приводим все изображения к одному размеру
        transforms.Resize((image_size, image_size)),
        # Случайно отражаем изображение по горизонтали
        transforms.RandomHorizontalFlip(),
        # Случайно поворачиваем изображение в пределах 10 градусов
        # Это помогает модели быть устойчивее к небольшим наклонам камеры
        transforms.RandomRotation(10),
        # Случайно меняем яркость и контраст(0.2 — это диапазон случайности)
        # Помогает модели лучше работать с разным освещением на фотографиях
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # Преобразуем изображение из PIL/numpy формата в PyTorch tensor
        transforms.ToTensor(),

        # Нормализуем изображение значениями mean/std от ImageNet
        # Это важно, потому что pretrained модели обычно обучались именно с такой нормализацией
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(image_size=224):
    return transforms.Compose([
        # Приводим validation изображения к тому же размеру, что и train
        transforms.Resize((image_size, image_size)),
        # Преобразуем изображение в PyTorch tensor
        transforms.ToTensor(),

        # Используем ту же нормализацию, что и для train
        # На validation нельзя применять случайные аугментации, но нормализация обязательна
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
