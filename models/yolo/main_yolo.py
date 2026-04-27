from huggingface_hub import hf_hub_download
from ultralyticsplus import YOLO, postprocess_classify_output
import os
import time
from pathlib import Path

IMAGES_PATH = Path('data/raw/val_images')
LOG_PATH = Path('data/log.txt')

N_IMAGES = 100

def main():
    
    start = time.perf_counter()

    model_path = Path('models/keremberke/yolov8m-scene-classification/best.pt')
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if model_path.exists():
        local_models = [x for x in os.listdir('./models') if not x.startswith('.')]
        if local_models:
            print('Найдены локальные модели:')
            for s in local_models:
                if '.' not in s:
                    print(f'\t{s}')
    else:
        model_path = hf_hub_download(
            repo_id="keremberke/yolov8m-scene-classification",
            filename="best.pt",
            local_dir="./models/keremberke/yolov8m-scene-classification",  # сохраняем в папку models/
            local_dir_use_symlinks=False,  # копируем, а не создаём симлинк
            token=hf_token,
        )
        print(f"Модель загружена и сохранена в: {model_path}")


    # load model
    model = YOLO(model_path)

    # set model parameters
    model.overrides['conf'] = 0.25  # model confidence threshold

    # set image
    images = os.listdir(IMAGES_PATH)[:N_IMAGES]

    load_time = time.perf_counter() - start
    print(f'Модель загружена за {load_time:.2f}с')

    with open(LOG_PATH, "w") as f:
        start = time.perf_counter()
        for im in images:
            # perform inference
            results = model.predict(IMAGES_PATH / im)

            processed_result = postprocess_classify_output(model, result=results[0])
            sorted_items = sorted(processed_result.items(), key=lambda x: x[1], reverse=True)[:2]

            s = f'\nФайл: {im}\n'
            for class_name, confidence in sorted_items:
                s += f"\t{class_name}: {confidence:.3f}\n"
            f.write(s)

        pred_time = time.perf_counter() - start

    print(f'\nПредсказание по {N_IMAGES} файлам выполнено за {pred_time:.2f}с')
    print(f'Среднее время на файл: {pred_time / N_IMAGES:.2f}с')

if __name__ == '__main__':
    main()