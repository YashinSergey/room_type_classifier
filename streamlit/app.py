from __future__ import annotations

import hashlib
import io
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]
YOLO_MODEL_PATH = ROOT_DIR / "models" / "yolo" / "keremberke" / "yolov8m-scene-classification" / "best.pt"
YOLO_REPO_ID = "keremberke/yolov8m-scene-classification"
YOLO_FILENAME = "best.pt"

ROOM_TYPES = [
    "bathroom",
    "bedroom",
    "children_room",
    "corridor",
    "dining_room",
    "dressing_room",
    "hall",
    "kitchen",
    "living_room",
    "office",
    "pantry",
    "studio",
]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    title: str
    description: str
    predictor: Callable[[bytes], tuple[str, float]]


def _image_hash(image_bytes: bytes, salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8") + image_bytes).hexdigest()
    return int(digest[:16], 16)


def _mock_predict(image_bytes: bytes, model_key: str) -> tuple[str, float]:
    rng = random.Random(_image_hash(image_bytes, model_key))
    prediction = ROOM_TYPES[rng.randrange(len(ROOM_TYPES))]
    probability = rng.uniform(0.71, 0.98)
    return prediction, probability


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


@st.cache_resource(show_spinner="Загружаем YOLO модель...")
def load_yolo_model() -> object | None:
    if not YOLO_MODEL_PATH.exists():
        if os.getenv("STREAMLIT_ALLOW_MODEL_DOWNLOAD") != "1":
            return None

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        hf_hub_download(
            repo_id=YOLO_REPO_ID,
            filename=YOLO_FILENAME,
            local_dir=YOLO_MODEL_PATH.parent,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

    try:
        from ultralyticsplus import YOLO
    except ImportError:
        return None

    model = YOLO(YOLO_MODEL_PATH)
    model.overrides["conf"] = 0.25
    return model


@st.cache_data(show_spinner=False)
def _predict_yolo_cached(image_bytes: bytes) -> tuple[str, float] | None:
    model = load_yolo_model()
    if model is None:
        return None

    try:
        from ultralyticsplus import postprocess_classify_output
    except ImportError:
        return None

    image = load_rgb_image(image_bytes)
    results = model.predict(image)
    processed_result = postprocess_classify_output(model, result=results[0])
    prediction, probability = max(processed_result.items(), key=lambda item: item[1])
    return prediction, float(probability)


def yolo_predict(image_bytes: bytes) -> tuple[str, float]:
    prediction = _predict_yolo_cached(image_bytes)
    if prediction is not None:
        return prediction
    return _mock_predict(image_bytes, "yolo_scene_classifier_mock")


MODELS = [
    ModelConfig(
        key="mock_cnn",
        title="Mock CNN classifier",
        description="Детерминированный мок классической CNN-модели.",
        predictor=lambda image_bytes: _mock_predict(image_bytes, "mock_cnn"),
    ),
    ModelConfig(
        key="mock_vit",
        title="Mock ViT classifier",
        description="Детерминированный мок transformer-based модели.",
        predictor=lambda image_bytes: _mock_predict(image_bytes, "mock_vit"),
    ),
    ModelConfig(
        key="yolo_scene_classifier",
        title="YOLO scene classifier",
        description="Использует локальный best.pt, если он есть; иначе работает как мок.",
        predictor=yolo_predict,
    ),
]


def configure_page() -> None:
    st.set_page_config(
        page_title="Room Type Classifier",
        layout="wide",
    )


def render_sidebar() -> list[ModelConfig]:
    st.sidebar.header("Модели")
    selected_models = []
    for model in MODELS:
        if st.sidebar.checkbox(model.title, value=True, help=model.description):
            selected_models.append(model)

    st.sidebar.divider()
    if YOLO_MODEL_PATH.exists():
        st.sidebar.success("YOLO best.pt найден локально")
    else:
        st.sidebar.info(
            "YOLO best.pt не найден. Для автозагрузки задайте STREAMLIT_ALLOW_MODEL_DOWNLOAD=1."
        )

    return selected_models


def render_results(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile], selected_models: list[ModelConfig]) -> None:
    rows = []
    progress = st.progress(0, text="Подготавливаем изображения...")
    total_steps = len(uploaded_files) * len(selected_models)
    completed_steps = 0

    for image_index, uploaded_file in enumerate(uploaded_files, start=1):
        image_bytes = uploaded_file.getvalue()
        for model in selected_models:
            time.sleep(0.25)
            prediction, probability = model.predictor(image_bytes)
            rows.append(
                {
                    "Изображение": uploaded_file.name or f"image_{image_index}",
                    "Номер": image_index,
                    "Модель": model.title,
                    "Предсказание": prediction,
                    "Вероятность": round(probability, 3),
                }
            )
            completed_steps += 1
            progress.progress(
                completed_steps / total_steps,
                text=f"Распознаем: {uploaded_file.name or image_index}",
            )

    progress.empty()

    results = pd.DataFrame(rows)
    st.subheader("Результаты")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Вероятность": st.column_config.ProgressColumn(
                "Вероятность",
                min_value=0,
                max_value=1,
                format="%.3f",
            )
        },
    )

    with st.expander("Загруженные изображения", expanded=False):
        columns = st.columns(min(3, len(uploaded_files)))
        for index, uploaded_file in enumerate(uploaded_files):
            with columns[index % len(columns)]:
                st.image(uploaded_file.getvalue(), caption=uploaded_file.name, use_container_width=True)


def main() -> None:
    configure_page()

    st.title("Room Type Classifier")
    st.caption("Загрузка изображений и сравнение результатов выбранных моделей.")

    selected_models = render_sidebar()
    uploaded_files = st.file_uploader(
        "Изображения",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    recognize = st.button(
        "Распознать",
        type="primary",
        disabled=not uploaded_files or not selected_models,
        use_container_width=False,
    )

    if not uploaded_files:
        st.info("Загрузите одно или несколько изображений.")
        return

    if not selected_models:
        st.warning("Выберите хотя бы одну модель.")
        return

    if recognize:
        render_results(uploaded_files, selected_models)


if __name__ == "__main__":
    main()
