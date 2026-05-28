"""Скачивание модели MediaPipe Hand Landmarker.

Запуск из корня проекта:
    python scripts/download_model.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Модель уже есть: {MODEL_PATH}")
        return
    print("Скачиваю модель MediaPipe...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Готово: {MODEL_PATH}")


if __name__ == "__main__":
    main()
