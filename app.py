# -*- coding: utf-8 -*-
"""Flask-приложение для демонстрации MediaPipe Hand Landmarker.

Важный момент: камера открывается НЕ на сервере через cv2.VideoCapture(0).
Камеру открывает браузер клиента через navigator.mediaDevices.getUserMedia(),
а Flask получает отдельные кадры по HTTP POST /process_frame.

Запуск:
    python app.py

После запуска откройте в браузере http://127.0.0.1:5000
Важно: камера в браузере работает только на localhost/127.0.0.1 или через HTTPS.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from detector import HandGestureDetector

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

app = Flask(__name__)
app.json.ensure_ascii = False  # JSON возвращает русские символы без \uXXXX
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

detector: HandGestureDetector | None = None
startup_error: str | None = None


def init_resources() -> None:
    """Инициализация модели. Камера здесь не открывается."""
    global detector, startup_error
    try:
        detector = HandGestureDetector(model_path=MODEL_PATH)
    except Exception as exc:
        startup_error = str(exc)


@app.route("/")
def index():
    return render_template("index.html", error=startup_error)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """Получает кадр от браузера клиента и возвращает обработанное изображение.

    Frontend отправляет файл `frame` в формате JPEG/PNG через FormData.
    Backend декодирует изображение, запускает MediaPipe, рисует скелет руки,
    кодирует результат обратно в JPEG и возвращает JSON.
    """
    if startup_error:
        return jsonify({"error": startup_error}), 500
    if detector is None:
        return jsonify({"error": "Detector is not initialized"}), 500
    if "frame" not in request.files:
        return jsonify({"error": "Файл frame не передан"}), 400

    uploaded = request.files["frame"].read()
    if not uploaded:
        return jsonify({"error": "Пустой кадр"}), 400

    encoded = np.frombuffer(uploaded, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Не удалось декодировать изображение"}), 400

    processed = detector.process_frame(frame)
    ok, buffer = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return jsonify({"error": "Не удалось закодировать результат"}), 500

    image_base64 = base64.b64encode(buffer).decode("ascii")
    return jsonify({
        "image": f"data:image/jpeg;base64,{image_base64}",
        "state": detector.state.__dict__,
    })


@app.route("/status")
def status():
    if detector is None:
        return jsonify({"error": startup_error or "Detector is not initialized"})
    return jsonify(detector.state.__dict__)


@app.route("/health")
def health():
    return jsonify({"ok": startup_error is None, "error": startup_error})


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    init_resources()

    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))

    print("\nПроект запущен.")
    print(f"Откройте: http://127.0.0.1:{port}")
    print("Для доступа к камере используйте localhost/127.0.0.1 или HTTPS.\n")

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        if detector is not None:
            detector.close()
