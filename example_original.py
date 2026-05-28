# Загрузить модель через терминал
# curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Определение связей (каркаса кисти) ---
# Так как модуля solutions больше нет, мы задаем соединения точек вручную
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),  # Указательный палец
    (5, 9), (9, 10), (10, 11), (11, 12),  # Средний палец
    (9, 13), (13, 14), (14, 15), (15, 16),  # Безымянный палец
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Мизинец и основание
]

# --- 2. Инициализация Mediapipe Tasks API ---
# Укажи правильный путь к скачанному файлу, если он лежит в другой папке
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Создаем объект детектора
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. Подготовка кадра для MediaPipe ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Запускаем распознавание
    detection_result = detector.detect(mp_image)
    h, w, _ = frame.shape

    # --- 4. Отрисовка результатов ---
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:

            # Отрисовка суставов (белые линии)
            for pt1_idx, pt2_idx in HAND_CONNECTIONS:
                pt1 = hand_landmarks[pt1_idx]
                pt2 = hand_landmarks[pt2_idx]
                cv2.line(frame,
                         (int(pt1.x * w), int(pt1.y * h)),
                         (int(pt2.x * w), int(pt2.y * h)),
                         (255, 255, 255), 2)

            # Отрисовка точек (красные круги)
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, (0, 0, 255), -1)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()  # Не забываем очищать ресурсы детектора
cv2.destroyAllWindows()