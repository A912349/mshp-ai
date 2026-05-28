# -*- coding: utf-8 -*-
"""Детекция рук, подсчёт пальцев и простая визуализация."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def get_cyrillic_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


@dataclass
class DetectionState:
    hands_count: int = 0
    fingers_count: int = 0
    gesture: str = "Рука не обнаружена"
    per_hand_counts: List[int] = field(default_factory=list)
    fps: float = 0.0
    frames: int = 0
    updated_at: float = field(default_factory=time.time)


class HandGestureDetector:
    """Обёртка над MediaPipe Hand Landmarker."""

    def __init__(
        self,
        model_path: str | Path = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Не найдена модель: {self.model_path}. "
                "Запустите: python scripts/download_model.py"
            )

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.state = DetectionState()
        self._last_time = time.time()

    def close(self) -> None:
        self.detector.close()

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        h, w, _ = frame_bgr.shape
        hands = result.hand_landmarks or []

        self.state.hands_count = len(hands)
        self.state.frames += 1
        self.state.updated_at = time.time()
        now = time.time()
        dt = now - self._last_time
        self.state.fps = 1.0 / dt if dt > 0 else 0.0
        self._last_time = now

        if hands:
            counts: List[int] = []
            for hand_landmarks in hands:
                fingers = self.count_fingers(hand_landmarks)
                counts.append(fingers)
                gesture = self.gesture_from_count(fingers)
                self._draw_hand(frame_bgr, hand_landmarks, w, h)
                self._draw_label(frame_bgr, hand_landmarks, w, h, fingers, gesture)

            total = sum(counts)
            self.state.per_hand_counts = counts
            self.state.fingers_count = total
            self.state.gesture = self.gesture_from_count(total) if len(counts) == 1 else f"{len(counts)} руки: {total} пальцев"
        else:
            self.state.fingers_count = 0
            self.state.per_hand_counts = []
            self.state.gesture = "Рука не обнаружена"

        self._draw_hud(frame_bgr)
        return frame_bgr

    @staticmethod
    def _point(hand_landmarks: Sequence, idx: int) -> np.ndarray:
        lm = hand_landmarks[idx]
        # x/y — положение на экране, z — глубина. Чем меньше z, тем ближе точка к камере.
        return np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=np.float32)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom <= 1e-8:
            return 0.0
        cos_value = float(np.dot(ba, bc) / denom)
        cos_value = max(-1.0, min(1.0, cos_value))
        return float(np.degrees(np.arccos(cos_value)))

    @classmethod
    def _finger_is_extended(
        cls,
        hand_landmarks: Sequence,
        palm_center: np.ndarray,
        wrist: np.ndarray,
        palm_size: float,
        mcp_idx: int,
        pip_idx: int,
        dip_idx: int,
        tip_idx: int,
    ) -> bool:
        mcp = cls._point(hand_landmarks, mcp_idx)
        pip = cls._point(hand_landmarks, pip_idx)
        dip = cls._point(hand_landmarks, dip_idx)
        tip = cls._point(hand_landmarks, tip_idx)

        chain_len = (
            cls._distance(mcp, pip)
            + cls._distance(pip, dip)
            + cls._distance(dip, tip)
        )
        if chain_len <= 1e-8:
            return False

        base_to_tip = cls._distance(mcp, tip)
        extension_ratio = base_to_tip / chain_len
        pip_angle = cls._angle(mcp, pip, dip)
        dip_angle = cls._angle(pip, dip, tip)

        # Обычная ладонь: кончик дальше от ладони/запястья и суставы почти прямые.
        straight_2d_like = pip_angle > 132 and dip_angle > 132
        tip_far_from_palm = cls._distance(palm_center, tip) > cls._distance(palm_center, pip) + 0.055 * palm_size
        tip_far_from_wrist = cls._distance(wrist, tip) > cls._distance(wrist, pip) + 0.055 * palm_size
        tip_above_pip = tip[1] < pip[1] - 0.014

        # Палец направлен в камеру: на экране он кажется коротким, но по z видно,
        # что кончик ушёл вперёд. Это исправляет частый баг "показывает 4 вместо 5".
        tip_forward = (mcp[2] - tip[2]) > max(0.018, 0.10 * palm_size)
        forward_straight = pip_angle > 98 and dip_angle > 98 and extension_ratio > 0.50

        fully_extended = extension_ratio > 0.68 and pip_angle > 118 and dip_angle > 118
        normally_open = straight_2d_like and (tip_far_from_palm or tip_far_from_wrist or tip_above_pip)
        forward_open = tip_forward and forward_straight

        return bool(fully_extended or normally_open or forward_open)

    @classmethod
    def _thumb_is_extended(
        cls,
        hand_landmarks: Sequence,
        palm_center: np.ndarray,
        palm_size: float,
    ) -> bool:
        cmc = cls._point(hand_landmarks, 1)
        mcp = cls._point(hand_landmarks, 2)
        ip = cls._point(hand_landmarks, 3)
        tip = cls._point(hand_landmarks, 4)
        index_mcp = cls._point(hand_landmarks, 5)

        chain_len = cls._distance(cmc, mcp) + cls._distance(mcp, ip) + cls._distance(ip, tip)
        if chain_len <= 1e-8:
            return False

        extension_ratio = cls._distance(cmc, tip) / chain_len
        mcp_angle = cls._angle(cmc, mcp, ip)
        ip_angle = cls._angle(mcp, ip, tip)
        far_from_palm = cls._distance(palm_center, tip) > cls._distance(palm_center, mcp) + 0.055 * palm_size
        far_from_index = cls._distance(tip, index_mcp) > cls._distance(ip, index_mcp) + 0.025 * palm_size
        tip_forward = (mcp[2] - tip[2]) > max(0.016, 0.08 * palm_size)

        straight = mcp_angle > 105 and ip_angle > 118
        return bool((extension_ratio > 0.60 and straight and (far_from_palm or far_from_index)) or (tip_forward and straight))

    @classmethod
    def count_fingers(cls, hand_landmarks: Sequence) -> int:
        """Считает поднятые пальцы с учётом глубины.

        Главная правка: палец, направленный в сторону камеры, больше не должен
        пропадать из счёта только потому, что в 2D-кадре он выглядит коротким.
        """
        if len(hand_landmarks) < 21:
            return 0

        wrist = cls._point(hand_landmarks, 0)
        index_mcp = cls._point(hand_landmarks, 5)
        middle_mcp = cls._point(hand_landmarks, 9)
        ring_mcp = cls._point(hand_landmarks, 13)
        pinky_mcp = cls._point(hand_landmarks, 17)
        palm_center = (wrist + index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 5.0
        palm_size = max(cls._distance(wrist, middle_mcp), cls._distance(index_mcp, pinky_mcp), 1e-6)

        count = 0
        if cls._thumb_is_extended(hand_landmarks, palm_center, palm_size):
            count += 1

        for indices in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            if cls._finger_is_extended(hand_landmarks, palm_center, wrist, palm_size, *indices):
                count += 1

        return count

    @staticmethod
    def gesture_from_count(count: int) -> str:
        mapping = {
            0: "Кулак",
            1: "Один палец",
            2: "Два пальца",
            3: "Три пальца",
            4: "Четыре пальца",
            5: "Открытая ладонь",
        }
        return mapping.get(count, "Жест")

    @staticmethod
    def _draw_hand(frame: np.ndarray, hand_landmarks: Sequence, w: int, h: int) -> None:
        # Спокойная разметка без неонового "AI"-вида.
        line_color = (116, 132, 102)   # BGR
        point_color = (83, 111, 91)
        point_fill = (245, 245, 238)

        for pt1_idx, pt2_idx in HAND_CONNECTIONS:
            pt1 = hand_landmarks[pt1_idx]
            pt2 = hand_landmarks[pt2_idx]
            cv2.line(
                frame,
                (int(pt1.x * w), int(pt1.y * h)),
                (int(pt2.x * w), int(pt2.y * h)),
                line_color,
                2,
                lineType=cv2.LINE_AA,
            )

        for lm in hand_landmarks:
            center = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, center, 5, point_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, center, 2, point_fill, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _put_text_box(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 22,
        text_color: Tuple[int, int, int] = (35, 33, 30),
        fill: Tuple[int, int, int] = (250, 248, 242),
        border: Tuple[int, int, int] = (210, 204, 194),
    ) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        font = get_cyrillic_font(font_size)
        x, y = position
        bbox = draw.textbbox((x, y), text, font=font)
        pad_x, pad_y = 10, 6
        rect = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
        draw.rounded_rectangle(rect, radius=10, fill=fill, outline=border, width=1)
        draw.text((x, y), text, font=font, fill=text_color)
        frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        hand_landmarks: Sequence,
        w: int,
        h: int,
        fingers: int,
        gesture: str,
    ) -> None:
        wrist = hand_landmarks[0]
        x = max(12, min(w - 220, int(wrist.x * w) - 70))
        y = max(12, min(h - 55, int(wrist.y * h) - 48))
        HandGestureDetector._put_text_box(frame, f"{fingers} • {gesture}", (x, y), font_size=22)

    def _draw_hud(self, frame: np.ndarray) -> None:
        text = f"Рук: {self.state.hands_count}   Пальцев: {self.state.fingers_count}"
        self._put_text_box(frame, text, (18, 18), font_size=22)
