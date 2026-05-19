from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


UNKNOWN_VALUE = "-"


@dataclass(frozen=True)
class ColorDetectionResult:
    color: str
    confidence: float
    ratios: dict[str, float]


def detect_price_tag_color(image: np.ndarray) -> ColorDetectionResult:
    """
    Определяет основной цвет ценника по crop.

    Возвращает:
    - белый
    - желтый
    - красный
    - смешанный
    - -
    """

    if image is None or image.size == 0:
        return ColorDetectionResult(
            color=UNKNOWN_VALUE,
            confidence=0.0,
            ratios={},
        )

    image = _ensure_bgr(image)

    # Слишком большой crop уменьшаем для скорости.
    max_side = 900
    height, width = image.shape[:2]

    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Игнорируем совсем темные области: текст, тени, фон.
    valid_mask = v > 55

    if np.mean(valid_mask) < 0.05:
        return ColorDetectionResult(
            color=UNKNOWN_VALUE,
            confidence=0.0,
            ratios={},
        )

    white_mask = (
        (s < 65)
        & (v > 145)
        & valid_mask
    )

    yellow_mask = (
        (h >= 15)
        & (h <= 42)
        & (s > 55)
        & (v > 80)
        & valid_mask
    )

    red_mask_1 = (
        (h <= 12)
        & (s > 55)
        & (v > 75)
        & valid_mask
    )

    red_mask_2 = (
        (h >= 165)
        & (h <= 179)
        & (s > 55)
        & (v > 75)
        & valid_mask
    )

    red_mask = red_mask_1 | red_mask_2

    # Цветная нижняя часть ценника важнее, поэтому считаем еще нижнюю половину.
    bottom = hsv[hsv.shape[0] // 2 :, :, :]
    bh, bs, bv = cv2.split(bottom)
    bottom_valid = bv > 55

    bottom_yellow = (
        (bh >= 15)
        & (bh <= 42)
        & (bs > 55)
        & (bv > 80)
        & bottom_valid
    )

    bottom_red = (
        (
            ((bh <= 12) | ((bh >= 165) & (bh <= 179)))
            & (bs > 55)
            & (bv > 75)
            & bottom_valid
        )
    )

    valid_count = max(int(np.sum(valid_mask)), 1)
    bottom_valid_count = max(int(np.sum(bottom_valid)), 1)

    white_ratio = float(np.sum(white_mask) / valid_count)
    yellow_ratio = float(np.sum(yellow_mask) / valid_count)
    red_ratio = float(np.sum(red_mask) / valid_count)

    bottom_yellow_ratio = float(np.sum(bottom_yellow) / bottom_valid_count)
    bottom_red_ratio = float(np.sum(bottom_red) / bottom_valid_count)

    # Усиливаем нижний цветной блок, потому что у промо-ценников цвет часто снизу.
    yellow_score = max(yellow_ratio, bottom_yellow_ratio * 0.9)
    red_score = max(red_ratio, bottom_red_ratio * 0.9)
    white_score = white_ratio

    ratios = {
        "white_ratio": round(white_ratio, 4),
        "yellow_ratio": round(yellow_ratio, 4),
        "red_ratio": round(red_ratio, 4),
        "bottom_yellow_ratio": round(bottom_yellow_ratio, 4),
        "bottom_red_ratio": round(bottom_red_ratio, 4),
        "white_score": round(white_score, 4),
        "yellow_score": round(yellow_score, 4),
        "red_score": round(red_score, 4),
    }

    colored_score = max(yellow_score, red_score)

    if red_score >= 0.10 and yellow_score >= 0.10:
        return ColorDetectionResult(
            color="смешанный",
            confidence=min(1.0, red_score + yellow_score),
            ratios=ratios,
        )

    if red_score >= 0.08 and red_score >= yellow_score * 1.15:
        return ColorDetectionResult(
            color="красный",
            confidence=min(1.0, red_score * 2.5),
            ratios=ratios,
        )

    if yellow_score >= 0.08 and yellow_score >= red_score * 1.05:
        return ColorDetectionResult(
            color="желтый",
            confidence=min(1.0, yellow_score * 2.5),
            ratios=ratios,
        )

    if white_score >= 0.25 and colored_score < 0.08:
        return ColorDetectionResult(
            color="белый",
            confidence=min(1.0, white_score * 1.5),
            ratios=ratios,
        )

    return ColorDetectionResult(
        color=UNKNOWN_VALUE,
        confidence=max(white_score, yellow_score, red_score),
        ratios=ratios,
    )


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image