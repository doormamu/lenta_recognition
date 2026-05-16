from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    sharpness: float
    brightness: float
    contrast: float
    glare_ratio: float
    dark_ratio: float
    score: float


def _normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0

    normalized = (value - min_value) / (max_value - min_value)
    return float(np.clip(normalized, 0.0, 1.0))


def calculate_sharpness(gray: np.ndarray) -> float:
    """
    Оценка резкости через дисперсию лапласиана.
    Чем больше значение, тем резче кадр.
    """

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_frame_quality(frame: np.ndarray) -> FrameQuality:
    """
    Возвращает набор простых признаков качества кадра.

    sharpness  — резкость;
    brightness — средняя яркость;
    contrast   — контраст;
    glare_ratio — доля пересвеченных пикселей;
    dark_ratio  — доля слишком темных пикселей;
    score      — итоговая оценка качества от 0 до 1.
    """

    if frame is None or frame.size == 0:
        raise ValueError("Передан пустой кадр")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness_raw = calculate_sharpness(gray)
    brightness_raw = float(np.mean(gray))
    contrast_raw = float(np.std(gray))

    glare_ratio = float(np.mean(gray > 245))
    dark_ratio = float(np.mean(gray < 20))

    sharpness_score = _normalize(sharpness_raw, 30.0, 500.0)

    brightness_score = 1.0 - abs(brightness_raw - 127.0) / 127.0
    brightness_score = float(np.clip(brightness_score, 0.0, 1.0))

    contrast_score = _normalize(contrast_raw, 20.0, 80.0)

    glare_penalty = float(np.clip(glare_ratio * 3.0, 0.0, 1.0))
    dark_penalty = float(np.clip(dark_ratio * 2.0, 0.0, 1.0))

    score = (
        0.45 * sharpness_score
        + 0.25 * brightness_score
        + 0.20 * contrast_score
        + 0.10 * (1.0 - glare_penalty)
    )

    score = score * (1.0 - 0.5 * dark_penalty)
    score = float(np.clip(score, 0.0, 1.0))

    return FrameQuality(
        sharpness=sharpness_raw,
        brightness=brightness_raw,
        contrast=contrast_raw,
        glare_ratio=glare_ratio,
        dark_ratio=dark_ratio,
        score=score,
    )