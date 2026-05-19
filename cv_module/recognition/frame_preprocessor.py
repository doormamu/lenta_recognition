from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from cv_module.detection.candidate_merger import BoundingBox


@dataclass(frozen=True)
class PreparedFrame:
    raw_frame: np.ndarray
    processed_frame: np.ndarray
    rotation_mode: str
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameLevelPreprocessor:
    """
    Обработка всего кадра один раз.

    Делает:
    - поворот всего кадра, если нужно;
    - легкое улучшение контраста;
    - легкое повышение резкости.

    Не делает:
    - full-frame upscale;
    - OCR thresholding;
    - perspective rectification.

    Почему:
    - full-frame upscale 3840x2160 слишком дорогой;
    - OCR-specific операции лучше делать на crop;
    - bbox можно корректно пересчитать только для простых геометрических операций.
    """

    def __init__(
        self,
        rotation_mode: str = "none",
        enable_enhance: bool = True,
        enable_denoise: bool = False,
        clahe_clip_limit: float = 2.0,
        sharpen_amount: float = 1.45,
    ) -> None:
        self.rotation_mode = rotation_mode
        self.enable_enhance = enable_enhance
        self.enable_denoise = enable_denoise
        self.clahe_clip_limit = clahe_clip_limit
        self.sharpen_amount = sharpen_amount

    def process(self, raw_frame: np.ndarray) -> PreparedFrame:
        if raw_frame is None or raw_frame.size == 0:
            empty = np.zeros((1, 1, 3), dtype=np.uint8)

            return PreparedFrame(
                raw_frame=empty,
                processed_frame=empty,
                rotation_mode="none",
                original_width=1,
                original_height=1,
                processed_width=1,
                processed_height=1,
                metadata={"error": "empty_frame"},
            )

        raw_frame = _ensure_bgr(raw_frame)

        original_height, original_width = raw_frame.shape[:2]

        processed = rotate_frame(
            frame=raw_frame,
            rotation_mode=self.rotation_mode,
        )

        if self.enable_enhance:
            processed = self._enhance_frame(processed)

        processed_height, processed_width = processed.shape[:2]

        return PreparedFrame(
            raw_frame=raw_frame,
            processed_frame=processed,
            rotation_mode=self.rotation_mode,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            metadata={
                "rotation_mode": self.rotation_mode,
                "enable_enhance": self.enable_enhance,
                "enable_denoise": self.enable_denoise,
                "clahe_clip_limit": self.clahe_clip_limit,
                "sharpen_amount": self.sharpen_amount,
            },
        )

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = _ensure_bgr(frame)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(8, 8),
        )

        l_channel = clahe.apply(l_channel)

        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.enable_denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                h=3,
                hColor=3,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        sharpened = cv2.addWeighted(
            enhanced,
            self.sharpen_amount,
            blurred,
            1.0 - self.sharpen_amount,
            0,
        )

        return sharpened


def rotate_frame(
    frame: np.ndarray,
    rotation_mode: str,
) -> np.ndarray:
    if rotation_mode in {"none", "", None}:
        return frame

    if rotation_mode == "rot90_ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if rotation_mode == "rot90_cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if rotation_mode == "rot180":
        return cv2.rotate(frame, cv2.ROTATE_180)

    raise ValueError(f"Unknown rotation_mode: {rotation_mode}")


def transform_bbox_by_rotation(
    bbox: BoundingBox,
    original_width: int,
    original_height: int,
    rotation_mode: str,
) -> BoundingBox:
    """
    Пересчитывает bbox из координат исходного кадра
    в координаты повернутого кадра.
    """

    if rotation_mode in {"none", "", None}:
        return bbox

    corners = [
        (bbox.x_min, bbox.y_min),
        (bbox.x_max, bbox.y_min),
        (bbox.x_max, bbox.y_max),
        (bbox.x_min, bbox.y_max),
    ]

    transformed = [
        transform_point_by_rotation(
            x=x,
            y=y,
            original_width=original_width,
            original_height=original_height,
            rotation_mode=rotation_mode,
        )
        for x, y in corners
    ]

    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]

    return BoundingBox(
        x_min=int(round(min(xs))),
        y_min=int(round(min(ys))),
        x_max=int(round(max(xs))),
        y_max=int(round(max(ys))),
    )


def transform_point_by_rotation(
    x: float,
    y: float,
    original_width: int,
    original_height: int,
    rotation_mode: str,
) -> tuple[float, float]:
    if rotation_mode in {"none", "", None}:
        return x, y

    if rotation_mode == "rot90_ccw":
        return y, original_width - x

    if rotation_mode == "rot90_cw":
        return original_height - y, x

    if rotation_mode == "rot180":
        return original_width - x, original_height - y

    raise ValueError(f"Unknown rotation_mode: {rotation_mode}")


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image