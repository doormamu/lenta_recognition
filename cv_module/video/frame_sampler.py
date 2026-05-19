from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv_module.video.preprocessing import build_preprocessor_for_video


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray
    quality_score: float


def sample_video_frames(
    video_path: str | Path,
    target_fps: float = 2.0,
    window_sec: float = 1.0,
    max_frames: int | None = 30,
    min_quality_score: float = 0.25,
    enable_undistort: bool = False,
    orientation: str = "none",
) -> list[SampledFrame]:
    """
    Выбирает хорошие кадры из видео.

    Теперь умеет:
    - исправлять широкоугольную дисторсию;
    - автоматически или вручную поворачивать кадры.

    orientation:
    - none
    - auto
    - rot90_ccw
    - rot90_cw
    - rot180
    """

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(target_fps, 0.1))))
    window_ms = max(1.0, window_sec * 1000.0)

    preprocessor = build_preprocessor_for_video(
        video_path=video_path,
        enable_undistort=enable_undistort,
        orientation=orientation,
    )

    selected: list[SampledFrame] = []

    current_window_id: int | None = None
    best_in_window: SampledFrame | None = None

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        if frame_index % step != 0:
            frame_index += 1
            continue

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        processed_frame = preprocessor.process(frame)

        quality_score = estimate_frame_quality(processed_frame)

        window_id = int(timestamp_ms // window_ms)

        sampled = SampledFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            image=processed_frame,
            quality_score=quality_score,
        )

        if current_window_id is None:
            current_window_id = window_id

        if window_id != current_window_id:
            if best_in_window is not None and best_in_window.quality_score >= min_quality_score:
                selected.append(best_in_window)

                if max_frames is not None and len(selected) >= max_frames:
                    break

            current_window_id = window_id
            best_in_window = sampled
        else:
            if best_in_window is None or sampled.quality_score > best_in_window.quality_score:
                best_in_window = sampled

        frame_index += 1

    if max_frames is None or len(selected) < max_frames:
        if best_in_window is not None and best_in_window.quality_score >= min_quality_score:
            selected.append(best_in_window)

    cap.release()

    if max_frames is not None:
        selected = selected[:max_frames]

    return selected


def estimate_frame_quality(frame: np.ndarray) -> float:
    """
    Оценка качества кадра от 0 до 1.

    Учитывает:
    - резкость;
    - контраст;
    - яркость;
    - пересветы.
    """

    if frame is None or frame.size == 0:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = np.clip(sharpness / 500.0, 0.0, 1.0)

    contrast = gray.std()
    contrast_score = np.clip(contrast / 64.0, 0.0, 1.0)

    brightness = gray.mean()
    brightness_score = 1.0 - abs(brightness - 128.0) / 128.0
    brightness_score = np.clip(brightness_score, 0.0, 1.0)

    overexposed_ratio = float(np.mean(gray > 245))
    underexposed_ratio = float(np.mean(gray < 10))
    exposure_penalty = np.clip(overexposed_ratio + underexposed_ratio, 0.0, 1.0)

    score = (
        0.45 * sharpness_score
        + 0.30 * contrast_score
        + 0.20 * brightness_score
        + 0.05 * (1.0 - exposure_penalty)
    )

    return float(np.clip(score, 0.0, 1.0))