from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv_module.video.quality import FrameQuality, calculate_frame_quality
from cv_module.video.reader import get_video_metadata, iter_video_frames


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray
    quality: FrameQuality


def sample_video_frames(
    video_path: str | Path,
    target_fps: float = 2.0,
    window_sec: float = 1.0,
    max_frames: int | None = 100,
    min_quality_score: float = 0.25,
) -> list[SampledFrame]:
    """
    Выбирает хорошие кадры из видео.

    1. Читаем видео не полностью, а с шагом target_fps.
    2. Для каждого кадра считаем качество.
    3. Внутри каждого временного окна window_sec оставляем лучший кадр.
    4. Убираем кадры с плохим quality score.
    5. Ограничиваем количество кадров через max_frames.
    """

    metadata = get_video_metadata(video_path)

    if target_fps <= 0:
        raise ValueError("target_fps должен быть положительным")

    if window_sec <= 0:
        raise ValueError("window_sec должен быть положительным")

    read_step = max(1, int(metadata.fps / target_fps))

    best_by_window: dict[int, SampledFrame] = {}

    for video_frame in iter_video_frames(video_path, step=read_step):
        quality = calculate_frame_quality(video_frame.image)

        if quality.score < min_quality_score:
            continue

        window_id = int((video_frame.timestamp_ms / 1000.0) // window_sec)

        sampled = SampledFrame(
            frame_index=video_frame.frame_index,
            timestamp_ms=video_frame.timestamp_ms,
            image=video_frame.image,
            quality=quality,
        )

        current_best = best_by_window.get(window_id)

        if current_best is None or sampled.quality.score > current_best.quality.score:
            best_by_window[window_id] = sampled

    selected = list(best_by_window.values())
    selected.sort(key=lambda item: item.timestamp_ms)

    if max_frames is not None and len(selected) > max_frames:
        selected = _limit_evenly(selected, max_frames)

    return selected


def save_sampled_frames(
    sampled_frames: list[SampledFrame],
    output_dir: str | Path,
    prefix: str = "frame",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for item in sampled_frames:
        timestamp_sec = item.timestamp_ms / 1000.0

        filename = (
            f"{prefix}_"
            f"idx_{item.frame_index:06d}_"
            f"time_{timestamp_sec:08.2f}_"
            f"score_{item.quality.score:.3f}.jpg"
        )

        cv2.imwrite(str(output_path / filename), item.image)


def _limit_evenly(
    frames: list[SampledFrame],
    max_frames: int,
) -> list[SampledFrame]:
    """
    Если кадров слишком много, оставляем max_frames равномерно по видео.
    """

    if max_frames <= 0:
        return []

    if len(frames) <= max_frames:
        return frames

    indexes = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return [frames[index] for index in indexes]