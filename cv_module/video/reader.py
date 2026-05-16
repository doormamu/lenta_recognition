from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Видео не найдено: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        fps = 25.0

    duration_sec = frame_count / fps if fps > 0 else 0.0

    return VideoMetadata(
        path=path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def iter_video_frames(
    video_path: str | Path,
    step: int = 1,
) -> Iterator[VideoFrame]:
    """
    Читает видео и возвращает каждый step-й кадр.

    """

    path = Path(video_path)

    if step <= 0:
        raise ValueError("step должен быть положительным числом")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_index % step == 0:
                timestamp_ms = frame_index / fps * 1000.0

                yield VideoFrame(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    image=frame,
                )

            frame_index += 1

    finally:
        cap.release()