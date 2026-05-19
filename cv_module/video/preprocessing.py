from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSettings:
    image_size: tuple[int, int]  # width, height
    diagonal_mm: float
    focal_len_mm: float


DEFAULT_CAMERA_SETTINGS = CameraSettings(
    image_size=(3840, 2160),
    diagonal_mm=16.0 / 2.8,
    focal_len_mm=2.8,
)

DEFAULT_DISTORTION_COEFFS = [
    -0.276,
    0.06,
    0.0084,
    -0.0016,
    -0.0044,
]


class DistortionCorrector:
    """
    Исправляет широкоугольную дисторсию камеры.

    Использует:
    - camera matrix K;
    - коэффициенты k1, k2, p1, p2, k3;
    - cv2.initUndistortRectifyMap;
    - cv2.remap.

    Карты remap считаются один раз и переиспользуются.
    """

    def __init__(
        self,
        camera_settings: CameraSettings,
        distortion_coeffs: list[float],
        alpha: float = 0.0,
    ) -> None:
        self.width = camera_settings.image_size[0]
        self.height = camera_settings.image_size[1]
        self.diagonal_mm = camera_settings.diagonal_mm
        self.focal_len_mm = camera_settings.focal_len_mm
        self.dist = np.array(distortion_coeffs, dtype=np.float32)
        self.alpha = alpha

        self.camera_matrix = self._calculate_camera_matrix()
        self.map1, self.map2, self.roi = self._create_undistort_maps()

    def _calculate_camera_matrix(self) -> np.ndarray:
        aspect_ratio = self.width / self.height

        height_mm = self.diagonal_mm / math.sqrt(aspect_ratio**2 + 1)
        width_mm = aspect_ratio * height_mm

        fx = (self.focal_len_mm * self.width) / width_mm
        fy = (self.focal_len_mm * self.height) / height_mm

        return np.array(
            [
                [fx, 0.0, self.width / 2.0],
                [0.0, fy, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _create_undistort_maps(self):
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist,
            (self.width, self.height),
            self.alpha,
            (self.width, self.height),
        )

        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist,
            None,
            new_camera_matrix,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        return map1, map2, roi

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        undistorted = cv2.remap(
            frame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
        )

        x, y, w, h = self.roi

        if w <= 0 or h <= 0:
            return undistorted

        return undistorted[y:y + h, x:x + w]


class DynamicDistortionCorrector:
    """
    Обертка над DistortionCorrector.

    Если видео вдруг не 3840x2160, пересчитывает camera settings
    под фактический размер кадра.
    """

    def __init__(
        self,
        base_camera_settings: CameraSettings = DEFAULT_CAMERA_SETTINGS,
        distortion_coeffs: list[float] | None = None,
        alpha: float = 0.0,
    ) -> None:
        self.base_camera_settings = base_camera_settings
        self.distortion_coeffs = distortion_coeffs or DEFAULT_DISTORTION_COEFFS
        self.alpha = alpha

        self._corrector: DistortionCorrector | None = None
        self._frame_size: tuple[int, int] | None = None

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        height, width = frame.shape[:2]
        frame_size = (width, height)

        if self._corrector is None or self._frame_size != frame_size:
            camera_settings = CameraSettings(
                image_size=frame_size,
                diagonal_mm=self.base_camera_settings.diagonal_mm,
                focal_len_mm=self.base_camera_settings.focal_len_mm,
            )

            self._corrector = DistortionCorrector(
                camera_settings=camera_settings,
                distortion_coeffs=self.distortion_coeffs,
                alpha=self.alpha,
            )

            self._frame_size = frame_size

        return self._corrector.undistort_frame(frame)


class FramePreprocessor:
    """
    Общий препроцессор кадра:

    1. Исправляет дисторсию.
    2. Поворачивает кадр.
    """

    def __init__(
        self,
        enable_undistort: bool = True,
        rotation_mode: str = "none",
        camera_settings: CameraSettings = DEFAULT_CAMERA_SETTINGS,
        distortion_coeffs: list[float] | None = None,
    ) -> None:
        self.enable_undistort = enable_undistort
        self.rotation_mode = rotation_mode

        self.distortion_corrector = DynamicDistortionCorrector(
            base_camera_settings=camera_settings,
            distortion_coeffs=distortion_coeffs or DEFAULT_DISTORTION_COEFFS,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        result = frame

        if self.enable_undistort:
            result = self.distortion_corrector.undistort_frame(result)

        result = rotate_frame(result, self.rotation_mode)

        return result


def rotate_frame(frame: np.ndarray, rotation_mode: str) -> np.ndarray:
    if rotation_mode in {"none", "", None}:
        return frame

    if rotation_mode == "rot90_ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if rotation_mode == "rot90_cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if rotation_mode == "rot180":
        return cv2.rotate(frame, cv2.ROTATE_180)

    raise ValueError(f"Unknown rotation_mode: {rotation_mode}")


def build_preprocessor_for_video(
    video_path: str | Path,
    enable_undistort: bool = True,
    orientation: str = "auto",
    camera_settings: CameraSettings = DEFAULT_CAMERA_SETTINGS,
    distortion_coeffs: list[float] | None = None,
) -> FramePreprocessor:
    """
    Создает препроцессор для видео.

    orientation:
    - "none"
    - "rot90_ccw"
    - "rot90_cw"
    - "rot180"
    - "auto"

    auto:
    - оценивает движение картинки по optical flow;
    - если картинка едет сверху вниз, ставит rot90_ccw;
    - если снизу вверх, ставит rot90_cw;
    - если справа налево, ставит rot180;
    - если уже слева направо, оставляет none.
    """

    if orientation == "auto":
        orientation = detect_rotation_mode_from_video(
            video_path=video_path,
            enable_undistort=enable_undistort,
            camera_settings=camera_settings,
            distortion_coeffs=distortion_coeffs or DEFAULT_DISTORTION_COEFFS,
        )

    return FramePreprocessor(
        enable_undistort=enable_undistort,
        rotation_mode=orientation,
        camera_settings=camera_settings,
        distortion_coeffs=distortion_coeffs or DEFAULT_DISTORTION_COEFFS,
    )


def detect_rotation_mode_from_video(
    video_path: str | Path,
    enable_undistort: bool = True,
    camera_settings: CameraSettings = DEFAULT_CAMERA_SETTINGS,
    distortion_coeffs: list[float] | None = None,
    max_pairs: int = 8,
) -> str:
    """
    Автоматически определяет поворот по движению картинки.

    Логика:
    - считаем optical flow между несколькими кадрами;
    - берем медианный сдвиг;
    - если вертикальный сдвиг сильнее горизонтального, поворачиваем на 90°;
    - хотим, чтобы движение стало слева направо.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return "none"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_count <= 2:
        cap.release()
        return "none"

    indices = np.linspace(
        0,
        max(1, frame_count - 1),
        num=min(max_pairs + 1, frame_count),
        dtype=int,
    )

    corrector = DynamicDistortionCorrector(
        base_camera_settings=camera_settings,
        distortion_coeffs=distortion_coeffs or DEFAULT_DISTORTION_COEFFS,
    )

    sampled: list[np.ndarray] = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        if enable_undistort:
            frame = corrector.undistort_frame(frame)

        frame = _resize_for_motion_estimation(frame)

        sampled.append(frame)

    cap.release()

    if len(sampled) < 2:
        return "none"

    shifts: list[tuple[float, float]] = []

    for prev_frame, next_frame in zip(sampled[:-1], sampled[1:]):
        shift = _estimate_global_shift(prev_frame, next_frame)

        if shift is not None:
            shifts.append(shift)

    if not shifts:
        return "none"

    dx = float(np.median([item[0] for item in shifts]))
    dy = float(np.median([item[1] for item in shifts]))

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dy > abs_dx * 1.25 and abs_dy > 0.30:
        # В координатах изображения y растет вниз.
        # Если картинка едет сверху вниз, dy > 0.
        # Чтобы это стало движением слева направо, нужен rot90_ccw.
        if dy > 0:
            return "rot90_ccw"

        return "rot90_cw"

    if abs_dx > abs_dy * 1.25 and abs_dx > 0.30:
        # Если картинка уже едет слева направо, оставляем как есть.
        if dx > 0:
            return "none"

        # Если справа налево — переворачиваем на 180.
        return "rot180"

    return "none"


def _resize_for_motion_estimation(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _estimate_global_shift(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
) -> tuple[float, float] | None:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=600,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )

    if points is None or len(points) < 20:
        return None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if next_points is None or status is None:
        return None

    status = status.reshape(-1).astype(bool)

    good_prev = points.reshape(-1, 2)[status]
    good_next = next_points.reshape(-1, 2)[status]

    if len(good_prev) < 20:
        return None

    flow = good_next - good_prev

    dx = float(np.median(flow[:, 0]))
    dy = float(np.median(flow[:, 1]))

    return dx, dy