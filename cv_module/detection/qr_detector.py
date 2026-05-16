from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cv_module.detection.candidate_merger import BoundingBox


@dataclass(frozen=True)
class CodeDetection:
    bbox: BoundingBox
    code_type: str
    value: str
    confidence: float
    source: str


def detect_codes(
    frame: np.ndarray,
    try_harder: bool = True,
) -> list[CodeDetection]:
    """
    Ищет QR-коды и штрихкоды на кадре.

    Основной способ:
    - OpenCV QRCodeDetector.

    Дополнительный способ:
    - pyzbar, если библиотека установлена.
    """

    if frame is None or frame.size == 0:
        return []

    detections: list[CodeDetection] = []

    scales = [1.0, 1.5, 2.0] if try_harder else [1.0]

    for scale in scales:
        detections.extend(_detect_qr_opencv(frame, scale=scale))

    detections.extend(_detect_codes_pyzbar_optional(frame))

    return _deduplicate_code_detections(detections)


def _detect_qr_opencv(
    frame: np.ndarray,
    scale: float = 1.0,
) -> list[CodeDetection]:
    detector = cv2.QRCodeDetector()

    original_height, original_width = frame.shape[:2]

    if scale != 1.0:
        work_frame = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        work_frame = frame

    variants = _make_preprocess_variants(work_frame)

    result: list[CodeDetection] = []

    for variant in variants:
        result.extend(
            _detect_qr_multi(
                detector=detector,
                image=variant,
                scale=scale,
                original_width=original_width,
                original_height=original_height,
            )
        )

        result.extend(
            _detect_qr_single(
                detector=detector,
                image=variant,
                scale=scale,
                original_width=original_width,
                original_height=original_height,
            )
        )

    return result


def _detect_qr_multi(
    detector: cv2.QRCodeDetector,
    image: np.ndarray,
    scale: float,
    original_width: int,
    original_height: int,
) -> list[CodeDetection]:
    result: list[CodeDetection] = []

    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    except Exception:
        return result

    if not ok or points is None:
        return result

    for decoded_value, qr_points in zip(decoded_info, points):
        bbox = _bbox_from_points(
            points=qr_points,
            scale=scale,
            original_width=original_width,
            original_height=original_height,
        )

        if bbox is None:
            continue

        value = decoded_value.strip() if decoded_value else ""

        result.append(
            CodeDetection(
                bbox=bbox,
                code_type="qr",
                value=value,
                confidence=0.95 if value else 0.60,
                source="opencv_qr_multi",
            )
        )

    return result


def _detect_qr_single(
    detector: cv2.QRCodeDetector,
    image: np.ndarray,
    scale: float,
    original_width: int,
    original_height: int,
) -> list[CodeDetection]:
    try:
        decoded_value, points, _ = detector.detectAndDecode(image)
    except Exception:
        return []

    if points is None:
        return []

    bbox = _bbox_from_points(
        points=points,
        scale=scale,
        original_width=original_width,
        original_height=original_height,
    )

    if bbox is None:
        return []

    value = decoded_value.strip() if decoded_value else ""

    return [
        CodeDetection(
            bbox=bbox,
            code_type="qr",
            value=value,
            confidence=0.90 if value else 0.55,
            source="opencv_qr_single",
        )
    ]


def _detect_codes_pyzbar_optional(
    frame: np.ndarray,
) -> list[CodeDetection]:
    try:
        from pyzbar.pyzbar import decode
    except Exception:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result: list[CodeDetection] = []

    try:
        decoded_objects = decode(gray)
    except Exception:
        return []

    for obj in decoded_objects:
        x, y, w, h = obj.rect

        bbox = BoundingBox(
            x_min=int(x),
            y_min=int(y),
            x_max=int(x + w),
            y_max=int(y + h),
        ).clamp(frame.shape[1], frame.shape[0])

        value = ""

        try:
            value = obj.data.decode("utf-8", errors="ignore").strip()
        except Exception:
            value = ""

        code_type = str(obj.type).lower()

        if "qrcode" in code_type:
            normalized_type = "qr"
        else:
            normalized_type = "barcode"

        result.append(
            CodeDetection(
                bbox=bbox,
                code_type=normalized_type,
                value=value,
                confidence=0.95 if value else 0.65,
                source="pyzbar",
            )
        )

    return result


def _make_preprocess_variants(frame: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gray)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    return [
        frame,
        gray,
        equalized,
        adaptive,
    ]


def _bbox_from_points(
    points: np.ndarray,
    scale: float,
    original_width: int,
    original_height: int,
) -> BoundingBox | None:
    points = np.asarray(points).reshape(-1, 2)

    if len(points) == 0:
        return None

    x_values = points[:, 0] / scale
    y_values = points[:, 1] / scale

    bbox = BoundingBox(
        x_min=int(np.min(x_values)),
        y_min=int(np.min(y_values)),
        x_max=int(np.max(x_values)),
        y_max=int(np.max(y_values)),
    ).clamp(original_width, original_height)

    if bbox.area <= 0:
        return None

    return bbox


def _deduplicate_code_detections(
    detections: list[CodeDetection],
    iou_threshold: float = 0.55,
) -> list[CodeDetection]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda item: item.confidence, reverse=True)

    result: list[CodeDetection] = []

    for detection in detections:
        duplicate_found = False

        for existing in result:
            same_type = detection.code_type == existing.code_type
            same_value = (
                detection.value
                and existing.value
                and detection.value == existing.value
            )
            high_overlap = detection.bbox.iou(existing.bbox) >= iou_threshold

            if same_type and (same_value or high_overlap):
                duplicate_found = True
                break

        if not duplicate_found:
            result.append(detection)

    return result