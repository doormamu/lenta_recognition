from __future__ import annotations

from dataclasses import dataclass
from string import printable

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
    Ищет только реально декодированные QR-коды и штрихкоды.

    Важно:
    - пустые OpenCV-срабатывания без decoded value отбрасываются;
    - случайные фоновые области без валидного значения отбрасываются;
    - QR должен быть похож на квадрат;
    - barcode должен быть похож на вытянутый код или иметь корректное значение.
    """

    if frame is None or frame.size == 0:
        return []

    detections: list[CodeDetection] = []

    scales = [1.0, 1.5, 2.0, 3.0] if try_harder else [1.0]

    for scale in scales:
        detections.extend(_detect_qr_opencv(frame, scale=scale))

    detections.extend(_detect_codes_pyzbar_optional(frame, try_harder=try_harder))

    detections = [
        detection
        for detection in detections
        if _is_valid_detection(detection, frame_width=frame.shape[1], frame_height=frame.shape[0])
    ]

    return _deduplicate_code_detections(detections)


def detect_codes_in_boxes(
    frame: np.ndarray,
    boxes: list[BoundingBox],
    padding_ratio: float = 0.10,
    min_crop_size: int = 40,
) -> list[CodeDetection]:
    """
    Ищет QR/штрихкоды внутри кандидатов.

    На полном кадре QR часто слишком мелкий.
    Поэтому сначала находим потенциальные области ценников,
    потом увеличиваем crop и ищем код внутри него.
    """

    if frame is None or frame.size == 0:
        return []

    frame_height, frame_width = frame.shape[:2]

    detections: list[CodeDetection] = []

    for box in boxes:
        padded_box = box.expand(
            frame_width=frame_width,
            frame_height=frame_height,
            left=padding_ratio,
            top=padding_ratio,
            right=padding_ratio,
            bottom=padding_ratio,
        )

        if padded_box.width < min_crop_size or padded_box.height < min_crop_size:
            continue

        crop = frame[
            padded_box.y_min:padded_box.y_max,
            padded_box.x_min:padded_box.x_max,
        ]

        if crop.size == 0:
            continue

        crop_detections = detect_codes(crop, try_harder=True)

        for detection in crop_detections:
            global_bbox = BoundingBox(
                x_min=detection.bbox.x_min + padded_box.x_min,
                y_min=detection.bbox.y_min + padded_box.y_min,
                x_max=detection.bbox.x_max + padded_box.x_min,
                y_max=detection.bbox.y_max + padded_box.y_min,
            ).clamp(frame_width, frame_height)

            global_detection = CodeDetection(
                bbox=global_bbox,
                code_type=detection.code_type,
                value=detection.value,
                confidence=min(1.0, detection.confidence + 0.03),
                source=f"crop_{detection.source}",
            )

            if _is_valid_detection(
                global_detection,
                frame_width=frame_width,
                frame_height=frame_height,
            ):
                detections.append(global_detection)

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
        value = decoded_value.strip() if decoded_value else ""

        # Ключевая правка:
        # если OpenCV нашел QR-подобные точки, но не смог декодировать значение,
        # это не считаем кодом.
        if not value:
            continue

        bbox = _bbox_from_points(
            points=qr_points,
            scale=scale,
            original_width=original_width,
            original_height=original_height,
        )

        if bbox is None:
            continue

        result.append(
            CodeDetection(
                bbox=bbox,
                code_type="qr",
                value=value,
                confidence=0.95,
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

    value = decoded_value.strip() if decoded_value else ""

    # Ключевая правка:
    # пустой decoded value не принимаем.
    if not value:
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

    return [
        CodeDetection(
            bbox=bbox,
            code_type="qr",
            value=value,
            confidence=0.92,
            source="opencv_qr_single",
        )
    ]


def _detect_codes_pyzbar_optional(
    frame: np.ndarray,
    try_harder: bool,
) -> list[CodeDetection]:
    try:
        from pyzbar.pyzbar import decode
    except Exception:
        return []

    variants = _make_pyzbar_variants(frame, try_harder=try_harder)

    result: list[CodeDetection] = []

    for variant, scale in variants:
        try:
            decoded_objects = decode(variant)
        except Exception:
            continue

        for obj in decoded_objects:
            x, y, w, h = obj.rect

            value = ""

            try:
                value = obj.data.decode("utf-8", errors="ignore").strip()
            except Exception:
                value = ""

            if not value:
                continue

            bbox = BoundingBox(
                x_min=int(x / scale),
                y_min=int(y / scale),
                x_max=int((x + w) / scale),
                y_max=int((y + h) / scale),
            ).clamp(frame.shape[1], frame.shape[0])

            raw_type = str(obj.type).lower()

            if "qrcode" in raw_type:
                normalized_type = "qr"
            else:
                normalized_type = "barcode"

            result.append(
                CodeDetection(
                    bbox=bbox,
                    code_type=normalized_type,
                    value=value,
                    confidence=0.97,
                    source=f"pyzbar_{raw_type}",
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

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    return [
        frame,
        gray,
        equalized,
        adaptive,
        sharp,
    ]


def _make_pyzbar_variants(
    frame: np.ndarray,
    try_harder: bool,
) -> list[tuple[np.ndarray, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    variants: list[tuple[np.ndarray, float]] = [
        (gray, 1.0),
    ]

    if not try_harder:
        return variants

    equalized = cv2.equalizeHist(gray)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    variants.extend(
        [
            (equalized, 1.0),
            (adaptive, 1.0),
        ]
    )

    for scale in [1.5, 2.0, 3.0]:
        resized = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        resized_equalized = cv2.equalizeHist(resized)

        variants.append((resized, scale))
        variants.append((resized_equalized, scale))

    return variants


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


def _is_valid_detection(
    detection: CodeDetection,
    frame_width: int,
    frame_height: int,
) -> bool:
    value = detection.value.strip()

    if not value:
        return False

    if not _is_valid_code_value(value, detection.code_type):
        return False

    bbox = detection.bbox

    if bbox.width < 6 or bbox.height < 6:
        return False

    frame_area = frame_width * frame_height

    if frame_area <= 0:
        return False

    area_ratio = bbox.area / frame_area

    if area_ratio < 0.000005:
        return False

    if area_ratio > 0.25:
        return False

    aspect = bbox.aspect_ratio

    if detection.code_type == "qr":
        # QR должен быть примерно квадратным.
        if aspect < 0.45 or aspect > 2.20:
            return False

    if detection.code_type == "barcode":
        # Штрихкод чаще вытянутый, но оставляем запас.
        if aspect < 0.45 or aspect > 15.0:
            return False

    return True


def _is_valid_code_value(
    value: str,
    code_type: str,
) -> bool:
    value = value.strip()

    if len(value) < 4:
        return False

    if len(value) > 800:
        return False

    printable_chars = set(printable)

    printable_count = sum(1 for char in value if char in printable_chars or ord(char) > 127)
    printable_ratio = printable_count / max(1, len(value))

    if printable_ratio < 0.85:
        return False

    if "\x00" in value:
        return False

    if code_type == "barcode":
        return _is_valid_barcode_value(value)

    if code_type == "qr":
        return _is_valid_qr_value(value)

    return True


def _is_valid_barcode_value(value: str) -> bool:
    compact = value.replace(" ", "").replace("-", "").strip()

    if not compact.isdigit():
        return False

    if len(compact) not in {8, 12, 13, 14}:
        return False

    if len(set(compact)) <= 2:
        return False

    if len(compact) == 8:
        return _check_ean8(compact)

    if len(compact) == 12:
        return _check_upca(compact)

    if len(compact) == 13:
        return _check_ean13(compact)

    # Для 14-значных кодов checksum может зависеть от формата,
    # поэтому проверяем только базовую структуру.
    return True


def _is_valid_qr_value(value: str) -> bool:
    cleaned = value.strip()

    if len(cleaned) < 6:
        return False

    lower = cleaned.lower()

    known_tokens = [
        "barcode",
        "price",
        "price1",
        "price2",
        "price3",
        "price4",
        "p1",
        "p2",
        "p3",
        "p4",
        "actionprice",
        "actioncode",
        "wholesale",
        "wl1",
        "wl2",
        "sku",
        "id",
        "code",
    ]

    if any(token in lower for token in known_tokens):
        return True

    # Часто QR содержит query-string, json или пары ключ=значение.
    structural_chars = ["=", "&", "?", "{", "}", ":", ";", "|"]

    if any(char in cleaned for char in structural_chars) and len(cleaned) >= 8:
        return True

    has_letter = any(char.isalpha() for char in cleaned)
    has_digit = any(char.isdigit() for char in cleaned)

    # Запасной вариант для обычных QR-строк.
    if has_letter and has_digit and len(cleaned) >= 10:
        return True

    # Иногда QR может содержать только длинный цифровой код.
    if cleaned.isdigit() and len(cleaned) in {8, 12, 13, 14}:
        return _is_valid_barcode_value(cleaned)

    return False


def _check_ean8(value: str) -> bool:
    if len(value) != 8 or not value.isdigit():
        return False

    digits = [int(char) for char in value]
    checksum = (3 * sum(digits[0:7:2]) + sum(digits[1:7:2])) % 10
    expected = (10 - checksum) % 10

    return expected == digits[-1]


def _check_ean13(value: str) -> bool:
    if len(value) != 13 or not value.isdigit():
        return False

    digits = [int(char) for char in value]
    checksum = (sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10
    expected = (10 - checksum) % 10

    return expected == digits[-1]


def _check_upca(value: str) -> bool:
    if len(value) != 12 or not value.isdigit():
        return False

    digits = [int(char) for char in value]
    checksum = (3 * sum(digits[0:11:2]) + sum(digits[1:11:2])) % 10
    expected = (10 - checksum) % 10

    return expected == digits[-1]


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
            same_value = detection.value and existing.value and detection.value == existing.value
            high_overlap = detection.bbox.iou(existing.bbox) >= iou_threshold

            if same_type and (same_value or high_overlap):
                duplicate_found = True
                break

        if not duplicate_found:
            result.append(detection)

    return result