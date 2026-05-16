from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv_module.detection.candidate_merger import (
    BoundingBox,
    PriceTagCandidate,
    filter_candidates_by_geometry,
    merge_candidates,
)
from cv_module.detection.qr_detector import CodeDetection, detect_codes


@dataclass(frozen=True)
class DetectionResult:
    candidates: list[PriceTagCandidate]
    codes: list[CodeDetection]


class PriceTagDetector:
    """
    Каскадный детектор кандидатов-ценников.

    Важно:
    это пока не финальное распознавание ценника, а поиск областей,
    которые дальше пойдут в OCR, QR-парсинг и постобработку.
    """

    def __init__(
        self,
        max_candidates: int = 80,
        min_candidate_score: float = 0.20,
    ) -> None:
        self.max_candidates = max_candidates
        self.min_candidate_score = min_candidate_score

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int | None = None,
        timestamp_ms: float | None = None,
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            return DetectionResult(candidates=[], codes=[])

        frame_height, frame_width = frame.shape[:2]

        codes = detect_codes(frame)

        candidates: list[PriceTagCandidate] = []

        candidates.extend(
            _candidates_from_codes(
                codes=codes,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

        candidates.extend(
            _candidates_from_text_like_regions(
                frame=frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

        candidates.extend(
            _candidates_from_color_regions(
                frame=frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

        candidates = filter_candidates_by_geometry(
            candidates=candidates,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= self.min_candidate_score
        ]

        candidates = merge_candidates(
            candidates=candidates,
            max_candidates=self.max_candidates,
        )

        candidates.sort(key=lambda item: item.score, reverse=True)

        return DetectionResult(
            candidates=candidates,
            codes=codes,
        )


def _candidates_from_codes(
    codes: list[CodeDetection],
    frame_width: int,
    frame_height: int,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> list[PriceTagCandidate]:
    result: list[PriceTagCandidate] = []

    for code in codes:
        anchor_boxes = _generate_anchor_boxes_around_code(
            code_bbox=code.bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        for index, bbox in enumerate(anchor_boxes):
            evidence = {
                "has_qr": code.code_type == "qr",
                "has_barcode": code.code_type == "barcode",
                "code_values": [code.value] if code.value else [],
                "anchor_code_type": code.code_type,
                "anchor_source": code.source,
                "anchor_confidence": code.confidence,
                "anchor_hypothesis": index,
            }

            base_score = 0.60 if code.value else 0.45

            if code.code_type == "qr":
                base_score += 0.08

            result.append(
                PriceTagCandidate(
                    bbox=bbox,
                    source=f"{code.code_type}_anchor",
                    score=float(np.clip(base_score - index * 0.03, 0.0, 1.0)),
                    evidence=evidence,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            )

    return result


def _generate_anchor_boxes_around_code(
    code_bbox: BoundingBox,
    frame_width: int,
    frame_height: int,
) -> list[BoundingBox]:
    """
    QR/штрихкод может находиться в разных местах ценника.
    Поэтому из одного кода генерируем несколько гипотез bbox.
    """

    hypotheses = [
        # Код справа вверху: ценник чаще всего уходит влево и вниз.
        code_bbox.expand(frame_width, frame_height, left=5.5, top=1.0, right=1.0, bottom=4.0),

        # Код справа внизу: ценник уходит влево и вверх.
        code_bbox.expand(frame_width, frame_height, left=5.5, top=4.0, right=1.0, bottom=1.0),

        # Код сверху по центру.
        code_bbox.expand(frame_width, frame_height, left=3.0, top=1.0, right=3.0, bottom=5.0),

        # Код снизу по центру.
        code_bbox.expand(frame_width, frame_height, left=3.0, top=5.0, right=3.0, bottom=1.0),

        # Маленький компактный ценник вокруг кода.
        code_bbox.expand(frame_width, frame_height, left=2.0, top=2.0, right=2.0, bottom=2.0),

        # Большой ценник, если код является малой частью A4/A3.
        code_bbox.expand(frame_width, frame_height, left=8.0, top=5.0, right=2.0, bottom=5.0),
    ]

    return _deduplicate_boxes(hypotheses)


def _candidates_from_text_like_regions(
    frame: np.ndarray,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> list[PriceTagCandidate]:
    """
    Ищет области, похожие на плотные группы текста.

    Это не OCR. Это быстрый CV-слой до распознавания:
    - ищем контрастные мелкие элементы;
    - склеиваем их морфологией;
    - получаем области-кластеры.
    """

    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Усиливаем локальный контраст.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Детектируем контрастные текстоподобные элементы.
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    edges = cv2.Canny(enhanced, 60, 160)

    text_mask = cv2.bitwise_or(adaptive, edges)

    # Склеиваем символы в строки/блоки.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 17))

    horizontal = cv2.dilate(text_mask, horizontal_kernel, iterations=1)
    vertical = cv2.dilate(text_mask, vertical_kernel, iterations=1)

    merged_mask = cv2.bitwise_or(horizontal, vertical)

    # Чистим шум.
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, clean_kernel)

    contours, _ = cv2.findContours(
        merged_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_boxes: list[BoundingBox] = []

    frame_area = frame_width * frame_height

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        bbox = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        if bbox.area < frame_area * 0.0003:
            continue

        if bbox.area > frame_area * 0.50:
            continue

        if bbox.width < 30 or bbox.height < 20:
            continue

        if bbox.aspect_ratio < 0.15 or bbox.aspect_ratio > 8.0:
            continue

        raw_boxes.append(bbox)

    grouped_boxes = _group_near_boxes(
        boxes=raw_boxes,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    candidates: list[PriceTagCandidate] = []

    for bbox in grouped_boxes:
        density = _estimate_text_density(text_mask, bbox)
        score = 0.25 + 0.45 * density
        score = float(np.clip(score, 0.0, 0.72))

        candidates.append(
            PriceTagCandidate(
                bbox=bbox,
                source="text_like_region",
                score=score,
                evidence={
                    "has_text_like_regions": True,
                    "text_density": round(float(density), 4),
                },
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

    return candidates


def _candidates_from_color_regions(
    frame: np.ndarray,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> list[PriceTagCandidate]:
    """
    Ищет цветовые области, похожие на ценники:
    - белые;
    - желтые;
    - красные элементы/половины.
    """

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Белые и светлые области.
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 160]),
        np.array([180, 70, 255]),
    )

    # Желтые области.
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 50, 80]),
        np.array([40, 255, 255]),
    )

    # Красные области, в HSV красный лежит на двух концах шкалы hue.
    red_mask_1 = cv2.inRange(
        hsv,
        np.array([0, 50, 60]),
        np.array([10, 255, 255]),
    )
    red_mask_2 = cv2.inRange(
        hsv,
        np.array([170, 50, 60]),
        np.array([180, 255, 255]),
    )
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    combined = cv2.bitwise_or(white_mask, yellow_mask)
    combined = cv2.bitwise_or(combined, red_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[PriceTagCandidate] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        bbox = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        area_ratio = bbox.area / frame_area if frame_area > 0 else 0.0

        if area_ratio < 0.001:
            continue

        if area_ratio > 0.60:
            continue

        if bbox.width < 40 or bbox.height < 25:
            continue

        if bbox.aspect_ratio < 0.20 or bbox.aspect_ratio > 6.00:
            continue

        color_hint = _estimate_color_hint(
            hsv=hsv,
            bbox=bbox,
            white_mask=white_mask,
            yellow_mask=yellow_mask,
            red_mask=red_mask,
        )

        score = 0.25

        if color_hint in {"yellow", "red", "mixed"}:
            score += 0.15

        rectangularity = _estimate_rectangularity(contour)
        score += 0.25 * rectangularity

        candidates.append(
            PriceTagCandidate(
                bbox=bbox,
                source="color_region",
                score=float(np.clip(score, 0.0, 0.65)),
                evidence={
                    "color_hint": color_hint,
                    "rectangularity": round(float(rectangularity), 4),
                },
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

    return candidates


def _group_near_boxes(
    boxes: list[BoundingBox],
    frame_width: int,
    frame_height: int,
    margin_ratio: float = 0.25,
    max_iterations: int = 3,
) -> list[BoundingBox]:
    """
    Группирует близкие текстоподобные области.

    Осторожно: не надо слишком сильно склеивать соседние ценники.
    Поэтому margin умеренный.
    """

    if not boxes:
        return []

    current = boxes[:]

    for _ in range(max_iterations):
        used = [False] * len(current)
        next_boxes: list[BoundingBox] = []

        changed = False

        for i, box in enumerate(current):
            if used[i]:
                continue

            group = box
            used[i] = True

            expanded_group = _expand_box_by_margin(
                group,
                frame_width,
                frame_height,
                margin_ratio,
            )

            for j in range(i + 1, len(current)):
                if used[j]:
                    continue

                other = current[j]
                expanded_other = _expand_box_by_margin(
                    other,
                    frame_width,
                    frame_height,
                    margin_ratio,
                )

                if expanded_group.iou(expanded_other) > 0:
                    group = group.union(other)
                    expanded_group = _expand_box_by_margin(
                        group,
                        frame_width,
                        frame_height,
                        margin_ratio,
                    )
                    used[j] = True
                    changed = True

            next_boxes.append(group)

        current = next_boxes

        if not changed:
            break

    return _deduplicate_boxes(current)


def _expand_box_by_margin(
    bbox: BoundingBox,
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
) -> BoundingBox:
    return bbox.expand(
        frame_width=frame_width,
        frame_height=frame_height,
        left=margin_ratio,
        top=margin_ratio,
        right=margin_ratio,
        bottom=margin_ratio,
    )


def _estimate_text_density(
    text_mask: np.ndarray,
    bbox: BoundingBox,
) -> float:
    crop = text_mask[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

    if crop.size == 0:
        return 0.0

    # Доля активных пикселей. Для текста она обычно не должна быть 0 и не должна быть 1.
    density = np.mean(crop > 0)

    # Переводим в условную оценку: слишком мало плохо, слишком много тоже плохо.
    if density <= 0.0:
        return 0.0

    if density > 0.75:
        return 0.15

    return float(np.clip(density / 0.35, 0.0, 1.0))


def _estimate_color_hint(
    hsv: np.ndarray,
    bbox: BoundingBox,
    white_mask: np.ndarray,
    yellow_mask: np.ndarray,
    red_mask: np.ndarray,
) -> str:
    crop_area = bbox.area

    if crop_area <= 0:
        return "unknown"

    white_ratio = _mask_ratio(white_mask, bbox)
    yellow_ratio = _mask_ratio(yellow_mask, bbox)
    red_ratio = _mask_ratio(red_mask, bbox)

    active_colors = []

    if white_ratio > 0.20:
        active_colors.append("white")

    if yellow_ratio > 0.08:
        active_colors.append("yellow")

    if red_ratio > 0.05:
        active_colors.append("red")

    if len(active_colors) > 1:
        return "mixed"

    if len(active_colors) == 1:
        return active_colors[0]

    return "unknown"


def _mask_ratio(mask: np.ndarray, bbox: BoundingBox) -> float:
    crop = mask[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

    if crop.size == 0:
        return 0.0

    return float(np.mean(crop > 0))


def _estimate_rectangularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)

    if area <= 0:
        return 0.0

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h

    if rect_area <= 0:
        return 0.0

    return float(np.clip(area / rect_area, 0.0, 1.0))


def _deduplicate_boxes(
    boxes: list[BoundingBox],
    iou_threshold: float = 0.85,
) -> list[BoundingBox]:
    result: list[BoundingBox] = []

    for box in boxes:
        duplicate = False

        for existing in result:
            if box.iou(existing) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            result.append(box)

    return result


def draw_detection_debug(
    frame: np.ndarray,
    result: DetectionResult,
    output_path: str | Path,
) -> None:
    """
    Сохраняет debug-картинку с найденными кандидатами.
    """

    debug = frame.copy()

    for code in result.codes:
        x_min, y_min, x_max, y_max = code.bbox.to_tuple()

        color = (255, 0, 0) if code.code_type == "qr" else (255, 0, 255)

        cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), color, 2)

        label = f"{code.code_type}:{code.confidence:.2f}"
        cv2.putText(
            debug,
            label,
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    for idx, candidate in enumerate(result.candidates):
        x_min, y_min, x_max, y_max = candidate.bbox.to_tuple()

        if candidate.source == "mixed":
            color = (0, 255, 0)
        elif "anchor" in candidate.source:
            color = (0, 200, 255)
        elif candidate.source == "text_like_region":
            color = (0, 255, 255)
        else:
            color = (0, 128, 255)

        cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), color, 2)

        label = f"{idx}:{candidate.source}:{candidate.score:.2f}"
        cv2.putText(
            debug,
            label,
            (x_min, min(frame.shape[0] - 10, y_max + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), debug)