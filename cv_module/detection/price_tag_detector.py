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
from cv_module.detection.qr_detector import (
    CodeDetection,
    detect_codes,
    detect_codes_in_boxes,
)


@dataclass(frozen=True)
class DetectionResult:
    candidates: list[PriceTagCandidate]
    codes: list[CodeDetection]


class PriceTagDetector:
    def __init__(
        self,
        max_candidates: int = 80,
        min_candidate_score: float = 0.50,
        enable_code_detection: bool = False,
    ) -> None:
        self.max_candidates = max_candidates
        self.min_candidate_score = min_candidate_score
        self.enable_code_detection = enable_code_detection

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int | None = None,
        timestamp_ms: float | None = None,
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            return DetectionResult(candidates=[], codes=[])

        frame_height, frame_width = frame.shape[:2]

        candidates: list[PriceTagCandidate] = []

        rectangular_candidates = _candidates_from_rectangular_regions(
            frame=frame,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

        text_candidates = _candidates_from_text_like_regions(
            frame=frame,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

        promo_layout_candidates = _candidates_from_promo_color_layouts(
            frame=frame,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

        color_candidates = _candidates_from_color_regions(
            frame=frame,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

        candidates.extend(rectangular_candidates)
        candidates.extend(text_candidates)
        candidates.extend(promo_layout_candidates)
        candidates.extend(color_candidates)

        candidates = _boost_rectangles_with_text(candidates)

        candidates = filter_candidates_by_geometry(
            candidates=candidates,
            frame_width=frame_width,
            frame_height=frame_height,
            max_area_ratio=0.18,
        )

        codes: list[CodeDetection] = []

        if self.enable_code_detection:
            probe_candidates = _select_diverse_candidates(
                candidates=candidates,
                max_candidates=20,
            )

            probe_boxes = [candidate.bbox for candidate in probe_candidates]

            full_frame_codes = detect_codes(frame, try_harder=False)

            crop_codes = detect_codes_in_boxes(
                frame=frame,
                boxes=probe_boxes,
                padding_ratio=0.10,
            )

            codes = _deduplicate_codes(full_frame_codes + crop_codes)

        candidates.extend(
            _candidates_from_codes(
                codes=codes,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

        candidates = _boost_candidates_with_codes(
            candidates=candidates,
            codes=codes,
        )

        candidates = filter_candidates_by_geometry(
            candidates=candidates,
            frame_width=frame_width,
            frame_height=frame_height,
            max_area_ratio=0.18,
        )

        candidates = merge_candidates(
            candidates=candidates,
            max_candidates=None,
        )

        candidates = _filter_final_candidates(
            candidates=candidates,
            min_candidate_score=self.min_candidate_score,
        )

        candidates = _select_diverse_candidates(
            candidates=candidates,
            max_candidates=self.max_candidates,
        )

        return DetectionResult(
            candidates=candidates,
            codes=codes,
        )


def _candidates_from_rectangular_regions(
    frame: np.ndarray,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> list[PriceTagCandidate]:
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed = cv2.dilate(closed, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[PriceTagCandidate] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        bbox = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        if bbox.width < 35 or bbox.height < 20:
            continue

        area_ratio = bbox.area / frame_area if frame_area > 0 else 0.0

        if area_ratio < 0.0004:
            continue

        if area_ratio > 0.08:
            continue

        if bbox.aspect_ratio < 0.25 or bbox.aspect_ratio > 5.5:
            continue

        rectangularity = _estimate_rectangularity(contour)
        edge_density = _estimate_edge_density(edges, bbox)
        size_score = _estimate_size_score(area_ratio)

        if edge_density < 0.025:
            continue

        if rectangularity < 0.25:
            continue

        score = (
            0.12
            + 0.40 * rectangularity
            + 0.30 * min(edge_density / 0.12, 1.0)
            + 0.10 * size_score
        )

        score = float(np.clip(score, 0.0, 0.78))

        candidates.append(
            PriceTagCandidate(
                bbox=bbox,
                source="rectangular_region",
                score=score,
                evidence={
                    "has_rectangular_shape": True,
                    "rectangularity": round(float(rectangularity), 4),
                    "edge_density": round(float(edge_density), 4),
                    "area_ratio": round(float(area_ratio), 5),
                },
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

    return candidates


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
                "code_values": [code.value],
                "anchor_code_type": code.code_type,
                "anchor_source": code.source,
                "anchor_confidence": code.confidence,
                "anchor_hypothesis": index,
                "anchor_bbox": code.bbox.to_tuple(),
            }

            if code.code_type == "qr":
                base_score = 0.95
            else:
                base_score = 0.88

            result.append(
                PriceTagCandidate(
                    bbox=bbox,
                    source=f"{code.code_type}_anchor",
                    score=float(np.clip(base_score - index * 0.06, 0.0, 1.0)),
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
    hypotheses = [
        code_bbox.expand(
            frame_width,
            frame_height,
            left=4.0,
            top=0.8,
            right=0.8,
            bottom=3.0,
        ),
        code_bbox.expand(
            frame_width,
            frame_height,
            left=4.0,
            top=3.0,
            right=0.8,
            bottom=0.8,
        ),
        code_bbox.expand(
            frame_width,
            frame_height,
            left=2.5,
            top=0.8,
            right=2.5,
            bottom=3.5,
        ),
        code_bbox.expand(
            frame_width,
            frame_height,
            left=2.5,
            top=3.5,
            right=2.5,
            bottom=0.8,
        ),
        code_bbox.expand(
            frame_width,
            frame_height,
            left=1.8,
            top=1.8,
            right=1.8,
            bottom=1.8,
        ),
    ]

    hypotheses = [
        _cap_box_size(
            bbox=box,
            anchor=code_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            max_width_ratio=0.35,
            max_height_ratio=0.35,
        )
        for box in hypotheses
    ]

    return _deduplicate_boxes(hypotheses)


def _cap_box_size(
    bbox: BoundingBox,
    anchor: BoundingBox,
    frame_width: int,
    frame_height: int,
    max_width_ratio: float,
    max_height_ratio: float,
) -> BoundingBox:
    max_width = int(frame_width * max_width_ratio)
    max_height = int(frame_height * max_height_ratio)

    if bbox.width <= max_width and bbox.height <= max_height:
        return bbox

    center_x, center_y = anchor.center

    width = min(bbox.width, max_width)
    height = min(bbox.height, max_height)

    capped = BoundingBox(
        x_min=int(center_x - width / 2),
        y_min=int(center_y - height / 2),
        x_max=int(center_x + width / 2),
        y_max=int(center_y + height / 2),
    )

    return capped.clamp(frame_width, frame_height)


def _candidates_from_text_like_regions(
    frame: np.ndarray,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> list[PriceTagCandidate]:
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

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

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 4))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))

    horizontal = cv2.dilate(text_mask, horizontal_kernel, iterations=1)
    vertical = cv2.dilate(text_mask, vertical_kernel, iterations=1)

    merged_mask = cv2.bitwise_or(horizontal, vertical)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, clean_kernel)

    contours, _ = cv2.findContours(
        merged_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_boxes: list[BoundingBox] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        bbox = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        if bbox.area < frame_area * 0.0003:
            continue

        if bbox.area > frame_area * 0.12:
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
        margin_ratio=0.08,
        max_iterations=1,
    )

    grouped_boxes = [
        box
        for box in grouped_boxes
        if box.area / frame_area <= 0.12
    ]

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
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 155]),
        np.array([180, 85, 255]),
    )

    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 50, 80]),
        np.array([40, 255, 255]),
    )

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
    color_mask = cv2.bitwise_or(red_mask, yellow_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

    white_clean = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, close_kernel)
    white_clean = cv2.morphologyEx(white_clean, cv2.MORPH_OPEN, open_kernel)

    color_clean = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel)
    color_clean = cv2.morphologyEx(color_clean, cv2.MORPH_OPEN, open_kernel)

    candidates: list[PriceTagCandidate] = []

    white_contours, _ = cv2.findContours(
        white_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in white_contours:
        x, y, w, h = cv2.boundingRect(contour)

        white_bbox = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        area_ratio = white_bbox.area / frame_area if frame_area > 0 else 0.0

        if area_ratio < 0.0005:
            continue

        if area_ratio > 0.08:
            continue

        if white_bbox.width < 35 or white_bbox.height < 20:
            continue

        if white_bbox.aspect_ratio < 0.25 or white_bbox.aspect_ratio > 5.8:
            continue

        white_ratio = _mask_ratio(white_mask, white_bbox)
        rectangularity = _estimate_rectangularity(contour)

        if white_ratio < 0.35:
            continue

        if rectangularity < 0.25:
            continue

        expanded_bbox, color_extension = _expand_box_to_adjacent_color_band(
            bbox=white_bbox,
            color_mask=color_clean,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        bottom_band_color = _estimate_bottom_band_color(
            bbox=expanded_bbox,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
        )

        side_band_color = _estimate_side_band_color(
            bbox=expanded_bbox,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
        )

        score = 0.10
        score += 0.18 * rectangularity
        score += 0.12 * min(white_ratio / 0.70, 1.0)

        if color_extension["extended"]:
            score += 0.08

        if bottom_band_color in {"red", "yellow"}:
            score += 0.05

        if side_band_color in {"red", "yellow"}:
            score += 0.04

        candidates.append(
            PriceTagCandidate(
                bbox=expanded_bbox,
                source="color_region",
                score=float(np.clip(score, 0.0, 0.40)),
                evidence={
                    "has_white_background": True,
                    "white_ratio": round(float(white_ratio), 4),
                    "bottom_band_color": bottom_band_color,
                    "side_band_color": side_band_color,
                    "color_extension": color_extension,
                    "rectangularity": round(float(rectangularity), 4),
                    "area_ratio": round(float(expanded_bbox.area / frame_area), 5),
                },
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

    candidates.extend(
        _candidates_from_standalone_color_bands(
            color_clean=color_clean,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_area=frame_area,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
    )

    return candidates


def _candidates_from_promo_color_layouts(
    frame: np.ndarray,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> list[PriceTagCandidate]:
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask_1 = cv2.inRange(
        hsv,
        np.array([0, 25, 75]),
        np.array([13, 255, 255]),
    )
    red_mask_2 = cv2.inRange(
        hsv,
        np.array([165, 25, 75]),
        np.array([180, 255, 255]),
    )
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 45, 85]),
        np.array([42, 255, 255]),
    )

    red_clean = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    red_clean = cv2.morphologyEx(
        red_clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
    )

    yellow_clean = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    yellow_clean = cv2.morphologyEx(
        yellow_clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
    )

    candidates: list[PriceTagCandidate] = []
    contour_items: list[tuple[str, np.ndarray]] = []

    for color_kind, mask in (("red", red_clean), ("yellow", yellow_clean)):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_items.extend((color_kind, contour) for contour in contours)

    for component_index, (component_color, contour) in enumerate(contour_items):
        x, y, w, h = cv2.boundingRect(contour)
        color_box = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        if color_box.width < 28 or color_box.height < 22:
            continue

        area_ratio = color_box.area / frame_area if frame_area > 0 else 0.0

        if area_ratio < 0.00025 or area_ratio > 0.035:
            continue

        color_aspect = color_box.aspect_ratio

        if color_aspect < 0.18 or color_aspect > 3.20:
            continue

        rectangularity = _estimate_rectangularity(contour)

        if rectangularity < 0.28:
            continue

        dominant_color = _estimate_dominant_color(
            bbox=color_box,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
        )

        if dominant_color == "none":
            dominant_color = component_color

        hypotheses = _generate_boxes_from_color_component(
            color_box=color_box,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        for hypothesis_index, bbox in enumerate(hypotheses):
            if bbox.width < 45 or bbox.height < 30:
                continue

            candidate_area_ratio = bbox.area / frame_area if frame_area > 0 else 0.0

            if candidate_area_ratio < 0.00045 or candidate_area_ratio > 0.09:
                continue

            if bbox.aspect_ratio < 0.25 or bbox.aspect_ratio > 5.80:
                continue

            color_coverage = color_box.area / max(1, bbox.area)

            if color_coverage < 0.12 or color_coverage > 0.72:
                continue

            score = 0.58
            score += 0.12 * rectangularity

            if dominant_color == "red":
                score += 0.06

            if 0.22 <= color_coverage <= 0.55:
                score += 0.08

            candidates.append(
                PriceTagCandidate(
                    bbox=bbox,
                    source="promo_color_layout",
                    score=float(np.clip(score, 0.0, 0.82)),
                    evidence={
                        "has_color_band": True,
                        "has_text_like_regions": True,
                        "color_band_kind": dominant_color,
                        "color_component_bbox": color_box.to_tuple(),
                        "layout_component_id": component_index,
                        "layout_hypothesis": hypothesis_index,
                        "rectangularity": round(float(rectangularity), 4),
                        "color_coverage": round(float(color_coverage), 4),
                        "area_ratio": round(float(candidate_area_ratio), 5),
                    },
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            )

    return candidates


def _generate_boxes_from_color_component(
    color_box: BoundingBox,
    frame_width: int,
    frame_height: int,
) -> list[BoundingBox]:
    x_min, y_min, x_max, y_max = color_box.to_tuple()
    w = color_box.width
    h = color_box.height

    raw_boxes = []

    horizontal_presets = [
        (-0.00, -0.00, 0.80, 0.00),
        (-0.00, -0.00, 0.90, 0.00),
        (-0.00, 0.10, 0.80, -0.10),
        (-0.10, 0.40, 0.90, -0.40),
        (-0.00, -0.00, 0.70, 0.00),
        (-0.00, -0.00, 0.80, 0.10),
        (-0.10, -0.00, 0.80, 0.00),
        (-0.10, 0.10, 0.90, -0.10),
        (-0.00, 0.20, 0.80, -0.20),
        (0.10, 0.10, 1.00, 0.10),
        (0.10, 0.10, 1.10, 0.10),
        (-0.10, 0.40, 0.90, -0.50),
        (-0.10, -0.10, 0.90, 0.10),
        (-0.00, -0.00, 0.90, 0.10),
        (-0.10, -0.20, 0.90, 0.00),
        (-0.10, -0.10, 0.80, 0.10),
        (0.10, -0.20, 1.00, 0.20),
        (0.10, -0.20, 1.10, 0.30),
        (-0.00, -0.20, 1.10, 0.20),
        (0.12, 0.10, 0.95, 0.06),
        (0.12, 0.10, 1.05, 0.06),
        (0.12, 0.10, 1.15, 0.06),
        (0.12, 0.10, 1.28, 0.06),
        (0.00, 0.06, 0.72, -0.10),
        (0.00, 0.06, 0.85, -0.10),
        (0.00, 0.05, 0.72, 0.05),
        (-0.07, -0.01, 0.75, -0.01),
        (0.00, 0.05, 0.85, 0.00),
        (0.00, 0.00, 0.76, 0.00),
        (0.08, 0.10, 0.72, -0.12),
        (0.25, 0.05, 1.00, 0.05),
        (0.20, 0.10, 1.05, 0.16),
    ]

    for left_pad, top_pad, right_ratio, bottom_pad in horizontal_presets:
        raw_boxes.append(
            BoundingBox(
                x_min=int(x_min - left_pad * w),
                y_min=int(y_min - top_pad * h),
                x_max=int(x_max + right_ratio * w),
                y_max=int(y_max + bottom_pad * h),
            )
        )

    for right_pad, top_pad, left_ratio, bottom_pad in horizontal_presets:
        raw_boxes.append(
            BoundingBox(
                x_min=int(x_min - left_ratio * w),
                y_min=int(y_min - top_pad * h),
                x_max=int(x_max + right_pad * w),
                y_max=int(y_max + bottom_pad * h),
            )
        )

    for bottom_ratio in (0.85, 1.05, 1.25):
        raw_boxes.append(
            BoundingBox(
                x_min=int(x_min - 0.08 * w),
                y_min=int(y_min - 0.10 * h),
                x_max=int(x_max + 0.08 * w),
                y_max=int(y_max + bottom_ratio * h),
            )
        )

    for top_ratio in (0.85, 1.05, 1.25):
        raw_boxes.append(
            BoundingBox(
                x_min=int(x_min - 0.08 * w),
                y_min=int(y_min - top_ratio * h),
                x_max=int(x_max + 0.08 * w),
                y_max=int(y_max + 0.10 * h),
            )
        )

    boxes = [
        box.clamp(frame_width, frame_height)
        for box in raw_boxes
    ]

    return _deduplicate_boxes(boxes, iou_threshold=0.995)


def _candidates_from_standalone_color_bands(
    color_clean: np.ndarray,
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
    frame_width: int,
    frame_height: int,
    frame_area: int,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> list[PriceTagCandidate]:
    contours, _ = cv2.findContours(
        color_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    result: list[PriceTagCandidate] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        band_box = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        if band_box.width < 35 or band_box.height < 12:
            continue

        area_ratio = band_box.area / frame_area if frame_area > 0 else 0.0

        if area_ratio < 0.0003:
            continue

        if area_ratio > 0.05:
            continue

        if band_box.aspect_ratio < 1.20:
            continue

        if band_box.aspect_ratio > 10.0:
            continue

        rectangularity = _estimate_rectangularity(contour)

        if rectangularity < 0.20:
            continue

        color_kind = _estimate_dominant_color(
            bbox=band_box,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
        )

        if color_kind == "none":
            continue

        expanded = band_box.expand(
            frame_width=frame_width,
            frame_height=frame_height,
            left=0.10,
            right=0.10,
            top=1.40,
            bottom=0.15,
        )

        score = 0.18 + 0.12 * rectangularity

        result.append(
            PriceTagCandidate(
                bbox=expanded,
                source="color_band_region",
                score=float(np.clip(score, 0.0, 0.34)),
                evidence={
                    "has_color_band": True,
                    "color_band_kind": color_kind,
                    "color_band_bbox": band_box.to_tuple(),
                    "rectangularity": round(float(rectangularity), 4),
                    "area_ratio": round(float(expanded.area / frame_area), 5),
                },
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )

    return result


def _expand_box_to_adjacent_color_band(
    bbox: BoundingBox,
    color_mask: np.ndarray,
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[BoundingBox, dict]:
    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    result_box = bbox
    extensions: list[dict] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        color_box = BoundingBox(x, y, x + w, y + h).clamp(frame_width, frame_height)

        if color_box.area <= 0:
            continue

        if color_box.width < 15 or color_box.height < 8:
            continue

        horizontal_overlap = _axis_overlap(
            bbox.x_min,
            bbox.x_max,
            color_box.x_min,
            color_box.x_max,
        )

        vertical_overlap = _axis_overlap(
            bbox.y_min,
            bbox.y_max,
            color_box.y_min,
            color_box.y_max,
        )

        horizontal_coverage = horizontal_overlap / max(1, bbox.width)
        vertical_coverage = vertical_overlap / max(1, bbox.height)

        gap_bottom = color_box.y_min - bbox.y_max
        gap_top = bbox.y_min - color_box.y_max
        gap_right = color_box.x_min - bbox.x_max
        gap_left = bbox.x_min - color_box.x_max

        max_vertical_gap = max(8, int(bbox.height * 0.25))
        max_horizontal_gap = max(8, int(bbox.width * 0.15))

        is_bottom_band = (
            gap_bottom <= max_vertical_gap
            and color_box.y_max >= bbox.y_min + int(bbox.height * 0.45)
            and horizontal_coverage >= 0.55
            and color_box.aspect_ratio >= 1.0
        )

        is_top_band = (
            gap_top <= max_vertical_gap
            and color_box.y_min <= bbox.y_max - int(bbox.height * 0.45)
            and horizontal_coverage >= 0.55
            and color_box.aspect_ratio >= 1.0
        )

        is_right_band = (
            gap_right <= max_horizontal_gap
            and color_box.x_max >= bbox.x_min + int(bbox.width * 0.45)
            and vertical_coverage >= 0.45
            and color_box.height >= color_box.width * 0.5
        )

        is_left_band = (
            gap_left <= max_horizontal_gap
            and color_box.x_min <= bbox.x_max - int(bbox.width * 0.45)
            and vertical_coverage >= 0.45
            and color_box.height >= color_box.width * 0.5
        )

        if not (is_bottom_band or is_top_band or is_right_band or is_left_band):
            continue

        color_kind = _estimate_dominant_color(
            bbox=color_box,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
        )

        if color_kind == "none":
            continue

        candidate_union = result_box.union(color_box).clamp(frame_width, frame_height)

        if candidate_union.area > bbox.area * 2.8:
            continue

        result_box = candidate_union

        if is_bottom_band:
            side = "bottom"
        elif is_top_band:
            side = "top"
        elif is_right_band:
            side = "right"
        else:
            side = "left"

        extensions.append(
            {
                "side": side,
                "color": color_kind,
                "bbox": color_box.to_tuple(),
            }
        )

    return result_box, {
        "extended": bool(extensions),
        "extensions": extensions,
    }


def _axis_overlap(
    first_min: int,
    first_max: int,
    second_min: int,
    second_max: int,
) -> int:
    return max(0, min(first_max, second_max) - max(first_min, second_min))


def _estimate_dominant_color(
    bbox: BoundingBox,
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
) -> str:
    red_ratio = _mask_ratio(red_mask, bbox)
    yellow_ratio = _mask_ratio(yellow_mask, bbox)

    if red_ratio < 0.05 and yellow_ratio < 0.05:
        return "none"

    if red_ratio >= yellow_ratio:
        return "red"

    return "yellow"


def _estimate_bottom_band_color(
    bbox: BoundingBox,
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
) -> str:
    if bbox.height <= 0 or bbox.width <= 0:
        return "none"

    y_mid = bbox.y_min + int(bbox.height * 0.50)

    bottom_box = BoundingBox(
        x_min=bbox.x_min,
        y_min=y_mid,
        x_max=bbox.x_max,
        y_max=bbox.y_max,
    )

    red_ratio = _mask_ratio(red_mask, bottom_box)
    yellow_ratio = _mask_ratio(yellow_mask, bottom_box)

    if red_ratio >= 0.08 and red_ratio >= yellow_ratio:
        return "red"

    if yellow_ratio >= 0.08 and yellow_ratio > red_ratio:
        return "yellow"

    return "none"


def _estimate_side_band_color(
    bbox: BoundingBox,
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
) -> str:
    if bbox.height <= 0 or bbox.width <= 0:
        return "none"

    left_box = BoundingBox(
        x_min=bbox.x_min,
        y_min=bbox.y_min,
        x_max=bbox.x_min + int(bbox.width * 0.35),
        y_max=bbox.y_max,
    )

    right_box = BoundingBox(
        x_min=bbox.x_max - int(bbox.width * 0.35),
        y_min=bbox.y_min,
        x_max=bbox.x_max,
        y_max=bbox.y_max,
    )

    left_color = _estimate_dominant_color(left_box, red_mask, yellow_mask)
    right_color = _estimate_dominant_color(right_box, red_mask, yellow_mask)

    if left_color != "none":
        return left_color

    if right_color != "none":
        return right_color

    return "none"


def _boost_rectangles_with_text(
    candidates: list[PriceTagCandidate],
) -> list[PriceTagCandidate]:
    rectangles = [
        candidate
        for candidate in candidates
        if candidate.source == "rectangular_region"
    ]

    text_regions = [
        candidate
        for candidate in candidates
        if candidate.source == "text_like_region"
    ]

    if not rectangles or not text_regions:
        return candidates

    result: list[PriceTagCandidate] = []

    for candidate in candidates:
        if candidate.source != "rectangular_region":
            result.append(candidate)
            continue

        matched_texts: list[PriceTagCandidate] = []

        for text_candidate in text_regions:
            text_inside_rectangle = candidate.bbox.contains_ratio(text_candidate.bbox) >= 0.45
            text_overlaps_rectangle = candidate.bbox.iou(text_candidate.bbox) >= 0.10

            if text_inside_rectangle or text_overlaps_rectangle:
                matched_texts.append(text_candidate)

        if not matched_texts:
            result.append(candidate)
            continue

        max_text_density = max(
            float(text.evidence.get("text_density", 0.0) or 0.0)
            for text in matched_texts
        )

        evidence = dict(candidate.evidence)
        evidence["has_text_like_regions"] = True
        evidence["text_density"] = max(
            float(evidence.get("text_density", 0.0) or 0.0),
            max_text_density,
        )
        evidence["text_regions_inside"] = len(matched_texts)

        boost = 0.15 + 0.20 * min(max_text_density, 1.0)

        result.append(
            PriceTagCandidate(
                bbox=candidate.bbox,
                source="mixed",
                score=float(np.clip(candidate.score + boost, 0.0, 1.0)),
                evidence=evidence,
                frame_index=candidate.frame_index,
                timestamp_ms=candidate.timestamp_ms,
            )
        )

    return result


def _boost_candidates_with_codes(
    candidates: list[PriceTagCandidate],
    codes: list[CodeDetection],
) -> list[PriceTagCandidate]:
    if not codes:
        return candidates

    result: list[PriceTagCandidate] = []

    for candidate in candidates:
        matched_codes: list[CodeDetection] = []

        for code in codes:
            contains_code = candidate.bbox.contains_ratio(code.bbox) >= 0.65
            overlaps_code = candidate.bbox.iou(code.bbox) >= 0.10

            if contains_code or overlaps_code:
                matched_codes.append(code)

        if not matched_codes:
            result.append(candidate)
            continue

        evidence = dict(candidate.evidence)

        code_values = [
            code.value
            for code in matched_codes
            if code.value
        ]

        has_qr = any(code.code_type == "qr" for code in matched_codes)
        has_barcode = any(code.code_type == "barcode" for code in matched_codes)

        evidence["has_qr"] = bool(evidence.get("has_qr", False) or has_qr)
        evidence["has_barcode"] = bool(evidence.get("has_barcode", False) or has_barcode)

        old_values = evidence.get("code_values", [])

        if isinstance(old_values, str):
            old_values = [old_values]

        if not isinstance(old_values, list):
            old_values = []

        evidence["code_values"] = list(dict.fromkeys(old_values + code_values))
        evidence["code_inside_candidate"] = True

        boost = 0.22

        if has_qr:
            boost += 0.08

        if has_barcode:
            boost += 0.05

        if code_values:
            boost += 0.08

        result.append(
            PriceTagCandidate(
                bbox=candidate.bbox,
                source="mixed" if candidate.source != "mixed" else candidate.source,
                score=float(np.clip(candidate.score + boost, 0.0, 1.0)),
                evidence=evidence,
                frame_index=candidate.frame_index,
                timestamp_ms=candidate.timestamp_ms,
            )
        )

    return result


def _filter_final_candidates(
    candidates: list[PriceTagCandidate],
    min_candidate_score: float,
) -> list[PriceTagCandidate]:
    result: list[PriceTagCandidate] = []

    for candidate in candidates:
        evidence = candidate.evidence

        has_qr = bool(evidence.get("has_qr", False))
        has_barcode = bool(evidence.get("has_barcode", False))
        has_code = has_qr or has_barcode or "anchor" in candidate.source

        has_text = bool(evidence.get("has_text_like_regions", False))
        text_density = float(evidence.get("text_density", 0.0) or 0.0)

        has_rectangle = bool(evidence.get("has_rectangular_shape", False))
        rectangularity = float(evidence.get("rectangularity", 0.0) or 0.0)
        edge_density = float(evidence.get("edge_density", 0.0) or 0.0)
        area_ratio = _evidence_float(evidence.get("area_ratio"), default=0.0)

        is_color_only = candidate.source in {"color_region", "color_band_region"}
        is_text = candidate.source == "text_like_region"
        is_rectangle = candidate.source == "rectangular_region"
        is_mixed = candidate.source == "mixed"
        is_promo_layout = candidate.source == "promo_color_layout"

        if is_promo_layout:
            if candidate.score >= 0.55 and has_text:
                result.append(candidate)

            continue

        if not has_code and area_ratio > 0.07:
            continue

        if has_code and candidate.score >= 0.45:
            result.append(candidate)
            continue

        if is_mixed:
            has_enough_structure = (
                has_code
                or (has_rectangle and has_text and text_density >= 0.12)
                or (has_text and candidate.score >= 0.62)
            )

            if candidate.score >= min_candidate_score and has_enough_structure:
                result.append(candidate)

            continue

        if is_rectangle:
            if not has_code and not has_text:
                continue

            if has_text:
                if (
                    candidate.score >= 0.55
                    and text_density >= 0.18
                    and rectangularity >= 0.35
                    and edge_density >= 0.025
                ):
                    result.append(candidate)

            continue

        if is_text:
            if (
                candidate.score >= 0.62
                and text_density >= 0.35
            ):
                result.append(candidate)

            continue

        if is_color_only:
            continue

        if candidate.score >= 0.75 and (has_code or has_text):
            result.append(candidate)

    result.sort(key=lambda item: item.score, reverse=True)

    return result


def _select_diverse_candidates(
    candidates: list[PriceTagCandidate],
    max_candidates: int | None,
    iou_threshold: float = 0.50,
    containment_threshold: float = 0.82,
) -> list[PriceTagCandidate]:
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=_candidate_ranking_key,
        reverse=True,
    )

    selected: list[PriceTagCandidate] = []
    used_layout_components: set[tuple[int, tuple[int, int, int, int] | None]] = set()

    for candidate in sorted_candidates:
        layout_key = _layout_component_key(candidate)

        if layout_key is not None and layout_key in used_layout_components:
            continue

        if any(_is_duplicate_output(candidate, existing, iou_threshold, containment_threshold) for existing in selected):
            continue

        selected.append(candidate)

        if layout_key is not None:
            used_layout_components.add(layout_key)

        if max_candidates is not None and len(selected) >= max_candidates:
            break

    selected.sort(key=lambda item: item.score, reverse=True)

    return selected


def _candidate_ranking_key(candidate: PriceTagCandidate) -> tuple[float, float, int]:
    evidence = candidate.evidence

    source_bonus = 0.0

    if candidate.source == "promo_color_layout":
        source_bonus += 0.12

    if candidate.source == "mixed":
        source_bonus += 0.08

    if evidence.get("has_qr", False) or evidence.get("has_barcode", False):
        source_bonus += 0.10

    area = candidate.bbox.area

    return (
        candidate.score + source_bonus,
        float(evidence.get("rectangularity", 0.0) or 0.0),
        area,
    )


def _evidence_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, list):
        numeric_values = [
            float(item)
            for item in value
            if isinstance(item, (int, float))
        ]

        if numeric_values:
            return max(numeric_values)

    return default


def _layout_component_key(
    candidate: PriceTagCandidate,
) -> tuple[int, tuple[int, int, int, int] | None] | None:
    component_id = candidate.evidence.get("layout_component_id")

    if component_id is None:
        return None

    try:
        normalized_component_id = int(component_id)
    except Exception:
        return None

    component_bbox = candidate.evidence.get("color_component_bbox")
    normalized_bbox: tuple[int, int, int, int] | None = None

    if isinstance(component_bbox, (tuple, list)) and len(component_bbox) == 4:
        try:
            normalized_bbox = tuple(int(item) for item in component_bbox)
        except Exception:
            normalized_bbox = None

    return normalized_component_id, normalized_bbox


def _is_duplicate_output(
    candidate: PriceTagCandidate,
    existing: PriceTagCandidate,
    iou_threshold: float,
    containment_threshold: float,
) -> bool:
    if candidate.bbox.iou(existing.bbox) >= iou_threshold:
        return True

    candidate_contains_existing = candidate.bbox.contains_ratio(existing.bbox)
    existing_contains_candidate = existing.bbox.contains_ratio(candidate.bbox)

    return (
        candidate_contains_existing >= containment_threshold
        or existing_contains_candidate >= containment_threshold
    )


def _deduplicate_codes(
    codes: list[CodeDetection],
    iou_threshold: float = 0.50,
) -> list[CodeDetection]:
    if not codes:
        return []

    codes = sorted(codes, key=lambda item: item.confidence, reverse=True)

    result: list[CodeDetection] = []

    for code in codes:
        duplicate = False

        for existing in result:
            same_type = code.code_type == existing.code_type
            same_value = code.value and existing.value and code.value == existing.value
            high_overlap = code.bbox.iou(existing.bbox) >= iou_threshold

            if same_type and (same_value or high_overlap):
                duplicate = True
                break

        if not duplicate:
            result.append(code)

    return result


def _group_near_boxes(
    boxes: list[BoundingBox],
    frame_width: int,
    frame_height: int,
    margin_ratio: float = 0.08,
    max_iterations: int = 1,
) -> list[BoundingBox]:
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

    density = np.mean(crop > 0)

    if density <= 0.0:
        return 0.0

    if density > 0.75:
        return 0.15

    return float(np.clip(density / 0.35, 0.0, 1.0))


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


def _estimate_edge_density(
    edges: np.ndarray,
    bbox: BoundingBox,
) -> float:
    crop = edges[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

    if crop.size == 0:
        return 0.0

    return float(np.mean(crop > 0))


def _estimate_size_score(area_ratio: float) -> float:
    if area_ratio <= 0:
        return 0.0

    if area_ratio < 0.001:
        return 0.35

    if area_ratio < 0.01:
        return 1.0

    if area_ratio < 0.04:
        return 0.75

    if area_ratio < 0.08:
        return 0.35

    return 0.0


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
    debug = frame.copy()

    for idx, candidate in enumerate(result.candidates):
        x_min, y_min, x_max, y_max = candidate.bbox.to_tuple()

        if candidate.source == "mixed":
            color = (0, 255, 0)
        elif "anchor" in candidate.source:
            color = (0, 200, 255)
        elif candidate.source == "text_like_region":
            color = (0, 255, 255)
        elif candidate.source == "rectangular_region":
            color = (255, 255, 0)
        elif candidate.source == "color_band_region":
            color = (0, 128, 255)
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

    for code_index, code in enumerate(result.codes):
        x_min, y_min, x_max, y_max = code.bbox.to_tuple()

        if code.code_type == "qr":
            color = (255, 0, 0)
        else:
            color = (255, 0, 255)

        cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), color, 4)

        short_value = code.value[:20] if code.value else "no-value"
        label = f"CODE {code_index}:{code.code_type}:{code.confidence:.2f}:{short_value}"

        label_y = max(22, y_min - 10)

        cv2.rectangle(
            debug,
            (x_min, label_y - 18),
            (min(frame.shape[1] - 1, x_min + 420), label_y + 5),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            debug,
            label,
            (x_min, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), debug)
