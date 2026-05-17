from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def clamp(self, width: int, height: int) -> "BoundingBox":
        if width <= 0 or height <= 0:
            return BoundingBox(0, 0, 0, 0)

        x_min = int(np.clip(self.x_min, 0, width - 1))
        y_min = int(np.clip(self.y_min, 0, height - 1))
        x_max = int(np.clip(self.x_max, x_min + 1, width))
        y_max = int(np.clip(self.y_max, y_min + 1, height))

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    def expand(
        self,
        frame_width: int,
        frame_height: int,
        left: float = 0.0,
        top: float = 0.0,
        right: float = 0.0,
        bottom: float = 0.0,
    ) -> "BoundingBox":
        w = self.width
        h = self.height

        expanded = BoundingBox(
            x_min=int(self.x_min - left * w),
            y_min=int(self.y_min - top * h),
            x_max=int(self.x_max + right * w),
            y_max=int(self.y_max + bottom * h),
        )

        return expanded.clamp(frame_width, frame_height)

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            x_min=min(self.x_min, other.x_min),
            y_min=min(self.y_min, other.y_min),
            x_max=max(self.x_max, other.x_max),
            y_max=max(self.y_max, other.y_max),
        )

    def intersection_area(self, other: "BoundingBox") -> int:
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)

        if x_right <= x_left or y_bottom <= y_top:
            return 0

        return (x_right - x_left) * (y_bottom - y_top)

    def iou(self, other: "BoundingBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter

        if union <= 0:
            return 0.0

        return inter / union

    def contains_ratio(self, other: "BoundingBox") -> float:
        if other.area <= 0:
            return 0.0

        return self.intersection_area(other) / other.area


@dataclass
class PriceTagCandidate:
    bbox: BoundingBox
    source: str
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)
    frame_index: int | None = None
    timestamp_ms: float | None = None


def merge_candidates(
    candidates: list[PriceTagCandidate],
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.85,
    max_candidates: int | None = 80,
) -> list[PriceTagCandidate]:
    if not candidates:
        return []

    prepared = [
        candidate
        for candidate in candidates
        if candidate.bbox.area > 0 and candidate.score > 0
    ]

    prepared.sort(key=lambda item: item.score, reverse=True)

    merged: list[PriceTagCandidate] = []

    for candidate in prepared:
        if _has_layout_hypothesis(candidate):
            merged.append(candidate)
            continue

        matched_index = None

        for index, existing in enumerate(merged):
            if _should_merge(
                existing,
                candidate,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
            ):
                matched_index = index
                break

        if matched_index is None:
            merged.append(candidate)
        else:
            merged[matched_index] = _merge_two_candidates(
                merged[matched_index],
                candidate,
            )

    merged.sort(key=lambda item: item.score, reverse=True)

    if max_candidates is not None:
        merged = merged[:max_candidates]

    return merged


def filter_candidates_by_geometry(
    candidates: list[PriceTagCandidate],
    frame_width: int,
    frame_height: int,
    min_area_ratio: float = 0.0003,
    max_area_ratio: float = 0.18,
    min_aspect_ratio: float = 0.20,
    max_aspect_ratio: float = 6.00,
) -> list[PriceTagCandidate]:
    frame_area = frame_width * frame_height

    if frame_area <= 0:
        return []

    result: list[PriceTagCandidate] = []

    for candidate in candidates:
        area_ratio = candidate.bbox.area / frame_area
        aspect_ratio = candidate.bbox.aspect_ratio

        if area_ratio < min_area_ratio:
            continue

        if area_ratio > max_area_ratio:
            continue

        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        result.append(candidate)

    return result


def _should_merge(
    first: PriceTagCandidate,
    second: PriceTagCandidate,
    iou_threshold: float,
    containment_threshold: float,
) -> bool:
    if _has_conflicting_codes(first, second):
        return False

    if _are_different_anchor_hypotheses(first, second):
        return False

    if _are_different_layout_hypotheses(first, second):
        return False

    if _has_layout_hypothesis(first) or _has_layout_hypothesis(second):
        return False

    iou = first.bbox.iou(second.bbox)

    first_contains_second = first.bbox.contains_ratio(second.bbox)
    second_contains_first = second.bbox.contains_ratio(first.bbox)

    has_strong_overlap = iou >= iou_threshold

    has_containment = (
        first_contains_second >= containment_threshold
        or second_contains_first >= containment_threshold
    )

    if not has_strong_overlap and not has_containment:
        return False

    if _union_growth_is_too_large(first, second):
        return False

    return True


def _merge_two_candidates(
    first: PriceTagCandidate,
    second: PriceTagCandidate,
) -> PriceTagCandidate:
    bbox = first.bbox.union(second.bbox)

    source = first.source if first.source == second.source else "mixed"

    evidence = _merge_evidence(first.evidence, second.evidence)

    score = max(first.score, second.score)

    if first.source != second.source:
        score += 0.08
    else:
        score += 0.03

    score = float(np.clip(score, 0.0, 1.0))

    frame_index = first.frame_index if first.frame_index is not None else second.frame_index
    timestamp_ms = first.timestamp_ms if first.timestamp_ms is not None else second.timestamp_ms

    return PriceTagCandidate(
        bbox=bbox,
        source=source,
        score=score,
        evidence=evidence,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )


def _merge_evidence(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    result = dict(first)

    for key, value in second.items():
        if key not in result:
            result[key] = value
            continue

        old_value = result[key]

        if isinstance(old_value, bool) and isinstance(value, bool):
            result[key] = old_value or value
        elif isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
            result[key] = max(old_value, value)
        elif isinstance(old_value, list) and isinstance(value, list):
            result[key] = _unique_list(old_value + value)
        elif old_value == value:
            result[key] = old_value
        else:
            old_list = old_value if isinstance(old_value, list) else [old_value]
            value_list = value if isinstance(value, list) else [value]
            result[key] = _unique_list(old_list + value_list)

    return result


def _unique_list(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()

    for value in values:
        marker = repr(value)

        if marker in seen:
            continue

        seen.add(marker)
        result.append(value)

    return result


def _extract_codes(candidate: PriceTagCandidate) -> set[str]:
    values = candidate.evidence.get("code_values", [])

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return set()

    return {
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    }


def _extract_anchor_bbox(candidate: PriceTagCandidate) -> tuple[int, int, int, int] | None:
    value = candidate.evidence.get("anchor_bbox")

    if not isinstance(value, (list, tuple)):
        return None

    if len(value) != 4:
        return None

    try:
        return tuple(int(item) for item in value)
    except Exception:
        return None


def _has_same_non_empty_code(
    first: PriceTagCandidate,
    second: PriceTagCandidate,
) -> bool:
    first_codes = _extract_codes(first)
    second_codes = _extract_codes(second)

    if not first_codes or not second_codes:
        return False

    return bool(first_codes & second_codes)


def _has_conflicting_codes(
    first: PriceTagCandidate,
    second: PriceTagCandidate,
) -> bool:
    first_codes = _extract_codes(first)
    second_codes = _extract_codes(second)

    if not first_codes or not second_codes:
        return False

    return not bool(first_codes & second_codes)


def _are_different_anchor_hypotheses(
    first: PriceTagCandidate,
    second: PriceTagCandidate,
) -> bool:
    first_is_anchor = "anchor" in first.source
    second_is_anchor = "anchor" in second.source

    if not first_is_anchor or not second_is_anchor:
        return False

    first_hypothesis = first.evidence.get("anchor_hypothesis")
    second_hypothesis = second.evidence.get("anchor_hypothesis")

    if first_hypothesis is None or second_hypothesis is None:
        return False

    if first_hypothesis == second_hypothesis:
        return False

    if _has_same_non_empty_code(first, second):
        return True

    first_anchor_bbox = _extract_anchor_bbox(first)
    second_anchor_bbox = _extract_anchor_bbox(second)

    if first_anchor_bbox is not None and first_anchor_bbox == second_anchor_bbox:
        return True

    return False


def _are_different_layout_hypotheses(
    first: PriceTagCandidate,
    second: PriceTagCandidate,
) -> bool:
    first_component = first.evidence.get("layout_component_id")
    second_component = second.evidence.get("layout_component_id")

    if first_component is None or second_component is None:
        return False

    if first_component != second_component:
        return False

    first_hypothesis = first.evidence.get("layout_hypothesis")
    second_hypothesis = second.evidence.get("layout_hypothesis")

    if first_hypothesis is None or second_hypothesis is None:
        return False

    return first_hypothesis != second_hypothesis


def _has_layout_hypothesis(candidate: PriceTagCandidate) -> bool:
    return (
        candidate.evidence.get("layout_component_id") is not None
        and candidate.evidence.get("layout_hypothesis") is not None
    )


def _union_growth_is_too_large(
    first: PriceTagCandidate,
    second: PriceTagCandidate,
    max_growth_ratio: float = 1.8,
) -> bool:
    union_box = first.bbox.union(second.bbox)

    max_original_area = max(first.bbox.area, second.bbox.area)

    if max_original_area <= 0:
        return True

    growth_ratio = union_box.area / max_original_area

    return growth_ratio > max_growth_ratio
