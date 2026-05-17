from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from cv_module.detection.candidate_merger import BoundingBox


@dataclass(frozen=True)
class Detection:
    frame_index: int
    timestamp_ms: float
    candidate_index: int
    bbox: BoundingBox
    source: str
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def first_detection(self) -> Detection:
        return self.detections[0]

    @property
    def last_detection(self) -> Detection:
        return self.detections[-1]

    @property
    def start_frame(self) -> int:
        return self.first_detection.frame_index

    @property
    def end_frame(self) -> int:
        return self.last_detection.frame_index

    @property
    def start_timestamp_ms(self) -> float:
        return self.first_detection.timestamp_ms

    @property
    def end_timestamp_ms(self) -> float:
        return self.last_detection.timestamp_ms

    @property
    def best_detection(self) -> Detection:
        return max(self.detections, key=_best_detection_score)

    @property
    def smoothed_bbox(self) -> BoundingBox:
        recent = self.detections[-3:]

        return BoundingBox(
            x_min=round(sum(item.bbox.x_min for item in recent) / len(recent)),
            y_min=round(sum(item.bbox.y_min for item in recent) / len(recent)),
            x_max=round(sum(item.bbox.x_max for item in recent) / len(recent)),
            y_max=round(sum(item.bbox.y_max for item in recent) / len(recent)),
        )

    def predicted_bbox(self, frame_index: int) -> BoundingBox:
        if len(self.detections) < 2:
            return self.smoothed_bbox

        previous = self.detections[-2]
        last = self.detections[-1]
        frame_gap = max(1, last.frame_index - previous.frame_index)
        target_gap = max(0, frame_index - last.frame_index)
        multiplier = min(2.5, target_gap / frame_gap)

        return BoundingBox(
            x_min=round(last.bbox.x_min + (last.bbox.x_min - previous.bbox.x_min) * multiplier),
            y_min=round(last.bbox.y_min + (last.bbox.y_min - previous.bbox.y_min) * multiplier),
            x_max=round(last.bbox.x_max + (last.bbox.x_max - previous.bbox.x_max) * multiplier),
            y_max=round(last.bbox.y_max + (last.bbox.y_max - previous.bbox.y_max) * multiplier),
        )


@dataclass(frozen=True)
class TrackConfig:
    min_score: float = 0.50
    min_width: int = 55
    min_height: int = 45
    min_area: int = 2_000
    max_area: int = 500_000
    min_aspect_ratio: float = 0.18
    max_aspect_ratio: float = 7.0
    nms_iou_threshold: float = 0.50
    nms_containment_threshold: float = 0.86
    max_frame_gap: int = 36
    min_match_score: float = 0.27
    min_iou_for_match: float = 0.04
    max_center_distance_factor: float = 3.2
    min_detections_per_track: int = 2
    max_detections_per_frame: int = 40
    merge_track_gap: int = 72
    merge_match_score: float = 0.34
    frame_width: int = 3840
    frame_height: int = 2160
    edge_margin: int = 180
    edge_max_frame_gap: int = 72


class PriceTagTracker:
    def __init__(self, config: TrackConfig | None = None) -> None:
        self.config = config or TrackConfig()

    def track(self, detections: list[Detection]) -> list[Track]:
        by_frame = _group_by_frame(self._prefilter(detections))
        active_tracks: list[Track] = []
        closed_tracks: list[Track] = []
        next_track_id = 1

        for frame_index in sorted(by_frame):
            frame_detections = _nms(by_frame[frame_index], self.config)[
                : self.config.max_detections_per_frame
            ]
            active_tracks, expired_tracks = self._split_active_tracks(
                tracks=active_tracks,
                frame_index=frame_index,
            )
            closed_tracks.extend(expired_tracks)

            matched_track_ids: set[int] = set()
            matched_detection_ids: set[int] = set()
            matches = self._match_tracks(active_tracks, frame_detections)

            for track, detection, _ in matches:
                if track.track_id in matched_track_ids:
                    continue

                detection_id = id(detection)

                if detection_id in matched_detection_ids:
                    continue

                track.detections.append(detection)
                matched_track_ids.add(track.track_id)
                matched_detection_ids.add(detection_id)

            for detection in frame_detections:
                if id(detection) in matched_detection_ids:
                    continue

                active_tracks.append(
                    Track(
                        track_id=next_track_id,
                        detections=[detection],
                    )
                )
                next_track_id += 1

        closed_tracks.extend(active_tracks)
        closed_tracks = _merge_fragmented_tracks(closed_tracks, self.config)
        closed_tracks.sort(key=lambda item: (item.start_frame, item.track_id))
        _renumber_tracks(closed_tracks)

        return [
            track
            for track in closed_tracks
            if len(track.detections) >= self.config.min_detections_per_track
        ]

    def _prefilter(self, detections: list[Detection]) -> list[Detection]:
        result: list[Detection] = []

        for detection in detections:
            bbox = detection.bbox

            if detection.score < self.config.min_score:
                continue

            if bbox.width < self.config.min_width or bbox.height < self.config.min_height:
                continue

            if bbox.area < self.config.min_area or bbox.area > self.config.max_area:
                continue

            if not self.config.min_aspect_ratio <= bbox.aspect_ratio <= self.config.max_aspect_ratio:
                continue

            result.append(detection)

        return result

    def _split_active_tracks(
        self,
        tracks: list[Track],
        frame_index: int,
    ) -> tuple[list[Track], list[Track]]:
        active: list[Track] = []
        expired: list[Track] = []

        for track in tracks:
            max_gap = self.config.max_frame_gap

            if _is_near_frame_edge(track.last_detection.bbox, self.config):
                max_gap = max(max_gap, self.config.edge_max_frame_gap)

            if frame_index - track.last_detection.frame_index <= max_gap:
                active.append(track)
            else:
                expired.append(track)

        return active, expired

    def _match_tracks(
        self,
        tracks: list[Track],
        detections: list[Detection],
    ) -> list[tuple[Track, Detection, float]]:
        candidates: list[tuple[Track, Detection, float]] = []

        for track in tracks:
            for detection in detections:
                score = max(
                    _match_score(
                        predicted_bbox=track.predicted_bbox(detection.frame_index),
                        detection_bbox=detection.bbox,
                        config=self.config,
                    ),
                    _match_score(
                        predicted_bbox=track.smoothed_bbox,
                        detection_bbox=detection.bbox,
                        config=self.config,
                    ),
                )

                if score < self.config.min_match_score:
                    continue

                candidates.append((track, detection, score))

        candidates.sort(key=lambda item: item[2], reverse=True)

        return candidates


def read_detection_report(path: Path) -> list[Detection]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [_detection_from_row(row) for row in reader]


def tracks_to_rows(tracks: list[Track]) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []

    for track in tracks:
        best = track.best_detection
        bbox = best.bbox

        rows.append(
            {
                "track_id": track.track_id,
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "start_timestamp_ms": round(track.start_timestamp_ms, 2),
                "end_timestamp_ms": round(track.end_timestamp_ms, 2),
                "duration_ms": round(track.end_timestamp_ms - track.start_timestamp_ms, 2),
                "detections_count": len(track.detections),
                "best_frame_index": best.frame_index,
                "best_timestamp_ms": round(best.timestamp_ms, 2),
                "best_candidate_index": best.candidate_index,
                "best_detection_score": round(best.score, 4),
                "best_track_score": round(_best_detection_score(best), 4),
                "source": best.source,
                "x_min": bbox.x_min,
                "y_min": bbox.y_min,
                "x_max": bbox.x_max,
                "y_max": bbox.y_max,
                "width": bbox.width,
                "height": bbox.height,
                "aspect_ratio": round(bbox.aspect_ratio, 4),
            }
        )

    return rows


def _group_by_frame(detections: list[Detection]) -> dict[int, list[Detection]]:
    by_frame: dict[int, list[Detection]] = {}

    for detection in detections:
        by_frame.setdefault(detection.frame_index, []).append(detection)

    return by_frame


def _nms(detections: list[Detection], config: TrackConfig) -> list[Detection]:
    kept: list[Detection] = []

    for detection in sorted(detections, key=_best_detection_score, reverse=True):
        if any(_should_suppress(detection, existing, config) for existing in kept):
            continue

        kept.append(detection)

    return kept


def _should_suppress(first: Detection, second: Detection, config: TrackConfig) -> bool:
    if first.bbox.iou(second.bbox) >= config.nms_iou_threshold:
        return True

    if first.bbox.contains_ratio(second.bbox) >= config.nms_containment_threshold:
        return True

    if second.bbox.contains_ratio(first.bbox) >= config.nms_containment_threshold:
        return True

    return False


def _match_score(
    predicted_bbox: BoundingBox,
    detection_bbox: BoundingBox,
    config: TrackConfig,
) -> float:
    iou = predicted_bbox.iou(detection_bbox)
    near_edge = (
        _is_near_frame_edge(predicted_bbox, config)
        or _is_near_frame_edge(detection_bbox, config)
    )
    center_distance_factor = config.max_center_distance_factor

    if near_edge:
        center_distance_factor *= 1.9

    center_similarity = _center_similarity(
        predicted_bbox,
        detection_bbox,
        center_distance_factor,
    )
    size_similarity = _size_similarity(predicted_bbox, detection_bbox)
    center_threshold = 0.38 if near_edge else 0.55
    min_iou = 0.0 if near_edge else config.min_iou_for_match

    if iou < min_iou and center_similarity < center_threshold:
        return 0.0

    if near_edge:
        return 0.45 * iou + 0.45 * center_similarity + 0.10 * size_similarity

    return 0.58 * iou + 0.30 * center_similarity + 0.12 * size_similarity


def _merge_fragmented_tracks(
    tracks: list[Track],
    config: TrackConfig,
) -> list[Track]:
    merged = sorted(tracks, key=lambda item: (item.start_frame, item.track_id))
    changed = True

    while changed:
        changed = False
        consumed: set[int] = set()
        result: list[Track] = []

        for index, track in enumerate(merged):
            if index in consumed:
                continue

            best_index = -1
            best_score = 0.0

            for candidate_index, candidate in enumerate(merged):
                if candidate_index == index or candidate_index in consumed:
                    continue

                gap = candidate.start_frame - track.end_frame

                if gap <= 0 or gap > config.merge_track_gap:
                    continue

                score = max(
                    _match_score(
                        predicted_bbox=track.predicted_bbox(candidate.start_frame),
                        detection_bbox=candidate.first_detection.bbox,
                        config=config,
                    ),
                    _match_score(
                        predicted_bbox=track.smoothed_bbox,
                        detection_bbox=candidate.first_detection.bbox,
                        config=config,
                    ),
                )

                if score <= best_score:
                    continue

                best_index = candidate_index
                best_score = score

            if best_index >= 0 and best_score >= config.merge_match_score:
                track.detections.extend(merged[best_index].detections)
                track.detections.sort(key=lambda item: (item.frame_index, item.candidate_index))
                consumed.add(best_index)
                changed = True

            result.append(track)

        merged = result

    return merged


def _renumber_tracks(tracks: list[Track]) -> None:
    for new_id, track in enumerate(tracks, start=1):
        track.track_id = new_id


def _is_near_frame_edge(bbox: BoundingBox, config: TrackConfig) -> bool:
    margin = config.edge_margin

    return (
        bbox.x_min <= margin
        or bbox.y_min <= margin
        or bbox.x_max >= config.frame_width - margin
        or bbox.y_max >= config.frame_height - margin
    )


def _center_similarity(
    first: BoundingBox,
    second: BoundingBox,
    max_distance_factor: float,
) -> float:
    first_center = first.center
    second_center = second.center
    dx = first_center[0] - second_center[0]
    dy = first_center[1] - second_center[1]
    distance = (dx * dx + dy * dy) ** 0.5
    scale = max(80.0, max(first.width, first.height, second.width, second.height))
    max_distance = scale * max_distance_factor

    return max(0.0, 1.0 - distance / max_distance)


def _size_similarity(first: BoundingBox, second: BoundingBox) -> float:
    width_ratio = min(first.width, second.width) / max(1, max(first.width, second.width))
    height_ratio = min(first.height, second.height) / max(1, max(first.height, second.height))

    return 0.5 * width_ratio + 0.5 * height_ratio


def _best_detection_score(detection: Detection) -> float:
    bbox = detection.bbox
    source_bonus = 0.0

    if "anchor" in detection.source:
        source_bonus = 0.08
    elif detection.source == "promo_color_layout":
        source_bonus = 0.05

    area_bonus = min(bbox.area / 120_000.0, 1.0) * 0.08
    aspect_penalty = 0.0

    if bbox.aspect_ratio < 0.25 or bbox.aspect_ratio > 5.5:
        aspect_penalty = 0.10

    return detection.score + source_bonus + area_bonus - aspect_penalty


def _detection_from_row(row: dict[str, str]) -> Detection:
    return Detection(
        frame_index=_int_value(row.get("frame_index", "")),
        timestamp_ms=_float_value(row.get("timestamp_ms", "")),
        candidate_index=_int_value(row.get("candidate_index", "")),
        source=row.get("source", ""),
        score=_float_value(row.get("score", "0")),
        bbox=BoundingBox(
            x_min=_int_value(row.get("x_min", "")),
            y_min=_int_value(row.get("y_min", "")),
            x_max=_int_value(row.get("x_max", "")),
            y_max=_int_value(row.get("y_max", "")),
        ),
    )


def _int_value(value: str) -> int:
    return int(round(_float_value(value)))


def _float_value(value: str) -> float:
    text = str(value).replace(",", ".").strip()

    if not text:
        return 0.0

    return float(text)
