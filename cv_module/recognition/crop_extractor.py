from __future__ import annotations

from pathlib import Path

import cv2

from cv_module.detection.candidate_merger import BoundingBox


def extract_price_tag_roi(image):
    height, width = image.shape[:2]

    if width < 80 or height < 45:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 135), (180, 95, 255))
    yellow_mask = cv2.inRange(hsv, (15, 45, 75), (42, 255, 255))
    red_mask_1 = cv2.inRange(hsv, (0, 35, 70), (13, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (165, 35, 70), (180, 255, 255))
    color_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask_1, red_mask_2), yellow_mask)

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
    )
    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
    )

    white_ratio = float((white_mask > 0).mean())
    color_ratio = float((color_mask > 0).mean())

    if white_ratio < 0.035 or color_ratio < 0.006:
        return None

    white_boxes = mask_component_boxes(
        white_mask,
        min_area=max(180, int(width * height * 0.006)),
    )
    color_boxes = mask_component_boxes(
        color_mask,
        min_area=max(120, int(width * height * 0.003)),
    )
    max_white_area = int(width * height * 0.42)
    white_boxes = [
        box
        for box in white_boxes
        if box.width * box.height <= max_white_area
        and not (box.width >= int(0.72 * width) and box.height >= int(0.72 * height))
    ]

    if not white_boxes or not color_boxes:
        return None

    best_pair: tuple[BoundingBox, BoundingBox] | None = None
    best_score = 0.0

    for color_box in color_boxes:
        if _touches_image_border(color_box, width, height):
            continue

        for white_box in white_boxes:
            if _touches_image_border(white_box, width, height):
                continue

            if boxes_are_price_tag_neighbors(color_box, white_box):
                union_width = max(color_box.x_max, white_box.x_max) - min(
                    color_box.x_min,
                    white_box.x_min,
                )
                union_height = max(color_box.y_max, white_box.y_max) - min(
                    color_box.y_min,
                    white_box.y_min,
                )
                union_area = union_width * union_height
                color_area = color_box.width * color_box.height
                white_area = white_box.width * white_box.height
                compactness = (color_area + white_area) / max(1, union_area)

                if compactness < 0.32:
                    continue

                score = (color_area + white_area) * compactness

                if score > best_score:
                    best_pair = (color_box, white_box)
                    best_score = score

    if best_pair is None:
        return None

    color_box, white_box = best_pair
    x_min = min(color_box.x_min, white_box.x_min)
    y_min = min(color_box.y_min, white_box.y_min)
    x_max = max(color_box.x_max, white_box.x_max)
    y_max = max(color_box.y_max, white_box.y_max)

    roi_width = x_max - x_min
    roi_height = y_max - y_min
    pad_x = max(14, int(0.26 * roi_width))
    pad_y = max(12, int(0.34 * roi_height))

    if is_same_column(color_box, white_box):
        block_height = max(color_box.height, white_box.height)
        pad_y = max(pad_y, int(1.05 * block_height))
        pad_x = max(pad_x, int(0.48 * max(color_box.width, white_box.width)))

        if white_box.center[1] < color_box.center[1]:
            y_min -= int(0.95 * block_height)
            y_max += int(0.35 * block_height)
        else:
            y_min -= int(0.35 * block_height)
            y_max += int(0.95 * block_height)
    elif is_same_row(color_box, white_box):
        block_width = max(color_box.width, white_box.width)
        pad_x = max(pad_x, int(1.0 * block_width))
        pad_y = max(pad_y, int(0.58 * max(color_box.height, white_box.height)))

        if white_box.center[0] < color_box.center[0]:
            x_min -= int(0.9 * block_width)
            x_max += int(0.4 * block_width)
        else:
            x_min -= int(0.4 * block_width)
            x_max += int(0.9 * block_width)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)

    roi = image[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return None

    if not _has_enough_tag_color_inside_roi(roi):
        return None

    return roi


def looks_like_price_tag_crop(image) -> bool:
    return extract_price_tag_roi(image) is not None


def _touches_image_border(box: BoundingBox, width: int, height: int) -> bool:
    margin = 4
    return (
        box.x_min <= margin
        or box.y_min <= margin
        or box.x_max >= width - margin
        or box.y_max >= height - margin
    )


def _has_enough_tag_color_inside_roi(image) -> bool:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 135), (180, 95, 255))
    yellow_mask = cv2.inRange(hsv, (15, 45, 75), (42, 255, 255))
    red_mask_1 = cv2.inRange(hsv, (0, 35, 70), (13, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (165, 35, 70), (180, 255, 255))
    color_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask_1, red_mask_2), yellow_mask)

    white_ratio = float((white_mask > 0).mean())
    color_ratio = float((color_mask > 0).mean())

    return white_ratio >= 0.025 and color_ratio >= 0.045


def load_detection_crop(
    crops_dir: Path,
    detection_row: dict[str, str],
    video_capture,
    crop_padding_ratio: float,
):
    if video_capture is None or not video_capture.isOpened():
        crop_path = find_crop_path(crops_dir, detection_row)

        if crop_path is not None:
            return cv2.imread(str(crop_path))

        return None

    frame_index = int_value(detection_row.get("frame_index", ""))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = video_capture.read()

    if not ok or frame is None:
        return None

    bbox = expand_box(
        bbox=box_from_row(detection_row),
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        padding_ratio=crop_padding_ratio,
    )

    return frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]


def is_promising_for_no_labels(detection_row: dict[str, str]) -> bool:
    source = detection_row.get("source", "")
    score = float_value(detection_row.get("score", "0"))
    width = float_value(detection_row.get("width", "0"))
    height = float_value(detection_row.get("height", "0"))
    area = width * height
    x_min = float_value(detection_row.get("x_min", "0"))
    y_min = float_value(detection_row.get("y_min", "0"))
    x_max = float_value(detection_row.get("x_max", "0"))
    y_max = float_value(detection_row.get("y_max", "0"))

    if min(x_min, y_min) < 12 or x_max > 3828 or y_max > 2148:
        return False

    if width < 90 or height < 70:
        return False

    if "anchor" in source:
        return True

    if source == "promo_color_layout" and score >= 0.74:
        return True

    if source == "mixed" and score >= 0.70 and area <= 180_000:
        return True

    return False


def no_labels_ranking_key(row: dict[str, str]) -> tuple[float, float, float]:
    source = row.get("source", "")
    score = float_value(row.get("score", "0"))
    width = float_value(row.get("width", "0"))
    height = float_value(row.get("height", "0"))
    area = width * height

    source_priority = 0.0

    if "anchor" in source:
        source_priority = 3.0
    elif source == "promo_color_layout":
        source_priority = 2.0
    elif source == "mixed" and area <= 180_000:
        source_priority = 1.0

    area_penalty = min(area / 500_000.0, 1.0)

    return (source_priority, score, -area_penalty)


def mask_component_boxes(mask, min_area: int) -> list[BoundingBox]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes: list[BoundingBox] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area:
            continue

        if w < 12 or h < 10:
            continue

        boxes.append(BoundingBox(x, y, x + w, y + h))

    return boxes


def boxes_are_price_tag_neighbors(
    color_box: BoundingBox,
    white_box: BoundingBox,
) -> bool:
    same_row = is_same_row(color_box, white_box)
    same_column = is_same_column(color_box, white_box)
    height_ratio = max(color_box.height, white_box.height) / max(
        1,
        min(color_box.height, white_box.height),
    )
    width_ratio = max(color_box.width, white_box.width) / max(
        1,
        min(color_box.width, white_box.width),
    )
    area_ratio = max(color_box.area, white_box.area) / max(
        1,
        min(color_box.area, white_box.area),
    )

    horizontal_gap = max(
        0,
        max(color_box.x_min, white_box.x_min) - min(color_box.x_max, white_box.x_max),
    )
    vertical_gap = max(
        0,
        max(color_box.y_min, white_box.y_min) - min(color_box.y_max, white_box.y_max),
    )

    max_horizontal_gap = max(14, int(0.75 * min(color_box.width, white_box.width)))
    max_vertical_gap = max(14, int(0.75 * min(color_box.height, white_box.height)))

    if (
        same_row
        and height_ratio <= 3.2
        and area_ratio <= 5.5
        and horizontal_gap <= max_horizontal_gap
    ):
        return True

    if (
        same_column
        and width_ratio <= 3.2
        and area_ratio <= 5.5
        and vertical_gap <= max_vertical_gap
    ):
        return True

    return False


def is_same_row(color_box: BoundingBox, white_box: BoundingBox) -> bool:
    overlap = axis_overlap(
        color_box.y_min,
        color_box.y_max,
        white_box.y_min,
        white_box.y_max,
    )
    return overlap >= 0.35 * min(color_box.height, white_box.height)


def is_same_column(color_box: BoundingBox, white_box: BoundingBox) -> bool:
    overlap = axis_overlap(
        color_box.x_min,
        color_box.x_max,
        white_box.x_min,
        white_box.x_max,
    )
    return overlap >= 0.35 * min(color_box.width, white_box.width)


def axis_overlap(
    first_min: int,
    first_max: int,
    second_min: int,
    second_max: int,
) -> int:
    return max(0, min(first_max, second_max) - max(first_min, second_min))


def expand_box(
    bbox: BoundingBox,
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> BoundingBox:
    width = max(1, bbox.width)
    height = max(1, bbox.height)
    aspect_ratio = bbox.aspect_ratio

    horizontal_padding = padding_ratio
    vertical_padding = padding_ratio

    if aspect_ratio >= 2.2:
        vertical_padding = max(vertical_padding, 2.4)
        horizontal_padding = max(horizontal_padding, 0.55)
    elif aspect_ratio <= 0.45:
        horizontal_padding = max(horizontal_padding, 2.4)
        vertical_padding = max(vertical_padding, 0.55)
    else:
        horizontal_padding = max(horizontal_padding, 1.0)
        vertical_padding = max(vertical_padding, 1.0)

    expanded = BoundingBox(
        x_min=int(bbox.x_min - horizontal_padding * width),
        y_min=int(bbox.y_min - vertical_padding * height),
        x_max=int(bbox.x_max + horizontal_padding * width),
        y_max=int(bbox.y_max + vertical_padding * height),
    )

    return expanded.clamp(frame_width, frame_height)


def find_crop_path(
    crops_dir: Path,
    detection_row: dict[str, str],
) -> Path | None:
    frame_index = int_value(detection_row.get("frame_index", ""))
    candidate_index = int_value(detection_row.get("candidate_index", ""))
    pattern = f"frame_{frame_index:06d}_candidate_{candidate_index:03d}_*.jpg"
    matches = sorted(crops_dir.glob(pattern))

    if not matches:
        return None

    return matches[0]


def box_from_row(row: dict[str, str]) -> BoundingBox:
    return BoundingBox(
        x_min=int_value(row.get("x_min", "")),
        y_min=int_value(row.get("y_min", "")),
        x_max=int_value(row.get("x_max", "")),
        y_max=int_value(row.get("y_max", "")),
    )


def int_value(value: str) -> int:
    return int(round(float_value(value)))


def float_value(value: str) -> float:
    text = str(value).replace(",", ".").strip()

    if not text:
        return 0.0

    return float(text)
