from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.candidate_merger import BoundingBox  # noqa: E402
from cv_module.recognition.barcode_reader import BarcodeReader  # noqa: E402
from cv_module.recognition.field_parser import (  # noqa: E402
    OUTPUT_FIELDS,
    PriceTagFieldParser,
)
from cv_module.recognition.ocr_engine import OCREngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка recognition-модуля на detection_report и размеченном CSV"
    )
    parser.add_argument(
        "--detection-report",
        default="data/output/detection_debug/detection_report.csv",
        help="CSV с кандидатами из tests/cv_module_detection.py",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="CSV с разметкой для того же видео из data/output/labeled",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Папка crops из tests/cv_module_detection.py; если не задана, берется рядом с detection_report",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Видео для no-labels режима, если crop-файлы не сохранены",
    )
    parser.add_argument(
        "--output",
        default="data/output/recognition_debug/recognition_report.csv",
        help="Куда сохранить распознанные поля",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.25,
        help="Минимальный IoU для связывания кандидата с размеченным ценником",
    )
    parser.add_argument(
        "--max-timestamp-delta-ms",
        type=float,
        default=2000.0,
        help="Максимальная разница timestamp между кандидатом и строкой разметки",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Сколько совпавших кандидатов сохранить",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
        help="Сколько кандидатов максимум просмотреть в no-labels режиме",
    )
    parser.add_argument(
        "--crop-padding-ratio",
        type=float,
        default=0.55,
        help="Насколько расширять bbox при вырезании crop из видео в no-labels режиме",
    )
    parser.add_argument(
        "--save-ocr-crops",
        action="store_true",
        help="Сохранять фактические crop-картинки, которые уходят в OCR",
    )
    args = parser.parse_args()

    detection_report_path = Path(args.detection_report)
    detection_rows = _read_csv(detection_report_path)
    label_rows = _read_csv(Path(args.labels)) if args.labels else []

    if not label_rows:
        detection_rows.sort(
            key=_no_labels_ranking_key,
            reverse=True,
        )

    crops_dir = Path(args.crops_dir) if args.crops_dir else detection_report_path.parent / "crops"
    output_path = Path(args.output)
    debug_crops_dir = output_path.parent / "ocr_crops" if args.save_ocr_crops else None

    parser_engine = PriceTagFieldParser()
    barcode_reader = BarcodeReader(try_harder=True)
    ocr_engine = OCREngine()
    video_capture = cv2.VideoCapture(args.video) if args.video else None
    used_label_indexes: set[int] = set()
    output_rows: list[dict[str, str | int | float]] = []

    try:
        checked_detections = 0

        for detection_row in detection_rows:
            if len(output_rows) >= args.max_rows:
                break

            if not label_rows:
                checked_detections += 1

                if checked_detections > args.max_detections:
                    break

                if not _is_promising_for_no_labels(detection_row):
                    continue

            if label_rows:
                fields, match_iou = _recognize_with_labels(
                    detection_row=detection_row,
                    label_rows=label_rows,
                    parser_engine=parser_engine,
                    used_label_indexes=used_label_indexes,
                    max_timestamp_delta_ms=args.max_timestamp_delta_ms,
                    iou_threshold=args.iou_threshold,
                )
            else:
                fields, match_iou = _recognize_without_labels(
                    detection_row=detection_row,
                    crops_dir=crops_dir,
                    parser_engine=parser_engine,
                    barcode_reader=barcode_reader,
                    ocr_engine=ocr_engine,
                    video_capture=video_capture,
                    crop_padding_ratio=args.crop_padding_ratio,
                    debug_crops_dir=debug_crops_dir,
                )

            if fields is None:
                continue

            output_row: dict[str, str | int | float] = {
                "frame_index": detection_row.get("frame_index", ""),
                "timestamp_ms": detection_row.get("timestamp_ms", ""),
                "candidate_index": detection_row.get("candidate_index", ""),
                "source": detection_row.get("source", ""),
                "score": detection_row.get("score", ""),
                "match_iou": match_iou,
            }
            output_row.update(_make_excel_friendly(fields.to_dict()))

            output_rows.append(output_row)
    finally:
        if video_capture is not None:
            video_capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_index",
        "timestamp_ms",
        "candidate_index",
        "source",
        "score",
        "match_iou",
        *OUTPUT_FIELDS,
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"labels: {len(label_rows) if label_rows else 'disabled'}")
    print(f"crops_dir: {crops_dir}")
    print(f"detections: {len(detection_rows)}")
    print(f"recognized_rows: {len(output_rows)}")
    print(f"output: {output_path}")

    for row in output_rows[:5]:
        print(
            " | ".join(
                [
                    f"frame={row['frame_index']}",
                    f"iou={row['match_iou']}",
                    str(row["product_name"]),
                    f"price_card={row['price_card']}",
                    f"barcode={row['barcode']}",
                ]
            )
        )


def _recognize_with_labels(
    detection_row: dict[str, str],
    label_rows: list[dict[str, str]],
    parser_engine: PriceTagFieldParser,
    used_label_indexes: set[int],
    max_timestamp_delta_ms: float,
    iou_threshold: float,
):
    detection_box = _box_from_row(detection_row)
    frame_timestamp = _float_value(detection_row.get("timestamp_ms", ""))

    label_index, label_row, iou = _find_best_label(
        detection_box=detection_box,
        detection_timestamp_ms=frame_timestamp,
        label_rows=label_rows,
        used_label_indexes=used_label_indexes,
        max_timestamp_delta_ms=max_timestamp_delta_ms,
    )

    if label_row is None or iou < iou_threshold:
        return None, "-"

    used_label_indexes.add(label_index)

    return parser_engine.parse(labeled_row=label_row), round(iou, 4)


def _recognize_without_labels(
    detection_row: dict[str, str],
    crops_dir: Path,
    parser_engine: PriceTagFieldParser,
    barcode_reader: BarcodeReader,
    ocr_engine: OCREngine,
    video_capture,
    crop_padding_ratio: float,
    debug_crops_dir: Path | None,
):
    image = _load_detection_crop(
        crops_dir=crops_dir,
        detection_row=detection_row,
        video_capture=video_capture,
        crop_padding_ratio=crop_padding_ratio,
    )

    if image is None or image.size == 0:
        return None, "-"

    price_tag_roi = _extract_price_tag_roi(image)

    if price_tag_roi is None:
        return None, "-"

    image = price_tag_roi

    if debug_crops_dir is not None:
        debug_crops_dir.mkdir(parents=True, exist_ok=True)
        frame_index = _int_value(detection_row.get("frame_index", ""))
        candidate_index = _int_value(detection_row.get("candidate_index", ""))
        cv2.imwrite(
            str(debug_crops_dir / f"frame_{frame_index:06d}_candidate_{candidate_index:03d}.jpg"),
            image,
        )

    barcode_reads = barcode_reader.read(image)
    ocr_result = ocr_engine.recognize(image)

    fields = parser_engine.parse(
        ocr_text=ocr_result.raw_text,
        code_values=[read.value for read in barcode_reads],
    )

    field_values = fields.to_dict()
    signal_fields = {
        "price_default",
        "price_card",
        "price_discount",
        "barcode",
        "discount_amount",
        "id_sku",
        "qr_code_barcode",
        "price1_qr",
        "price2_qr",
        "price3_qr",
        "price4_qr",
        "action_price_qr",
        "action_code_qr",
    }
    has_signal = any(
        field_values[field_name] not in {"-", "нет"}
        for field_name in signal_fields
    )

    if not has_signal:
        return None, "-"

    return fields, "-"


def _looks_like_price_tag_crop(image) -> bool:
    return _extract_price_tag_roi(image) is not None


def _extract_price_tag_roi(image):
    height, width = image.shape[:2]

    if width < 80 or height < 45:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        (0, 0, 135),
        (180, 95, 255),
    )
    yellow_mask = cv2.inRange(
        hsv,
        (15, 45, 75),
        (42, 255, 255),
    )
    red_mask_1 = cv2.inRange(
        hsv,
        (0, 35, 70),
        (13, 255, 255),
    )
    red_mask_2 = cv2.inRange(
        hsv,
        (165, 35, 70),
        (180, 255, 255),
    )
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

    white_boxes = _mask_component_boxes(white_mask, min_area=max(180, int(width * height * 0.006)))
    color_boxes = _mask_component_boxes(color_mask, min_area=max(120, int(width * height * 0.003)))
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

            if _boxes_are_price_tag_neighbors(color_box, white_box):
                union_width = max(color_box.x_max, white_box.x_max) - min(color_box.x_min, white_box.x_min)
                union_height = max(color_box.y_max, white_box.y_max) - min(color_box.y_min, white_box.y_min)
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

    if _is_same_column(color_box, white_box):
        block_height = max(color_box.height, white_box.height)
        pad_y = max(pad_y, int(1.05 * block_height))
        pad_x = max(pad_x, int(0.48 * max(color_box.width, white_box.width)))

        if white_box.center[1] < color_box.center[1]:
            y_min -= int(0.95 * block_height)
            y_max += int(0.35 * block_height)
        else:
            y_min -= int(0.35 * block_height)
            y_max += int(0.95 * block_height)
    elif _is_same_row(color_box, white_box):
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


def _mask_component_boxes(mask, min_area: int) -> list[BoundingBox]:
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


def _boxes_are_price_tag_neighbors(
    color_box: BoundingBox,
    white_box: BoundingBox,
) -> bool:
    same_row = _is_same_row(color_box, white_box)
    same_column = _is_same_column(color_box, white_box)
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


def _is_same_row(color_box: BoundingBox, white_box: BoundingBox) -> bool:
    overlap = _axis_overlap(
        color_box.y_min,
        color_box.y_max,
        white_box.y_min,
        white_box.y_max,
    )
    return overlap >= 0.35 * min(color_box.height, white_box.height)


def _is_same_column(color_box: BoundingBox, white_box: BoundingBox) -> bool:
    overlap = _axis_overlap(
        color_box.x_min,
        color_box.x_max,
        white_box.x_min,
        white_box.x_max,
    )
    return overlap >= 0.35 * min(color_box.width, white_box.width)


def _axis_overlap(
    first_min: int,
    first_max: int,
    second_min: int,
    second_max: int,
) -> int:
    return max(0, min(first_max, second_max) - max(first_min, second_min))


def _is_promising_for_no_labels(detection_row: dict[str, str]) -> bool:
    source = detection_row.get("source", "")
    score = _float_value(detection_row.get("score", "0"))
    width = _float_value(detection_row.get("width", "0"))
    height = _float_value(detection_row.get("height", "0"))
    area = width * height
    x_min = _float_value(detection_row.get("x_min", "0"))
    y_min = _float_value(detection_row.get("y_min", "0"))
    x_max = _float_value(detection_row.get("x_max", "0"))
    y_max = _float_value(detection_row.get("y_max", "0"))

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


def _no_labels_ranking_key(row: dict[str, str]) -> tuple[float, float, float]:
    source = row.get("source", "")
    score = _float_value(row.get("score", "0"))
    width = _float_value(row.get("width", "0"))
    height = _float_value(row.get("height", "0"))
    area = width * height

    source_priority = 0.0

    if "anchor" in source:
        source_priority = 3.0
    elif source == "promo_color_layout":
        source_priority = 2.0
    elif source == "mixed" and area <= 180_000:
        source_priority = 1.0

    area_penalty = min(area / 500_000.0, 1.0)

    return (
        source_priority,
        score,
        -area_penalty,
    )


def _load_detection_crop(
    crops_dir: Path,
    detection_row: dict[str, str],
    video_capture,
    crop_padding_ratio: float,
):
    if video_capture is None or not video_capture.isOpened():
        crop_path = _find_crop_path(crops_dir, detection_row)

        if crop_path is not None:
            return cv2.imread(str(crop_path))

        return None

    frame_index = _int_value(detection_row.get("frame_index", ""))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = video_capture.read()

    if not ok or frame is None:
        return None

    bbox = _expand_box(
        bbox=_box_from_row(detection_row),
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        padding_ratio=crop_padding_ratio,
    )

    return frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]


def _expand_box(
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


def _find_crop_path(
    crops_dir: Path,
    detection_row: dict[str, str],
) -> Path | None:
    frame_index = _int_value(detection_row.get("frame_index", ""))
    candidate_index = _int_value(detection_row.get("candidate_index", ""))
    pattern = f"frame_{frame_index:06d}_candidate_{candidate_index:03d}_*.jpg"
    matches = sorted(crops_dir.glob(pattern))

    if not matches:
        return None

    return matches[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _make_excel_friendly(row: dict[str, str]) -> dict[str, str]:
    return {
        key: " ".join(str(value).split())
        for key, value in row.items()
    }


def _find_best_label(
    detection_box: BoundingBox,
    detection_timestamp_ms: float,
    label_rows: list[dict[str, str]],
    used_label_indexes: set[int],
    max_timestamp_delta_ms: float,
) -> tuple[int, dict[str, str] | None, float]:
    best_index = -1
    best_row = None
    best_iou = 0.0

    for index, row in enumerate(label_rows):
        if index in used_label_indexes:
            continue

        label_timestamp = _float_value(row.get("frame_timestamp", ""))

        if abs(label_timestamp - detection_timestamp_ms) > max_timestamp_delta_ms:
            continue

        label_box = _box_from_row(row)
        iou = detection_box.iou(label_box)

        if iou > best_iou:
            best_index = index
            best_row = row
            best_iou = iou

    return best_index, best_row, best_iou


def _box_from_row(row: dict[str, str]) -> BoundingBox:
    return BoundingBox(
        x_min=_int_value(row.get("x_min", "")),
        y_min=_int_value(row.get("y_min", "")),
        x_max=_int_value(row.get("x_max", "")),
        y_max=_int_value(row.get("y_max", "")),
    )


def _int_value(value: str) -> int:
    return int(round(_float_value(value)))


def _float_value(value: str) -> float:
    text = str(value).replace(",", ".").strip()

    if not text:
        return 0.0

    return float(text)


if __name__ == "__main__":
    main()
