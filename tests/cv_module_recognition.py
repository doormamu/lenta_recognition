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
        default=0.18,
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
            key=lambda row: _float_value(row.get("score", "0")),
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


def _is_promising_for_no_labels(detection_row: dict[str, str]) -> bool:
    source = detection_row.get("source", "")
    score = _float_value(detection_row.get("score", "0"))

    if "anchor" in source:
        return True

    if source == "promo_color_layout" and score >= 0.74:
        return True

    if source == "mixed" and score >= 0.70:
        return True

    return False


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
    return bbox.expand(
        frame_width=frame_width,
        frame_height=frame_height,
        left=padding_ratio,
        top=padding_ratio,
        right=padding_ratio,
        bottom=padding_ratio,
    )


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
