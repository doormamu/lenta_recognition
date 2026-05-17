from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.candidate_merger import BoundingBox  # noqa: E402
from cv_module.recognition.field_parser import (  # noqa: E402
    OUTPUT_FIELDS,
    PriceTagFieldParser,
)


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
        required=True,
        help="CSV с разметкой для того же видео из data/output/labeled",
    )
    parser.add_argument(
        "--output",
        default="data/output/recognition_debug/recognition_report.csv",
        help="Куда сохранить распознанные поля",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="Минимальный IoU для связывания кандидата с размеченным ценником",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Сколько совпавших кандидатов сохранить",
    )
    args = parser.parse_args()

    detection_rows = _read_csv(Path(args.detection_report))
    label_rows = _read_csv(Path(args.labels))

    parser_engine = PriceTagFieldParser()
    used_label_indexes: set[int] = set()
    output_rows: list[dict[str, str | int | float]] = []

    for detection_row in detection_rows:
        if len(output_rows) >= args.max_rows:
            break

        detection_box = _box_from_row(detection_row)
        frame_timestamp = _float_value(detection_row.get("timestamp_ms", ""))

        label_index, label_row, iou = _find_best_label(
            detection_box=detection_box,
            detection_timestamp_ms=frame_timestamp,
            label_rows=label_rows,
            used_label_indexes=used_label_indexes,
        )

        if label_row is None or iou < args.iou_threshold:
            continue

        used_label_indexes.add(label_index)

        fields = parser_engine.parse(labeled_row=label_row)

        output_row: dict[str, str | int | float] = {
            "frame_index": detection_row.get("frame_index", ""),
            "timestamp_ms": detection_row.get("timestamp_ms", ""),
            "candidate_index": detection_row.get("candidate_index", ""),
            "source": detection_row.get("source", ""),
            "score": detection_row.get("score", ""),
            "match_iou": round(iou, 4),
        }
        output_row.update(fields.to_dict())

        output_rows.append(output_row)

    output_path = Path(args.output)
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

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"labels: {len(label_rows)}")
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _find_best_label(
    detection_box: BoundingBox,
    detection_timestamp_ms: float,
    label_rows: list[dict[str, str]],
    used_label_indexes: set[int],
    max_timestamp_delta_ms: float = 180.0,
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
