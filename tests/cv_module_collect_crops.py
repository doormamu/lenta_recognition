from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.recognition.crop_extractor import (  # noqa: E402
    extract_price_tag_roi,
    is_promising_for_no_labels,
    load_detection_crop,
    no_labels_ranking_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Собрать OCR-crops из detection_report без использования labels"
    )
    parser.add_argument(
        "--detection-report",
        required=True,
        help="CSV с кандидатами из tests/cv_module_detection.py",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Исходное видео; если задано, crop вырезается из видео с padding",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Папка crops из detection; используется если --video не задан",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Куда сохранить нормализованные OCR-crops",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="CSV-отчет по сохраненным crops; по умолчанию output-dir/crops_report.csv",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=200,
        help="Сколько crops сохранить максимум",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=1200,
        help="Сколько кандидатов detection просмотреть максимум",
    )
    parser.add_argument(
        "--crop-padding-ratio",
        type=float,
        default=0.55,
        help="Насколько расширять bbox при вырезании crop из видео",
    )
    args = parser.parse_args()

    detection_report_path = Path(args.detection_report)
    crops_dir = Path(args.crops_dir) if args.crops_dir else detection_report_path.parent / "crops"
    output_dir = Path(args.output_dir)
    report_path = Path(args.report) if args.report else output_dir / "crops_report.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_rows = _read_csv(detection_report_path)
    detection_rows.sort(key=no_labels_ranking_key, reverse=True)

    video_capture = cv2.VideoCapture(args.video) if args.video else None
    output_rows: list[dict[str, str | int | float]] = []

    try:
        checked = 0

        for detection_row in detection_rows:
            if len(output_rows) >= args.max_crops:
                break

            checked += 1

            if checked > args.max_detections:
                break

            if not is_promising_for_no_labels(detection_row):
                continue

            image = load_detection_crop(
                crops_dir=crops_dir,
                detection_row=detection_row,
                video_capture=video_capture,
                crop_padding_ratio=args.crop_padding_ratio,
            )

            if image is None or image.size == 0:
                continue

            roi = extract_price_tag_roi(image)

            if roi is None:
                continue

            frame_index = _int_value(detection_row.get("frame_index", ""))
            candidate_index = _int_value(detection_row.get("candidate_index", ""))
            crop_name = f"frame_{frame_index:06d}_candidate_{candidate_index:03d}.jpg"
            crop_path = output_dir / crop_name
            cv2.imwrite(str(crop_path), roi)

            output_rows.append(
                {
                    "crop_path": str(crop_path),
                    "frame_index": detection_row.get("frame_index", ""),
                    "timestamp_ms": detection_row.get("timestamp_ms", ""),
                    "candidate_index": detection_row.get("candidate_index", ""),
                    "source": detection_row.get("source", ""),
                    "score": detection_row.get("score", ""),
                    "x_min": detection_row.get("x_min", ""),
                    "y_min": detection_row.get("y_min", ""),
                    "x_max": detection_row.get("x_max", ""),
                    "y_max": detection_row.get("y_max", ""),
                    "crop_width": roi.shape[1],
                    "crop_height": roi.shape[0],
                }
            )
    finally:
        if video_capture is not None:
            video_capture.release()

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "crop_path",
                "frame_index",
                "timestamp_ms",
                "candidate_index",
                "source",
                "score",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "crop_width",
                "crop_height",
            ],
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"detections: {len(detection_rows)}")
    print(f"saved_crops: {len(output_rows)}")
    print(f"output_dir: {output_dir}")
    print(f"report: {report_path}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _int_value(value: str) -> int:
    text = str(value).replace(",", ".").strip()

    if not text:
        return 0

    return int(round(float(text)))


if __name__ == "__main__":
    main()


'''
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 tests/cv_module_collect_crops.py \
  --detection-report data/output/detection_debug/detection_report.csv \
  --video data/input/labeled/25_2-10.mp4 \
  --output-dir data/output/recognition_debug/crops_25_2_10 \
  --report data/output/recognition_debug/crops_25_2_10.csv \
  --max-crops 200 \
  --max-detections 1200
'''