from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.candidate_merger import BoundingBox  # noqa: E402
from cv_module.tracking import (  # noqa: E402
    PriceTagTracker,
    TrackConfig,
    read_detection_report,
    tracks_to_rows,
)


TRACK_FIELDNAMES = [
    "track_id",
    "start_frame",
    "end_frame",
    "start_timestamp_ms",
    "end_timestamp_ms",
    "duration_ms",
    "detections_count",
    "best_frame_index",
    "best_timestamp_ms",
    "best_candidate_index",
    "best_detection_score",
    "best_track_score",
    "source",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
    "aspect_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить треки ценников по detection_report.csv"
    )
    parser.add_argument(
        "--detection-report",
        required=True,
        help="CSV с кандидатами из tests/cv_module_detection.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Куда сохранить tracks.csv",
    )
    parser.add_argument(
        "--track-detections-output",
        default=None,
        help="Куда сохранить покадровые detection->track связи; по умолчанию рядом с tracks.csv",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Видео для сохранения best crops",
    )
    parser.add_argument(
        "--save-best-crops",
        action="store_true",
        help="Сохранить лучший crop каждого трека рядом с tracks.csv",
    )
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--min-detections", type=int, default=2)
    parser.add_argument("--max-frame-gap", type=int, default=36)
    parser.add_argument("--min-match-score", type=float, default=0.27)
    parser.add_argument("--max-detections-per-frame", type=int, default=40)
    args = parser.parse_args()

    detection_report_path = Path(args.detection_report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detections = read_detection_report(detection_report_path)
    tracker = PriceTagTracker(
        TrackConfig(
            min_score=args.min_score,
            min_detections_per_track=args.min_detections,
            max_frame_gap=args.max_frame_gap,
            min_match_score=args.min_match_score,
            max_detections_per_frame=args.max_detections_per_frame,
        )
    )
    tracks = tracker.track(detections)
    rows = tracks_to_rows(tracks)
    track_detection_rows = _track_detection_rows(tracks)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TRACK_FIELDNAMES,
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    track_detections_output = (
        Path(args.track_detections_output)
        if args.track_detections_output
        else output_path.parent / "track_detections.csv"
    )
    _save_track_detection_rows(track_detection_rows, track_detections_output)

    if args.save_best_crops:
        if not args.video:
            raise SystemExit("--save-best-crops требует --video")

        _save_best_crops(
            video_path=Path(args.video),
            rows=rows,
            output_dir=output_path.parent / "best_crops",
        )

    print(f"detections: {len(detections)}")
    print(f"tracks: {len(tracks)}")
    print(f"output: {output_path}")
    print(f"track_detections: {track_detections_output}")

    if args.save_best_crops:
        print(f"best_crops: {output_path.parent / 'best_crops'}")

    for row in rows[:8]:
        print(
            " | ".join(
                [
                    f"T{row['track_id']}",
                    f"frames={row['start_frame']}-{row['end_frame']}",
                    f"n={row['detections_count']}",
                    f"best={row['best_frame_index']}/{row['best_candidate_index']}",
                    f"score={row['best_track_score']}",
                ]
            )
        )


def _save_best_crops(
    video_path: Path,
    rows: list[dict[str, int | float | str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))

    try:
        for row in rows:
            frame_index = int(row["best_frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()

            if not ok or frame is None:
                continue

            bbox = BoundingBox(
                x_min=int(row["x_min"]),
                y_min=int(row["y_min"]),
                x_max=int(row["x_max"]),
                y_max=int(row["y_max"]),
            ).expand(
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                left=0.20,
                top=0.20,
                right=0.20,
                bottom=0.20,
            )
            crop = frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

            if crop.size == 0:
                continue

            crop_path = output_dir / (
                f"track_{int(row['track_id']):04d}_"
                f"frame_{frame_index:06d}_"
                f"candidate_{int(row['best_candidate_index']):03d}.jpg"
            )
            cv2.imwrite(str(crop_path), crop)
    finally:
        capture.release()


def _track_detection_rows(tracks) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []

    for track in tracks:
        for detection in track.detections:
            bbox = detection.bbox
            rows.append(
                {
                    "track_id": track.track_id,
                    "frame_index": detection.frame_index,
                    "timestamp_ms": round(detection.timestamp_ms, 2),
                    "candidate_index": detection.candidate_index,
                    "source": detection.source,
                    "score": round(detection.score, 4),
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                    "width": bbox.width,
                    "height": bbox.height,
                }
            )

    rows.sort(key=lambda item: (item["frame_index"], item["track_id"]))
    return rows


def _save_track_detection_rows(
    rows: list[dict[str, int | float | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "track_id",
                "frame_index",
                "timestamp_ms",
                "candidate_index",
                "source",
                "score",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "width",
                "height",
            ],
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()


'''
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 tests/cv_module_tracking.py \
  --detection-report data/output/detection_debug/detection_report.csv \
  --video data/input/labeled/25_2-10.mp4 \
  --output data/output/tracking_debug/tracks_25_2_10.csv \
  --save-best-crops \
  --min-score 0.50 \
  --min-detections 3 \
  --max-frame-gap 44 \
  --min-match-score 0.27 \
  --max-detections-per-frame 35 \
  --merge-track-gap 88 \
  --merge-match-score 0.28 \
  --edge-margin 220 \
  --edge-max-frame-gap 96
'''