from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.candidate_merger import BoundingBox  # noqa: E402
from cv_module.tracking import read_detection_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Собрать debug-видео с трекингом ценников"
    )
    parser.add_argument("--video", required=True, help="Исходное видео")
    parser.add_argument("--tracks", required=True, help="tracks.csv из cv_module_tracking.py")
    parser.add_argument(
        "--track-detections",
        default=None,
        help="track_detections.csv из cv_module_tracking.py; по умолчанию рядом с tracks.csv",
    )
    parser.add_argument(
        "--detection-report",
        default=None,
        help="Исходный detection_report.csv; нужен только для серых рамок --draw-all-detections",
    )
    parser.add_argument("--output", required=True, help="Путь к mp4")
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="FPS debug-видео; это не FPS исходника, а скорость просмотра выбранных кадров",
    )
    parser.add_argument(
        "--draw-all-detections",
        action="store_true",
        help="Серыми рамками показать все detection-кандидаты на кадре",
    )
    parser.add_argument(
        "--draw-tails",
        action="store_true",
        help="Показать линии движения центров треков",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks_path = Path(args.tracks)
    track_rows = _read_tracks(tracks_path)
    track_detections_path = (
        Path(args.track_detections)
        if args.track_detections
        else tracks_path.parent / "track_detections.csv"
    )
    track_detection_rows = _read_track_detections(track_detections_path)
    detections = read_detection_report(Path(args.detection_report)) if args.detection_report else []
    detections_by_frame: dict[int, list] = {}

    for detection in detections:
        detections_by_frame.setdefault(detection.frame_index, []).append(detection)

    track_detections_by_frame: dict[int, list[tuple[int, BoundingBox]]] = {}
    track_centers: dict[int, list[tuple[int, int, int]]] = {}

    known_track_ids = {int(row["track_id"]) for row in track_rows}

    for row in track_detection_rows:
        track_id = int(row["track_id"])

        if track_id not in known_track_ids:
            continue

        frame_index = int(row["frame_index"])
        bbox = BoundingBox(
            x_min=int(row["x_min"]),
            y_min=int(row["y_min"]),
            x_max=int(row["x_max"]),
            y_max=int(row["y_max"]),
        )
        track_detections_by_frame.setdefault(frame_index, []).append((track_id, bbox))
        center_x, center_y = bbox.center
        track_centers.setdefault(track_id, []).append(
            (frame_index, int(center_x), int(center_y))
        )

    frames_to_render = sorted(track_detections_by_frame)

    if not frames_to_render:
        raise SystemExit("Нет кадров для визуализации: tracks не связались с detection_report")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    try:
        for frame_index in frames_to_render:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()

            if not ok or frame is None:
                continue

            if args.draw_all_detections:
                for detection in detections_by_frame.get(frame_index, []):
                    _draw_box(
                        frame=frame,
                        bbox=detection.bbox,
                        color=(130, 130, 130),
                        label="",
                        thickness=1,
                    )

            for track_id, bbox in track_detections_by_frame.get(frame_index, []):
                color = _track_color(track_id)
                _draw_box(
                    frame=frame,
                    bbox=bbox,
                    color=color,
                    label=f"T{track_id}",
                    thickness=3,
                )
                if args.draw_tails:
                    _draw_tail(
                        frame=frame,
                        centers=track_centers.get(track_id, []),
                        current_frame=frame_index,
                        color=color,
                    )

            cv2.putText(
                frame,
                f"frame {frame_index} | tracks {len(track_detections_by_frame.get(frame_index, []))}",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    print(f"frames: {len(frames_to_render)}")
    print(f"output: {output_path}")


def _read_tracks(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def _read_track_detections(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def _draw_box(
    frame,
    bbox: BoundingBox,
    color: tuple[int, int, int],
    label: str,
    thickness: int,
) -> None:
    cv2.rectangle(
        frame,
        (bbox.x_min, bbox.y_min),
        (bbox.x_max, bbox.y_max),
        color,
        thickness,
    )

    if not label:
        return

    cv2.putText(
        frame,
        label,
        (bbox.x_min, max(24, bbox.y_min - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_tail(
    frame,
    centers: list[tuple[int, int, int]],
    current_frame: int,
    color: tuple[int, int, int],
) -> None:
    visible = [
        (x, y)
        for frame_index, x, y in centers
        if current_frame - 90 <= frame_index <= current_frame
    ]

    for first, second in zip(visible, visible[1:]):
        cv2.line(frame, first, second, color, 2, cv2.LINE_AA)


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 220, 255),
        (80, 180, 255),
        (0, 255, 120),
        (255, 180, 80),
        (255, 100, 180),
        (180, 255, 80),
        (140, 140, 255),
        (255, 255, 80),
    ]
    return palette[track_id % len(palette)]


if __name__ == "__main__":
    main()
