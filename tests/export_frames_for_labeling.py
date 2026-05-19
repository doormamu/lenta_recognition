from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def estimate_frame_quality(frame) -> float:
    if frame is None or frame.size == 0:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = np.clip(sharpness / 500.0, 0.0, 1.0)

    contrast = gray.std()
    contrast_score = np.clip(contrast / 64.0, 0.0, 1.0)

    brightness = gray.mean()
    brightness_score = 1.0 - abs(brightness - 128.0) / 128.0
    brightness_score = np.clip(brightness_score, 0.0, 1.0)

    overexposed_ratio = float(np.mean(gray > 245))
    underexposed_ratio = float(np.mean(gray < 10))
    exposure_penalty = np.clip(overexposed_ratio + underexposed_ratio, 0.0, 1.0)

    score = (
        0.45 * sharpness_score
        + 0.30 * contrast_score
        + 0.20 * brightness_score
        + 0.05 * (1.0 - exposure_penalty)
    )

    return float(np.clip(score, 0.0, 1.0))


def save_empty_yolo_label(labels_dir: Path, image_name: str) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{Path(image_name).stem}.txt"
    label_path.write_text("", encoding="utf-8")


def export_raw_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    window_sec: float,
    max_frames: int | None,
    min_quality: float,
    prefix: str,
    create_empty_labels: bool,
) -> None:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    metadata_path = output_dir / "frames_metadata.csv"

    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(target_fps, 0.1))))
    window_ms = max(1.0, window_sec * 1000.0)

    print("Видео:")
    print(f"  path: {video_path}")
    print(f"  fps: {fps:.2f}")
    print(f"  frame_count: {frame_count}")
    print(f"  size: {width}x{height}")
    print()
    print("Важно: кадры сохраняются RAW, без поворота и без undistort.")
    print("Именно на них надо размечать bbox, чтобы координаты совпадали с исходным видео.")
    print()

    rows: list[dict] = []

    current_window_id: int | None = None
    best_frame = None
    best_frame_index = -1
    best_timestamp_ms = 0.0
    best_quality = -1.0

    exported_count = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        if frame_index % step != 0:
            frame_index += 1
            continue

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        quality = estimate_frame_quality(frame)
        window_id = int(timestamp_ms // window_ms)

        if current_window_id is None:
            current_window_id = window_id

        if window_id != current_window_id:
            if best_frame is not None and best_quality >= min_quality:
                image_name = (
                    f"{prefix}_frame_{best_frame_index:06d}_"
                    f"time_{best_timestamp_ms / 1000.0:08.2f}.jpg"
                )

                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), best_frame)

                if create_empty_labels:
                    save_empty_yolo_label(labels_dir, image_name)

                rows.append(
                    {
                        "image_name": image_name,
                        "video_path": str(video_path),
                        "frame_index": best_frame_index,
                        "timestamp_ms": round(best_timestamp_ms, 2),
                        "timestamp_sec": round(best_timestamp_ms / 1000.0, 2),
                        "quality_score": round(best_quality, 5),
                        "width": best_frame.shape[1],
                        "height": best_frame.shape[0],
                    }
                )

                exported_count += 1
                print(f"saved {exported_count:04d}: {image_path}")

                if max_frames is not None and exported_count >= max_frames:
                    break

            current_window_id = window_id
            best_frame = frame
            best_frame_index = frame_index
            best_timestamp_ms = timestamp_ms
            best_quality = quality
        else:
            if quality > best_quality:
                best_frame = frame
                best_frame_index = frame_index
                best_timestamp_ms = timestamp_ms
                best_quality = quality

        frame_index += 1

    if max_frames is None or exported_count < max_frames:
        if best_frame is not None and best_quality >= min_quality:
            image_name = (
                f"{prefix}_frame_{best_frame_index:06d}_"
                f"time_{best_timestamp_ms / 1000.0:08.2f}.jpg"
            )

            image_path = images_dir / image_name
            cv2.imwrite(str(image_path), best_frame)

            if create_empty_labels:
                save_empty_yolo_label(labels_dir, image_name)

            rows.append(
                {
                    "image_name": image_name,
                    "video_path": str(video_path),
                    "frame_index": best_frame_index,
                    "timestamp_ms": round(best_timestamp_ms, 2),
                    "timestamp_sec": round(best_timestamp_ms / 1000.0, 2),
                    "quality_score": round(best_quality, 5),
                    "width": best_frame.shape[1],
                    "height": best_frame.shape[0],
                }
            )

            exported_count += 1
            print(f"saved {exported_count:04d}: {image_path}")

    cap.release()

    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "image_name",
            "video_path",
            "frame_index",
            "timestamp_ms",
            "timestamp_sec",
            "quality_score",
            "width",
            "height",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Итог:")
    print(f"  exported: {exported_count}")
    print(f"  images: {images_dir}")
    print(f"  metadata: {metadata_path}")

    if create_empty_labels:
        print(f"  labels: {labels_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Экспорт RAW-кадров из видео для разметки price_tag bbox"
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="data/labeling/raw_frames")
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--min-quality", type=float, default=0.10)
    parser.add_argument("--prefix", default="lenta")
    parser.add_argument(
        "--create-empty-labels",
        action="store_true",
        help="Создать пустые .txt labels для YOLO-разметки",
    )

    args = parser.parse_args()

    export_raw_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        target_fps=args.target_fps,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
        min_quality=args.min_quality,
        prefix=args.prefix,
        create_empty_labels=args.create_empty_labels,
    )


if __name__ == "__main__":
    main()