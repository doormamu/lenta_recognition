from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.model_detector import (  # noqa: E402
    YOLOPriceTagDetector,
    draw_model_candidates_debug,
)


@dataclass(frozen=True)
class RawSampledFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray
    quality_score: float


def estimate_frame_quality(frame: np.ndarray) -> float:
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


def sample_raw_video_frames(
    video_path: Path,
    target_fps: float,
    window_sec: float,
    max_frames: int | None,
    min_quality_score: float,
) -> list[RawSampledFrame]:
    """
    Выбирает кадры из видео без preprocessing.

    Важно:
    - не поворачивает;
    - не undistort;
    - координаты модели остаются координатами исходного видео.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(target_fps, 0.1))))
    window_ms = max(1.0, window_sec * 1000.0)

    selected: list[RawSampledFrame] = []

    current_window_id: int | None = None
    best_in_window: RawSampledFrame | None = None

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        if frame_index % step != 0:
            frame_index += 1
            continue

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        quality_score = estimate_frame_quality(frame)
        window_id = int(timestamp_ms // window_ms)

        sampled = RawSampledFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            image=frame,
            quality_score=quality_score,
        )

        if current_window_id is None:
            current_window_id = window_id

        if window_id != current_window_id:
            if best_in_window is not None and best_in_window.quality_score >= min_quality_score:
                selected.append(best_in_window)

                if max_frames is not None and len(selected) >= max_frames:
                    break

            current_window_id = window_id
            best_in_window = sampled
        else:
            if best_in_window is None or sampled.quality_score > best_in_window.quality_score:
                best_in_window = sampled

        frame_index += 1

    if max_frames is None or len(selected) < max_frames:
        if best_in_window is not None and best_in_window.quality_score >= min_quality_score:
            selected.append(best_in_window)

    cap.release()

    if max_frames is not None:
        selected = selected[:max_frames]

    return selected


def save_candidate_crops(
    frame: np.ndarray,
    candidates,
    output_dir: Path,
    frame_index: int,
) -> None:
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for candidate_index, candidate in enumerate(candidates):
        bbox = candidate.bbox

        crop = frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

        if crop.size == 0:
            continue

        filename = (
            f"frame_{frame_index:06d}_"
            f"candidate_{candidate_index:03d}_"
            f"{candidate.source}_"
            f"score_{candidate.score:.2f}.jpg"
        )

        cv2.imwrite(str(crops_dir / filename), crop)


def save_detection_report(
    rows: list[dict],
    output_dir: Path,
) -> None:
    report_path = output_dir / "detection_report.csv"

    fieldnames = [
        "frame_index",
        "timestamp_ms",
        "timestamp_sec",
        "candidate_index",
        "source",
        "score",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
        "aspect_ratio",
        "evidence",
    ]

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO detection ценников на RAW-кадрах видео"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Путь до видео",
    )

    parser.add_argument(
        "--model",
        default="models/detector/price_tag_detector_yolo11n.pt",
        help="Путь до best.pt / обученной YOLO-модели",
    )

    parser.add_argument(
        "--output",
        default="data/output/model_detection_debug",
        help="Папка для результатов",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности YOLO",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU для NMS",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Например: cpu, 0, mps",
    )

    parser.add_argument(
        "--save-crops",
        action="store_true",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_frames_dir = output_dir / "debug_frames"
    debug_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    print("Видео:")
    print(f"  path: {video_path}")
    print(f"  fps: {fps:.2f}")
    print(f"  frame_count: {frame_count}")
    print(f"  size: {width}x{height}")
    print(f"  duration_sec: {duration_sec:.2f}")
    print()
    print("Детекция идет на RAW-кадрах. Координаты bbox соответствуют исходному видео.")
    print()

    sampled_frames = sample_raw_video_frames(
        video_path=video_path,
        target_fps=args.target_fps,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
        min_quality_score=args.min_quality,
    )

    print(f"Выбрано кадров: {len(sampled_frames)}")
    print()

    detector = YOLOPriceTagDetector(
        model_path=model_path,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        target_class_names=None,
    )

    report_rows: list[dict] = []
    total_candidates = 0

    for sampled in sampled_frames:
        candidates = detector.detect_as_candidates(
            frame=sampled.image,
            frame_index=sampled.frame_index,
            timestamp_ms=sampled.timestamp_ms,
            max_candidates=args.max_candidates,
        )

        total_candidates += len(candidates)

        debug_path = (
            debug_frames_dir
            / (
                f"frame_{sampled.frame_index:06d}_"
                f"time_{sampled.timestamp_ms / 1000.0:08.2f}_"
                f"candidates_{len(candidates):03d}.jpg"
            )
        )

        draw_model_candidates_debug(
            frame=sampled.image,
            candidates=candidates,
            output_path=debug_path,
        )

        if args.save_crops:
            save_candidate_crops(
                frame=sampled.image,
                candidates=candidates,
                output_dir=output_dir,
                frame_index=sampled.frame_index,
            )

        for candidate_index, candidate in enumerate(candidates):
            bbox = candidate.bbox

            report_rows.append(
                {
                    "frame_index": sampled.frame_index,
                    "timestamp_ms": round(sampled.timestamp_ms, 2),
                    "timestamp_sec": round(sampled.timestamp_ms / 1000.0, 2),
                    "candidate_index": candidate_index,
                    "source": candidate.source,
                    "score": round(candidate.score, 5),
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                    "width": bbox.width,
                    "height": bbox.height,
                    "aspect_ratio": round(bbox.aspect_ratio, 5),
                    "evidence": json.dumps(
                        candidate.evidence,
                        ensure_ascii=False,
                    ),
                }
            )

        print(
            f"frame={sampled.frame_index:06d}, "
            f"time={sampled.timestamp_ms / 1000.0:8.2f}s, "
            f"quality={sampled.quality_score:.3f}, "
            f"candidates={len(candidates):3d}, "
            f"debug={debug_path}"
        )

    save_detection_report(
        rows=report_rows,
        output_dir=output_dir,
    )

    summary = {
        "video": str(video_path),
        "model": str(model_path),
        "frames": len(sampled_frames),
        "total_candidates": total_candidates,
        "conf": args.conf,
        "iou": args.iou,
        "raw_coordinates": True,
    }

    (output_dir / "model_detection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Итог:")
    print(f"  frames: {len(sampled_frames)}")
    print(f"  total_candidates: {total_candidates}")
    print(f"  detection_report: {output_dir / 'detection_report.csv'}")
    print(f"  debug_frames: {debug_frames_dir}")

    if args.save_crops:
        print(f"  crops: {output_dir / 'crops'}")


if __name__ == "__main__":
    main()