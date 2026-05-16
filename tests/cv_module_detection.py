from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.price_tag_detector import (
    PriceTagDetector,
    draw_detection_debug,
)
from cv_module.video.frame_sampler import sample_video_frames
from cv_module.video.reader import get_video_metadata


def save_candidate_crops(
    frame,
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


def save_report(
    rows: list[dict],
    output_dir: Path,
) -> None:
    report_path = output_dir / "detection_report.csv"

    if not rows:
        with report_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
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
                    "aspect_ratio",
                    "evidence",
                ]
            )
        return

    fieldnames = list(rows[0].keys())

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Тестирование detection-модуля на выбранных кадрах видео"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Путь до видеофайла",
    )

    parser.add_argument(
        "--output",
        default="data/output/detection_debug",
        help="Папка для debug-результатов",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=2.0,
        help="Сколько кадров в секунду предварительно просматривать",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Размер окна, внутри которого выбирается лучший кадр",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Максимальное число кадров для проверки detection-модуля",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.25,
        help="Минимальная оценка качества кадра",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=80,
        help="Максимальное число кандидатов на одном кадре",
    )

    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Сохранять вырезанные области кандидатов",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)

    debug_frames_dir = output_dir / "debug_frames"
    debug_frames_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_video_metadata(video_path)

    print("Видео:")
    print(f"  path: {metadata.path}")
    print(f"  fps: {metadata.fps:.2f}")
    print(f"  frame_count: {metadata.frame_count}")
    print(f"  size: {metadata.width}x{metadata.height}")
    print(f"  duration_sec: {metadata.duration_sec:.2f}")

    sampled_frames = sample_video_frames(
        video_path=video_path,
        target_fps=args.target_fps,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
        min_quality_score=args.min_quality,
    )

    print()
    print(f"Выбрано кадров для проверки: {len(sampled_frames)}")

    detector = PriceTagDetector(
        max_candidates=args.max_candidates,
    )

    report_rows: list[dict] = []

    for sampled in sampled_frames:
        detection_result = detector.detect(
            frame=sampled.image,
            frame_index=sampled.frame_index,
            timestamp_ms=sampled.timestamp_ms,
        )

        debug_path = (
            debug_frames_dir
            / f"frame_{sampled.frame_index:06d}_"
              f"time_{sampled.timestamp_ms / 1000.0:08.2f}_"
              f"candidates_{len(detection_result.candidates):03d}.jpg"
        )

        draw_detection_debug(
            frame=sampled.image,
            result=detection_result,
            output_path=debug_path,
        )

        if args.save_crops:
            save_candidate_crops(
                frame=sampled.image,
                candidates=detection_result.candidates,
                output_dir=output_dir,
                frame_index=sampled.frame_index,
            )

        for candidate_index, candidate in enumerate(detection_result.candidates):
            bbox = candidate.bbox

            report_rows.append(
                {
                    "frame_index": sampled.frame_index,
                    "timestamp_ms": round(sampled.timestamp_ms, 2),
                    "timestamp_sec": round(sampled.timestamp_ms / 1000.0, 2),
                    "candidate_index": candidate_index,
                    "source": candidate.source,
                    "score": round(candidate.score, 4),
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                    "width": bbox.width,
                    "height": bbox.height,
                    "aspect_ratio": round(bbox.aspect_ratio, 4),
                    "evidence": json.dumps(
                        candidate.evidence,
                        ensure_ascii=False,
                    ),
                }
            )

        print(
            f"frame={sampled.frame_index:06d}, "
            f"time={sampled.timestamp_ms / 1000.0:8.2f}s, "
            f"candidates={len(detection_result.candidates):3d}, "
            f"codes={len(detection_result.codes):2d}, "
            f"debug={debug_path}"
        )

    save_report(report_rows, output_dir)

    print()
    print(f"Debug-кадры сохранены в: {debug_frames_dir}")
    print(f"Отчет сохранен в: {output_dir / 'detection_report.csv'}")

    if args.save_crops:
        print(f"Crop-картинки сохранены в: {output_dir / 'crops'}")


if __name__ == "__main__":
    main()


'''
запуск

python tests/cv_module_detection.py \
  --video data/input/labeled/25_2-10.mp4 \
  --output data/output/detection_debug \
  --target-fps 2 \
  --window-sec 1 \
  --max-frames 30 \
  --min-quality 0.25 \
  --save-crops
'''