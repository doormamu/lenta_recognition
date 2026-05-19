from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.price_tag_detector import (  # noqa: E402
    PriceTagDetector,
    draw_detection_debug,
)
from cv_module.video.frame_sampler import sample_video_frames  # noqa: E402
from cv_module.video.reader import get_video_metadata  # noqa: E402


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


def save_code_crops(
    frame,
    codes,
    output_dir: Path,
    frame_index: int,
) -> None:
    codes_dir = output_dir / "code_crops"
    codes_dir.mkdir(parents=True, exist_ok=True)

    for code_index, code in enumerate(codes):
        bbox = code.bbox

        crop = frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

        if crop.size == 0:
            continue

        filename = (
            f"frame_{frame_index:06d}_"
            f"code_{code_index:03d}_"
            f"{code.code_type}_"
            f"confidence_{code.confidence:.2f}.jpg"
        )

        cv2.imwrite(str(codes_dir / filename), crop)


def save_report(
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


def save_codes_report(
    rows: list[dict],
    output_dir: Path,
) -> None:
    report_path = output_dir / "codes_report.csv"

    fieldnames = [
        "frame_index",
        "timestamp_ms",
        "timestamp_sec",
        "code_index",
        "code_type",
        "value",
        "confidence",
        "source",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
    ]

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
        default=4.0,
        help="Сколько кадров в секунду предварительно просматривать",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.5,
        help="Размер окна, внутри которого выбирается лучший кадр",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help="Максимальное число кадров для проверки detection-модуля",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.15,
        help="Минимальная оценка качества кадра",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=80,
        help="Максимальное число кандидатов на одном кадре",
    )

    parser.add_argument(
        "--min-candidate-score",
        type=float,
        default=0.50,
        help="Минимальная итоговая уверенность кандидата",
    )

    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Сохранять вырезанные области кандидатов",
    )

    parser.add_argument(
        "--save-code-crops",
        action="store_true",
        help="Сохранять вырезанные области QR/штрихкодов",
    )

    parser.add_argument(
        "--enable-code-detection",
        action="store_true",
        help="Включить медленный поиск QR/штрихкодов внутри найденных кандидатов",
    )

    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Исправлять широкоугольную дисторсию камеры",
    )

    parser.add_argument(
        "--orientation",
        default="none",
        choices=["none", "auto", "rot90_ccw", "rot90_cw", "rot180"],
        help="Поворот кадров перед detection",
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
        enable_undistort=args.undistort,
        orientation=args.orientation,
    )

    print()
    print(f"Выбрано кадров для проверки: {len(sampled_frames)}")

    detector = PriceTagDetector(
        max_candidates=args.max_candidates,
        min_candidate_score=args.min_candidate_score,
        enable_code_detection=args.enable_code_detection,
    )

    candidate_report_rows: list[dict] = []
    codes_report_rows: list[dict] = []

    total_candidates = 0
    total_codes = 0

    for sampled in sampled_frames:
        detection_result = detector.detect(
            frame=sampled.image,
            frame_index=sampled.frame_index,
            timestamp_ms=sampled.timestamp_ms,
        )

        total_candidates += len(detection_result.candidates)
        total_codes += len(detection_result.codes)

        debug_path = (
            debug_frames_dir
            / (
                f"frame_{sampled.frame_index:06d}_"
                f"time_{sampled.timestamp_ms / 1000.0:08.2f}_"
                f"candidates_{len(detection_result.candidates):03d}_"
                f"codes_{len(detection_result.codes):02d}.jpg"
            )
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

        if args.save_code_crops:
            save_code_crops(
                frame=sampled.image,
                codes=detection_result.codes,
                output_dir=output_dir,
                frame_index=sampled.frame_index,
            )

        for candidate_index, candidate in enumerate(detection_result.candidates):
            bbox = candidate.bbox

            candidate_report_rows.append(
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

        for code_index, code in enumerate(detection_result.codes):
            bbox = code.bbox

            codes_report_rows.append(
                {
                    "frame_index": sampled.frame_index,
                    "timestamp_ms": round(sampled.timestamp_ms, 2),
                    "timestamp_sec": round(sampled.timestamp_ms / 1000.0, 2),
                    "code_index": code_index,
                    "code_type": code.code_type,
                    "value": code.value,
                    "confidence": round(code.confidence, 4),
                    "source": code.source,
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                    "width": bbox.width,
                    "height": bbox.height,
                }
            )

        print(
            f"frame={sampled.frame_index:06d}, "
            f"time={sampled.timestamp_ms / 1000.0:8.2f}s, "
            f"candidates={len(detection_result.candidates):3d}, "
            f"codes={len(detection_result.codes):2d}, "
            f"debug={debug_path}"
        )

    save_report(candidate_report_rows, output_dir)
    save_codes_report(codes_report_rows, output_dir)

    print()
    print("Итог:")
    print(f"  frames: {len(sampled_frames)}")
    print(f"  total_candidates: {total_candidates}")
    print(f"  total_codes: {total_codes}")
    print()
    print(f"Debug-кадры сохранены в: {debug_frames_dir}")
    print(f"Отчет по кандидатам сохранен в: {output_dir / 'detection_report.csv'}")
    print(f"Отчет по QR/штрихкодам сохранен в: {output_dir / 'codes_report.csv'}")

    if args.save_crops:
        print(f"Crop-картинки сохранены в: {output_dir / 'crops'}")

    if args.save_code_crops:
        print(f"Crop QR/штрихкодов сохранены в: {output_dir / 'code_crops'}")


if __name__ == "__main__":
    main()
'''
запуск

python tests/cv_module_detection.py \
  --video data/input/labeled/25_2-10.mp4 \
  --output data/output/detection_debug \
  --target-fps 4 \
  --window-sec 0.5 \
  --max-frames 80 \
  --min-quality 0.15 \
  --max-candidates 20 \
  --min-candidate-score 0.50 \
  --undistort \
  --orientation auto \
  --save-crops
'''
