import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.video.frame_sampler import sample_video_frames, save_sampled_frames
from cv_module.video.reader import get_video_metadata


def save_report(sampled_frames, output_dir: Path) -> None:
    report_path = output_dir / "frame_sampling_report.csv"

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "frame_index",
                "timestamp_ms",
                "timestamp_sec",
                "quality_score",
                "sharpness",
                "brightness",
                "contrast",
                "glare_ratio",
                "dark_ratio",
                "price_tag_ratio",
            ]
        )

        for item in sampled_frames:
            writer.writerow(
                [
                    item.frame_index,
                    round(item.timestamp_ms, 2),
                    round(item.timestamp_ms / 1000.0, 2),
                    round(item.quality.score, 4),
                    round(item.quality.sharpness, 2),
                    round(item.quality.brightness, 2),
                    round(item.quality.contrast, 2),
                    round(item.quality.glare_ratio, 4),
                    round(item.quality.dark_ratio, 4),
                    round(item.quality.price_tag_ratio, 4),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка выбора кадров из видео"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Путь до видеофайла",
    )

    parser.add_argument(
        "--output",
        default="data/output/frame_sampling",
        help="Папка для сохранения выбранных кадров",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=2.0,
        help="Сколько кадров в секунду предварительно смотреть",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Размер временного окна, внутри которого выбирается лучший кадр",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Максимальное число выбранных кадров",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.25,
        help="Минимальная оценка качества кадра от 0 до 1",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    save_sampled_frames(
        sampled_frames=sampled_frames,
        output_dir=output_dir,
        prefix=video_path.stem,
    )

    save_report(sampled_frames, output_dir)

    print()
    print(f"Выбрано кадров: {len(sampled_frames)}")
    print(f"Кадры сохранены в: {output_dir}")
    print(f"Отчет сохранен в: {output_dir / 'frame_sampling_report.csv'}")


if __name__ == "__main__":
    main()


'''
запуск

python tests/cv_module_video.py \
  --video data/input/labeled/25_2-10.mp4 \
  --output data/output/frame_sampling \
  --target-fps 2 \
  --window-sec 1 \
  --max-frames 100 \
  --min-quality 0.25
'''
