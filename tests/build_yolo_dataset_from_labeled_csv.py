from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)


@dataclass(frozen=True)
class DatasetItem:
    image_name: str
    video_path: Path
    csv_path: Path
    frame_timestamp_ms: float
    frame_index: int
    boxes: list[Box]
    width: int
    height: int


def parse_number(value: Any) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace("\xa0", "")
    text = text.replace(" ", "")
    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def safe_stem(value: str) -> str:
    value = Path(str(value).strip()).stem
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", value)
    return value.strip("_") or "video"


def collect_csv_files(csv_dir: Path) -> list[Path]:
    if csv_dir.is_file():
        return [csv_dir]

    return sorted(csv_dir.glob("*.csv"))


def resolve_video_path(
    csv_path: Path,
    videos_dir: Path,
) -> Path:
    """
    Основное правило:
    data/output/labeled/25_2-10.csv
    соответствует
    data/input/labeled/25_2-10.mp4

    filename внутри CSV может быть грязным: с пробелами, без .mp4,
    или вроде 25_12-20/2.mp4, поэтому сначала используем имя CSV.
    """

    csv_stem = csv_path.stem.strip()

    candidates: list[Path] = []

    for ext in VIDEO_EXTENSIONS:
        candidates.append(videos_dir / f"{csv_stem}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for ext in VIDEO_EXTENSIONS:
        matches = list(videos_dir.rglob(f"{csv_stem}{ext}"))

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Не нашел видео для {csv_path.name}. "
        f"Ожидал что-то вроде {videos_dir / (csv_stem + '.mp4')}"
    )


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    cap.release()

    if fps <= 0:
        fps = 25.0

    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def timestamp_to_frame_index(timestamp_ms: float, fps: float, frame_count: int) -> int:
    frame_index = int(round(timestamp_ms / 1000.0 * fps))

    if frame_count > 0:
        frame_index = max(0, min(frame_index, frame_count - 1))

    return frame_index


def read_frame_by_index(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def clamp_box(box: Box, width: int, height: int) -> Box | None:
    x_min = max(0.0, min(box.x_min, width - 1.0))
    y_min = max(0.0, min(box.y_min, height - 1.0))
    x_max = max(0.0, min(box.x_max, width - 1.0))
    y_max = max(0.0, min(box.y_max, height - 1.0))

    if x_max < x_min:
        x_min, x_max = x_max, x_min

    if y_max < y_min:
        y_min, y_max = y_max, y_min

    clamped = Box(x_min, y_min, x_max, y_max)

    if clamped.width <= 2 or clamped.height <= 2:
        return None

    return clamped


def row_to_box(row: pd.Series, width: int, height: int) -> Box | None:
    x_min = parse_number(row.get("x_min"))
    y_min = parse_number(row.get("y_min"))
    x_max = parse_number(row.get("x_max"))
    y_max = parse_number(row.get("y_max"))

    if None in {x_min, y_min, x_max, y_max}:
        return None

    return clamp_box(
        Box(
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
        ),
        width=width,
        height=height,
    )


def box_to_yolo_line(class_id: int, box: Box, width: int, height: int) -> str:
    x_center = ((box.x_min + box.x_max) / 2.0) / width
    y_center = ((box.y_min + box.y_max) / 2.0) / height
    box_width = box.width / width
    box_height = box.height / height

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{box_width:.6f} "
        f"{box_height:.6f}"
    )


def read_labeled_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path)

    required = {"frame_timestamp", "x_min", "y_min", "x_max", "y_max"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"В {csv_path.name} не хватает колонок: {sorted(missing)}. "
            f"Есть колонки: {list(df.columns)}"
        )

    return df


def build_items_from_csv(
    csv_path: Path,
    videos_dir: Path,
) -> list[DatasetItem]:
    df = read_labeled_csv(csv_path)

    video_path = resolve_video_path(
        csv_path=csv_path,
        videos_dir=videos_dir,
    )

    video_info = get_video_info(video_path)

    items: list[DatasetItem] = []

    for timestamp_value, group in df.groupby("frame_timestamp", dropna=True):
        timestamp_ms = parse_number(timestamp_value)

        if timestamp_ms is None:
            continue

        frame_index = timestamp_to_frame_index(
            timestamp_ms=timestamp_ms,
            fps=video_info.fps,
            frame_count=video_info.frame_count,
        )

        boxes: list[Box] = []

        for _, row in group.iterrows():
            box = row_to_box(
                row=row,
                width=video_info.width,
                height=video_info.height,
            )

            if box is not None:
                boxes.append(box)

        if not boxes:
            continue

        image_name = (
            f"{safe_stem(csv_path.stem)}_"
            f"t_{int(round(timestamp_ms)):08d}_"
            f"f_{frame_index:06d}.jpg"
        )

        items.append(
            DatasetItem(
                image_name=image_name,
                video_path=video_path,
                csv_path=csv_path,
                frame_timestamp_ms=timestamp_ms,
                frame_index=frame_index,
                boxes=boxes,
                width=video_info.width,
                height=video_info.height,
            )
        )

    return items


def draw_debug_boxes(frame, boxes: list[Box]):
    debug = frame.copy()

    for idx, box in enumerate(boxes):
        x_min = int(round(box.x_min))
        y_min = int(round(box.y_min))
        x_max = int(round(box.x_max))
        y_max = int(round(box.y_max))

        cv2.rectangle(
            debug,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            debug,
            f"price_tag {idx}",
            (x_min, max(24, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

    return debug


def write_dataset_yaml(output_dir: Path, class_name: str) -> None:
    text = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: {class_name}
"""

    (output_dir / "dataset.yaml").write_text(text, encoding="utf-8")


def write_label_file(
    label_path: Path,
    boxes: list[Box],
    image_width: int,
    image_height: int,
    class_id: int,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        box_to_yolo_line(
            class_id=class_id,
            box=box,
            width=image_width,
            height=image_height,
        )
        for box in boxes
    ]

    label_path.write_text("\n".join(lines), encoding="utf-8")


def save_metadata(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "image_name",
        "video_path",
        "csv_path",
        "frame_timestamp_ms",
        "frame_timestamp_sec",
        "frame_index",
        "width",
        "height",
        "boxes_count",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_items(
    items: list[DatasetItem],
    output_dir: Path,
    split_name: str,
    class_id: int,
    draw_debug: bool,
) -> list[dict]:
    metadata_rows: list[dict] = []

    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    debug_dir = output_dir / "debug" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if draw_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        frame = read_frame_by_index(
            video_path=item.video_path,
            frame_index=item.frame_index,
        )

        if frame is None:
            print(f"skip: cannot read frame {item.frame_index} from {item.video_path}")
            continue

        image_path = images_dir / item.image_name
        label_path = labels_dir / f"{Path(item.image_name).stem}.txt"

        cv2.imwrite(str(image_path), frame)

        write_label_file(
            label_path=label_path,
            boxes=item.boxes,
            image_width=item.width,
            image_height=item.height,
            class_id=class_id,
        )

        if draw_debug:
            debug = draw_debug_boxes(frame, item.boxes)
            cv2.imwrite(str(debug_dir / item.image_name), debug)

        metadata_rows.append(
            {
                "split": split_name,
                "image_name": item.image_name,
                "video_path": str(item.video_path),
                "csv_path": str(item.csv_path),
                "frame_timestamp_ms": round(item.frame_timestamp_ms, 2),
                "frame_timestamp_sec": round(item.frame_timestamp_ms / 1000.0, 3),
                "frame_index": item.frame_index,
                "width": item.width,
                "height": item.height,
                "boxes_count": len(item.boxes),
            }
        )

        print(
            f"{split_name}: {item.image_name}, "
            f"frame={item.frame_index}, "
            f"t={item.frame_timestamp_ms / 1000.0:.2f}s, "
            f"boxes={len(item.boxes)}"
        )

    return metadata_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сборка YOLO-датасета из CSV-разметки best frames"
    )

    parser.add_argument(
        "--csv-dir",
        default="data/output/labeled",
        help="Папка с CSV-разметкой",
    )

    parser.add_argument(
        "--videos-dir",
        default="data/input/labeled",
        help="Папка с одноименными видео",
    )

    parser.add_argument(
        "--output",
        default="data/datasets/price_tags_yolo",
        help="Куда сохранить YOLO dataset",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--class-name",
        default="price_tag",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--draw-debug",
        action="store_true",
        help="Сохранить debug-картинки с bbox поверх кадров",
    )

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output)

    csv_files = collect_csv_files(csv_dir)

    if not csv_files:
        raise FileNotFoundError(f"CSV-файлы не найдены: {csv_dir}")

    all_items: list[DatasetItem] = []

    for csv_path in csv_files:
        items = build_items_from_csv(
            csv_path=csv_path,
            videos_dir=videos_dir,
        )

        print(f"{csv_path.name}: frames={len(items)}, boxes={sum(len(i.boxes) for i in items)}")

        all_items.extend(items)

    if not all_items:
        raise RuntimeError("Не удалось собрать ни одного размеченного кадра")

    random.seed(args.seed)
    random.shuffle(all_items)

    val_count = int(round(len(all_items) * args.val_ratio))

    val_items = all_items[:val_count]
    train_items = all_items[val_count:]

    metadata_rows: list[dict] = []

    metadata_rows.extend(
        save_items(
            items=train_items,
            output_dir=output_dir,
            split_name="train",
            class_id=args.class_id,
            draw_debug=args.draw_debug,
        )
    )

    metadata_rows.extend(
        save_items(
            items=val_items,
            output_dir=output_dir,
            split_name="val",
            class_id=args.class_id,
            draw_debug=args.draw_debug,
        )
    )

    write_dataset_yaml(
        output_dir=output_dir,
        class_name=args.class_name,
    )

    save_metadata(
        rows=metadata_rows,
        output_path=output_dir / "dataset_metadata.csv",
    )

    summary = {
        "csv_files": [str(path) for path in csv_files],
        "videos_dir": str(videos_dir),
        "output_dir": str(output_dir),
        "items_total": len(all_items),
        "items_train": len(train_items),
        "items_val": len(val_items),
        "boxes_total": sum(len(item.boxes) for item in all_items),
        "class_id": args.class_id,
        "class_name": args.class_name,
    }

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Итог:")
    print(f"  csv files: {len(csv_files)}")
    print(f"  images total: {len(all_items)}")
    print(f"  train: {len(train_items)}")
    print(f"  val: {len(val_items)}")
    print(f"  boxes total: {summary['boxes_total']}")
    print()
    print(f"YOLO dataset: {output_dir}")
    print(f"Dataset YAML: {output_dir / 'dataset.yaml'}")
    print(f"Metadata: {output_dir / 'dataset_metadata.csv'}")

    if args.draw_debug:
        print(f"Debug images: {output_dir / 'debug'}")


if __name__ == "__main__":
    main()