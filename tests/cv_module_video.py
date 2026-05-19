from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.video.preprocessing import build_preprocessor_for_video  # noqa: E402


def make_before_after_collage(raw, processed):
    raw_small = resize_to_height(raw, 480)
    processed_small = resize_to_height(processed, 480)

    h = max(raw_small.shape[0], processed_small.shape[0])

    raw_small = pad_to_height(raw_small, h)
    processed_small = pad_to_height(processed_small, h)

    cv2.putText(
        raw_small,
        "RAW",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    cv2.putText(
        processed_small,
        "UNDISTORT + ROTATE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    return np.hstack([raw_small, processed_small])


def resize_to_height(image, height):
    h, w = image.shape[:2]

    scale = height / h
    new_w = int(w * scale)

    return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)


def pad_to_height(image, height):
    h, w = image.shape[:2]

    if h == height:
        return image

    pad = np.zeros((height - h, w, 3), dtype=image.dtype)

    return np.vstack([image, pad])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="data/output/video_preprocessing_debug")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--orientation", default="auto")
    parser.add_argument("--undistort", action="store_true")

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor_for_video(
        video_path=video_path,
        enable_undistort=args.undistort,
        orientation=args.orientation,
    )

    print("rotation_mode:", preprocessor.rotation_mode)
    print("enable_undistort:", preprocessor.enable_undistort)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_count <= 0:
        frame_indices = list(range(args.frames))
    else:
        frame_indices = np.linspace(
            0,
            max(0, frame_count - 1),
            num=args.frames,
            dtype=int,
        ).tolist()

    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, raw = cap.read()

        if not ok or raw is None:
            continue

        processed = preprocessor.process(raw)

        collage = make_before_after_collage(raw, processed)

        output_path = output_dir / f"frame_{frame_index:06d}_before_after.jpg"

        cv2.imwrite(str(output_path), collage)

        print(output_path)

    cap.release()


if __name__ == "__main__":
    main()

'''
python tests/cv_module_video.py \
  --video data/input/labeled/25_2-10.mp4 \
  --output data/output/video_preprocessing_debug \
  --frames 12 \
  --undistort \
  --orientation auto
'''