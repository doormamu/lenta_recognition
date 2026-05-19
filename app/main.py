from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Все это лежит на сервере/локально в проекте.
# Пользователь через API загружает ТОЛЬКО видео.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "detector" / "price_tag_detector.pt"
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "data" / "input" / "references" / "products.csv"

RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"

# Настройки YOLO-детекции.
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.50
YOLO_IMAGE_SIZE = 1280
YOLO_DEVICE = None  # можно "cpu", "0", "mps", если нужно

# Настройки выбора кадров.
TARGET_FPS = 2.0
MAX_FRAMES = 50
MAX_CANDIDATES_PER_FRAME = 30

# Настройки recognition/parsing.
RECOGNITION_MIN_SCORE = 0.35
PADDING_RATIO = 0.12
UPSCALE_FACTOR = 1.8
MAX_PARSING_VARIANTS = 2
FRAME_ROTATION = "rot90_ccw"
REFERENCE_MIN_SCORE = 55.0


app = FastAPI(
    title="Lenta Price Recognition",
    description="Сырое видео → YOLO → OCR/парсинг → справочник → итоговый CSV",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
def process_video(
    video: UploadFile = File(...),
) -> FileResponse:
    """
    Единственная точка входа.

    Пользователь загружает только сырое видео.
    Все остальное приложение берет само:
    - YOLO-модель из models/detector/price_tag_detector.pt;
    - справочник products.csv из data/input/references/products.csv, если он есть.
    """

    run_id = time.strftime("%Y%m%d_%H%M%S")
    video_stem = Path(video.filename or "video").stem
    safe_stem = safe_filename(video_stem)

    run_dir = RUNTIME_DIR / f"{safe_stem}_{run_id}"

    uploads_dir = run_dir / "uploads"
    detection_dir = run_dir / "detection"
    recognition_dir = run_dir / "recognition"
    final_dir = run_dir / "final"

    uploads_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)
    recognition_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    video_path = uploads_dir / safe_filename(video.filename or "input.mp4")
    save_upload(video, video_path)

    try:
        detection_report_path = run_yolo_detection(
            video_path=video_path,
            output_dir=detection_dir,
        )

        recognition_csv = run_recognition(
            video_path=video_path,
            detection_report_path=detection_report_path,
            output_dir=recognition_dir,
        )

        final_csv = run_export(
            recognition_csv=recognition_csv,
            output_dir=final_dir,
            filename=safe_stem,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return FileResponse(
        path=str(final_csv),
        filename=final_csv.name,
        media_type="text/csv",
    )


def run_yolo_detection(
    video_path: Path,
    output_dir: Path,
) -> Path:
    """
    Запускает YOLO на raw-видео и формирует detection_report.csv.

    Детекция выполняется на исходных кадрах, чтобы bbox-координаты
    оставались в системе координат исходного видео.
    """

    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {DEFAULT_MODEL_PATH}\n"
            "Положи модель сюда: models/detector/price_tag_detector.pt"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(DEFAULT_MODEL_PATH))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0 or TARGET_FPS <= 0:
        frame_step = 1
    else:
        frame_step = max(1, int(round(source_fps / TARGET_FPS)))

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
        "class_id",
        "class_name",
        "model_path",
    ]

    processed_frames = 0
    absolute_frame_index = -1

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                break

            absolute_frame_index += 1

            if absolute_frame_index % frame_step != 0:
                continue

            frame_height, frame_width = frame.shape[:2]
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = timestamp_ms / 1000.0 if timestamp_ms else 0.0

            predict_kwargs = {
                "source": frame,
                "conf": YOLO_CONFIDENCE,
                "iou": YOLO_IOU,
                "imgsz": YOLO_IMAGE_SIZE,
                "verbose": False,
            }

            if YOLO_DEVICE:
                predict_kwargs["device"] = YOLO_DEVICE

            results = model.predict(**predict_kwargs)

            candidates = []

            if results:
                result = results[0]
                boxes = getattr(result, "boxes", None)

                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].detach().cpu().numpy()
                        confidence = float(box.conf[0].detach().cpu().item())

                        class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
                        class_name = get_class_name(model, class_id)

                        x_min, y_min, x_max, y_max = map(float, xyxy)

                        x_min = clamp_int(x_min, 0, frame_width - 1)
                        y_min = clamp_int(y_min, 0, frame_height - 1)
                        x_max = clamp_int(x_max, 0, frame_width - 1)
                        y_max = clamp_int(y_max, 0, frame_height - 1)

                        if x_max <= x_min or y_max <= y_min:
                            continue

                        candidates.append(
                            {
                                "score": confidence,
                                "x_min": x_min,
                                "y_min": y_min,
                                "x_max": x_max,
                                "y_max": y_max,
                                "width": x_max - x_min,
                                "height": y_max - y_min,
                                "class_id": class_id,
                                "class_name": class_name,
                            }
                        )

            candidates.sort(key=lambda item: item["score"], reverse=True)
            candidates = candidates[:MAX_CANDIDATES_PER_FRAME]

            for candidate_index, candidate in enumerate(candidates):
                writer.writerow(
                    {
                        "frame_index": absolute_frame_index,
                        "timestamp_ms": round(float(timestamp_ms), 3),
                        "timestamp_sec": round(float(timestamp_sec), 3),
                        "candidate_index": candidate_index,
                        "source": "model_yolo",
                        "score": round(float(candidate["score"]), 6),
                        "x_min": candidate["x_min"],
                        "y_min": candidate["y_min"],
                        "x_max": candidate["x_max"],
                        "y_max": candidate["y_max"],
                        "width": candidate["width"],
                        "height": candidate["height"],
                        "class_id": candidate["class_id"],
                        "class_name": candidate["class_name"],
                        "model_path": str(DEFAULT_MODEL_PATH),
                    }
                )

            processed_frames += 1

            if processed_frames >= MAX_FRAMES:
                break

    cap.release()

    if not report_path.exists():
        raise RuntimeError("YOLO detection failed: detection_report.csv was not created")

    return report_path


def run_recognition(
    video_path: Path,
    detection_report_path: Path,
    output_dir: Path,
) -> Path:
    """
    Запускает текущий recognition-скрипт:
    detection_report.csv → crop → OCR → field_parser → products.csv → recognition_results.csv.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "tests/cv_module_recognition_from_detection.py",
        "--video",
        str(video_path),
        "--detection-report",
        str(detection_report_path),
        "--output",
        str(output_dir),
        "--max-rows",
        str(MAX_FRAMES * MAX_CANDIDATES_PER_FRAME),
        "--min-score",
        str(RECOGNITION_MIN_SCORE),
        "--padding-ratio",
        str(PADDING_RATIO),
        "--upscale-factor",
        str(UPSCALE_FACTOR),
        "--max-parsing-variants",
        str(MAX_PARSING_VARIANTS),
        "--frame-rotation",
        FRAME_ROTATION,
        "--reference-min-score",
        str(REFERENCE_MIN_SCORE),
        "--disable-barcode",
    ]

    if DEFAULT_PRODUCTS_PATH.exists():
        command.extend(
            [
                "--reference-path",
                str(DEFAULT_PRODUCTS_PATH),
            ]
        )

    run_command(command)

    recognition_csv = output_dir / "recognition_results.csv"

    if not recognition_csv.exists():
        raise RuntimeError(f"recognition_results.csv not found: {recognition_csv}")

    return recognition_csv


def run_export(
    recognition_csv: Path,
    output_dir: Path,
    filename: str,
) -> Path:
    """
    Формирует финальный CSV с фиксированным набором колонок.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{safe_filename(filename)}.csv"

    command = [
        sys.executable,
        "-m",
        "cv_module.export.csv_exporter",
        "--input",
        str(recognition_csv),
        "--output",
        str(output_csv),
        "--filename",
        filename,
    ]

    run_command(command)

    if not output_csv.exists():
        raise RuntimeError(f"Final CSV not found: {output_csv}")

    return output_csv


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    output = "\n".join(
        part
        for part in [completed.stdout, completed.stderr]
        if part
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\n"
            + output
        )

    return output


def save_upload(upload_file: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        shutil.copyfileobj(upload_file.file, file)


def safe_filename(filename: str) -> str:
    return (
        Path(filename).name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def clamp_int(value: float, min_value: int, max_value: int) -> int:
    return int(max(min_value, min(max_value, round(value))))


def get_class_name(model: YOLO, class_id: int) -> str:
    names = getattr(model, "names", None)

    if isinstance(names, dict):
        return str(names.get(class_id, "price_tag"))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return "price_tag"