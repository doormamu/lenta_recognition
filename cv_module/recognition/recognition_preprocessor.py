from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv_module.detection.candidate_merger import BoundingBox
from cv_module.recognition.frame_preprocessor import (
    PreparedFrame,
    transform_bbox_by_rotation,
)


@dataclass(frozen=True)
class RecognitionCrop:
    raw_crop: np.ndarray
    processed_crop: np.ndarray
    crop_bbox_raw: BoundingBox
    source_bbox_raw: BoundingBox
    variants: dict[str, np.ndarray] = field(default_factory=dict)
    parsing_variant_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_parsing_images(self) -> list[tuple[str, np.ndarray]]:
        result: list[tuple[str, np.ndarray]] = []

        for name in self.parsing_variant_names:
            image = self.variants.get(name)

            if image is not None and image.size > 0:
                result.append((name, image))

        if not result:
            result.append(("processed", self.processed_crop))

        return result


class RecognitionPreprocessor:
    """
    Быстрая подготовка кандидата перед OCR.

    Работает по схеме:

    raw_frame
      ↓
    frame-level preprocessing один раз на кадр
      ↓
    bbox пересчитывается под обработанный кадр
      ↓
    crop вырезается уже из обработанного кадра
      ↓
    OCR получает только 2 варианта:
        1. crop
        2. crop_rot90_ccw

    Важно:
    - исходные координаты YOLO не теряются;
    - crop_bbox_raw остается в координатах исходного видео;
    - prepared_frame используется только для внутреннего OCR-crop.
    """

    def __init__(
        self,
        padding_ratio: float = 0.12,
        min_padding_px: int = 8,
        upscale_factor: float = 1.8,
        max_parsing_variants: int = 2,
        generate_rotations: bool = False,
        enable_perspective_rectification: bool = False,
        auto_choose_best_variant: bool = False,
        frame_rotation_mode: str = "none",
    ) -> None:
        self.padding_ratio = padding_ratio
        self.min_padding_px = min_padding_px
        self.upscale_factor = upscale_factor
        self.max_parsing_variants = max_parsing_variants

        # Оставлены для совместимости со старым кодом.
        self.generate_rotations = generate_rotations
        self.enable_perspective_rectification = enable_perspective_rectification
        self.auto_choose_best_variant = auto_choose_best_variant
        self.frame_rotation_mode = frame_rotation_mode

    def extract(
        self,
        raw_frame: np.ndarray,
        bbox: BoundingBox,
        prepared_frame: PreparedFrame | None = None,
    ) -> RecognitionCrop:
        if raw_frame is None or raw_frame.size == 0:
            empty = np.zeros((1, 1, 3), dtype=np.uint8)

            return RecognitionCrop(
                raw_crop=empty,
                processed_crop=empty,
                crop_bbox_raw=BoundingBox(0, 0, 1, 1),
                source_bbox_raw=bbox,
                variants={"crop": empty},
                parsing_variant_names=["crop"],
                metadata={"error": "empty_raw_frame"},
            )

        raw_frame = _ensure_bgr(raw_frame)
        frame_height, frame_width = raw_frame.shape[:2]

        source_bbox_raw = bbox.clamp(frame_width, frame_height)

        crop_bbox_raw = self._pad_bbox(
            bbox=source_bbox_raw,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        raw_crop = raw_frame[
            crop_bbox_raw.y_min:crop_bbox_raw.y_max,
            crop_bbox_raw.x_min:crop_bbox_raw.x_max,
        ]

        if raw_crop.size == 0:
            raw_crop = np.zeros((1, 1, 3), dtype=np.uint8)

        if prepared_frame is None:
            prepared_frame = PreparedFrame(
                raw_frame=raw_frame,
                processed_frame=raw_frame,
                rotation_mode="none",
                original_width=frame_width,
                original_height=frame_height,
                processed_width=frame_width,
                processed_height=frame_height,
                metadata={"mode": "no_frame_preprocessing"},
            )

        crop_bbox_processed = transform_bbox_by_rotation(
            bbox=crop_bbox_raw,
            original_width=prepared_frame.original_width,
            original_height=prepared_frame.original_height,
            rotation_mode=prepared_frame.rotation_mode,
        ).clamp(
            prepared_frame.processed_width,
            prepared_frame.processed_height,
        )

        processed_crop = prepared_frame.processed_frame[
            crop_bbox_processed.y_min:crop_bbox_processed.y_max,
            crop_bbox_processed.x_min:crop_bbox_processed.x_max,
        ]

        if processed_crop.size == 0:
            processed_crop = raw_crop.copy()

        processed_crop = _ensure_bgr(processed_crop)

        if self.upscale_factor and self.upscale_factor != 1.0:
            processed_crop = cv2.resize(
                processed_crop,
                None,
                fx=self.upscale_factor,
                fy=self.upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

        crop_rot90_ccw = cv2.rotate(
            processed_crop,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

        variants = {
            "crop": processed_crop,
            "crop_rot90_ccw": crop_rot90_ccw,
        }

        parsing_variant_names = list(variants.keys())[: self.max_parsing_variants]

        return RecognitionCrop(
            raw_crop=raw_crop,
            processed_crop=processed_crop,
            crop_bbox_raw=crop_bbox_raw,
            source_bbox_raw=source_bbox_raw,
            variants=variants,
            parsing_variant_names=parsing_variant_names,
            metadata={
                "mode": "frame_level_fast",
                "processed_variant": "crop",
                "frame_rotation_mode": prepared_frame.rotation_mode,
                "crop_bbox_raw": _bbox_to_tuple(crop_bbox_raw),
                "crop_bbox_processed": _bbox_to_tuple(crop_bbox_processed),
                "padding_ratio": self.padding_ratio,
                "upscale_factor": self.upscale_factor,
                "parsing_variant_names": parsing_variant_names,
                "frame_metadata": prepared_frame.metadata,
            },
        )

    def _pad_bbox(
        self,
        bbox: BoundingBox,
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox:
        pad_x = max(self.min_padding_px, int(bbox.width * self.padding_ratio))
        pad_y = max(self.min_padding_px, int(bbox.height * self.padding_ratio))

        padded = BoundingBox(
            x_min=bbox.x_min - pad_x,
            y_min=bbox.y_min - pad_y,
            x_max=bbox.x_max + pad_x,
            y_max=bbox.y_max + pad_y,
        )

        return padded.clamp(frame_width, frame_height)


def parse_bbox_from_row(row: dict[str, Any]) -> BoundingBox:
    return BoundingBox(
        x_min=_int_value(row["x_min"]),
        y_min=_int_value(row["y_min"]),
        x_max=_int_value(row["x_max"]),
        y_max=_int_value(row["y_max"]),
    )


def save_recognition_crop_debug(
    recognition_crop: RecognitionCrop,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    raw_path = output_dir / f"{prefix}_raw.jpg"
    processed_path = output_dir / f"{prefix}_processed.jpg"
    collage_path = output_dir / f"{prefix}_collage.jpg"
    metadata_path = output_dir / f"{prefix}_metadata.json"

    cv2.imwrite(str(raw_path), recognition_crop.raw_crop)
    cv2.imwrite(str(processed_path), recognition_crop.processed_crop)

    paths["raw"] = str(raw_path)
    paths["processed"] = str(processed_path)

    variants_dir = output_dir / f"{prefix}_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    for name, image in recognition_crop.variants.items():
        path = variants_dir / f"{name}.jpg"
        cv2.imwrite(str(path), image)
        paths[f"variant_{name}"] = str(path)

    collage = make_variants_collage(
        variants=recognition_crop.variants,
        selected_names=recognition_crop.parsing_variant_names,
    )

    if collage is not None:
        cv2.imwrite(str(collage_path), collage)
        paths["collage"] = str(collage_path)

    metadata = {
        "source_bbox_raw": _bbox_to_tuple(recognition_crop.source_bbox_raw),
        "crop_bbox_raw": _bbox_to_tuple(recognition_crop.crop_bbox_raw),
        "metadata": recognition_crop.metadata,
        "variant_names": list(recognition_crop.variants.keys()),
        "parsing_variant_names": recognition_crop.parsing_variant_names,
    }

    metadata_path.write_text(
        _json_dumps(metadata),
        encoding="utf-8",
    )

    paths["metadata"] = str(metadata_path)

    return paths


def make_variants_collage(
    variants: dict[str, np.ndarray],
    selected_names: list[str],
    tile_height: int = 260,
) -> np.ndarray | None:
    if not variants:
        return None

    names = selected_names[:]

    for name in variants.keys():
        if name not in names:
            names.append(name)

    tiles: list[np.ndarray] = []

    for name in names:
        image = variants.get(name)

        if image is None or image.size == 0:
            continue

        tile = _resize_to_height(_ensure_bgr(image), tile_height)

        marker = "[P] " if name in selected_names else ""

        cv2.putText(
            tile,
            f"{marker}{name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        tiles.append(tile)

    if not tiles:
        return None

    rows: list[np.ndarray] = []
    current_row: list[np.ndarray] = []

    for tile in tiles:
        current_row.append(tile)

        if len(current_row) == 2:
            rows.append(_hstack_same_height(current_row))
            current_row = []

    if current_row:
        while len(current_row) < 2:
            current_row.append(np.zeros_like(current_row[0]))

        rows.append(_hstack_same_height(current_row))

    max_width = max(row.shape[1] for row in rows)
    normalized_rows = []

    for row in rows:
        if row.shape[1] < max_width:
            pad = np.zeros(
                (row.shape[0], max_width - row.shape[1], 3),
                dtype=row.dtype,
            )
            row = np.hstack([row, pad])

        normalized_rows.append(row)

    return np.vstack(normalized_rows)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]

    if h <= 0:
        return image

    scale = height / h
    new_width = max(1, int(w * scale))

    return cv2.resize(
        image,
        (new_width, height),
        interpolation=cv2.INTER_AREA,
    )


def _hstack_same_height(images: list[np.ndarray]) -> np.ndarray:
    min_height = min(image.shape[0] for image in images)

    resized = []

    for image in images:
        if image.shape[0] != min_height:
            scale = min_height / image.shape[0]
            new_width = max(1, int(image.shape[1] * scale))
            image = cv2.resize(
                image,
                (new_width, min_height),
                interpolation=cv2.INTER_AREA,
            )

        resized.append(image)

    return np.hstack(resized)


def _bbox_to_tuple(bbox: BoundingBox) -> tuple[int, int, int, int]:
    return bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max


def _int_value(value: Any) -> int:
    return int(round(float(str(value).replace(",", ".").strip())))


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)