from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv_module.detection.candidate_merger import BoundingBox, PriceTagCandidate


@dataclass(frozen=True)
class ModelDetection:
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    metadata: dict[str, Any]


class YOLOPriceTagDetector:
    """
    Детектор ценников на базе обученной YOLO-модели.

    Важно:
    - модель работает на RAW-кадре;
    - bbox возвращается в координатах исходного видео;
    - поворот/undistort/улучшение crop делаются позже, только для OCR.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        device: str | None = None,
        target_class_names: set[str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Не установлен ultralytics. Установи: python -m pip install ultralytics"
            ) from exc

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_class_names = target_class_names

    def detect(self, frame: np.ndarray) -> list[ModelDetection]:
        if frame is None or frame.size == 0:
            return []

        frame_height, frame_width = frame.shape[:2]

        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }

        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)

        detections: list[ModelDetection] = []

        for result in results:
            names = getattr(result, "names", {}) or {}

            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                confidence = float(box.conf[0].detach().cpu().item())
                class_id = int(box.cls[0].detach().cpu().item())

                class_name = self._resolve_class_name(
                    class_id=class_id,
                    model_names=names,
                )

                if self.target_class_names is not None and class_name not in self.target_class_names:
                    continue

                bbox = BoundingBox(
                    x_min=int(round(xyxy[0])),
                    y_min=int(round(xyxy[1])),
                    x_max=int(round(xyxy[2])),
                    y_max=int(round(xyxy[3])),
                ).clamp(frame_width, frame_height)

                if bbox.area <= 0:
                    continue

                detections.append(
                    ModelDetection(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        metadata={
                            "model_path": str(self.model_path),
                            "raw_xyxy": xyxy,
                        },
                    )
                )

        detections.sort(key=lambda item: item.confidence, reverse=True)

        return detections

    def detect_as_candidates(
        self,
        frame: np.ndarray,
        frame_index: int | None = None,
        timestamp_ms: float | None = None,
        max_candidates: int | None = None,
    ) -> list[PriceTagCandidate]:
        detections = self.detect(frame)

        if max_candidates is not None:
            detections = detections[:max_candidates]

        candidates: list[PriceTagCandidate] = []

        for detection in detections:
            candidates.append(
                PriceTagCandidate(
                    bbox=detection.bbox,
                    source="model_yolo",
                    score=detection.confidence,
                    evidence={
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "model_confidence": detection.confidence,
                        **detection.metadata,
                    },
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            )

        return candidates

    def _resolve_class_name(
        self,
        class_id: int,
        model_names: dict[int, str] | dict[str, str],
    ) -> str:
        if class_id in model_names:
            return str(model_names[class_id])

        if str(class_id) in model_names:
            return str(model_names[str(class_id)])

        if class_id == 0:
            return "price_tag"

        return f"class_{class_id}"


def draw_model_candidates_debug(
    frame: np.ndarray,
    candidates: list[PriceTagCandidate],
    output_path: str | Path,
) -> None:
    debug = frame.copy()

    for idx, candidate in enumerate(candidates):
        x_min, y_min, x_max, y_max = candidate.bbox.to_tuple()

        color = (0, 255, 0)

        cv2.rectangle(
            debug,
            (x_min, y_min),
            (x_max, y_max),
            color,
            3,
        )

        class_name = candidate.evidence.get("class_name", "price_tag")

        label = f"{idx}:{class_name}:{candidate.score:.2f}"

        cv2.putText(
            debug,
            label,
            (x_min, max(28, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), debug)