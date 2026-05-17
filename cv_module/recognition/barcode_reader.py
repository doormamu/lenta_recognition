from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv_module.detection.candidate_merger import BoundingBox
from cv_module.detection.qr_detector import detect_codes


@dataclass(frozen=True)
class BarcodeRead:
    value: str
    code_type: str
    confidence: float
    bbox: BoundingBox | None = None
    source: str = "unknown"


class BarcodeReader:
    def __init__(self, try_harder: bool = True) -> None:
        self.try_harder = try_harder

    def read(self, image: np.ndarray) -> list[BarcodeRead]:
        if image is None or image.size == 0:
            return []

        detections = detect_codes(image, try_harder=self.try_harder)

        return [
            BarcodeRead(
                value=detection.value,
                code_type=detection.code_type,
                confidence=detection.confidence,
                bbox=detection.bbox,
                source=detection.source,
            )
            for detection in detections
        ]
