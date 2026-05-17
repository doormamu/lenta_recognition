from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv_module.detection.candidate_merger import BoundingBox


@dataclass(frozen=True)
class TextBlock:
    text: str
    confidence: float
    bbox: BoundingBox | None = None


@dataclass(frozen=True)
class OCRResult:
    raw_text: str
    blocks: list[TextBlock]
    confidence: float
    engine: str


class OCREngine:
    """
    Минимальный интерфейс OCR.

    Сейчас внешнего OCR-движка в requirements нет, поэтому класс честно
    возвращает пустой результат. Поля уже можно заполнять через QR/CSV oracle,
    а позже сюда можно подключить PaddleOCR/EasyOCR/Tesseract без изменений
    field_parser и barcode_reader.
    """

    def recognize(self, image: np.ndarray) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(raw_text="", blocks=[], confidence=0.0, engine="none")

        return OCRResult(raw_text="", blocks=[], confidence=0.0, engine="none")
