from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile

import cv2
import numpy as np

from cv_module.detection.candidate_merger import BoundingBox
from cv_module.recognition.ocr_engine import OCRResult, TextBlock


class PaddleOCREngine:
    """
    OCR-движок на базе PaddleOCR.

    Задача:
    - принять crop-картинку ценника;
    - запустить PaddleOCR;
    - привести результат к общему формату OCRResult,
      чтобы дальше использовать существующий field_parser.py.

    Поддерживает:
    - новый API PaddleOCR 3.x: ocr.predict(input=...)
    - старый API PaddleOCR 2.x: ocr.ocr(...)
    """

    def __init__(
        self,
        language: str = "ru",
        use_gpu: bool = False,
        device: str | None = None,
        use_angle_cls: bool = False,
        min_confidence: float = 0.15,
    ) -> None:
        self.language = language
        self.use_gpu = use_gpu
        self.device = device
        self.use_angle_cls = use_angle_cls
        self.min_confidence = min_confidence

        self._ocr: Any | None = None
        self._api_kind: str = "unknown"

    def recognize(self, image: np.ndarray) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(
                raw_text="",
                blocks=[],
                confidence=0.0,
                engine="paddleocr",
            )

        try:
            ocr = self._get_ocr()
        except Exception:
            return OCRResult(
                raw_text="",
                blocks=[],
                confidence=0.0,
                engine="paddleocr_unavailable",
            )

        image = _ensure_bgr(image)

        raw_payload = self._run_ocr(ocr, image)
        blocks = _extract_text_blocks(raw_payload)

        blocks = [
            block
            for block in blocks
            if block.text.strip() and block.confidence >= self.min_confidence
        ]

        blocks = _deduplicate_blocks(blocks)

        raw_text = "\n".join(block.text for block in blocks)

        if blocks:
            confidence = float(np.mean([block.confidence for block in blocks]))
        else:
            confidence = 0.0

        return OCRResult(
            raw_text=raw_text,
            blocks=blocks,
            confidence=confidence,
            engine=f"paddleocr_{self._api_kind}",
        )

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with: "
                "python -m pip install paddleocr paddlepaddle"
            ) from exc

        # PaddleOCR 3.x.
        # Новый API использует predict(input=...), а в конструкторе появились
        # параметры use_doc_orientation_classify/use_doc_unwarping/use_textline_orientation.
        v3_kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self.use_angle_cls,
        }

        # В разных версиях lang может поддерживаться или нет.
        if self.language:
            v3_kwargs["lang"] = self.language

        if self.device:
            v3_kwargs["device"] = self.device

        try:
            self._ocr = PaddleOCR(**v3_kwargs)
            self._api_kind = "v3_predict"
            return self._ocr
        except TypeError:
            pass
        except Exception:
            pass

        # PaddleOCR 2.x.
        v2_kwargs: dict[str, Any] = {
            "lang": self.language,
            "use_angle_cls": self.use_angle_cls,
            "use_gpu": self.use_gpu,
            "show_log": False,
        }

        try:
            self._ocr = PaddleOCR(**v2_kwargs)
            self._api_kind = "v2_ocr"
            return self._ocr
        except TypeError:
            pass
        except Exception:
            pass

        # Максимально совместимый fallback.
        try:
            self._ocr = PaddleOCR()
            self._api_kind = "fallback"
            return self._ocr
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize PaddleOCR: {exc}") from exc

    def _run_ocr(self, ocr: Any, image: np.ndarray) -> Any:
        # Новый PaddleOCR 3.x чаще работает через predict(input=path).
        if hasattr(ocr, "predict"):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
                temp_path = Path(file.name)

            try:
                cv2.imwrite(str(temp_path), image)

                try:
                    return ocr.predict(input=str(temp_path))
                except TypeError:
                    return ocr.predict(str(temp_path))
            finally:
                temp_path.unlink(missing_ok=True)

        # Старый PaddleOCR 2.x.
        if hasattr(ocr, "ocr"):
            try:
                return ocr.ocr(image, cls=self.use_angle_cls)
            except TypeError:
                return ocr.ocr(image)

        return None


def is_paddleocr_available() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except Exception:
        return False


def _extract_text_blocks(payload: Any) -> list[TextBlock]:
    blocks: list[TextBlock] = []

    _collect_blocks_recursive(payload, blocks)

    return blocks


def _collect_blocks_recursive(payload: Any, blocks: list[TextBlock]) -> None:
    if payload is None:
        return

    # PaddleOCR 3.x result object может иметь json / to_dict.
    if hasattr(payload, "json"):
        try:
            json_value = payload.json

            if callable(json_value):
                json_value = json_value()

            _collect_blocks_recursive(json_value, blocks)
            return
        except Exception:
            pass

    if hasattr(payload, "to_dict"):
        try:
            _collect_blocks_recursive(payload.to_dict(), blocks)
            return
        except Exception:
            pass

    if isinstance(payload, dict):
        _collect_blocks_from_dict(payload, blocks)

        # На случай вложенной структуры.
        for value in payload.values():
            if isinstance(value, (dict, list, tuple)):
                _collect_blocks_recursive(value, blocks)

        return

    if isinstance(payload, (list, tuple)):
        # Старый стиль PaddleOCR:
        # [
        #   [
        #     [[[x1,y1],...], ("text", score)],
        #     ...
        #   ]
        # ]
        if _looks_like_old_style_line(payload):
            block = _parse_old_style_line(payload)

            if block is not None:
                blocks.append(block)

            return

        for item in payload:
            _collect_blocks_recursive(item, blocks)

        return


def _collect_blocks_from_dict(payload: dict[str, Any], blocks: list[TextBlock]) -> None:
    # Частый новый стиль:
    # {
    #   "rec_texts": [...],
    #   "rec_scores": [...],
    #   "rec_polys": [...]
    # }
    for container_key in ("res", "result", "data"):
        nested = payload.get(container_key)

        if isinstance(nested, dict):
            _collect_blocks_from_dict(nested, blocks)

    text_keys = [
        "rec_texts",
        "texts",
        "text",
    ]

    score_keys = [
        "rec_scores",
        "scores",
        "score",
        "confidence",
    ]

    box_keys = [
        "rec_boxes",
        "rec_polys",
        "dt_polys",
        "dt_boxes",
        "boxes",
        "polys",
    ]

    texts = _first_existing(payload, text_keys)
    scores = _first_existing(payload, score_keys)
    boxes = _first_existing(payload, box_keys)

    if isinstance(texts, str):
        text = texts.strip()

        if text:
            blocks.append(
                TextBlock(
                    text=text,
                    confidence=_safe_float(scores, 1.0),
                    bbox=_bbox_from_any(boxes),
                )
            )

        return

    if isinstance(texts, (list, tuple)):
        for index, text_value in enumerate(texts):
            text = str(text_value).strip()

            if not text:
                continue

            score = _item_at(scores, index, default=1.0)
            box = _item_at(boxes, index, default=None)

            blocks.append(
                TextBlock(
                    text=text,
                    confidence=_safe_float(score, 1.0),
                    bbox=_bbox_from_any(box),
                )
            )

        return

    # Иногда текст лежит в отдельных ключах.
    single_text = payload.get("transcription") or payload.get("label")

    if isinstance(single_text, str) and single_text.strip():
        blocks.append(
            TextBlock(
                text=single_text.strip(),
                confidence=_safe_float(scores, 1.0),
                bbox=_bbox_from_any(boxes),
            )
        )


def _looks_like_old_style_line(payload: Any) -> bool:
    if not isinstance(payload, (list, tuple)):
        return False

    if len(payload) != 2:
        return False

    second = payload[1]

    if not isinstance(second, (list, tuple)):
        return False

    if len(second) < 2:
        return False

    return isinstance(second[0], str)


def _parse_old_style_line(payload: Any) -> TextBlock | None:
    try:
        box_raw = payload[0]
        text_raw = payload[1][0]
        score_raw = payload[1][1]

        text = str(text_raw).strip()

        if not text:
            return None

        return TextBlock(
            text=text,
            confidence=_safe_float(score_raw, 1.0),
            bbox=_bbox_from_any(box_raw),
        )
    except Exception:
        return None


def _bbox_from_any(value: Any) -> BoundingBox | None:
    if value is None:
        return None

    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        return None

    # Формат [x_min, y_min, x_max, y_max].
    if arr.ndim == 1 and arr.shape[0] == 4:
        x_min, y_min, x_max, y_max = arr.tolist()

        return _make_bbox(x_min, y_min, x_max, y_max)

    # Формат [[x1,y1], [x2,y2], ...].
    try:
        points = arr.reshape(-1, 2)

        xs = points[:, 0]
        ys = points[:, 1]

        return _make_bbox(
            float(np.min(xs)),
            float(np.min(ys)),
            float(np.max(xs)),
            float(np.max(ys)),
        )
    except Exception:
        return None


def _make_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> BoundingBox | None:
    x1 = int(round(min(x_min, x_max)))
    y1 = int(round(min(y_min, y_max)))
    x2 = int(round(max(x_min, x_max)))
    y2 = int(round(max(y_min, y_max)))

    if x2 <= x1 or y2 <= y1:
        return None

    return BoundingBox(
        x_min=x1,
        y_min=y1,
        x_max=x2,
        y_max=y2,
    )


def _first_existing(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]

    return None


def _item_at(value: Any, index: int, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        if 0 <= index < len(value):
            return value[index]

    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _deduplicate_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    result: list[TextBlock] = []
    seen: set[tuple[str, int, int, int, int]] = set()

    for block in blocks:
        text = " ".join(block.text.split())

        if not text:
            continue

        if block.bbox is None:
            key = (text, -1, -1, -1, -1)
        else:
            key = (
                text,
                block.bbox.x_min,
                block.bbox.y_min,
                block.bbox.x_max,
                block.bbox.y_max,
            )

        if key in seen:
            continue

        seen.add(key)

        result.append(
            TextBlock(
                text=text,
                confidence=block.confidence,
                bbox=block.bbox,
            )
        )

    return result


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image