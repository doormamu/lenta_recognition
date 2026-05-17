from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
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
    def __init__(
        self,
        tesseract_path: str | None = None,
        language: str = "rus+eng+snum",
        tessdata_dir: str | Path | None = None,
        psm: int = 6,
    ) -> None:
        self.tesseract_path = tesseract_path or shutil.which("tesseract")
        self.language = language
        self.tessdata_dir = _resolve_tessdata_dir(tessdata_dir)
        self.psm = psm

    def recognize(self, image: np.ndarray) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(raw_text="", blocks=[], confidence=0.0, engine="none")

        if not self.tesseract_path:
            return OCRResult(raw_text="", blocks=[], confidence=0.0, engine="none")

        texts: list[str] = []

        for variant in _make_ocr_variants(image):
            for psm in (6, 11):
                text = self._recognize_variant(variant, psm=psm)

                if text:
                    texts.append(text)

        raw_text = "\n".join(dict.fromkeys(texts))
        confidence = 0.45 if raw_text else 0.0

        return OCRResult(
            raw_text=raw_text,
            blocks=[
                TextBlock(
                    text=raw_text,
                    confidence=confidence,
                    bbox=None,
                )
            ] if raw_text else [],
            confidence=confidence,
            engine="tesseract",
        )

    def _recognize_variant(self, image: np.ndarray, psm: int) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            dir="/private/tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)

        try:
            cv2.imwrite(str(temp_path), image)

            command = [
                str(self.tesseract_path),
                str(temp_path),
                "stdout",
                "-l",
                self.language,
                "--psm",
                str(psm),
            ]

            if self.tessdata_dir is not None:
                command.extend(["--tessdata-dir", str(self.tessdata_dir)])

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )

            if completed.returncode != 0:
                return ""

            return completed.stdout.strip()
        except Exception:
            return ""
        finally:
            temp_path.unlink(missing_ok=True)


def _make_ocr_variants(image: np.ndarray) -> list[np.ndarray]:
    rotations = [
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        image,
    ]

    variants: list[np.ndarray] = []

    for rotated in rotations:
        prepared = _prepare_single_orientation(rotated)
        variants.extend(prepared)

    return _deduplicate_images(variants)


def _prepare_single_orientation(image: np.ndarray) -> list[np.ndarray]:
    scaled = cv2.resize(
        image,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=9,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
    sharp = cv2.addWeighted(contrast, 1.8, blurred, -0.8, 0)

    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        9,
    )

    otsu_blur = cv2.GaussianBlur(sharp, (3, 3), 0)
    _, otsu = cv2.threshold(
        otsu_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive_clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, morph_kernel)
    otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, morph_kernel)

    return [
        sharp,
        adaptive_clean,
        otsu_clean,
    ]


def _deduplicate_images(images: list[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()

    for image in images:
        marker = (
            image.shape[0],
            image.shape[1],
            int(np.mean(image)),
        )

        if marker in seen:
            continue

        seen.add(marker)
        result.append(image)

    return result


def _resolve_tessdata_dir(tessdata_dir: str | Path | None) -> Path | None:
    if tessdata_dir is not None:
        path = Path(tessdata_dir)
        return path if path.exists() else None

    project_tessdata = Path(__file__).resolve().parents[2] / "models" / "tessdata"

    if (project_tessdata / "rus.traineddata").exists():
        return project_tessdata

    return None
