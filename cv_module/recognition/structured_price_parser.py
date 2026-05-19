from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import io
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from cv_module.detection.candidate_merger import BoundingBox


UNKNOWN_VALUE = "-"


@dataclass(frozen=True)
class OCRWord:
    text: str
    confidence: float
    bbox: BoundingBox

    @property
    def x_center(self) -> float:
        return (self.bbox.x_min + self.bbox.x_max) / 2.0

    @property
    def y_center(self) -> float:
        return (self.bbox.y_min + self.bbox.y_max) / 2.0

    @property
    def height(self) -> int:
        return self.bbox.height

    @property
    def width(self) -> int:
        return self.bbox.width


@dataclass(frozen=True)
class OCRLine:
    words: list[OCRWord]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(
            x_min=min(word.bbox.x_min for word in self.words),
            y_min=min(word.bbox.y_min for word in self.words),
            x_max=max(word.bbox.x_max for word in self.words),
            y_max=max(word.bbox.y_max for word in self.words),
        )


@dataclass(frozen=True)
class PriceCandidate:
    value: float
    text: str
    bbox: BoundingBox | None
    score: float
    source: str


@dataclass(frozen=True)
class StructuredPriceParseResult:
    values: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_field_values(self) -> dict[str, str]:
        return {
            key: value
            for key, value in self.values.items()
            if value not in {"", UNKNOWN_VALUE, None}
        }

    def has_values(self) -> bool:
        return bool(self.to_field_values())


class StructuredPriceParser:
    """
    Структурный парсер ценника.

    Идея:
    1. Получаем OCR-слова с координатами через tesseract TSV.
    2. Парсим поля не из всего текста подряд, а с учетом структуры ценника:
       - скидка: -X%;
       - цена: крупные рубли + маленькие копейки справа;
       - цена по карте <= цена без карты;
       - скидка должна примерно согласовываться с двумя ценами;
       - штрихкод проверяется по длине и checksum;
       - id_sku не должен совпадать со штрихкодом, датой или ценой.
    """

    def __init__(
        self,
        tesseract_path: str | None = None,
        language: str = "rus+eng+snum",
        timeout_sec: float = 3.0,
        min_word_confidence: float = 10.0,
        max_tsv_variants: int = 2,
    ) -> None:
        self.tesseract_path = tesseract_path or shutil.which("tesseract")
        self.language = language
        self.timeout_sec = timeout_sec
        self.min_word_confidence = min_word_confidence
        self.max_tsv_variants = max_tsv_variants

    def parse(
        self,
        image: np.ndarray,
        fallback_text: str = "",
    ) -> StructuredPriceParseResult:
        if image is None or image.size == 0:
            return StructuredPriceParseResult()

        if not self.tesseract_path:
            return StructuredPriceParseResult(
                diagnostics={"error": "tesseract_not_found"},
            )

        image = _ensure_bgr(image)
        height, width = image.shape[:2]

        words = self._recognize_words(image)
        words = _deduplicate_words(words)

        full_text = _build_full_text(words, fallback_text)
        lines = _group_words_into_lines(words)

        values: dict[str, str] = {}
        sources: dict[str, str] = {}
        diagnostics: dict[str, Any] = {
            "words_count": len(words),
            "lines_count": len(lines),
        }

        discount = _extract_discount(full_text)
        if discount:
            values["discount_amount"] = discount
            sources["discount_amount"] = "structured_discount"

        print_datetime = _extract_print_datetime(full_text)
        if print_datetime:
            values["print_datetime"] = print_datetime
            sources["print_datetime"] = "structured_datetime"

        barcode = _extract_barcode(full_text)
        if barcode:
            values["barcode"] = barcode
            sources["barcode"] = "structured_barcode"

        price_candidates = _extract_price_candidates(
            words=words,
            full_text=full_text,
            image_width=width,
            image_height=height,
        )

        diagnostics["price_candidates"] = [
            {
                "value": candidate.value,
                "text": candidate.text,
                "score": round(candidate.score, 4),
                "source": candidate.source,
                "bbox": candidate.bbox.to_tuple() if candidate.bbox else None,
            }
            for candidate in price_candidates
        ]

        price_default, price_card, price_diag = _choose_prices(
            candidates=price_candidates,
            discount_text=discount,
        )

        diagnostics["price_choice"] = price_diag

        if price_default:
            values["price_default"] = price_default
            sources["price_default"] = "structured_price"

        if price_card:
            values["price_card"] = price_card
            sources["price_card"] = "structured_price"

        id_sku = _extract_id_sku(
            full_text=full_text,
            barcode=barcode,
            known_prices=[price_default, price_card],
        )

        if id_sku:
            values["id_sku"] = id_sku
            sources["id_sku"] = "structured_id_sku"

        code = _extract_zone_or_tag_code(full_text)

        if code:
            values["code"] = code
            sources["code"] = "structured_code"

        product_name = _extract_product_name(
            lines=lines,
            image_width=width,
            image_height=height,
        )

        if product_name:
            values["product_name"] = product_name
            sources["product_name"] = "structured_product_name"

        confidence = _estimate_confidence(values, diagnostics)

        return StructuredPriceParseResult(
            values=values,
            sources=sources,
            confidence=confidence,
            diagnostics=diagnostics,
        )

    def _recognize_words(self, image: np.ndarray) -> list[OCRWord]:
        variants = _make_tsv_variants(image)[: self.max_tsv_variants]
        all_words: list[OCRWord] = []

        for variant in variants:
            for psm in (6, 11):
                words = self._run_tesseract_tsv(variant, psm=psm)
                all_words.extend(words)

        return all_words

    def _run_tesseract_tsv(
        self,
        image: np.ndarray,
        psm: int,
    ) -> list[OCRWord]:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
            temp_path = Path(file.name)

        try:
            cv2.imwrite(str(temp_path), image)

            command = [
                str(self.tesseract_path),
                str(temp_path),
                "stdout",
                "-l",
                self.language,
                "--oem",
                "3",
                "--psm",
                str(psm),
                "tsv",
            ]

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )

            if completed.returncode != 0:
                return []

            return _parse_tesseract_tsv(
                completed.stdout,
                min_confidence=self.min_word_confidence,
            )
        except Exception:
            return []
        finally:
            temp_path.unlink(missing_ok=True)


def _parse_tesseract_tsv(
    payload: str,
    min_confidence: float,
) -> list[OCRWord]:
    words: list[OCRWord] = []

    if not payload.strip():
        return words

    reader = csv.DictReader(io.StringIO(payload), delimiter="\t")

    for row in reader:
        text = str(row.get("text", "") or "").strip()

        if not text:
            continue

        try:
            confidence = float(row.get("conf", "-1"))
        except ValueError:
            confidence = -1.0

        if confidence < min_confidence:
            continue

        try:
            x = int(float(row.get("left", "0")))
            y = int(float(row.get("top", "0")))
            w = int(float(row.get("width", "0")))
            h = int(float(row.get("height", "0")))
        except ValueError:
            continue

        if w <= 0 or h <= 0:
            continue

        words.append(
            OCRWord(
                text=text,
                confidence=confidence / 100.0,
                bbox=BoundingBox(
                    x_min=x,
                    y_min=y,
                    x_max=x + w,
                    y_max=y + h,
                ),
            )
        )

    return words


def _make_tsv_variants(image: np.ndarray) -> list[np.ndarray]:
    image = _ensure_bgr(image)

    result: list[np.ndarray] = []

    result.append(image)

    scaled = cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    result.append(scaled)

    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
    sharp = cv2.addWeighted(contrast, 1.8, blurred, -0.8, 0)

    result.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

    return _deduplicate_images(result)


def _group_words_into_lines(words: list[OCRWord]) -> list[OCRLine]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: (word.y_center, word.bbox.x_min))

    lines: list[list[OCRWord]] = []

    for word in sorted_words:
        placed = False

        for line in lines:
            line_center = np.mean([item.y_center for item in line])
            line_height = np.mean([item.height for item in line])
            tolerance = max(8.0, line_height * 0.55)

            if abs(word.y_center - line_center) <= tolerance:
                line.append(word)
                placed = True
                break

        if not placed:
            lines.append([word])

    result = []

    for line_words in lines:
        line_words = sorted(line_words, key=lambda word: word.bbox.x_min)
        result.append(OCRLine(words=line_words))

    return result


def _extract_discount(text: str) -> str:
    normalized = _normalize_ocr_digits(text)

    matches = re.findall(r"[-−—]?\s*(\d{1,2})\s*[%оo]", normalized)

    values: list[int] = []

    for item in matches:
        try:
            value = int(item)
        except ValueError:
            continue

        if 1 <= value <= 90:
            values.append(value)

    if not values:
        return ""

    return f"-{max(values)}%"


def _extract_print_datetime(text: str) -> str:
    normalized = _normalize_ocr_digits(text)

    patterns = [
        r"\b(\d{2})[.](\d{2})[.](\d{4})\s+(\d{1,2})[:.](\d{2})\b",
        r"\b(\d{2})[.](\d{2})[.](\d{2})\s+(\d{1,2})[:.](\d{2})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)

        if not match:
            continue

        day, month, year, hour, minute = match.groups()

        if len(year) == 2:
            year = f"20{year}"

        try:
            day_i = int(day)
            month_i = int(month)
            hour_i = int(hour)
            minute_i = int(minute)
        except ValueError:
            continue

        if not (1 <= day_i <= 31 and 1 <= month_i <= 12):
            continue

        if not (0 <= hour_i <= 23 and 0 <= minute_i <= 59):
            continue

        return f"{day}.{month}.{year} {hour_i:02d}:{minute_i:02d}"

    return ""


def _extract_barcode(text: str) -> str:
    normalized = _normalize_ocr_digits(text)
    compact_candidates = re.findall(r"\b\d[\d\s]{6,22}\d\b", normalized)

    candidates: list[str] = []

    for item in compact_candidates:
        digits = re.sub(r"\D", "", item)

        if len(digits) in {8, 12, 13, 14}:
            candidates.append(digits)

    candidates = sorted(set(candidates), key=len, reverse=True)

    for digits in candidates:
        if _is_repeated_digits(digits):
            continue

        if len(digits) in {8, 12, 13, 14} and _gtin_checksum_ok(digits):
            return digits

    # Если checksum не сошелся, для 14 цифр иногда оставляем fallback,
    # но только если кандидат выглядит реалистично.
    for digits in candidates:
        if len(digits) == 14 and not _is_repeated_digits(digits):
            return digits

    return ""


def _extract_id_sku(
    full_text: str,
    barcode: str,
    known_prices: list[str],
) -> str:
    normalized = _normalize_ocr_digits(full_text)

    price_digits = {
        re.sub(r"\D", "", price or "")
        for price in known_prices
        if price
    }

    candidates: list[str] = []

    for match in re.finditer(r"\b\d{6}\s+\d{6}\b", normalized):
        digits = re.sub(r"\D", "", match.group(0))
        candidates.append(digits)

    for match in re.finditer(r"\b\d{6,12}\b", normalized):
        digits = match.group(0)
        candidates.append(digits)

    for digits in candidates:
        if not digits:
            continue

        if digits == barcode:
            continue

        if digits in price_digits:
            continue

        if len(digits) in {8, 12, 13, 14} and _gtin_checksum_ok(digits):
            continue

        if _looks_like_date_fragment(digits):
            continue

        if 6 <= len(digits) <= 12:
            return digits

    return ""


def _extract_zone_or_tag_code(text: str) -> str:
    normalized = _normalize_ocr_digits(text)

    match = re.search(
        r"\b\d{2,3}[_\s-]\d{3,6}(?:\s*[-–]\s*\d{2,3}[_\s-]?\d{3,6})?\b",
        normalized,
    )

    if not match:
        return ""

    return " ".join(match.group(0).replace("_", " ").split())


def _extract_price_candidates(
    words: list[OCRWord],
    full_text: str,
    image_width: int,
    image_height: int,
) -> list[PriceCandidate]:
    candidates: list[PriceCandidate] = []

    word_candidates = _extract_price_candidates_from_words(words)
    candidates.extend(word_candidates)

    text_candidates = _extract_price_candidates_from_text(full_text)
    candidates.extend(text_candidates)

    candidates = [
        candidate
        for candidate in candidates
        if 10.0 <= candidate.value <= 99999.99
    ]

    candidates = _deduplicate_price_candidates(candidates)

    candidates.sort(key=lambda item: item.score, reverse=True)

    return candidates[:8]


def _extract_price_candidates_from_words(words: list[OCRWord]) -> list[PriceCandidate]:
    candidates: list[PriceCandidate] = []

    if not words:
        return candidates

    digit_words: list[tuple[OCRWord, str]] = []

    for word in words:
        digits = _only_digits(word.text)

        if not digits:
            continue

        digit_words.append((word, digits))

    heights = [word.height for word, digits in digit_words if len(digits) >= 2]
    median_height = float(np.median(heights)) if heights else 1.0

    rub_words = [
        (word, digits)
        for word, digits in digit_words
        if 2 <= len(digits) <= 5
    ]

    kop_words = [
        (word, digits)
        for word, digits in digit_words
        if len(digits) == 2 and 0 <= int(digits) <= 99
    ]

    for rub_word, rub_digits in rub_words:
        rub = int(rub_digits)

        if rub < 10:
            continue

        # Компактная цена вида 110499 -> 1104.99.
        if len(rub_digits) >= 4:
            compact_price = _compact_digits_to_price(rub_digits)

            if compact_price is not None:
                candidates.append(
                    PriceCandidate(
                        value=compact_price,
                        text=f"{compact_price:.2f}",
                        bbox=rub_word.bbox,
                        score=0.45 + min(rub_word.height / max(median_height, 1.0), 2.0) * 0.12,
                        source="word_compact",
                    )
                )

        for kop_word, kop_digits in kop_words:
            if kop_word is rub_word:
                continue

            if not _looks_like_cents_pair(rub_word, kop_word):
                continue

            price = float(f"{rub}.{int(kop_digits):02d}")

            score = 0.65

            if rub_word.height >= median_height * 1.4:
                score += 0.20

            if kop_word.bbox.x_min >= rub_word.bbox.x_max - int(0.10 * rub_word.width):
                score += 0.10

            vertical_delta = abs(kop_word.y_center - rub_word.y_center)
            score += max(0.0, 0.10 - vertical_delta / max(rub_word.height * 6.0, 1.0))

            bbox = BoundingBox(
                x_min=min(rub_word.bbox.x_min, kop_word.bbox.x_min),
                y_min=min(rub_word.bbox.y_min, kop_word.bbox.y_min),
                x_max=max(rub_word.bbox.x_max, kop_word.bbox.x_max),
                y_max=max(rub_word.bbox.y_max, kop_word.bbox.y_max),
            )

            candidates.append(
                PriceCandidate(
                    value=price,
                    text=f"{rub}.{int(kop_digits):02d}",
                    bbox=bbox,
                    score=score,
                    source="rub_kop_pair",
                )
            )

    return candidates


def _extract_price_candidates_from_text(text: str) -> list[PriceCandidate]:
    normalized = _normalize_ocr_digits(text)
    candidates: list[PriceCandidate] = []

    for match in re.finditer(r"\b(\d{1,5})[,.](\d{2})\b", normalized):
        rub = int(match.group(1))
        kop = int(match.group(2))

        if 10 <= rub <= 99999 and 0 <= kop <= 99:
            value = float(f"{rub}.{kop:02d}")

            candidates.append(
                PriceCandidate(
                    value=value,
                    text=f"{rub}.{kop:02d}",
                    bbox=None,
                    score=0.52,
                    source="text_decimal",
                )
            )

    for match in re.finditer(r"\b\d{4,7}\b", normalized):
        digits = match.group(0)

        if len(digits) in {8, 12, 13, 14}:
            continue

        value = _compact_digits_to_price(digits)

        if value is not None:
            candidates.append(
                PriceCandidate(
                    value=value,
                    text=f"{value:.2f}",
                    bbox=None,
                    score=0.32,
                    source="text_compact",
                )
            )

    return candidates


def _choose_prices(
    candidates: list[PriceCandidate],
    discount_text: str,
) -> tuple[str, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}

    if not candidates:
        return "", "", diagnostics

    discount_value = _discount_to_number(discount_text)

    unique = _deduplicate_price_candidates(candidates)

    if len(unique) == 1:
        return "", _format_price(unique[0].value), {
            "reason": "single_price",
            "selected_card": unique[0].value,
        }

    pair_candidates: list[tuple[float, PriceCandidate, PriceCandidate, dict[str, Any]]] = []

    for first in unique:
        for second in unique:
            if first is second:
                continue

            high = max(first.value, second.value)
            low = min(first.value, second.value)

            if high <= low:
                continue

            if low < 10 or high < 10:
                continue

            ratio = high / max(low, 1.0)

            if ratio > 4.0:
                continue

            calc_discount = round((1.0 - low / high) * 100.0)

            score = first.score + second.score

            if discount_value is not None:
                diff = abs(calc_discount - discount_value)
                score += max(0.0, 1.0 - diff / 15.0)

                if diff > 18:
                    score -= 0.75

            if high > low:
                score += 0.25

            pair_candidates.append(
                (
                    score,
                    first if first.value == high else second,
                    first if first.value == low else second,
                    {
                        "high": high,
                        "low": low,
                        "ratio": ratio,
                        "calc_discount": calc_discount,
                        "discount_ocr": discount_value,
                    },
                )
            )

    if pair_candidates:
        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        score, default_candidate, card_candidate, diag = pair_candidates[0]

        diagnostics.update(diag)
        diagnostics["reason"] = "pair_selected"
        diagnostics["pair_score"] = round(score, 4)

        return (
            _format_price(default_candidate.value),
            _format_price(card_candidate.value),
            diagnostics,
        )

    best = max(unique, key=lambda item: item.score)

    return "", _format_price(best.value), {
        "reason": "fallback_best_single",
        "selected_card": best.value,
    }


def _extract_product_name(
    lines: list[OCRLine],
    image_width: int,
    image_height: int,
) -> str:
    if not lines:
        return ""

    candidate_lines: list[tuple[float, str]] = []

    for line in lines:
        text = " ".join(line.text.split()).strip()

        if len(text) < 3:
            continue

        bbox = line.bbox

        # Название обычно сверху и левее центра.
        if bbox.y_min > image_height * 0.60:
            continue

        if bbox.x_min > image_width * 0.72:
            continue

        if not re.search(r"[A-Za-zА-Яа-яЁё]", text):
            continue

        if _line_looks_service_or_numeric(text):
            continue

        cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё .,%/()-]+", " ", text)
        cleaned = " ".join(cleaned.split())

        if len(cleaned) < 3:
            continue

        letters_count = len(re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned))

        if letters_count < 3:
            continue

        score = letters_count

        if bbox.y_min < image_height * 0.35:
            score += 15

        if bbox.x_min < image_width * 0.55:
            score += 10

        candidate_lines.append((score, cleaned))

    if not candidate_lines:
        return ""

    candidate_lines.sort(key=lambda item: item[0], reverse=True)

    result_lines = [text for _, text in candidate_lines[:4]]

    return " ".join(result_lines)[:240]


def _line_looks_service_or_numeric(text: str) -> bool:
    lowered = text.lower()
    normalized = _normalize_ocr_digits(lowered)

    service_tokens = [
        "без карты",
        "по карте",
        "цена",
        "руб",
        "коп",
        "qr",
        "код",
        "штрих",
        "скид",
        "id",
        "sku",
    ]

    if any(token in lowered for token in service_tokens):
        return True

    if re.search(r"\d{2}[.]\d{2}[.]\d{2,4}", normalized):
        return True

    if re.search(r"[-−—]?\s*\d{1,2}\s*%", normalized):
        return True

    digits = re.sub(r"\D", "", normalized)

    if len(digits) >= 6:
        return True

    return False


def _estimate_confidence(
    values: dict[str, str],
    diagnostics: dict[str, Any],
) -> float:
    score = 0.0

    if values.get("product_name"):
        score += 0.16

    if values.get("price_card"):
        score += 0.26

    if values.get("price_default"):
        score += 0.22

    if values.get("discount_amount"):
        score += 0.12

    if values.get("barcode"):
        score += 0.12

    if values.get("id_sku"):
        score += 0.06

    if values.get("print_datetime"):
        score += 0.06

    price_choice = diagnostics.get("price_choice", {})

    if price_choice.get("reason") == "pair_selected":
        score += 0.08

    calc_discount = price_choice.get("calc_discount")
    discount_ocr = price_choice.get("discount_ocr")

    if calc_discount is not None and discount_ocr is not None:
        if abs(float(calc_discount) - float(discount_ocr)) <= 10:
            score += 0.08

    return float(np.clip(score, 0.0, 1.0))


def _looks_like_cents_pair(rub_word: OCRWord, kop_word: OCRWord) -> bool:
    # Копейки обычно справа от рублей и чуть выше/рядом.
    if kop_word.bbox.x_min < rub_word.x_center:
        return False

    max_horizontal_gap = max(12, int(rub_word.height * 2.8))
    horizontal_gap = kop_word.bbox.x_min - rub_word.bbox.x_max

    if horizontal_gap > max_horizontal_gap:
        return False

    if kop_word.y_center > rub_word.bbox.y_max + rub_word.height * 0.45:
        return False

    if kop_word.y_center < rub_word.bbox.y_min - rub_word.height * 0.80:
        return False

    if kop_word.height > rub_word.height * 1.25:
        return False

    return True


def _compact_digits_to_price(digits: str) -> float | None:
    if not digits.isdigit():
        return None

    if len(digits) < 4 or len(digits) > 7:
        return None

    rub = int(digits[:-2])
    kop = int(digits[-2:])

    if rub < 10 or rub > 99999:
        return None

    if kop < 0 or kop > 99:
        return None

    return float(f"{rub}.{kop:02d}")


def _deduplicate_price_candidates(
    candidates: list[PriceCandidate],
) -> list[PriceCandidate]:
    result: list[PriceCandidate] = []

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        already_exists = False

        for existing in result:
            if abs(existing.value - candidate.value) <= 0.01:
                already_exists = True
                break

        if not already_exists:
            result.append(candidate)

    return result


def _deduplicate_words(words: list[OCRWord]) -> list[OCRWord]:
    result: list[OCRWord] = []
    seen: set[tuple[str, int, int, int, int]] = set()

    for word in words:
        text = " ".join(word.text.split())

        key = (
            text,
            word.bbox.x_min,
            word.bbox.y_min,
            word.bbox.x_max,
            word.bbox.y_max,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(
            OCRWord(
                text=text,
                confidence=word.confidence,
                bbox=word.bbox,
            )
        )

    return result


def _build_full_text(words: list[OCRWord], fallback_text: str) -> str:
    word_text = " ".join(word.text for word in words)

    if fallback_text:
        return f"{word_text}\n{fallback_text}"

    return word_text


def _normalize_ocr_digits(text: str) -> str:
    return (
        str(text)
        .replace("O", "0")
        .replace("o", "0")
        .replace("О", "0")
        .replace("о", "0")
        .replace("З", "3")
        .replace("з", "3")
        .replace("Б", "6")
        .replace("б", "6")
        .replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
    )


def _only_digits(text: str) -> str:
    return re.sub(r"\D", "", _normalize_ocr_digits(text))


def _discount_to_number(value: str) -> int | None:
    if not value:
        return None

    match = re.search(r"\d{1,2}", value)

    if not match:
        return None

    try:
        result = int(match.group(0))
    except ValueError:
        return None

    if 1 <= result <= 90:
        return result

    return None


def _format_price(value: float) -> str:
    return f"{value:.2f}"


def _gtin_checksum_ok(digits: str) -> bool:
    if not digits.isdigit():
        return False

    if len(digits) not in {8, 12, 13, 14}:
        return False

    body = digits[:-1]
    expected_check = int(digits[-1])

    total = 0
    reversed_body = list(map(int, body[::-1]))

    for index, digit in enumerate(reversed_body):
        weight = 3 if index % 2 == 0 else 1
        total += digit * weight

    actual_check = (10 - (total % 10)) % 10

    return actual_check == expected_check


def _is_repeated_digits(digits: str) -> bool:
    return len(set(digits)) <= 2


def _looks_like_date_fragment(digits: str) -> bool:
    if len(digits) not in {6, 8, 12}:
        return False

    if digits.startswith("20") or digits.endswith("2025") or digits.endswith("2026"):
        return True

    return False


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


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image