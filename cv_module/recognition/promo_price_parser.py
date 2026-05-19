from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from cv_module.detection.candidate_merger import BoundingBox


@dataclass(frozen=True)
class PromoPriceParseResult:
    price_default: str | None = None
    price_card: str | None = None
    discount_amount: str | None = None
    confidence: float = 0.0
    orientation: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    def has_values(self) -> bool:
        return bool(
            self.price_default
            or self.price_card
            or self.discount_amount
        )

    def to_field_values(self) -> dict[str, str]:
        result: dict[str, str] = {}

        if self.price_default:
            result["price_default"] = self.price_default

        if self.price_card:
            result["price_card"] = self.price_card

        if self.discount_amount:
            result["discount_amount"] = self.discount_amount

        return result


class PromoPriceParser:
    """
    Специализированный парсер акционных ценников.

    Идея:
    - ищем красную/желтую акционную зону;
    - делим ценник на верхнюю и цветную часть;
    - в цветной зоне ищем скидку и цену по карте;
    - около границы зон ищем обычную цену.

    Это не заменяет общий OCR, а дополняет его для акционных ценников.
    """

    def __init__(
        self,
        tesseract_path: str | None = None,
        language: str = "eng",
        timeout_sec: float = 1.2,
    ) -> None:
        self.tesseract_path = tesseract_path or shutil.which("tesseract")
        self.language = language
        self.timeout_sec = timeout_sec

    def parse(
        self,
        image: np.ndarray,
        with_debug: bool = False,
    ) -> PromoPriceParseResult:
        if image is None or image.size == 0:
            return PromoPriceParseResult()

        if not self.tesseract_path:
            return PromoPriceParseResult(
                metadata={"error": "tesseract_not_found"},
            )

        orientation_variants = [
            ("original", image),
            ("rot90_ccw", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ("rot90_cw", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ]

        parsed_results = []

        for name, variant in orientation_variants:
            result = self._parse_orientation(name, variant, with_debug=with_debug)
            parsed_results.append(result)

            if result.confidence >= 0.82 and result.price_card:
                break

        parsed_results.sort(
            key=lambda item: (
                item.confidence,
                bool(item.price_card),
                bool(item.price_default),
                bool(item.discount_amount),
            ),
            reverse=True,
        )

        return parsed_results[0] if parsed_results else PromoPriceParseResult()

    def _parse_orientation(
        self,
        orientation_name: str,
        image: np.ndarray,
        with_debug: bool,
    ) -> PromoPriceParseResult:
        debug_images: dict[str, np.ndarray] = {}

        tag, tag_bbox, tag_debug = self._find_promo_tag(image)

        if with_debug:
            debug_images["tag_debug"] = tag_debug

        if tag is None or tag.size == 0:
            return PromoPriceParseResult(
                orientation=orientation_name,
                metadata={"error": "promo_tag_not_found"},
                debug_images=debug_images,
            )

        split = self._split_colored_zone(tag)

        if split is None:
            return PromoPriceParseResult(
                orientation=orientation_name,
                metadata={"error": "colored_zone_not_found"},
                debug_images=debug_images,
            )

        upper_zone, colored_zone, boundary, split_debug = split

        if with_debug:
            debug_images["split_debug"] = split_debug

        discount_amount, circle, discount_debug, discount_ocr_debug = (
            self._detect_discount(colored_zone)
        )

        price_card, card_ocr_debug = self._detect_card_price(
            colored_zone=colored_zone,
            circle=circle,
        )

        price_default, default_ocr_debug = self._detect_default_price(
            tag=tag,
            boundary=boundary,
        )

        if with_debug:
            debug_images["discount_debug"] = discount_debug
            debug_images["discount_ocr"] = discount_ocr_debug
            debug_images["card_ocr"] = card_ocr_debug
            debug_images["default_ocr"] = default_ocr_debug

        confidence = self._estimate_confidence(
            price_default=price_default,
            price_card=price_card,
            discount_amount=discount_amount,
            tag_bbox=tag_bbox,
            tag_shape=tag.shape,
        )

        return PromoPriceParseResult(
            price_default=price_default,
            price_card=price_card,
            discount_amount=discount_amount,
            confidence=confidence,
            orientation=orientation_name,
            metadata={
                "tag_bbox": tag_bbox.to_tuple() if tag_bbox else None,
                "boundary": boundary,
                "has_circle": circle is not None,
            },
            debug_images=debug_images,
        )

    def _find_promo_tag(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray | None, BoundingBox | None, np.ndarray]:
        red_mask = _get_red_mask(image)
        yellow_mask = _get_yellow_mask(image)
        promo_mask = cv2.bitwise_or(red_mask, yellow_mask)

        contours, _ = cv2.findContours(
            promo_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        height, width = image.shape[:2]
        frame_area = width * height

        debug = image.copy()

        best_box: BoundingBox | None = None
        best_score = 0.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            box = BoundingBox(x, y, x + w, y + h).clamp(width, height)

            if box.width < max(35, int(width * 0.08)):
                continue

            if box.height < max(18, int(height * 0.04)):
                continue

            area_ratio = box.area / max(1, frame_area)

            if area_ratio < 0.002:
                continue

            if area_ratio > 0.65:
                continue

            rectangularity = _contour_rectangularity(contour)

            if rectangularity < 0.18:
                continue

            aspect = box.aspect_ratio

            if aspect < 0.35 or aspect > 8.5:
                continue

            score = box.area * (0.5 + rectangularity)

            if score > best_score:
                best_box = box
                best_score = score

        if best_box is None:
            promo_ratio = float((promo_mask > 0).mean())

            if promo_ratio < 0.01:
                return None, None, debug

            full_box = BoundingBox(0, 0, width, height)
            return image, full_box, debug

        pad_x = max(12, int(best_box.width * 0.18))
        pad_y = max(10, int(best_box.height * 0.28))

        expanded = BoundingBox(
            x_min=best_box.x_min - pad_x,
            y_min=best_box.y_min - pad_y,
            x_max=best_box.x_max + pad_x,
            y_max=best_box.y_max + pad_y,
        ).clamp(width, height)

        cv2.rectangle(
            debug,
            (expanded.x_min, expanded.y_min),
            (expanded.x_max, expanded.y_max),
            (0, 255, 0),
            3,
        )

        tag = image[expanded.y_min:expanded.y_max, expanded.x_min:expanded.x_max]

        return tag, expanded, debug

    def _split_colored_zone(
        self,
        tag: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray] | None:
        red_mask = _get_red_mask(tag)
        yellow_mask = _get_yellow_mask(tag)
        promo_mask = cv2.bitwise_or(red_mask, yellow_mask)

        row_mean = promo_mask.mean(axis=1)

        active_rows = np.where(row_mean > 25)[0]

        if len(active_rows) == 0:
            return None

        first_active = int(active_rows[0])
        last_active = int(active_rows[-1])

        height = tag.shape[0]

        colored_height = last_active - first_active + 1

        if colored_height < max(12, int(height * 0.08)):
            return None

        boundary = first_active

        boundary = int(np.clip(boundary, 1, height - 2))

        upper_zone = tag[:boundary]
        colored_zone = tag[boundary:]

        if colored_zone.size == 0:
            return None

        debug = tag.copy()

        cv2.line(
            debug,
            (0, boundary),
            (tag.shape[1], boundary),
            (255, 0, 0),
            3,
        )

        return upper_zone, colored_zone, boundary, debug

    def _detect_discount(
        self,
        colored_zone: np.ndarray,
    ) -> tuple[str | None, tuple[int, int, int] | None, np.ndarray, np.ndarray]:
        height, width = colored_zone.shape[:2]

        if width < 40 or height < 30:
            return None, None, colored_zone, colored_zone

        roi = colored_zone[:, :int(width * 0.48)]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        min_radius = max(10, int(min(width, height) * 0.06))
        max_radius = max(min_radius + 4, int(min(width, height) * 0.28))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(width * 0.15)),
            param1=100,
            param2=20,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        debug = roi.copy()

        if circles is None:
            text, prepared = self._ocr_digits(roi)

            discount = _extract_discount_from_text(text)

            return discount, None, debug, prepared

        circles = np.uint16(np.around(circles))

        best_circle = circles[0][0]
        x, y, r = (int(best_circle[0]), int(best_circle[1]), int(best_circle[2]))

        cv2.circle(debug, (x, y), r, (0, 255, 0), 3)

        crop = roi[
            max(0, y - r):min(roi.shape[0], y + r),
            max(0, x - r):min(roi.shape[1], x + r),
        ]

        text, prepared = self._ocr_digits(crop)

        discount = _extract_discount_from_text(text)

        if discount is None:
            text, prepared = self._ocr_digits(roi)
            discount = _extract_discount_from_text(text)

        return discount, (x, y, r), debug, prepared

    def _detect_card_price(
        self,
        colored_zone: np.ndarray,
        circle: tuple[int, int, int] | None,
    ) -> tuple[str | None, np.ndarray]:
        height, width = colored_zone.shape[:2]

        if width < 50 or height < 25:
            return None, colored_zone

        if circle:
            x, _, r = circle
            start_x = min(width - 10, x + r + max(8, int(width * 0.03)))
        else:
            start_x = int(width * 0.35)

        roi = colored_zone[
            :max(1, int(height * 0.88)),
            start_x:,
        ]

        if roi.size == 0 or roi.shape[1] < 12:
            roi = colored_zone[:, int(width * 0.30):]

        text, prepared = self._ocr_digits(roi)

        price = _extract_price_from_digit_text(text)

        return price, prepared

    def _detect_default_price(
        self,
        tag: np.ndarray,
        boundary: int,
    ) -> tuple[str | None, np.ndarray]:
        height, width = tag.shape[:2]

        y_min = max(0, boundary - max(30, int(height * 0.12)))
        y_max = min(height, boundary + max(30, int(height * 0.12)))

        x_min = int(width * 0.35)
        x_max = int(width * 0.95)

        roi = tag[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return None, tag

        text, prepared = self._ocr_digits(roi)

        price = _extract_default_price_from_text(text)

        return price, prepared

    def _ocr_digits(
        self,
        image: np.ndarray,
    ) -> tuple[str, np.ndarray]:
        variants = _make_digit_ocr_variants(image)

        best_text = ""
        best_variant = variants[0] if variants else image
        best_score = -1

        for variant in variants:
            for psm in (6, 7):
                text = self._run_tesseract_digits(variant, psm=psm)

                score = _digit_text_score(text)

                if score > best_score:
                    best_score = score
                    best_text = text
                    best_variant = variant

        return best_text, best_variant

    def _run_tesseract_digits(
        self,
        image: np.ndarray,
        psm: int,
    ) -> str:
        if not self.tesseract_path:
            return ""

        with tempfile.NamedTemporaryFile(
            suffix=".png",
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
                "--oem",
                "3",
                "--psm",
                str(psm),
                "-c",
                "tessedit_char_whitelist=0123456789%",
            ]

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )

            if completed.returncode != 0:
                return ""

            return completed.stdout.strip()
        except Exception:
            return ""
        finally:
            temp_path.unlink(missing_ok=True)

    def _estimate_confidence(
        self,
        price_default: str | None,
        price_card: str | None,
        discount_amount: str | None,
        tag_bbox: BoundingBox | None,
        tag_shape: tuple[int, ...],
    ) -> float:
        score = 0.0

        if tag_bbox is not None:
            score += 0.18

        if price_card:
            score += 0.38

        if price_default:
            score += 0.28

        if discount_amount:
            score += 0.16

        if len(tag_shape) >= 2:
            h, w = tag_shape[:2]

            if w >= 80 and h >= 45:
                score += 0.08

        return float(np.clip(score, 0.0, 1.0))


def _get_red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_1 = np.array([0, 35, 45])
    upper_1 = np.array([15, 255, 255])

    lower_2 = np.array([160, 35, 45])
    upper_2 = np.array([180, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_1, upper_1)
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)

    mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = np.ones((5, 5), np.uint8)

    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _get_yellow_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 35, 55])
    upper = np.array([42, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)

    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _make_digit_ocr_variants(image: np.ndarray) -> list[np.ndarray]:
    if image is None or image.size == 0:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    result: list[np.ndarray] = []

    for scale in (3.0, 4.0):
        scaled = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        blurred = cv2.GaussianBlur(scaled, (3, 3), 0)

        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            7,
        )

        inverted_otsu = cv2.bitwise_not(otsu)
        inverted_adaptive = cv2.bitwise_not(adaptive)

        result.extend(
            [
                otsu,
                adaptive,
                inverted_otsu,
            ]
        )

    return _deduplicate_images(result)


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


def _digit_text_score(text: str) -> int:
    digits = re.findall(r"\d", text)
    percents = text.count("%")

    return len(digits) * 10 + percents * 3


def _extract_discount_from_text(text: str) -> str | None:
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("О", "0").replace("о", "0")

    nums = re.findall(r"\d{1,2}", text)

    if not nums:
        return None

    values = []

    for item in nums:
        try:
            value = int(item)
        except ValueError:
            continue

        if 1 <= value <= 99:
            values.append(value)

    if not values:
        return None

    discount = max(values)

    return f"-{discount}%"


def _extract_price_from_digit_text(text: str) -> str | None:
    nums = re.findall(r"\d+", text)

    if not nums:
        return None

    nums = sorted(nums, key=len, reverse=True)

    main = nums[0]

    if len(main) >= 5:
        rub = int(main[:-2])
        kop = int(main[-2:])
    else:
        rub = int(main)
        kop = 99

        for item in nums[1:]:
            if len(item) == 2:
                candidate = int(item)

                if 0 <= candidate <= 99:
                    kop = candidate
                    break

    if rub < 10 or rub > 9999:
        return None

    if kop < 0 or kop > 99:
        kop = 99

    return f"{rub}.{kop:02d}"


def _extract_default_price_from_text(text: str) -> str | None:
    nums = re.findall(r"\d{2,6}", text)

    if not nums:
        return None

    nums = sorted(nums, key=len, reverse=True)

    for item in nums:
        if len(item) >= 5:
            rub = int(item[:-2])
            kop = int(item[-2:])
        else:
            rub = int(item)
            kop = 99

        if 10 <= rub <= 9999 and 0 <= kop <= 99:
            return f"{rub}.{kop:02d}"

    return None


def _contour_rectangularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)

    if area <= 0:
        return 0.0

    _, _, w, h = cv2.boundingRect(contour)
    rect_area = w * h

    if rect_area <= 0:
        return 0.0

    return float(np.clip(area / rect_area, 0.0, 1.0))
