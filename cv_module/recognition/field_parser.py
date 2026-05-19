from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote_plus, urlsplit


ABSENT_VALUE = "нет"
UNKNOWN_VALUE = "-"

MIN_PRICE = 10.0
MAX_PRICE = 4999.99

OUTPUT_FIELDS = [
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

FIELD_ALIASES = {
    "barcode": "qr_code_barcode",
    "bar_code": "qr_code_barcode",
    "ean": "qr_code_barcode",
    "ean13": "qr_code_barcode",
    "gtin": "qr_code_barcode",
    "sku": "id_sku",
    "id": "id_sku",
    "idsku": "id_sku",
    "id_sku": "id_sku",
    "price": "price1_qr",
    "price1": "price1_qr",
    "price_1": "price1_qr",
    "price2": "price2_qr",
    "price_2": "price2_qr",
    "price3": "price3_qr",
    "price_3": "price3_qr",
    "price4": "price4_qr",
    "price_4": "price4_qr",
    "actionprice": "action_price_qr",
    "action_price": "action_price_qr",
    "actioncode": "action_code_qr",
    "action_code": "action_code_qr",
    "wl1": "wholesale_level_1_count",
    "wl1count": "wholesale_level_1_count",
    "wholesale_level_1_count": "wholesale_level_1_count",
    "wl1price": "wholesale_level_1_price",
    "wholesale_level_1_price": "wholesale_level_1_price",
    "wl2": "wholesale_level_2_count",
    "wl2count": "wholesale_level_2_count",
    "wholesale_level_2_count": "wholesale_level_2_count",
    "wl2price": "wholesale_level_2_price",
    "wholesale_level_2_price": "wholesale_level_2_price",
}

CSV_FIELD_ALIASES = {
    "wholesale_level_1_coun": "wholesale_level_1_count",
}


@dataclass
class PriceTagFields:
    values: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PriceTagFields":
        return cls(values={name: UNKNOWN_VALUE for name in OUTPUT_FIELDS})

    def set_value(self, field_name: str, value: Any, source: str) -> None:
        normalized_name = CSV_FIELD_ALIASES.get(field_name, field_name)

        if normalized_name not in OUTPUT_FIELDS:
            return

        normalized_value = normalize_value(value)

        if normalized_value == UNKNOWN_VALUE:
            return

        self.values[normalized_name] = normalized_value
        self.sources[normalized_name] = source

    def set_value_if_unknown(self, field_name: str, value: Any, source: str) -> None:
        normalized_name = CSV_FIELD_ALIASES.get(field_name, field_name)

        if normalized_name not in OUTPUT_FIELDS:
            return

        current_value = self.values.get(normalized_name, UNKNOWN_VALUE)

        if current_value != UNKNOWN_VALUE:
            return

        self.set_value(normalized_name, value, source)

    def to_dict(self) -> dict[str, str]:
        return {name: self.values.get(name, UNKNOWN_VALUE) for name in OUTPUT_FIELDS}


class PriceTagFieldParser:
    def parse(
        self,
        ocr_text: str = "",
        code_values: Iterable[str] | None = None,
        labeled_row: dict[str, Any] | None = None,
        promo_result: Any | None = None,
    ) -> PriceTagFields:
        fields = PriceTagFields.empty()

        if labeled_row is not None:
            self._apply_labeled_row(fields, labeled_row)

        # ВАЖНО:
        # code_values — это реально считанные QR/штрихкоды.
        # Им доверяем больше, чем OCR-тексту.
        for code_value in code_values or []:
            self._apply_code_payload(fields, code_value)

        if ocr_text:
            self._apply_ocr_text(fields, ocr_text)

        if promo_result is not None:
            self._apply_promo_result(fields, promo_result)

        self._derive_missing_fields(fields)

        return fields

    def _apply_labeled_row(
        self,
        fields: PriceTagFields,
        row: dict[str, Any],
    ) -> None:
        for raw_name, value in row.items():
            field_name = CSV_FIELD_ALIASES.get(raw_name, raw_name)

            if field_name in OUTPUT_FIELDS:
                fields.set_value(field_name, value, source="labeled_csv")

    def _apply_code_payload(
        self,
        fields: PriceTagFields,
        value: str,
    ) -> None:
        payload = str(value).strip()

        if not payload:
            return

        barcode = _normalize_barcode(payload)

        if barcode:
            fields.set_value("barcode", barcode, source="code_raw")
            fields.set_value("qr_code_barcode", barcode, source="code_raw")
            return

        fields.set_value("qr_code_barcode", payload, source="code_raw")

        for key, item_value in _extract_key_values(payload).items():
            normalized_key = _normalize_key(key)
            field_name = FIELD_ALIASES.get(normalized_key)

            if field_name is None:
                continue

            fields.set_value(field_name, item_value, source="code_payload")

        qr_barcode = _normalize_barcode(fields.values.get("qr_code_barcode", ""))

        if qr_barcode:
            fields.set_value("barcode", qr_barcode, source="code_payload")
            fields.set_value("qr_code_barcode", qr_barcode, source="code_payload")

    def _apply_ocr_text(
        self,
        fields: PriceTagFields,
        text: str,
    ) -> None:
        compact_text = " ".join(text.split())
        normalized_text = normalize_ocr_text(compact_text)

        # ВАЖНО:
        # Баркод из OCR НЕ достаем.
        # На твоих данных OCR часто принимает случайные цифры за barcode,
        # а настоящий barcode с ценника пока не парсится.
        # Баркод будем получать либо из реально считанного QR/штрихкода,
        # либо из products.csv по найденному названию.

        discount = _extract_discount(normalized_text)

        if discount:
            fields.set_value("discount_amount", discount, source="ocr_text")

        print_datetime = _extract_print_datetime(normalized_text)

        if print_datetime:
            fields.set_value("print_datetime", print_datetime, source="ocr_text")

        code = _extract_price_tag_code(normalized_text)

        if code:
            fields.set_value("code", code, source="ocr_text")

        prices = _extract_prices_conservative(normalized_text)
        price_default, price_card = _choose_default_and_card_prices(
            prices=prices,
            discount=fields.values.get("discount_amount", UNKNOWN_VALUE),
        )

        if price_default:
            fields.set_value("price_default", price_default, source="ocr_text")

        if price_card:
            fields.set_value("price_card", price_card, source="ocr_text")

        id_sku = _extract_id_sku(
            text=normalized_text,
            barcode=fields.values.get("barcode", UNKNOWN_VALUE),
            prices=[price_default, price_card],
        )

        if id_sku:
            fields.set_value("id_sku", id_sku, source="ocr_text")

        product_name = _extract_product_hint(text)

        if product_name:
            fields.set_value("product_name", product_name, source="ocr_text")

    def _apply_promo_result(
        self,
        fields: PriceTagFields,
        promo_result: Any,
    ) -> None:
        confidence = _get_promo_attr(promo_result, "confidence")

        try:
            confidence_value = float(confidence or 0.0)
        except ValueError:
            confidence_value = 0.0

        if confidence_value < 0.35:
            return

        promo_values = _promo_to_dict(promo_result)

        promo_card = _normalize_price(promo_values.get("price_card", ""))
        promo_default = _normalize_price(promo_values.get("price_default", ""))
        promo_discount = promo_values.get("discount_amount", "")

        if promo_card and not _is_price_in_range(promo_card):
            promo_card = ""

        if promo_default and not _is_price_in_range(promo_default):
            promo_default = ""

        if promo_card and promo_default:
            if float(promo_card) > float(promo_default):
                promo_card = ""
                promo_default = ""

        if promo_default:
            _set_if_not_labeled(fields, "price_default", promo_default, "promo_price_parser")

        if promo_card:
            _set_if_not_labeled(fields, "price_card", promo_card, "promo_price_parser")

        if promo_discount and _discount_to_number(promo_discount) is not None:
            _set_if_not_labeled(fields, "discount_amount", promo_discount, "promo_price_parser")

    def _derive_missing_fields(
        self,
        fields: PriceTagFields,
    ) -> None:
        barcode = _normalize_barcode(fields.values.get("barcode", UNKNOWN_VALUE))
        qr_barcode = _normalize_barcode(fields.values.get("qr_code_barcode", UNKNOWN_VALUE))

        if barcode and not qr_barcode:
            fields.set_value("qr_code_barcode", barcode, source="derived_from_barcode")

        if qr_barcode and not barcode:
            fields.set_value("barcode", qr_barcode, source="derived_from_qr_code_barcode")


def normalize_value(value: Any) -> str:
    if value is None:
        return UNKNOWN_VALUE

    text = str(value).replace("\xa0", " ").strip()

    if not text:
        return UNKNOWN_VALUE

    lowered = text.lower()

    if lowered in {"нет", "no", "none", "null", "nan"}:
        return ABSENT_VALUE

    if lowered in {"-", "—", "–"}:
        return UNKNOWN_VALUE

    return text


def normalize_ocr_text(text: str) -> str:
    return (
        str(text)
        .replace("\xa0", " ")
        .replace("−", "-")
        .replace("—", "-")
        .replace("O", "0")
        .replace("o", "0")
        .replace("О", "0")
        .replace("о", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
    )


def _set_if_not_labeled(
    fields: PriceTagFields,
    field_name: str,
    value: str,
    source: str,
) -> None:
    current_source = fields.sources.get(field_name)

    if current_source == "labeled_csv":
        return

    fields.set_value(field_name, value, source=source)


def _promo_to_dict(promo_result: Any) -> dict[str, str]:
    if isinstance(promo_result, dict):
        return {
            key: str(value)
            for key, value in promo_result.items()
            if value is not None
        }

    if hasattr(promo_result, "to_field_values"):
        values = promo_result.to_field_values()

        if isinstance(values, dict):
            return {
                key: str(value)
                for key, value in values.items()
                if value is not None
            }

    result: dict[str, str] = {}

    for field_name in ("price_default", "price_card", "discount_amount"):
        value = _get_promo_attr(promo_result, field_name)

        if value is not None:
            result[field_name] = str(value)

    return result


def _get_promo_attr(promo_result: Any, name: str) -> Any:
    if isinstance(promo_result, dict):
        return promo_result.get(name)

    return getattr(promo_result, name, None)


def _extract_key_values(payload: str) -> dict[str, str]:
    result: dict[str, str] = {}
    decoded = unquote_plus(str(payload))

    parsed_url = urlsplit(decoded)

    if parsed_url.query:
        for key, values in parse_qs(parsed_url.query, keep_blank_values=False).items():
            if values:
                result[key] = values[-1]

    for chunk in re.split(r"[;&|,\n\r]+", decoded):
        if "=" not in chunk and ":" not in chunk:
            continue

        if "=" in chunk:
            key, value = chunk.split("=", 1)
        else:
            key, value = chunk.split(":", 1)

        key = key.strip().strip("{}[]\"'")
        value = value.strip().strip("{}[]\"'")

        if key and value:
            result[key] = value

    return result


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", key.strip().lower())


def _extract_discount(text: str) -> str:
    matches = re.findall(r"[-]?\s*(\d{1,2})\s*[%оo]", text)

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


def _discount_to_number(value: str) -> int | None:
    if not value or value == UNKNOWN_VALUE:
        return None

    match = re.search(r"\d{1,2}", value)

    if not match:
        return None

    try:
        number = int(match.group(0))
    except ValueError:
        return None

    if 1 <= number <= 90:
        return number

    return None


def _normalize_barcode(value: str) -> str:
    digits = re.sub(r"\D", "", normalize_ocr_text(str(value)))

    if len(digits) in {8, 12, 13, 14} and not _is_repeated_digits(digits):
        return digits

    return ""


def _gtin_checksum_ok(digits: str) -> bool:
    if not digits.isdigit():
        return False

    if len(digits) not in {8, 12, 13, 14}:
        return False

    body = digits[:-1]
    expected_check = int(digits[-1])

    total = 0

    for index, digit_char in enumerate(reversed(body)):
        digit = int(digit_char)
        weight = 3 if index % 2 == 0 else 1
        total += digit * weight

    actual_check = (10 - (total % 10)) % 10

    return actual_check == expected_check


def _is_repeated_digits(digits: str) -> bool:
    return len(set(digits)) <= 2


def _extract_print_datetime(text: str) -> str:
    patterns = [
        r"\b(\d{2})[.](\d{2})[.](\d{4})\s+(\d{1,2})[:.](\d{2})\b",
        r"\b(\d{2})[.](\d{2})[.](\d{2})\s+(\d{1,2})[:.](\d{2})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

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


def _extract_price_tag_code(text: str) -> str:
    match = re.search(
        r"\b\d{2,3}[_\s-]\d{3,6}(?:\s*[-–]\s*\d{2,3}[_\s-]?\d{3,6})?\b",
        text,
    )

    if not match:
        return ""

    return " ".join(match.group(0).replace("_", " ").split())


def _extract_id_sku(
    text: str,
    barcode: str,
    prices: list[str],
) -> str:
    price_digits = {
        re.sub(r"\D", "", price)
        for price in prices
        if price and price != UNKNOWN_VALUE
    }

    barcode_digits = re.sub(r"\D", "", barcode or "")

    candidates: list[str] = []

    for match in re.finditer(r"\b\d{6}\s+\d{6}\b", text):
        candidates.append(re.sub(r"\D", "", match.group(0)))

    for match in re.finditer(r"\b\d{6,12}\b", text):
        candidates.append(match.group(0))

    for digits in candidates:
        if not digits:
            continue

        if digits == barcode_digits:
            continue

        if digits in price_digits:
            continue

        if len(digits) in {8, 12, 13, 14} and _gtin_checksum_ok(digits):
            continue

        if _looks_like_date_fragment(digits):
            continue

        if _is_repeated_digits(digits):
            continue

        if 6 <= len(digits) <= 12:
            return digits

    return ""


def _looks_like_date_fragment(digits: str) -> bool:
    if len(digits) not in {6, 8, 12}:
        return False

    if digits.startswith("20"):
        return True

    if digits.endswith("2024") or digits.endswith("2025") or digits.endswith("2026"):
        return True

    return False


def _extract_prices_conservative(text: str) -> list[str]:
    candidates: list[float] = []

    cleaned = _remove_known_non_price_fragments(text)

    for match in re.finditer(r"\b(\d{1,5})[,.](\d{2})\b", cleaned):
        rub = int(match.group(1))
        kop = int(match.group(2))
        price = float(f"{rub}.{kop:02d}")

        if _is_price_in_range(price):
            candidates.append(price)

    for whole, cents in re.findall(r"\b(\d{2,4})\s{1,3}(\d{2})\b", cleaned):
        rub = int(whole)
        kop = int(cents)
        price = float(f"{rub}.{kop:02d}")

        if _is_price_in_range(price):
            candidates.append(price)

    for match in re.finditer(r"\b\d{4,6}\b", cleaned):
        digits = match.group(0)

        if len(digits) in {8, 12, 13, 14}:
            continue

        if _looks_like_date_fragment(digits):
            continue

        price = _compact_digits_to_price(digits)

        if price is not None and _is_price_in_range(price):
            candidates.append(price)

    unique = sorted(set(round(price, 2) for price in candidates), reverse=True)

    return [f"{price:.2f}" for price in unique[:6]]


def _remove_known_non_price_fragments(text: str) -> str:
    result = text

    result = re.sub(r"\b\d{2}[.]\d{2}[.]\d{2,4}\s+\d{1,2}[:.]\d{2}\b", " ", result)
    result = re.sub(r"\b\d[\d ]{7,22}\d\b", " ", result)
    result = re.sub(r"[-]?\s*\d{1,2}\s*[%оo]", " ", result)

    return result


def _compact_digits_to_price(digits: str) -> float | None:
    if not digits.isdigit():
        return None

    if len(digits) < 4 or len(digits) > 6:
        return None

    rub = int(digits[:-2])
    kop = int(digits[-2:])

    if kop < 0 or kop > 99:
        return None

    price = float(f"{rub}.{kop:02d}")

    if not _is_price_in_range(price):
        return None

    return price


def _normalize_price(value: str) -> str:
    text = str(value).replace(",", ".").strip()

    match = re.search(r"\d{1,5}[.]\d{2}", text)

    if not match:
        return ""

    price = float(match.group(0))

    if not _is_price_in_range(price):
        return ""

    return f"{price:.2f}"


def _is_price_in_range(value: float | str) -> bool:
    try:
        price = float(str(value).replace(",", "."))
    except ValueError:
        return False

    return MIN_PRICE <= price <= MAX_PRICE


def _choose_default_and_card_prices(
    prices: list[str],
    discount: str,
) -> tuple[str, str]:
    numeric_prices = []

    for price in prices:
        try:
            numeric_prices.append(float(price.replace(",", ".")))
        except ValueError:
            continue

    numeric_prices = sorted(set(numeric_prices), reverse=True)

    if not numeric_prices:
        return "", ""

    if len(numeric_prices) == 1:
        return "", f"{numeric_prices[0]:.2f}"

    discount_value = _discount_to_number(discount)

    best_pair = None
    best_score = -999.0

    for high in numeric_prices:
        for low in numeric_prices:
            if high <= low:
                continue

            if high / max(low, 1.0) > 4.0:
                continue

            score = 0.0
            calc_discount = round((1.0 - low / high) * 100.0)

            if discount_value is not None:
                diff = abs(calc_discount - discount_value)
                score += max(0.0, 20.0 - diff)

                if diff > 20:
                    score -= 20.0

            score += high / 1000.0
            score -= abs(high - low) / max(high, 1.0)

            if score > best_score:
                best_score = score
                best_pair = (high, low)

    if best_pair:
        high, low = best_pair
        return f"{high:.2f}", f"{low:.2f}"

    return "", f"{numeric_prices[0]:.2f}"


def _extract_product_hint(text: str) -> str:
    candidate_lines: list[tuple[int, str]] = []

    product_tokens = [
        "вино",
        "сыр",
        "молоко",
        "йогурт",
        "кефир",
        "сок",
        "чай",
        "кофе",
        "напиток",
        "пиво",
        "вода",
        "масло",
        "шоколад",
        "конфеты",
        "хлеб",
        "батон",
        "pure",
        "haut",
        "marin",
        "altitude",
        "jardin",
        "charmes",
        "sauvignon",
        "merlot",
        "blanc",
        "блан",
        "социньон",
        "совиньон",
    ]

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip(" |[]{}()")

        if len(line) < 4:
            continue

        if not re.search(r"[A-Za-zА-Яа-яЁё]", line):
            continue

        if _line_looks_service_or_numeric(line):
            continue

        cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё .,%/()-]+", " ", line)
        cleaned = " ".join(cleaned.split())

        if len(cleaned) < 4:
            continue

        tokens = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", cleaned)
        letters_count = len(re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned))
        digits_count = len(re.findall(r"\d", cleaned))

        if letters_count < 3:
            continue

        short_tokens = [token for token in tokens if len(token) <= 3]
        latin_tokens = [token for token in tokens if re.fullmatch(r"[A-Za-z]+", token)]
        cyrillic_tokens = [token for token in tokens if re.search(r"[А-Яа-яЁё]", token)]

        # Отсекаем строки вида "TH US al fe hte"
        if len(tokens) >= 4 and len(short_tokens) / max(len(tokens), 1) > 0.65:
            continue

        # Если вся строка латиницей и выглядит как набор случайных слов,
        # оставляем только если в ней есть известный товарный токен/бренд.
        if len(latin_tokens) >= 4 and not cyrillic_tokens:
            lowered_tmp = cleaned.lower()

            if not any(product_token in lowered_tmp for product_token in product_tokens):
                continue

        score = 0
        score += letters_count

        lowered = cleaned.lower()

        for product_token in product_tokens:
            if product_token in lowered:
                score += 35

        meaningful_tokens = [
            token
            for token in tokens
            if len(token) >= 4
        ]

        score += 8 * len(meaningful_tokens)

        if digits_count >= 6:
            score -= 30

        if cyrillic_tokens:
            score += 20

        candidate_lines.append((score, cleaned))

    if not candidate_lines:
        return ""

    candidate_lines.sort(key=lambda item: item[0], reverse=True)

    selected_lines: list[str] = []

    for _, line in candidate_lines:
        if line in selected_lines:
            continue

        selected_lines.append(line)

        if len(selected_lines) >= 3:
            break

    return " ".join(selected_lines)[:240]


def _line_looks_service_or_numeric(text: str) -> bool:
    lowered = text.lower()
    normalized = normalize_ocr_text(lowered)

    service_tokens = [
        "без карты",
        "по карте",
        "цена",
        "руб",
        "коп",
        "скид",
        "штрих",
        "qr",
        "код",
        "id",
        "sku",
        "печати",
        "дата",
    ]

    if any(token in lowered for token in service_tokens):
        return True

    if re.search(r"\d{2}[.]\d{2}[.]\d{2,4}", normalized):
        return True

    if re.search(r"[-]?\s*\d{1,2}\s*%", normalized):
        return True

    digits = re.sub(r"\D", "", normalized)

    if len(digits) >= 8:
        return True

    return False