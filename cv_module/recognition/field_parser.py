from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote_plus, urlsplit


ABSENT_VALUE = "нет"
UNKNOWN_VALUE = "-"

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

    def to_dict(self) -> dict[str, str]:
        return {name: self.values.get(name, UNKNOWN_VALUE) for name in OUTPUT_FIELDS}


class PriceTagFieldParser:
    def parse(
        self,
        ocr_text: str = "",
        code_values: Iterable[str] | None = None,
        labeled_row: dict[str, Any] | None = None,
    ) -> PriceTagFields:
        fields = PriceTagFields.empty()

        if labeled_row is not None:
            self._apply_labeled_row(fields, labeled_row)

        for code_value in code_values or []:
            self._apply_code_payload(fields, code_value)

        if ocr_text:
            self._apply_ocr_text(fields, ocr_text)

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
        payload = value.strip()

        if not payload:
            return

        fields.set_value("qr_code_barcode", payload, source="code_raw")

        for key, item_value in _extract_key_values(payload).items():
            normalized_key = _normalize_key(key)
            field_name = FIELD_ALIASES.get(normalized_key)

            if field_name is None:
                continue

            fields.set_value(field_name, item_value, source="code_payload")

    def _apply_ocr_text(
        self,
        fields: PriceTagFields,
        text: str,
    ) -> None:
        compact_text = " ".join(text.split())

        barcode_match = _extract_barcode(compact_text)

        if barcode_match:
            fields.set_value("barcode", barcode_match, source="ocr_text")

        discount_match = re.search(r"[-−]?\s?[0-9OoОо]{1,2}\s?[%оo]", compact_text)

        if discount_match:
            discount = discount_match.group(0).replace(" ", "")
            discount = discount.replace("−", "-")
            discount = (
                discount
                .replace("O", "0")
                .replace("o", "0")
                .replace("О", "0")
                .replace("о", "0")
            )
            discount = discount.rstrip("оo") + "%"

            if not discount.startswith(("-", "−")):
                discount = f"-{discount}"

            discount = re.sub(r"^-0+(\d)", r"-\1", discount)

            fields.set_value("discount_amount", discount, source="ocr_text")

        print_datetime = _extract_print_datetime(compact_text)

        if print_datetime:
            fields.set_value("print_datetime", print_datetime, source="ocr_text")

        code = _extract_price_tag_code(compact_text)

        if code:
            fields.set_value("code", code, source="ocr_text")

        price_matches = _extract_prices(compact_text)

        if price_matches:
            default_price, card_price = _choose_default_and_card_prices(price_matches)

            if default_price:
                fields.set_value("price_default", default_price, source="ocr_text")

            if card_price:
                fields.set_value("price_card", card_price, source="ocr_text")

        if fields.values.get("product_name") == UNKNOWN_VALUE:
            product_name = _extract_product_hint(text)

            if product_name:
                fields.set_value("product_name", product_name, source="ocr_text")

    def _derive_missing_fields(
        self,
        fields: PriceTagFields,
    ) -> None:
        values = fields.values

        if values.get("barcode", UNKNOWN_VALUE) == UNKNOWN_VALUE:
            qr_barcode = values.get("qr_code_barcode", UNKNOWN_VALUE)

            if qr_barcode not in {UNKNOWN_VALUE, ABSENT_VALUE}:
                values["barcode"] = qr_barcode
                fields.sources["barcode"] = "derived_from_qr_code_barcode"


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


def _extract_key_values(payload: str) -> dict[str, str]:
    result: dict[str, str] = {}
    decoded = unquote_plus(payload)

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


def _extract_product_hint(text: str) -> str:
    cleaned_lines: list[str] = []
    product_tokens = [
        "вино",
        "напиток",
        "продукт",
        "мед",
        "молоко",
        "сыр",
        "йогурт",
        "кефир",
        "сок",
        "чай",
        "кофе",
        "пиво",
    ]

    candidate_lines: list[tuple[int, str]] = []

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip(" |[]{}()")

        if len(line) < 4:
            continue

        if not re.search(r"[А-Яа-яЁё]", line):
            continue

        if re.search(r"\d{2,5}", line):
            continue

        if len(re.findall(r"[А-Яа-яЁё]", line)) < 3:
            continue

        lowered = line.lower()

        if any(token in lowered for token in ["руб", "коп", "цена", "скид", "штрих", "qr", "код"]):
            continue

        line = re.sub(r"[^0-9A-Za-zА-Яа-яЁё .,%/-]+", " ", line)
        line = " ".join(line.split())

        if not line:
            continue

        score = len(re.findall(r"[А-Яа-яЁёA-Za-z]", line))

        if any(token in lowered for token in product_tokens):
            score += 30

        if re.search(r"\b\d{1,2}[,.]\s?\d{1,2}\b", line):
            score += 5

        candidate_lines.append((score, line))

        if len(candidate_lines) >= 5:
            break

    if not candidate_lines:
        return ""

    candidate_lines.sort(key=lambda item: item[0], reverse=True)
    cleaned_lines = [line for _, line in candidate_lines[:2]]

    return " ".join(cleaned_lines[:3])[:240]


def _extract_barcode(text: str) -> str:
    for match in re.finditer(r"\b\d[\d ]{6,22}\d\b", text):
        compact = re.sub(r"\D", "", match.group(0))

        if len(compact) in {8, 12, 13, 14}:
            return compact

    return ""


def _extract_print_datetime(text: str) -> str:
    match = re.search(
        r"\b(\d{2})[.](\d{2})[.](\d{4})\s+(\d{1,2})[:.](\d{2})\b",
        text,
    )

    if not match:
        return ""

    day, month, year, hour, minute = match.groups()
    return f"{day}.{month}.{year} {int(hour)}:{minute}"


def _extract_price_tag_code(text: str) -> str:
    match = re.search(r"\b\d{2,3}[_\s-]\d{3,6}(?:\s*[-–]\s*\d{2,3}[_\s-]?\d{3,6})?\b", text)

    if not match:
        return ""

    return " ".join(match.group(0).replace("_", " ").split())


def _extract_prices(text: str) -> list[str]:
    result: list[str] = []

    for match in re.findall(r"\b\d{1,5}[,.]\d{2}\b", text):
        result.append(match)

    for whole, cents in re.findall(r"\b(\d{2,5})\s{1,3}(\d{2})\b", text):
        result.append(f"{whole}.{cents}")

    for match in re.finditer(r"\b\d{3,5}\b", text):
        whole = match.group(0)
        previous_char = text[match.start() - 1] if match.start() > 0 else ""
        next_char = text[match.end()] if match.end() < len(text) else ""

        if previous_char in {".", ","} or next_char in {".", ","}:
            continue

        if any(whole in price for price in result):
            continue

        if len(whole) == 5:
            prefix = text[max(0, match.start() - 2):match.start()]

            if "#" in prefix or whole[0] not in {"1", "2"}:
                continue

            if whole.startswith("10") or whole.startswith("20"):
                trimmed = whole[0] + whole[-3:]
            else:
                trimmed = whole[-4:]

            if 100 <= int(trimmed) <= 9999:
                result.append(f"{trimmed}.99")

            continue

        result.append(f"{whole}.99")

    deduplicated: list[str] = []

    for price in result:
        normalized = price.replace(",", ".")

        try:
            numeric_price = float(normalized)
        except ValueError:
            continue

        if numeric_price < 10.0 or numeric_price > 9999.99:
            continue

        if normalized not in deduplicated:
            deduplicated.append(normalized)

    return deduplicated[:4]


def _choose_default_and_card_prices(prices: list[str]) -> tuple[str, str]:
    numeric_prices = sorted(
        {
            float(price.replace(",", "."))
            for price in prices
            if _is_float(price)
        },
        reverse=True,
    )

    numeric_prices = [
        price
        for price in numeric_prices
        if price >= 50.0
    ]

    if not numeric_prices:
        return "", ""

    if len(numeric_prices) == 1:
        return "", _format_price(numeric_prices[0])

    highest = numeric_prices[0]
    second = numeric_prices[1]

    if highest / max(second, 1.0) >= 1.20:
        return _format_price(highest), _format_price(second)

    return "", _format_price(highest)


def _is_float(value: str) -> bool:
    try:
        float(value.replace(",", "."))
        return True
    except ValueError:
        return False


def _format_price(value: float) -> str:
    return f"{value:.2f}"
