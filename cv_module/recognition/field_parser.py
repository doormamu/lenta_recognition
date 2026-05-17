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

        barcode_match = re.search(r"\b\d[\d ]{6,18}\d\b", compact_text)

        if barcode_match:
            fields.set_value("barcode", barcode_match.group(0), source="ocr_text")

        discount_match = re.search(r"[-−]\s?\d{1,2}\s?%", compact_text)

        if discount_match:
            fields.set_value("discount_amount", discount_match.group(0), source="ocr_text")

        price_matches = re.findall(r"\b\d{1,5}[,.]\d{2}\b", compact_text)

        if price_matches:
            fields.set_value("price_default", price_matches[0], source="ocr_text")

        if len(price_matches) >= 2:
            fields.set_value("price_card", price_matches[1], source="ocr_text")

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
