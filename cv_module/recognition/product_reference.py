from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

from cv_module.recognition.field_parser import (
    OUTPUT_FIELDS,
    UNKNOWN_VALUE,
    PriceTagFields,
    normalize_value,
)


REFERENCE_FIELDS = [
    "product_name",
    "price_default",
    "price_card",
    "barcode",
    "discount_amount",
    "id_sku",
    "additional_info",
    "color",
    "special_symbols",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "action_price_qr",
    "action_code_qr",
]


@dataclass(frozen=True)
class ProductReferenceEntry:
    values: dict[str, str]


def load_product_references(path: Path | None) -> list[ProductReferenceEntry]:
    if path is None:
        return []

    csv_paths: list[Path]

    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
    else:
        csv_paths = [path]

    entries: list[ProductReferenceEntry] = []
    seen: set[tuple[str, str, str, str]] = set()

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                values = {
                    field: normalize_value(row.get(field, ""))
                    for field in OUTPUT_FIELDS
                }
                key = (
                    values.get("product_name", UNKNOWN_VALUE),
                    values.get("barcode", UNKNOWN_VALUE),
                    values.get("price_default", UNKNOWN_VALUE),
                    values.get("price_card", UNKNOWN_VALUE),
                )

                if key in seen:
                    continue

                seen.add(key)
                entries.append(ProductReferenceEntry(values=values))

    return entries


def apply_reference_match(
    fields: PriceTagFields,
    references: list[ProductReferenceEntry],
) -> ProductReferenceEntry | None:
    if not references:
        return None

    best_entry = None
    best_score = 0.0

    for entry in references:
        score = score_reference(fields, entry)

        if score > best_score:
            best_entry = entry
            best_score = score

    if best_entry is None or best_score < 28:
        return None

    for field_name in REFERENCE_FIELDS:
        current_value = fields.values.get(field_name, UNKNOWN_VALUE)
        reference_value = best_entry.values.get(field_name, UNKNOWN_VALUE)

        if current_value == UNKNOWN_VALUE and reference_value != UNKNOWN_VALUE:
            fields.set_value(field_name, reference_value, source="product_reference")

    return best_entry


def score_reference(
    fields: PriceTagFields,
    entry: ProductReferenceEntry,
) -> float:
    values = fields.values
    reference_values = entry.values
    score = 0.0

    barcode = _first_known(values, ["barcode", "qr_code_barcode"])
    reference_barcode = _first_known(reference_values, ["barcode", "qr_code_barcode"])

    if barcode and reference_barcode and barcode == reference_barcode:
        score += 100.0

    id_sku = values.get("id_sku", UNKNOWN_VALUE)
    reference_id_sku = reference_values.get("id_sku", UNKNOWN_VALUE)

    if _known(id_sku) and id_sku == reference_id_sku:
        score += 70.0

    detected_prices = _known_prices(values)
    reference_prices = _known_prices(reference_values)

    for detected_price in detected_prices:
        for reference_price in reference_prices:
            if abs(detected_price - reference_price) <= 1.0:
                score += 16.0
                break

    discount = values.get("discount_amount", UNKNOWN_VALUE)
    reference_discount = reference_values.get("discount_amount", UNKNOWN_VALUE)

    if _known(discount) and discount == reference_discount:
        score += 10.0

    name = values.get("product_name", UNKNOWN_VALUE)
    reference_name = reference_values.get("product_name", UNKNOWN_VALUE)

    if _known(name) and _known(reference_name):
        score += 18.0 * _token_overlap(name, reference_name)

    return score


def _known_prices(values: dict[str, str]) -> list[float]:
    prices: list[float] = []

    for field_name in ("price_default", "price_card", "price1_qr", "price2_qr", "price3_qr", "price4_qr"):
        value = values.get(field_name, UNKNOWN_VALUE)

        if not _known(value):
            continue

        try:
            prices.append(float(value.replace(",", ".")))
        except ValueError:
            continue

    return prices


def _first_known(values: dict[str, str], field_names: list[str]) -> str:
    for field_name in field_names:
        value = values.get(field_name, UNKNOWN_VALUE)

        if _known(value):
            return value

    return ""


def _known(value: str) -> bool:
    return value not in {"", UNKNOWN_VALUE, "нет"}


def _token_overlap(first: str, second: str) -> float:
    first_tokens = _name_tokens(first)
    second_tokens = _name_tokens(second)

    if not first_tokens or not second_tokens:
        return 0.0

    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def _name_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", value.lower())
        if token not in {"цен", "руб", "коп"}
    }
