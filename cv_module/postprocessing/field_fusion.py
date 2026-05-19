from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cv_module.postprocessing.validators import (
    UNKNOWN_VALUE,
    normalize_barcode,
    normalize_coordinate,
    normalize_datetime,
    normalize_discount,
    normalize_id_sku,
    normalize_output_value,
    normalize_price,
    normalize_text,
    normalize_timestamp,
    validate_price_pair,
)


FINAL_COLUMNS = [
    "filename",
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
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
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


TEXT_FIELDS = {
    "product_name",
    "price_discount",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "wholesale_level_1_count",
    "wholesale_level_2_count",
    "action_code_qr",
}


PRICE_FIELDS = {
    "price_default",
    "price_card",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}


COORD_FIELDS = {
    "x_min",
    "y_min",
    "x_max",
    "y_max",
}


def fuse_recognition_row(
    row: dict[str, Any],
    filename: str | None = None,
) -> dict[str, str]:
    """
    Приводит строку recognition_results.csv к финальному формату.

    Важно:
    - barcode может быть взят из products.csv по названию;
    - qr_code_barcode заполняется только если QR/штрихкод реально считался с картинки;
    - color берется из field_parser или detected_color;
    - все нераспознанные поля становятся "-".
    """

    result = {column: UNKNOWN_VALUE for column in FINAL_COLUMNS}

    result["filename"] = _resolve_filename(row, filename)

    for field_name in FINAL_COLUMNS:
        if field_name == "filename":
            continue

        if field_name in row:
            result[field_name] = row.get(field_name, UNKNOWN_VALUE)

    result["product_name"] = _first_known(
        row,
        [
            "product_name",
            "reference_product_name",
            "matched_product_name",
        ],
        default=result["product_name"],
    )

    result["barcode"] = _first_known(
        row,
        [
            "barcode",
            "reference_barcode",
            "matched_barcode",
        ],
        default=result["barcode"],
    )

    result["color"] = _first_known(
        row,
        [
            "color",
            "detected_color",
        ],
        default=result["color"],
    )

    if result["frame_timestamp"] == UNKNOWN_VALUE:
        result["frame_timestamp"] = _first_known(
            row,
            ["frame_timestamp", "timestamp_ms", "timestamp_sec"],
        )

    for field_name in COORD_FIELDS:
        result[field_name] = _first_known(
            row,
            [field_name],
            default=result[field_name],
        )

    code_values = _parse_json_list(row.get("code_values", ""))

    real_code_barcode = UNKNOWN_VALUE

    for code_value in code_values:
        barcode = normalize_barcode(code_value)

        if barcode != UNKNOWN_VALUE:
            real_code_barcode = barcode
            break

    if real_code_barcode != UNKNOWN_VALUE:
        result["barcode"] = real_code_barcode
        result["qr_code_barcode"] = real_code_barcode
    else:
        # barcode из справочника не означает, что QR/штрихкод считался с картинки.
        result["qr_code_barcode"] = UNKNOWN_VALUE

    result = normalize_final_row(result)

    return result


def normalize_final_row(row: dict[str, Any]) -> dict[str, str]:
    normalized = {column: UNKNOWN_VALUE for column in FINAL_COLUMNS}

    for column in FINAL_COLUMNS:
        value = row.get(column, UNKNOWN_VALUE)

        if column in PRICE_FIELDS:
            normalized[column] = normalize_price(value)

        elif column in COORD_FIELDS:
            normalized[column] = normalize_coordinate(value)

        elif column == "frame_timestamp":
            normalized[column] = normalize_timestamp(value)

        elif column == "barcode":
            normalized[column] = normalize_barcode(value)

        elif column == "qr_code_barcode":
            normalized[column] = normalize_barcode(value)

        elif column == "id_sku":
            normalized[column] = normalize_id_sku(value)

        elif column == "discount_amount":
            normalized[column] = normalize_discount(value)

        elif column == "print_datetime":
            normalized[column] = normalize_datetime(value)

        elif column == "product_name":
            normalized[column] = normalize_text(value, max_length=500)

        elif column in TEXT_FIELDS:
            normalized[column] = normalize_text(value, max_length=300)

        else:
            normalized[column] = normalize_output_value(value)

    normalized["price_default"], normalized["price_card"] = validate_price_pair(
        normalized["price_default"],
        normalized["price_card"],
    )

    return normalized


def fuse_rows(
    rows: list[dict[str, Any]],
    filename: str | None = None,
) -> list[dict[str, str]]:
    return [
        fuse_recognition_row(row=row, filename=filename)
        for row in rows
    ]


def _resolve_filename(row: dict[str, Any], filename: str | None) -> str:
    if filename:
        return Path(filename).stem

    value = _first_known(row, ["filename"])

    if value != UNKNOWN_VALUE:
        return Path(value).stem

    source_video = _first_known(row, ["source_video"])

    if source_video != UNKNOWN_VALUE:
        return Path(source_video).stem

    return UNKNOWN_VALUE


def _first_known(
    row: dict[str, Any],
    keys: list[str],
    default: Any = UNKNOWN_VALUE,
) -> str:
    for key in keys:
        if key not in row:
            continue

        value = row.get(key)

        if value is None:
            continue

        text = str(value).strip()

        if not text:
            continue

        if text.lower() in {"-", "—", "–", "none", "null", "nan"}:
            continue

        return text

    return normalize_output_value(default)


def _parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()

    if not text or text in {"-", "[]"}:
        return []

    try:
        parsed = json.loads(text)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item) for item in parsed if item is not None]