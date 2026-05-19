from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any


UNKNOWN_VALUE = "-"
ABSENT_VALUE = "нет"


def is_unknown(value: Any) -> bool:
    if value is None:
        return True

    text = str(value).strip()

    if not text:
        return True

    lowered = text.lower()

    return lowered in {
        "-",
        "—",
        "–",
        "none",
        "null",
        "nan",
        "нет данных",
        "не распознано",
    }


def normalize_unknown(value: Any) -> str:
    if is_unknown(value):
        return UNKNOWN_VALUE

    text = str(value).replace("\xa0", " ").strip()

    if text.lower() == "nan":
        return UNKNOWN_VALUE

    return text


def normalize_text(value: Any, max_length: int | None = None) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    text = " ".join(text.split())

    if max_length is not None:
        text = text[:max_length]

    return text if text else UNKNOWN_VALUE


def normalize_price(value: Any) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    text = text.replace(",", ".")
    text = re.sub(r"[^\d.]", "", text)

    if not text:
        return UNKNOWN_VALUE

    matches = re.findall(r"\d{1,5}(?:\.\d{1,2})?", text)

    if not matches:
        return UNKNOWN_VALUE

    try:
        price = float(matches[0])
    except ValueError:
        return UNKNOWN_VALUE

    if not math.isfinite(price):
        return UNKNOWN_VALUE

    if price <= 0 or price > 99999:
        return UNKNOWN_VALUE

    return f"{price:.2f}"


def normalize_discount(value: Any) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    text = text.replace("−", "-").replace("—", "-").replace("–", "-")
    text = text.replace("о", "0").replace("О", "0").replace("o", "0").replace("O", "0")

    match = re.search(r"-?\s*(\d{1,2})\s*%", text)

    if not match:
        return UNKNOWN_VALUE

    try:
        discount = int(match.group(1))
    except ValueError:
        return UNKNOWN_VALUE

    if not 1 <= discount <= 90:
        return UNKNOWN_VALUE

    return f"-{discount}%"


def normalize_barcode(value: Any, allow_internal_codes: bool = True) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    text = (
        text
        .replace("O", "0")
        .replace("o", "0")
        .replace("О", "0")
        .replace("о", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
    )

    digits = re.sub(r"\D", "", text)

    if not digits:
        return UNKNOWN_VALUE

    # В products.csv поле code может быть внутренним кодом, поэтому разрешаем 6–14 цифр.
    if allow_internal_codes:
        if 6 <= len(digits) <= 14 and len(set(digits)) > 1:
            return digits
        return UNKNOWN_VALUE

    if len(digits) in {8, 12, 13, 14} and len(set(digits)) > 1:
        return digits

    return UNKNOWN_VALUE


def normalize_id_sku(value: Any) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    digits = re.sub(r"\D", "", text)

    if 6 <= len(digits) <= 14 and len(set(digits)) > 1:
        return digits

    return UNKNOWN_VALUE


def normalize_datetime(value: Any) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    text = text.replace(",", ".")
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"(\d{2})[.](\d{2})[.](\d{4})\s+(\d{1,2})[:.](\d{2})",
        r"(\d{2})[.](\d{2})[.](\d{2})\s+(\d{1,2})[:.](\d{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if not match:
            continue

        day, month, year, hour, minute = match.groups()

        if len(year) == 2:
            year = f"20{year}"

        try:
            dt = datetime(
                year=int(year),
                month=int(month),
                day=int(day),
                hour=int(hour),
                minute=int(minute),
            )
        except ValueError:
            continue

        return dt.strftime("%d.%m.%Y %H:%M")

    return UNKNOWN_VALUE


def normalize_coordinate(value: Any) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    text = text.replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return UNKNOWN_VALUE

    if not math.isfinite(number):
        return UNKNOWN_VALUE

    return f"{number:.1f}".replace(".", ",")


def normalize_timestamp(value: Any) -> str:
    text = normalize_unknown(value)

    if text == UNKNOWN_VALUE:
        return UNKNOWN_VALUE

    text = text.replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return UNKNOWN_VALUE

    if not math.isfinite(number):
        return UNKNOWN_VALUE

    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))

    return f"{number:.2f}".rstrip("0").rstrip(".")


def validate_price_pair(price_default: str, price_card: str) -> tuple[str, str]:
    price_default = normalize_price(price_default)
    price_card = normalize_price(price_card)

    if price_default == UNKNOWN_VALUE or price_card == UNKNOWN_VALUE:
        return price_default, price_card

    try:
        default_value = float(price_default)
        card_value = float(price_card)
    except ValueError:
        return price_default, price_card

    # На ценнике цена по карте не должна быть выше обычной.
    # Если наоборот — оставляем более надежную одиночную цену, вторую сбрасываем.
    if card_value > default_value:
        return UNKNOWN_VALUE, price_card

    return price_default, price_card


def normalize_output_value(value: Any) -> str:
    value = normalize_unknown(value)
    return value if value != "" else UNKNOWN_VALUE