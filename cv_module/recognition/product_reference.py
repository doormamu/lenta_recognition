from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from cv_module.recognition.field_parser import (
    OUTPUT_FIELDS,
    UNKNOWN_VALUE,
    PriceTagFields,
    normalize_value,
)


REFERENCE_FIELDS = [
    "product_name",
    "barcode",
    "qr_code_barcode",
    "price_default",
    "price_card",
    "discount_amount",
    "id_sku",
    "additional_info",
    "color",
    "special_symbols",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "action_price_qr",
    "action_code_qr",
]


STOP_WORDS = {
    "цен",
    "цена",
    "руб",
    "коп",
    "скид",
    "карта",
    "карты",
    "код",
    "qr",
    "id",
    "sku",
    "товар",
    "товара",
    "штрих",
    "штрихкод",
    "без",
    "для",
    "при",
    "или",
    "the",
    "and",
    "with",
    "from",
}


PRODUCT_NAME_ALIASES = {
    "product_name",
    "name",
    "title",
    "full_name",
    "fullname",
    "full name",
    "product",
    "товар",
    "название",
    "наименование",
    "полное название",
    "полное_название",
    "название товара",
    "наименование товара",
}


BARCODE_ALIASES = {
    "barcode",
    "bar_code",
    "ean",
    "ean13",
    "gtin",
    "code",
    "код",
    "штрихкод",
    "штрих-код",
    "штрих код",
    "штрих_код",
    "баркод",
    "barcode_digits",
}


@dataclass(frozen=True)
class ProductReferenceEntry:
    values: dict[str, str]
    product_name: str
    barcode: str
    normalized_name: str
    name_tokens: set[str]
    trigrams: set[str]


@dataclass(frozen=True)
class ProductReferenceMatch:
    entry: ProductReferenceEntry
    score: float
    reason: str


class ProductReferenceIndex:
    def __init__(self, entries: list[ProductReferenceEntry]) -> None:
        self.entries = entries
        self.by_token: dict[str, list[int]] = {}

        for index, entry in enumerate(entries):
            for token in entry.name_tokens:
                self.by_token.setdefault(token, []).append(index)

    def __len__(self) -> int:
        return len(self.entries)

    def find_name_candidates(
        self,
        detected_name: str,
        max_candidates: int = 4000,
    ) -> list[ProductReferenceEntry]:
        detected_tokens = _name_tokens(detected_name)

        if not detected_tokens:
            return []

        candidate_ids: set[int] = set()

        # 1. Быстрый поиск по общим токенам.
        for token in detected_tokens:
            for index in self.by_token.get(token, []):
                candidate_ids.add(index)

        # 2. Если точных токенов нет, пробуем по похожим токенам.
        # Это нужно, когда OCR немного исказил слово.
        if not candidate_ids:
            all_index_ids: set[int] = set()

            for token in detected_tokens:
                if len(token) < 5:
                    continue

                prefix = token[:4]

                for reference_token, indexes in self.by_token.items():
                    if not reference_token.startswith(prefix):
                        continue

                    all_index_ids.update(indexes)

            candidate_ids = all_index_ids

        if not candidate_ids:
            return []

        candidates = [self.entries[index] for index in candidate_ids]

        return candidates[:max_candidates]


def load_product_references(path: Path | None) -> ProductReferenceIndex:
    if path is None:
        return ProductReferenceIndex([])

    if not path.exists():
        print(f"WARNING: reference path not found: {path}")
        return ProductReferenceIndex([])

    csv_paths = sorted(path.glob("*.csv")) if path.is_dir() else [path]

    entries: list[ProductReferenceEntry] = []
    seen: set[tuple[str, str]] = set()

    for csv_path in csv_paths:
        rows = _read_csv_flexible(csv_path)

        for row in rows:
            product_name = _get_value_by_aliases(row, PRODUCT_NAME_ALIASES)
            barcode = _get_value_by_aliases(row, BARCODE_ALIASES)

            product_name = normalize_value(product_name)
            barcode = _normalize_barcode_loose(barcode)

            if product_name == UNKNOWN_VALUE:
                continue

            normalized_name = _normalize_name(product_name)
            name_tokens = _name_tokens(product_name)

            if not normalized_name or not name_tokens:
                continue

            key = (normalized_name, barcode)

            if key in seen:
                continue

            seen.add(key)

            values = {field: UNKNOWN_VALUE for field in OUTPUT_FIELDS}
            values["product_name"] = product_name

            if barcode:
                values["barcode"] = barcode
                values["qr_code_barcode"] = barcode

            entries.append(
                ProductReferenceEntry(
                    values=values,
                    product_name=product_name,
                    barcode=barcode,
                    normalized_name=normalized_name,
                    name_tokens=name_tokens,
                    trigrams=_trigrams(normalized_name),
                )
            )

    return ProductReferenceIndex(entries)


def apply_reference_match(
    fields: PriceTagFields,
    references: ProductReferenceIndex | list[ProductReferenceEntry],
    min_score: float = 55.0,
) -> ProductReferenceMatch | None:
    if isinstance(references, list):
        index = ProductReferenceIndex(references)
    else:
        index = references

    if len(index) == 0:
        return None

    detected_name = fields.values.get("product_name", UNKNOWN_VALUE)

    if not _known(detected_name):
        return None

    candidates = index.find_name_candidates(detected_name)

    if not candidates:
        return None

    best_entry: ProductReferenceEntry | None = None
    best_score = 0.0
    best_reason = ""

    for entry in candidates:
        score, reason = score_reference_by_name(detected_name, entry)

        if score > best_score:
            best_entry = entry
            best_score = score
            best_reason = reason

    if best_entry is None or best_score < min_score:
        return None

    # ВАЖНО:
    # Так как barcode с ценника не парсится, мы берем barcode именно из справочника,
    # но только после достаточно уверенного совпадения по названию.
    fields.set_value(
        "product_name",
        best_entry.product_name,
        source="product_reference_name_match",
    )

    if best_entry.barcode:
        fields.set_value(
            "barcode",
            best_entry.barcode,
            source="product_reference_name_match",
        )
        fields.set_value(
            "qr_code_barcode",
            best_entry.barcode,
            source="product_reference_name_match",
        )

    for field_name in REFERENCE_FIELDS:
        if field_name in {"product_name", "barcode", "qr_code_barcode"}:
            continue

        current_value = fields.values.get(field_name, UNKNOWN_VALUE)
        reference_value = best_entry.values.get(field_name, UNKNOWN_VALUE)

        if current_value == UNKNOWN_VALUE and reference_value != UNKNOWN_VALUE:
            fields.set_value(
                field_name,
                reference_value,
                source="product_reference_name_match",
            )

    return ProductReferenceMatch(
        entry=best_entry,
        score=best_score,
        reason=best_reason,
    )


def score_reference(
    fields: PriceTagFields,
    entry: ProductReferenceEntry,
) -> tuple[float, str]:
    detected_name = fields.values.get("product_name", UNKNOWN_VALUE)

    if not _known(detected_name):
        return 0.0, ""

    return score_reference_by_name(detected_name, entry)


def score_reference_by_name(
    detected_name: str,
    entry: ProductReferenceEntry,
) -> tuple[float, str]:
    detected_normalized = _normalize_name(detected_name)
    detected_tokens = _name_tokens(detected_name)
    detected_trigrams = _trigrams(detected_normalized)

    if not detected_normalized or not detected_tokens:
        return 0.0, ""

    reasons: list[str] = []
    score = 0.0

    token_overlap = _token_overlap_sets(detected_tokens, entry.name_tokens)
    token_partial = _partial_token_score(detected_tokens, entry.name_tokens)
    trigram_score = _trigram_similarity(detected_trigrams, entry.trigrams)

    sequence_score = SequenceMatcher(
        None,
        detected_normalized,
        entry.normalized_name,
    ).ratio()

    # Токены важнее всего: если OCR вытащил хотя бы бренд/часть названия,
    # это самый полезный сигнал.
    if token_overlap > 0:
        score += 80.0 * token_overlap
        reasons.append(f"token_overlap_{token_overlap:.2f}")

    if token_partial > 0:
        score += 65.0 * token_partial
        reasons.append(f"token_partial_{token_partial:.2f}")

    if trigram_score > 0:
        score += 35.0 * trigram_score
        reasons.append(f"trigram_{trigram_score:.2f}")

    if sequence_score >= 0.25:
        score += 25.0 * sequence_score
        reasons.append(f"sequence_{sequence_score:.2f}")

    # Защита от ложных матчей:
    # если совпал только один короткий токен, сильно штрафуем.
    matched_tokens = _matched_token_count(detected_tokens, entry.name_tokens)

    if matched_tokens == 0:
        score = 0.0
        reasons.append("no_matched_tokens")

    elif matched_tokens == 1:
        longest_detected = max((len(token) for token in detected_tokens), default=0)

        if longest_detected < 5:
            score *= 0.35
            reasons.append("single_short_token_penalty")
        else:
            score *= 0.70
            reasons.append("single_token_penalty")

    # Если OCR-текст почти весь из мусорных латинских коротких токенов,
    # не даем ему случайно матчиться на огромном справочнике.
    if _looks_like_ocr_garbage(detected_name):
        score *= 0.55
        reasons.append("ocr_garbage_penalty")

    return score, "+".join(reasons)


def _read_csv_flexible(csv_path: Path) -> list[dict[str, Any]]:
    encodings = ["cp1251", "utf-8-sig", "utf-8"]
    separators = [";", ",", "\t"]

    last_error: Exception | None = None

    for encoding in encodings:
        for separator in separators:
            try:
                with csv_path.open("r", encoding=encoding, newline="") as file:
                    sample = file.read(8192)
                    file.seek(0)

                    if separator not in sample:
                        continue

                    reader = csv.DictReader(file, delimiter=separator)
                    rows = [dict(row) for row in reader]

                    if rows and reader.fieldnames:
                        print(
                            f"loaded reference: {csv_path}, "
                            f"encoding={encoding}, sep={separator!r}, rows={len(rows)}"
                        )
                        return rows
            except Exception as exc:
                last_error = exc

    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                sample = file.read(8192)
                file.seek(0)

                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(file, dialect=dialect)
                rows = [dict(row) for row in reader]

                if rows and reader.fieldnames:
                    print(
                        f"loaded reference: {csv_path}, "
                        f"encoding={encoding}, sep=sniffer, rows={len(rows)}"
                    )
                    return rows
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error

    return []


def _get_value_by_aliases(
    row: dict[str, Any],
    aliases: set[str],
) -> str:
    normalized_columns = {
        _normalize_column_name(column): column
        for column in row.keys()
    }

    for alias in aliases:
        source_column = normalized_columns.get(_normalize_column_name(alias))

        if source_column is None:
            continue

        value = row.get(source_column, "")

        if value is not None and str(value).strip():
            return str(value)

    return ""


def _normalize_column_name(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("\ufeff", "")
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def _normalize_barcode_loose(value: str) -> str:
    digits = _normalize_digits(value)

    # code в справочнике считаем barcode.
    # Допускаем 6-14 цифр, потому что в реальных справочниках бывают
    # внутренние коды и EAN-коды разной длины.
    if 6 <= len(digits) <= 14 and len(set(digits)) > 1:
        return digits

    return ""


def _normalize_digits(value: str) -> str:
    text = str(value)

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

    return re.sub(r"\D", "", text)


def _known(value: str) -> bool:
    return value not in {"", UNKNOWN_VALUE, "нет", None}


def _normalize_name(value: str) -> str:
    text = str(value).lower()
    text = text.replace("ё", "е")

    # Частые OCR-замены оставляем очень мягкими,
    # чтобы не превратить весь текст в цифры.
    text = text.replace("|", " ")
    text = text.replace("_", " ")

    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _name_tokens(value: str) -> set[str]:
    normalized = _normalize_name(value)

    tokens = set()

    for token in re.findall(r"[0-9a-zа-я]{3,}", normalized):
        if token in STOP_WORDS:
            continue

        if token.isdigit():
            continue

        # Короткие латинские OCR-обрывки типа TH, US, Ni, Rs не нужны.
        if re.fullmatch(r"[a-z]{3}", token):
            continue

        tokens.add(token)

    return tokens


def _token_overlap_sets(
    first_tokens: set[str],
    second_tokens: set[str],
) -> float:
    if not first_tokens or not second_tokens:
        return 0.0

    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def _partial_token_score(
    detected_tokens: set[str],
    reference_tokens: set[str],
) -> float:
    if not detected_tokens or not reference_tokens:
        return 0.0

    matched = 0

    for detected_token in detected_tokens:
        for reference_token in reference_tokens:
            ratio = SequenceMatcher(None, detected_token, reference_token).ratio()

            if (
                ratio >= 0.78
                or detected_token in reference_token
                or reference_token in detected_token
            ):
                matched += 1
                break

    return matched / max(len(detected_tokens), 1)


def _matched_token_count(
    detected_tokens: set[str],
    reference_tokens: set[str],
) -> int:
    count = 0

    for detected_token in detected_tokens:
        for reference_token in reference_tokens:
            ratio = SequenceMatcher(None, detected_token, reference_token).ratio()

            if (
                ratio >= 0.78
                or detected_token in reference_token
                or reference_token in detected_token
            ):
                count += 1
                break

    return count


def _trigrams(value: str) -> set[str]:
    value = _normalize_name(value).replace(" ", "")

    if len(value) < 3:
        return set()

    return {
        value[index:index + 3]
        for index in range(len(value) - 2)
    }


def _trigram_similarity(
    first: set[str],
    second: set[str],
) -> float:
    if not first or not second:
        return 0.0

    return len(first & second) / len(first | second)


def _looks_like_ocr_garbage(value: str) -> bool:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", value)

    if not tokens:
        return True

    short_tokens = [token for token in tokens if len(token) <= 3]
    latin_tokens = [token for token in tokens if re.fullmatch(r"[A-Za-z]+", token)]
    cyrillic_tokens = [token for token in tokens if re.search(r"[А-Яа-яЁё]", token)]

    if len(tokens) >= 4 and len(short_tokens) / len(tokens) > 0.65:
        return True

    if len(latin_tokens) >= 4 and not cyrillic_tokens:
        # Если все распозналось как английский мусор,
        # лучше занизить уверенность матчинга.
        return True

    return False