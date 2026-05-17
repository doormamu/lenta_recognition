from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.recognition.barcode_reader import BarcodeReader  # noqa: E402
from cv_module.recognition.field_parser import (  # noqa: E402
    OUTPUT_FIELDS,
    PriceTagFieldParser,
)
from cv_module.recognition.ocr_engine import OCREngine  # noqa: E402
from cv_module.recognition.product_reference import (  # noqa: E402
    apply_reference_match,
    load_product_references,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Распознать заранее собранные OCR-crops"
    )
    parser.add_argument(
        "--crops-report",
        required=True,
        help="CSV из tests/cv_module_collect_crops.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Куда сохранить распознанные поля",
    )
    parser.add_argument(
        "--reference-labels",
        default=None,
        help="CSV или папка CSV из data/output/labeled как справочник товаров, без использования bbox",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Сколько распознанных строк сохранить",
    )
    parser.add_argument(
        "--strict-signal",
        action="store_true",
        help="Сохранять только строки с распарсенными ценами/кодами, без слабого OCR-текста",
    )
    args = parser.parse_args()

    crop_rows = _read_csv(Path(args.crops_report))
    output_path = Path(args.output)
    references = load_product_references(Path(args.reference_labels)) if args.reference_labels else []

    parser_engine = PriceTagFieldParser()
    barcode_reader = BarcodeReader(try_harder=True)
    ocr_engine = OCREngine()
    output_rows: list[dict[str, str | int | float]] = []

    for crop_row in crop_rows:
        if len(output_rows) >= args.max_rows:
            break

        crop_path = Path(crop_row.get("crop_path", ""))
        image = cv2.imread(str(crop_path))

        if image is None or image.size == 0:
            continue

        barcode_reads = barcode_reader.read(image)
        ocr_result = ocr_engine.recognize(image)
        fields = parser_engine.parse(
            ocr_text=ocr_result.raw_text,
            code_values=[read.value for read in barcode_reads],
        )
        reference_match = apply_reference_match(fields, references)
        field_values = fields.to_dict()

        if not _has_signal(field_values) and (
            args.strict_signal or not _has_weak_ocr_text(ocr_result.raw_text)
        ):
            continue

        output_row: dict[str, str | int | float] = {
            "crop_path": crop_row.get("crop_path", ""),
            "frame_index": crop_row.get("frame_index", ""),
            "timestamp_ms": crop_row.get("timestamp_ms", ""),
            "candidate_index": crop_row.get("candidate_index", ""),
            "source": crop_row.get("source", ""),
            "score": crop_row.get("score", ""),
            "reference_match": "yes" if reference_match is not None else "no",
            "ocr_text": _compact_ocr_text(ocr_result.raw_text),
        }
        output_row.update(_make_excel_friendly(field_values))
        output_rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "crop_path",
                "frame_index",
                "timestamp_ms",
                "candidate_index",
                "source",
                "score",
                "reference_match",
                "ocr_text",
                *OUTPUT_FIELDS,
            ],
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"crops: {len(crop_rows)}")
    print(f"references: {len(references)}")
    print(f"recognized_rows: {len(output_rows)}")
    print(f"output: {output_path}")

    for row in output_rows[:5]:
        print(
            " | ".join(
                [
                    f"frame={row['frame_index']}",
                    str(row["product_name"]),
                    f"price_card={row['price_card']}",
                    f"barcode={row['barcode']}",
                    f"ref={row['reference_match']}",
                ]
            )
        )


def _has_signal(field_values: dict[str, str]) -> bool:
    signal_fields = {
        "price_default",
        "price_card",
        "price_discount",
        "barcode",
        "discount_amount",
        "id_sku",
        "qr_code_barcode",
        "price1_qr",
        "price2_qr",
        "price3_qr",
        "price4_qr",
        "action_price_qr",
        "action_code_qr",
    }
    return any(
        field_values[field_name] not in {"-", "нет"}
        for field_name in signal_fields
    )


def _has_weak_ocr_text(text: str) -> bool:
    compact_text = " ".join(text.split())

    if len(compact_text) < 5:
        return False

    if len([char for char in compact_text if char.isdigit()]) >= 2:
        return True

    if len([char for char in compact_text if "а" <= char.lower() <= "я" or char.lower() == "ё"]) >= 4:
        return True

    return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def _make_excel_friendly(row: dict[str, str]) -> dict[str, str]:
    return {
        key: " ".join(str(value).split())
        for key, value in row.items()
    }


def _compact_ocr_text(text: str) -> str:
    return " | ".join(" ".join(line.split()) for line in text.splitlines() if line.strip())[:600]


if __name__ == "__main__":
    main()
