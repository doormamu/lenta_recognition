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
from cv_module.recognition.promo_price_parcer import PromoPriceParser  # noqa: E402


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
        "--max-crops",
        type=int,
        default=None,
        help="Сколько crop-картинок максимум обработать; удобно для быстрых тестов",
    )
    parser.add_argument(
        "--promo-only",
        action="store_true",
        help="Запускать только promo price parser без общего OCR и barcode reader",
    )
    args = parser.parse_args()

    crop_rows = _read_csv(Path(args.crops_report))

    if args.max_crops is not None:
        crop_rows = crop_rows[:args.max_crops]

    output_path = Path(args.output)
    references = load_product_references(Path(args.reference_labels)) if args.reference_labels else []

    parser_engine = PriceTagFieldParser()
    barcode_reader = BarcodeReader(try_harder=True)
    ocr_engine = OCREngine()
    promo_parser = PromoPriceParser()
    output_rows: list[dict[str, str | int | float]] = []

    for crop_row in crop_rows:
        if len(output_rows) >= args.max_rows:
            break

        crop_path = Path(crop_row.get("crop_path", ""))
        image = cv2.imread(str(crop_path))

        if image is None or image.size == 0:
            continue

        promo_result = promo_parser.parse(image)
        barcode_reads = [] if args.promo_only else barcode_reader.read(image)
        ocr_text = "" if args.promo_only else ocr_engine.recognize(image).raw_text
        fields = parser_engine.parse(
            ocr_text=ocr_text,
            code_values=[read.value for read in barcode_reads],
            promo_result=promo_result,
        )
        reference_match = apply_reference_match(fields, references)
        field_values = fields.to_dict()

        if not _has_signal(field_values):
            continue

        output_row: dict[str, str | int | float] = {
            "crop_path": crop_row.get("crop_path", ""),
            "frame_index": crop_row.get("frame_index", ""),
            "timestamp_ms": crop_row.get("timestamp_ms", ""),
            "candidate_index": crop_row.get("candidate_index", ""),
            "source": crop_row.get("source", ""),
            "score": crop_row.get("score", ""),
            "reference_match": "yes" if reference_match is not None else "no",
            "promo_confidence": round(promo_result.confidence, 4),
            "promo_orientation": promo_result.orientation,
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
                "promo_confidence",
                "promo_orientation",
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def _make_excel_friendly(row: dict[str, str]) -> dict[str, str]:
    return {
        key: " ".join(str(value).split())
        for key, value in row.items()
    }


if __name__ == "__main__":
    main()

'''
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 tests/cv_module_parse_crops.py \
  --crops-report data/output/recognition_debug/crops_25_2_10.csv \
  --output data/output/recognition_debug/parsed_crops_25_2_10.csv \
  --reference-labels data/output/labeled \
  --max-rows 80
'''
