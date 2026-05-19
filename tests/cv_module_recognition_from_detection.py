from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from cv_module.recognition.barcode_reader import BarcodeReader
from cv_module.recognition.field_parser import PriceTagFieldParser
from cv_module.recognition.frame_preprocessor import FrameLevelPreprocessor
from cv_module.recognition.ocr_engine import OCREngine
from cv_module.recognition.recognition_preprocessor import (
    RecognitionPreprocessor,
    parse_bbox_from_row,
    save_recognition_crop_debug,
)

try:
    from cv_module.recognition.color_detector import detect_price_tag_color
except Exception:
    detect_price_tag_color = None

try:
    from cv_module.recognition.promo_price_parser import PromoPriceParser
except ModuleNotFoundError:
    try:
        from cv_module.recognition.promo_price_parcer import PromoPriceParser
    except ModuleNotFoundError:
        PromoPriceParser = None

try:
    from cv_module.recognition.product_reference import (
        apply_reference_match,
        load_product_references,
    )
except Exception:
    apply_reference_match = None
    load_product_references = None


@dataclass(frozen=True)
class OCRCandidate:
    engine_name: str
    variant_name: str
    raw_text: str
    confidence: float
    result: Any | None = None


@dataclass(frozen=True)
class ColorFallbackResult:
    color: str
    confidence: float
    ratios: dict[str, float]


def read_detection_rows(
    detection_report_path: Path,
    min_score: float,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with detection_report_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            score = safe_float(row.get("score", "0"))

            if score < min_score:
                continue

            rows.append(dict(row))

            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def read_frame_at_index(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def fields_to_row(fields: Any) -> dict[str, str]:
    if hasattr(fields, "to_dict"):
        return fields.to_dict()

    if isinstance(fields, dict):
        return {str(key): str(value) for key, value in fields.items()}

    return {}


def parse_fields_safely(
    parser: PriceTagFieldParser,
    ocr_text: str,
    code_values: list[str],
    promo_result: Any | None,
):
    try:
        return parser.parse(
            ocr_text=ocr_text,
            code_values=code_values,
            promo_result=promo_result,
        )
    except TypeError:
        fields = parser.parse(
            ocr_text=ocr_text,
            code_values=code_values,
        )

        if promo_result is not None:
            if hasattr(promo_result, "to_field_values") and hasattr(fields, "set_value"):
                for field_name, value in promo_result.to_field_values().items():
                    fields.set_value(
                        field_name=field_name,
                        value=value,
                        source="promo_price_parser",
                    )

        return fields


def run_ocr_on_variants(
    engine: OCREngine,
    parsing_images: list[tuple[str, Any]],
) -> list[OCRCandidate]:
    candidates: list[OCRCandidate] = []

    for variant_name, image in parsing_images:
        try:
            result = engine.recognize(image)
        except Exception as exc:
            print(f"OCR error: variant={variant_name}, error={exc}")
            continue

        raw_text = getattr(result, "raw_text", "") or ""
        confidence = safe_float(getattr(result, "confidence", 0.0))

        candidates.append(
            OCRCandidate(
                engine_name="tesseract",
                variant_name=variant_name,
                raw_text=raw_text,
                confidence=confidence,
                result=result,
            )
        )

    return candidates


def choose_best_ocr_candidate(
    candidates: list[OCRCandidate],
) -> OCRCandidate:
    if not candidates:
        return OCRCandidate(
            engine_name="none",
            variant_name="",
            raw_text="",
            confidence=0.0,
            result=None,
        )

    return sorted(
        candidates,
        key=ocr_candidate_score,
        reverse=True,
    )[0]


def ocr_candidate_score(candidate: OCRCandidate) -> float:
    raw_text = candidate.raw_text or ""
    text_len = len(raw_text.strip())

    if text_len == 0:
        return 0.0

    digits_count = sum(char.isdigit() for char in raw_text)
    cyrillic_count = sum(("А" <= char <= "я") or char in "Ёё" for char in raw_text)
    latin_count = sum(("A" <= char <= "z") for char in raw_text)

    text_score = min(text_len / 140.0, 1.0)
    digit_score = min(digits_count / 14.0, 1.0)
    cyrillic_score = min(cyrillic_count / 24.0, 1.0)
    latin_score = min(latin_count / 18.0, 1.0)

    variant_bonus = 0.0

    if candidate.variant_name == "crop":
        variant_bonus += 0.05

    if candidate.variant_name == "crop_rot90_ccw":
        variant_bonus += 0.03

    short_penalty = 0.30 if text_len < 4 else 0.0

    return max(
        0.0,
        0.25 * candidate.confidence
        + 0.25 * text_score
        + 0.25 * digit_score
        + 0.15 * cyrillic_score
        + 0.05 * latin_score
        + variant_bonus
        - short_penalty,
    )


def collect_barcode_values(
    barcode_reader: BarcodeReader,
    images: list[tuple[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    values: list[str] = []
    details: list[dict[str, Any]] = []
    seen: set[str] = set()

    for variant_name, image in images:
        try:
            reads = barcode_reader.read(image)
        except Exception as exc:
            details.append(
                {
                    "variant": variant_name,
                    "error": str(exc),
                }
            )
            continue

        for item in reads:
            value = str(getattr(item, "value", "") or "").strip()

            if not value:
                continue

            details.append(
                {
                    "variant": variant_name,
                    "value": value,
                    "code_type": getattr(item, "code_type", ""),
                    "confidence": getattr(item, "confidence", ""),
                    "source": getattr(item, "source", ""),
                }
            )

            if value in seen:
                continue

            seen.add(value)
            values.append(value)

    return values, details


def choose_best_promo_result(
    promo_results: list[tuple[str, Any]],
) -> tuple[str, Any | None]:
    if not promo_results:
        return "", None

    def score(item: tuple[str, Any]) -> float:
        variant_name, result = item

        confidence = safe_float(getattr(result, "confidence", 0.0))

        has_card = bool(getattr(result, "price_card", None))
        has_default = bool(getattr(result, "price_default", None))
        has_discount = bool(getattr(result, "discount_amount", None))

        variant_bonus = 0.0

        if variant_name == "crop":
            variant_bonus += 0.04

        if variant_name == "crop_rot90_ccw":
            variant_bonus += 0.03

        return (
            confidence
            + 0.18 * has_card
            + 0.12 * has_default
            + 0.08 * has_discount
            + variant_bonus
        )

    return sorted(promo_results, key=score, reverse=True)[0]


def run_promo_parser(
    promo_parser: Any,
    parsing_images: list[tuple[str, Any]],
) -> tuple[str, Any | None]:
    if promo_parser is None:
        return "", None

    promo_results: list[tuple[str, Any]] = []

    for variant_name, image in parsing_images:
        try:
            promo_result = promo_parser.parse(
                image,
                with_debug=False,
            )
            promo_results.append((variant_name, promo_result))
        except Exception as exc:
            print(f"Promo parser error: variant={variant_name}, error={exc}")

    return choose_best_promo_result(promo_results)


def apply_reference_match_safely(
    fields: Any,
    references: Any,
    min_score: float,
):
    if references is None:
        return None

    try:
        if len(references) == 0:
            return None
    except Exception:
        pass

    if apply_reference_match is None:
        return None

    try:
        return apply_reference_match(
            fields=fields,
            references=references,
            min_score=min_score,
        )
    except TypeError:
        try:
            return apply_reference_match(
                fields,
                references,
            )
        except Exception as exc:
            print(f"Reference match error: {exc}")
            return None
    except Exception as exc:
        print(f"Reference match error: {exc}")
        return None


def make_recognition_preprocessor(args: argparse.Namespace) -> RecognitionPreprocessor:
    try:
        return RecognitionPreprocessor(
            padding_ratio=args.padding_ratio,
            min_padding_px=8,
            upscale_factor=args.upscale_factor,
            max_parsing_variants=args.max_parsing_variants,
            generate_rotations=False,
            enable_perspective_rectification=False,
            frame_rotation_mode=args.frame_rotation,
        )
    except TypeError:
        return RecognitionPreprocessor(
            padding_ratio=args.padding_ratio,
            min_padding_px=8,
            upscale_factor=args.upscale_factor,
            max_parsing_variants=args.max_parsing_variants,
            generate_rotations=False,
            enable_perspective_rectification=False,
        )


def extract_recognition_crop(
    preprocessor: RecognitionPreprocessor,
    raw_frame: Any,
    bbox: Any,
    prepared_frame: Any | None,
):
    try:
        return preprocessor.extract(
            raw_frame=raw_frame,
            bbox=bbox,
            prepared_frame=prepared_frame,
        )
    except TypeError:
        return preprocessor.extract(
            raw_frame=raw_frame,
            bbox=bbox,
        )


def detect_color_safely(image: Any) -> ColorFallbackResult:
    if detect_price_tag_color is None:
        return ColorFallbackResult(
            color="-",
            confidence=0.0,
            ratios={},
        )

    try:
        result = detect_price_tag_color(image)

        return ColorFallbackResult(
            color=str(getattr(result, "color", "-") or "-"),
            confidence=safe_float(getattr(result, "confidence", 0.0)),
            ratios=dict(getattr(result, "ratios", {}) or {}),
        )
    except Exception as exc:
        print(f"Color detection error: {exc}")

        return ColorFallbackResult(
            color="-",
            confidence=0.0,
            ratios={},
        )


def clear_qr_code_if_not_read_from_image(
    fields: Any,
    code_values: list[str],
) -> None:
    """
    Важно:
    barcode может быть подтянут из products.csv по названию товара.
    Но qr_code_barcode должен заполняться только если QR/штрихкод реально считался с картинки.
    """

    if code_values:
        return

    if hasattr(fields, "values"):
        fields.values["qr_code_barcode"] = "-"

    if hasattr(fields, "sources"):
        fields.sources["qr_code_barcode"] = "not_read_from_image"


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True

    text = str(value).strip()

    return text in {"", "-", "—", "–", "None", "null", "nan"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).replace(",", ".").strip()

        if not text:
            return default

        return float(text)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(safe_float(value, default=float(default))))
    except Exception:
        return default


def write_output_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recognition/parsing по detection_report.csv: "
            "frame-level preprocessing once per frame + Tesseract + field_parser + reference + color"
        )
    )

    parser.add_argument("--video", required=True)

    parser.add_argument(
        "--detection-report",
        default="data/output/model_detection_debug/detection_report.csv",
    )

    parser.add_argument(
        "--output",
        default="data/output/recognition_from_detection",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--save-debug-crops",
        action="store_true",
    )

    parser.add_argument(
        "--try-hard-barcode",
        action="store_true",
    )

    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--upscale-factor",
        type=float,
        default=1.8,
    )

    parser.add_argument(
        "--max-parsing-variants",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--frame-rotation",
        default="none",
        choices=["none", "rot90_ccw", "rot90_cw", "rot180"],
    )

    parser.add_argument(
        "--disable-frame-enhance",
        action="store_true",
    )

    parser.add_argument(
        "--frame-denoise",
        action="store_true",
    )

    parser.add_argument(
        "--enable-promo-parser",
        action="store_true",
    )

    parser.add_argument(
        "--disable-barcode",
        action="store_true",
    )

    parser.add_argument(
        "--reference-path",
        default=None,
    )

    parser.add_argument(
        "--reference-min-score",
        type=float,
        default=55.0,
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    detection_report_path = Path(args.detection_report)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not detection_report_path.exists():
        raise FileNotFoundError(f"Detection report not found: {detection_report_path}")

    detection_rows = read_detection_rows(
        detection_report_path=detection_report_path,
        min_score=args.min_score,
        max_rows=args.max_rows,
    )

    print("Входные данные:")
    print(f"  video: {video_path}")
    print(f"  detection_report: {detection_report_path}")
    print(f"  rows selected: {len(detection_rows)}")
    print()

    frame_preprocessor = FrameLevelPreprocessor(
        rotation_mode=args.frame_rotation,
        enable_enhance=not args.disable_frame_enhance,
        enable_denoise=args.frame_denoise,
    )

    preprocessor = make_recognition_preprocessor(args)

    ocr_engine = OCREngine()
    barcode_reader = BarcodeReader(try_harder=args.try_hard_barcode)
    field_parser = PriceTagFieldParser()

    if args.enable_promo_parser and PromoPriceParser is not None:
        promo_parser = PromoPriceParser()
    else:
        promo_parser = None

    if args.reference_path and load_product_references is not None:
        references = load_product_references(Path(args.reference_path))
    else:
        references = []

    try:
        references_count = len(references)
    except Exception:
        references_count = 0

    print(f"references loaded: {references_count}")
    print()

    result_rows: list[dict[str, Any]] = []

    frame_cache: dict[int, Any] = {}
    prepared_frame_cache: dict[int, Any] = {}

    for row_index, detection_row in enumerate(detection_rows):
        frame_index = safe_int(detection_row.get("frame_index", 0))

        if frame_index not in frame_cache:
            frame_cache[frame_index] = read_frame_at_index(
                video_path=video_path,
                frame_index=frame_index,
            )

        raw_frame = frame_cache[frame_index]

        if raw_frame is None:
            print(f"skip row={row_index:05d}: cannot read frame={frame_index}")
            continue

        if frame_index not in prepared_frame_cache:
            prepared_frame_cache[frame_index] = frame_preprocessor.process(raw_frame)

        prepared_frame = prepared_frame_cache[frame_index]

        bbox = parse_bbox_from_row(detection_row)

        recognition_crop = extract_recognition_crop(
            preprocessor=preprocessor,
            raw_frame=raw_frame,
            bbox=bbox,
            prepared_frame=prepared_frame,
        )

        color_result = detect_color_safely(recognition_crop.processed_crop)

        candidate_index = safe_int(
            detection_row.get("candidate_index", row_index),
            default=row_index,
        )

        crop_prefix = (
            f"row_{row_index:05d}_"
            f"frame_{frame_index:06d}_"
            f"cand_{candidate_index:03d}"
        )

        crop_paths: dict[str, str] = {}

        if args.save_debug_crops:
            crop_paths = save_recognition_crop_debug(
                recognition_crop=recognition_crop,
                output_dir=output_dir / "crops_debug",
                prefix=crop_prefix,
            )

        parsing_images = recognition_crop.get_parsing_images()

        if args.disable_barcode:
            code_values = []
            barcode_details = []
        else:
            barcode_images = [("raw_crop", recognition_crop.raw_crop)] + parsing_images
            code_values, barcode_details = collect_barcode_values(
                barcode_reader=barcode_reader,
                images=barcode_images,
            )

        ocr_candidates = run_ocr_on_variants(
            engine=ocr_engine,
            parsing_images=parsing_images,
        )

        best_ocr = choose_best_ocr_candidate(ocr_candidates)

        if promo_parser is not None:
            best_promo_variant, best_promo_result = run_promo_parser(
                promo_parser=promo_parser,
                parsing_images=parsing_images,
            )
        else:
            best_promo_variant = ""
            best_promo_result = None

        fields = parse_fields_safely(
            parser=field_parser,
            ocr_text=best_ocr.raw_text,
            code_values=code_values,
            promo_result=best_promo_result,
        )

        reference_match = apply_reference_match_safely(
            fields=fields,
            references=references,
            min_score=args.reference_min_score,
        )

        clear_qr_code_if_not_read_from_image(
            fields=fields,
            code_values=code_values,
        )

        parsed_fields = fields_to_row(fields)

        result_row: dict[str, Any] = {
            "source_video": str(video_path),
            "frame_index": frame_index,
            "timestamp_ms": detection_row.get("timestamp_ms", ""),
            "timestamp_sec": detection_row.get("timestamp_sec", ""),
            "candidate_index": detection_row.get("candidate_index", candidate_index),
            "source": detection_row.get("source", ""),
            "score": detection_row.get("score", ""),
            "x_min": bbox.x_min,
            "y_min": bbox.y_min,
            "x_max": bbox.x_max,
            "y_max": bbox.y_max,
            "width": bbox.width,
            "height": bbox.height,
            "recognition_bbox_x_min": recognition_crop.crop_bbox_raw.x_min,
            "recognition_bbox_y_min": recognition_crop.crop_bbox_raw.y_min,
            "recognition_bbox_x_max": recognition_crop.crop_bbox_raw.x_max,
            "recognition_bbox_y_max": recognition_crop.crop_bbox_raw.y_max,
            "processed_variant": recognition_crop.metadata.get("processed_variant", ""),
            "frame_rotation_mode": recognition_crop.metadata.get("frame_rotation_mode", ""),
            "parsing_variants": json.dumps(
                recognition_crop.parsing_variant_names,
                ensure_ascii=False,
            ),
            "best_ocr_engine": best_ocr.engine_name,
            "best_ocr_variant": best_ocr.variant_name,
            "best_ocr_confidence": best_ocr.confidence,
            "best_ocr_score": round(ocr_candidate_score(best_ocr), 5),
            "best_ocr_text": best_ocr.raw_text,
            "best_promo_variant": best_promo_variant,
            "promo_confidence": getattr(best_promo_result, "confidence", ""),
            "promo_orientation": getattr(best_promo_result, "orientation", ""),
            "promo_price_card": getattr(best_promo_result, "price_card", ""),
            "promo_price_default": getattr(best_promo_result, "price_default", ""),
            "promo_discount_amount": getattr(best_promo_result, "discount_amount", ""),
            "code_values": json.dumps(code_values, ensure_ascii=False),
            "barcode_details": json.dumps(barcode_details, ensure_ascii=False),
            "reference_matched": reference_match is not None,
            "reference_score": getattr(reference_match, "score", ""),
            "reference_reason": getattr(reference_match, "reason", ""),
            "detected_color": color_result.color,
            "color_confidence": round(color_result.confidence, 4),
            "color_ratios": json.dumps(color_result.ratios, ensure_ascii=False),
            "raw_crop_path": crop_paths.get("raw", ""),
            "processed_crop_path": crop_paths.get("processed", ""),
            "collage_path": crop_paths.get("collage", ""),
        }

        result_row.update(parsed_fields)

        if is_empty_value(result_row.get("color")):
            result_row["color"] = color_result.color

        result_rows.append(result_row)

        print(
            f"row={row_index:05d}, "
            f"frame={frame_index:06d}, "
            f"score={detection_row.get('score', '')}, "
            f"variants={len(parsing_images)}, "
            f"best_ocr={best_ocr.variant_name}, "
            f"ocr_len={len(best_ocr.raw_text)}, "
            f"codes={len(code_values)}, "
            f"color={color_result.color}, "
            f"ref={reference_match is not None}"
        )

    output_csv = output_dir / "recognition_results.csv"

    write_output_csv(
        rows=result_rows,
        output_path=output_csv,
    )

    summary = {
        "video": str(video_path),
        "detection_report": str(detection_report_path),
        "rows_input": len(detection_rows),
        "rows_output": len(result_rows),
        "min_score": args.min_score,
        "padding_ratio": args.padding_ratio,
        "upscale_factor": args.upscale_factor,
        "max_parsing_variants": args.max_parsing_variants,
        "frame_rotation": args.frame_rotation,
        "disable_frame_enhance": args.disable_frame_enhance,
        "frame_denoise": args.frame_denoise,
        "enable_promo_parser": args.enable_promo_parser,
        "disable_barcode": args.disable_barcode,
        "reference_path": args.reference_path,
        "references_loaded": references_count,
        "reference_min_score": args.reference_min_score,
        "pipeline": "raw_frame -> frame_preprocessing_once -> transformed_bbox -> crop/crop_rot90_ccw -> tesseract -> reference -> color",
    }

    (output_dir / "recognition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Итог:")
    print(f"  detections input: {len(detection_rows)}")
    print(f"  rows output: {len(result_rows)}")
    print(f"  output: {output_csv}")
    print(f"  summary: {output_dir / 'recognition_summary.json'}")

    if args.save_debug_crops:
        print(f"  debug crops: {output_dir / 'crops_debug'}")


if __name__ == "__main__":
    main()