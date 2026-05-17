from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from cv_module.detection.candidate_merger import BoundingBox  # noqa: E402
from cv_module.detection.price_tag_detector import PriceTagDetector  # noqa: E402
from cv_module.video.frame_sampler import sample_video_frames  # noqa: E402
from cv_module.video.reader import get_video_metadata  # noqa: E402


@dataclass(frozen=True)
class ProbeRegion:
    frame_index: int
    timestamp_ms: float
    region_type: str
    bbox: BoundingBox
    image: np.ndarray


@dataclass(frozen=True)
class ProbeFinding:
    frame_index: int
    timestamp_ms: float
    region_type: str
    region_bbox: BoundingBox
    reader: str
    code_type: str
    value: str
    variant_name: str
    scale: float
    crop_path: str | None = None


def make_preprocess_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []

    if image is None or image.size == 0:
        return variants

    variants.append(("original", image))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))

    equalized = cv2.equalizeHist(gray)
    variants.append(("equalized", equalized))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    variants.append(("sharp", sharp))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    variants.append(("adaptive", adaptive))

    adaptive_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    variants.append(("adaptive_inv", adaptive_inv))

    return variants


def make_scaled_variants(
    image: np.ndarray,
    scales: list[float],
) -> list[tuple[str, float, np.ndarray]]:
    result: list[tuple[str, float, np.ndarray]] = []

    for base_name, variant in make_preprocess_variants(image):
        for scale in scales:
            if scale == 1.0:
                scaled = variant
            else:
                scaled = cv2.resize(
                    variant,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            result.append((base_name, scale, scaled))

    return result


def read_with_opencv_qr(image: np.ndarray) -> list[tuple[str, str]]:
    detector = cv2.QRCodeDetector()
    result: list[tuple[str, str]] = []

    try:
        value, _, _ = detector.detectAndDecode(image)
        if value and value.strip():
            result.append(("qr", value.strip()))
    except Exception:
        pass

    try:
        ok, decoded_info, _, _ = detector.detectAndDecodeMulti(image)

        if ok and decoded_info:
            for value in decoded_info:
                if value and value.strip():
                    result.append(("qr", value.strip()))
    except Exception:
        pass

    return result


def read_with_pyzbar(image: np.ndarray) -> list[tuple[str, str]]:
    try:
        from pyzbar.pyzbar import decode
    except Exception:
        return []

    result: list[tuple[str, str]] = []

    try:
        decoded_objects = decode(image)
    except Exception:
        return []

    for obj in decoded_objects:
        value = ""

        try:
            value = obj.data.decode("utf-8", errors="ignore").strip()
        except Exception:
            value = ""

        if not value:
            continue

        raw_type = str(obj.type).lower()

        if "qrcode" in raw_type:
            code_type = "qr"
        else:
            code_type = "barcode"

        result.append((code_type, value))

    return result


def read_with_zxingcpp(image: np.ndarray) -> list[tuple[str, str]]:
    try:
        import zxingcpp
    except Exception:
        return []

    result: list[tuple[str, str]] = []

    try:
        barcodes = zxingcpp.read_barcodes(image)
    except Exception:
        return []

    for barcode in barcodes:
        value = str(getattr(barcode, "text", "") or "").strip()

        if not value:
            continue

        raw_format = str(getattr(barcode, "format", "") or "").lower()

        if "qr" in raw_format:
            code_type = "qr"
        else:
            code_type = "barcode"

        result.append((code_type, value))

    return result


def normalize_value(value: str) -> str:
    return " ".join(value.strip().split())


def probe_region(
    region: ProbeRegion,
    output_dir: Path,
    save_positive_crops: bool,
    scales: list[float],
) -> list[ProbeFinding]:
    findings: list[ProbeFinding] = []

    variants = make_scaled_variants(region.image, scales=scales)

    for variant_name, scale, image_variant in variants:
        reader_results: list[tuple[str, str, str]] = []

        for code_type, value in read_with_zxingcpp(image_variant):
            reader_results.append(("zxingcpp", code_type, value))

        for code_type, value in read_with_pyzbar(image_variant):
            reader_results.append(("pyzbar", code_type, value))

        for code_type, value in read_with_opencv_qr(image_variant):
            reader_results.append(("opencv_qr", code_type, value))

        for reader, code_type, value in reader_results:
            value = normalize_value(value)

            if not value:
                continue

            crop_path: str | None = None

            if save_positive_crops:
                crop_path = save_positive_crop(
                    image=region.image,
                    output_dir=output_dir,
                    frame_index=region.frame_index,
                    region_type=region.region_type,
                    reader=reader,
                    code_type=code_type,
                    value=value,
                    variant_name=variant_name,
                    scale=scale,
                )

            findings.append(
                ProbeFinding(
                    frame_index=region.frame_index,
                    timestamp_ms=region.timestamp_ms,
                    region_type=region.region_type,
                    region_bbox=region.bbox,
                    reader=reader,
                    code_type=code_type,
                    value=value,
                    variant_name=variant_name,
                    scale=scale,
                    crop_path=crop_path,
                )
            )

    return deduplicate_findings(findings)


def save_positive_crop(
    image: np.ndarray,
    output_dir: Path,
    frame_index: int,
    region_type: str,
    reader: str,
    code_type: str,
    value: str,
    variant_name: str,
    scale: float,
) -> str:
    crops_dir = output_dir / "positive_code_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    safe_value = "".join(
        char if char.isalnum() else "_"
        for char in value[:40]
    )

    filename = (
        f"frame_{frame_index:06d}_"
        f"{region_type}_"
        f"{reader}_"
        f"{code_type}_"
        f"{variant_name}_"
        f"scale_{scale:.1f}_"
        f"{safe_value}.jpg"
    )

    path = crops_dir / filename
    cv2.imwrite(str(path), image)

    return str(path)


def deduplicate_findings(
    findings: list[ProbeFinding],
) -> list[ProbeFinding]:
    result: list[ProbeFinding] = []
    seen: set[tuple[Any, ...]] = set()

    for item in findings:
        key = (
            item.frame_index,
            item.region_type,
            item.reader,
            item.code_type,
            item.value,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def generate_tile_regions(
    frame: np.ndarray,
    frame_index: int,
    timestamp_ms: float,
    tile_size: int,
    overlap: float,
) -> list[ProbeRegion]:
    frame_height, frame_width = frame.shape[:2]

    if tile_size <= 0:
        return []

    overlap = float(np.clip(overlap, 0.0, 0.90))
    step = max(1, int(tile_size * (1.0 - overlap)))

    regions: list[ProbeRegion] = []

    y_values = list(range(0, max(1, frame_height - tile_size + 1), step))
    x_values = list(range(0, max(1, frame_width - tile_size + 1), step))

    if not y_values or y_values[-1] + tile_size < frame_height:
        y_values.append(max(0, frame_height - tile_size))

    if not x_values or x_values[-1] + tile_size < frame_width:
        x_values.append(max(0, frame_width - tile_size))

    for y in y_values:
        for x in x_values:
            bbox = BoundingBox(
                x_min=x,
                y_min=y,
                x_max=min(frame_width, x + tile_size),
                y_max=min(frame_height, y + tile_size),
            )

            crop = frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

            if crop.size == 0:
                continue

            regions.append(
                ProbeRegion(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    region_type="tile",
                    bbox=bbox,
                    image=crop,
                )
            )

    return regions


def generate_candidate_regions(
    frame: np.ndarray,
    frame_index: int,
    timestamp_ms: float,
    max_candidates: int,
    padding_ratio: float,
) -> list[ProbeRegion]:
    frame_height, frame_width = frame.shape[:2]

    detector = PriceTagDetector(
        max_candidates=max_candidates,
        min_candidate_score=0.35,
    )

    result = detector.detect(
        frame=frame,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )

    regions: list[ProbeRegion] = []

    for candidate_index, candidate in enumerate(result.candidates):
        bbox = candidate.bbox.expand(
            frame_width=frame_width,
            frame_height=frame_height,
            left=padding_ratio,
            top=padding_ratio,
            right=padding_ratio,
            bottom=padding_ratio,
        )

        crop = frame[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]

        if crop.size == 0:
            continue

        regions.append(
            ProbeRegion(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                region_type=f"candidate_{candidate_index}_{candidate.source}",
                bbox=bbox,
                image=crop,
            )
        )

    return regions


def save_probe_report(
    findings: list[ProbeFinding],
    output_dir: Path,
) -> None:
    report_path = output_dir / "qr_probe_report.csv"

    fieldnames = [
        "frame_index",
        "timestamp_ms",
        "timestamp_sec",
        "region_type",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
        "reader",
        "code_type",
        "value",
        "variant_name",
        "scale",
        "crop_path",
    ]

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in findings:
            bbox = item.region_bbox

            writer.writerow(
                {
                    "frame_index": item.frame_index,
                    "timestamp_ms": round(item.timestamp_ms, 2),
                    "timestamp_sec": round(item.timestamp_ms / 1000.0, 2),
                    "region_type": item.region_type,
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                    "width": bbox.width,
                    "height": bbox.height,
                    "reader": item.reader,
                    "code_type": item.code_type,
                    "value": item.value,
                    "variant_name": item.variant_name,
                    "scale": item.scale,
                    "crop_path": item.crop_path or "",
                }
            )


def save_debug_frame(
    frame: np.ndarray,
    frame_index: int,
    timestamp_ms: float,
    findings: list[ProbeFinding],
    output_dir: Path,
) -> None:
    debug_dir = output_dir / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)

    debug = frame.copy()

    for idx, finding in enumerate(findings):
        bbox = finding.region_bbox
        x_min, y_min, x_max, y_max = bbox.to_tuple()

        color = (255, 0, 0) if finding.code_type == "qr" else (255, 0, 255)

        cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), color, 4)

        label = f"{idx}:{finding.reader}:{finding.code_type}:{finding.value[:24]}"

        label_y = max(22, y_min - 10)

        cv2.rectangle(
            debug,
            (x_min, label_y - 18),
            (min(debug.shape[1] - 1, x_min + 520), label_y + 5),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            debug,
            label,
            (x_min, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    path = (
        debug_dir
        / f"frame_{frame_index:06d}_time_{timestamp_ms / 1000.0:08.2f}_findings_{len(findings):02d}.jpg"
    )

    cv2.imwrite(str(path), debug)


def save_probe_regions(
    regions: list[ProbeRegion],
    output_dir: Path,
    frame_index: int,
    limit: int,
) -> None:
    regions_dir = output_dir / "probe_regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    for idx, region in enumerate(regions[:limit]):
        path = (
            regions_dir
            / f"frame_{frame_index:06d}_region_{idx:03d}_{region.region_type}.jpg"
        )

        cv2.imwrite(str(path), region.image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Отдельная диагностика чтения QR/штрихкодов на видео"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Путь до видеофайла",
    )

    parser.add_argument(
        "--output",
        default="data/output/qr_probe",
        help="Папка для результатов",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=4.0,
        help="Сколько кадров в секунду предварительно смотреть",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.5,
        help="Размер окна выбора лучшего кадра",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help="Максимальное число кадров",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.10,
        help="Минимальная оценка качества кадра",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=768,
        help="Размер tile для поиска QR на частях кадра",
    )

    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.35,
        help="Перекрытие tile от 0 до 0.9",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="Сколько candidate crops брать из детектора",
    )

    parser.add_argument(
        "--candidate-padding",
        type=float,
        default=0.20,
        help="Расширение candidate bbox перед QR-probe",
    )

    parser.add_argument(
        "--save-positive-crops",
        action="store_true",
        help="Сохранять crops, где что-то считалось",
    )

    parser.add_argument(
        "--save-probe-regions",
        action="store_true",
        help="Сохранять часть всех регионов, где пытались читать QR",
    )

    parser.add_argument(
        "--probe-regions-limit",
        type=int,
        default=80,
        help="Сколько probe regions сохранять на кадр",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_video_metadata(video_path)

    print("Видео:")
    print(f"  path: {metadata.path}")
    print(f"  fps: {metadata.fps:.2f}")
    print(f"  frame_count: {metadata.frame_count}")
    print(f"  size: {metadata.width}x{metadata.height}")
    print(f"  duration_sec: {metadata.duration_sec:.2f}")

    sampled_frames = sample_video_frames(
        video_path=video_path,
        target_fps=args.target_fps,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
        min_quality_score=args.min_quality,
    )

    print()
    print(f"Выбрано кадров: {len(sampled_frames)}")

    all_findings: list[ProbeFinding] = []

    scales = [1.0, 2.0, 3.0, 4.0]

    for sampled in sampled_frames:
        frame = sampled.image
        frame_height, frame_width = frame.shape[:2]

        full_region = ProbeRegion(
            frame_index=sampled.frame_index,
            timestamp_ms=sampled.timestamp_ms,
            region_type="full_frame",
            bbox=BoundingBox(
                x_min=0,
                y_min=0,
                x_max=frame_width,
                y_max=frame_height,
            ),
            image=frame,
        )

        tile_regions = generate_tile_regions(
            frame=frame,
            frame_index=sampled.frame_index,
            timestamp_ms=sampled.timestamp_ms,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
        )

        candidate_regions = generate_candidate_regions(
            frame=frame,
            frame_index=sampled.frame_index,
            timestamp_ms=sampled.timestamp_ms,
            max_candidates=args.max_candidates,
            padding_ratio=args.candidate_padding,
        )

        regions = [full_region] + tile_regions + candidate_regions

        if args.save_probe_regions:
            save_probe_regions(
                regions=regions,
                output_dir=output_dir,
                frame_index=sampled.frame_index,
                limit=args.probe_regions_limit,
            )

        frame_findings: list[ProbeFinding] = []

        for region in regions:
            findings = probe_region(
                region=region,
                output_dir=output_dir,
                save_positive_crops=args.save_positive_crops,
                scales=scales,
            )

            frame_findings.extend(findings)

        frame_findings = deduplicate_findings(frame_findings)

        save_debug_frame(
            frame=frame,
            frame_index=sampled.frame_index,
            timestamp_ms=sampled.timestamp_ms,
            findings=frame_findings,
            output_dir=output_dir,
        )

        all_findings.extend(frame_findings)

        print(
            f"frame={sampled.frame_index:06d}, "
            f"time={sampled.timestamp_ms / 1000.0:8.2f}s, "
            f"regions={len(regions):4d}, "
            f"findings={len(frame_findings):3d}"
        )

    all_findings = deduplicate_findings(all_findings)

    save_probe_report(
        findings=all_findings,
        output_dir=output_dir,
    )

    summary = {
        "video": str(video_path),
        "frames": len(sampled_frames),
        "findings": len(all_findings),
        "unique_values": sorted({item.value for item in all_findings}),
        "readers": sorted({item.reader for item in all_findings}),
    }

    with (output_dir / "qr_probe_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print()
    print("Итог:")
    print(f"  frames: {len(sampled_frames)}")
    print(f"  findings: {len(all_findings)}")
    print(f"  unique values: {len(summary['unique_values'])}")
    print()
    print(f"Отчет: {output_dir / 'qr_probe_report.csv'}")
    print(f"Summary: {output_dir / 'qr_probe_summary.json'}")
    print(f"Debug frames: {output_dir / 'debug_frames'}")

    if args.save_positive_crops:
        print(f"Positive crops: {output_dir / 'positive_code_crops'}")

    if args.save_probe_regions:
        print(f"Probe regions: {output_dir / 'probe_regions'}")


if __name__ == "__main__":
    main()


'''
python tests/qr_probe.py \
  --video data/input/labeled/25_2-10.mp4 \
  --output data/output/qr_probe \
  --target-fps 4 \
  --window-sec 0.5 \
  --max-frames 80 \
  --min-quality 0.10 \
  --tile-size 768 \
  --tile-overlap 0.35 \
  --max-candidates 40 \
  --candidate-padding 0.20 \
  --save-positive-crops
'''