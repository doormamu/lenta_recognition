from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from cv_module.postprocessing.field_fusion import FINAL_COLUMNS, fuse_rows


class FinalCsvExporter:
    def __init__(
        self,
        columns: list[str] | None = None,
        delimiter: str = ",",
    ) -> None:
        self.columns = columns or FINAL_COLUMNS
        self.delimiter = delimiter

    def export(
        self,
        rows: list[dict[str, Any]],
        output_path: str | Path,
        filename: str | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_rows = fuse_rows(
            rows=rows,
            filename=filename,
        )

        with output_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.columns,
                delimiter=self.delimiter,
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in final_rows:
                writer.writerow(
                    {
                        column: row.get(column, "-")
                        for column in self.columns
                    }
                )

        return output_path


def read_csv_rows(input_path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(input_path)

    encodings = ["utf-8-sig", "utf-8", "cp1251"]
    delimiters = [",", ";", "\t"]

    last_error: Exception | None = None

    for encoding in encodings:
        for delimiter in delimiters:
            try:
                with input_path.open("r", encoding=encoding, newline="") as file:
                    sample = file.read(8192)
                    file.seek(0)

                    if delimiter not in sample:
                        continue

                    reader = csv.DictReader(file, delimiter=delimiter)
                    rows = [dict(row) for row in reader]

                    if reader.fieldnames:
                        return rows
            except Exception as exc:
                last_error = exc

    for encoding in encodings:
        try:
            with input_path.open("r", encoding=encoding, newline="") as file:
                sample = file.read(8192)
                file.seek(0)

                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(file, dialect=dialect)
                rows = [dict(row) for row in reader]

                if reader.fieldnames:
                    return rows
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Экспорт recognition_results.csv в финальный CSV-формат"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Путь до recognition_results.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Путь до итогового CSV",
    )

    parser.add_argument(
        "--filename",
        default=None,
        help="Значение для поля filename. Если не указано, берется из source_video/input.",
    )

    parser.add_argument(
        "--delimiter",
        default=",",
        help="Разделитель итогового CSV",
    )

    args = parser.parse_args()

    rows = read_csv_rows(args.input)

    exporter = FinalCsvExporter(
        delimiter=args.delimiter,
    )

    output_path = exporter.export(
        rows=rows,
        output_path=args.output,
        filename=args.filename,
    )

    print("Итоговый CSV сформирован:")
    print(f"  input rows: {len(rows)}")
    print(f"  output: {output_path}")
    print(f"  columns: {len(FINAL_COLUMNS)}")


if __name__ == "__main__":
    main()