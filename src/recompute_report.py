import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

from .aggregation import (
    aggregate_runs,
    harmonize_locations,
    DEFAULT_NUMERIC_TOLERANCES,
    normalize_location_string,
    normalize_text_value,
)


def parse_tolerance_overrides(values: List[str]) -> Dict[Tuple[str, str], float]:
    overrides: Dict[Tuple[str, str], float] = {}
    for raw in values:
        key_part, value_part = raw.split("=", 1)
        section, measurement = key_part.split(":", 1)
        overrides[(section.strip(), measurement.strip())] = float(value_part.strip())
    return overrides


def load_run_csv(path: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    numeric_groups: Dict[Tuple[str, str, str, str, str], List[float]] = {}
    for row in rows:
        group_key = (
            row.get("SourceFile"),
            row.get("Page"),
            row.get("Section"),
            row.get("Measurement"),
            row.get("MeasurementType"),
        )
        value_numeric = row.get("ValueNumeric")
        if value_numeric not in ("", None):
            try:
                numeric_groups.setdefault(group_key, []).append(float(value_numeric))
            except ValueError:
                continue

    drop_groups: set[Tuple[str, str, str, str, str]] = set()
    for key, values in numeric_groups.items():
        if values and all(abs(v) < 1e-9 for v in values):
            drop_groups.add(key)

    row_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        group_key = (
            row.get("SourceFile"),
            row.get("Page"),
            row.get("Section"),
            row.get("Measurement"),
            row.get("MeasurementType"),
        )
        if group_key in drop_groups:
            continue
        key = (
            row.get("SourceFile"),
            int(row.get("Page")),
            row.get("Section"),
            row.get("Measurement"),
            row.get("MeasurementType"),
            int(row.get("HourIndex")),
            row.get("HourLabel"),
        )
        value_numeric = row.get("ValueNumeric")
        if value_numeric in ("", None):
            numeric_value: Any = ""
        else:
            try:
                numeric_value = float(value_numeric)
            except ValueError:
                numeric_value = value_numeric
        location = normalize_location_string(row.get("Location"))
        value_text = normalize_text_value(row.get("Section"), row.get("MeasurementType"), row.get("ValueText"))
        row_map[key] = {
            **row,
            "Location": location,
            "ValueNumeric": numeric_value,
            "ValueText": value_text,
        }
    return row_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute final batch CSV and disagreement report from existing run outputs",
    )
    parser.add_argument("--runs", nargs="+", required=True, help="CSV files from individual extraction runs")
    parser.add_argument("--out_csv", default="out/batch_results_recomputed.csv", help="Path for recomputed merged CSV")
    parser.add_argument(
        "--report",
        default="out/disagreement_report_recomputed.json",
        help="Path for recomputed disagreement report JSON",
    )
    parser.add_argument(
        "--default_numeric_tolerance",
        type=float,
        default=1.0,
        help="Default tolerance when comparing numeric values across runs",
    )
    parser.add_argument(
        "--tolerance_override",
        action="append",
        default=[],
        help="Override tolerance for Section:MeasurementType (e.g. 'Wind:Speed=2.5'). May repeat.",
    )
    args = parser.parse_args()

    run_maps: List[Dict[Tuple[Any, ...], Dict[str, Any]]] = []
    for path_str in args.runs:
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"Run CSV not found: {path}")
        run_maps.append(load_run_csv(path))

    tolerance_map = dict(DEFAULT_NUMERIC_TOLERANCES)
    if args.tolerance_override:
        tolerance_map.update(parse_tolerance_overrides(args.tolerance_override))

    final_rows_map, disagreements = aggregate_runs(
        run_maps,
        tolerance_map,
        default_numeric_tolerance=args.default_numeric_tolerance,
    )
    harmonize_locations(final_rows_map)

    final_rows = list(final_rows_map.values())
    fieldnames = [
        "SourceFile",
        "Page",
        "Location",
        "Section",
        "Measurement",
        "MeasurementType",
        "Units",
        "ForecastWindowStartLocal",
        "HourLabel",
        "HourIndex",
        "TimestampLocal",
        "ValueNumeric",
        "ValueText",
        "Notes",
    ]

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            final_rows,
            key=lambda r: (
                r.get("SourceFile"),
                int(r.get("Page", 0)),
                r.get("Section"),
                int(r.get("HourIndex", 0)),
                r.get("MeasurementType"),
            ),
        ):
            writer.writerow(row)

    report = {
        "run_count": len(run_maps),
        "source_runs": [str(Path(p).resolve()) for p in args.runs],
        "disagreements": disagreements,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Recomputed CSV written to {out_csv_path}")
    print(f"Recomputed disagreement report written to {report_path}")


if __name__ == "__main__":
    main()
