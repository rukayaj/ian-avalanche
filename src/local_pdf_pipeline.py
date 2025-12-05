import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Dict

import fitz
import pandas as pd

from .extractors import series_to_rows


ALLOWED_DIRS = {
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
}
ALLOWED_PTYPE = {
    "Clear",
    "Cloudy",
    "Rain",
    "Fog",
    "Mist",
    "Partly cloudy",
    "Sunny",
    "Snow",
    "Snow showers",
    "Snow shower",
    "Sleet",
    "Drizzle",
    "Overcast",
}
HOUR_LABELS = [
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
]


def pad(vals: List, fill, length: int = 24) -> List:
    return (vals + [fill] * length)[:length]


def parse_numeric_arrays(pdf_path: Path, page_num: int) -> List[List[float]]:
    """Use pdftotext -layout to get numeric arrays on a page."""
    proc = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page_num), "-l", str(page_num), str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    nums_all: List[List[float]] = []
    for line in proc.stdout.splitlines():
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", line)]
        if len(nums) >= 24:
            nums_all.append(nums)
    return nums_all


def extract_directions(page: fitz.Page) -> List[str]:
    words = page.get_text("words")
    dirs = [(x0, t) for x0, y0, x1, y1, t, blk, line, wordno in words if t in ALLOWED_DIRS and y0 < 200]
    dirs = sorted(dirs, key=lambda a: a[0])
    return pad([t for _, t in dirs], fill="", length=24)


def extract_precip_types(page: fitz.Page) -> List[str]:
    blocks = page.get_text("blocks")
    ptypes: List[str] = []
    for b in blocks:
        x0, y0, x1, y1, text, *rest = b
        if 300 <= y0 <= 380 and text:
            for line in text.splitlines():
                t = line.strip()
                if t in ("Snow shower", "Snow showers"):
                    t = "Snow showers"
                if t in ALLOWED_PTYPE:
                    ptypes.append((x0, t))
    ptypes = sorted(ptypes, key=lambda a: a[0])
    return pad([t for _, t in ptypes], fill="", length=24)


def extract_location(page: fitz.Page, pdf_path: Path) -> str:
    for b in page.get_text("blocks"):
        txt = (b[4] or "").strip()
        if txt.startswith("Wind - "):
            return txt.replace("Wind - ", "").strip()
    return pdf_path.stem


def extract_page9_rows(pdf_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-f", "9", "-l", "9", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return rows

    text = result.stdout.splitlines()
    date_str = ""
    for line in text:
        m = re.search(r"1800 on (.+?) until", line)
        if m:
            date_str = m.group(1).strip()
            break
    if date_str:
        rows.append(
            {
                "SourceFile": pdf_path.name,
                "Page": 9,
                "Location": "",
                "Section": "Meta",
                "Measurement": "ForecastWindowStart",
                "MeasurementType": "Date",
                "Units": "",
                "ForecastWindowStartLocal": "",
                "HourLabel": "",
                "HourIndex": "",
                "TimestampLocal": "",
                "ValueNumeric": "",
                "ValueText": date_str,
                "Notes": "",
            }
        )

    row_re = re.compile(r"\s*([A-Za-z' ]{3,})\s+([0-9]+\.?[0-9]*)\s+([0-9]+\.?[0-9]*)\s*$")
    for line in text:
        m = row_re.match(line)
        if not m:
            continue
        site = m.group(1).strip()
        snow_val = float(m.group(2))
        rain_val = float(m.group(3))
        base = {
            "SourceFile": pdf_path.name,
            "Page": 9,
            "Location": site,
            "Section": "Precip",
            "ForecastWindowStartLocal": "",
            "HourLabel": "",
            "HourIndex": "",
            "TimestampLocal": "",
            "Notes": "",
        }
        rows.append(
            {
                **base,
                "Measurement": "Accumulation",
                "MeasurementType": "SnowTotal_cm",
                "Units": "cm",
                "ValueNumeric": snow_val,
                "ValueText": "",
            }
        )
        rows.append(
            {
                **base,
                "Measurement": "Accumulation",
                "MeasurementType": "RainTotal_mm",
                "Units": "mm",
                "ValueNumeric": rain_val,
                "ValueText": "",
            }
        )
    return rows


def process_pdf_local(pdf_path: Path) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    all_rows: List[Dict] = []
    try:
        # Graph pages assumed on pages 2..(min(8, page_count))
        last_graph_page = min(doc.page_count, 9)
        for page_idx in range(1, last_graph_page - 1):  # skip first page, stop before page 9
            page_num = page_idx + 1
            page = doc[page_idx]
            nums_all = parse_numeric_arrays(pdf_path, page_num)
            if len(nums_all) < 11:
                continue
            speed = pad(nums_all[1][:24], fill=0.0)
            gust = pad(nums_all[2][:24], fill=0.0)
            snow = pad(nums_all[4][:24], fill=0.0)
            rain = pad(nums_all[5][:24], fill=0.0)
            air = pad(nums_all[8][:24], fill=0.0)
            fl = pad(nums_all[9][1:25], fill=0.0)
            wbfl = pad([x for x in nums_all[10] if x < 900][:24], fill=0.0)

            dir_vals = extract_directions(page)
            ptypes = extract_precip_types(page)
            loc = extract_location(page, pdf_path)

            for i, h in enumerate(HOUR_LABELS):
                all_rows.extend(
                    series_to_rows(
                        pdf_path.name,
                        page_idx,
                        loc,
                        "wind",
                        {
                            "hours": [
                                {
                                    "hour_label": h,
                                    "hour_index": i,
                                    "wind_speed_mph": speed[i],
                                    "wind_gust_mph": gust[i],
                                    "wind_direction": dir_vals[i],
                                }
                            ]
                        },
                    )
                )
                all_rows.extend(
                    series_to_rows(
                        pdf_path.name,
                        page_idx,
                        loc,
                        "precipitation",
                        {
                            "hours": [
                                {
                                    "hour_label": h,
                                    "hour_index": i,
                                    "rain_mm": rain[i],
                                    "snow_cm": snow[i],
                                    "precip_type": ptypes[i],
                                }
                            ]
                        },
                    )
                )
                all_rows.extend(
                    series_to_rows(
                        pdf_path.name,
                        page_idx,
                        loc,
                        "temperature",
                        {
                            "hours": [
                                {
                                    "hour_label": h,
                                    "hour_index": i,
                                    "air_temp_c": air[i],
                                    "freezing_level_m": fl[i],
                                    "wet_bulb_freezing_level_m": wbfl[i],
                                }
                            ]
                        },
                    )
                )
        # Page 9 accumulations
        all_rows.extend(extract_page9_rows(pdf_path))
    finally:
        doc.close()

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Local PDF extractor (no OpenAI) for SAIS forecasts")
    parser.add_argument("--input", nargs="*", help="Optional explicit PDF paths")
    parser.add_argument("--input-dir", default="in", help="Directory to scan for PDFs when --input is not supplied")
    parser.add_argument("--out-csv", default="out/local_results.csv", help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    pdfs = [Path(p) for p in args.input] if args.input else sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found. Add files to the input directory or pass --input.")

    all_dfs = []
    for pdf in pdfs:
        df = process_pdf_local(pdf)
        all_dfs.append(df)
    out_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
