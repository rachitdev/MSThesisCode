"""Build a provenance-labelled FSI panel without altering the submitted thesis.

2024 is a single, manually verified Total/Rank from the official annual report;
indicator-level 2024 scores are intentionally left missing, not imputed.
"""

import argparse
import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "MainCode"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "fsi_2006_2024.csv"
SOURCES = {
    2021: ("https://fragilestatesindex.org/wp-content/uploads/2021/05/fsi-2021.xlsx", "b21a891071f48ee00746b25c6b5c2ebbe08323b46a9daf3b8935243f0e2325d4"),
    2022: ("https://fragilestatesindex.org/wp-content/uploads/2022/07/fsi-2022-download.xlsx", "fdbecfb98bb15478905e4fa2e1be1b552e95c914b57f0ff983134fca3413fe0a"),
    2023: ("https://fragilestatesindex.org/wp-content/uploads/2023/06/FSI-2023-DOWNLOAD.xlsx", "76adfd95695ad8da1eb6233dc14e06ba2c1f55080f604b5310f2acd52ffc1d2d"),
}
REPORT_2024 = "https://fragilestatesindex.org/wp-content/uploads/2025/02/FSI-2024-Report-A-World-Adrift.pdf"
INDICATORS = [
    "C1: Security Apparatus", "C2: Factionalized Elites", "C3: Group Grievance",
    "E1: Economy", "E2: Economic Inequality", "E3: Human Flight and Brain Drain",
    "P1: State Legitimacy", "P2: Public Services", "P3: Human Rights",
    "S1: Demographic Pressures", "S2: Refugees and IDPs", "X1: External Intervention",
]
COLUMNS = ["Country", "Year", "Rank", "Total", *INDICATORS, "source_url", "source_detail"]


def source_file(year):
    return LEGACY / f"fsi-{year}.xlsx" if year <= 2020 else RAW / f"fsi-{year}.xlsx"


def download_missing():
    """Fetch only missing source workbooks; reject unexpected content."""
    RAW.mkdir(parents=True, exist_ok=True)
    for year, (url, digest) in SOURCES.items():
        path = source_file(year)
        if path.exists():
            continue
        with urlopen(url, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Unexpected checksum for {year}; inspect publisher revision")
        path.write_bytes(data)


def build():
    frames = []
    for year in range(2006, 2024):
        path = source_file(year)
        if year in SOURCES:
            url, digest = SOURCES[year]
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Changed publisher workbook: {path}")
        else:
            url = f"https://fragilestatesindex.org/excel/ (archived in MainCode/fsi-{year}.xlsx)"
        df = pd.read_excel(path)
        missing = set(COLUMNS[:16]) - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        df = df[COLUMNS[:16]].copy()
        df["Year"] = year  # Published file is authoritative for its own annual year.
        df["Country"] = df["Country"].astype("string").str.strip()
        df["Rank"] = pd.to_numeric(df["Rank"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
        for col in ["Total", *INDICATORS]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # "n/r" is a legitimate published rank (e.g. South Sudan in 2012),
        # so preserve the score and leave its numeric Rank blank.
        required = ["Country", "Total", *INDICATORS]
        if df[required].isna().any().any():
            bad_rows = df.index[df[required].isna().any(axis=1)].tolist()
            raise ValueError(f"{year}: missing scores or keys in rows {bad_rows[:10]}")
        if df["Country"].duplicated().any() or not df["Total"].between(0, 120).all():
            raise ValueError(f"{year}: invalid country keys or Total scores")
        if not df[INDICATORS].apply(lambda scores: scores.between(0, 10)).all().all():
            raise ValueError(f"{year}: indicator outside 0–10")
        if year >= 2021 and len(df) != 179:
            raise ValueError(f"{year}: expected 179 publisher rows, found {len(df)}")
        df["source_url"] = url
        df["source_detail"] = "publisher_workbook" if year in SOURCES else "thesis_archived_workbook"
        frames.append(df)

    # Report page 7 (printed page 6): "72.3 - India (75th)". No indicator
    # values have been inferred from the rasterized 2024 indicator tables.
    frames.append(pd.DataFrame([{
        "Country": "India", "Year": 2024, "Rank": 75, "Total": 72.3,
        "source_url": REPORT_2024, "source_detail": "report_page_7_total_only",
    }]))
    panel = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)
    if panel.duplicated(["Country", "Year"]).any():
        raise ValueError("Duplicate country/year observations")
    panel = panel.sort_values(["Country", "Year"], kind="stable")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {OUT}: {len(panel)} rows, 2006–2023 full; India 2024 Total only")
    print(panel[panel.Country.eq("India")][["Year", "Total", "Rank", "source_detail"]].tail(5).to_string(index=False))
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-missing", action="store_true", help="fetch absent 2021–2023 official files")
    args = parser.parse_args()
    if args.download_missing:
        download_missing()
    build()
