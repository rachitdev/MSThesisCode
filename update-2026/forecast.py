"""Exploratory India FSI forecast with TimesFM 3.0 MLX (non-commercial only).

Annual observations are shorter than the recommended 32-point context; do not
interpret this experiment as validated conflict/riot prediction or policy advice.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = "google/timesfm-3.0-pytorch"


def load_series(path):
    frame = pd.read_csv(path)
    india = frame.loc[frame.Country.eq("India"), ["Year", "Total"]].sort_values("Year")
    if india.Year.duplicated().any() or india.Total.isna().any():
        raise ValueError("India Total must have one non-null observation per year")
    years = india.Year.astype(int).to_numpy()
    if not np.array_equal(years, np.arange(2006, 2025)):
        raise ValueError("Expected uninterrupted observed India totals, 2006–2024")
    return years, india.Total.to_numpy(dtype=np.float32)


def estimate(model, history, horizon):
    result = model.predict(history.astype(np.float32), horizon=horizon, return_quantiles=True)
    point = np.asarray(result.forecast, dtype=float)
    quantiles = np.asarray(result.quantiles, dtype=float)
    if point.shape != (horizon,) or quantiles.shape != (horizon, 9):
        raise ValueError(f"Unexpected TimesFM 3.0 output shapes: {point.shape}, {quantiles.shape}")
    if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
        raise ValueError("TimesFM returned non-finite forecasts")
    # The published score is bounded [0, 120]; this is an explicit postprocess,
    # not an assertion that model quantiles are calibrated prediction intervals.
    return np.clip(point, 0, 120), np.clip(quantiles, 0, 120)


def run(data, output, horizon):
    from timesfm3.mlx import TimesFM3Forecaster

    years, values = load_series(data)
    model = TimesFM3Forecaster.from_pretrained(MODEL)
    backtests = []
    for target_year in (2022, 2023, 2024):
        end = int(np.flatnonzero(years == target_year)[0])
        predicted, bands = estimate(model, values[:end], 1)
        actual = float(values[end])
        backtests.append({
            "year": target_year, "training_end": target_year - 1, "context_length": end,
            "actual": round(actual, 4), "timesfm_median": round(float(predicted[0]), 4),
            "timesfm_abs_error": round(abs(actual - float(predicted[0])), 4),
            "last_value_baseline": round(float(values[end-1]), 4),
            "baseline_abs_error": round(abs(actual - float(values[end-1])), 4),
            "q10": round(float(bands[0, 0]), 4), "q90": round(float(bands[0, 8]), 4),
        })
    predicted, bands = estimate(model, values, horizon)
    forecast = [{
        "year": int(years[-1] + index + 1), "median": round(float(predicted[index]), 4),
        "q10": round(float(bands[index, 0]), 4),
        "q90": round(float(bands[index, 8]), 4),
    } for index in range(horizon)]
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("backtest.csv", backtests), ("forecast.csv", forecast)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "model": MODEL, "backend": "timesfm3.mlx", "series": "India FSI Total",
        "last_observed_year": 2024, "last_observed_total": float(values[-1]),
        "observations": len(values), "context_warning": "19 annual points; fewer than recommended 32. Exploratory only.",
        "backtest_years": [row["year"] for row in backtests],
        "timesfm_mae": round(float(np.mean([row["timesfm_abs_error"] for row in backtests])), 4),
        "last_value_mae": round(float(np.mean([row["baseline_abs_error"] for row in backtests])), 4),
        "bounds": "q10/q90 model deciles, clipped to [0,120]; not calibrated intervals",
        "license": "TimesFM 3.0 weights: non-commercial, non-production only",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": summary, "forecast": forecast}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fsi_2006_2024.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 10:
        parser.error("horizon must be 1–10 annual steps")
    run(args.data, args.output, args.horizon)
