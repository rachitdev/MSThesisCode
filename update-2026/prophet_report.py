"""Run the thesis's Prophet method on the latest verified India FSI observations.

The original notebook fitted one Prophet model per Total/indicator and generated
five annual future rows. This addendum preserves that approach, but disables
unidentifiable within-year seasonality on one-observation-per-year data.
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from build_dataset import INDICATORS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "fsi_2006_2024.csv"
RESULTS = ROOT / "results"
AS_OF = "2026-10-06"
LAST_PROJECTION_YEAR = 2029
TARGETS = ["Total", *INDICATORS]


def history(panel, target):
    india = panel.loc[panel.Country.eq("India"), ["Year", target]].sort_values("Year")
    india = india.dropna(subset=[target])
    last = 2024 if target == "Total" else 2023
    years = india.Year.to_numpy(dtype=int)
    if not np.array_equal(years, np.arange(2006, last + 1)):
        raise ValueError(f"{target}: non-contiguous or missing India observations, 2006–{last}")
    if not india[target].between(0, 120 if target == "Total" else 10).all():
        raise ValueError(f"{target}: observed value outside published score range")
    return pd.DataFrame({
        "ds": pd.to_datetime(india.Year.astype(str) + "-01-01"),
        "y": india[target].astype(float),
    }).reset_index(drop=True)


def forecast_rows(observed, horizon, seed):
    """Return future-only Prophet rows, without in-sample values or leakage."""
    np.random.seed(seed)
    model = Prophet(
        yearly_seasonality=False, weekly_seasonality=False,
        daily_seasonality=False, interval_width=0.80,
    )
    model.fit(observed)
    future = model.make_future_dataframe(periods=horizon, freq="YS")
    predicted = model.predict(future).tail(horizon).reset_index(drop=True)
    expected = np.arange(observed.ds.dt.year.iloc[-1] + 1, observed.ds.dt.year.iloc[-1] + horizon + 1)
    if not np.array_equal(predicted.ds.dt.year.to_numpy(), expected):
        raise ValueError("Prophet returned unexpected annual dates")
    cols = ["yhat", "yhat_lower", "yhat_upper"]
    if not np.isfinite(predicted[cols].to_numpy()).all():
        raise ValueError("Prophet returned non-finite predictions")
    return predicted


def bound(row, maximum):
    low, value, high = (float(np.clip(row[col], 0, maximum)) for col in
                        ("yhat_lower", "yhat", "yhat_upper"))
    if not low <= high:
        raise ValueError("Prophet produced reversed forecast bounds")
    return round(value, 3), round(low, 3), round(high, 3)


def seed_for(target, year):
    return int.from_bytes(hashlib.sha256(f"{target}:{year}".encode()).digest()[:4], "big")


def calculate(panel):
    projections, backtests, metrics = [], [], []
    for target in TARGETS:
        observed = history(panel, target)
        last_observed = int(observed.ds.dt.year.iloc[-1])
        maximum = 120 if target == "Total" else 10
        # Three expanding-origin, genuine one-step tests per target. Never
        # include the target year's score in its training set.
        for target_year in range(last_observed - 2, last_observed + 1):
            train = observed.loc[observed.ds.dt.year.lt(target_year)].copy()
            result = forecast_rows(train, 1, seed_for(target, target_year)).iloc[0]
            point, lower, upper = bound(result, maximum)
            actual = float(observed.loc[observed.ds.dt.year.eq(target_year), "y"].iloc[0])
            baseline = float(train.y.iloc[-1])
            backtests.append({
                "series": target, "target_year": target_year,
                "training_end": target_year - 1, "training_points": len(train),
                "actual": round(actual, 3), "prophet": point,
                "prophet_lower_80": lower, "prophet_upper_80": upper,
                "baseline_last_value": round(baseline, 3),
                "prophet_abs_error": round(abs(actual - point), 3),
                "baseline_abs_error": round(abs(actual - baseline), 3),
                "within_nominal_80": lower <= actual <= upper,
            })
        horizon = LAST_PROJECTION_YEAR - last_observed
        prediction = forecast_rows(observed, horizon, seed_for(target, last_observed + 1))
        for row in prediction.itertuples(index=False):
            point, lower, upper = bound(row._asdict(), maximum)
            year = int(row.ds.year)
            projections.append({
                "series": target, "year": year, "last_observed": last_observed,
                "status": "unobserved_as_of_snapshot" if year <= 2026 else "future_projection",
                "prophet": point, "lower_80": lower, "upper_80": upper,
            })
        folds = [row for row in backtests if row["series"] == target]
        metrics.append({
            "series": target, "last_observed": last_observed,
            "n_backtests": len(folds),
            "prophet_mae": round(float(np.mean([row["prophet_abs_error"] for row in folds])), 3),
            "baseline_mae": round(float(np.mean([row["baseline_abs_error"] for row in folds])), 3),
            "nominal_80_coverage": round(sum(row["within_nominal_80"] for row in folds) / len(folds), 3),
        })
    return projections, backtests, metrics


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(projections, metrics):
    total = [row for row in projections if row["series"] == "Total"]
    indicator_years = (2024, 2025, 2026, 2029)
    index = {(row["series"], row["year"]): row for row in projections}
    lines = [
        "# MS thesis — updated India FSI projections (6 October 2026)", "",
        "**Exploratory research, not a political-violence prediction.** The submitted 2021 thesis remains unchanged. "
        "This report reruns its separate-per-series [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) "
        "trend forecasting approach against the latest verified observations; it does **not** use TimesFM.", "",
        "## Verified observations and provenance", "",
        "- Full publisher [annual FSI workbooks](https://fragilestatesindex.org/excel/) were available through **2023** "
        "on the checked official page. The thesis archive supplies 2006–2020; new pinned publisher workbooks supply 2021–2023.",
        "- The [2024 official annual report, PDF page 7](https://fragilestatesindex.org/wp-content/uploads/2025/02/FSI-2024-Report-A-World-Adrift.pdf) "
        "reports **India Total 72.3, rank 75**. Its indicator scores were not reliably extractable, so 2024 indicator values are **not** treated as observations.",
        "- Total is trained through 2024 (19 annual points); each of 12 indicator series is trained through 2023 "
        "(18 annual points). No publisher-verified 2025/2026 observations are included. A lower FSI score indicates lower measured fragility.", "",
        "## India Total (scale 0–120)", "",
        "| Year | Status | Prophet point | Nominal lower 80% | Nominal upper 80% |", "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in total:
        lines.append(f"| {row['year']} | {row['status']} | {row['prophet']:.3f} | {row['lower_80']:.3f} | {row['upper_80']:.3f} |")
    lines += [
        "", "2025–2026 are **unobserved estimates as of the snapshot date**, not confirmed outcomes; 2027–2029 are future projections.", "",
        "## India indicator projections (scale 0–10)", "",
        "The 2024 indicator column is an **estimate**, not the measured 2024 score. "
        "All 2024–2026 indicator entries are unobserved as of this report; the full yearly 2024–2029 outputs and nominal bounds are in `prophet_forecast.csv`.", "",
        "| Indicator | 2024 estimate | 2025 estimate | 2026 estimate | 2029 projection |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for target in INDICATORS:
        values = " | ".join(f"{index[(target, year)]['prophet']:.3f}" for year in indicator_years)
        lines.append(f"| {target} | {values} |")
    lines += [
        "", "The Total and indicators are modelled **independently** as in the thesis notebook. "
        "Do not add the indicator forecasts and expect them to match the separately forecast Total.", "",
        "## Three expanding-origin, one-year backtests per series", "",
        "| Series | Last observed | Prophet MAE | Last-value MAE | Nominal 80% coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        lines.append(f"| {metric['series']} | {metric['last_observed']} | {metric['prophet_mae']:.3f} | "
                     f"{metric['baseline_mae']:.3f} | {metric['nominal_80_coverage']:.0%} |")
    lines += [
        "", "Backtests use targets 2022–2024 for Total and 2021–2023 for indicators. "
        "Each training window ends *before* its target year; `prophet_backtest.csv` gives individual errors. "
        "Three folds per target are insufficient to establish interval calibration or generalize predictive accuracy. "
        "Compare Prophet MAE with the naive last-value baseline rather than assuming the more complex model wins.", "",
        "## Method, scope and limitations", "",
        "- Uses the thesis notebook's separate `Prophet` model for Total and each indicator, with annual year-start timestamps "
        "and `make_future_dataframe(..., freq='YS')`. The archived notebook used `Prophet()` defaults; here **yearly, weekly and daily "
        "seasonalities are disabled** because one observation per year cannot identify within-year seasonality. "
        "Leaving yearly seasonality on creates aliasing at January 1 across leap years and misleading swings.",
        "- Prophet 1.4.0, 80% model intervals, fixed random seeds per fit; forecast values and bounds are clipped "
        "to the published Total [0,120] and indicator [0,10] scales. These model intervals have **not** been calibrated. "
        "Annual sampling (18–19 points), country-definition changes and index-methodology changes limit reliability.",
        "- FSI is a state-fragility index, **not** a measured riot incidence or conflict-probability target. "
        "A projection is not a causal explanation, warning of imminent violence, policy recommendation, or verified future fact.",
        "- The TimesFM 3.0 experiment remains separate: its checkpoint load was blocked by cache-write policy, "
        "so these numerical results are from Prophet alone. The original thesis, clustering and Wikipedia riot-list "
        "notebooks have not been rerun or rewritten.", "",
        "## Reproduce", "",
        "From the repository root, create an isolated Python environment with `pandas`, `openpyxl`, "
        "`numpy`, and `prophet==1.4.0`; then run:", "",
        "```bash", "python3 update-2026/build_dataset.py", "python3 update-2026/prophet_report.py",
        "python3 -m unittest discover -s update-2026 -p 'test_*.py' -v", "```", "",
        "The reproducible CSV outputs are `prophet_forecast.csv`, `prophet_backtest.csv`, and `prophet_metrics.csv` "
        "in this directory. Results are from the snapshot data and do not imply a newly published FSI 2025/2026 report.", "",
    ]
    return "\n".join(lines)


def run(data, output):
    panel = pd.read_csv(data)
    projections, backtests, metrics = calculate(panel)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "prophet_forecast.csv", projections)
    write_csv(output / "prophet_backtest.csv", backtests)
    write_csv(output / "prophet_metrics.csv", metrics)
    (output / "prophet_report.md").write_text(report(projections, metrics))
    metadata = {"as_of": AS_OF, "method": "Prophet 1.4.0 (annual, no seasonality)",
                "source_data": str(data.relative_to(ROOT)) if data.is_relative_to(ROOT) else str(data),
                "series": len(TARGETS), "forecast_rows": len(projections),
                "backtest_rows": len(backtests), "last_projection_year": LAST_PROJECTION_YEAR}
    (output / "prophet_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("India Total projections:")
    for row in projections:
        if row["series"] == "Total":
            print(row)
    return projections, backtests, metrics


if __name__ == "__main__":
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    run(args.data, args.output)
