"""Regression checks for actual Prophet projections and dated report artifacts."""

import csv
import unittest
from unittest.mock import patch

import pandas as pd

try:
    import prophet_report as analysis
except ImportError:  # The dataset tests also run in Python without Prophet installed.
    analysis = None


@unittest.skipIf(analysis is None, "install prophet==1.4.0 for forecasting tests")
class ProphetReportTests(unittest.TestCase):
    def test_indicator_history_excludes_unobserved_2024(self):
        panel = pd.read_csv(analysis.DATA)
        total = analysis.history(panel, "Total")
        c1 = analysis.history(panel, "C1: Security Apparatus")
        self.assertEqual((total.ds.dt.year.iloc[-1], len(total)), (2024, 19))
        self.assertEqual((c1.ds.dt.year.iloc[-1], len(c1)), (2023, 18))
        self.assertEqual(float(total.y.iloc[-1]), 72.3)

    def test_annual_model_has_no_unidentifiable_seasonality(self):
        panel = pd.read_csv(analysis.DATA)
        india = analysis.history(panel, "Total")
        predicted = analysis.forecast_rows(india, 2, analysis.seed_for("Total", 2025))
        self.assertEqual(predicted.ds.dt.year.tolist(), [2025, 2026])
        self.assertEqual(len(predicted), 2)
        self.assertTrue(predicted[["yhat", "yhat_lower", "yhat_upper"]].notna().all().all())

    def test_expanding_origin_never_trains_on_target_year(self):
        panel = pd.read_csv(analysis.DATA)
        seen = []

        def predictable_future(train, horizon, seed):
            last = int(train.ds.dt.year.iloc[-1])
            seen.append((last, horizon))
            return pd.DataFrame({
                "ds": pd.to_datetime([f"{year}-01-01" for year in range(last + 1, last + horizon + 1)]),
                "yhat": [float(train.y.iloc[-1])] * horizon,
                "yhat_lower": [0.0] * horizon,
                "yhat_upper": [120.0] * horizon,
            })

        with patch.object(analysis, "forecast_rows", side_effect=predictable_future):
            projections, folds, metrics = analysis.calculate(panel)
        self.assertEqual((len(projections), len(folds), len(metrics)), (77, 39, 13))
        self.assertTrue(all((row["training_end"], 1) in seen for row in folds))
        self.assertEqual([row["target_year"] for row in folds if row["series"] == "Total"], [2022, 2023, 2024])
        self.assertEqual([row["target_year"] for row in folds if row["series"] == "C1: Security Apparatus"], [2021, 2022, 2023])
        self.assertTrue(all(row["training_end"] == row["target_year"] - 1 for row in folds))

    def test_saved_report_matches_inputs_and_backtest_counts(self):
        output = analysis.RESULTS
        with (output / "prophet_forecast.csv").open(newline="") as stream:
            forecasts = list(csv.DictReader(stream))
        with (output / "prophet_backtest.csv").open(newline="") as stream:
            backtests = list(csv.DictReader(stream))
        self.assertEqual((len(forecasts), len(backtests)), (77, 39))
        total = [row for row in forecasts if row["series"] == "Total"]
        self.assertEqual([int(row["year"]) for row in total], list(range(2025, 2030)))
        self.assertTrue(all(row["status"] == "unobserved_as_of_snapshot" for row in total[:2]))
        self.assertTrue(all(row["status"] == "future_projection" for row in total[2:]))
        self.assertTrue(all(int(row["training_end"]) < int(row["target_year"]) for row in backtests))
        self.assertTrue(all(row["within_nominal_80"] in ("True", "False") for row in backtests))
        text = (output / "prophet_report.md").read_text()
        self.assertIn("India Total 72.3", text)
        self.assertIn(f"| 2026 | unobserved_as_of_snapshot | {float(total[1]['prophet']):.3f}", text)


if __name__ == "__main__":
    unittest.main()
