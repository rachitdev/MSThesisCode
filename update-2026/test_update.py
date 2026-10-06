"""Regression tests for the dated FSI dataset and TimesFM output contract."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_dataset = load_module("build_dataset")
forecast = load_module("forecast")


class DatasetTests(unittest.TestCase):
    def test_panel_provenance_and_partial_year(self):
        panel = build_dataset.build()
        self.assertFalse(panel.duplicated(["Country", "Year"]).any())
        self.assertEqual(sorted(panel.Year.unique()), list(range(2006, 2025)))
        self.assertEqual(len(panel[panel.Year.eq(2024)]), 1)
        india_2024 = panel.loc[panel.Year.eq(2024)].iloc[0]
        self.assertEqual((india_2024.Country, india_2024.Total, india_2024.Rank), ("India", 72.3, 75))
        self.assertTrue(india_2024[build_dataset.INDICATORS].isna().all())
        self.assertEqual(len(panel.loc[panel.Year.eq(2023)]), 179)
        # The publisher marked South Sudan 2012 "n/r": retain its Total
        # rather than silently removing this legitimate observation.
        unranked = panel.loc[panel.Year.eq(2012) & panel.Country.eq("South Sudan")].iloc[0]
        self.assertAlmostEqual(unranked.Total, 108.4)
        self.assertTrue(pd.isna(unranked.Rank))
        row = panel.loc[panel.Year.eq(2023) & panel.Country.eq("India")].iloc[0]
        self.assertAlmostEqual(row.Total, 74.1)
        self.assertAlmostEqual(row["C1: Security Apparatus"], 6.0)

    def test_contiguous_india_history(self):
        years, values = forecast.load_series(build_dataset.OUT)
        self.assertEqual(len(years), 19)
        self.assertEqual(float(values[-1]), float(np.float32(72.3)))

    def test_fails_on_observation_gap(self):
        panel = pd.read_csv(build_dataset.OUT)
        path = ROOT / "data" / "test_gap.csv"
        try:
            panel.loc[~(panel.Country.eq("India") & panel.Year.eq(2015))].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "uninterrupted"):
                forecast.load_series(path)
        finally:
            path.unlink(missing_ok=True)


class OutputTests(unittest.TestCase):
    def test_quantile_shape_bounds_and_finite_values(self):
        class FakeModel:
            def predict(self, context, horizon, return_quantiles):
                self.context = context
                self.return_quantiles = return_quantiles
                return type("Result", (), {
                    "forecast": np.array([-2.0, 125.0]),
                    "quantiles": np.tile(np.linspace(-1, 121, 9), (2, 1)),
                })()

        fake = FakeModel()
        point, quantiles = forecast.estimate(fake, np.arange(19, dtype=float), 2)
        np.testing.assert_array_equal(point, [0, 120])
        self.assertEqual(quantiles.shape, (2, 9))
        self.assertEqual(fake.context.dtype, np.float32)
        self.assertTrue(fake.return_quantiles)

    def test_bad_model_output_is_rejected(self):
        class BadModel:
            def predict(self, *args, **kwargs):
                return type("Result", (), {"forecast": [np.nan], "quantiles": [[0] * 9]})()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            forecast.estimate(BadModel(), np.ones(19), 1)


if __name__ == "__main__":
    unittest.main()
