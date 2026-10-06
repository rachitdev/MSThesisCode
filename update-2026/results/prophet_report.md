# MS thesis — updated India FSI projections (6 October 2026)

**Exploratory research, not a political-violence prediction.** The submitted 2021 thesis remains unchanged. This report reruns its separate-per-series [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) trend forecasting approach against the latest verified observations; it does **not** use TimesFM.

## Verified observations and provenance

- Full publisher [annual FSI workbooks](https://fragilestatesindex.org/excel/) were available through **2023** on the checked official page. The thesis archive supplies 2006–2020; new pinned publisher workbooks supply 2021–2023.
- The [2024 official annual report, PDF page 7](https://fragilestatesindex.org/wp-content/uploads/2025/02/FSI-2024-Report-A-World-Adrift.pdf) reports **India Total 72.3, rank 75**. Its indicator scores were not reliably extractable, so 2024 indicator values are **not** treated as observations.
- Total is trained through 2024 (19 annual points); each of 12 indicator series is trained through 2023 (18 annual points). No publisher-verified 2025/2026 observations are included. A lower FSI score indicates lower measured fragility.

## India Total (scale 0–120)

| Year | Status | Prophet point | Nominal lower 80% | Nominal upper 80% |
| --- | --- | ---: | ---: | ---: |
| 2025 | unobserved_as_of_snapshot | 73.278 | 71.765 | 74.948 |
| 2026 | unobserved_as_of_snapshot | 72.786 | 71.086 | 74.508 |
| 2027 | future_projection | 72.294 | 70.417 | 74.045 |
| 2028 | future_projection | 71.802 | 69.763 | 73.615 |
| 2029 | future_projection | 71.308 | 69.110 | 73.631 |

2025–2026 are **unobserved estimates as of the snapshot date**, not confirmed outcomes; 2027–2029 are future projections.

## India indicator projections (scale 0–10)

The 2024 indicator column is an **estimate**, not the measured 2024 score. All 2024–2026 indicator entries are unobserved as of this report; the full yearly 2024–2029 outputs and nominal bounds are in `prophet_forecast.csv`.

| Indicator | 2024 estimate | 2025 estimate | 2026 estimate | 2029 projection |
| --- | ---: | ---: | ---: | ---: |
| C1: Security Apparatus | 7.256 | 7.291 | 7.325 | 7.427 |
| C2: Factionalized Elites | 7.395 | 7.417 | 7.439 | 7.507 |
| C3: Group Grievance | 8.268 | 8.274 | 8.280 | 8.299 |
| E1: Economy | 6.258 | 6.344 | 6.431 | 6.689 |
| E2: Economic Inequality | 5.442 | 5.211 | 4.980 | 4.286 |
| E3: Human Flight and Brain Drain | 5.096 | 4.999 | 4.902 | 4.612 |
| P1: State Legitimacy | 4.261 | 4.195 | 4.129 | 3.931 |
| P2: Public Services | 7.374 | 7.405 | 7.436 | 7.530 |
| P3: Human Rights | 7.212 | 7.318 | 7.423 | 7.740 |
| S1: Demographic Pressures | 8.266 | 8.315 | 8.365 | 8.513 |
| S2: Refugees and IDPs | 4.901 | 4.936 | 4.970 | 5.072 |
| X1: External Intervention | 4.851 | 4.857 | 4.863 | 4.880 |

The Total and indicators are modelled **independently** as in the thesis notebook. Do not add the indicator forecasts and expect them to match the separately forecast Total.

## Three expanding-origin, one-year backtests per series

| Series | Last observed | Prophet MAE | Last-value MAE | Nominal 80% coverage |
| --- | ---: | ---: | ---: | ---: |
| Total | 2024 | 1.399 | 1.567 | 67% |
| C1: Security Apparatus | 2023 | 1.490 | 0.302 | 0% |
| C2: Factionalized Elites | 2023 | 0.142 | 0.000 | 67% |
| C3: Group Grievance | 2023 | 0.320 | 0.133 | 33% |
| E1: Economy | 2023 | 0.671 | 0.733 | 33% |
| E2: Economic Inequality | 2023 | 0.319 | 0.167 | 33% |
| E3: Human Flight and Brain Drain | 2023 | 0.195 | 0.300 | 100% |
| P1: State Legitimacy | 2023 | 0.141 | 0.167 | 100% |
| P2: Public Services | 2023 | 0.258 | 0.467 | 100% |
| P3: Human Rights | 2023 | 0.811 | 0.167 | 33% |
| S1: Demographic Pressures | 2023 | 0.391 | 0.233 | 33% |
| S2: Refugees and IDPs | 2023 | 1.340 | 0.300 | 33% |
| X1: External Intervention | 2023 | 1.089 | 0.300 | 0% |

Backtests use targets 2022–2024 for Total and 2021–2023 for indicators. Each training window ends *before* its target year; `prophet_backtest.csv` gives individual errors. Three folds per target are insufficient to establish interval calibration or generalize predictive accuracy. Compare Prophet MAE with the naive last-value baseline rather than assuming the more complex model wins.

## Method, scope and limitations

- Uses the thesis notebook's separate `Prophet` model for Total and each indicator, with annual year-start timestamps and `make_future_dataframe(..., freq='YS')`. The archived notebook used `Prophet()` defaults; here **yearly, weekly and daily seasonalities are disabled** because one observation per year cannot identify within-year seasonality. Leaving yearly seasonality on creates aliasing at January 1 across leap years and misleading swings.
- Prophet 1.4.0, 80% model intervals, fixed random seeds per fit; forecast values and bounds are clipped to the published Total [0,120] and indicator [0,10] scales. These model intervals have **not** been calibrated. Annual sampling (18–19 points), country-definition changes and index-methodology changes limit reliability.
- FSI is a state-fragility index, **not** a measured riot incidence or conflict-probability target. A projection is not a causal explanation, warning of imminent violence, policy recommendation, or verified future fact.
- The TimesFM 3.0 experiment remains separate: its checkpoint load was blocked by cache-write policy, so these numerical results are from Prophet alone. The original thesis, clustering and Wikipedia riot-list notebooks have not been rerun or rewritten.

## Reproduce

From the repository root, create an isolated Python environment with `pandas`, `openpyxl`, `numpy`, and `prophet==1.4.0`; then run:

```bash
python3 update-2026/build_dataset.py
python3 update-2026/prophet_report.py
python3 -m unittest discover -s update-2026 -p 'test_*.py' -v
```

The reproducible CSV outputs are `prophet_forecast.csv`, `prophet_backtest.csv`, and `prophet_metrics.csv` in this directory. Results are from the snapshot data and do not imply a newly published FSI 2025/2026 report.
