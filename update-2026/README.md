# 2026 research update — Fragile States Index and Prophet projections

**As-of date:** 6 October 2026. **Purpose:** personal, non-commercial research addendum to the 2021 MS thesis, *Predicting Political Instability across the World*. The submitted notebooks, thesis and report have **not** been rewritten; this directory is a separate reproducible update. A Fragile States Index (FSI) score measures multidimensional state fragility, **not** the probability, date, or location of riots or armed conflict. Higher scores indicate greater measured fragility.

## What is current and what is not

- The [Fund for Peace Excel download page](https://fragilestatesindex.org/excel/) lists full country/12-indicator workbooks **through 2023** (checked 6 October 2026). The original repository had workbooks through 2020. The newly archived 2021–2023 workbooks and SHA-256 pins are in `data/raw/` and `build_dataset.py`.
- The [publisher's 2024 annual report](https://fragilestatesindex.org/2025/02/18/fragile-states-index-2024-annual-report-a-world-adrift/) provides a newer observation. Its [report PDF, PDF page 7](https://fragilestatesindex.org/wp-content/uploads/2025/02/FSI-2024-Report-A-World-Adrift.pdf) reads **India: 72.3, rank 75**. Only this manually checked 2024 total/rank has been added. The indicator-by-country tables in that PDF are rasterized; **no 2024 indicator values or full global panel have been invented**. The panel has 3,170 full-score records from 2006–2023 and one partial 2024 India record. One historical country (South Sudan, 2012) has an officially unranked (`n/r`) score: its Total and indicators are retained while its numeric Rank is blank.
- No publisher-verified 2025 or 2026 FSI release was identified at the cited official download/report pages on this date. A projection for these years is **not** an observation.

`data/fsi_2006_2024.csv` contains each observed country/year's `Total`, rank, 12 indicators, source URL and source-detail label. Missing 2024 indicator values are blank. Historic 2006–2020 workbooks are read in place from `../MainCode/`; the published original files remain untouched. The 2023 workbook orders its indicator columns differently from prior years; the builder selects by **name**, not position. The label `thesis_archived_workbook` means the corresponding original workbook is already tracked in the thesis; for the fresh sources, the builder verifies byte checksums. Country names are retained exactly as published (no unreviewed cross-year aliases). For an individual-country history, check identity and continuity before forecasting.

## Reproduce the data

Use a Python environment with `pandas` and `openpyxl` (the default Python on this Mac has both):

```bash
python3 update-2026/build_dataset.py
python3 -m unittest discover -s update-2026 -p 'test_*.py' -v
```

For a fresh clone missing the tracked raw files, use `python3 update-2026/build_dataset.py --download-missing`. This uses only the pinned publisher URLs and rejects unexpected bytes, requiring an explicit review of a changed source before acceptance. If the publisher later releases 2024+ full Excel data, replace the manually sourced partial row **only after verifying the year, schema and provenance**, and update the data builder/tests. Do not silently extend a CSV with unverified internet tables.

## Executed Prophet projections and latest report

The submitted thesis notebook `MainCode/Time Series with Facebook's Prophet Library.ipynb` fitted separate Prophet models to India's Total and 12 indicator scores. [`prophet_report.py`](prophet_report.py) reruns that approach against the verified updated panel and **generates actual estimates**: [the dated India projections report](results/prophet_report.md), [all yearly estimates and nominal bounds](results/prophet_forecast.csv), [39 out-of-sample one-year predictions](results/prophet_backtest.csv), and [per-series accuracy summaries](results/prophet_metrics.csv). Total uses observations through **2024** and projections for 2025–2029. The indicator models use observations through **2023** and estimates for 2024–2029. The 2024 indicator estimates, and all 2025/2026 estimates, must **not** be confused with publisher observations. Three expanding-origin holdouts per series compare Prophet against a last-value baseline; several indicators perform **worse** than the baseline. Details, units, source links, limitations and model output are in the report.

Prophet 1.4.0 uses the original notebook's per-series fit and annual year-start future periods, but disables yearly/weekly/daily seasonality: at one observation per year, fitting within-year patterns would be unsupported and Prophet's default yearly setting produced a leap-year aliasing artifact. The notebook's results have not been overwritten. To reproduce from the repository root, install `prophet==1.4.0`, `pandas`, `numpy`, and `openpyxl` in an isolated Python environment, then run:

```bash
python3 update-2026/build_dataset.py
python3 update-2026/prophet_report.py
python3 -m unittest discover -s update-2026 -p 'test_*.py' -v
```

These projections concern the **FSI score**, not the occurrence or probability of riots, and are exploratory given the 18–19-year annual series. Prophet's nominal 80% bounds are not calibrated confidence intervals.

## TimesFM 3.0 research experiment

`forecast.py` implements an **India Total** (not riot incidence or twelve-indicator) forecast from the observed 2006–2024 annual series. It calls `timesfm3.mlx.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")`, uses float32 history, evaluates one-step historical holdouts in 2022–2024 against a last-value baseline, and requests five annual steps for **2025–2029**. Results, when run, go to `results/backtest.csv`, `results/forecast.csv`, and `results/summary.json`. It loads the model once, checks output shapes and finite values, and clips predictions/deciles to the publisher's 0–120 score scale. TimesFM's q10/q90 are **model deciles, not validated confidence bounds**; three holdouts do not establish calibration. Compare MAE to the baseline before interpreting a projection.

**Critical limitation:** there are only **19 annual India observations**, below the local skill's recommended 32-point context (holdout contexts are 16–18). We deliberately do not fabricate or upsample historical observations to meet that recommendation. This is an exploratory short-context evaluation; model performance may be poor, and it should not be used for public-safety, financial or governmental decisions.

The TimesFM 3.0 **pretrained weights are non-commercial and non-production only**. Do not use them for client projects, paid services or production. For such work use an appropriately licensed alternative. On the installed Apple Silicon machine, after a successful preflight and with permission to access the Hugging Face weight cache:

```bash
conda run -n timesfm python ~/.dsh/skills/timesfm-forecasting/scripts/check_system.py
conda run -n timesfm python update-2026/forecast.py
```

The initial model load downloads the checkpoint (roughly 1–2 GB). **Execution status on 6 October 2026:** preflight passed, but the attempted run was blocked when the environment denied writing `~/.cache/huggingface`; **there are no generated TimesFM results or TimesFM backtest metrics in this update**. The published estimates and measured backtests in `results/` were generated with **Prophet**, not TimesFM. Do not treat the TimesFM command or model integration as evidence that a TimesFM forecast was produced. No sandbox workaround or unverified forecast was used.

## Reproducibility and interpretation

| Source | Latest included observation | Granularity | Limitation |
| --- | --- | --- | --- |
| Original thesis workbooks | 2020 | Annual country Total + 12 indicators | Historical copy, not retroactively edited |
| Publisher Excel workbooks | 2023 | Annual country Total + 12 indicators | None published in Excel for 2024 on the checked page |
| Publisher 2024 report | 2024 | India Total/rank only | No transcribed indicator rows |
| Prophet output | 2025–2029 Total; 2024–2029 indicators | Executed India-only, separate per-series annual projections | Model estimates, not published observations; inspect backtest baseline |
| TimesFM output | Not generated in this environment | Proposed India Total forecast, 2025–2029 | First checkpoint load blocked by filesystem policy |

A score trend is an index projection, not a causal analysis or a political-instability warning system. The original riot-list extraction and clustering notebooks are historical artifacts and have not been re-run or claimed to be updated here. Data availability, name changes, different country counts in early years, index methodology, and the unusually short series constrain any comparison across years.
