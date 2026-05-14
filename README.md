# F1 Lap Time Forecasting

Predicting Formula 1 lap times from a driver's most recent ten laps for their next five laps using a Multilayer Perceptron and an LSTM, trained on three seasons (2024–2026) of race data from the FastF1 library.

## Headline finding

Both models converged to a similar test RMSE of ~7.5 seconds. The bottleneck is structural, not architectural: pit-stop timing dominates squared error and is a strategic team decision that past lap-time data does not encode. The more interesting result is **"data unpredictability dominates architectural sophistication"** — a more capable LSTM did not beat a simpler MLP.

See [report/report.md](report/report.md) for the full write-up.

## Project layout

- `F1_Lap_Time_Forecasting.ipynb` — full pipeline: data ingest, feature engineering, windowing, training, evaluation, plotting
- `report/report.md` — technical write-up (abstract / methods / results / conclusions)
- `report/figures/` — saved plots referenced by the report
- `results/` — per-run JSON metrics (one file per training run)
- `data/processed/laps.csv` — cached processed lap dataset (regenerated from FastF1 if absent)
- `data/fastf1_cache/` — FastF1 raw cache (not committed)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook F1_Lap_Time_Forecasting.ipynb
```

The first run will download race data from the FastF1 API (a few minutes, subject to FastF1 rate limits). Subsequent runs read from `data/processed/laps.csv`.

## Stack

PyTorch (MPS / CUDA / CPU auto-select), FastF1, pandas, scikit-learn, matplotlib, seaborn.
