# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **final thesis (tugas akhir)** project for forecasting SMS International service revenue. The goal is to predict monthly revenue per route/client combination (Series_ID) for Jan–Mar 2025, using a two-stage ML pipeline optimized with a Genetic Algorithm.

The thesis write-up (LaTeX, ITS format) lives in `Tugas_Akhir_Jiryan_Farokhi/tugas-akhir/`. Compile entry-point: `compile-here/skripsi-main.tex`.

## Environment Setup

```bash
# Activate virtualenv
source .venv/bin/activate

# Install dependencies
pip install -r artifacts/requirements.txt
# Key packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, xgboost, lightgbm, pygad

# Run notebook
jupyter notebook main_experiment.ipynb
```

## Data

- **Raw input**: `data/SMS International.csv` — transactional SMS data with columns: `Client`, `Route`, `SupplierConnection`, `Network`, `OAProfiledCategory`, `TotalRevenue`, `TotalCost`, `TotalProfit`, `TotalSentMessages`, `TotalDeliveredMessages`, `Year`, `Month`, etc.
- **Masked version**: `data/SMS_International_masked.csv` — anonymized with hashed client/supplier/route names (written by cell 10)
- **Pre-encoded version**: `data/final.csv` — an earlier iteration of the cleaned data with label-encoded columns (`Client_enc`, `Route_enc`, etc.); not used by the current notebook pipeline
- **Intermediate**: `artifacts/monthly_panel.csv` — the fully-featured monthly panel; can be loaded directly to skip data cleaning steps

The `Series_ID` key is constructed as: `Client_Route_SupplierConnection_Network`

> **Artifact path note**: Several cells save to the working directory root (e.g., cell 64 saves `monthly_panel.csv` to root, not `artifacts/`). The `artifacts/` copies are the canonical versions kept for reference.

## Architecture: Two-Stage Forecasting Pipeline

All experiment logic lives in `main_experiment.ipynb` (78 cells). The pipeline flows:

```
Raw CSV  (cell 2)
  → Data Cleaning & Masking (cells 2–25)
      └─ df_terurut: sorted, cleaned, currency columns dropped
  → df_model = df_terurut.copy() + minor cleanup (cells 43–46)
  → EDA & Visualization (cells 27–42)
  → Series_ID construction + Monthly Aggregation (cells 54–55)
      ├─ Series_ID = Client_Route_SupplierConnection_Network
      ├─ Aggregate revenue/cost/messages/anomaly flags per Series_ID × month
      └─ Complete panel: fill missing months with Revenue=0, is_active=0 (cell 53 markdown)
  → STEP 2 — Feature Engineering (cell 62)
      ├─ Lag features: rev_lag1/2/3, active_lag1/2/3
      ├─ Rolling features: rolling_mean_3/6, rolling_std_3/6
      ├─ Active streak, months_since_active, pct_active_3m/6m
      └─ Cyclical encoding: month_sin, month_cos
  → STEP 3 — Imputation & panel save (cells 63–64)
      └─ Fills lag/rolling NaN with 0; saves monthly_panel.csv
  → STEP 4 — Stage 1: Classification (cell 67)
      └─ XGBClassifier predicts `target_active` (will this series have revenue > 0?)
  → STEP 5 — Stage 2: Regression + GA Tuning (cell 68)
      ├─ Only runs on series predicted active by Stage 1
      ├─ Models: RandomForest, XGBoost, LightGBM
      ├─ PyGAD optimizes hyperparameters (fitness = −WMAPE on train)
      └─ Metrics: MAE, RMSE, R², WMAPE
  → STEP 5b–5d — Error analysis & visualization (cells 73–75)
  → STEP 6 — Future Forecast Jan–Mar 2025 (cell 76)
      └─ Output: forecast_jan_mar_2025.csv
  → STEP 6b — Sample series plot (cell 77)
```

> **Note**: Cell 66 contains a commented-out alternative Stage 1 approach (RandomForest + LightGBM classifier). The active implementation is cell 67.

**Time-based train/test split**: 80% train (earlier months), 20% test (last 5 months). Never random split — preserves temporal ordering.

## Key Design Decisions

- **Panel completion**: Every Series_ID gets rows for ALL months (not just months with transactions). Missing months get `Revenue=0, is_active=0`. This prevents lag features from skipping inactive months.
- **Two-stage approach**: Stage 1 classifies active/inactive first to avoid predicting near-zero revenue for dormant series; Stage 2 regresses only on active-predicted rows.
- **GA tuning**: PyGAD (`pygad`) searches continuous hyperparameter spaces for RF and XGBoost. Fitness function is negative WMAPE. Results compared against untuned baselines.
- **Anomaly flags**: `is_anomaly_rev_only`, `is_anomaly_cost_only`, `anomaly_count`, `anomaly_revenue_pct` are engineered during aggregation to capture irregular transaction patterns.
- **WMAPE** (Weighted Mean Absolute Percentage Error) is the primary metric: `sum(|actual - pred|) / sum(|actual|)`.

## Feature Sets

**FEATURES_STAGE1** (classification): `active_lag1/2/3`, `active_streak`, `months_since_active`, `pct_active_3m/6m`, `rev_lag1/2/3`, `rolling_mean_3/6`, `TotalSentMessages`, `TotalDeliveredMessages`, `delivery_rate`, `unique_OA_count`, `anomaly_count`, `anomaly_revenue_pct`, `month_sin`, `month_cos`

**FEATURES_STAGE2** (regression): all Stage 1 features plus `rolling_std_3/6`, `profit_margin`, `rateCost`, `max_OA_revenue`

## Artifacts

All outputs are saved to `artifacts/`:
- `monthly_panel.csv` — processed panel (reusable checkpoint)
- `forecast_jan_mar_2025.csv` — final 3-month forecast per Series_ID
- PNG charts: `tren_revenue/cost/sent_messages/mom_growth`, `distribusi_revenue_series`, `konvergensi_ga`, `scatter_actual_predicted`, `boxplot_error_models`, `relative_absolute_error`, `overfit_check`, `forecast_jan_mar_2025`, `forecast_sample_series`
