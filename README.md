# Credit Risk Prediction & Portfolio Segmentation


> **Note:** Portfolio segmentation and portfolio-level plots are generated on the full dataset (default **N=50,000**). Model scores (ROC/F1/confusion matrix) are computed on a hold-out test split.

[![CI](https://github.com/PiyushUCD/Credit-Risk-Prediction-and-Portfolio-Segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/PiyushUCD/Credit-Risk-Prediction-and-Portfolio-Segmentation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b.svg)](app.py)

<!-- After you deploy on Streamlit Community Cloud, replace the link below with your app URL -->
[![Deploy](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

Train a **Probability of Default (PD)** model (leakage-safe preprocessing), then convert PDs into **portfolio risk bands** with segment-level analytics.
Includes a **Streamlit demo** that lets anyone upload a CSV and get scored outputs.

---

## Demo (what recruiters will see)


> Tip: run the app locally in ~60 seconds (instructions below).

---

## Portfolio case study (how to talk about this project)

**Problem**
- Predict default risk (PD) for loan applicants and transform model scores into actionable **portfolio segments**.

**Approach**
- Leakage-safe preprocessing using `Pipeline` + `ColumnTransformer` (imputation, one-hot encoding, scaling)
- Train multiple candidate models → pick best by validation metrics
- Segment the portfolio into risk bands and summarize **mix / observed default rate / expected-loss style proxy**

**Example results (from a 50,000-loan run)**
- Strong rank-ordering (ROC AUC around **0.70** is typical for this baseline feature set)
- Portfolio segmentation highlights concentration in higher-risk bands, helping you prioritize reviews and monitoring

> To reproduce the results on **50,000 loans**, run `python project/main.py --sample-size 50000` and open `project/results/REPORT.md`.

### Model metrics (holdout evaluation)

Metrics are computed on a **held‑out test split** (default: 20% of the loaded data) to avoid optimistic results.
The project reports **ROC‑AUC** and a simple **defaults‑captured@top‑10%** lift-style metric (business-friendly).

- Test metrics CSV: `project/results/model_metrics.csv`
- Best model pipeline: `models/best_credit_risk_model.pkl`

### Portfolio segmentation summary (full portfolio)

Portfolio segmentation is computed on **the full loaded portfolio** (e.g., **50,000 loans** when `--sample-size 50000`).
This is what drives the business-facing visuals and the risk-band summary.

- Portfolio table CSV: `project/results/portfolio_analysis.csv`
- Loan-level output (if enabled): `project/results/portfolio_loan_level.csv`

> Tip: After you run `python project/main.py --sample-size 50000`, open the CSVs above and the plots in `project/results/`.
> You can paste your latest tables back into this README if you want the repo to always display your most recent run.


## Results preview (generated artifacts)

**Model comparison**  
![Model Performance](project/assets/plots/01_model_performance_comparison.png)

**ROC curves**  
![ROC](project/assets/plots/02_roc_curves.png)


**Feature importance (model-dependent)**  
![Feature Importance](project/assets/plots/04_feature_importance.png)
**PD distribution**  
![PD Distribution](project/assets/plots/06_probability_distribution.png)

**Confusion matrix (chosen threshold)**  
![Confusion Matrix](project/assets/plots/03_confusion_matrix.png)

**Portfolio segmentation**  
![Portfolio Analysis](project/assets/plots/05_portfolio_analysis.png)

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Run training + evaluation + segmentation
```bash
python project/main.py --sample-size 50000
```

This run also writes a GitHub-friendly report:
- `project/results/REPORT.md`

Update the README plots (copies `project/results/*.png` → `project/assets/plots/`):
```bash
python scripts/sync_assets.py
# or: make sync-assets
```

Useful flags:
```bash
python project/main.py --tune --tune-iter 30
python project/main.py --threshold cost
python project/main.py --no-calibrate
```

---

## Streamlit app

The one-command install already includes Streamlit/ via `requirements.txt`.

Run:
```bash
streamlit run project/app.py
```

The app supports:
- Upload CSV → score PD → assign risk band → download scored dataset
- Optional explainability (global importance;  if installed)

### Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to Streamlit Community Cloud → **Create app**
3. Select your repo and set:
   - **Main file path**: `app.py`
4. Deploy 🎉

Notes:
- The Cloud builder installs from `requirements.txt` (this repo's file includes both core + app deps)
- If you want a lighter deploy, edit `requirements.txt` to remove 

See: `docs/DEPLOY_STREAMLIT.md`

## Project structure

```text
credit_risk/
  data_loader.py
  model_trainer.py
  model_evaluator.py
  portfolio_analyzer.py
  explainability.py
  visualizer.py
  helpers.py
assets/
  plots/
data/
  _input.csv
tests/
.github/
```

---

## Documentation
- `MODEL_CARD.md` — what the model is, how it was trained, limitations
- `ARCHITECTURE.md` — pipeline + design notes
- `DATA.md` — expected schema & assumptions

---

## License
MIT — see `LICENSE`.

## Repository layout

Only `README.md` and `requirements.txt` live at the repo root (clean GitHub view).  
All code, docs, and assets are inside the `project/` folder.

- Run pipeline: `python project/main.py --sample-size 50000`
- Run app: `streamlit run project/app.py`