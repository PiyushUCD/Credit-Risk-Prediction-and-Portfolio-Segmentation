# Changelog

## 1.3.1
- Portfolio segmentation and business visuals are now computed on the full loaded portfolio (e.g., 50,000 loans), while model metrics remain holdout-test evaluation.
- Improved plot alignment and readability for GitHub.
- Added scripts/sync_assets.py to sync generated plots into assets/plots for the README.
## 1.4.0 — 2025-12-25
- README upgraded into a portfolio-style case study (metrics + segmentation highlights).
- Added `assets/streamlit_demo.gif` for a quick visual walkthrough.
- Refreshed `docs/MODEL_CARD.md` with responsible use + clearer limitations.
- Updated GitHub badges/links to match `PiyushUCD/Credit-Risk-Prediction-and-Portfolio-Segmentation`.

## 1.3.0 — 2025-12-25
- Converted code into a proper package structure (`credit_risk/`).
- Added Streamlit demo app and explainability utilities.
- Added optional hyperparameter tuning and CI lint checks.
