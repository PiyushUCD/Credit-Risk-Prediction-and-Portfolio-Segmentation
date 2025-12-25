# Model Card — Credit Risk Probability of Default (PD)

## Intended use
This model estimates **Probability of Default (PD)** for loan/credit applicants and supports **portfolio segmentation** into risk bands.
It is designed for:
- credit risk scoring prototypes and analytics
- scenario / portfolio monitoring demos
- education and portfolio projects

It is **not** production-approved credit decisioning.

## Model overview
- **Task:** binary classification (default vs non-default)
- **Outputs:** calibrated PD (0–1), predicted class at chosen threshold, and assigned risk band
- **Algorithms:** logistic regression / tree models (configurable)
- **Preprocessing:** leakage-safe `Pipeline` + `ColumnTransformer` (imputation, one-hot encoding, scaling)

## Training & evaluation
The training pipeline:
1. Split data into train/test (or CV)
2. Fit preprocessing + model inside a single sklearn pipeline
3. Evaluate with ROC AUC, PR AUC, F1, confusion matrix, and threshold optimization
4. Optionally calibrate probabilities (improves PD interpretability)

Artifacts are saved under `results/` (plots, metrics) and `models/` (trained pipeline).

## Data assumptions
- Tabular loan/customer features
- Contains a **binary target** column (default indicator)
- Feature schema may contain both numeric and categorical columns

See `DATA.md` for expected schema guidelines and examples.

## Limitations
- Performance depends heavily on data quality and representativeness.
- PD calibration may degrade if the population shifts (dataset drift).
- A single threshold is rarely optimal for all business objectives; cost-based thresholds are preferred.

## Fairness & responsible use
Credit models can encode historical bias. Before real-world use, consider:
- protected class correlation checks
- disparate impact / equalized odds metrics
- feature review (exclude obviously sensitive attributes)
- human-in-the-loop decision policies and documentation

## Reproducibility
- Primary entrypoints: `python main.py` (pipeline), `streamlit run app.py` (demo)
- Dependencies: `requirements.txt` (+ `requirements-app.txt` for demo)
