# Credit Risk Prediction & Portfolio Segmentation

[![CI](https://github.com/PiyushUCD/Credit-Risk-Prediction-and-Portfolio-Segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/PiyushUCD/Credit-Risk-Prediction-and-Portfolio-Segmentation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b.svg)](app.py)

Train a **Probability of Default (PD)** model (leakage-safe preprocessing), then convert PDs into **portfolio risk bands** with segment-level analytics.
Includes a **Streamlit demo** that lets anyone upload a CSV and get scored outputs.

---

## Demo (what recruiters will see)

![Demo](assets/streamlit_demo.gif)

> Tip: run the app locally in ~60 seconds (instructions below).

---

## Portfolio case study (how to talk about this project)

**Problem**
- Predict default risk (PD) for loan applicants and transform model scores into actionable **portfolio segments**.

**Approach**
- Leakage-safe preprocessing using `Pipeline` + `ColumnTransformer` (imputation, one-hot encoding, scaling)
- Train multiple candidate models → pick best by validation metrics
- Evaluate with ROC/PR curves, confusion matrix, threshold tuning, PD distribution
- Segment the portfolio into risk bands and summarize **mix / observed default rate / expected-loss style proxy**

**Example results from the included run**
- Best model by ROC AUC: **Logistic Regression** (ROC AUC **0.700**, F1 **0.404**)
- Concentration: the top risk bands (**High + Very High + Extreme**) represent **89.43%** of the portfolio and account for about **97.96%** of observed defaults in the sample output.

### Model metrics (sample run)

| Model | ROC AUC | F1 | N |
|---|---|---|---|
| Logistic Regression | 0.700 | 0.404 | 10000 |
| Random Forest | 0.680 | 0.072 | 10000 |


### Portfolio segmentation summary (sample run)

| Risk band | Loans | Portfolio % | Avg PD | Observed default rate |
|---|---|---|---|---|
| Very Low | 4 | 0.040 | 0.035 | 0.000 |
| Low | 46 | 0.460 | 0.123 | 0.000 |
| Medium | 1007 | 10.070 | 0.212 | 0.038 |
| High | 1959 | 19.590 | 0.301 | 0.085 |
| Very High | 3173 | 31.730 | 0.425 | 0.162 |
| Extreme | 3811 | 38.110 | 0.638 | 0.300 |


---

## Results preview (generated artifacts)

**Model comparison**  
![Model Performance](assets/plots/01_model_performance_comparison.png)

**ROC curves**  
![ROC](assets/plots/02_roc_curves.png)

**Feature importance (model-dependent)**  
![Feature Importance](assets/plots/feature_importance.png)

**PD distribution**  
![PD Distribution](assets/plots/06_probability_distribution.png)

**Confusion matrix (chosen threshold)**  
![Confusion Matrix](assets/plots/03_confusion_matrix.png)

**Portfolio segmentation**  
![Portfolio Analysis](assets/plots/05_portfolio_analysis.png)

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
python main.py
```

Useful flags:
```bash
python main.py --tune --tune-iter 30
python main.py --threshold cost
python main.py --no-calibrate
```

---

## Streamlit app

Install app extras:
```bash
pip install -r requirements.txt -r requirements-app.txt
```

Run:
```bash
streamlit run app.py
```

The app supports:
- Upload CSV → score PD → assign risk band → download scored dataset
- Optional explainability (global importance; SHAP if installed)

---

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
  streamlit_demo.gif
  plots/
data/
  sample_input.csv
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
