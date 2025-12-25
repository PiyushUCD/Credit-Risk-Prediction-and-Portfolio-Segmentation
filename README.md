# Credit Risk Prediction & Portfolio Segmentation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](project/LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b.svg)](project/app.py)

This project builds an end-to-end **Probability of Default (PD)** model and converts model scores into **portfolio risk bands** (Very Low → Extreme) to support practical credit decisions like:

- **Modeling:** learn a PD score from historical loan performance (default vs non-default)
- **Evaluation:** validate on a **holdout test split** to avoid overly optimistic results
- **Portfolio analytics:** score the **full 50,000-loan run** and segment loans into risk bands
- **Delivery:** export a report, CSV outputs, and plots that are easy to share with stakeholders
- **App layer:** optional **Streamlit** app to score new CSV files using the trained pipeline

**Why this is useful:** PD is not the final answer—teams use it to drive actions like approval thresholds, pricing tiers, exposure limits, and portfolio monitoring. This project demonstrates how to go from a PD model to an interpretable segmentation that can be communicated to both technical and non-technical audiences.


> This project runs on **50,000 loans** by default and generates the visuals from that run.

---

## Table of contents

## Table of contents
- [Background](#background)
- [Project](#project)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Model performance](#model-performance)
- [Deliverables](#deliverables)
- [Results (50,000-loan run)](#results-50000-loan-run)
- [Getting started](#getting-started)
- [Technologies](#technologies)
- [Top-directory layout](#top-directory-layout)
- [License](#license)
- [Author](#author)


---

## Background

**Credit risk modelling** estimates the likelihood that a borrower will **default** on a loan. A common approach is to train a supervised model that outputs a **Probability of Default (PD)** for each loan. PD is especially valuable because it provides a continuous risk score, not just a yes/no decision.

In real lending workflows, PD is usually combined with business rules to answer questions like:

- **Approval / decline:** at what PD threshold should we reject applications?
- **Pricing:** how should interest rates or risk premiums change with PD?
- **Exposure management:** how much portfolio concentration is acceptable in higher-risk segments?
- **Portfolio monitoring:** is the risk mix shifting over time (e.g., more “high risk” loans this quarter)?
- **Collections prioritization:** which accounts should receive attention first?

However, PD alone is often hard to communicate. That’s why many institutions convert PD scores into **risk bands** (e.g., Very Low → Extreme) and track portfolio share and performance metrics per band. This project follows that exact pattern: train a PD model, evaluate it properly, then translate it into a segmentation that is easy to interpret and act on.


---

## Project

This project builds a baseline **Probability of Default (PD)** model using **leakage-safe preprocessing** (so we don’t accidentally train on information that would not be available at decision time). After training, the pipeline **scores the full portfolio** and converts probabilities into **risk bands** that summarize portfolio quality and help guide decisions.

**Objectives**

1. **Predict PD:** estimate default probability for each loan using a clean ML pipeline.
2. **Validate properly:** evaluate on a **holdout split** (test set) to estimate real-world performance.
3. **Segment the portfolio:** score the **full 50k run** and create risk bands to understand the portfolio mix.
4. **Produce stakeholder-ready outputs:** plots + CSV tables + a markdown report that reads well on GitHub.

**What you get:**

- **Trained PD model** saved as a scikit-learn Pipeline (preprocessing + model together)
- **Holdout evaluation** artifacts (metrics + ROC curve + confusion matrix)
- **Portfolio segmentation** on the **full 50k loans**:
  - portfolio share by band
  - average predicted PD per band
  - observed default rate per band (sanity check / monitoring view)
- **Streamlit app (optional)** to score new CSV files and export results

**How to interpret the outputs**

- **Model plots** (ROC + confusion matrix) answer: *“Can the model separate defaults from non-defaults?”*
- **Segmentation outputs** answer: *“How is risk distributed across the portfolio, and does higher PD correspond to higher observed defaults?”*
- The combination makes the project practical: it shows both **model skill** and **portfolio-level business insight**.

---

## Pipeline

1. Load LendingClub data (50k sample)
2. Clean + feature selection
3. Train/test split
4. Preprocess (impute + one-hot + scale)
5. Train models
6. Evaluate on holdout
7. Score full portfolio (50k)
8. Risk bands + portfolio summary

---

## Dataset

- Source: LendingClub public loan data (downloaded via `kagglehub`)
- File used: `accepted_2007_to_2018Q4.csv`
- Run size: **50,000 rows** (controlled by `--sample-size 50000`)

---

## Model performance

Model performance is reported on a **holdout test split** (by default 20% of the loaded data), so the results are not overly optimistic.

Generated artifacts:

- Metrics: `project/results/model_metrics.csv`
- Confusion matrix: `project/results/03_confusion_matrix.png`
- ROC curves: `project/results/02_roc_curves.png`

---

## Deliverables

After a run, you will have:

- `project/results/REPORT.md` (GitHub-friendly report)
- `project/results/model_metrics.csv`
- `project/results/portfolio_analysis.csv`
- Plots (PNG) saved into `project/results/`

To keep the README plots up to date, copy the latest plot images:

```bash
python project/scripts/sync_assets.py --results-dir project/results --assets-dir project/assets/plots
```

---

## Results (50,000-loan run)


**Model comparison**  
![Model Performance](project/assets/plots/01_model_performance_comparison.png)

**ROC curves**  
![ROC](project/assets/plots/02_roc_curves.png)

**Confusion matrix (chosen threshold)**  
![Confusion Matrix](project/assets/plots/03_confusion_matrix.png)

**Feature importance (model-dependent)**  
![Feature Importance](project/assets/plots/04_feature_importance.png)

**Portfolio segmentation (risk bands)**  
![Portfolio Analysis](project/assets/plots/05_portfolio_analysis.png)

**PD distribution (full portfolio)**  
![PD Distribution](project/assets/plots/06_probability_distribution.png)

---

## Getting started

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Run training + evaluation + segmentation (50,000 loans)

```bash
python project/main.py --sample-size 50000
```

Open:

- `project/results/REPORT.md`
- `project/results/portfolio_analysis.csv`

### 3) Run Streamlit app (optional)

```bash
streamlit run project/app.py
```

---

## Technologies

- Python
- pandas, NumPy
- scikit-learn (Pipelines + ColumnTransformer)
- matplotlib
- Streamlit (optional)

---

## Top-directory layout

```text
.
├── README.md
├── requirements.txt
└── project/
    ├── app.py
    ├── main.py
    ├── credit_risk/
    ├── assets/
    ├── scripts/
    └── results/
```

---

## License

Distributed under the MIT License. See `project/LICENSE` for more information.

---

## Author

- Piyush Patil
- Dheeraj Chavan
