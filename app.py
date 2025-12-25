"""Streamlit demo app.

Run:
    streamlit run app.py

What it does:
- Upload Lending Club CSV (raw) OR a feature-only CSV
- Load the trained model (models/best_credit_risk_model.pkl) OR train quickly in-app
- Predict probability of default (PD)
- Segment the portfolio into risk bands
- Show basic explainability (feature importance + optional SHAP for a single row)
"""

from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from explainability import (
    global_feature_importance,
    try_shap_local_explanation,
    unwrap_estimator,
)

from credit_risk.config import PROJECT_CONFIG
from credit_risk.data_loader import DataLoader
from credit_risk.model_trainer import ModelTrainer
from credit_risk.portfolio_analyzer import PortfolioAnalyzer

st.set_page_config(
    page_title="Credit Risk – PD Prediction & Segmentation", layout="wide"
)


def _load_model(models_dir: str) -> object | None:
    path = os.path.join(models_dir, "best_credit_risk_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _is_raw_lendingclub(df: pd.DataFrame, cfg: dict) -> bool:
    return str(cfg.get("TARGET_COLUMN", "loan_status")) in df.columns


def main() -> None:
    cfg = dict(PROJECT_CONFIG)

    st.set_page_config(
        page_title="Credit Risk Dashboard", page_icon="📊", layout="wide"
    )

    st.title("Credit Risk Default Prediction & Portfolio Segmentation")
    st.caption(
        "Upload a dataset → predict PD → segment into risk bands → export results."
    )

    with st.sidebar:
        st.header("Settings")
        models_dir = st.text_input(
            "Models directory", value=str(cfg.get("MODELS_DIR", "models"))
        )
        # Kept for future extension (e.g., saving scored files server-side). The app uses download buttons today.
        st.text_input(
            "Results output directory", value=str(cfg.get("OUTPUT_DIR", "results"))
        )
        st.divider()
        model_mode = st.radio(
            "Model source",
            options=["Load saved model", "Train quickly"],
            index=0,
        )
        max_train_rows = st.number_input(
            "Max rows to use for quick training",
            min_value=1000,
            max_value=200000,
            value=20000,
            step=1000,
        )
        enable_calibration = st.checkbox(
            "Calibrate probabilities (training)",
            value=bool(cfg.get("CALIBRATE_BEST_MODEL", True)),
        )
        st.divider()
        st.markdown("**Upload data**")
        uploaded = st.file_uploader(
            "CSV file", type=["csv"], accept_multiple_files=False
        )

    if uploaded is None:
        st.info(
            "Upload a CSV to begin. Tip: use the raw Lending Club 'accepted' file OR a feature-only CSV."
        )
        return

    df = pd.read_csv(uploaded, low_memory=False)
    st.write("### Preview")
    st.dataframe(df.head(20), use_container_width=True)

    loader = DataLoader(cfg)
    y_true = None

    if "loan_default" in df.columns:
        # Already a modeling frame
        model_df = df.copy()
        y_true = model_df["loan_default"].astype(int)
        feature_cols = [c for c in model_df.columns if c != "loan_default"]
    elif _is_raw_lendingclub(df, cfg):
        loaded = loader.build_model_frame(df)
        model_df = loaded.df
        feature_cols = loaded.feature_cols
        y_true = model_df[loaded.target_col].astype(int)
    else:
        # Feature-only file
        num_cols = list(cfg.get("NUMERICAL_FEATURES", []))
        cat_cols = list(cfg.get("CATEGORICAL_FEATURES", []))
        engineered = list(cfg.get("ENGINEERED_FEATURES", []))
        wanted = num_cols + cat_cols + engineered
        feature_cols = [c for c in wanted if c in df.columns]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            st.warning(
                "Your file does not look like the raw Lending Club dataset and is missing some configured features. "
                "Predictions will use the columns that exist.\n\n"
                f"Missing: {missing}"
            )
        model_df = df[feature_cols].copy()

    X = model_df[feature_cols].copy()

    # Model selection
    model = None
    if model_mode == "Load saved model":
        model = _load_model(models_dir)
        if model is None:
            st.warning(
                "No saved model found. Switch to 'Train quickly' or run `python main.py` to create a model."
            )
    else:
        # Train quickly on uploaded data (requires target)
        if y_true is None:
            st.error(
                "Quick training requires labels. Upload raw Lending Club data (with loan_status) or a dataset with loan_default."
            )
        else:
            cfg_train = dict(cfg)
            cfg_train["USE_CV"] = False
            cfg_train["CALIBRATE_BEST_MODEL"] = bool(enable_calibration)

            train_df = model_df.copy()
            if len(train_df) > int(max_train_rows):
                train_df = train_df.sample(
                    int(max_train_rows), random_state=int(cfg.get("RANDOM_STATE", 42))
                )

            trainer = ModelTrainer(cfg_train)
            train_results = trainer.train(
                train_df, feature_cols=feature_cols, target_col="loan_default"
            )
            model = train_results.best_model
            st.success(f"Trained model: {train_results.best_model_name}")

    if model is None:
        return

    # Predict
    probs = _predict_pd(model, X)
    out = df.copy()
    out["predicted_pd"] = probs

    st.write("### Predictions")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(out.head(50), use_container_width=True)
    with col2:
        st.metric("Rows scored", f"{len(out):,}")
        st.metric("Avg predicted PD", f"{float(np.mean(probs)):.3f}")
        st.metric("Median predicted PD", f"{float(np.median(probs)):.3f}")

        st.subheader("PD distribution")
        fig, ax = plt.subplots()
        ax.hist(probs, bins=30)
        ax.set_xlabel("Predicted PD")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    st.download_button(
        "Download scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="scored_portfolio.csv",
        mime="text/csv",
    )

    # Portfolio segmentation
    st.write("### Portfolio segmentation")
    analyzer = PortfolioAnalyzer(cfg)
    if y_true is None:
        # If no labels, segment with dummy 0s (metrics will be meaningless but bands are still useful)
        y_tmp = pd.Series(np.zeros(len(probs), dtype=int))
        port = analyzer.segment(probs, y_tmp)
        st.info(
            "No labels found, so 'actual_default_rate' is not meaningful. Risk bands are still computed from PD."
        )
    else:
        port = analyzer.segment(probs, y_true)

    st.dataframe(port.portfolio_table, use_container_width=True)

    st.subheader("Loans by risk band")
    fig2, ax2 = plt.subplots()
    tbl = port.portfolio_table.copy()
    if "loan_count" in tbl.columns:
        ax2.bar(tbl.index.astype(str), tbl["loan_count"].values)
        ax2.set_xlabel("Risk band")
        ax2.set_ylabel("Loan count")
        ax2.tick_params(axis="x", rotation=30)
        st.pyplot(fig2, use_container_width=True)

    # Explainability
    st.write("### Explainability")
    base = unwrap_estimator(model)
    if hasattr(base, "named_steps"):
        importance_df = global_feature_importance(base, top_k=30)
        if len(importance_df) == 0:
            st.info("This model type does not expose feature importances/coefs.")
        else:
            st.write("**Top features (global importance)**")
            st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Could not extract feature importance from this model wrapper.")

    st.write("**Optional: SHAP local explanation**")
    idx = st.number_input(
        "Row index to explain",
        min_value=0,
        max_value=max(0, len(X) - 1),
        value=0,
        step=1,
    )
    bg_n = st.slider(
        "Background sample size", min_value=50, max_value=1000, value=200, step=50
    )

    if st.button("Explain row with SHAP"):
        X_row = X.iloc[[int(idx)]].copy()
        bg = X.sample(
            min(int(bg_n), len(X)), random_state=int(cfg.get("RANDOM_STATE", 42))
        )

        # Use base estimator (Pipeline) for explainability when possible
        if hasattr(base, "named_steps"):
            shap_df, msg = try_shap_local_explanation(base, X_row=X_row, background=bg)
            if msg:
                st.warning(msg)
            if shap_df is not None:
                st.dataframe(shap_df, use_container_width=True)
        else:
            st.warning(
                "SHAP explanation requires a Pipeline with preprocess/model steps."
            )

    with st.expander("About"):
        st.markdown(
            """        **What this app demonstrates**
- Scoring a portfolio with a Probability of Default (PD) model
- Segmenting results into risk bands for decisioning/monitoring
- Basic explainability (feature importance + optional SHAP)

**Tip:** run the full pipeline locally with `python main.py` to train/evaluate and save a model to `models/`.
"""
        )


if __name__ == "__main__":
    main()
