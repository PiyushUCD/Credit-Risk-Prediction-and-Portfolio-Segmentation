"""Visualization utilities.

This module saves plots to OUTPUT_DIR and avoids plt.show() so it works in
headless environments (GitHub Actions, servers).

Plots included (intentionally minimal / recruiter-friendly):
1) Model comparison (ROC AUC + F1)
2) ROC curves
3) Confusion matrix (best model, chosen threshold)
4) Feature importance (tree/linear)
5) Portfolio segmentation (donut + bars) — based on FULL portfolio (e.g., 50k)
6) Probability distribution — based on FULL portfolio (e.g., 50k)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = str(self.config.get("OUTPUT_DIR", "results"))
        os.makedirs(self.output_dir, exist_ok=True)

        # Consistent, readable palette (explicit because the README screenshots matter).
        # These match matplotlib's "tab:" colors but are specified to keep visuals stable.
        self.colors = {
            "primary": "#1f77b4",   # blue
            "secondary": "#ff7f0e", # orange
            "accent": "#2ca02c",    # green
            "danger": "#d62728",    # red
        }

        # Risk band palette (low risk -> high risk)
        self.risk_band_colors = [
            "#2ca02c",  # Very Low
            "#98df8a",  # Low
            "#ffbb78",  # Medium
            "#ff7f0e",  # High
            "#ff9896",  # Very High
            "#d62728",  # Extreme
        ]

        # A readable default across platforms
        plt.rcParams.update(
            {
                "figure.dpi": int(self.config.get("PLOTS_DPI", 300)),
                "savefig.dpi": int(self.config.get("PLOTS_DPI", 300)),
                "font.size": 12,
            }
        )

    def _save(self, fig, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_model_comparison(self, evaluation_results: dict[str, dict]) -> str:
        models = list(evaluation_results.keys())
        aucs = [float(evaluation_results[m]["metrics"]["auc"]) for m in models]
        f1s = [float(evaluation_results[m]["metrics"]["f1"]) for m in models]

        x = np.arange(len(models))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.bar(
            x - width / 2,
            aucs,
            width,
            label="ROC AUC",
            color=self.colors["primary"],
        )
        ax.bar(
            x + width / 2,
            f1s,
            width,
            label="F1 (chosen threshold)",
            color=self.colors["secondary"],
        )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend()

        return self._save(fig, "01_model_performance_comparison.png")

    def plot_roc_curves(self, evaluation_results: dict[str, dict]) -> str:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        # Stable color cycle across runs
        ax.set_prop_cycle(
            color=[
                self.colors["primary"],
                self.colors["secondary"],
                self.colors["accent"],
                self.colors["danger"],
                "#9467bd",
                "#8c564b",
            ]
        )
        for name, res in evaluation_results.items():
            roc = res.get("roc_curve", {})
            auc = float(res.get("metrics", {}).get("auc", float("nan")))
            ax.plot(roc.get("fpr", []), roc.get("tpr", []), label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        return self._save(fig, "02_roc_curves.png")

    def plot_confusion_matrix(self, evaluation_results: dict[str, dict], best_model_name: str) -> str:
        """Plot the confusion matrix for `best_model_name`.

        `best_model_name` should exist in `evaluation_results`.
        If it doesn't, we fall back to the first model.
        """
        if best_model_name not in evaluation_results and len(evaluation_results) > 0:
            best_model_name = list(evaluation_results.keys())[0]

        cm = np.asarray(evaluation_results[best_model_name]["metrics"].get("confusion_matrix"))
        fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

        ax.set_title(f"Confusion Matrix — {best_model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Default", "Default"])
        ax.set_yticklabels(["No Default", "Default"])

        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return self._save(fig, "03_confusion_matrix.png")

    def plot_feature_importance(self, model, top_n: int = 15) -> str | None:
        """Plot feature importance / coefficients for pipeline models.

        Works for:
        - Pipeline(preprocess, model) where model has feature_importances_ or coef_
        - CalibratedClassifierCV where underlying estimator is stored in `.estimator`
        """

        # Unwrap calibrated model if necessary
        base = getattr(model, "estimator", model)

        if not hasattr(base, "named_steps"):
            return None

        preprocess = base.named_steps.get("preprocess")
        estimator = base.named_steps.get("model")
        if preprocess is None or estimator is None or not hasattr(preprocess, "get_feature_names_out"):
            return None

        try:
            feature_names = preprocess.get_feature_names_out()
        except Exception:
            return None

        importances = None
        title = "Feature Importance"
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
            title = "Feature Importance"
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float).ravel()
            importances = np.abs(coef)
            title = "Logistic Regression |coef|"
        else:
            return None

        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(int(top_n)).iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        ax.barh(imp_df["feature"], imp_df["importance"], color=self.colors["primary"])
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.25, axis="x")

        return self._save(fig, "04_feature_importance.png")

    def plot_portfolio(self, portfolio_table: pd.DataFrame) -> str:
        """Portfolio segmentation plot.

        Left: donut chart (portfolio composition)
        Right: grouped bars (avg predicted PD vs observed default rate)

        Key requirement: labels must be readable (percentages in legend).
        """

        df = portfolio_table.copy()

        # Stable ordering (when available)
        if "risk_band" in df.columns:
            order = ["Very Low", "Low", "Medium", "High", "Very High", "Extreme"]
            if set(order).issubset(set(df["risk_band"].unique())):
                df = df.set_index("risk_band").loc[order].reset_index()

        # Infer N (loan count)
        n_loans = None
        for col in ["loan_count", "loans", "Loans", "count"]:
            if col in df.columns:
                try:
                    n_loans = int(df[col].sum())
                except Exception:
                    n_loans = None
                break

        # Percent column is expected to sum to ~100
        pct = df["portfolio_percentage"].astype(float).to_numpy()
        risk = df["risk_band"].astype(str).to_list()
        legend_labels = [f"{r} — {p:.1f}%" for r, p in zip(risk, pct)]

        fig = plt.figure(figsize=(19, 7.5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.35])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Colors must be stable for the README screenshots.
        # Use one color per risk band, low-risk -> high-risk.
        colors = self.risk_band_colors[: len(pct)]
        wedges, _ = ax1.pie(
            pct,
            startangle=90,
            labels=None,
            colors=colors,
            wedgeprops=dict(width=0.42, edgecolor="white"),
        )

        title = "Portfolio Composition by Risk Band"
        if n_loans is not None:
            title += f" (N={n_loans:,})"
        ax1.set_title(title)
        ax1.legend(
            wedges,
            legend_labels,
            title="Risk band",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=12,
        )
        ax1.axis("equal")

        # Bars: predicted vs observed
        x = np.arange(len(df))
        width = 0.40
        ax2.bar(
            x - width / 2,
            df["avg_predicted_pd"].astype(float),
            width,
            label="Avg Predicted PD",
            color=self.colors["primary"],
        )
        ax2.bar(
            x + width / 2,
            df["actual_default_rate"].astype(float),
            width,
            label="Observed Default Rate",
            color=self.colors["secondary"],
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["risk_band"].astype(str), rotation=20, ha="right")
        ax2.set_ylabel("Rate")
        ax2.set_title("Predicted vs Observed Default by Risk Band")
        ax2.grid(True, alpha=0.25, axis="y")
        ax2.legend()

        # Make sure the y-limits give enough headroom for labels/legend
        ymax = float(
            max(df["avg_predicted_pd"].max(), df["actual_default_rate"].max())
        )
        ax2.set_ylim(0, min(1.0, ymax * 1.25 + 0.02))

        return self._save(fig, "05_portfolio_analysis.png")

    def plot_probability_distribution(self, y_true: pd.Series, probabilities: np.ndarray) -> str:
        n = int(len(probabilities))
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

        y = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
        mask_default = y == 1
        mask_nondefault = ~mask_default

        ax.hist(
            probabilities[mask_nondefault],
            bins=30,
            alpha=0.75,
            label="Non-Default",
            color=self.colors["primary"],
        )
        ax.hist(
            probabilities[mask_default],
            bins=30,
            alpha=0.75,
            label="Default",
            color=self.colors["secondary"],
        )

        ax.set_xlabel("Predicted PD")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Predicted Probability Distribution (N={n:,})")
        ax.grid(True, alpha=0.25)
        ax.legend()

        return self._save(fig, "06_probability_distribution.png")
