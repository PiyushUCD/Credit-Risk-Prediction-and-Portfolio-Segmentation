"""Visualization utilities.

All plots are saved to OUTPUT_DIR.
The module avoids plt.show() so it works in headless environments (GitHub Actions, servers).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Visualizer:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = str(self.config.get("OUTPUT_DIR", "results"))
        os.makedirs(self.output_dir, exist_ok=True)

        sns.set_palette(str(self.config.get("COLOR_PALETTE", "viridis")))
        plt.rcParams["figure.figsize"] = (12, 8)

    def plot_model_comparison(self, evaluation_results: dict[str, dict]) -> str:
        models = list(evaluation_results.keys())
        aucs = [evaluation_results[m]["metrics"]["auc"] for m in models]
        f1s = [evaluation_results[m]["metrics"]["f1"] for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, aucs, width, label="ROC AUC")
        ax.bar(x + width / 2, f1s, width, label="F1 (opt threshold)")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

        path = os.path.join(self.output_dir, "01_model_performance_comparison.png")
        fig.tight_layout()
        fig.savefig(
            path, dpi=int(self.config.get("PLOTS_DPI", 300)), bbox_inches="tight"
        )
        plt.close(fig)
        return path

    def plot_roc_curves(self, evaluation_results: dict[str, dict]) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in evaluation_results.items():
            roc = res["roc_curve"]
            auc = res["metrics"]["auc"]
            ax.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        path = os.path.join(self.output_dir, "02_roc_curves.png")
        fig.tight_layout()
        fig.savefig(
            path, dpi=int(self.config.get("PLOTS_DPI", 300)), bbox_inches="tight"
        )
        plt.close(fig)
        return path

    def plot_pr_curves(self, evaluation_results: dict[str, dict]) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in evaluation_results.items():
            pr = res["pr_curve"]
            pr_auc = res["metrics"]["pr_auc"]
            ax.plot(pr["recall"], pr["precision"], label=f"{name} (AP={pr_auc:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        path = os.path.join(self.output_dir, "02b_precision_recall_curves.png")
        fig.tight_layout()
        fig.savefig(
            path, dpi=int(self.config.get("PLOTS_DPI", 300)), bbox_inches="tight"
        )
        plt.close(fig)
        return path

    def plot_confusion_matrix(
        self, evaluation_results: dict[str, dict], best_model_name: str
    ) -> str:
        # Pick calibrated if present
        key = best_model_name
        if f"{best_model_name} (Calibrated)" in evaluation_results:
            key = f"{best_model_name} (Calibrated)"

        cm = evaluation_results[key]["metrics"]["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {key}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Non-Default", "Default"])
        ax.set_yticklabels(["Non-Default", "Default"], rotation=0)

        path = os.path.join(self.output_dir, "03_confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(
            path, dpi=int(self.config.get("PLOTS_DPI", 300)), bbox_inches="tight"
        )
        plt.close(fig)
        return path

    def plot_feature_importance(self, model, top_n: int = 15) -> str | None:
        """Plot feature importance / coefficients for pipeline models.

        Works for:
        - Pipeline(preprocess, model) where model has feature_importances_ or coef_
        - CalibratedClassifierCV where underlying estimator is stored in `.estimator`
        """
        # Unwrap calibrated model if necessary
        if hasattr(model, "estimator"):
            base = model.estimator
        else:
            base = model

        if not hasattr(base, "named_steps"):
            return None

        preprocess = base.named_steps.get("preprocess")
        estimator = base.named_steps.get("model")

        if (
            preprocess is None
            or estimator is None
            or not hasattr(preprocess, "get_feature_names_out")
        ):
            return None

        try:
            feature_names = preprocess.get_feature_names_out()
        except Exception:
            return None

        importances = None
        title = "Feature Importance"

        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_)
            title = "Feature Importance"
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_).ravel()
            importances = np.abs(coef)
            title = "Logistic Regression |coef|"
        else:
            return None

        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
        imp_df = imp_df.iloc[::-1]  # for horizontal bar plot

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(imp_df["feature"], imp_df["importance"])
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")

        path = os.path.join(self.output_dir, "04_feature_importance.png")
        fig.tight_layout()
        fig.savefig(
            path, dpi=int(self.config.get("PLOTS_DPI", 300)), bbox_inches="tight"
        )
        plt.close(fig)
        return path

    def plot_portfolio(self, portfolio_table: pd.DataFrame) -> str:
        # Donut chart + bar chart (clean layout for GitHub)
        # Expect these columns from PortfolioAnalyzer:
        #   risk_band, portfolio_percentage, avg_predicted_pd, actual_default_rate, loan_count
        n_loans = None
        for col in ["loan_count", "loans", "Loans", "count"]:
            if col in portfolio_table.columns:
                n_loans = int(portfolio_table[col].sum())
                break

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # --- Donut: use legend to avoid label overlap
        wedges, _, _ = ax1.pie(
            portfolio_table["portfolio_percentage"],
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white"),
        )
        title = "Portfolio Composition by Risk Band"
        if n_loans is not None:
            title += f" (N={n_loans:,})"
        ax1.set_title(title)
        ax1.legend(
            wedges,
            portfolio_table["risk_band"],
            title="Risk band",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )
        ax1.axis("equal")

        # --- Bars: align and order
        x = np.arange(len(portfolio_table))
        width = 0.38
        ax2.bar(
            x - width / 2,
            portfolio_table["avg_predicted_pd"],
            width,
            label="Avg Predicted PD",
        )
        ax2.bar(
            x + width / 2,
            portfolio_table["actual_default_rate"],
            width,
            label="Observed Default Rate",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(portfolio_table["risk_band"], rotation=30, ha="right")
        ax2.set_ylabel("Rate")
        ax2.set_title("Predicted vs Observed Default by Risk Band")
        ax2.grid(True, alpha=0.25, axis="y")
        ax2.legend()

        path = os.path.join(self.output_dir, "05_portfolio_analysis.png")
        fig.savefig(
            path, dpi=int(self.config.get("PLOTS_DPI", 300)), bbox_inches="tight"
        )
        plt.close(fig)
        return path

    def plot_probability_distribution(
        self, y_true: pd.Series, probabilities: np.ndarray
    ) -> str:
        n = int(len(probabilities))
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

        # Two distributions: default vs non-default
        mask_default = (y_true == 1).values if hasattr(y_true, "values") else (y_true == 1)
        mask_nondefault = ~mask_default

        ax.hist(probabilities[mask_nondefault], bins=30, alpha=0.7, label="Non-Default")
        ax.hist(probabilities[mask_default], bins=30, alpha=0.7, label="Default")

        ax.set_xlabel("Predicted PD")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Predicted Probability Distribution (N={n:,})")
        ax.grid(True, alpha=0.25)
        ax.legend()

        path = os.path.join(self.output_dir, "06_probability_distribution.png")
        fig.savefig(
            path, dpi=int(self.config.get("PLOTS_DPI", 300)), bbox_inches="tight"
        )
        plt.close(fig)
        return path

