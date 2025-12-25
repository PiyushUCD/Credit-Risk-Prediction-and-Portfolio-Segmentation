"""Portfolio segmentation and business-oriented analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResults:
    portfolio_table: pd.DataFrame
    loan_level: pd.DataFrame


class PortfolioAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    def segment(self, probabilities: np.ndarray, y_true: pd.Series) -> PortfolioResults:
        """Assign risk bands and build a portfolio summary table."""
        risk_bands: dict[str, tuple[float, float]] = dict(
            self.config.get("RISK_BANDS", {})
        )
        if not risk_bands:
            raise ValueError("RISK_BANDS not configured")

        # Ensure ordered edges
        labels = list(risk_bands.keys())
        edges = [risk_bands[k][0] for k in labels] + [1.0]

        loan_level = pd.DataFrame(
            {
                "actual_default": y_true.astype(int).values,
                "predicted_pd": probabilities,
            }
        )

        loan_level["risk_band"] = pd.cut(
            loan_level["predicted_pd"],
            bins=edges,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        # Summary statistics
        grouped = loan_level.groupby("risk_band", observed=True).agg(
            loan_count=("predicted_pd", "size"),
            avg_predicted_pd=("predicted_pd", "mean"),
            std_predicted_pd=("predicted_pd", "std"),
            actual_default_rate=("actual_default", "mean"),
            total_defaults=("actual_default", "sum"),
        )

        grouped = grouped.fillna(0.0)
        grouped["portfolio_percentage"] = (
            grouped["loan_count"] / grouped["loan_count"].sum() * 100.0
        )
        grouped["expected_defaults"] = (
            grouped["avg_predicted_pd"] * grouped["loan_count"]
        )

        # Lift vs overall
        overall_dr = loan_level["actual_default"].mean()
        grouped["lift_vs_overall"] = grouped["actual_default_rate"] / (
            overall_dr + 1e-9
        )

        # Pretty rounding for presentation (keep raw in loan_level)
        portfolio_table = grouped.round(
            {
                "avg_predicted_pd": 4,
                "std_predicted_pd": 4,
                "actual_default_rate": 4,
                "portfolio_percentage": 2,
                "expected_defaults": 2,
                "lift_vs_overall": 2,
            }
        ).reset_index()

        return PortfolioResults(portfolio_table=portfolio_table, loan_level=loan_level)
