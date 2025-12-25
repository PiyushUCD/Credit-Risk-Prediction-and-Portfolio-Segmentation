"""Markdown report generation.

Creates a lightweight, GitHub-friendly report that you can:
- open locally after a run (results/REPORT.md)
- copy/paste into README

The report intentionally avoids heavy templating dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Return a GitHub-flavored markdown table."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else float_fmt.format(float(x)))
    return out.to_markdown(index=False)


def write_markdown_report(
    *,
    out_path: str,
    cfg: dict,
    metrics_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    best_model_name: str,
    plots_dir: str,
) -> None:
    """Write a run report to `out_path`."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plots = {
        "Model comparison": "01_model_performance_comparison.png",
        "ROC curves": "02_roc_curves.png",
        "PR curves": "02b_precision_recall_curves.png",
        "Confusion matrix": "03_confusion_matrix.png",
        "Feature importance": "04_feature_importance.png",
        "Portfolio segmentation": "05_portfolio_analysis.png",
        "PD distribution": "06_probability_distribution.png",
        "Calibration": "07_calibration_curves.png",
        "SHAP summary (optional)": "08_shap_summary.png",
    }

    # Keep tables compact for markdown
    mcols = [
        c
        for c in [
            "model",
            "total_n",
            "test_n",
            "auc",
            "pr_auc",
            "brier",
            "f1",
            "precision",
            "recall",
            "threshold",
            "defaults_captured@top_pct",
        ]
        if c in metrics_df.columns
    ]
    metrics_small = metrics_df[mcols].copy()

    pcols = [
        c
        for c in [
            "risk_band",
            "loan_count",
            "portfolio_percentage",
            "avg_predicted_pd",
            "actual_default_rate",
        ]
        if c in portfolio_df.columns
    ]
    portfolio_small = portfolio_df[pcols].copy()

    lines: list[str] = []
    lines.append(f"# Run report – {cfg.get('PROJECT_NAME', 'Credit Risk')} (v{cfg.get('VERSION')})")
    lines.append("")
    lines.append(f"- Sample size (loaded): **{int(cfg.get('SAMPLE_SIZE', 0)):,}**")
    lines.append(f"- Best model: **{best_model_name}**")
    lines.append(f"- Threshold strategy: **{cfg.get('THRESHOLD_STRATEGY')}**")
    lines.append("")

    lines.append("## Model metrics (holdout test)")
    lines.append(_md_table(metrics_small))
    lines.append("")

    lines.append("## Portfolio segmentation (full portfolio)")
    lines.append(_md_table(portfolio_small))
    lines.append("")

    lines.append("## Plots")
    for title, fn in plots.items():
        img_path = Path(plots_dir) / fn
        if img_path.exists():
            lines.append(f"### {title}")
            lines.append(f"![]({fn})")
            lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
