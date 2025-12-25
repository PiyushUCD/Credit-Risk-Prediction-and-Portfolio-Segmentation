"""Entry point for the Credit Risk project.

Run:
    python main.py

Optional:
    python main.py --sample-size 100000 --no-cv
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy

import pandas as pd

from credit_risk.config import PROJECT_CONFIG
from credit_risk.data_loader import DataLoader
from credit_risk.helpers import (
    ensure_dir,
    format_metrics_table,
    save_csv,
    save_joblib,
    save_json,
    save_text,
)
from credit_risk.model_evaluator import ModelEvaluator
from credit_risk.model_trainer import ModelTrainer
from credit_risk.portfolio_analyzer import PortfolioAnalyzer
from credit_risk.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit Risk Default Prediction & Portfolio Segmentation"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of rows to load from the raw dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write outputs (plots, CSVs)",
    )
    parser.add_argument(
        "--models-dir", type=str, default=None, help="Where to write serialized models"
    )
    parser.add_argument(
        "--no-cv", action="store_true", help="Disable CV-based model selection"
    )
    parser.add_argument(
        "--no-calibrate", action="store_true", help="Disable probability calibration"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable RandomizedSearchCV hyperparameter tuning",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=None,
        help="RandomizedSearchCV n_iter (when --tune)",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default=None,
        choices=["f1", "youden", "cost"],
        help="Threshold selection strategy",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg = deepcopy(PROJECT_CONFIG)
    if args.sample_size is not None:
        cfg["SAMPLE_SIZE"] = args.sample_size
    if args.output_dir is not None:
        cfg["OUTPUT_DIR"] = args.output_dir
    if args.models_dir is not None:
        cfg["MODELS_DIR"] = args.models_dir
    if args.no_cv:
        cfg["USE_CV"] = False
    if args.no_calibrate:
        cfg["CALIBRATE_BEST_MODEL"] = False
    if args.tune:
        cfg["ENABLE_TUNING"] = True
    if args.tune_iter is not None:
        cfg["TUNE_N_ITER"] = int(args.tune_iter)
    if args.threshold is not None:
        cfg["THRESHOLD_STRATEGY"] = args.threshold
    return cfg


def write_summary(
    cfg: dict, metrics_df: pd.DataFrame, portfolio_df: pd.DataFrame, best_model_key: str
) -> str:
    lines = []
    lines.append("CREDIT RISK PREDICTION PROJECT SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Project: {cfg['PROJECT_NAME']} (v{cfg['VERSION']})")
    lines.append(f"Sample size: {cfg['SAMPLE_SIZE']:,}")
    lines.append(f"Threshold strategy: {cfg['THRESHOLD_STRATEGY']}")
    lines.append("")

    lines.append("MODEL PERFORMANCE (sorted by AUC):")
    lines.append(metrics_df.to_string(index=False))
    lines.append("")

    lines.append(f"BEST MODEL USED FOR PORTFOLIO: {best_model_key}")
    lines.append("")

    lines.append("PORTFOLIO SEGMENTATION:")
    lines.append(portfolio_df.to_string(index=False))
    lines.append("")

    lines.append("NOTES:")
    lines.append(
        "- Categorical features are one-hot encoded; missing values are imputed inside the pipeline."
    )
    lines.append(
        "- The reported F1/precision/recall use the optimized threshold, not necessarily 0.5."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    output_dir = str(cfg.get("OUTPUT_DIR", "results"))
    models_dir = str(cfg.get("MODELS_DIR", "models"))
    ensure_dir(output_dir)
    ensure_dir(models_dir)

    print("🚀 CREDIT RISK DEFAULT PREDICTION & PORTFOLIO SEGMENTATION")
    print("=" * 72)

    # 1) Load
    loader = DataLoader(cfg)
    raw_df = loader.load_dataset()
    if raw_df is None:
        raise SystemExit(1)

    loaded = loader.build_model_frame(raw_df)

    # 2) Train
    trainer = ModelTrainer(cfg)
    train_results = trainer.train(loaded.df, loaded.feature_cols, loaded.target_col)

    # 3) Evaluate
    evaluator = ModelEvaluator(cfg)
    eval_results = evaluator.evaluate(train_results)
    metrics_df = format_metrics_table(
        eval_results, test_n=len(train_results.y_test), total_n=len(loaded.df)
    )

    # 4) Choose best model key for scoring (prefer calibrated if available)
    best_key = train_results.best_model_name
    use_calibrated = False
    if f"{best_key} (Calibrated)" in eval_results:
        best_key = f"{best_key} (Calibrated)"
        use_calibrated = True

    # Test-set probabilities (used for evaluation plots like ROC / CM)
    best_probs_test = eval_results[best_key]["probabilities"]

    # Full-portfolio probabilities (ALL rows: e.g., 50,000 loans)
    scoring_model = (
        train_results.best_model
        if use_calibrated
        else train_results.models[train_results.best_model_name]
    )
    X_all = loaded.df[train_results.feature_cols]
    y_all = loaded.df[train_results.target_col]
    best_probs_all = scoring_model.predict_proba(X_all)[:, 1]

    # 5) Portfolio segmentation (use FULL portfolio for business segmentation)
    analyzer = PortfolioAnalyzer(cfg)
    portfolio = analyzer.segment(best_probs_all, y_all)

    # 6) Save artifacts
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    portfolio_path = os.path.join(output_dir, "portfolio_analysis.csv")
    portfolio_loan_path = os.path.join(output_dir, "portfolio_loan_level.csv")
    summary_path = os.path.join(output_dir, "project_summary.txt")
    config_path = os.path.join(output_dir, "run_config.json")

    save_csv(metrics_df, metrics_path, index=False)
    save_csv(portfolio.portfolio_table, portfolio_path, index=False)
    save_csv(portfolio.loan_level, portfolio_loan_path, index=False)
    save_text(
        write_summary(cfg, metrics_df, portfolio.portfolio_table, best_key),
        summary_path,
    )
    save_json(cfg, config_path)

    # 6b) Markdown report for GitHub (tables + plots)
    report_path = os.path.join(output_dir, "REPORT.md")
    from credit_risk.reporter import write_markdown_report

    write_markdown_report(
        out_path=report_path,
        cfg=cfg,
        metrics_df=metrics_df,
        portfolio_df=portfolio.portfolio_table,
        best_model_name=best_key,
        plots_dir=output_dir,
    )

    # Save best model (pipeline or calibrated wrapper)
    save_joblib(
        train_results.best_model, os.path.join(models_dir, "best_credit_risk_model.pkl")
    )

    # 7) Visuals
    viz = Visualizer(cfg)
    viz.plot_model_comparison(eval_results)
    viz.plot_roc_curves(eval_results)
    viz.plot_confusion_matrix(eval_results, best_key)
    viz.plot_feature_importance(train_results.best_model)
    viz.plot_portfolio(portfolio.portfolio_table)
    viz.plot_probability_distribution(y_all, best_probs_all)

    print("\n✅ Done. Outputs written to:")
    print(f"   • {output_dir}/")
    print(f"   • {models_dir}/best_credit_risk_model.pkl")


if __name__ == "__main__":
    main()
