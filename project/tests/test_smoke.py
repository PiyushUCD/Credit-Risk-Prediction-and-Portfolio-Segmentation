import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on PYTHONPATH when running in CI.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_risk.config import PROJECT_CONFIG  # noqa: E402
from credit_risk.model_evaluator import ModelEvaluator  # noqa: E402
from credit_risk.model_trainer import ModelTrainer  # noqa: E402
from credit_risk.portfolio_analyzer import PortfolioAnalyzer  # noqa: E402


def _make_synthetic(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cfg = dict(PROJECT_CONFIG)

    num_cols = list(cfg.get("NUMERICAL_FEATURES", []))
    cat_cols = list(cfg.get("CATEGORICAL_FEATURES", []))

    df = pd.DataFrame(index=range(n))

    # Numeric features (roughly plausible ranges)
    df["loan_amnt"] = rng.integers(1000, 35000, size=n)
    df["int_rate"] = rng.uniform(5, 28, size=n)
    df["installment"] = rng.uniform(50, 1200, size=n)
    df["annual_inc"] = rng.uniform(20000, 200000, size=n)
    df["dti"] = rng.uniform(0, 35, size=n)
    df["delinq_2yrs"] = rng.integers(0, 4, size=n)
    df["inq_last_6mths"] = rng.integers(0, 6, size=n)
    df["open_acc"] = rng.integers(1, 25, size=n)
    df["revol_bal"] = rng.uniform(0, 60000, size=n)
    df["revol_util"] = rng.uniform(0, 120, size=n)
    df["total_acc"] = rng.integers(1, 60, size=n)

    # Categoricals
    df["term"] = rng.choice([" 36 months", " 60 months"], size=n)
    df["grade"] = rng.choice(list("ABCDEFG"), size=n)
    df["emp_length"] = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "10+ years"],
        size=n,
    )
    df["home_ownership"] = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n)
    df["verification_status"] = rng.choice(
        ["Verified", "Source Verified", "Not Verified"], size=n
    )
    df["purpose"] = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business"],
        size=n,
    )

    # Keep only configured cols (future-proof if config changes)
    keep = [c for c in (num_cols + cat_cols) if c in df.columns]
    df = df[keep].copy()

    # Synthetic target: higher PD for higher rate + higher dti + higher revol_util
    logit = (
        0.08 * (df["int_rate"] - 12)
        + 0.05 * (df["dti"] - 15)
        + 0.02 * (df["revol_util"] - 50)
    )
    p = 1 / (1 + np.exp(-logit))
    df["loan_default"] = (rng.uniform(0, 1, size=n) < p).astype(int)

    return df


def test_end_to_end_smoke():
    cfg = dict(PROJECT_CONFIG)
    cfg["USE_CV"] = False
    cfg["ENABLE_TUNING"] = False
    cfg["CALIBRATE_BEST_MODEL"] = False
    cfg["TEST_SIZE"] = 0.25

    df = _make_synthetic(n=400, seed=int(cfg.get("RANDOM_STATE", 42)))

    feature_cols = [c for c in df.columns if c != "loan_default"]

    trainer = ModelTrainer(cfg)
    train_res = trainer.train(df, feature_cols=feature_cols, target_col="loan_default")

    assert train_res.best_model is not None
    assert train_res.best_model_name

    evaluator = ModelEvaluator(cfg)
    eval_res = evaluator.evaluate(train_res)

    assert train_res.best_model_name in eval_res
    probs = eval_res[train_res.best_model_name]["probabilities"]
    assert len(probs) == len(train_res.y_test)

    analyzer = PortfolioAnalyzer(cfg)
    port = analyzer.segment(probs, train_res.y_test)

    assert "loan_count" in port.portfolio_table.columns
    assert int(port.portfolio_table["loan_count"].sum()) == int(len(train_res.y_test))
