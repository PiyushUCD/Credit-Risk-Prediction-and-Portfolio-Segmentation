"""Helper utilities (filesystem, serialization, summaries)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_joblib(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(obj, path)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=index)


def save_text(text: str, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")

    def _to_jsonable(x: Any):
        if is_dataclass(x):
            return asdict(x)
        if hasattr(x, "tolist"):
            return x.tolist()
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_to_jsonable)


def format_metrics_table(
    evaluation_results: dict[str, dict], *, test_n: int | None = None, total_n: int | None = None
) -> pd.DataFrame:
    rows = []
    for name, res in evaluation_results.items():
        m = res["metrics"]
        rows.append(
            {
                "model": name,
                "auc": m["auc"],
                "pr_auc": m["pr_auc"],
                "brier": m.get("brier"),
                "log_loss": m.get("log_loss"),
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "accuracy": m["accuracy"],
                "threshold": m["threshold"],
                "threshold_strategy": m["threshold_strategy"],
                "defaults_captured@top_pct": m.get("defaults_captured@top_pct"),
            }
        )
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    if test_n is not None:
        df.insert(1, "test_n", int(test_n))
    if total_n is not None:
        df.insert(1, "total_n", int(total_n))
    return df
