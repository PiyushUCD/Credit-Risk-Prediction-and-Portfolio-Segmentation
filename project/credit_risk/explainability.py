"""Lightweight explainability helpers (no SHAP).

Goal: provide simple, model-agnostic *global* feature importance for GitHub/Streamlit demos.
- For linear models: absolute value of coefficients
- For tree models: feature_importances_
- For Pipelines: tries to extract the final estimator
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def unwrap_estimator(model):
    """Return the final estimator if model is a sklearn Pipeline-like object."""
    if hasattr(model, "named_steps") and len(getattr(model, "named_steps", {})) > 0:
        # return the last step
        return list(model.named_steps.values())[-1]
    return model


def global_feature_importance(model, feature_names: list[str] | None = None, top_k: int = 30) -> pd.DataFrame:
    """Compute global feature importance for common estimator types.

    Returns a DataFrame with columns: feature, importance
    """
    est = unwrap_estimator(model)

    importances = None
    names = feature_names

    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        importances = np.abs(coef).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if names is None:
        names = [f"feature_{i}" for i in range(len(importances))]
    if len(names) != len(importances):
        # safety: align to min length
        n = min(len(names), len(importances))
        names = names[:n]
        importances = importances[:n]

    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(int(top_k)).reset_index(drop=True)
    return df
