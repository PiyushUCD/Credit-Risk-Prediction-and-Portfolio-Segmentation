"""Model explainability utilities.

This module is intentionally lightweight and safe:
- Works with scikit-learn Pipelines (ColumnTransformer + model)
- Supports global feature importance for common sklearn estimators
- Optionally supports SHAP *if* installed (used in the Streamlit app)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def unwrap_estimator(model):
    """Unwrap common wrappers (e.g., CalibratedClassifierCV) to the underlying estimator."""
    # scikit-learn's CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_"):
        try:
            return model.calibrated_classifiers_[0].estimator
        except Exception:
            pass
    if hasattr(model, "estimator"):
        try:
            return model.estimator
        except Exception:
            pass
    return model


def get_feature_names_from_pipeline(pipe) -> np.ndarray:
    """Return post-preprocessing feature names from a Pipeline with a 'preprocess' step."""
    preprocess = getattr(pipe, "named_steps", {}).get("preprocess")
    if preprocess is None:
        return np.array([])

    # sklearn >= 1.0
    try:
        return preprocess.get_feature_names_out()
    except Exception:
        return np.array([])


def global_feature_importance(pipe, top_k: int = 30) -> pd.DataFrame:
    """Compute a global feature importance table.

    Supports:
    - tree models: feature_importances_
    - linear models: coef_
    """
    feature_names = get_feature_names_from_pipeline(pipe)
    model = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else None
    if model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    importance: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importance = np.abs(coef).ravel()

    if importance is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if feature_names.size != importance.size:
        # Fallback: if names cannot be extracted, create generic names.
        feature_names = np.array([f"f{i}" for i in range(importance.size)])

    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df = (
        df.sort_values("importance", ascending=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )
    return df


def try_shap_local_explanation(
    pipe, X_row: pd.DataFrame, background: pd.DataFrame | None = None
) -> tuple[pd.DataFrame | None, str]:
    """Best-effort SHAP explanation for a single row.

    Returns (explanation_df_or_none, message).
    - Works best for tree-based models.
    - For other models, may fall back or return None.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return (
            None,
            "SHAP not installed. Install with: pip install -r requirements-app.txt",
        )

    if (
        not hasattr(pipe, "named_steps")
        or "preprocess" not in pipe.named_steps
        or "model" not in pipe.named_steps
    ):
        return None, "Pipeline does not expose 'preprocess' and 'model' steps."

    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    feature_names = get_feature_names_from_pipeline(pipe)

    # Transform inputs to model space
    X_row_t = preprocess.transform(X_row)
    if background is not None:
        bg_t = preprocess.transform(background)
    else:
        bg_t = None

    # Prefer TreeExplainer when possible
    try:
        explainer = shap.TreeExplainer(model, data=bg_t)
        shap_vals = explainer.shap_values(X_row_t)
    except Exception:
        # Generic explainer fallback
        try:
            explainer = shap.Explainer(model, bg_t)
            shap_vals = explainer(X_row_t)
        except Exception as e:
            return None, f"Could not compute SHAP explanation: {e}"

    # Normalize output shape
    try:
        # TreeExplainer may return list for binary classification
        if isinstance(shap_vals, list):
            vals = np.asarray(shap_vals[1]).ravel()
        else:
            vals = np.asarray(getattr(shap_vals, "values", shap_vals)).ravel()
    except Exception:
        return None, "Could not parse SHAP values output."

    if feature_names.size != vals.size:
        feature_names = np.array([f"f{i}" for i in range(vals.size)])

    out = pd.DataFrame({"feature": feature_names, "shap_value": vals})
    out["abs_shap"] = out["shap_value"].abs()
    out = (
        out.sort_values("abs_shap", ascending=False)
        .head(20)
        .drop(columns=["abs_shap"])
        .reset_index(drop=True)
    )
    return out, ""


def try_shap_global_summary(
    pipe,
    X_sample: pd.DataFrame,
    *,
    max_display: int = 20,
    out_png: str | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Best-effort SHAP global summary (bar) for a sample of rows.

    Returns (importance_df_or_none, message).
    If `out_png` is provided, also writes a PNG summary plot.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return (
            None,
            "SHAP not installed. Install with: pip install -r requirements-app.txt",
        )

    if (
        not hasattr(pipe, "named_steps")
        or "preprocess" not in pipe.named_steps
        or "model" not in pipe.named_steps
    ):
        return None, "Pipeline does not expose 'preprocess' and 'model' steps."

    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    feature_names = get_feature_names_from_pipeline(pipe)

    # Transform to model space
    X_t = preprocess.transform(X_sample)

    # Prefer TreeExplainer when possible
    try:
        explainer = shap.TreeExplainer(model, data=X_t)
        shap_vals = explainer.shap_values(X_t)
    except Exception:
        try:
            explainer = shap.Explainer(model, X_t)
            shap_vals = explainer(X_t)
        except Exception as e:
            return None, f"Could not compute SHAP values: {e}"

    # Normalize output
    try:
        if isinstance(shap_vals, list):
            vals = np.asarray(shap_vals[1])
        else:
            vals = np.asarray(getattr(shap_vals, "values", shap_vals))
    except Exception:
        return None, "Could not parse SHAP output."

    if vals.ndim == 1:
        vals = vals.reshape(1, -1)

    if feature_names.size != vals.shape[1]:
        feature_names = np.array([f"f{i}" for i in range(vals.shape[1])])

    mean_abs = np.mean(np.abs(vals), axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).head(int(max_display)).reset_index(drop=True)

    if out_png is not None:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            plot_df = df.iloc[::-1]
            ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
            ax.set_title("SHAP global feature impact (mean |SHAP|)")
            ax.set_xlabel("mean(|SHAP|)")
            ax.grid(True, alpha=0.25, axis="x")
            fig.tight_layout()
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            return df, f"Computed SHAP importance, but could not save plot: {e}"

    return df, ""
