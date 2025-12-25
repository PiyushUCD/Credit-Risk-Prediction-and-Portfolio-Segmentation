"""Model evaluation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def evaluate(self, train_results) -> dict[str, dict]:
        """Evaluate trained models on the held-out test set."""
        X_test = train_results.X_test
        y_test = train_results.y_test

        results: dict[str, dict] = {}

        # Evaluate all base models
        for name, model in train_results.models.items():
            results[name] = self._evaluate_single(model, X_test, y_test)

        # Evaluate calibrated best model if different
        best_name = train_results.best_model_name
        best_model = train_results.best_model
        if best_model is not train_results.models.get(best_name):
            results[f"{best_name} (Calibrated)"] = self._evaluate_single(
                best_model, X_test, y_test
            )

        return results

    def _evaluate_single(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_proba = model.predict_proba(X_test)[:, 1]

        # Default threshold metrics
        y_pred_05 = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)

        # Optimize threshold according to config
        threshold, threshold_reason = self._select_threshold(y_test.values, y_proba)
        y_pred_opt = (y_proba >= threshold).astype(int)

        metrics = {
            "auc": float(auc),
            "accuracy@0.5": float(accuracy_score(y_test, y_pred_05)),
            "precision@0.5": float(precision_score(y_test, y_pred_05, zero_division=0)),
            "recall@0.5": float(recall_score(y_test, y_pred_05, zero_division=0)),
            "f1@0.5": float(f1_score(y_test, y_pred_05, zero_division=0)),
            "threshold": float(threshold),
            "threshold_strategy": threshold_reason,
            "accuracy": float(accuracy_score(y_test, y_pred_opt)),
            "precision": float(precision_score(y_test, y_pred_opt, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_opt, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_opt, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_opt),
        }

        # Business-friendly lift style metric: what fraction of defaults are captured
        # by the top X% highest-risk loans.
        top_pct = float(self.config.get("TOP_RISK_PERCENTILE", 0.10))
        k = max(1, int(len(y_proba) * top_pct))
        order = np.argsort(-y_proba)
        top_idx = order[:k]
        defaults_total = int(np.sum(y_test.values))
        defaults_top = int(np.sum(y_test.values[top_idx]))
        metrics["top_risk_percentile"] = float(top_pct)
        metrics["defaults_captured@top_pct"] = (
            float(defaults_top / defaults_total) if defaults_total > 0 else float("nan")
        )

        # Curves (for plots)
        fpr, tpr, roc_thr = roc_curve(y_test, y_proba)

        return {
            "model": model,
            "_y_true": y_test.values,
            "probabilities": y_proba,
            "predictions": y_pred_opt,
            "metrics": metrics,
            "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thr},
        }

    def _select_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> tuple[float, str]:
        strategy = str(self.config.get("THRESHOLD_STRATEGY", "f1")).lower()

        if strategy == "youden":
            fpr, tpr, thr = roc_curve(y_true, y_proba)
            j = tpr - fpr
            idx = int(np.argmax(j))
            return float(thr[idx]), "youden"

        if strategy == "cost":
            cost_fn = float(self.config.get("COST_FN", 5.0))
            cost_fp = float(self.config.get("COST_FP", 1.0))
            thresholds = np.linspace(0.01, 0.99, 99)
            best_thr = 0.5
            best_cost = float("inf")

            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                cost = cost_fn * fn + cost_fp * fp
                if cost < best_cost:
                    best_cost = cost
                    best_thr = float(t)
            return best_thr, "cost"

        # default: F1 optimization
        thresholds = np.linspace(0.05, 0.95, 91)
        best_thr = 0.5
        best_f1 = -1.0
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_thr = float(t)
        return best_thr, "f1"