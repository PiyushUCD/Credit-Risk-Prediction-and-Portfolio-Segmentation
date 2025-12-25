"""Model training module.

Key improvements vs the original version:
- Uses sklearn Pipelines + ColumnTransformer (prevents leakage)
- One-hot encodes categoricals (stronger baseline than label encoding)
- Optional cross-validated model selection
- Optional probability calibration for the best model
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Registry used for safe class lookup
SKLEARN_MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}

# Optional: imbalanced-learn models (used for stronger imbalanced baselines)
try:
    from imblearn.ensemble import BalancedRandomForestClassifier  # type: ignore

    SKLEARN_MODEL_REGISTRY["BalancedRandomForestClassifier"] = BalancedRandomForestClassifier
except Exception:
    # imbalanced-learn not installed; the rest of the project still works
    pass


@dataclass
class TrainResults:
    """Artifacts produced by training."""

    models: dict[str, Pipeline]
    best_model_name: str
    best_model: object  # Pipeline or CalibratedClassifierCV
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_cols: list[str]
    target_col: str
    cv_auc: dict[str, float]


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config

    def _build_preprocessor(
        self, X: pd.DataFrame
    ) -> tuple[ColumnTransformer, list[str], list[str]]:
        """Create a preprocessing ColumnTransformer based on dtypes."""
        numeric_features = list(X.select_dtypes(include=["number", "bool"]).columns)
        categorical_features = [c for c in X.columns if c not in numeric_features]

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", categorical_pipe, categorical_features),
            ]
        )

        return preprocessor, numeric_features, categorical_features

    def _make_pipeline(
        self, preprocessor: ColumnTransformer, model_class_name: str, params: dict
    ) -> Pipeline:
        if model_class_name not in SKLEARN_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model class '{model_class_name}'. Allowed: {list(SKLEARN_MODEL_REGISTRY)}"
            )
        model_cls = SKLEARN_MODEL_REGISTRY[model_class_name]
        model = model_cls(**params)
        return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    def train(
        self, df: pd.DataFrame, feature_cols: list[str], target_col: str
    ) -> TrainResults:
        """Train models and pick the best one."""
        X = df[feature_cols].copy()
        y = df[target_col].astype(int).copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(self.config.get("TEST_SIZE", 0.2)),
            random_state=int(self.config.get("RANDOM_STATE", 42)),
            stratify=y,
        )

        preprocessor, _, _ = self._build_preprocessor(X_train)

        models_cfg = dict(self.config.get("MODELS", {}))
        use_cv = bool(self.config.get("USE_CV", True))
        cv_folds = int(self.config.get("CV_FOLDS", 5))

        trained: dict[str, Pipeline] = {}
        cv_auc: dict[str, float] = {}

        enable_tuning = bool(self.config.get("ENABLE_TUNING", False))
        tune_n_iter = int(self.config.get("TUNE_N_ITER", 20))
        tune_scoring = str(self.config.get("TUNE_SCORING", "roc_auc"))
        hyperparam_dists = dict(self.config.get("HYPERPARAM_DISTS", {}))

        # Cross-validate on training split to select the best model.
        if use_cv:
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=int(self.config.get("RANDOM_STATE", 42)),
            )

        for name, cfg in models_cfg.items():
            model_class_name = str(cfg["class"])
            params = dict(cfg.get("params", {}))
            # Allow optional models that depend on extra packages
            if model_class_name not in SKLEARN_MODEL_REGISTRY:
                print(
                    f"⚠️ Skipping model '{name}' because '{model_class_name}' is not available. "
                    "Install optional dependencies if you want to include it."
                )
                continue

            pipe = self._make_pipeline(preprocessor, model_class_name, params)

            # Optional hyperparameter tuning (uses CV on training split)
            if enable_tuning and name in hyperparam_dists:
                cv_tune = (
                    cv
                    if use_cv
                    else StratifiedKFold(
                        n_splits=3,
                        shuffle=True,
                        random_state=int(self.config.get("RANDOM_STATE", 42)),
                    )
                )
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=hyperparam_dists[name],
                    n_iter=tune_n_iter,
                    scoring=tune_scoring,
                    cv=cv_tune,
                    n_jobs=-1,
                    refit=True,
                    random_state=int(self.config.get("RANDOM_STATE", 42)),
                    verbose=0,
                )
                search.fit(X_train, y_train)
                pipe = search.best_estimator_
                cv_auc[name] = float(search.best_score_)
            else:
                if use_cv:
                    scores = cross_val_score(
                        pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
                    )
                    cv_auc[name] = float(np.mean(scores))
                else:
                    cv_auc[name] = float("nan")

                # Fit a final instance for evaluation on the held-out test set
                pipe.fit(X_train, y_train)

            trained[name] = pipe

        if not trained:
            raise ValueError(
                "No models were trained. Check PROJECT_CONFIG['MODELS'] and your installed dependencies."
            )

        # Choose best model by CV AUC if enabled, otherwise by test AUC will happen in evaluator.
        best_model_name = (
            max(cv_auc, key=cv_auc.get) if use_cv else list(trained.keys())[0]
        )

        best_model: object = trained[best_model_name]

        # Optional calibration (uses only training data)
        if bool(self.config.get("CALIBRATE_BEST_MODEL", True)):
            method = str(self.config.get("CALIBRATION_METHOD", "sigmoid"))
            # Calibrate using internal CV on training split for better probability quality.
            calibrated = CalibratedClassifierCV(
                estimator=trained[best_model_name], cv=3, method=method
            )
            calibrated.fit(X_train, y_train)
            best_model = calibrated

        return TrainResults(
            models=trained,
            best_model_name=best_model_name,
            best_model=best_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_cols=feature_cols,
            target_col=target_col,
            cv_auc=cv_auc,
        )
