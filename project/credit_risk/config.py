"""Project configuration.

Keep all tunables here so runs are reproducible and easy to tweak.
"""

from __future__ import annotations

PROJECT_CONFIG: dict[str, object] = {
    "PROJECT_NAME": "Credit Risk Default Prediction & Portfolio Segmentation",
    "VERSION": "1.3.1",
    # ----------------------------
    # Data settings
    # ----------------------------
    # KaggleHub dataset slug
    "KAGGLE_DATASET": "wordsforthewise/lending-club",
    "SAMPLE_SIZE": 50000,
    "TEST_SIZE": 0.20,
    "RANDOM_STATE": 42,
    # Output folders (these should generally be gitignored)
    "OUTPUT_DIR": "results",
    "MODELS_DIR": "models",
    # ----------------------------
    # Feature selection
    # ----------------------------
    "NUMERICAL_FEATURES": [
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "revol_bal",
        "revol_util",
        "total_acc",
    ],
    "CATEGORICAL_FEATURES": [
        "term",
        "grade",
        "emp_length",
        "home_ownership",
        "verification_status",
        "purpose",
    ],
    # Engineered features (computed if inputs exist)
    "ENGINEERED_FEATURES": [
        "loan_to_income",
        "revol_bal_to_loan",
    ],
    # ----------------------------
    # Target
    # ----------------------------
    "TARGET_COLUMN": "loan_status",
    "DEFAULT_STATUSES": [
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Late (16-30 days)",
        "Does not meet the credit policy. Status:Charged Off",
    ],
    # ----------------------------
    # Modeling
    # ----------------------------
    # NOTE: training uses sklearn Pipelines with a ColumnTransformer.
    # Model classes are looked up via sklearn imports in model_trainer.py.
    "MODELS": {
        "Logistic Regression": {
            "class": "LogisticRegression",
            "params": {
                "random_state": 42,
                "max_iter": 2000,
                "class_weight": "balanced",
                "solver": "lbfgs",
            },
        },
        "Random Forest": {
            "class": "RandomForestClassifier",
            "params": {
                "n_estimators": 500,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": -1,
                "min_samples_leaf": 2,
            },
        },
        "Gradient Boosting": {
            "class": "GradientBoostingClassifier",
            "params": {"n_estimators": 300, "random_state": 42},
        },
    },
    # Cross-validation (used for more stable model selection)
    "USE_CV": True,
    "CV_FOLDS": 5,
    # Probability calibration for the final (best) model
    "CALIBRATE_BEST_MODEL": True,
    "CALIBRATION_METHOD": "sigmoid",  # 'sigmoid' or 'isotonic'
    # Optional: hyperparameter tuning using RandomizedSearchCV
    # (kept OFF by default so `python main.py` runs fast).
    "ENABLE_TUNING": False,
    "TUNE_N_ITER": 20,
    "TUNE_SCORING": "roc_auc",
    # Parameter distributions use sklearn-style names under the pipeline's 'model__' prefix.
    # These are reasonable defaults; feel free to customize for your data.
    "HYPERPARAM_DISTS": {
        "Logistic Regression": {
            "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "Random Forest": {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 6, 10, 16],
            "model__min_samples_leaf": [1, 2, 5],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "model__n_estimators": [150, 300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0],
        },
    },
    # Threshold selection
    # - 'f1': maximize F1
    # - 'youden': maximize (tpr - fpr)
    # - 'cost': minimize expected cost using COST_FN / COST_FP
    "THRESHOLD_STRATEGY": "f1",
    "COST_FN": 5.0,
    "COST_FP": 1.0,
    # ----------------------------
    # Portfolio segmentation
    # ----------------------------
    # Fixed bands (probability of default). Kept ordered on purpose.
    "RISK_BANDS": {
        "Very Low": (0.00, 0.05),
        "Low": (0.05, 0.15),
        "Medium": (0.15, 0.25),
        "High": (0.25, 0.35),
        "Very High": (0.35, 0.50),
        "Extreme": (0.50, 1.00),
    },
    # ----------------------------
    # Visualization
    # ----------------------------
    "COLOR_PALETTE": "viridis",
    "PLOTS_DPI": 300,
}


def risk_band_edges() -> tuple[list[float], list[str]]:
    """Return ordered bin edges + labels from PROJECT_CONFIG['RISK_BANDS']."""
    bands = PROJECT_CONFIG["RISK_BANDS"]  # type: ignore[assignment]
    edges = [bands[k][0] for k in bands] + [1.0]  # type: ignore[index]
    labels = list(bands.keys())
    return edges, labels
