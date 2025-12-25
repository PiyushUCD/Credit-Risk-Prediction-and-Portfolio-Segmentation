"""Data loading and light preprocessing.

Design choice:
- This module is responsible for data acquisition + creating the binary target.
- Model-specific preprocessing (imputation, encoding, scaling) happens in sklearn Pipelines
  in `model_trainer.py` to avoid leakage.

Kaggle access:
- `kagglehub` will use your configured Kaggle credentials.
  If you don't have them set up, see the README.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import kagglehub
import pandas as pd


@dataclass
class LoadedData:
    """Container for model-ready frame and metadata."""

    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str


class DataLoader:
    def __init__(self, config: dict):
        self.config = config

    def load_dataset(self) -> pd.DataFrame | None:
        """Download the Lending Club dataset from Kaggle (via KaggleHub) and load a sample."""
        dataset_slug = str(
            self.config.get("KAGGLE_DATASET", "wordsforthewise/lending-club")
        )
        sample_size = int(self.config.get("SAMPLE_SIZE", 50000))

        # Allow users to override with a local CSV path for reproducibility / offline runs.
        local_path = os.environ.get("LENDING_CLUB_CSV")
        if local_path and os.path.exists(local_path):
            print(f"📄 Loading local CSV from LENDING_CLUB_CSV: {local_path}")
            return pd.read_csv(local_path, nrows=sample_size, low_memory=False)

        print("📥 Downloading dataset from Kaggle (kagglehub)...")
        try:
            path = kagglehub.dataset_download(dataset_slug)
        except Exception as e:
            print("❌ Kaggle download failed.")
            print("   Tip: ensure Kaggle credentials are configured (see README).")
            print(f"   Error: {e}")
            return None

        csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        accepted_files = [
            f for f in csv_files if "accepted" in os.path.basename(f).lower()
        ]

        if not accepted_files:
            print("❌ No 'accepted' loans CSV found in downloaded dataset.")
            return None

        # Use the largest accepted-loans file (usually the main one)
        main_file = max(accepted_files, key=lambda p: os.path.getsize(p))
        print(f"🎯 Loading file: {os.path.basename(main_file)}")

        try:
            df = pd.read_csv(main_file, nrows=sample_size, low_memory=False)
            print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            return df
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            return None

    def build_model_frame(self, raw_df: pd.DataFrame) -> LoadedData:
        """Create binary target + select features + add a couple of engineered features.

        Returns a DataFrame with columns = selected features + 'loan_default'.
        Missing values are left as-is (handled later by pipelines).
        """
        df = raw_df.copy()

        target_source = str(self.config.get("TARGET_COLUMN", "loan_status"))
        default_statuses = set(self.config.get("DEFAULT_STATUSES", []))
        if target_source not in df.columns:
            raise ValueError(f"Target column '{target_source}' not found in dataset.")

        df["loan_default"] = df[target_source].apply(
            lambda x: 1 if x in default_statuses else 0
        )

        num_features = list(self.config.get("NUMERICAL_FEATURES", []))
        cat_features = list(self.config.get("CATEGORICAL_FEATURES", []))
        all_features = num_features + cat_features

        available = [c for c in all_features if c in df.columns]
        missing = [c for c in all_features if c not in df.columns]
        if missing:
            print(f"⚠️  Missing features in this dataset slice: {missing}")

        if not available:
            raise ValueError("No configured features found in dataset.")

        # --- engineered features (only if source columns exist) ---
        if "loan_amnt" in df.columns and "annual_inc" in df.columns:
            df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1.0)
            if "loan_to_income" not in available:
                available.append("loan_to_income")

        if "revol_bal" in df.columns and "loan_amnt" in df.columns:
            df["revol_bal_to_loan"] = df["revol_bal"] / (df["loan_amnt"] + 1.0)
            if "revol_bal_to_loan" not in available:
                available.append("revol_bal_to_loan")

        model_df = df[available + ["loan_default"]].copy()

        default_rate = float(model_df["loan_default"].mean())
        print(
            f"🎯 Default rate: {default_rate:.2%}  |  Features used: {len(available)}"
        )

        return LoadedData(
            df=model_df, feature_cols=available, target_col="loan_default"
        )
