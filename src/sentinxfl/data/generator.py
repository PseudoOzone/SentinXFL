"""
SentinXFL - Synthetic Dataset Generator
========================================

Generates realistic synthetic fraud detection datasets for training
and testing. Uses statistical distributions modelled on real-world
fraud patterns.

Author: Anshuman Bakshi
"""

import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from sentinxfl.core.config import settings
from sentinxfl.core.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────
# Generator Configuration
# ──────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_shopping",
    "electronics", "travel", "entertainment", "pharmacy",
    "clothing", "jewelry", "cash_advance", "wire_transfer",
    "atm_withdrawal", "subscription", "utility", "insurance",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
    "London", "Mumbai", "Tokyo", "Berlin", "Sydney",
]

DEVICE_TYPES = ["mobile", "desktop", "tablet", "pos_terminal", "atm"]
PAYMENT_METHODS = ["credit_card", "debit_card", "bank_transfer", "digital_wallet"]


class FraudDatasetGenerator:
    """
    Generates synthetic fraud detection datasets with configurable
    parameters for fraud ratio, number of rows, and feature complexity.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate(
        self,
        n_samples: int = 10_000,
        fraud_ratio: float = 0.05,
        n_features: int = 20,
        include_pii: bool = False,
        dataset_name: str = "synthetic_fraud",
    ) -> pl.DataFrame:
        """
        Generate a synthetic fraud detection dataset.

        Args:
            n_samples: Number of transactions to generate
            fraud_ratio: Fraction of fraudulent transactions (0.0-1.0)
            n_features: Number of features (10-50)
            include_pii: Whether to include PII columns (for privacy testing)
            dataset_name: Name for the generated dataset

        Returns:
            Polars DataFrame with synthetic fraud data
        """
        n_features = max(10, min(50, n_features))
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud

        log.info(
            f"Generating {n_samples} samples ({n_fraud} fraud, {n_legit} legit) "
            f"with {n_features} features"
        )

        # ── Labels ───────────────────────────────────────
        labels = np.array([0] * n_legit + [1] * n_fraud)
        self.rng.shuffle(labels)
        is_fraud = labels.astype(bool)

        # ── Transaction metadata ─────────────────────────
        data: dict[str, list | np.ndarray] = {}

        # Transaction ID
        data["transaction_id"] = [
            f"TXN_{i:08d}" for i in range(n_samples)
        ]

        # Timestamp — spread over 30 days
        base_time = datetime(2025, 1, 1)
        timestamps = [
            base_time + timedelta(seconds=int(self.rng.uniform(0, 30 * 86400)))
            for _ in range(n_samples)
        ]
        data["timestamp"] = [t.isoformat() for t in timestamps]
        data["trans_hour"] = np.array([t.hour for t in timestamps])
        data["trans_day_of_week"] = np.array([t.weekday() for t in timestamps])
        data["is_weekend"] = (data["trans_day_of_week"] >= 5).astype(int)
        data["is_night"] = ((data["trans_hour"] >= 22) | (data["trans_hour"] <= 5)).astype(int)

        # ── Amount ───────────────────────────────────────
        legit_amounts = self.rng.lognormal(mean=3.5, sigma=1.2, size=n_samples)
        fraud_boost = is_fraud * self.rng.lognormal(mean=5.0, sigma=1.5, size=n_samples)
        amounts = np.where(is_fraud, fraud_boost, legit_amounts)
        amounts = np.round(np.clip(amounts, 0.01, 50000.0), 2)
        data["amount"] = amounts

        # Amount z-score (will be relative to legit mean/std)
        legit_mean = np.mean(legit_amounts)
        legit_std = np.std(legit_amounts) + 1e-8
        data["amount_zscore"] = np.round((amounts - legit_mean) / legit_std, 4)

        # ── Velocity features ────────────────────────────
        data["velocity_1h"] = np.where(
            is_fraud,
            self.rng.poisson(lam=4, size=n_samples),
            self.rng.poisson(lam=1, size=n_samples),
        )
        data["velocity_24h"] = np.where(
            is_fraud,
            self.rng.poisson(lam=12, size=n_samples),
            self.rng.poisson(lam=3, size=n_samples),
        )
        data["velocity_7d"] = data["velocity_24h"] * self.rng.uniform(2, 5, size=n_samples)
        data["velocity_7d"] = np.round(data["velocity_7d"]).astype(int)

        # ── Distance features ────────────────────────────
        data["distance_from_home"] = np.where(
            is_fraud,
            self.rng.exponential(scale=200, size=n_samples),
            self.rng.exponential(scale=15, size=n_samples),
        )
        data["distance_from_home"] = np.round(data["distance_from_home"], 1)

        data["distance_from_last_txn"] = np.where(
            is_fraud,
            self.rng.exponential(scale=150, size=n_samples),
            self.rng.exponential(scale=5, size=n_samples),
        )
        data["distance_from_last_txn"] = np.round(data["distance_from_last_txn"], 1)

        # ── Merchant features ────────────────────────────
        categories = [random.choice(MERCHANT_CATEGORIES) for _ in range(n_samples)]
        # Fraud slightly favors certain categories
        for i in range(n_samples):
            if is_fraud[i] and self.rng.random() > 0.4:
                categories[i] = random.choice([
                    "jewelry", "electronics", "wire_transfer", "cash_advance"
                ])
        data["merchant_category"] = categories

        data["merchant_fraud_rate"] = np.where(
            is_fraud,
            self.rng.beta(a=3, b=10, size=n_samples),
            self.rng.beta(a=1, b=50, size=n_samples),
        )
        data["merchant_fraud_rate"] = np.round(data["merchant_fraud_rate"], 4)

        # ── Device / Channel features ────────────────────
        data["device_type"] = [random.choice(DEVICE_TYPES) for _ in range(n_samples)]
        data["payment_method"] = [random.choice(PAYMENT_METHODS) for _ in range(n_samples)]

        data["is_foreign_transaction"] = np.where(
            is_fraud,
            (self.rng.random(n_samples) > 0.5).astype(int),
            (self.rng.random(n_samples) > 0.9).astype(int),
        )

        # ── Account / Customer features ──────────────────
        data["account_age_days"] = np.where(
            is_fraud,
            self.rng.exponential(scale=60, size=n_samples).astype(int),
            self.rng.exponential(scale=400, size=n_samples).astype(int),
        )

        data["num_previous_frauds"] = np.where(
            is_fraud,
            self.rng.poisson(lam=0.8, size=n_samples),
            self.rng.poisson(lam=0.05, size=n_samples),
        )

        data["credit_score"] = np.where(
            is_fraud,
            self.rng.normal(loc=580, scale=80, size=n_samples),
            self.rng.normal(loc=710, scale=60, size=n_samples),
        )
        data["credit_score"] = np.clip(np.round(data["credit_score"]), 300, 850).astype(int)

        # ── Additional numeric features ──────────────────
        remaining = n_features - len([
            k for k in data
            if k not in ("transaction_id", "timestamp", "merchant_category",
                         "device_type", "payment_method")
        ])
        for i in range(max(0, remaining)):
            noise_legit = self.rng.normal(0, 1, size=n_samples)
            noise_fraud = self.rng.normal(0.5 + i * 0.1, 1.2, size=n_samples)
            data[f"feature_{i + 1}"] = np.round(
                np.where(is_fraud, noise_fraud, noise_legit), 4
            )

        # ── PII columns (optional, for privacy testing) ──
        if include_pii:
            data["customer_name"] = [
                f"{'John' if self.rng.random() > 0.5 else 'Jane'} "
                f"{'Smith' if self.rng.random() > 0.5 else 'Doe'}"
                for _ in range(n_samples)
            ]
            data["email"] = [
                f"user{i}@{'gmail' if self.rng.random() > 0.5 else 'yahoo'}.com"
                for i in range(n_samples)
            ]
            data["phone"] = [
                f"+1{self.rng.integers(2000000000, 9999999999)}"
                for _ in range(n_samples)
            ]
            data["city"] = [random.choice(CITIES) for _ in range(n_samples)]

        # ── Label ────────────────────────────────────────
        data["is_fraud"] = labels

        # ── Build DataFrame ──────────────────────────────
        df = pl.DataFrame(data)

        log.info(
            f"Generated dataset '{dataset_name}': "
            f"{df.shape[0]} rows × {df.shape[1]} cols, "
            f"fraud ratio = {df['is_fraud'].mean():.2%}"
        )
        return df

    def save_dataset(
        self,
        df: pl.DataFrame,
        name: str = "synthetic_fraud",
        format: str = "csv",
    ) -> Path:
        """
        Save generated dataset to the data directory.

        Args:
            df: The generated DataFrame
            name: File name (without extension)
            format: 'csv' or 'parquet'

        Returns:
            Path to saved file
        """
        out_dir = settings.data_dir_abs
        out_dir.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            path = out_dir / f"{name}.parquet"
            df.write_parquet(path)
        else:
            path = out_dir / f"{name}.csv"
            df.write_csv(path)

        log.info(f"Saved dataset to {path} ({path.stat().st_size / 1024:.1f} KB)")
        return path

    def get_dataset_info(self, df: pl.DataFrame) -> dict:
        """Get summary statistics for a generated dataset."""
        numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
        categorical_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]

        fraud_count = int(df["is_fraud"].sum())
        total = len(df)

        return {
            "total_rows": total,
            "total_columns": len(df.columns),
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "fraud_count": fraud_count,
            "legitimate_count": total - fraud_count,
            "fraud_ratio": round(fraud_count / total, 4) if total > 0 else 0,
            "memory_mb": round(df.estimated_size("mb"), 2),
            "columns": df.columns,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "sample_rows": df.head(5).to_dicts(),
        }
