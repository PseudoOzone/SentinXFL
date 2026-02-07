"""
SentinXFL Data Splitter
========================

Stratified data splitting for imbalanced fraud detection datasets.
Supports FL-style client partitioning.

Author: Anshuman Bakshi
"""

from typing import Optional, Tuple, List
import hashlib

import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from sentinxfl.core.config import settings
from sentinxfl.core.logging import get_logger
from sentinxfl.data.schemas import DataSplitInfo

log = get_logger(__name__)


class DataSplitter:
    """
    Stratified data splitting for imbalanced classification.

    Supports:
    - Standard train/val/test split with stratification
    - K-fold cross-validation
    - FL client partitioning (IID and non-IID)
    """

    def __init__(
        self,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None,
        random_seed: int = None,
    ):
        """Initialize the data splitter.

        Args:
            train_ratio: Fraction for training set. Defaults to settings.
            val_ratio: Fraction for validation set. Defaults to settings.
            test_ratio: Fraction for test set. Defaults to settings.
            random_seed: Random seed for reproducibility.
        """
        self.train_ratio = train_ratio or settings.ml_train_split
        self.val_ratio = val_ratio or settings.ml_val_split
        self.test_ratio = test_ratio or settings.ml_test_split
        self.random_seed = random_seed or settings.ml_random_seed

        # Validate ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        log.info(
            f"DataSplitter initialized: train={self.train_ratio}, "
            f"val={self.val_ratio}, test={self.test_ratio}"
        )

    def split(
        self,
        df: pl.DataFrame,
        label_col: str = "is_fraud",
        stratify: bool = True,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split DataFrame into train/val/test sets.

        Args:
            df: Input DataFrame
            label_col: Name of the label column for stratification
            stratify: Whether to use stratified splitting

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        log.info(f"Splitting {len(df)} rows into train/val/test...")

        # Get indices
        indices = np.arange(len(df))
        labels = df[label_col].to_numpy() if stratify and label_col in df.columns else None

        # First split: separate test set
        test_size = self.test_ratio
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=labels,
        )

        # Second split: separate train and val from remaining
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_val_labels = labels[train_val_idx] if labels is not None else None

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=train_val_labels,
        )

        # Create DataFrames from indices
        train_df = df[train_idx.tolist()]
        val_df = df[val_idx.tolist()]
        test_df = df[test_idx.tolist()]

        # Log split info
        self._log_split_info(train_df, val_df, test_df, label_col)

        return train_df, val_df, test_df

    def _log_split_info(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        label_col: str,
    ) -> DataSplitInfo:
        """Log and return split information."""
        def get_fraud_ratio(df: pl.DataFrame) -> float:
            if label_col not in df.columns:
                return 0.0
            return df.filter(pl.col(label_col)).shape[0] / len(df) if len(df) > 0 else 0.0

        info = DataSplitInfo(
            total_rows=len(train_df) + len(val_df) + len(test_df),
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            train_fraud_ratio=get_fraud_ratio(train_df),
            val_fraud_ratio=get_fraud_ratio(val_df),
            test_fraud_ratio=get_fraud_ratio(test_df),
            stratified=True,
            random_seed=self.random_seed,
        )

        log.info(
            f"Split complete: train={info.train_rows} ({info.train_fraud_ratio:.2%} fraud), "
            f"val={info.val_rows} ({info.val_fraud_ratio:.2%} fraud), "
            f"test={info.test_rows} ({info.test_fraud_ratio:.2%} fraud)"
        )

        return info

    def kfold_split(
        self,
        df: pl.DataFrame,
        n_splits: int = 5,
        label_col: str = "is_fraud",
    ) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
        """Generate K-fold cross-validation splits.

        Args:
            df: Input DataFrame
            n_splits: Number of folds
            label_col: Name of label column for stratification

        Returns:
            List of (train_df, val_df) tuples
        """
        log.info(f"Creating {n_splits}-fold stratified splits...")

        labels = df[label_col].to_numpy() if label_col in df.columns else None
        indices = np.arange(len(df))

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_seed,
        )

        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
            train_df = df[train_idx.tolist()]
            val_df = df[val_idx.tolist()]
            splits.append((train_df, val_df))
            log.debug(
                f"Fold {fold_idx + 1}: train={len(train_df)}, val={len(val_df)}"
            )

        return splits

    def partition_for_fl(
        self,
        df: pl.DataFrame,
        num_clients: int,
        partition_type: str = "iid",
        label_col: str = "is_fraud",
        alpha: float = 0.5,
    ) -> List[pl.DataFrame]:
        """Partition data for federated learning clients.

        Args:
            df: Input DataFrame
            num_clients: Number of FL clients
            partition_type: 'iid' for uniform, 'noniid' for Dirichlet distribution
            label_col: Name of label column
            alpha: Dirichlet concentration parameter (lower = more non-IID)

        Returns:
            List of DataFrames, one per client
        """
        log.info(
            f"Partitioning {len(df)} rows for {num_clients} FL clients "
            f"(type={partition_type}, alpha={alpha})"
        )

        if partition_type == "iid":
            return self._iid_partition(df, num_clients)
        elif partition_type == "noniid":
            return self._noniid_partition(df, num_clients, label_col, alpha)
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")

    def _iid_partition(
        self,
        df: pl.DataFrame,
        num_clients: int,
    ) -> List[pl.DataFrame]:
        """Create IID (uniformly distributed) partitions."""
        # Shuffle and split evenly
        shuffled = df.sample(fraction=1.0, seed=self.random_seed)
        indices = np.arange(len(shuffled))
        partition_size = len(shuffled) // num_clients

        partitions = []
        for i in range(num_clients):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < num_clients - 1 else len(shuffled)
            partition_indices = indices[start_idx:end_idx].tolist()
            partitions.append(shuffled[partition_indices])

        for i, p in enumerate(partitions):
            log.debug(f"Client {i}: {len(p)} samples")

        return partitions

    def _noniid_partition(
        self,
        df: pl.DataFrame,
        num_clients: int,
        label_col: str,
        alpha: float,
    ) -> List[pl.DataFrame]:
        """Create non-IID partitions using Dirichlet distribution.

        Lower alpha = more heterogeneous (non-IID)
        Higher alpha = more homogeneous (closer to IID)
        """
        np.random.seed(self.random_seed)

        labels = df[label_col].to_numpy()
        unique_labels = np.unique(labels)

        # Get indices for each label
        label_indices = {
            label: np.where(labels == label)[0] for label in unique_labels
        }

        # Allocate indices to clients using Dirichlet distribution
        client_indices = [[] for _ in range(num_clients)]

        for label, indices in label_indices.items():
            np.random.shuffle(indices)

            # Sample from Dirichlet
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)

            # Assign to clients
            prev_idx = 0
            for client_id, end_idx in enumerate(proportions):
                client_indices[client_id].extend(indices[prev_idx:end_idx].tolist())
                prev_idx = end_idx

        # Create DataFrames
        partitions = []
        for i, indices in enumerate(client_indices):
            partition = df[indices]
            partitions.append(partition)
            fraud_ratio = partition.filter(pl.col(label_col)).shape[0] / len(partition) if len(partition) > 0 else 0
            log.debug(
                f"Client {i}: {len(partition)} samples, {fraud_ratio:.2%} fraud"
            )

        return partitions

    @staticmethod
    def get_class_weights(df: pl.DataFrame, label_col: str = "is_fraud") -> dict:
        """Calculate class weights for imbalanced data.

        Uses 'balanced' strategy: n_samples / (n_classes * np.bincount(y))

        Args:
            df: Input DataFrame
            label_col: Name of label column

        Returns:
            Dictionary mapping class label to weight
        """
        labels = df[label_col].to_numpy()
        unique, counts = np.unique(labels, return_counts=True)

        n_samples = len(labels)
        n_classes = len(unique)

        weights = {}
        for label, count in zip(unique, counts):
            weights[label] = n_samples / (n_classes * count)

        log.info(f"Class weights: {weights}")
        return weights

    @staticmethod
    def undersample_majority(
        df: pl.DataFrame,
        label_col: str = "is_fraud",
        ratio: float = 1.0,
        seed: int = None,
    ) -> pl.DataFrame:
        """Undersample majority class to reduce imbalance.

        Args:
            df: Input DataFrame
            label_col: Name of label column
            ratio: Desired ratio of majority:minority (1.0 = balanced)
            seed: Random seed

        Returns:
            Undersampled DataFrame
        """
        seed = seed or settings.ml_random_seed

        minority = df.filter(pl.col(label_col))
        majority = df.filter(~pl.col(label_col))

        minority_count = len(minority)
        target_majority = int(minority_count * ratio)

        if target_majority >= len(majority):
            log.info("No undersampling needed, returning original data")
            return df

        majority_sampled = majority.sample(n=target_majority, seed=seed)
        result = pl.concat([minority, majority_sampled])

        log.info(
            f"Undersampled from {len(df)} to {len(result)} rows "
            f"(minority={minority_count}, majority={target_majority})"
        )

        return result.sample(fraction=1.0, seed=seed)  # Shuffle
