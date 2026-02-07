"""
SentinXFL Data Loader
======================

High-performance data loading using DuckDB and Polars.
Handles all three datasets: Bank Account Fraud, Credit Card, and PaySim.

Author: Anshuman Bakshi
"""

from pathlib import Path
from typing import Optional
import hashlib

import duckdb
import polars as pl

from sentinxfl.core.config import settings
from sentinxfl.core.logging import get_logger
from sentinxfl.data.schemas import DatasetType, DatasetStats, UnifiedDatasetStats

log = get_logger(__name__)


class DataLoader:
    """
    High-performance data loader using DuckDB for analytical queries
    and Polars for DataFrame operations.

    Supports:
    - Bank Account Fraud Dataset (6M rows, 6 files)
    - Credit Card Fraud Dataset (284K rows)
    - PaySim Dataset (6.3M rows)
    """

    # Column mappings for each dataset
    COLUMN_MAPPINGS = {
        DatasetType.BANK_ACCOUNT_FRAUD: {
            "fraud_bool": "is_fraud",
            "income": "income",
            "name_email_similarity": "name_email_similarity",
            "current_address_months_count": "address_months",
            "customer_age": "customer_age",
            "days_since_request": "days_since_request",
            "intended_balcon_amount": "intended_balance_amount",
            "payment_type": "payment_type",
            "zip_count_4w": "zip_activity_4w",
            "velocity_6h": "velocity_6h",
            "velocity_24h": "velocity_24h",
            "velocity_4w": "velocity_4w",
            "bank_branch_count_8w": "bank_branch_activity_8w",
            "date_of_birth_distinct_emails_4w": "dob_distinct_emails_4w",
            "employment_status": "employment_status",
            "credit_risk_score": "credit_risk_score",
            "email_is_free": "email_is_free",
            "housing_status": "housing_status",
            "phone_home_valid": "phone_home_valid",
            "phone_mobile_valid": "phone_mobile_valid",
            "bank_months_count": "bank_months",
            "has_other_cards": "has_other_cards",
            "proposed_credit_limit": "proposed_credit_limit",
            "foreign_request": "foreign_request",
            "source": "source_channel",
            "session_length_in_minutes": "session_length_minutes",
            "device_os": "device_os",
            "keep_alive_session": "keep_alive_session",
            "device_distinct_emails_8w": "device_distinct_emails_8w",
            "device_fraud_count": "device_fraud_count",
            "month": "month",
        },
        DatasetType.CREDIT_CARD_FRAUD: {
            "Time": "time_seconds",
            "Amount": "amount",
            "Class": "is_fraud",
            # V1-V28 are PCA components - keep as-is
        },
        DatasetType.PAYSIM: {
            "step": "step",
            "type": "transaction_type",
            "amount": "amount",
            "nameOrig": "account_orig",  # Will be hashed
            "oldbalanceOrg": "balance_orig_before",
            "newbalanceOrig": "balance_orig_after",
            "nameDest": "account_dest",  # Will be hashed
            "oldbalanceDest": "balance_dest_before",
            "newbalanceDest": "balance_dest_after",
            "isFraud": "is_fraud",
            "isFlaggedFraud": "is_flagged_fraud",
        },
    }

    # Files for Bank Account Fraud dataset
    BANK_FRAUD_FILES = [
        "Base.csv",
        "Variant I.csv",
        "Variant II.csv",
        "Variant III.csv",
        "Variant IV.csv",
        "Variant V.csv",
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the data loader.

        Args:
            data_dir: Path to data directory. Defaults to settings.data_dir_abs.
        """
        self.data_dir = data_dir or settings.data_dir_abs
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._dataset_stats: dict[DatasetType, DatasetStats] = {}

        log.info(f"DataLoader initialized with data_dir: {self.data_dir}")

    def connect(self) -> "DataLoader":
        """Connect to DuckDB database."""
        db_path = settings.duckdb_path_abs
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(
            str(db_path),
            config={
                "memory_limit": settings.duckdb_memory_limit,
                "threads": settings.duckdb_threads,
            },
        )
        log.info(f"Connected to DuckDB at {db_path}")
        return self

    def close(self) -> None:
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            log.info("DuckDB connection closed")

    def __enter__(self) -> "DataLoader":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        """Ensure database connection is active."""
        if self.conn is None:
            self.connect()

    @staticmethod
    def _hash_identifier(value: str) -> str:
        """Hash an identifier using SHA-256 truncated to 16 chars."""
        if not value or value == "":
            return ""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def load_bank_account_fraud(self, sample_frac: Optional[float] = None) -> pl.DataFrame:
        """Load Bank Account Fraud dataset (all 6 variant files).

        Args:
            sample_frac: If provided, sample this fraction of data (0.0-1.0)

        Returns:
            Polars DataFrame with standardized columns
        """
        log.info("Loading Bank Account Fraud dataset...")
        dfs = []

        for filename in self.BANK_FRAUD_FILES:
            filepath = self.data_dir / filename
            if not filepath.exists():
                log.warning(f"File not found: {filepath}")
                continue

            df = pl.read_csv(filepath, infer_schema_length=10000)
            # Add source variant column
            variant = filename.replace(".csv", "").replace("Variant ", "V")
            df = df.with_columns(pl.lit(variant).alias("variant"))
            dfs.append(df)
            log.debug(f"Loaded {filename}: {len(df)} rows")

        if not dfs:
            raise FileNotFoundError("No Bank Account Fraud files found")

        # Concatenate all variants (use diagonal_relaxed for schema mismatches)
        combined = pl.concat(dfs, how="diagonal_relaxed")

        # Rename columns
        mapping = self.COLUMN_MAPPINGS[DatasetType.BANK_ACCOUNT_FRAUD]
        for old_col, new_col in mapping.items():
            if old_col in combined.columns:
                combined = combined.rename({old_col: new_col})

        # Add dataset source
        combined = combined.with_columns(
            pl.lit(DatasetType.BANK_ACCOUNT_FRAUD.value).alias("dataset_source")
        )

        # Convert fraud column to boolean
        if "is_fraud" in combined.columns:
            combined = combined.with_columns(
                pl.col("is_fraud").cast(pl.Boolean)
            )

        # Sample if requested
        if sample_frac and 0 < sample_frac < 1:
            combined = combined.sample(fraction=sample_frac, seed=settings.ml_random_seed)
            log.info(f"Sampled to {len(combined)} rows ({sample_frac*100:.1f}%)")

        # Calculate stats
        self._dataset_stats[DatasetType.BANK_ACCOUNT_FRAUD] = DatasetStats(
            name="Bank Account Fraud",
            dataset_type=DatasetType.BANK_ACCOUNT_FRAUD,
            total_rows=len(combined),
            fraud_count=combined.filter(pl.col("is_fraud")).shape[0],
            legitimate_count=combined.filter(~pl.col("is_fraud")).shape[0],
            fraud_ratio=combined.filter(pl.col("is_fraud")).shape[0] / len(combined),
            num_features=len(combined.columns),
            memory_mb=combined.estimated_size("mb"),
            file_path=str(self.data_dir),
        )

        log.info(
            f"Bank Account Fraud loaded: {len(combined)} rows, "
            f"{self._dataset_stats[DatasetType.BANK_ACCOUNT_FRAUD].fraud_ratio:.2%} fraud"
        )
        return combined

    def load_credit_card_fraud(self, sample_frac: Optional[float] = None) -> pl.DataFrame:
        """Load Credit Card Fraud dataset.

        Args:
            sample_frac: If provided, sample this fraction of data (0.0-1.0)

        Returns:
            Polars DataFrame with standardized columns
        """
        log.info("Loading Credit Card Fraud dataset...")
        filepath = self.data_dir / "creditcard.csv"

        if not filepath.exists():
            raise FileNotFoundError(f"Credit Card dataset not found: {filepath}")

        # Time and Amount can have scientific notation, read as float
        df = pl.read_csv(
            filepath, 
            infer_schema_length=10000,
            schema_overrides={"Time": pl.Float64, "Amount": pl.Float64}
        )

        # Rename columns
        mapping = self.COLUMN_MAPPINGS[DatasetType.CREDIT_CARD_FRAUD]
        for old_col, new_col in mapping.items():
            if old_col in df.columns:
                df = df.rename({old_col: new_col})

        # Add dataset source
        df = df.with_columns(
            pl.lit(DatasetType.CREDIT_CARD_FRAUD.value).alias("dataset_source")
        )

        # Convert fraud column to boolean
        if "is_fraud" in df.columns:
            df = df.with_columns(
                pl.col("is_fraud").cast(pl.Boolean)
            )

        # Sample if requested
        if sample_frac and 0 < sample_frac < 1:
            df = df.sample(fraction=sample_frac, seed=settings.ml_random_seed)
            log.info(f"Sampled to {len(df)} rows ({sample_frac*100:.1f}%)")

        # Calculate stats
        self._dataset_stats[DatasetType.CREDIT_CARD_FRAUD] = DatasetStats(
            name="Credit Card Fraud",
            dataset_type=DatasetType.CREDIT_CARD_FRAUD,
            total_rows=len(df),
            fraud_count=df.filter(pl.col("is_fraud")).shape[0],
            legitimate_count=df.filter(~pl.col("is_fraud")).shape[0],
            fraud_ratio=df.filter(pl.col("is_fraud")).shape[0] / len(df) if len(df) > 0 else 0,
            num_features=len(df.columns),
            memory_mb=df.estimated_size("mb"),
            file_path=str(filepath),
        )

        log.info(
            f"Credit Card Fraud loaded: {len(df)} rows, "
            f"{self._dataset_stats[DatasetType.CREDIT_CARD_FRAUD].fraud_ratio:.2%} fraud"
        )
        return df

    def load_paysim(self, sample_frac: Optional[float] = None) -> pl.DataFrame:
        """Load PaySim dataset.

        Args:
            sample_frac: If provided, sample this fraction of data (0.0-1.0)

        Returns:
            Polars DataFrame with standardized columns
        """
        log.info("Loading PaySim dataset...")

        # Find PaySim file (may have different names)
        paysim_patterns = ["PS_*.csv", "paysim*.csv"]
        filepath = None

        for pattern in paysim_patterns:
            matches = list(self.data_dir.glob(pattern))
            if matches:
                filepath = matches[0]
                break

        if filepath is None or not filepath.exists():
            raise FileNotFoundError(f"PaySim dataset not found in {self.data_dir}")

        df = pl.read_csv(filepath, infer_schema_length=10000)

        # Rename columns
        mapping = self.COLUMN_MAPPINGS[DatasetType.PAYSIM]
        for old_col, new_col in mapping.items():
            if old_col in df.columns:
                df = df.rename({old_col: new_col})

        # Hash account identifiers (PII protection)
        if "account_orig" in df.columns:
            df = df.with_columns(
                pl.col("account_orig").map_elements(
                    self._hash_identifier, return_dtype=pl.Utf8
                ).alias("account_orig_hash")
            ).drop("account_orig")

        if "account_dest" in df.columns:
            df = df.with_columns(
                pl.col("account_dest").map_elements(
                    self._hash_identifier, return_dtype=pl.Utf8
                ).alias("account_dest_hash")
            ).drop("account_dest")

        # Add dataset source
        df = df.with_columns(
            pl.lit(DatasetType.PAYSIM.value).alias("dataset_source")
        )

        # Convert fraud column to boolean
        if "is_fraud" in df.columns:
            df = df.with_columns(
                pl.col("is_fraud").cast(pl.Boolean)
            )

        # Sample if requested
        if sample_frac and 0 < sample_frac < 1:
            df = df.sample(fraction=sample_frac, seed=settings.ml_random_seed)
            log.info(f"Sampled to {len(df)} rows ({sample_frac*100:.1f}%)")

        # Calculate stats
        self._dataset_stats[DatasetType.PAYSIM] = DatasetStats(
            name="PaySim",
            dataset_type=DatasetType.PAYSIM,
            total_rows=len(df),
            fraud_count=df.filter(pl.col("is_fraud")).shape[0],
            legitimate_count=df.filter(~pl.col("is_fraud")).shape[0],
            fraud_ratio=df.filter(pl.col("is_fraud")).shape[0] / len(df) if len(df) > 0 else 0,
            num_features=len(df.columns),
            memory_mb=df.estimated_size("mb"),
            file_path=str(filepath),
        )

        log.info(
            f"PaySim loaded: {len(df)} rows, "
            f"{self._dataset_stats[DatasetType.PAYSIM].fraud_ratio:.2%} fraud"
        )
        return df

    def load_all_datasets(
        self,
        sample_frac: Optional[float] = None,
        datasets: Optional[list[DatasetType]] = None,
    ) -> dict[DatasetType, pl.DataFrame]:
        """Load all (or specified) datasets.

        Args:
            sample_frac: If provided, sample this fraction of each dataset
            datasets: List of datasets to load. Defaults to all.

        Returns:
            Dictionary mapping DatasetType to DataFrame
        """
        if datasets is None:
            datasets = [
                DatasetType.BANK_ACCOUNT_FRAUD,
                DatasetType.CREDIT_CARD_FRAUD,
                DatasetType.PAYSIM,
            ]

        loaders = {
            DatasetType.BANK_ACCOUNT_FRAUD: self.load_bank_account_fraud,
            DatasetType.CREDIT_CARD_FRAUD: self.load_credit_card_fraud,
            DatasetType.PAYSIM: self.load_paysim,
        }

        result = {}
        for dtype in datasets:
            try:
                result[dtype] = loaders[dtype](sample_frac=sample_frac)
            except FileNotFoundError as e:
                log.error(f"Failed to load {dtype.value}: {e}")

        return result

    def get_unified_stats(self) -> UnifiedDatasetStats:
        """Get statistics for all loaded datasets."""
        stats_list = list(self._dataset_stats.values())

        if not stats_list:
            raise ValueError("No datasets loaded yet")

        total_rows = sum(s.total_rows for s in stats_list)
        total_fraud = sum(s.fraud_count for s in stats_list)
        total_legitimate = sum(s.legitimate_count for s in stats_list)
        total_memory = sum(s.memory_mb for s in stats_list)

        return UnifiedDatasetStats(
            total_rows=total_rows,
            total_fraud=total_fraud,
            total_legitimate=total_legitimate,
            overall_fraud_ratio=total_fraud / total_rows if total_rows > 0 else 0,
            datasets=stats_list,
            combined_features=max(s.num_features for s in stats_list),
            total_memory_mb=total_memory,
        )

    def register_in_duckdb(
        self,
        df: pl.DataFrame,
        table_name: str,
    ) -> None:
        """Register a Polars DataFrame as a DuckDB table.

        Args:
            df: Polars DataFrame to register
            table_name: Name for the DuckDB table
        """
        self._ensure_connected()
        # Convert to Arrow for zero-copy transfer to DuckDB
        arrow_table = df.to_arrow()
        self.conn.register(table_name, arrow_table)
        log.info(f"Registered table '{table_name}' with {len(df)} rows")

    def query(self, sql: str) -> pl.DataFrame:
        """Execute SQL query and return results as Polars DataFrame.

        Args:
            sql: SQL query string

        Returns:
            Polars DataFrame with query results
        """
        self._ensure_connected()
        result = self.conn.execute(sql).pl()
        return result

    def save_processed(
        self,
        df: pl.DataFrame,
        name: str,
        format: str = "parquet",
    ) -> Path:
        """Save processed DataFrame to file.

        Args:
            df: DataFrame to save
            name: Output filename (without extension)
            format: Output format ('parquet' or 'csv')

        Returns:
            Path to saved file
        """
        output_dir = settings.processed_dir_abs
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            output_path = output_dir / f"{name}.parquet"
            df.write_parquet(output_path, compression="zstd")
        else:
            output_path = output_dir / f"{name}.csv"
            df.write_csv(output_path)

        log.info(f"Saved processed data to {output_path}")
        return output_path
