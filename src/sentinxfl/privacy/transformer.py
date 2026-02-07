"""
SentinXFL PII Transformer
==========================

Data transformation engine for PII sanitization.

Supported transformations:
- Hashing (SHA-256, keyed HMAC)
- Masking (partial redaction)
- Generalization (binning, k-anonymity)
- Pseudonymization (consistent mapping)
- Redaction (removal)

This is a PATENT-CORE component.

Author: Anshuman Bakshi
Copyright (c) 2026. All rights reserved.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, Union

import polars as pl

from sentinxfl.core.config import settings
from sentinxfl.core.logging import get_logger
from sentinxfl.privacy.detector import PIIDetectionResult, PIIMatch, PIISensitivity, PIIType

log = get_logger(__name__)


class TransformationType(str, Enum):
    """Types of PII transformations."""
    HASH = "hash"                  # One-way cryptographic hash
    HMAC = "hmac"                  # Keyed hash (deterministic)
    MASK = "mask"                  # Partial redaction (* characters)
    GENERALIZE = "generalize"     # Binning/bucketing
    PSEUDONYMIZE = "pseudonymize" # Consistent fake replacement
    REDACT = "redact"             # Complete removal
    TOKENIZE = "tokenize"         # Format-preserving tokenization
    NOISE = "noise"               # Add differential privacy noise
    SUPPRESS = "suppress"         # Replace with null/special value


@dataclass
class TransformationResult:
    """Result of a transformation operation."""
    column_name: str
    transformation_type: TransformationType
    original_dtype: str
    rows_transformed: int
    null_count: int
    success: bool
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TransformationPlan:
    """Plan for transforming a dataset."""
    columns: dict[str, TransformationType]
    hash_key: Optional[str] = None
    generalization_params: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PIITransformer:
    """
    PII transformation engine with multiple sanitization strategies.
    
    Implements the transformation layer of the 5-Gate Pipeline.
    """
    
    # Default transformation by PII type
    DEFAULT_TRANSFORMATIONS = {
        PIIType.NAME: TransformationType.HASH,
        PIIType.EMAIL: TransformationType.HASH,
        PIIType.PHONE: TransformationType.MASK,
        PIIType.SSN: TransformationType.REDACT,
        PIIType.NATIONAL_ID: TransformationType.REDACT,
        PIIType.CREDIT_CARD: TransformationType.REDACT,
        PIIType.BANK_ACCOUNT: TransformationType.HASH,
        PIIType.ADDRESS: TransformationType.GENERALIZE,
        PIIType.ZIP_CODE: TransformationType.GENERALIZE,
        PIIType.IP_ADDRESS: TransformationType.MASK,
        PIIType.DEVICE_ID: TransformationType.HASH,
        PIIType.DATE_OF_BIRTH: TransformationType.GENERALIZE,
        PIIType.AGE: TransformationType.GENERALIZE,
        PIIType.GENDER: TransformationType.SUPPRESS,  # High re-id risk when combined
        PIIType.INCOME: TransformationType.GENERALIZE,
        PIIType.GPS_COORDINATES: TransformationType.GENERALIZE,
        PIIType.USERNAME: TransformationType.HASH,
        PIIType.PASSWORD: TransformationType.REDACT,
    }
    
    # Age bins for generalization
    AGE_BINS = [0, 18, 25, 35, 45, 55, 65, 100]
    AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    
    # Income bins (in USD)
    INCOME_BINS = [0, 25000, 50000, 75000, 100000, 150000, 200000, float("inf")]
    INCOME_LABELS = ["<25K", "25K-50K", "50K-75K", "75K-100K", "100K-150K", "150K-200K", "200K+"]
    
    def __init__(self, hash_key: Optional[str] = None):
        """Initialize PII Transformer.
        
        Args:
            hash_key: Secret key for HMAC operations. Auto-generated if not provided.
        """
        self.hash_key = hash_key or secrets.token_urlsafe(32)
        self._transformation_log: list[TransformationResult] = []
        self._pseudonym_maps: dict[str, dict] = {}  # For consistent pseudonymization
        
        log.info("PIITransformer initialized")
    
    def transform(
        self,
        df: pl.DataFrame,
        detection_result: PIIDetectionResult,
        custom_transforms: Optional[dict[str, TransformationType]] = None,
    ) -> tuple[pl.DataFrame, list[TransformationResult]]:
        """Transform detected PII columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            detection_result: Result from PIIDetector.detect()
            custom_transforms: Optional override for transformation types
            
        Returns:
            Tuple of (transformed DataFrame, list of transformation results)
        """
        log.info(f"Starting PII transformation on {len(detection_result.matches)} columns")
        
        results: list[TransformationResult] = []
        transformed_df = df.clone()
        
        custom_transforms = custom_transforms or {}
        
        for match in detection_result.matches:
            col = match.column_name
            
            if col not in transformed_df.columns:
                log.warning(f"Column {col} not found in DataFrame, skipping")
                continue
            
            # Determine transformation type
            if col in custom_transforms:
                transform_type = custom_transforms[col]
            else:
                transform_type = self.DEFAULT_TRANSFORMATIONS.get(
                    match.pii_type, TransformationType.HASH
                )
            
            # Apply transformation
            try:
                transformed_df, result = self._apply_transformation(
                    transformed_df, col, transform_type, match
                )
                results.append(result)
                log.debug(f"Transformed {col}: {transform_type.value}")
            except Exception as e:
                log.error(f"Failed to transform {col}: {e}")
                results.append(TransformationResult(
                    column_name=col,
                    transformation_type=transform_type,
                    original_dtype=str(df[col].dtype),
                    rows_transformed=0,
                    null_count=0,
                    success=False,
                    error_message=str(e),
                ))
        
        self._transformation_log.extend(results)
        
        successful = sum(1 for r in results if r.success)
        log.info(f"Transformation complete: {successful}/{len(results)} columns transformed")
        
        return transformed_df, results
    
    def _apply_transformation(
        self,
        df: pl.DataFrame,
        column: str,
        transform_type: TransformationType,
        match: PIIMatch,
    ) -> tuple[pl.DataFrame, TransformationResult]:
        """Apply a specific transformation to a column."""
        original_dtype = str(df[column].dtype)
        null_count = df[column].null_count()
        
        transformers = {
            TransformationType.HASH: self._transform_hash,
            TransformationType.HMAC: self._transform_hmac,
            TransformationType.MASK: self._transform_mask,
            TransformationType.GENERALIZE: self._transform_generalize,
            TransformationType.PSEUDONYMIZE: self._transform_pseudonymize,
            TransformationType.REDACT: self._transform_redact,
            TransformationType.TOKENIZE: self._transform_tokenize,
            TransformationType.SUPPRESS: self._transform_suppress,
            TransformationType.NOISE: self._transform_noise,
        }
        
        transformer = transformers.get(transform_type, self._transform_hash)
        transformed_df = transformer(df, column, match)
        
        return transformed_df, TransformationResult(
            column_name=column,
            transformation_type=transform_type,
            original_dtype=original_dtype,
            rows_transformed=len(df) - null_count,
            null_count=null_count,
            success=True,
            metadata={"pii_type": match.pii_type.value},
        )
    
    def _transform_hash(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Apply SHA-256 hash transformation."""
        def hash_value(val):
            if val is None or str(val) == "":
                return None
            return hashlib.sha256(str(val).encode()).hexdigest()[:16]
        
        return df.with_columns(
            pl.col(column).map_elements(hash_value, return_dtype=pl.Utf8).alias(column)
        )
    
    def _transform_hmac(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Apply keyed HMAC hash (deterministic, consistent across runs)."""
        def hmac_value(val):
            if val is None or str(val) == "":
                return None
            return hmac.new(
                self.hash_key.encode(),
                str(val).encode(),
                hashlib.sha256
            ).hexdigest()[:16]
        
        return df.with_columns(
            pl.col(column).map_elements(hmac_value, return_dtype=pl.Utf8).alias(column)
        )
    
    def _transform_mask(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Apply partial masking (show first/last few characters)."""
        def mask_value(val):
            if val is None:
                return None
            s = str(val)
            if len(s) <= 4:
                return "*" * len(s)
            # Show first 2 and last 2 characters
            return s[:2] + "*" * (len(s) - 4) + s[-2:]
        
        return df.with_columns(
            pl.col(column).map_elements(mask_value, return_dtype=pl.Utf8).alias(column)
        )
    
    def _transform_generalize(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Apply generalization (binning/bucketing)."""
        col_dtype = df[column].dtype
        
        # Age generalization
        if match.pii_type == PIIType.AGE or "age" in column.lower():
            return df.with_columns(
                pl.col(column).cut(
                    self.AGE_BINS[1:-1],
                    labels=self.AGE_LABELS
                ).alias(column)
            )
        
        # Income generalization
        if match.pii_type == PIIType.INCOME or "income" in column.lower():
            return df.with_columns(
                pl.col(column).cut(
                    self.INCOME_BINS[1:-1],
                    labels=self.INCOME_LABELS
                ).alias(column)
            )
        
        # ZIP code generalization (keep first 3 digits)
        if match.pii_type == PIIType.ZIP_CODE:
            def generalize_zip(val):
                if val is None:
                    return None
                s = str(val)
                return s[:3] + "XX" if len(s) >= 3 else s
            
            return df.with_columns(
                pl.col(column).map_elements(generalize_zip, return_dtype=pl.Utf8).alias(column)
            )
        
        # Date of birth -> Age range
        if match.pii_type == PIIType.DATE_OF_BIRTH:
            def generalize_dob(val):
                if val is None:
                    return None
                # Just extract year decade
                try:
                    from datetime import datetime
                    if isinstance(val, str):
                        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
                            try:
                                dt = datetime.strptime(val, fmt)
                                decade = (datetime.now().year - dt.year) // 10 * 10
                                return f"{decade}s"
                            except ValueError:
                                continue
                    return "unknown"
                except Exception:
                    return "unknown"
            
            return df.with_columns(
                pl.col(column).map_elements(generalize_dob, return_dtype=pl.Utf8).alias(column)
            )
        
        # GPS coordinates (reduce precision to ~10km)
        if match.pii_type == PIIType.GPS_COORDINATES:
            def generalize_coords(val):
                if val is None:
                    return None
                try:
                    parts = str(val).split(",")
                    if len(parts) == 2:
                        lat = round(float(parts[0]), 1)
                        lon = round(float(parts[1]), 1)
                        return f"{lat},{lon}"
                except Exception:
                    pass
                return None
            
            return df.with_columns(
                pl.col(column).map_elements(generalize_coords, return_dtype=pl.Utf8).alias(column)
            )
        
        # Default: bucket numeric values
        if col_dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            # Create 10 equal-width bins
            return df.with_columns(
                pl.col(column).qcut(10, labels=[f"bin_{i}" for i in range(10)]).alias(column)
            )
        
        # For string columns, keep first portion
        def generalize_string(val):
            if val is None:
                return None
            s = str(val)
            return s[:len(s)//2] + "..." if len(s) > 4 else s
        
        return df.with_columns(
            pl.col(column).map_elements(generalize_string, return_dtype=pl.Utf8).alias(column)
        )
    
    def _transform_pseudonymize(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Apply consistent pseudonymization (same input -> same output)."""
        if column not in self._pseudonym_maps:
            self._pseudonym_maps[column] = {}
        
        pseudo_map = self._pseudonym_maps[column]
        counter = len(pseudo_map)
        
        def pseudonymize(val):
            nonlocal counter
            if val is None:
                return None
            key = str(val)
            if key not in pseudo_map:
                pseudo_map[key] = f"PSEUDO_{column.upper()}_{counter:06d}"
                counter += 1
            return pseudo_map[key]
        
        return df.with_columns(
            pl.col(column).map_elements(pseudonymize, return_dtype=pl.Utf8).alias(column)
        )
    
    def _transform_redact(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Complete redaction (remove column or replace with placeholder)."""
        # Replace all values with [REDACTED]
        return df.with_columns(
            pl.lit("[REDACTED]").alias(column)
        )
    
    def _transform_tokenize(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Format-preserving tokenization."""
        def tokenize(val):
            if val is None:
                return None
            s = str(val)
            # Preserve format but replace with random characters
            import random
            result = ""
            for c in s:
                if c.isdigit():
                    result += str(random.randint(0, 9))
                elif c.isalpha():
                    result += random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                else:
                    result += c
            return result
        
        return df.with_columns(
            pl.col(column).map_elements(tokenize, return_dtype=pl.Utf8).alias(column)
        )
    
    def _transform_suppress(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Suppress column (set to null/special value)."""
        return df.with_columns(
            pl.lit(None).alias(column)
        )
    
    def _transform_noise(self, df: pl.DataFrame, column: str, match: PIIMatch) -> pl.DataFrame:
        """Add Laplacian noise for differential privacy."""
        import numpy as np
        
        col_dtype = df[column].dtype
        if col_dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            # Fall back to hash for non-numeric
            return self._transform_hash(df, column, match)
        
        # Calculate sensitivity based on data range
        col_values = df[column].drop_nulls()
        if len(col_values) == 0:
            return df
        
        sensitivity = float(col_values.max() - col_values.min()) / 100  # Rough estimate
        scale = sensitivity / settings.dp_epsilon
        
        # Add Laplacian noise
        noise = np.random.laplace(0, scale, len(df))
        
        return df.with_columns(
            (pl.col(column) + pl.lit(noise.tolist()).list.eval(pl.element())).alias(column)
        )
    
    def create_transformation_plan(
        self,
        detection_result: PIIDetectionResult,
    ) -> TransformationPlan:
        """Create a transformation plan from detection results.
        
        Args:
            detection_result: Result from PIIDetector
            
        Returns:
            TransformationPlan to be applied
        """
        columns = {}
        for match in detection_result.matches:
            transform = self.DEFAULT_TRANSFORMATIONS.get(
                match.pii_type, TransformationType.HASH
            )
            columns[match.column_name] = transform
        
        return TransformationPlan(
            columns=columns,
            hash_key=self.hash_key[:8] + "...",  # Truncated for security
        )
    
    def get_transformation_log(self) -> list[TransformationResult]:
        """Get the log of all transformations applied."""
        return self._transformation_log.copy()
    
    def clear_log(self) -> None:
        """Clear the transformation log."""
        self._transformation_log.clear()
    
    def export_audit_trail(self) -> dict:
        """Export audit trail for compliance documentation."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_transformations": len(self._transformation_log),
            "transformations": [
                {
                    "column": r.column_name,
                    "type": r.transformation_type.value,
                    "rows": r.rows_transformed,
                    "success": r.success,
                    "metadata": r.metadata,
                }
                for r in self._transformation_log
            ],
            "pseudonym_map_sizes": {
                col: len(mapping) for col, mapping in self._pseudonym_maps.items()
            },
        }
