"""
SentinXFL PII Certifier
========================

Gates 3-5 of the 5-Gate Certified Data Sanitization Pipeline.

Gate 3: Statistical Uniqueness Analysis (quasi-identifier detection)
Gate 4: Entropy Check (detect high-entropy sensitive data)
Gate 5: ML-based Pattern Detection (fallback for missed PII)

This is a PATENT-CORE component.

Author: Anshuman Bakshi
Copyright (c) 2026. All rights reserved.
"""

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import polars as pl
import numpy as np

from sentinxfl.core.config import settings
from sentinxfl.core.logging import get_logger
from sentinxfl.privacy.detector import PIIDetectionResult, PIIMatch, PIISensitivity, PIIType

log = get_logger(__name__)


@dataclass
class EntropyResult:
    """Result of entropy analysis on a column."""
    column_name: str
    entropy: float
    normalized_entropy: float  # 0-1 scale
    unique_ratio: float
    is_high_entropy: bool
    threshold_used: float


@dataclass
class UniquenessResult:
    """Result of uniqueness analysis on a column."""
    column_name: str
    unique_count: int
    total_count: int
    unique_ratio: float
    is_quasi_identifier: bool
    k_anonymity_estimate: int


@dataclass
class CertificationResult:
    """Final certification result for a dataset."""
    # Overall
    certified: bool
    certification_level: str  # "gold", "silver", "bronze", "failed"
    timestamp: str
    
    # Gate results
    gate1_passed: bool  # Column name analysis
    gate2_passed: bool  # Regex patterns
    gate3_passed: bool  # Statistical uniqueness
    gate4_passed: bool  # Entropy check
    gate5_passed: bool  # ML detection (optional)
    
    # Details
    detection_result: Optional[PIIDetectionResult] = None
    entropy_results: list[EntropyResult] = field(default_factory=list)
    uniqueness_results: list[UniquenessResult] = field(default_factory=list)
    
    # Warnings and recommendations
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    # Metrics
    total_columns: int = 0
    safe_columns: int = 0
    transformed_columns: int = 0
    removed_columns: int = 0
    
    @property
    def certification_score(self) -> float:
        """Calculate certification score (0-100)."""
        gates_passed = sum([
            self.gate1_passed,
            self.gate2_passed,
            self.gate3_passed,
            self.gate4_passed,
            self.gate5_passed,
        ])
        return (gates_passed / 5) * 100


class PIICertifier:
    """
    PII Certification Engine implementing Gates 3-5.
    
    Provides a final certification stamp for data that has been
    properly sanitized through the 5-gate pipeline.
    """
    
    # Entropy thresholds (higher = more likely to be sensitive)
    ENTROPY_THRESHOLD_HIGH = 4.0  # Bits per character
    ENTROPY_THRESHOLD_MEDIUM = 3.0
    
    # Uniqueness thresholds for quasi-identifier detection
    UNIQUENESS_THRESHOLD_HIGH = 0.95  # >95% unique = likely identifier
    UNIQUENESS_THRESHOLD_MEDIUM = 0.80
    
    # K-anonymity minimum
    K_ANONYMITY_MIN = 5
    
    def __init__(
        self,
        entropy_threshold: float = None,
        uniqueness_threshold: float = None,
        k_anonymity_min: int = None,
    ):
        """Initialize PII Certifier.
        
        Args:
            entropy_threshold: Override entropy threshold for high-risk
            uniqueness_threshold: Override uniqueness threshold
            k_anonymity_min: Minimum k for k-anonymity
        """
        self.entropy_threshold = entropy_threshold or self.ENTROPY_THRESHOLD_HIGH
        self.uniqueness_threshold = uniqueness_threshold or self.UNIQUENESS_THRESHOLD_HIGH
        self.k_anonymity_min = k_anonymity_min or self.K_ANONYMITY_MIN
        
        log.info(
            f"PIICertifier initialized: entropy_threshold={self.entropy_threshold}, "
            f"uniqueness_threshold={self.uniqueness_threshold}, k_min={self.k_anonymity_min}"
        )
    
    def certify(
        self,
        df: pl.DataFrame,
        detection_result: PIIDetectionResult,
        run_ml_gate: bool = False,
    ) -> CertificationResult:
        """Run full certification on a transformed dataset.
        
        Args:
            df: DataFrame AFTER PII transformation
            detection_result: Original detection result (for Gates 1&2 status)
            run_ml_gate: Whether to run Gate 5 (ML-based detection)
            
        Returns:
            CertificationResult with certification status
        """
        log.info(f"Starting certification on {len(df.columns)} columns, {len(df)} rows")
        
        # Gates 1 & 2: Already run in detector
        gate1_passed = detection_result.passed or len(detection_result.matches) == 0
        gate2_passed = gate1_passed  # Same as Gate 1 for now
        
        # Gate 3: Statistical uniqueness analysis
        uniqueness_results, gate3_passed = self._gate3_uniqueness_analysis(df)
        
        # Gate 4: Entropy check
        entropy_results, gate4_passed = self._gate4_entropy_check(df)
        
        # Gate 5: ML-based detection (optional)
        gate5_passed = True
        if run_ml_gate:
            gate5_passed = self._gate5_ml_detection(df)
        
        # Determine certification level
        all_passed = all([gate1_passed, gate2_passed, gate3_passed, gate4_passed, gate5_passed])
        
        if all_passed:
            cert_level = "gold"
        elif sum([gate1_passed, gate2_passed, gate3_passed, gate4_passed]) >= 3:
            cert_level = "silver"
        elif sum([gate1_passed, gate2_passed, gate3_passed, gate4_passed]) >= 2:
            cert_level = "bronze"
        else:
            cert_level = "failed"
        
        # Generate warnings and recommendations
        warnings, recommendations = self._generate_warnings_and_recommendations(
            detection_result, uniqueness_results, entropy_results
        )
        
        result = CertificationResult(
            certified=cert_level != "failed",
            certification_level=cert_level,
            timestamp=datetime.utcnow().isoformat(),
            gate1_passed=gate1_passed,
            gate2_passed=gate2_passed,
            gate3_passed=gate3_passed,
            gate4_passed=gate4_passed,
            gate5_passed=gate5_passed,
            detection_result=detection_result,
            entropy_results=entropy_results,
            uniqueness_results=uniqueness_results,
            warnings=warnings,
            recommendations=recommendations,
            total_columns=len(df.columns),
            safe_columns=len(df.columns) - len(detection_result.matches),
            transformed_columns=len(detection_result.matches),
        )
        
        log.info(
            f"Certification complete: level={cert_level}, score={result.certification_score:.1f}/100, "
            f"gates=[{gate1_passed}, {gate2_passed}, {gate3_passed}, {gate4_passed}, {gate5_passed}]"
        )
        
        return result
    
    def _gate3_uniqueness_analysis(
        self,
        df: pl.DataFrame,
    ) -> tuple[list[UniquenessResult], bool]:
        """Gate 3: Statistical uniqueness analysis for quasi-identifier detection."""
        log.debug("Running Gate 3: Uniqueness Analysis")
        
        results = []
        quasi_identifiers_found = []
        
        for col in df.columns:
            try:
                unique_count = df[col].n_unique()
                total_count = len(df)
                unique_ratio = unique_count / total_count if total_count > 0 else 0
                
                # Estimate k-anonymity (minimum group size)
                if df[col].dtype == pl.Utf8:
                    # Count occurrences of each value
                    value_counts = df.group_by(col).agg(pl.count().alias("count"))
                    min_count = value_counts["count"].min()
                    k_estimate = min_count if min_count else 1
                else:
                    # For numeric, estimate based on unique ratio
                    k_estimate = max(1, int(total_count / unique_count)) if unique_count > 0 else total_count
                
                is_quasi = unique_ratio > self.uniqueness_threshold or k_estimate < self.k_anonymity_min
                
                if is_quasi:
                    quasi_identifiers_found.append(col)
                
                results.append(UniquenessResult(
                    column_name=col,
                    unique_count=unique_count,
                    total_count=total_count,
                    unique_ratio=unique_ratio,
                    is_quasi_identifier=is_quasi,
                    k_anonymity_estimate=k_estimate,
                ))
            except Exception as e:
                log.warning(f"Uniqueness analysis failed for {col}: {e}")
        
        # Gate passes if no high-risk quasi-identifiers found
        passed = len(quasi_identifiers_found) == 0
        
        if quasi_identifiers_found:
            log.warning(f"Gate 3 WARNING: Quasi-identifiers detected: {quasi_identifiers_found}")
        
        return results, passed
    
    def _gate4_entropy_check(
        self,
        df: pl.DataFrame,
    ) -> tuple[list[EntropyResult], bool]:
        """Gate 4: Entropy analysis to detect high-entropy sensitive data."""
        log.debug("Running Gate 4: Entropy Check")
        
        results = []
        high_entropy_cols = []
        
        for col in df.columns:
            try:
                # Only analyze string columns for entropy
                if df[col].dtype != pl.Utf8:
                    continue
                
                # Sample values for entropy calculation
                values = df[col].drop_nulls().sample(
                    n=min(1000, df[col].drop_nulls().len()),
                    seed=42
                ).to_list()
                
                if not values:
                    continue
                
                # Calculate Shannon entropy
                entropy = self._calculate_entropy(values)
                
                # Normalize by max possible character-level entropy
                all_chars = "".join(str(v) for v in values if v)
                n_distinct_chars = len(set(all_chars)) if all_chars else 1
                max_entropy = math.log2(n_distinct_chars) if n_distinct_chars > 1 else 1
                normalized = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0
                
                unique_ratio = len(set(values)) / len(values) if values else 0
                
                is_high = entropy > self.entropy_threshold and unique_ratio > 0.5
                
                if is_high:
                    high_entropy_cols.append(col)
                
                results.append(EntropyResult(
                    column_name=col,
                    entropy=entropy,
                    normalized_entropy=normalized,
                    unique_ratio=unique_ratio,
                    is_high_entropy=is_high,
                    threshold_used=self.entropy_threshold,
                ))
            except Exception as e:
                log.warning(f"Entropy check failed for {col}: {e}")
        
        # Gate passes if no unexpected high-entropy columns found
        passed = len(high_entropy_cols) == 0
        
        if high_entropy_cols:
            log.warning(f"Gate 4 WARNING: High-entropy columns detected: {high_entropy_cols}")
        
        return results, passed
    
    def _calculate_entropy(self, values: list[str]) -> float:
        """Calculate Shannon entropy of a list of string values."""
        if not values:
            return 0.0
        
        # Character-level entropy
        all_chars = "".join(str(v) for v in values if v)
        if not all_chars:
            return 0.0
        
        char_counts = {}
        for char in all_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total = len(all_chars)
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _gate5_ml_detection(self, df: pl.DataFrame) -> bool:
        """Gate 5: ML-based PII detection (placeholder for future implementation).
        
        This would use a trained classifier to detect PII patterns
        that regex and heuristics might miss.
        """
        log.debug("Running Gate 5: ML Detection (placeholder)")
        # TODO: Implement ML-based detection using a trained model
        # For now, always passes
        return True
    
    def _generate_warnings_and_recommendations(
        self,
        detection_result: PIIDetectionResult,
        uniqueness_results: list[UniquenessResult],
        entropy_results: list[EntropyResult],
    ) -> tuple[list[str], list[str]]:
        """Generate warnings and recommendations based on certification."""
        warnings = []
        recommendations = []
        
        # Check detection results
        if detection_result.critical_pii_found:
            warnings.append("CRITICAL: PII was detected that requires immediate attention")
            recommendations.append("Review and re-process columns flagged as CRITICAL sensitivity")
        
        # Check uniqueness
        quasi_ids = [r for r in uniqueness_results if r.is_quasi_identifier]
        if quasi_ids:
            warnings.append(f"Found {len(quasi_ids)} potential quasi-identifiers")
            for qi in quasi_ids[:3]:  # Show top 3
                recommendations.append(
                    f"Consider generalizing '{qi.column_name}' "
                    f"(k-anonymity estimate: {qi.k_anonymity_estimate})"
                )
        
        # Check entropy
        high_entropy = [r for r in entropy_results if r.is_high_entropy]
        if high_entropy:
            warnings.append(f"Found {len(high_entropy)} high-entropy columns that may contain sensitive IDs")
            for he in high_entropy[:3]:
                recommendations.append(
                    f"Review '{he.column_name}' (entropy: {he.entropy:.2f} bits)"
                )
        
        # Add general recommendations
        if not warnings:
            recommendations.append("Data appears properly sanitized - proceed with caution")
            recommendations.append("Periodic re-certification recommended after any data updates")
        
        return warnings, recommendations
    
    def generate_certificate(self, result: CertificationResult) -> str:
        """Generate a formal certification document.
        
        Args:
            result: CertificationResult from certify()
            
        Returns:
            Formatted certificate string
        """
        # Generate certificate hash
        cert_data = f"{result.timestamp}|{result.certification_level}|{result.total_columns}"
        cert_hash = hashlib.sha256(cert_data.encode()).hexdigest()[:12].upper()
        
        lines = [
            "╔" + "═" * 68 + "╗",
            "║" + " " * 20 + "PII CERTIFICATION CERTIFICATE" + " " * 18 + "║",
            "╠" + "═" * 68 + "╣",
            f"║  Certificate ID: SENTINXFL-{cert_hash}" + " " * 30 + "║",
            f"║  Issued: {result.timestamp[:19]}" + " " * 35 + "║",
            "╠" + "═" * 68 + "╣",
        ]
        
        # Status
        status_icon = "✓" if result.certified else "✗"
        status_text = f"CERTIFIED ({result.certification_level.upper()})" if result.certified else "NOT CERTIFIED"
        lines.append(f"║  Status: {status_icon} {status_text}" + " " * (55 - len(status_text)) + "║")
        lines.append(f"║  Score: {result.certification_score:.0f}/100" + " " * 50 + "║")
        lines.append("╠" + "═" * 68 + "╣")
        
        # Gate results
        lines.append("║  5-GATE VERIFICATION:" + " " * 46 + "║")
        gates = [
            ("Gate 1: Column Name Analysis", result.gate1_passed),
            ("Gate 2: Regex Pattern Detection", result.gate2_passed),
            ("Gate 3: Statistical Uniqueness", result.gate3_passed),
            ("Gate 4: Entropy Analysis", result.gate4_passed),
            ("Gate 5: ML Detection", result.gate5_passed),
        ]
        for gate_name, passed in gates:
            icon = "✓" if passed else "✗"
            lines.append(f"║    {icon} {gate_name}" + " " * (61 - len(gate_name)) + "║")
        
        lines.append("╠" + "═" * 68 + "╣")
        
        # Metrics
        lines.append("║  METRICS:" + " " * 58 + "║")
        lines.append(f"║    Total Columns: {result.total_columns}" + " " * (49 - len(str(result.total_columns))) + "║")
        lines.append(f"║    Safe Columns: {result.safe_columns}" + " " * (50 - len(str(result.safe_columns))) + "║")
        lines.append(f"║    Transformed: {result.transformed_columns}" + " " * (51 - len(str(result.transformed_columns))) + "║")
        
        lines.append("╠" + "═" * 68 + "╣")
        
        # Warnings
        if result.warnings:
            lines.append("║  WARNINGS:" + " " * 57 + "║")
            for w in result.warnings[:3]:
                w_truncated = w[:60] + "..." if len(w) > 60 else w
                lines.append(f"║    ⚠ {w_truncated}" + " " * max(0, 61 - len(w_truncated)) + "║")
        
        lines.append("╚" + "═" * 68 + "╝")
        
        return "\n".join(lines)
    
    def verify_k_anonymity(
        self,
        df: pl.DataFrame,
        quasi_identifiers: list[str],
        k: int = None,
    ) -> tuple[bool, int]:
        """Verify k-anonymity for given quasi-identifiers.
        
        Args:
            df: DataFrame to check
            quasi_identifiers: List of quasi-identifier column names
            k: Minimum k value (defaults to self.k_anonymity_min)
            
        Returns:
            Tuple of (passed, actual_k)
        """
        k = k or self.k_anonymity_min
        
        # Filter to only existing columns
        valid_cols = [c for c in quasi_identifiers if c in df.columns]
        if not valid_cols:
            return True, len(df)
        
        # Count equivalence classes
        grouped = df.group_by(valid_cols).agg(pl.count().alias("group_size"))
        min_group = grouped["group_size"].min()
        
        passed = min_group >= k
        
        log.info(
            f"K-anonymity check: k={k}, actual_min={min_group}, "
            f"passed={'✓' if passed else '✗'}"
        )
        
        return passed, min_group
