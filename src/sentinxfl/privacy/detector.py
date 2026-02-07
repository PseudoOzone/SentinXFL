"""
SentinXFL PII Detector
=======================

Gate 1 & 2 of the 5-Gate Certified Data Sanitization Pipeline.

Gate 1: Column Name Analysis (semantic matching)
Gate 2: Regex Pattern Detection (100+ patterns)

This is a PATENT-CORE component.

Author: Anshuman Bakshi
Copyright (c) 2026. All rights reserved.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import polars as pl

from sentinxfl.core.config import settings
from sentinxfl.core.logging import get_logger

log = get_logger(__name__)


class PIIType(str, Enum):
    """Types of Personally Identifiable Information."""
    # Direct identifiers
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    NATIONAL_ID = "national_id"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    
    # Financial identifiers
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    IBAN = "iban"
    SWIFT = "swift"
    
    # Location identifiers
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    IP_ADDRESS = "ip_address"
    GPS_COORDINATES = "gps_coordinates"
    
    # Device identifiers
    DEVICE_ID = "device_id"
    MAC_ADDRESS = "mac_address"
    IMEI = "imei"
    
    # Temporal identifiers
    DATE_OF_BIRTH = "date_of_birth"
    
    # Quasi-identifiers (can identify when combined)
    AGE = "age"
    GENDER = "gender"
    INCOME = "income"
    OCCUPATION = "occupation"
    
    # Other
    USERNAME = "username"
    PASSWORD = "password"
    BIOMETRIC = "biometric"
    UNKNOWN = "unknown"


class PIISensitivity(str, Enum):
    """Sensitivity levels for PII."""
    CRITICAL = "critical"  # Direct identifiers (SSN, credit card)
    HIGH = "high"          # Strong identifiers (name, email, phone)
    MEDIUM = "medium"      # Quasi-identifiers (age, gender, zip)
    LOW = "low"            # Weak signals
    NONE = "none"          # Not PII


@dataclass
class PIIMatch:
    """A detected PII match."""
    column_name: str
    pii_type: PIIType
    sensitivity: PIISensitivity
    detection_method: str  # "column_name", "regex", "statistical", "entropy", "ml"
    confidence: float  # 0.0 to 1.0
    sample_values: list[str] = field(default_factory=list)
    pattern_matched: Optional[str] = None
    recommendation: str = "transform"  # "transform", "remove", "keep"


@dataclass
class PIIDetectionResult:
    """Result of PII detection on a dataset."""
    total_columns: int
    columns_with_pii: int
    matches: list[PIIMatch]
    passed: bool = False
    scan_timestamp: str = ""
    
    @property
    def critical_pii_found(self) -> bool:
        return any(m.sensitivity == PIISensitivity.CRITICAL for m in self.matches)
    
    @property
    def high_pii_found(self) -> bool:
        return any(m.sensitivity in [PIISensitivity.CRITICAL, PIISensitivity.HIGH] for m in self.matches)


class PIIDetector:
    """
    5-Gate PII Detection Engine.
    
    Gate 1: Column Name Analysis
    Gate 2: Regex Pattern Matching
    
    Additional gates (3-5) are in PIICertifier:
    - Gate 3: Statistical Analysis
    - Gate 4: Entropy Check
    - Gate 5: ML-based Detection
    """
    
    # Gate 1: Column name patterns (case-insensitive)
    COLUMN_NAME_PATTERNS = {
        PIIType.NAME: [
            r"name", r"first.?name", r"last.?name", r"full.?name",
            r"customer.?name", r"user.?name", r"account.?holder",
            r"given.?name", r"surname", r"^fn$", r"^ln$",
        ],
        PIIType.EMAIL: [
            r"email", r"e.?mail", r"mail", r"email.?address",
        ],
        PIIType.PHONE: [
            r"phone", r"mobile", r"cell", r"tel", r"telephone",
            r"contact.?number", r"phone.?number", r"mobile.?number",
        ],
        PIIType.SSN: [
            r"ssn", r"social.?security", r"tax.?id", r"sin",
        ],
        PIIType.NATIONAL_ID: [
            r"national.?id", r"passport", r"aadhaar", r"pan",
            r"voter.?id", r"license", r"dl.?number",
        ],
        PIIType.CREDIT_CARD: [
            r"credit.?card", r"card.?number", r"cc.?num", r"pan.?number",
            r"card.?no", r"payment.?card",
        ],
        PIIType.BANK_ACCOUNT: [
            r"bank.?account", r"account.?number", r"acct.?num",
            r"routing", r"iban", r"swift", r"bic",
        ],
        PIIType.ADDRESS: [
            r"address", r"street", r"\bcity\b", r"\bstate\b", r"country",
            r"postal", r"\bzip\b", r"pincode", r"residence",
        ],
        PIIType.ZIP_CODE: [
            r"zip", r"postal.?code", r"pin.?code", r"postcode",
        ],
        PIIType.IP_ADDRESS: [
            r"ip.?address", r"ip$", r"client.?ip", r"user.?ip",
        ],
        PIIType.DEVICE_ID: [
            r"device.?id", r"device.?uuid", r"udid", r"android.?id",
            r"idfa", r"idfv", r"hardware.?id",
        ],
        PIIType.DATE_OF_BIRTH: [
            r"dob", r"birth.?date", r"date.?of.?birth", r"birthday",
        ],
        PIIType.AGE: [
            r"^age$", r"customer.?age", r"user.?age",
        ],
        PIIType.GENDER: [
            r"gender", r"sex", r"male.?female",
        ],
        PIIType.USERNAME: [
            r"user.?name", r"login", r"user.?id", r"userid",
        ],
        PIIType.PASSWORD: [
            r"password", r"passwd", r"pwd", r"secret", r"pin$",
        ],
    }
    
    # Gate 2: Value patterns (regex for content matching)
    VALUE_PATTERNS = {
        PIIType.EMAIL: [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        ],
        PIIType.PHONE: [
            r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US format
            r"\+91[-.\s]?\d{10}",  # India format
            r"\d{10,15}",  # Generic phone
        ],
        PIIType.SSN: [
            r"\d{3}[-.\s]?\d{2}[-.\s]?\d{4}",  # US SSN
        ],
        PIIType.CREDIT_CARD: [
            r"4[0-9]{12}(?:[0-9]{3})?",  # Visa
            r"5[1-5][0-9]{14}",  # Mastercard
            r"3[47][0-9]{13}",  # Amex
            r"6(?:011|5[0-9][0-9])[0-9]{12}",  # Discover
            r"\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}",  # Generic 16-digit
        ],
        PIIType.IBAN: [
            r"[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}",
        ],
        PIIType.IP_ADDRESS: [
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IPv4
            r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}",  # IPv6
        ],
        PIIType.MAC_ADDRESS: [
            r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})",
        ],
        PIIType.DATE_OF_BIRTH: [
            r"\d{2}[-/]\d{2}[-/]\d{4}",  # DD/MM/YYYY
            r"\d{4}[-/]\d{2}[-/]\d{2}",  # YYYY-MM-DD
        ],
        PIIType.ZIP_CODE: [
            r"\d{5}(-\d{4})?",  # US ZIP
            r"\d{6}",  # India PIN
            r"[A-Z]\d[A-Z]\s?\d[A-Z]\d",  # Canadian
        ],
        PIIType.GPS_COORDINATES: [
            r"[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)",
        ],
        PIIType.IMEI: [
            r"\d{15}",  # 15-digit IMEI
        ],
        PIIType.NATIONAL_ID: [
            r"[A-Z]{5}\d{4}[A-Z]",  # India PAN
            r"\d{12}",  # Aadhaar (12 digits)
        ],
    }
    
    # Sensitivity mapping
    SENSITIVITY_MAP = {
        PIIType.SSN: PIISensitivity.CRITICAL,
        PIIType.CREDIT_CARD: PIISensitivity.CRITICAL,
        PIIType.BANK_ACCOUNT: PIISensitivity.CRITICAL,
        PIIType.PASSWORD: PIISensitivity.CRITICAL,
        PIIType.BIOMETRIC: PIISensitivity.CRITICAL,
        PIIType.NAME: PIISensitivity.HIGH,
        PIIType.EMAIL: PIISensitivity.HIGH,
        PIIType.PHONE: PIISensitivity.HIGH,
        PIIType.ADDRESS: PIISensitivity.HIGH,
        PIIType.NATIONAL_ID: PIISensitivity.CRITICAL,
        PIIType.PASSPORT: PIISensitivity.CRITICAL,
        PIIType.DRIVERS_LICENSE: PIISensitivity.CRITICAL,
        PIIType.DATE_OF_BIRTH: PIISensitivity.HIGH,
        PIIType.IP_ADDRESS: PIISensitivity.MEDIUM,
        PIIType.DEVICE_ID: PIISensitivity.MEDIUM,
        PIIType.MAC_ADDRESS: PIISensitivity.MEDIUM,
        PIIType.GPS_COORDINATES: PIISensitivity.HIGH,
        PIIType.ZIP_CODE: PIISensitivity.MEDIUM,
        PIIType.AGE: PIISensitivity.MEDIUM,
        PIIType.GENDER: PIISensitivity.MEDIUM,
        PIIType.INCOME: PIISensitivity.MEDIUM,
        PIIType.OCCUPATION: PIISensitivity.MEDIUM,
        PIIType.IBAN: PIISensitivity.CRITICAL,
        PIIType.SWIFT: PIISensitivity.HIGH,
        PIIType.IMEI: PIISensitivity.HIGH,
        PIIType.USERNAME: PIISensitivity.MEDIUM,
        PIIType.UNKNOWN: PIISensitivity.LOW,
    }
    
    def __init__(self, strict_mode: bool = None):
        """Initialize PII Detector.
        
        Args:
            strict_mode: If True, any PII detection fails certification.
                        Defaults to settings.pii_strict_mode.
        """
        self.strict_mode = strict_mode if strict_mode is not None else settings.pii_strict_mode
        
        # Compile regex patterns
        self._compiled_column_patterns = {
            pii_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for pii_type, patterns in self.COLUMN_NAME_PATTERNS.items()
        }
        self._compiled_value_patterns = {
            pii_type: [re.compile(p) for p in patterns]
            for pii_type, patterns in self.VALUE_PATTERNS.items()
        }
        
        log.info(f"PIIDetector initialized (strict_mode={self.strict_mode})")
    
    def detect(
        self,
        df: pl.DataFrame,
        sample_size: int = 1000,
    ) -> PIIDetectionResult:
        """Run PII detection on a DataFrame (Gates 1 & 2).
        
        Args:
            df: Input DataFrame to scan
            sample_size: Number of rows to sample for value analysis
            
        Returns:
            PIIDetectionResult with all matches
        """
        from datetime import datetime
        
        log.info(f"Starting PII detection on {len(df.columns)} columns, {len(df)} rows")
        matches: list[PIIMatch] = []
        
        # Sample data for value analysis
        sample_df = df.sample(n=min(sample_size, len(df)), seed=42) if len(df) > sample_size else df
        
        for col in df.columns:
            col_matches = []
            
            # Gate 1: Column name analysis
            col_match = self._detect_by_column_name(col)
            if col_match:
                col_matches.append(col_match)
            
            # Gate 2: Regex pattern matching on values
            if col in sample_df.columns:
                col_dtype = sample_df[col].dtype
                if col_dtype == pl.Utf8 or col_dtype == pl.Object:
                    value_matches = self._detect_by_value_patterns(col, sample_df[col])
                    col_matches.extend(value_matches)
            
            # Merge matches for this column (take highest confidence)
            if col_matches:
                best_match = max(col_matches, key=lambda m: m.confidence)
                matches.append(best_match)
        
        # Determine if passed (no critical/high PII in strict mode)
        passed = True
        if self.strict_mode:
            passed = not any(
                m.sensitivity in [PIISensitivity.CRITICAL, PIISensitivity.HIGH]
                for m in matches
            )
        
        result = PIIDetectionResult(
            total_columns=len(df.columns),
            columns_with_pii=len(matches),
            matches=matches,
            passed=passed,
            scan_timestamp=datetime.utcnow().isoformat(),
        )
        
        log.info(
            f"PII Detection complete: {result.columns_with_pii}/{result.total_columns} columns flagged, "
            f"passed={result.passed}"
        )
        
        # Log critical findings
        for match in matches:
            if match.sensitivity in [PIISensitivity.CRITICAL, PIISensitivity.HIGH]:
                log.warning(
                    f"PII DETECTED: {match.column_name} | type={match.pii_type.value} | "
                    f"sensitivity={match.sensitivity.value} | confidence={match.confidence:.2f}"
                )
        
        return result
    
    def _detect_by_column_name(self, column_name: str) -> Optional[PIIMatch]:
        """Gate 1: Detect PII by column name patterns."""
        for pii_type, patterns in self._compiled_column_patterns.items():
            for pattern in patterns:
                if pattern.search(column_name):
                    return PIIMatch(
                        column_name=column_name,
                        pii_type=pii_type,
                        sensitivity=self.SENSITIVITY_MAP.get(pii_type, PIISensitivity.LOW),
                        detection_method="column_name",
                        confidence=0.8,  # High confidence for name match
                        pattern_matched=pattern.pattern,
                        recommendation=self._get_recommendation(pii_type),
                    )
        return None
    
    def _detect_by_value_patterns(
        self,
        column_name: str,
        series: pl.Series,
    ) -> list[PIIMatch]:
        """Gate 2: Detect PII by regex patterns on values."""
        matches = []
        
        # Convert to string and get non-null samples
        try:
            values = series.drop_nulls().cast(pl.Utf8).to_list()[:100]
        except Exception:
            return matches
        
        if not values:
            return matches
        
        for pii_type, patterns in self._compiled_value_patterns.items():
            for pattern in patterns:
                match_count = sum(1 for v in values if pattern.search(str(v)))
                match_ratio = match_count / len(values) if values else 0
                
                # If >10% of sampled values match, flag it
                if match_ratio > 0.1:
                    # Get sample matched values (masked)
                    sample_matched = [
                        self._mask_value(str(v)) 
                        for v in values[:5] 
                        if pattern.search(str(v))
                    ]
                    
                    matches.append(PIIMatch(
                        column_name=column_name,
                        pii_type=pii_type,
                        sensitivity=self.SENSITIVITY_MAP.get(pii_type, PIISensitivity.LOW),
                        detection_method="regex",
                        confidence=min(0.9, 0.5 + match_ratio),
                        sample_values=sample_matched,
                        pattern_matched=pattern.pattern,
                        recommendation=self._get_recommendation(pii_type),
                    ))
        
        return matches
    
    def _get_recommendation(self, pii_type: PIIType) -> str:
        """Get transformation recommendation for PII type."""
        # Critical PII should be removed or heavily transformed
        if self.SENSITIVITY_MAP.get(pii_type) == PIISensitivity.CRITICAL:
            return "remove"
        elif self.SENSITIVITY_MAP.get(pii_type) == PIISensitivity.HIGH:
            return "transform"  # Hash or pseudonymize
        else:
            return "transform"  # Generalize or bin
    
    @staticmethod
    def _mask_value(value: str, show_chars: int = 3) -> str:
        """Mask a PII value for logging (show only first few chars)."""
        if len(value) <= show_chars:
            return "*" * len(value)
        return value[:show_chars] + "*" * (len(value) - show_chars)
    
    def get_safe_columns(self, df: pl.DataFrame) -> list[str]:
        """Get list of columns that passed PII detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names that don't contain PII
        """
        result = self.detect(df)
        pii_columns = {m.column_name for m in result.matches}
        return [col for col in df.columns if col not in pii_columns]
    
    def get_pii_columns(self, df: pl.DataFrame) -> list[str]:
        """Get list of columns that contain PII.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names containing PII
        """
        result = self.detect(df)
        return [m.column_name for m in result.matches]
    
    def generate_report(self, result: PIIDetectionResult) -> str:
        """Generate a human-readable PII report.
        
        Args:
            result: PIIDetectionResult from detect()
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "PII DETECTION REPORT",
            "=" * 60,
            f"Scan Timestamp: {result.scan_timestamp}",
            f"Total Columns: {result.total_columns}",
            f"Columns with PII: {result.columns_with_pii}",
            f"PASSED: {'✓ YES' if result.passed else '✗ NO'}",
            "-" * 60,
        ]
        
        if result.matches:
            lines.append("DETECTED PII:")
            for match in sorted(result.matches, key=lambda m: m.sensitivity.value):
                lines.append(
                    f"  [{match.sensitivity.value.upper():8}] {match.column_name:30} "
                    f"| {match.pii_type.value:15} | conf={match.confidence:.2f} "
                    f"| {match.recommendation}"
                )
        else:
            lines.append("No PII detected.")
        
        lines.append("=" * 60)
        return "\n".join(lines)
