"""
SentinXFL Sprint 1 Test Script
===============================

Tests for the 5-Gate PII Pipeline implementation.

Run with: python -m pytest tests/test_sprint1.py -v

Author: Anshuman Bakshi
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import polars as pl

from sentinxfl.core.config import settings
from sentinxfl.data.loader import DataLoader
from sentinxfl.data.splitter import DataSplitter
from sentinxfl.data.schemas import DatasetType
from sentinxfl.privacy.detector import PIIDetector, PIIType, PIISensitivity
from sentinxfl.privacy.transformer import PIITransformer, TransformationType
from sentinxfl.privacy.certifier import PIICertifier
from sentinxfl.privacy.audit import PIIAuditLog


class TestConfiguration:
    """Test configuration module."""

    def test_settings_loaded(self):
        """Settings should load with defaults."""
        assert settings.app_name == "SentinXFL"
        assert settings.app_version == "2.0.0"

    def test_paths_configured(self):
        """Data paths should be configured."""
        assert settings.data_dir is not None
        assert settings.processed_dir is not None

    def test_privacy_settings(self):
        """Privacy settings should have safe defaults."""
        assert settings.dp_epsilon >= 0.1
        assert settings.dp_epsilon <= 10.0
        assert settings.pii_strict_mode is True


class TestDataLoader:
    """Test data loading functionality."""

    @pytest.fixture
    def loader(self):
        """Create a data loader instance."""
        return DataLoader()

    def test_loader_initialization(self, loader):
        """Loader should initialize with correct paths."""
        assert loader.data_dir == settings.data_dir_abs

    def test_column_mappings_defined(self, loader):
        """Column mappings should be defined for all datasets."""
        assert DatasetType.BANK_ACCOUNT_FRAUD in loader.COLUMN_MAPPINGS
        assert DatasetType.CREDIT_CARD_FRAUD in loader.COLUMN_MAPPINGS
        assert DatasetType.PAYSIM in loader.COLUMN_MAPPINGS

    def test_bank_fraud_files_defined(self, loader):
        """Bank fraud file list should be defined."""
        assert len(loader.BANK_FRAUD_FILES) == 6
        assert "Base.csv" in loader.BANK_FRAUD_FILES

    @pytest.mark.skipif(
        not (settings.data_dir_abs / "Base.csv").exists(),
        reason="Dataset not available"
    )
    def test_load_bank_fraud_sample(self, loader):
        """Should load bank fraud dataset with sampling."""
        df = loader.load_bank_account_fraud(sample_frac=0.001)
        assert len(df) > 0
        assert "is_fraud" in df.columns

    @pytest.mark.skipif(
        not (settings.data_dir_abs / "creditcard.csv").exists(),
        reason="Dataset not available"
    )
    def test_load_credit_card_sample(self, loader):
        """Should load credit card dataset with sampling."""
        df = loader.load_credit_card_fraud(sample_frac=0.01)
        assert len(df) > 0
        assert "is_fraud" in df.columns


class TestDataSplitter:
    """Test data splitting functionality."""

    @pytest.fixture
    def splitter(self):
        """Create a data splitter instance."""
        return DataSplitter()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pl.DataFrame({
            "feature1": list(range(1000)),
            "feature2": [i * 0.5 for i in range(1000)],
            "is_fraud": [True if i < 100 else False for i in range(1000)],
        })

    def test_splitter_initialization(self, splitter):
        """Splitter should initialize with correct ratios."""
        assert splitter.train_ratio + splitter.val_ratio + splitter.test_ratio == 1.0

    def test_stratified_split(self, splitter, sample_df):
        """Should perform stratified split maintaining fraud ratio."""
        train, val, test = splitter.split(sample_df, label_col="is_fraud")

        assert len(train) + len(val) + len(test) == len(sample_df)

        # Check fraud ratios are approximately maintained
        original_ratio = sample_df.filter(pl.col("is_fraud")).shape[0] / len(sample_df)
        train_ratio = train.filter(pl.col("is_fraud")).shape[0] / len(train)

        assert abs(train_ratio - original_ratio) < 0.05  # Within 5%

    def test_fl_partition_iid(self, splitter, sample_df):
        """Should partition data for FL clients (IID)."""
        partitions = splitter.partition_for_fl(
            sample_df, num_clients=3, partition_type="iid"
        )

        assert len(partitions) == 3
        total_rows = sum(len(p) for p in partitions)
        assert total_rows == len(sample_df)

    def test_class_weights(self, splitter, sample_df):
        """Should calculate class weights for imbalanced data."""
        weights = splitter.get_class_weights(sample_df, label_col="is_fraud")

        assert True in weights
        assert False in weights
        assert weights[True] > weights[False]  # Minority class has higher weight


class TestPIIDetector:
    """Test PII detection functionality."""

    @pytest.fixture
    def detector(self):
        """Create a PII detector instance."""
        return PIIDetector(strict_mode=True)

    @pytest.fixture
    def sample_df_with_pii(self):
        """Create sample DataFrame with PII."""
        return pl.DataFrame({
            "customer_name": ["John Doe", "Jane Smith", "Bob Wilson"],
            "email": ["john@email.com", "jane@test.com", "bob@example.com"],
            "phone_number": ["1234567890", "9876543210", "5555555555"],
            "customer_age": [25, 35, 45],
            "amount": [100.0, 250.0, 500.0],
            "is_fraud": [False, True, False],
        })

    @pytest.fixture
    def sample_df_safe(self):
        """Create sample DataFrame without PII."""
        return pl.DataFrame({
            "transaction_id_hash": ["abc123", "def456", "ghi789"],
            "amount": [100.0, 250.0, 500.0],
            "velocity_24h": [5, 3, 8],
            "is_fraud": [False, True, False],
        })

    def test_detector_initialization(self, detector):
        """Detector should initialize with compiled patterns."""
        assert len(detector._compiled_column_patterns) > 0
        assert len(detector._compiled_value_patterns) > 0

    def test_detect_pii_by_column_name(self, detector, sample_df_with_pii):
        """Should detect PII by column name patterns."""
        result = detector.detect(sample_df_with_pii)

        pii_columns = {m.column_name for m in result.matches}
        assert "customer_name" in pii_columns
        assert "email" in pii_columns
        assert "phone_number" in pii_columns

    def test_detect_email_pattern(self, detector, sample_df_with_pii):
        """Should detect email patterns."""
        result = detector.detect(sample_df_with_pii)

        email_matches = [m for m in result.matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) > 0

    def test_pii_sensitivity_classification(self, detector, sample_df_with_pii):
        """Should classify PII sensitivity correctly."""
        result = detector.detect(sample_df_with_pii)

        for match in result.matches:
            assert match.sensitivity in [
                PIISensitivity.CRITICAL,
                PIISensitivity.HIGH,
                PIISensitivity.MEDIUM,
                PIISensitivity.LOW,
            ]

    def test_safe_df_passes(self, detector, sample_df_safe):
        """Safe DataFrame should pass PII detection."""
        result = detector.detect(sample_df_safe)
        # Should have no high-sensitivity matches
        high_risk = [
            m for m in result.matches
            if m.sensitivity in [PIISensitivity.CRITICAL, PIISensitivity.HIGH]
        ]
        assert len(high_risk) == 0

    def test_generate_report(self, detector, sample_df_with_pii):
        """Should generate human-readable report."""
        result = detector.detect(sample_df_with_pii)
        report = detector.generate_report(result)

        assert "PII DETECTION REPORT" in report
        assert "customer_name" in report


class TestPIITransformer:
    """Test PII transformation functionality."""

    @pytest.fixture
    def transformer(self):
        """Create a PII transformer instance."""
        return PIITransformer()

    @pytest.fixture
    def detector(self):
        """Create a PII detector instance."""
        return PIIDetector()

    @pytest.fixture
    def sample_df_with_pii(self):
        """Create sample DataFrame with PII."""
        return pl.DataFrame({
            "customer_name": ["John Doe", "Jane Smith", "Bob Wilson"],
            "email": ["john@email.com", "jane@test.com", "bob@example.com"],
            "customer_age": [25, 35, 45],
            "amount": [100.0, 250.0, 500.0],
            "is_fraud": [False, True, False],
        })

    def test_transformer_initialization(self, transformer):
        """Transformer should initialize with hash key."""
        assert transformer.hash_key is not None
        assert len(transformer.hash_key) > 0

    def test_hash_transformation(self, transformer, detector, sample_df_with_pii):
        """Should apply hash transformation."""
        detection_result = detector.detect(sample_df_with_pii)
        transformed_df, results = transformer.transform(
            sample_df_with_pii,
            detection_result,
            custom_transforms={"customer_name": TransformationType.HASH},
        )

        # Name should be hashed (16 chars hex)
        name_values = transformed_df["customer_name"].to_list()
        assert all(len(v) == 16 if v else True for v in name_values)

    def test_mask_transformation(self, transformer, detector, sample_df_with_pii):
        """Should apply mask transformation."""
        detection_result = detector.detect(sample_df_with_pii)
        transformed_df, results = transformer.transform(
            sample_df_with_pii,
            detection_result,
            custom_transforms={"email": TransformationType.MASK},
        )

        # Email should be masked
        email_values = transformed_df["email"].to_list()
        assert all("*" in v if v else True for v in email_values)

    def test_generalize_age(self, transformer, detector, sample_df_with_pii):
        """Should generalize age to bins."""
        detection_result = detector.detect(sample_df_with_pii)
        transformed_df, results = transformer.transform(
            sample_df_with_pii,
            detection_result,
            custom_transforms={"customer_age": TransformationType.GENERALIZE},
        )

        # Age should be in categorical bins
        age_values = transformed_df["customer_age"].to_list()
        valid_bins = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        assert all(str(v) in valid_bins for v in age_values if v)

    def test_transformation_log(self, transformer, detector, sample_df_with_pii):
        """Should maintain transformation log."""
        transformer.clear_log()
        detection_result = detector.detect(sample_df_with_pii)
        transformer.transform(sample_df_with_pii, detection_result)

        log = transformer.get_transformation_log()
        assert len(log) > 0

    def test_audit_trail_export(self, transformer, detector, sample_df_with_pii):
        """Should export audit trail."""
        transformer.clear_log()
        detection_result = detector.detect(sample_df_with_pii)
        transformer.transform(sample_df_with_pii, detection_result)

        audit = transformer.export_audit_trail()
        assert "timestamp" in audit
        assert "transformations" in audit


class TestPIICertifier:
    """Test PII certification functionality."""

    @pytest.fixture
    def certifier(self):
        """Create a PII certifier instance."""
        return PIICertifier()

    @pytest.fixture
    def detector(self):
        """Create a PII detector instance."""
        return PIIDetector()

    @pytest.fixture
    def transformer(self):
        """Create a PII transformer instance."""
        return PIITransformer()

    @pytest.fixture
    def sample_df_sanitized(self):
        """Create a sanitized sample DataFrame."""
        return pl.DataFrame({
            "id_hash": ["abc123def456", "ghi789jkl012", "mno345pqr678"],
            "amount": [100.0, 250.0, 500.0],
            "velocity_24h": [5, 3, 8],
            "is_fraud": [False, True, False],
        })

    def test_certifier_initialization(self, certifier):
        """Certifier should initialize with thresholds."""
        assert certifier.entropy_threshold > 0
        assert certifier.uniqueness_threshold > 0
        assert certifier.k_anonymity_min > 0

    def test_certify_clean_data(self, certifier, detector, sample_df_sanitized):
        """Should certify clean (sanitized) data."""
        detection_result = detector.detect(sample_df_sanitized)
        cert_result = certifier.certify(sample_df_sanitized, detection_result)

        assert cert_result.certified is True
        assert cert_result.certification_level in ["gold", "silver", "bronze"]

    def test_gate3_uniqueness_analysis(self, certifier, sample_df_sanitized):
        """Gate 3 should analyze uniqueness."""
        uniqueness_results, passed = certifier._gate3_uniqueness_analysis(
            sample_df_sanitized
        )

        assert len(uniqueness_results) > 0
        for result in uniqueness_results:
            assert result.unique_ratio >= 0
            assert result.unique_ratio <= 1

    def test_gate4_entropy_check(self, certifier, sample_df_sanitized):
        """Gate 4 should check entropy."""
        entropy_results, passed = certifier._gate4_entropy_check(sample_df_sanitized)

        # Some columns may not be string type, so we may have fewer results
        for result in entropy_results:
            assert result.entropy >= 0
            assert result.normalized_entropy >= 0
            assert result.normalized_entropy <= 1

    def test_generate_certificate(self, certifier, detector, sample_df_sanitized):
        """Should generate formal certificate."""
        detection_result = detector.detect(sample_df_sanitized)
        cert_result = certifier.certify(sample_df_sanitized, detection_result)
        certificate = certifier.generate_certificate(cert_result)

        assert "CERTIFICATION CERTIFICATE" in certificate
        assert "SENTINXFL-" in certificate

    def test_verify_k_anonymity(self, certifier, sample_df_sanitized):
        """Should verify k-anonymity."""
        passed, actual_k = certifier.verify_k_anonymity(
            sample_df_sanitized,
            quasi_identifiers=["velocity_24h"],
            k=2,
        )

        assert isinstance(passed, bool)
        assert actual_k >= 1


class TestPIIAuditLog:
    """Test audit logging functionality."""

    @pytest.fixture
    def audit(self):
        """Create an audit log instance."""
        return PIIAuditLog(actor="test")

    def test_audit_initialization(self, audit):
        """Audit log should initialize with session ID."""
        assert audit.session_id is not None
        assert audit.session_id.startswith("AUDIT-")

    def test_log_scan_started(self, audit):
        """Should log scan start event."""
        event = audit.log_scan_started("test_dataset", 10, 1000)

        assert event.event_type.value == "pii_scan_started"
        assert event.resource == "test_dataset"

    def test_chain_integrity(self, audit):
        """Should maintain chain integrity."""
        audit.log_scan_started("test", 10, 1000)
        audit.log_data_access("test", "testing")

        is_valid, issues = audit.verify_chain_integrity()
        assert is_valid is True
        assert len(issues) == 0

    def test_compliance_report(self, audit):
        """Should generate compliance report."""
        audit.log_scan_started("test", 10, 1000)
        audit.log_data_access("test", "testing")

        report = audit.generate_compliance_report()
        assert "COMPLIANCE REPORT" in report
        assert audit.session_id in report


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def sample_df(self):
        """Create a realistic sample DataFrame."""
        return pl.DataFrame({
            "customer_name": ["Alice Johnson", "Bob Smith", "Carol Davis"],
            "email": ["alice@company.com", "bob@test.org", "carol@example.net"],
            "phone": ["555-123-4567", "555-987-6543", "555-456-7890"],
            "ssn": ["123-45-6789", "987-65-4321", "456-78-9012"],
            "age": [28, 42, 35],
            "income": [65000, 85000, 72000],
            "transaction_amount": [150.00, 2500.00, 890.00],
            "is_fraud": [False, True, False],
        })

    def test_full_pipeline(self, sample_df):
        """Test complete detect -> transform -> certify pipeline."""
        # Step 1: Detect PII
        detector = PIIDetector(strict_mode=True)
        detection_result = detector.detect(sample_df)

        assert detection_result.columns_with_pii > 0
        assert not detection_result.passed  # Should fail due to PII

        # Step 2: Transform PII
        transformer = PIITransformer()
        transformed_df, transform_results = transformer.transform(
            sample_df, detection_result
        )

        successful = sum(1 for r in transform_results if r.success)
        assert successful > 0

        # Step 3: Certify transformed data
        certifier = PIICertifier()
        # Re-detect on transformed data
        new_detection = detector.detect(transformed_df)
        cert_result = certifier.certify(transformed_df, new_detection)

        assert cert_result.certification_score > 0

    def test_audit_throughout_pipeline(self, sample_df):
        """Test audit logging throughout pipeline."""
        audit = PIIAuditLog(actor="integration_test")

        # Log operations
        audit.log_scan_started("test_df", len(sample_df.columns), len(sample_df))

        detector = PIIDetector()
        result = detector.detect(sample_df)
        audit.log_scan_completed("test_df", result)

        # Verify chain
        is_valid, _ = audit.verify_chain_integrity()
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
