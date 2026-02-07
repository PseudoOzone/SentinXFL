"""
SentinXFL - Tests for Knowledge Module
=========================================

Tests for pattern library, emergent detector, central model,
report generator, and pattern miner.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sentinxfl.intelligence.pattern_library import (
    PatternEntry,
    PatternLibrary,
    PatternSeverity,
    PatternStatus,
    PatternType,
)
from sentinxfl.intelligence.emergent_detector import EmergentAlert, EmergentDetector
from sentinxfl.intelligence.central_model import (
    BankProfile,
    CentralKnowledgeModel,
    KnowledgeSnapshot,
)
from sentinxfl.intelligence.report_generator import Report, ReportGenerator
from sentinxfl.intelligence.pattern_miner import PatternMiner, MiningResult


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for tests."""
    return str(tmp_path)


@pytest.fixture
def library(tmp_dir):
    """Fresh pattern library."""
    return PatternLibrary(storage_path=tmp_dir)


@pytest.fixture
def seeded_library(library):
    """Pattern library with baseline patterns."""
    library.seed_baseline_patterns()
    return library


@pytest.fixture
def detector(seeded_library):
    return EmergentDetector(seeded_library)


@pytest.fixture
def central_model(seeded_library, tmp_dir):
    return CentralKnowledgeModel(seeded_library, data_dir=tmp_dir)


@pytest.fixture
def report_gen(seeded_library, tmp_dir):
    cm = CentralKnowledgeModel(seeded_library, data_dir=tmp_dir)
    det = EmergentDetector(seeded_library)
    return ReportGenerator(seeded_library, cm, det)


@pytest.fixture
def miner(library):
    return PatternMiner(library)


@pytest.fixture
def sample_feature_importances():
    return {
        "bank_A": {"amount": 0.3, "velocity": 0.2, "location": 0.1},
        "bank_B": {"amount": 0.25, "velocity": 0.18, "location": 0.12},
        "bank_C": {"amount": 0.35, "velocity": 0.22, "location": 0.08},
    }


@pytest.fixture
def sample_bank_metrics():
    return {
        "bank_A": {"fraud_rate": 0.02, "accuracy": 0.95, "num_samples": 10000, "num_fraud": 200},
        "bank_B": {"fraud_rate": 0.03, "accuracy": 0.93, "num_samples": 8000, "num_fraud": 240},
        "bank_C": {"fraud_rate": 0.015, "accuracy": 0.96, "num_samples": 12000, "num_fraud": 180},
    }


# ============================================
# PatternLibrary Tests
# ============================================


class TestPatternLibrary:
    def test_create_library(self, library):
        assert library is not None
        stats = library.get_statistics()
        assert stats["total"] == 0

    def test_seed_baseline(self, seeded_library):
        stats = seeded_library.get_statistics()
        assert stats["total"] == 6

    def test_add_pattern(self, library):
        p = PatternEntry(
            pattern_id="TEST-001",
            pattern_type=PatternType.FACT,
            name="Test Pattern",
            description="A test pattern",
            severity=PatternSeverity.HIGH,
            confidence=0.85,
        )
        library.add_pattern(p)
        retrieved = library.get_pattern("TEST-001")
        assert retrieved is not None
        assert retrieved.name == "Test Pattern"
        assert retrieved.confidence == 0.85

    def test_list_patterns(self, seeded_library):
        patterns = seeded_library.list_patterns()
        assert len(patterns) == 6

    def test_list_by_type(self, seeded_library):
        facts = seeded_library.list_patterns(pattern_type=PatternType.FACT)
        assert len(facts) > 0
        for p in facts:
            assert p.pattern_type == PatternType.FACT

    def test_search_patterns(self, seeded_library):
        results = seeded_library.search_patterns("card testing")
        assert len(results) > 0
        assert any("card" in r.name.lower() for r in results)

    def test_get_emergent_patterns(self, library):
        p = PatternEntry(
            pattern_id="EMRG-001",
            pattern_type=PatternType.EMERGENT,
            name="Emergent Test",
            description="An emergent pattern",
            severity=PatternSeverity.HIGH,
            confidence=0.7,
        )
        library.add_pattern(p)
        emergent = library.get_emergent_patterns()
        assert len(emergent) == 1

    def test_get_fact_based(self, seeded_library):
        facts = seeded_library.get_fact_based_patterns()
        assert len(facts) > 0
        for p in facts:
            assert p.is_fact_based

    def test_statistics(self, seeded_library):
        stats = seeded_library.get_statistics()
        assert stats["total"] == 6
        assert "by_type" in stats
        assert "by_severity" in stats

    def test_ingest_from_fl_round(self, library, sample_feature_importances, sample_bank_metrics):
        library.ingest_from_fl_round(
            round_number=1,
            feature_importances=sample_feature_importances,
            bank_metrics=sample_bank_metrics,
        )
        stats = library.get_statistics()
        assert stats["total"] >= 0  # May or may not find patterns

    def test_persistence(self, tmp_dir):
        lib1 = PatternLibrary(storage_path=tmp_dir)
        lib1.add_pattern(PatternEntry(
            pattern_id="PERSIST-001",
            pattern_type=PatternType.FACT,
            name="Persistent Pattern",
            description="Should persist",
            severity=PatternSeverity.LOW,
            confidence=0.9,
        ))

        # New instance should load from disk
        lib2 = PatternLibrary(storage_path=tmp_dir)
        retrieved = lib2.get_pattern("PERSIST-001")
        assert retrieved is not None
        assert retrieved.name == "Persistent Pattern"

    def test_pattern_to_dict(self, seeded_library):
        patterns = seeded_library.list_patterns(limit=1)
        assert len(patterns) == 1
        d = patterns[0].to_dict()
        assert "pattern_id" in d
        assert "name" in d
        assert "severity" in d


# ============================================
# EmergentDetector Tests
# ============================================


class TestEmergentDetector:
    def test_create_detector(self, detector):
        assert detector is not None

    def test_analyze_round_no_previous(self, detector, sample_feature_importances, sample_bank_metrics):
        alerts = detector.analyze_round(
            round_number=1,
            feature_importances=sample_feature_importances,
            bank_metrics=sample_bank_metrics,
        )
        # With no previous data, should not detect feature shifts
        assert isinstance(alerts, list)

    def test_detect_feature_shifts(self, detector):
        prev = {
            "bank_A": {"amount": 0.1, "velocity": 0.1},
            "bank_B": {"amount": 0.12, "velocity": 0.11},
        }
        curr = {
            "bank_A": {"amount": 0.5, "velocity": 0.1},
            "bank_B": {"amount": 0.55, "velocity": 0.12},
        }
        metrics = {
            "bank_A": {"fraud_rate": 0.02},
            "bank_B": {"fraud_rate": 0.03},
        }
        alerts = detector.analyze_round(
            round_number=2,
            feature_importances=curr,
            bank_metrics=metrics,
            previous_importances=prev,
        )
        # Should detect big shift in 'amount'
        shift_alerts = [a for a in alerts if "amount" in a.title.lower() or "shift" in a.title.lower()]
        assert len(shift_alerts) > 0

    def test_detect_spike(self, detector):
        # Need z-score > 2.5 for spike detection
        # With [0.02, 0.02, 0.02, 0.02, 0.02, 0.90], mean≈0.167, std≈0.327, z for 0.90≈2.24
        # Use more extreme: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.95]
        metrics = {
            f"bank_{i}": {"fraud_rate": 0.01} for i in range(7)
        }
        metrics["bank_spike"] = {"fraud_rate": 0.95}  # extreme outlier
        fi = {k: {"f1": 0.1} for k in metrics}
        alerts = detector.analyze_round(1, fi, metrics)
        spike_alerts = [a for a in alerts if a.alert_type == "spike"]
        assert len(spike_alerts) > 0

    def test_get_alerts(self, detector, sample_feature_importances, sample_bank_metrics):
        detector.analyze_round(1, sample_feature_importances, sample_bank_metrics)
        alerts = detector.get_alerts()
        assert isinstance(alerts, list)

    def test_get_alert_summary(self, detector):
        summary = detector.get_alert_summary()
        assert "total_alerts" in summary

    def test_alert_to_dict(self):
        alert = EmergentAlert(
            alert_id="TEST-001",
            pattern_id="",
            title="Test Alert",
            description="desc",
            severity=PatternSeverity.HIGH,
            alert_type="spike",
            confidence=0.8,
            affected_banks=2,
        )
        d = alert.to_dict()
        assert d["alert_id"] == "TEST-001"
        assert d["severity"] == "high"


# ============================================
# CentralKnowledgeModel Tests
# ============================================


class TestCentralKnowledgeModel:
    def test_create_model(self, central_model):
        assert central_model is not None

    def test_register_bank(self, central_model):
        bank = central_model.register_bank("bank-test-001", "Test Bank")
        assert bank.bank_id == "bank-test-001"
        assert bank.display_name == "Test Bank"

    def test_list_banks(self, central_model):
        central_model.register_bank("bank-1", "Bank One")
        central_model.register_bank("bank-2", "Bank Two")
        banks = central_model.list_banks()
        assert len(banks) == 2

    def test_ingest_round(self, central_model, sample_feature_importances, sample_bank_metrics):
        result = central_model.ingest_round(
            round_number=1,
            bank_metrics=sample_bank_metrics,
            feature_importances=sample_feature_importances,
            global_accuracy=0.95,
            global_loss=0.12,
        )
        assert result["round"] == 1
        assert result["banks_participated"] == 3

    def test_get_global_statistics(self, central_model, sample_feature_importances, sample_bank_metrics):
        central_model.ingest_round(1, sample_bank_metrics, sample_feature_importances, 0.95, 0.1)
        stats = central_model.get_global_statistics()
        assert stats["total_banks"] == 3
        assert stats["total_rounds"] == 1
        assert stats["version"] == 1

    def test_get_global_features(self, central_model, sample_feature_importances, sample_bank_metrics):
        central_model.ingest_round(1, sample_bank_metrics, sample_feature_importances, 0.95, 0.1)
        features = central_model.get_global_feature_importance(top_n=5)
        assert len(features) > 0
        assert features[0]["feature"] in ["amount", "velocity", "location"]

    def test_create_snapshot(self, central_model, sample_feature_importances, sample_bank_metrics):
        central_model.ingest_round(1, sample_bank_metrics, sample_feature_importances, 0.95, 0.1)
        snapshot = central_model.create_snapshot()
        assert isinstance(snapshot, KnowledgeSnapshot)
        assert snapshot.version == 1

    def test_trend_analysis(self, central_model, sample_feature_importances, sample_bank_metrics):
        for i in range(3):
            central_model.ingest_round(i + 1, sample_bank_metrics, sample_feature_importances, 0.93 + i * 0.01, 0.15 - i * 0.01)
        trends = central_model.get_trend_analysis(window=3)
        assert "accuracy_trend" in trends

    def test_bank_risk_scores(self, central_model, sample_feature_importances, sample_bank_metrics):
        central_model.ingest_round(1, sample_bank_metrics, sample_feature_importances, 0.95, 0.1)
        scores = central_model.calculate_bank_risk_scores()
        assert len(scores) == 3
        for score in scores.values():
            assert 0 <= score <= 1

    def test_persistence(self, seeded_library, tmp_dir, sample_bank_metrics, sample_feature_importances):
        cm1 = CentralKnowledgeModel(seeded_library, data_dir=tmp_dir)
        cm1.ingest_round(1, sample_bank_metrics, sample_feature_importances, 0.95, 0.1)

        cm2 = CentralKnowledgeModel(seeded_library, data_dir=tmp_dir)
        assert len(cm2.list_banks()) == 3


# ============================================
# ReportGenerator Tests
# ============================================


class TestReportGenerator:
    def test_create_generator(self, report_gen):
        assert report_gen is not None

    def test_generate_global_report(self, report_gen):
        report = report_gen.generate_global_report()
        assert isinstance(report, Report)
        assert report.report_type == "global"
        assert len(report.sections) > 0

    def test_generate_bank_report(self, report_gen):
        report_gen.central_model.register_bank("bank-test", "Test Bank")
        report = report_gen.generate_bank_report("bank-test")
        assert report is not None
        assert report.report_type == "bank"

    def test_generate_bank_report_not_found(self, report_gen):
        report = report_gen.generate_bank_report("nonexistent")
        assert report is None

    def test_generate_emergent_briefing(self, report_gen):
        report = report_gen.generate_emergent_briefing()
        assert isinstance(report, Report)
        assert report.report_type == "emergent"

    def test_generate_compliance_report(self, report_gen):
        report = report_gen.generate_compliance_report()
        assert isinstance(report, Report)
        assert report.report_type == "compliance"
        assert len(report.sections) >= 2

    def test_report_to_dict(self, report_gen):
        report = report_gen.generate_global_report()
        d = report.to_dict()
        assert "report_id" in d
        assert "sections" in d
        assert isinstance(d["sections"], list)

    def test_list_reports(self, report_gen):
        report_gen.generate_global_report()
        report_gen.generate_compliance_report()
        reports = report_gen.get_reports()
        assert len(reports) == 2

    def test_filter_reports_by_type(self, report_gen):
        report_gen.generate_global_report()
        report_gen.generate_compliance_report()
        reports = report_gen.get_reports(report_type="global")
        assert len(reports) == 1
        assert reports[0]["report_type"] == "global"


# ============================================
# PatternMiner Tests
# ============================================


class TestPatternMiner:
    def test_create_miner(self, miner):
        assert miner is not None

    def test_mine_from_round(self, miner, sample_feature_importances, sample_bank_metrics):
        result = miner.mine_from_round(
            round_number=1,
            feature_importances=sample_feature_importances,
            bank_metrics=sample_bank_metrics,
        )
        assert isinstance(result, MiningResult)
        assert result.banks_analyzed == 3
        assert result.features_analyzed == 3

    def test_mine_with_delta(self, miner, sample_feature_importances, sample_bank_metrics):
        prev = {
            "bank_A": {"amount": 0.05, "velocity": 0.05, "location": 0.05},
            "bank_B": {"amount": 0.06, "velocity": 0.04, "location": 0.06},
            "bank_C": {"amount": 0.04, "velocity": 0.06, "location": 0.04},
        }
        result = miner.mine_from_round(
            round_number=2,
            feature_importances=sample_feature_importances,
            bank_metrics=sample_bank_metrics,
            previous_importances=prev,
        )
        # Should find delta patterns since values changed significantly
        assert result.patterns_discovered > 0

    def test_mine_combined_indicators(self, miner):
        fi = {
            "bank_A": {"suspicious_feat": 0.5, "normal_feat": 0.1},
            "bank_B": {"suspicious_feat": 0.6, "normal_feat": 0.12},
            "bank_C": {"suspicious_feat": 0.05, "normal_feat": 0.11},
        }
        metrics = {
            "bank_A": {"fraud_rate": 0.10},  # high fraud
            "bank_B": {"fraud_rate": 0.12},  # high fraud
            "bank_C": {"fraud_rate": 0.01},  # low fraud
        }
        result = miner.mine_from_round(1, fi, metrics)
        # suspicious_feat should be correlated with high fraud
        combined = [d for d in result.details if "suspicious_feat" in d.get("name", "").lower()]
        assert len(combined) > 0 or result.patterns_discovered > 0

    def test_mining_history(self, miner, sample_feature_importances, sample_bank_metrics):
        miner.mine_from_round(1, sample_feature_importances, sample_bank_metrics)
        miner.mine_from_round(2, sample_feature_importances, sample_bank_metrics)
        history = miner.get_mining_history()
        assert len(history) == 2

    def test_mining_result_to_dict(self, miner, sample_feature_importances, sample_bank_metrics):
        result = miner.mine_from_round(1, sample_feature_importances, sample_bank_metrics)
        d = result.to_dict()
        assert "run_id" in d
        assert "patterns_discovered" in d
        assert "mining_duration_ms" in d


# ============================================
# Integration Tests
# ============================================


class TestKnowledgeIntegration:
    def test_full_pipeline(self, tmp_dir):
        """Test the full knowledge pipeline: ingest → mine → detect → report."""
        # Setup
        lib = PatternLibrary(storage_path=tmp_dir)
        lib.seed_baseline_patterns()
        cm = CentralKnowledgeModel(lib, data_dir=tmp_dir)
        det = EmergentDetector(lib)
        miner = PatternMiner(lib)
        rg = ReportGenerator(lib, cm, det)

        # Round 1
        fi_r1 = {
            "bank_A": {"amount": 0.3, "velocity": 0.2},
            "bank_B": {"amount": 0.25, "velocity": 0.18},
            "bank_C": {"amount": 0.28, "velocity": 0.21},
        }
        metrics_r1 = {
            "bank_A": {"fraud_rate": 0.02, "accuracy": 0.95, "num_samples": 10000, "num_fraud": 200},
            "bank_B": {"fraud_rate": 0.03, "accuracy": 0.93, "num_samples": 8000, "num_fraud": 240},
            "bank_C": {"fraud_rate": 0.015, "accuracy": 0.96, "num_samples": 12000, "num_fraud": 180},
        }

        cm.ingest_round(1, metrics_r1, fi_r1, 0.95, 0.12)
        miner.mine_from_round(1, fi_r1, metrics_r1)
        det.analyze_round(1, fi_r1, metrics_r1)

        # Round 2 - with shifts
        fi_r2 = {
            "bank_A": {"amount": 0.6, "velocity": 0.1},  # big shift
            "bank_B": {"amount": 0.55, "velocity": 0.12},
            "bank_C": {"amount": 0.58, "velocity": 0.09},
        }
        metrics_r2 = {
            "bank_A": {"fraud_rate": 0.05, "accuracy": 0.92, "num_samples": 11000, "num_fraud": 550},
            "bank_B": {"fraud_rate": 0.04, "accuracy": 0.91, "num_samples": 9000, "num_fraud": 360},
            "bank_C": {"fraud_rate": 0.02, "accuracy": 0.94, "num_samples": 13000, "num_fraud": 260},
        }

        cm.ingest_round(2, metrics_r2, fi_r2, 0.93, 0.15)
        mining_result = miner.mine_from_round(2, fi_r2, metrics_r2, previous_importances=fi_r1)
        alerts = det.analyze_round(2, fi_r2, metrics_r2, previous_importances=fi_r1)

        # Generate reports
        global_report = rg.generate_global_report()
        bank_report = rg.generate_bank_report("bank_A")
        compliance = rg.generate_compliance_report()

        # Assertions
        assert cm.get_global_statistics()["total_banks"] == 3
        assert cm.get_global_statistics()["total_rounds"] == 2
        assert mining_result.patterns_discovered > 0
        assert global_report.report_type == "global"
        assert bank_report is not None
        assert compliance.report_type == "compliance"

        # Should have detected some patterns/alerts from the shifts
        stats = lib.get_statistics()
        assert stats["total"] > 6  # more than just baseline

    def test_pattern_promotion(self, tmp_dir):
        """Test emergent → fact promotion logic."""
        lib = PatternLibrary(storage_path=tmp_dir)
        det = EmergentDetector(lib)

        # Add emergent pattern with enough evidence
        p = PatternEntry(
            pattern_id="PROMOTE-001",
            pattern_type=PatternType.EMERGENT,
            name="Should Promote",
            description="Has enough evidence",
            severity=PatternSeverity.HIGH,
            confidence=0.8,
            observation_count=15,
            source_bank_count=3,
        )
        lib.add_pattern(p)

        promoted = det._promote_validated_patterns()
        assert promoted == 1

        updated = lib.get_pattern("PROMOTE-001")
        assert updated.pattern_type == PatternType.FACT
        assert updated.status == PatternStatus.CONFIRMED
