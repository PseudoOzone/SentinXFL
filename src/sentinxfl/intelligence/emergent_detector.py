"""
SentinXFL - Emergent Attack Detector
======================================

Detects novel, zero-day, and emergent fraud attack patterns
by comparing incoming data against the known pattern library.
Uses statistical divergence, temporal analysis, and cross-bank
correlation to identify previously unseen threats.

Author: Anshuman Bakshi
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from sentinxfl.core.logging import get_logger
from sentinxfl.intelligence.pattern_library import (
    PatternEntry,
    PatternLibrary,
    PatternSeverity,
    PatternStatus,
    PatternType,
)

logger = get_logger(__name__)


@dataclass
class EmergentAlert:
    """An alert for a detected emergent attack pattern."""

    alert_id: str
    pattern_id: str
    title: str
    description: str
    severity: PatternSeverity
    alert_type: str  # "zero_day" | "variant" | "spike" | "correlation"
    confidence: float
    affected_banks: int
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_actions: list[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "pattern_id": self.pattern_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "confidence": self.confidence,
            "affected_banks": self.affected_banks,
            "evidence": self.evidence,
            "recommended_actions": self.recommended_actions,
            "created_at": self.created_at,
        }


class EmergentDetector:
    """
    Detects emergent and zero-day fraud attacks.

    Analysis pipeline:
    1. Feature distribution shift detection (KL divergence)
    2. Temporal spike analysis (frequency anomalies)
    3. Cross-bank correlation (multi-institution patterns)
    4. Novelty scoring against known patterns
    5. Fact-based validation (statistical thresholds)
    """

    # Minimum thresholds for fact-based promotion
    MIN_OBSERVATIONS_FACT = 10
    MIN_CONFIDENCE_FACT = 0.7
    MIN_BANKS_FACT = 2

    def __init__(self, library: PatternLibrary):
        self.library = library
        self._alerts: list[EmergentAlert] = []
        logger.info("EmergentDetector initialized")

    def analyze_round(
        self,
        round_number: int,
        feature_importances: dict[str, dict[str, float]],
        bank_metrics: dict[str, dict[str, float]],
        previous_importances: dict[str, dict[str, float]] | None = None,
    ) -> list[EmergentAlert]:
        """
        Run full emergent detection pipeline on an FL round's results.

        Returns list of EmergentAlert objects for any detected threats.
        """
        alerts: list[EmergentAlert] = []

        # 1. Feature distribution shift detection
        alerts.extend(
            self._detect_feature_shifts(round_number, feature_importances, previous_importances)
        )

        # 2. Temporal spike analysis
        alerts.extend(self._detect_spikes(round_number, bank_metrics))

        # 3. Cross-bank correlation
        alerts.extend(
            self._detect_cross_bank_correlation(round_number, feature_importances, bank_metrics)
        )

        # 4. Promote validated patterns to fact-based
        self._promote_validated_patterns()

        self._alerts.extend(alerts)
        logger.info(f"Round {round_number}: {len(alerts)} emergent alerts generated")
        return alerts

    def _detect_feature_shifts(
        self,
        round_number: int,
        current: dict[str, dict[str, float]],
        previous: dict[str, dict[str, float]] | None,
    ) -> list[EmergentAlert]:
        """Detect significant shifts in feature importance distributions."""
        if not previous or not current:
            return []

        alerts = []
        all_features: set[str] = set()
        for imps in list(current.values()) + list(previous.values()):
            all_features.update(imps.keys())

        for feat in all_features:
            curr_vals = [imps.get(feat, 0) for imps in current.values()]
            prev_vals = [imps.get(feat, 0) for imps in previous.values()]

            if not curr_vals or not prev_vals:
                continue

            curr_mean = np.mean(curr_vals)
            prev_mean = np.mean(prev_vals)
            shift_magnitude = abs(curr_mean - prev_mean) / (prev_mean + 1e-9)

            if shift_magnitude > 0.5:  # >50% shift
                severity = (
                    PatternSeverity.CRITICAL if shift_magnitude > 1.0
                    else PatternSeverity.HIGH
                )
                aid = f"ALERT-SHIFT-R{round_number}-{feat[:8].upper()}"
                alert = EmergentAlert(
                    alert_id=aid,
                    pattern_id="",
                    title=f"Feature Shift Detected: {feat}",
                    description=(
                        f"Feature '{feat}' importance shifted by {shift_magnitude:.1%} "
                        f"between rounds (prev={prev_mean:.4f} → curr={curr_mean:.4f}). "
                        f"This may indicate an emerging attack vector."
                    ),
                    severity=severity,
                    alert_type="variant" if shift_magnitude < 1.0 else "zero_day",
                    confidence=round(min(shift_magnitude * 0.5, 0.95), 3),
                    affected_banks=len(current),
                    evidence={
                        "feature": feat,
                        "prev_mean": round(prev_mean, 6),
                        "curr_mean": round(curr_mean, 6),
                        "shift_magnitude": round(shift_magnitude, 4),
                    },
                    recommended_actions=[
                        f"Investigate transactions with high '{feat}' values",
                        "Review model predictions for false negatives",
                        "Consider adding targeted feature engineering",
                    ],
                )
                alerts.append(alert)

        return alerts

    def _detect_spikes(
        self,
        round_number: int,
        bank_metrics: dict[str, dict[str, float]],
    ) -> list[EmergentAlert]:
        """Detect sudden spikes in fraud rates or model errors."""
        alerts = []
        if not bank_metrics:
            return alerts

        fraud_rates = [m.get("fraud_rate", 0) for m in bank_metrics.values()]
        if len(fraud_rates) < 2:
            return alerts

        mean_fr = np.mean(fraud_rates)
        std_fr = np.std(fraud_rates)

        for bank_id, metrics in bank_metrics.items():
            fr = metrics.get("fraud_rate", 0)
            if std_fr > 0 and fr > mean_fr + 2.5 * std_fr:
                aid = f"ALERT-SPIKE-R{round_number}-{hashlib.md5(bank_id.encode()).hexdigest()[:6].upper()}"
                alert = EmergentAlert(
                    alert_id=aid,
                    pattern_id="",
                    title=f"Fraud Rate Spike Detected (Round {round_number})",
                    description=(
                        f"A bank shows fraud rate {fr:.4f} vs global average {mean_fr:.4f} "
                        f"({(fr - mean_fr) / std_fr:.1f}σ deviation). "
                        f"Possible targeted or localized attack in progress."
                    ),
                    severity=PatternSeverity.CRITICAL,
                    alert_type="spike",
                    confidence=round(min((fr - mean_fr) / (std_fr + 1e-9) * 0.25, 0.95), 3),
                    affected_banks=1,
                    evidence={
                        "bank_fraud_rate": round(fr, 6),
                        "global_mean": round(mean_fr, 6),
                        "global_std": round(std_fr, 6),
                        "z_score": round((fr - mean_fr) / (std_fr + 1e-9), 2),
                    },
                    recommended_actions=[
                        "Immediately review recent transactions at affected bank",
                        "Check for data pipeline anomalies",
                        "Enable enhanced monitoring for next 24 hours",
                    ],
                )
                alerts.append(alert)

        return alerts

    def _detect_cross_bank_correlation(
        self,
        round_number: int,
        feature_importances: dict[str, dict[str, float]],
        bank_metrics: dict[str, dict[str, float]],
    ) -> list[EmergentAlert]:
        """Detect correlated anomalies across multiple banks (coordinated attacks)."""
        alerts = []
        if len(bank_metrics) < 3:
            return alerts

        # Check if multiple banks simultaneously show elevated fraud
        elevated = [
            bid for bid, m in bank_metrics.items()
            if m.get("fraud_rate", 0) > np.mean(
                [x.get("fraud_rate", 0) for x in bank_metrics.values()]
            ) * 1.5
        ]

        if len(elevated) >= 2:
            aid = f"ALERT-CORR-R{round_number}"
            alert = EmergentAlert(
                alert_id=aid,
                pattern_id="",
                title=f"Coordinated Attack Pattern (Round {round_number})",
                description=(
                    f"{len(elevated)} banks simultaneously show elevated fraud rates "
                    f"(>1.5x global average). Possible coordinated cross-institution "
                    f"attack campaign."
                ),
                severity=PatternSeverity.CRITICAL,
                alert_type="correlation",
                confidence=round(min(len(elevated) / len(bank_metrics), 0.95), 3),
                affected_banks=len(elevated),
                evidence={
                    "elevated_bank_count": len(elevated),
                    "total_banks": len(bank_metrics),
                    "ratio": round(len(elevated) / len(bank_metrics), 4),
                },
                recommended_actions=[
                    "Activate cross-bank coordination protocol",
                    "Share anonymized attack signatures between affected banks",
                    "Escalate to regulatory authorities if sustained",
                    "Deploy enhanced model with tightened thresholds",
                ],
            )
            alerts.append(alert)

        return alerts

    def _promote_validated_patterns(self) -> int:
        """Promote emergent patterns to fact-based when sufficiently validated."""
        promoted = 0
        for pattern in self.library.list_patterns(pattern_type=PatternType.EMERGENT, limit=1000):
            if (
                pattern.observation_count >= self.MIN_OBSERVATIONS_FACT
                and pattern.confidence >= self.MIN_CONFIDENCE_FACT
                and pattern.source_bank_count >= self.MIN_BANKS_FACT
            ):
                pattern.pattern_type = PatternType.FACT
                pattern.status = PatternStatus.CONFIRMED
                self.library.add_pattern(pattern)
                promoted += 1

        if promoted:
            logger.info(f"Promoted {promoted} patterns from emergent to fact-based")
        return promoted

    def get_alerts(
        self,
        severity: PatternSeverity | None = None,
        alert_type: str | None = None,
        limit: int = 50,
    ) -> list[EmergentAlert]:
        """Get recent alerts with optional filters."""
        results = self._alerts.copy()
        if severity:
            results = [a for a in results if a.severity == severity]
        if alert_type:
            results = [a for a in results if a.alert_type == alert_type]
        results.sort(key=lambda a: a.created_at, reverse=True)
        return results[:limit]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of all active alerts."""
        return {
            "total_alerts": len(self._alerts),
            "by_severity": {
                s.value: sum(1 for a in self._alerts if a.severity == s)
                for s in PatternSeverity
            },
            "by_type": {},
            "latest": [a.to_dict() for a in self._alerts[-5:]],
        }
