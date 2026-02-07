"""
SentinXFL - Pattern Miner
===========================

Automated pattern extraction from federated learning rounds.
Analyzes feature importance deltas, anomaly clustering, and
statistical outliers to discover new fraud patterns at scale.

Designed to handle huge datasets by working with aggregated
statistics rather than raw transaction data.

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from sentinxfl.core.logging import get_logger
from sentinxfl.intelligence.pattern_library import (
    PatternEntry,
    PatternLibrary,
    PatternSeverity,
    PatternType,
)

logger = get_logger(__name__)


@dataclass
class MiningResult:
    """Result of a pattern mining run."""

    run_id: str
    timestamp: str
    patterns_discovered: int
    patterns_updated: int
    features_analyzed: int
    banks_analyzed: int
    mining_duration_ms: float
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "patterns_discovered": self.patterns_discovered,
            "patterns_updated": self.patterns_updated,
            "features_analyzed": self.features_analyzed,
            "banks_analyzed": self.banks_analyzed,
            "mining_duration_ms": self.mining_duration_ms,
            "details": self.details,
        }


class PatternMiner:
    """
    Automated pattern mining from FL round aggregates.

    Mining strategies:
    1. Feature importance delta analysis
    2. Cross-bank outlier detection
    3. Temporal pattern correlation
    4. Combined indicator clustering
    """

    # Thresholds
    IMPORTANCE_DELTA_THRESHOLD = 0.3  # 30% change triggers mining
    OUTLIER_Z_THRESHOLD = 2.0
    MIN_FEATURE_IMPORTANCE = 0.01
    MAX_PATTERNS_PER_RUN = 50

    def __init__(self, library: PatternLibrary):
        self.library = library
        self._history: list[MiningResult] = []
        self._feature_baselines: dict[str, list[float]] = {}
        logger.info("PatternMiner initialized")

    def mine_from_round(
        self,
        round_number: int,
        feature_importances: dict[str, dict[str, float]],
        bank_metrics: dict[str, dict[str, float]],
        previous_importances: dict[str, dict[str, float]] | None = None,
    ) -> MiningResult:
        """
        Run full pattern mining pipeline on an FL round.

        Args:
            round_number: Current FL round
            feature_importances: {bank_id: {feature: importance}}
            bank_metrics: {bank_id: {metric: value}}
            previous_importances: Previous round's importances for delta analysis
        """
        start_time = datetime.utcnow()
        discovered = []
        updated = 0

        # Strategy 1: Feature importance delta analysis
        if previous_importances:
            delta_patterns = self._mine_feature_deltas(
                round_number, feature_importances, previous_importances
            )
            discovered.extend(delta_patterns)

        # Strategy 2: Cross-bank outlier detection
        outlier_patterns = self._mine_cross_bank_outliers(
            round_number, feature_importances, bank_metrics
        )
        discovered.extend(outlier_patterns)

        # Strategy 3: Feature consensus analysis
        consensus_patterns = self._mine_feature_consensus(
            round_number, feature_importances
        )
        discovered.extend(consensus_patterns)

        # Strategy 4: Combined indicator analysis
        combined_patterns = self._mine_combined_indicators(
            round_number, feature_importances, bank_metrics
        )
        discovered.extend(combined_patterns)

        # Deduplicate and limit
        discovered = self._deduplicate(discovered)[:self.MAX_PATTERNS_PER_RUN]

        # Add to library
        for pattern in discovered:
            self.library.add_pattern(pattern)

        # Update feature baselines
        self._update_baselines(feature_importances)

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        result = MiningResult(
            run_id=f"MINE-R{round_number}-{datetime.utcnow().strftime('%H%M%S')}",
            timestamp=datetime.utcnow().isoformat(),
            patterns_discovered=len(discovered),
            patterns_updated=updated,
            features_analyzed=len(set().union(*(fi.keys() for fi in feature_importances.values())) if feature_importances else set()),
            banks_analyzed=len(bank_metrics),
            mining_duration_ms=round(elapsed, 2),
            details=[
                {"id": p.pattern_id, "name": p.name, "type": p.pattern_type.value}
                for p in discovered
            ],
        )

        self._history.append(result)
        logger.info(
            "Mining round %d: %d new patterns (%.1fms)",
            round_number, len(discovered), elapsed,
        )
        return result

    # ----------------------------------------------------------
    # Mining Strategies
    # ----------------------------------------------------------

    def _mine_feature_deltas(
        self,
        round_number: int,
        current: dict[str, dict[str, float]],
        previous: dict[str, dict[str, float]],
    ) -> list[PatternEntry]:
        """Strategy 1: Find features with large importance changes."""
        patterns = []
        all_features: set[str] = set()
        for imps in list(current.values()) + list(previous.values()):
            all_features.update(imps.keys())

        for feat in all_features:
            curr_vals = [imps.get(feat, 0) for imps in current.values()]
            prev_vals = [imps.get(feat, 0) for imps in previous.values()]

            curr_mean = float(np.mean(curr_vals)) if curr_vals else 0
            prev_mean = float(np.mean(prev_vals)) if prev_vals else 0

            if prev_mean < self.MIN_FEATURE_IMPORTANCE and curr_mean < self.MIN_FEATURE_IMPORTANCE:
                continue

            delta = abs(curr_mean - prev_mean) / (max(prev_mean, self.MIN_FEATURE_IMPORTANCE))

            if delta >= self.IMPORTANCE_DELTA_THRESHOLD:
                direction = "increased" if curr_mean > prev_mean else "decreased"
                severity = (
                    PatternSeverity.HIGH if delta > 0.7
                    else PatternSeverity.MEDIUM
                )
                novelty = min(delta, 1.0)

                p = PatternEntry(
                    pattern_id=f"DELTA-R{round_number}-{feat[:12]}",
                    pattern_type=PatternType.EMERGENT if delta > 0.7 else PatternType.VARIANT,
                    name=f"Feature Delta: {feat} ({direction})",
                    description=(
                        f"Feature '{feat}' importance {direction} by {delta:.1%} "
                        f"(from {prev_mean:.4f} to {curr_mean:.4f}) across {len(current)} banks."
                    ),
                    severity=severity,
                    confidence=round(min(0.5 + delta * 0.3, 0.9), 3),
                    observation_count=len(current),
                    source_bank_count=len(current),
                    feature_signature={feat: round(curr_mean, 6)},
                    attack_vector="feature_delta",
                    tags=["delta", direction, f"round-{round_number}"],
                )
                patterns.append(p)

        return patterns

    def _mine_cross_bank_outliers(
        self,
        round_number: int,
        feature_importances: dict[str, dict[str, float]],
        bank_metrics: dict[str, dict[str, float]],
    ) -> list[PatternEntry]:
        """Strategy 2: Find banks with outlier feature importance profiles."""
        patterns = []
        if len(feature_importances) < 3:
            return patterns

        all_features: set[str] = set()
        for imps in feature_importances.values():
            all_features.update(imps.keys())

        for feat in all_features:
            vals = [(bid, imps.get(feat, 0)) for bid, imps in feature_importances.items()]
            values = [v for _, v in vals]
            if not values:
                continue

            mean = float(np.mean(values))
            std = float(np.std(values))

            if std < 1e-6:
                continue

            outlier_banks = [
                (bid, val, (val - mean) / std)
                for bid, val in vals
                if abs(val - mean) > self.OUTLIER_Z_THRESHOLD * std
            ]

            if outlier_banks:
                p = PatternEntry(
                    pattern_id=f"OUTLIER-R{round_number}-{feat[:12]}",
                    pattern_type=PatternType.VARIANT,
                    name=f"Cross-Bank Outlier: {feat}",
                    description=(
                        f"Feature '{feat}' shows {len(outlier_banks)} outlier bank(s) "
                        f"with importance deviating >2σ from mean ({mean:.4f}). "
                        f"May indicate localized attack targeting specific institutions."
                    ),
                    severity=PatternSeverity.MEDIUM,
                    confidence=round(min(0.4 + len(outlier_banks) * 0.1, 0.8), 3),
                    observation_count=len(feature_importances),
                    source_bank_count=len(outlier_banks),
                    feature_signature={feat: round(mean, 6)},
                    attack_vector="cross_bank_outlier",
                    tags=["outlier", feat, f"round-{round_number}"],
                )
                patterns.append(p)

        return patterns

    def _mine_feature_consensus(
        self,
        round_number: int,
        feature_importances: dict[str, dict[str, float]],
    ) -> list[PatternEntry]:
        """Strategy 3: Find features that are unanimously important."""
        patterns = []
        if len(feature_importances) < 2:
            return patterns

        all_features: set[str] = set()
        for imps in feature_importances.values():
            all_features.update(imps.keys())

        for feat in all_features:
            vals = [imps.get(feat, 0) for imps in feature_importances.values()]
            if not vals:
                continue

            mean = float(np.mean(vals))
            # Low variance + high mean = consensus important feature
            cv = float(np.std(vals)) / (mean + 1e-9)  # coefficient of variation

            if mean > 0.05 and cv < 0.3:
                # Already known as a baseline feature → check if it's in baselines
                baseline = self._feature_baselines.get(feat, [])
                if baseline and abs(mean - np.mean(baseline)) < 0.1 * mean:
                    continue  # Known stable feature, skip

                p = PatternEntry(
                    pattern_id=f"CONSENSUS-R{round_number}-{feat[:12]}",
                    pattern_type=PatternType.FACT,
                    name=f"Consensus Feature: {feat}",
                    description=(
                        f"Feature '{feat}' shows strong consensus importance ({mean:.4f}) "
                        f"with low variance (CV={cv:.3f}) across {len(feature_importances)} banks. "
                        f"This is a reliable fraud indicator."
                    ),
                    severity=PatternSeverity.LOW,
                    confidence=round(min(0.7 + (1 - cv) * 0.2, 0.95), 3),
                    observation_count=len(feature_importances),
                    source_bank_count=len(feature_importances),
                    feature_signature={feat: round(mean, 6)},
                    attack_vector="consensus",
                    tags=["consensus", feat, f"round-{round_number}"],
                )
                patterns.append(p)

        return patterns

    def _mine_combined_indicators(
        self,
        round_number: int,
        feature_importances: dict[str, dict[str, float]],
        bank_metrics: dict[str, dict[str, float]],
    ) -> list[PatternEntry]:
        """Strategy 4: Correlate feature spikes with fraud rate spikes."""
        patterns = []
        if len(bank_metrics) < 3:
            return patterns

        fraud_rates = {bid: m.get("fraud_rate", 0) for bid, m in bank_metrics.items()}
        mean_fr = float(np.mean(list(fraud_rates.values())))

        # Find banks with high fraud rates
        high_fraud_banks = {bid for bid, fr in fraud_rates.items() if fr > mean_fr * 1.5}

        if not high_fraud_banks:
            return patterns

        # Find features that are specifically important for high-fraud banks
        for feat in set().union(*(fi.keys() for fi in feature_importances.values())):
            high_vals = [
                feature_importances[bid].get(feat, 0)
                for bid in high_fraud_banks
                if bid in feature_importances
            ]
            low_vals = [
                feature_importances[bid].get(feat, 0)
                for bid in feature_importances
                if bid not in high_fraud_banks
            ]

            if not high_vals or not low_vals:
                continue

            high_mean = float(np.mean(high_vals))
            low_mean = float(np.mean(low_vals))

            ratio = high_mean / (low_mean + 1e-9)
            if ratio > 2.0:
                p = PatternEntry(
                    pattern_id=f"COMBINED-R{round_number}-{feat[:12]}",
                    pattern_type=PatternType.EMERGENT,
                    name=f"Fraud-Correlated Feature: {feat}",
                    description=(
                        f"Feature '{feat}' is {ratio:.1f}x more important in banks "
                        f"with elevated fraud rates (>{mean_fr*1.5:.4f}). "
                        f"Strong indicator of targeted attack vector."
                    ),
                    severity=PatternSeverity.HIGH,
                    confidence=round(min(0.5 + (ratio - 2) * 0.1, 0.9), 3),
                    observation_count=len(bank_metrics),
                    source_bank_count=len(high_fraud_banks),
                    feature_signature={feat: round(high_mean, 6)},
                    attack_vector="combined",
                    tags=["combined", feat, f"round-{round_number}"],
                )
                patterns.append(p)

        return patterns

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------

    def _deduplicate(self, patterns: list[PatternEntry]) -> list[PatternEntry]:
        """Remove duplicate patterns based on title similarity."""
        seen_titles: set[str] = set()
        unique = []
        for p in patterns:
            key = p.name.lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(p)
        return unique

    def _update_baselines(self, feature_importances: dict[str, dict[str, float]]):
        """Update rolling baselines for feature importances."""
        for imps in feature_importances.values():
            for feat, val in imps.items():
                if feat not in self._feature_baselines:
                    self._feature_baselines[feat] = []
                self._feature_baselines[feat].append(val)
                if len(self._feature_baselines[feat]) > 100:
                    self._feature_baselines[feat] = self._feature_baselines[feat][-100:]

    def get_mining_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent mining run history."""
        return [r.to_dict() for r in self._history[-limit:]]
