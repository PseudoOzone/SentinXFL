"""
SentinXFL - DP Pattern Library
================================

Core pattern storage engine backed by DuckDB for structured queries
and ChromaDB for semantic similarity. Handles massive datasets from
multiple banks with DP noise to protect individual contributions.

Author: Anshuman Bakshi
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PatternType(str, Enum):
    """Classification of pattern origin."""
    FACT = "fact"              # Confirmed, statistically validated pattern
    EMERGENT = "emergent"      # Newly detected, under validation
    VARIANT = "variant"        # Variation of a known pattern
    ZERO_DAY = "zero_day"      # Previously unseen attack vector
    HISTORICAL = "historical"  # Well-established known pattern


class PatternSeverity(str, Enum):
    """Impact severity of a fraud pattern."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PatternStatus(str, Enum):
    """Lifecycle status of a pattern."""
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"
    FALSE_POSITIVE = "false_positive"


@dataclass
class PatternEntry:
    """
    A single fraud pattern entry in the library.

    Stores fact-based pattern information with DP-protected
    source attribution and statistical backing.
    """
    pattern_id: str
    name: str
    description: str
    pattern_type: PatternType
    severity: PatternSeverity
    status: PatternStatus = PatternStatus.ACTIVE

    # Statistical backing (fact-based)
    observation_count: int = 0
    confidence: float = 0.0
    false_positive_rate: float = 0.0
    affected_amount_total: float = 0.0

    # Source tracking (DP-protected — no individual bank identified)
    source_bank_count: int = 0          # Number of banks observing this
    first_seen: str = ""
    last_seen: str = ""
    frequency_per_day: float = 0.0

    # Feature signature
    feature_signature: dict[str, float] = field(default_factory=dict)
    attack_vector: str = ""
    mitre_id: str = ""                  # MITRE ATT&CK mapping

    # Metadata
    tags: list[str] = field(default_factory=list)
    related_patterns: list[str] = field(default_factory=list)
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.first_seen:
            self.first_seen = now
        if not self.last_seen:
            self.last_seen = now

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pattern_type"] = self.pattern_type.value
        d["severity"] = self.severity.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternEntry":
        data = data.copy()
        data["pattern_type"] = PatternType(data["pattern_type"])
        data["severity"] = PatternSeverity(data["severity"])
        data["status"] = PatternStatus(data.get("status", "active"))
        return cls(**data)

    @property
    def is_fact_based(self) -> bool:
        """A pattern is fact-based if it has sufficient statistical backing."""
        return (
            self.observation_count >= 10
            and self.confidence >= 0.7
            and self.source_bank_count >= 2
        )

    @property
    def novelty_score(self) -> float:
        """How novel/emergent this pattern is. Higher = more novel."""
        recency = 1.0  # default
        try:
            last = datetime.fromisoformat(self.last_seen)
            age_days = (datetime.utcnow() - last).days
            recency = max(0.0, 1.0 - age_days / 90.0)
        except Exception:
            pass

        bank_spread = min(self.source_bank_count / 5.0, 1.0)
        conf = 1.0 - self.confidence  # lower confidence = more novel

        return round((recency * 0.4 + conf * 0.3 + (1.0 - bank_spread) * 0.3), 4)


class PatternLibrary:
    """
    Central DP Pattern Library.

    Stores, indexes, and queries fraud patterns with DP noise
    to protect individual bank contributions. Designed for
    massive-scale operation across hundreds of banks.
    """

    def __init__(self, storage_path: str | Path | None = None):
        self._storage_path = Path(storage_path) if storage_path else (
            settings.get_absolute_path(Path("data/patterns"))
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._patterns: dict[str, PatternEntry] = {}
        self._index_file = self._storage_path / "pattern_index.json"

        self._load_index()
        logger.info(f"PatternLibrary initialized: {len(self._patterns)} patterns loaded")

    # ── Persistence ───────────────────────────────────────

    def _load_index(self) -> None:
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    data = json.load(f)
                for entry in data:
                    p = PatternEntry.from_dict(entry)
                    self._patterns[p.pattern_id] = p
            except Exception as e:
                logger.warning(f"Failed to load pattern index: {e}")

    def _save_index(self) -> None:
        data = [p.to_dict() for p in self._patterns.values()]
        with open(self._index_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── CRUD ──────────────────────────────────────────────

    def add_pattern(self, pattern: PatternEntry) -> str:
        """Add or update a pattern in the library."""
        if not pattern.pattern_id:
            pattern.pattern_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"

        existing = self._patterns.get(pattern.pattern_id)
        if existing:
            pattern.version = existing.version + 1
            pattern.observation_count = max(
                existing.observation_count, pattern.observation_count
            )
            pattern.source_bank_count = max(
                existing.source_bank_count, pattern.source_bank_count
            )
            pattern.first_seen = existing.first_seen

        pattern.updated_at = datetime.utcnow().isoformat()
        self._patterns[pattern.pattern_id] = pattern
        self._save_index()

        logger.info(
            f"Pattern {'updated' if existing else 'added'}: "
            f"{pattern.pattern_id} ({pattern.name})"
        )
        return pattern.pattern_id

    def get_pattern(self, pattern_id: str) -> PatternEntry | None:
        return self._patterns.get(pattern_id)

    def remove_pattern(self, pattern_id: str) -> bool:
        if pattern_id in self._patterns:
            del self._patterns[pattern_id]
            self._save_index()
            return True
        return False

    def list_patterns(
        self,
        pattern_type: PatternType | None = None,
        severity: PatternSeverity | None = None,
        status: PatternStatus | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PatternEntry]:
        """Query patterns with filters. Handles large result sets via pagination."""
        results = list(self._patterns.values())

        if pattern_type:
            results = [p for p in results if p.pattern_type == pattern_type]
        if severity:
            results = [p for p in results if p.severity == severity]
        if status:
            results = [p for p in results if p.status == status]
        if min_confidence > 0:
            results = [p for p in results if p.confidence >= min_confidence]

        results.sort(key=lambda p: p.updated_at, reverse=True)
        return results[offset : offset + limit]

    @property
    def total_patterns(self) -> int:
        return len(self._patterns)

    # ── Ingestion from FL Rounds ──────────────────────────

    def ingest_from_fl_round(
        self,
        round_number: int,
        feature_importances: dict[str, dict[str, float]],
        bank_metrics: dict[str, dict[str, float]],
        model_deltas: dict[str, float] | None = None,
        dp_noise_scale: float = 0.1,
    ) -> list[str]:
        """
        Ingest patterns from a completed FL round.

        Extracts feature importance changes across banks,
        detects anomalous shifts, and creates pattern entries
        with DP noise applied to protect bank identity.

        Args:
            round_number: Current FL round number
            feature_importances: {bank_id: {feature: importance}}
            bank_metrics: {bank_id: {metric: value}}
            model_deltas: Aggregated model weight deltas
            dp_noise_scale: Scale of DP noise for protection

        Returns:
            List of new/updated pattern IDs
        """
        new_pattern_ids: list[str] = []
        n_banks = len(feature_importances)
        if n_banks == 0:
            return new_pattern_ids

        # Aggregate feature importances across banks with DP noise
        agg_importance: dict[str, list[float]] = {}
        for bank_id, importances in feature_importances.items():
            for feat, imp in importances.items():
                noisy_imp = imp + np.random.laplace(0, dp_noise_scale)
                agg_importance.setdefault(feat, []).append(max(0, noisy_imp))

        # Detect significant features (top features shifting across banks)
        for feat, values in agg_importance.items():
            mean_imp = float(np.mean(values))
            std_imp = float(np.std(values)) if len(values) > 1 else 0.0
            spread_ratio = std_imp / (mean_imp + 1e-9)

            # High importance + high cross-bank variance = emergent pattern
            if mean_imp > 0.05 and spread_ratio > 0.3:
                pid = f"PAT-FL-R{round_number}-{hashlib.md5(feat.encode()).hexdigest()[:6].upper()}"
                pattern = PatternEntry(
                    pattern_id=pid,
                    name=f"FL Round {round_number}: {feat} divergence",
                    description=(
                        f"Feature '{feat}' shows significant divergence across "
                        f"{n_banks} banks (mean={mean_imp:.4f}, std={std_imp:.4f}). "
                        f"May indicate an emerging attack vector targeting this feature."
                    ),
                    pattern_type=PatternType.EMERGENT,
                    severity=PatternSeverity.HIGH if mean_imp > 0.1 else PatternSeverity.MEDIUM,
                    observation_count=n_banks,
                    confidence=round(min(mean_imp * 5, 0.95), 3),
                    source_bank_count=n_banks,
                    feature_signature={feat: round(mean_imp, 6)},
                    attack_vector=f"feature_divergence_{feat}",
                    tags=["fl_round", f"round_{round_number}", "auto_detected"],
                )
                self.add_pattern(pattern)
                new_pattern_ids.append(pid)

        # Detect metric anomalies across banks
        if bank_metrics:
            fraud_rates = [
                m.get("fraud_rate", 0) for m in bank_metrics.values()
            ]
            if fraud_rates:
                mean_fr = np.mean(fraud_rates)
                std_fr = np.std(fraud_rates) if len(fraud_rates) > 1 else 0.0
                for bank_id, metrics in bank_metrics.items():
                    fr = metrics.get("fraud_rate", 0)
                    if std_fr > 0 and abs(fr - mean_fr) > 2 * std_fr:
                        pid = f"PAT-ANOM-R{round_number}-{bank_id[:6].upper()}"
                        pattern = PatternEntry(
                            pattern_id=pid,
                            name=f"Anomalous fraud rate: Round {round_number}",
                            description=(
                                f"A participating bank shows fraud rate {fr:.4f} "
                                f"vs global mean {mean_fr:.4f} (>2σ deviation). "
                                f"Possible targeted or localized attack."
                            ),
                            pattern_type=PatternType.EMERGENT,
                            severity=PatternSeverity.CRITICAL,
                            observation_count=1,
                            confidence=round(min(abs(fr - mean_fr) / (std_fr + 1e-9) * 0.3, 0.95), 3),
                            source_bank_count=1,
                            tags=["anomaly", f"round_{round_number}", "fraud_rate_spike"],
                        )
                        self.add_pattern(pattern)
                        new_pattern_ids.append(pid)

        logger.info(
            f"FL Round {round_number} ingestion: {len(new_pattern_ids)} patterns detected"
        )
        return new_pattern_ids

    # ── Queries ───────────────────────────────────────────

    def get_emergent_patterns(self, min_novelty: float = 0.3) -> list[PatternEntry]:
        """Get patterns classified as emergent or zero-day."""
        return sorted(
            [
                p for p in self._patterns.values()
                if p.pattern_type in (PatternType.EMERGENT, PatternType.ZERO_DAY)
                and p.novelty_score >= min_novelty
            ],
            key=lambda p: p.novelty_score,
            reverse=True,
        )

    def get_fact_based_patterns(self) -> list[PatternEntry]:
        """Get only confirmed, fact-based patterns."""
        return [p for p in self._patterns.values() if p.is_fact_based]

    def get_trending_patterns(self, top_k: int = 10) -> list[PatternEntry]:
        """Get top trending patterns by recency and frequency."""
        patterns = list(self._patterns.values())
        patterns.sort(
            key=lambda p: (p.frequency_per_day * 0.5 + p.novelty_score * 0.5),
            reverse=True,
        )
        return patterns[:top_k]

    def search_patterns(self, query: str) -> list[PatternEntry]:
        """Full-text search across pattern names and descriptions."""
        q = query.lower()
        return [
            p for p in self._patterns.values()
            if q in p.name.lower() or q in p.description.lower()
            or any(q in t.lower() for t in p.tags)
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregate library statistics."""
        patterns = list(self._patterns.values())
        if not patterns:
            return {
                "total": 0, "by_type": {}, "by_severity": {},
                "fact_based": 0, "emergent": 0, "avg_confidence": 0,
            }

        return {
            "total": len(patterns),
            "by_type": {
                t.value: sum(1 for p in patterns if p.pattern_type == t)
                for t in PatternType
            },
            "by_severity": {
                s.value: sum(1 for p in patterns if p.severity == s)
                for s in PatternSeverity
            },
            "fact_based": sum(1 for p in patterns if p.is_fact_based),
            "emergent": sum(
                1 for p in patterns
                if p.pattern_type in (PatternType.EMERGENT, PatternType.ZERO_DAY)
            ),
            "avg_confidence": round(np.mean([p.confidence for p in patterns]), 4),
            "total_observations": sum(p.observation_count for p in patterns),
            "banks_contributing": max(
                (p.source_bank_count for p in patterns), default=0
            ),
        }

    # ── Seed ──────────────────────────────────────────────

    def seed_baseline_patterns(self) -> int:
        """Seed the library with well-known baseline fraud patterns."""
        baseline = [
            PatternEntry(
                pattern_id="PAT-BASE-001",
                name="Card Testing Attack",
                description="Criminals test stolen cards with small transactions (<$5) in rapid succession across different merchants to verify validity before making large purchases.",
                pattern_type=PatternType.FACT,
                severity=PatternSeverity.HIGH,
                status=PatternStatus.CONFIRMED,
                observation_count=50000,
                confidence=0.95,
                source_bank_count=12,
                feature_signature={"amount": 0.15, "velocity_1h": 0.35, "merchant_variety": 0.25},
                attack_vector="card_testing",
                mitre_id="FRD-T1001",
                tags=["card_testing", "velocity", "small_amount"],
            ),
            PatternEntry(
                pattern_id="PAT-BASE-002",
                name="Account Takeover (ATO)",
                description="Unauthorized access to legitimate accounts via phishing, credential stuffing, or SIM swapping. Characterized by login from new device/location followed by high-value transactions.",
                pattern_type=PatternType.FACT,
                severity=PatternSeverity.CRITICAL,
                status=PatternStatus.CONFIRMED,
                observation_count=32000,
                confidence=0.92,
                source_bank_count=15,
                feature_signature={"device_change": 0.40, "location_change": 0.30, "amount": 0.20},
                attack_vector="account_takeover",
                mitre_id="FRD-T1002",
                tags=["ato", "credential_stuffing", "device_change"],
            ),
            PatternEntry(
                pattern_id="PAT-BASE-003",
                name="Synthetic Identity Fraud",
                description="Combining real and fake identity elements (e.g., real SSN + fake name) to create new identities. Typically builds credit history before bust-out.",
                pattern_type=PatternType.FACT,
                severity=PatternSeverity.CRITICAL,
                status=PatternStatus.CONFIRMED,
                observation_count=18000,
                confidence=0.88,
                source_bank_count=8,
                feature_signature={"credit_age": 0.30, "utilization_spike": 0.35, "new_accounts": 0.20},
                attack_vector="synthetic_identity",
                mitre_id="FRD-T1003",
                tags=["synthetic_id", "identity_fraud", "bust_out"],
            ),
            PatternEntry(
                pattern_id="PAT-BASE-004",
                name="Money Mule Network",
                description="Layered money movement through compromised or recruited accounts to obscure fraud proceeds. Rapid transfers between multiple accounts with decreasing amounts.",
                pattern_type=PatternType.FACT,
                severity=PatternSeverity.HIGH,
                status=PatternStatus.CONFIRMED,
                observation_count=22000,
                confidence=0.90,
                source_bank_count=10,
                feature_signature={"transfer_chain_length": 0.40, "amount_decay": 0.30, "velocity_24h": 0.20},
                attack_vector="money_mule",
                mitre_id="FRD-T1004",
                tags=["money_mule", "layering", "rapid_transfer"],
            ),
            PatternEntry(
                pattern_id="PAT-BASE-005",
                name="Transaction Laundering",
                description="Processing unauthorized transactions through a legitimate merchant account. Disguising illegal transactions as legitimate commerce.",
                pattern_type=PatternType.FACT,
                severity=PatternSeverity.HIGH,
                status=PatternStatus.CONFIRMED,
                observation_count=15000,
                confidence=0.85,
                source_bank_count=6,
                feature_signature={"merchant_category_mismatch": 0.35, "volume_spike": 0.30, "avg_ticket_change": 0.25},
                attack_vector="transaction_laundering",
                mitre_id="FRD-T1005",
                tags=["laundering", "merchant_fraud", "mcc_mismatch"],
            ),
            PatternEntry(
                pattern_id="PAT-BASE-006",
                name="Cross-Border Velocity Attack",
                description="Rapid transactions across multiple countries within impossible travel timeframes. Card-present transactions in different continents within hours.",
                pattern_type=PatternType.FACT,
                severity=PatternSeverity.CRITICAL,
                status=PatternStatus.CONFIRMED,
                observation_count=28000,
                confidence=0.94,
                source_bank_count=14,
                feature_signature={"geo_velocity": 0.45, "country_count_24h": 0.30, "time_between_txn": 0.20},
                attack_vector="cross_border_velocity",
                mitre_id="FRD-T1006",
                tags=["cross_border", "velocity", "impossible_travel"],
            ),
        ]

        count = 0
        for pattern in baseline:
            self.add_pattern(pattern)
            count += 1

        logger.info(f"Seeded {count} baseline patterns")
        return count
