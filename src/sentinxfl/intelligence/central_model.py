"""
SentinXFL - Central Knowledge Model
=====================================

Versioned knowledge graph that aggregates patterns from all banks,
maintains a global fraud intelligence model, and feeds the RAG
pipeline. Supports version-controlled snapshots with rollback.

Author: Anshuman Bakshi
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sentinxfl.core.logging import get_logger
from sentinxfl.intelligence.pattern_library import (
    PatternLibrary,
    PatternSeverity,
    PatternType,
)

logger = get_logger(__name__)


@dataclass
class KnowledgeSnapshot:
    """A versioned snapshot of central knowledge state."""

    version: int
    created_at: str
    total_patterns: int
    total_alerts: int
    bank_count: int
    global_fraud_rate: float
    top_attack_vectors: list[dict[str, Any]]
    model_performance: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "total_patterns": self.total_patterns,
            "total_alerts": self.total_alerts,
            "bank_count": self.bank_count,
            "global_fraud_rate": self.global_fraud_rate,
            "top_attack_vectors": self.top_attack_vectors,
            "model_performance": self.model_performance,
            "metadata": self.metadata,
        }


@dataclass
class BankProfile:
    """Profile of a participating bank's data and behavior."""

    bank_id: str
    display_name: str
    joined_at: str
    last_active: str
    total_transactions: int = 0
    total_fraud_flagged: int = 0
    avg_fraud_rate: float = 0.0
    model_accuracy: float = 0.0
    rounds_participated: int = 0
    risk_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "display_name": self.display_name,
            "joined_at": self.joined_at,
            "last_active": self.last_active,
            "total_transactions": self.total_transactions,
            "total_fraud_flagged": self.total_fraud_flagged,
            "avg_fraud_rate": self.avg_fraud_rate,
            "model_accuracy": self.model_accuracy,
            "rounds_participated": self.rounds_participated,
            "risk_score": self.risk_score,
        }


class CentralKnowledgeModel:
    """
    Central Knowledge Model aggregating global fraud intelligence.

    Maintains:
    - Global pattern registry from all banks
    - Versioned snapshots of knowledge state
    - Bank profiles with risk scoring
    - Cross-bank feature importance consensus
    - Global statistics and trend analysis
    """

    def __init__(self, library: PatternLibrary, data_dir: str = "data/knowledge"):
        self.library = library
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._banks: dict[str, BankProfile] = {}
        self._snapshots: list[KnowledgeSnapshot] = []
        self._global_feature_importances: dict[str, list[float]] = {}
        self._round_history: list[dict[str, Any]] = []
        self._current_version = 0

        self._load_state()
        logger.info("CentralKnowledgeModel initialized (v%d)", self._current_version)

    def _state_path(self) -> Path:
        return self.data_dir / "central_model_state.json"

    def _load_state(self):
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._current_version = data.get("version", 0)
                for bid, bdata in data.get("banks", {}).items():
                    self._banks[bid] = BankProfile(**bdata)
                self._global_feature_importances = data.get("feature_importances", {})
                self._round_history = data.get("round_history", [])
                logger.info("Loaded central model state v%d", self._current_version)
            except Exception as e:
                logger.warning("Could not load central model state: %s", e)

    def _save_state(self):
        data = {
            "version": self._current_version,
            "updated_at": datetime.utcnow().isoformat(),
            "banks": {bid: b.to_dict() for bid, b in self._banks.items()},
            "feature_importances": self._global_feature_importances,
            "round_history": self._round_history[-500:],
        }
        self._state_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ----------------------------------------------------------
    # Bank Management
    # ----------------------------------------------------------

    def register_bank(self, bank_id: str, display_name: str) -> BankProfile:
        """Register a new participating bank."""
        now = datetime.utcnow().isoformat()
        if bank_id in self._banks:
            logger.info("Bank %s already registered", bank_id)
            return self._banks[bank_id]

        profile = BankProfile(
            bank_id=bank_id,
            display_name=display_name,
            joined_at=now,
            last_active=now,
        )
        self._banks[bank_id] = profile
        self._save_state()
        logger.info("Registered bank: %s (%s)", display_name, bank_id)
        return profile

    def get_bank(self, bank_id: str) -> BankProfile | None:
        return self._banks.get(bank_id)

    def list_banks(self) -> list[BankProfile]:
        return list(self._banks.values())

    def update_bank_metrics(
        self,
        bank_id: str,
        transactions: int = 0,
        fraud_flagged: int = 0,
        fraud_rate: float | None = None,
        accuracy: float | None = None,
    ):
        """Update bank metrics after a training round or data upload."""
        bank = self._banks.get(bank_id)
        if not bank:
            return
        bank.last_active = datetime.utcnow().isoformat()
        bank.total_transactions += transactions
        bank.total_fraud_flagged += fraud_flagged
        bank.rounds_participated += 1
        if fraud_rate is not None:
            bank.avg_fraud_rate = (bank.avg_fraud_rate * 0.7) + (fraud_rate * 0.3)
        if accuracy is not None:
            bank.model_accuracy = accuracy
        self._save_state()

    # ----------------------------------------------------------
    # Round Ingestion
    # ----------------------------------------------------------

    def ingest_round(
        self,
        round_number: int,
        bank_metrics: dict[str, dict[str, float]],
        feature_importances: dict[str, dict[str, float]],
        global_accuracy: float,
        global_loss: float,
    ) -> dict[str, Any]:
        """
        Ingest results from a federated learning round.

        - Updates bank profiles
        - Aggregates feature importances
        - Creates pattern library entries
        - Updates version and round history
        """
        self._current_version += 1

        # Update per-bank profiles
        for bank_id, metrics in bank_metrics.items():
            if bank_id not in self._banks:
                self.register_bank(bank_id, f"Bank-{bank_id[:8]}")
            self.update_bank_metrics(
                bank_id,
                transactions=int(metrics.get("num_samples", 0)),
                fraud_flagged=int(metrics.get("num_fraud", 0)),
                fraud_rate=metrics.get("fraud_rate"),
                accuracy=metrics.get("accuracy"),
            )

        # Aggregate global feature importances (exponential moving average)
        for bank_id, imps in feature_importances.items():
            for feat, val in imps.items():
                if feat not in self._global_feature_importances:
                    self._global_feature_importances[feat] = []
                self._global_feature_importances[feat].append(val)

        # Ingest into pattern library
        self.library.ingest_from_fl_round(
            round_number=round_number,
            feature_importances=feature_importances,
            bank_metrics=bank_metrics,
        )

        # Record round
        round_entry = {
            "round": round_number,
            "version": self._current_version,
            "timestamp": datetime.utcnow().isoformat(),
            "banks_participated": len(bank_metrics),
            "global_accuracy": global_accuracy,
            "global_loss": global_loss,
        }
        self._round_history.append(round_entry)

        self._save_state()
        logger.info(
            "Ingested round %d → knowledge v%d (%d banks)",
            round_number,
            self._current_version,
            len(bank_metrics),
        )
        return round_entry

    # ----------------------------------------------------------
    # Global Intelligence Queries
    # ----------------------------------------------------------

    def get_global_feature_importance(self, top_n: int = 20) -> list[dict[str, Any]]:
        """Get globally aggregated feature importances ranked."""
        result = []
        for feat, vals in self._global_feature_importances.items():
            result.append({
                "feature": feat,
                "mean_importance": round(float(np.mean(vals[-50:])), 6),
                "std": round(float(np.std(vals[-50:])), 6),
                "observations": len(vals),
            })
        result.sort(key=lambda x: x["mean_importance"], reverse=True)
        return result[:top_n]

    def get_global_statistics(self) -> dict[str, Any]:
        """Get overall system statistics."""
        lib_stats = self.library.get_statistics()
        banks = self.list_banks()
        return {
            "version": self._current_version,
            "total_banks": len(banks),
            "active_banks": sum(1 for b in banks if b.rounds_participated > 0),
            "total_transactions_processed": sum(b.total_transactions for b in banks),
            "total_fraud_flagged": sum(b.total_fraud_flagged for b in banks),
            "global_avg_fraud_rate": (
                round(float(np.mean([b.avg_fraud_rate for b in banks if b.avg_fraud_rate > 0])), 6)
                if banks
                else 0
            ),
            "total_rounds": len(self._round_history),
            "pattern_library": lib_stats,
            "last_updated": (
                self._round_history[-1]["timestamp"] if self._round_history else None
            ),
        }

    def get_trend_analysis(self, window: int = 10) -> dict[str, Any]:
        """Analyze trends over recent rounds."""
        recent = self._round_history[-window:] if self._round_history else []
        if len(recent) < 2:
            return {"message": "Insufficient data for trend analysis", "rounds_available": len(recent)}

        accuracies = [r.get("global_accuracy", 0) for r in recent]
        losses = [r.get("global_loss", 0) for r in recent]
        bank_counts = [r.get("banks_participated", 0) for r in recent]

        return {
            "window": window,
            "rounds_analyzed": len(recent),
            "accuracy_trend": {
                "current": round(accuracies[-1], 4),
                "mean": round(float(np.mean(accuracies)), 4),
                "direction": "improving" if accuracies[-1] > accuracies[0] else "declining",
            },
            "loss_trend": {
                "current": round(losses[-1], 4),
                "mean": round(float(np.mean(losses)), 4),
                "direction": "improving" if losses[-1] < losses[0] else "worsening",
            },
            "participation_trend": {
                "current": bank_counts[-1],
                "mean": round(float(np.mean(bank_counts)), 1),
            },
        }

    def create_snapshot(self) -> KnowledgeSnapshot:
        """Create a versioned snapshot of current knowledge state."""
        stats = self.get_global_statistics()
        top_features = self.get_global_feature_importance(top_n=10)
        top_attacks = [
            {"feature": f["feature"], "importance": f["mean_importance"]}
            for f in top_features[:5]
        ]

        recent_accuracy = 0.0
        if self._round_history:
            recent_accuracy = self._round_history[-1].get("global_accuracy", 0)

        snapshot = KnowledgeSnapshot(
            version=self._current_version,
            created_at=datetime.utcnow().isoformat(),
            total_patterns=stats["pattern_library"]["total"],
            total_alerts=0,
            bank_count=stats["total_banks"],
            global_fraud_rate=stats["global_avg_fraud_rate"],
            top_attack_vectors=top_attacks,
            model_performance={"accuracy": recent_accuracy},
        )
        self._snapshots.append(snapshot)
        logger.info("Created knowledge snapshot v%d", self._current_version)
        return snapshot

    def get_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._snapshots[-limit:]]

    # ----------------------------------------------------------
    # Risk Scoring
    # ----------------------------------------------------------

    def calculate_bank_risk_scores(self) -> dict[str, float]:
        """Calculate risk scores for all banks (0-1 scale)."""
        scores = {}
        if not self._banks:
            return scores

        all_fraud_rates = [b.avg_fraud_rate for b in self._banks.values() if b.avg_fraud_rate > 0]
        if not all_fraud_rates:
            return {bid: 0.5 for bid in self._banks}

        max_fr = max(all_fraud_rates) if all_fraud_rates else 1
        for bid, bank in self._banks.items():
            # Weighted risk: 60% fraud rate, 40% recent activity
            fr_component = bank.avg_fraud_rate / (max_fr + 1e-9)
            activity_component = 1.0 - min(bank.model_accuracy, 1.0) if bank.model_accuracy > 0 else 0.5
            risk = 0.6 * fr_component + 0.4 * activity_component
            bank.risk_score = round(min(max(risk, 0), 1), 4)
            scores[bid] = bank.risk_score

        self._save_state()
        return scores
