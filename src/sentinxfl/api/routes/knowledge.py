"""
SentinXFL - Knowledge & Pattern Intelligence API Routes
=========================================================

REST API endpoints for the DP Pattern Library, Emergent Detector,
Central Knowledge Model, and Report Generator.

Author: Anshuman Bakshi
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinxfl.core.logging import get_logger
from sentinxfl.intelligence.pattern_library import (
    PatternLibrary,
    PatternSeverity,
    PatternType,
)
from sentinxfl.intelligence.emergent_detector import EmergentDetector
from sentinxfl.intelligence.central_model import CentralKnowledgeModel
from sentinxfl.intelligence.report_generator import ReportGenerator
from sentinxfl.intelligence.pattern_miner import PatternMiner

log = get_logger(__name__)
router = APIRouter()

# ============================================
# Singleton instances
# ============================================
_library: PatternLibrary | None = None
_central_model: CentralKnowledgeModel | None = None
_detector: EmergentDetector | None = None
_report_gen: ReportGenerator | None = None
_miner: PatternMiner | None = None


def _get_library() -> PatternLibrary:
    global _library
    if _library is None:
        _library = PatternLibrary()
        _library.seed_baseline_patterns()
    return _library


def _get_central_model() -> CentralKnowledgeModel:
    global _central_model
    if _central_model is None:
        _central_model = CentralKnowledgeModel(_get_library())
    return _central_model


def _get_detector() -> EmergentDetector:
    global _detector
    if _detector is None:
        _detector = EmergentDetector(_get_library())
    return _detector


def _get_report_gen() -> ReportGenerator:
    global _report_gen
    if _report_gen is None:
        _report_gen = ReportGenerator(_get_library(), _get_central_model(), _get_detector())
    return _report_gen


def _get_miner() -> PatternMiner:
    global _miner
    if _miner is None:
        _miner = PatternMiner(_get_library())
    return _miner


# ============================================
# Request/Response Models
# ============================================
class RegisterBankRequest(BaseModel):
    bank_id: str
    display_name: str


class IngestRoundRequest(BaseModel):
    round_number: int
    bank_metrics: dict[str, dict[str, float]]
    feature_importances: dict[str, dict[str, float]]
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    previous_importances: Optional[dict[str, dict[str, float]]] = None


# ============================================
# Pattern Library Endpoints
# ============================================


@router.get("/knowledge/patterns", tags=["knowledge"])
async def list_patterns(
    pattern_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(default=50, le=500),
):
    """List all patterns with optional filters."""
    lib = _get_library()
    pt = PatternType(pattern_type) if pattern_type else None
    patterns = lib.list_patterns(pattern_type=pt, limit=limit)
    if severity:
        sev = PatternSeverity(severity)
        patterns = [p for p in patterns if p.severity == sev]
    return {"patterns": [p.to_dict() for p in patterns], "total": len(patterns)}


@router.get("/knowledge/patterns/emergent", tags=["knowledge"])
async def get_emergent_patterns(limit: int = Query(default=20, le=100)):
    """Get emerging threat patterns."""
    lib = _get_library()
    patterns = lib.get_emergent_patterns()[:limit]
    return {"patterns": [p.to_dict() for p in patterns], "count": len(patterns)}


@router.get("/knowledge/patterns/fact-based", tags=["knowledge"])
async def get_fact_based_patterns(limit: int = Query(default=50, le=200)):
    """Get confirmed fact-based patterns."""
    lib = _get_library()
    patterns = lib.get_fact_based_patterns()[:limit]
    return {"patterns": [p.to_dict() for p in patterns], "count": len(patterns)}


@router.get("/knowledge/patterns/search", tags=["knowledge"])
async def search_patterns(q: str = Query(..., min_length=2)):
    """Full-text search across patterns."""
    lib = _get_library()
    patterns = lib.search_patterns(q)
    return {"patterns": [p.to_dict() for p in patterns], "query": q, "count": len(patterns)}


@router.get("/knowledge/patterns/{pattern_id}", tags=["knowledge"])
async def get_pattern(pattern_id: str):
    """Get a specific pattern by ID."""
    lib = _get_library()
    pattern = lib.get_pattern(pattern_id)
    if not pattern:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")
    return pattern.to_dict()


@router.get("/knowledge/statistics", tags=["knowledge"])
async def get_library_statistics():
    """Get pattern library statistics."""
    return _get_library().get_statistics()


# ============================================
# Emergent Alerts Endpoints
# ============================================


@router.get("/knowledge/alerts", tags=["knowledge"])
async def get_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = Query(default=50, le=200),
):
    """Get emergent attack alerts."""
    det = _get_detector()
    sev = PatternSeverity(severity) if severity else None
    alerts = det.get_alerts(severity=sev, alert_type=alert_type, limit=limit)
    return {"alerts": [a.to_dict() for a in alerts], "count": len(alerts)}


@router.get("/knowledge/alerts/summary", tags=["knowledge"])
async def get_alert_summary():
    """Get summary of all active alerts."""
    return _get_detector().get_alert_summary()


# ============================================
# Central Knowledge Model Endpoints
# ============================================


@router.get("/knowledge/global/statistics", tags=["knowledge"])
async def get_global_statistics():
    """Get global system statistics."""
    return _get_central_model().get_global_statistics()


@router.get("/knowledge/global/trends", tags=["knowledge"])
async def get_global_trends(window: int = Query(default=10, le=100)):
    """Get trend analysis over recent rounds."""
    return _get_central_model().get_trend_analysis(window=window)


@router.get("/knowledge/global/features", tags=["knowledge"])
async def get_global_features(top_n: int = Query(default=20, le=100)):
    """Get globally aggregated feature importances."""
    return _get_central_model().get_global_feature_importance(top_n=top_n)


@router.get("/knowledge/global/snapshots", tags=["knowledge"])
async def get_snapshots(limit: int = Query(default=20, le=100)):
    """Get knowledge snapshots."""
    return _get_central_model().get_snapshots(limit=limit)


@router.post("/knowledge/global/snapshot", tags=["knowledge"])
async def create_snapshot():
    """Create a versioned knowledge snapshot."""
    snapshot = _get_central_model().create_snapshot()
    return snapshot.to_dict()


# ============================================
# Bank Management Endpoints
# ============================================


@router.get("/knowledge/banks", tags=["knowledge"])
async def list_banks():
    """List all registered banks."""
    banks = _get_central_model().list_banks()
    return {"banks": [b.to_dict() for b in banks], "count": len(banks)}


@router.post("/knowledge/banks", tags=["knowledge"])
async def register_bank(req: RegisterBankRequest):
    """Register a new participating bank."""
    bank = _get_central_model().register_bank(req.bank_id, req.display_name)
    return bank.to_dict()


@router.get("/knowledge/banks/{bank_id}", tags=["knowledge"])
async def get_bank(bank_id: str):
    """Get bank profile."""
    bank = _get_central_model().get_bank(bank_id)
    if not bank:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    return bank.to_dict()


@router.get("/knowledge/banks/risk-scores", tags=["knowledge"])
async def get_risk_scores():
    """Get risk scores for all banks."""
    return _get_central_model().calculate_bank_risk_scores()


# ============================================
# Ingestion Endpoints
# ============================================


@router.post("/knowledge/ingest", tags=["knowledge"])
async def ingest_round(req: IngestRoundRequest):
    """
    Ingest FL round results into the knowledge system.
    Runs pattern mining + emergent detection + central model update.
    """
    cm = _get_central_model()
    det = _get_detector()
    miner = _get_miner()

    # 1. Central model ingestion
    round_entry = cm.ingest_round(
        round_number=req.round_number,
        bank_metrics=req.bank_metrics,
        feature_importances=req.feature_importances,
        global_accuracy=req.global_accuracy,
        global_loss=req.global_loss,
    )

    # 2. Pattern mining
    mining_result = miner.mine_from_round(
        round_number=req.round_number,
        feature_importances=req.feature_importances,
        bank_metrics=req.bank_metrics,
        previous_importances=req.previous_importances,
    )

    # 3. Emergent detection
    alerts = det.analyze_round(
        round_number=req.round_number,
        feature_importances=req.feature_importances,
        bank_metrics=req.bank_metrics,
        previous_importances=req.previous_importances,
    )

    return {
        "round_entry": round_entry,
        "mining": mining_result.to_dict(),
        "alerts": [a.to_dict() for a in alerts],
    }


# ============================================
# Report Endpoints
# ============================================


@router.post("/knowledge/reports/global", tags=["knowledge"])
async def generate_global_report():
    """Generate comprehensive global intelligence report."""
    report = _get_report_gen().generate_global_report()
    return report.to_dict()


@router.post("/knowledge/reports/bank/{bank_id}", tags=["knowledge"])
async def generate_bank_report(bank_id: str):
    """Generate report for a specific bank."""
    report = _get_report_gen().generate_bank_report(bank_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Bank {bank_id} not found")
    return report.to_dict()


@router.post("/knowledge/reports/emergent", tags=["knowledge"])
async def generate_emergent_briefing():
    """Generate emergent attack briefing."""
    report = _get_report_gen().generate_emergent_briefing()
    return report.to_dict()


@router.post("/knowledge/reports/compliance", tags=["knowledge"])
async def generate_compliance_report(bank_id: Optional[str] = None):
    """Generate compliance/audit report."""
    report = _get_report_gen().generate_compliance_report(bank_id)
    return report.to_dict()


@router.get("/knowledge/reports", tags=["knowledge"])
async def list_reports(
    report_type: Optional[str] = None,
    bank_id: Optional[str] = None,
    limit: int = Query(default=20, le=100),
):
    """List generated reports."""
    return _get_report_gen().get_reports(report_type=report_type, bank_id=bank_id, limit=limit)


# ============================================
# Mining Endpoints
# ============================================


@router.get("/knowledge/mining/history", tags=["knowledge"])
async def get_mining_history(limit: int = Query(default=20, le=100)):
    """Get pattern mining run history."""
    return _get_miner().get_mining_history(limit=limit)
