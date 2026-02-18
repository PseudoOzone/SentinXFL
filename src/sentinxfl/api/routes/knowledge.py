"""
SentinXFL - Knowledge & Pattern Intelligence API Routes
=========================================================

REST API endpoints for the DP Pattern Library, Emergent Detector,
Central Knowledge Model, and Report Generator.

Author: Anshuman Bakshi
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinxfl.core.logging import get_logger
from sentinxfl.intelligence.pattern_library import (
    PatternEntry,
    PatternLibrary,
    PatternSeverity,
    PatternStatus,
    PatternType,
)
from sentinxfl.intelligence.emergent_detector import EmergentAlert, EmergentDetector
from sentinxfl.intelligence.central_model import BankProfile, CentralKnowledgeModel
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
_demo_seeded: bool = False


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


def _seed_demo_data():
    """Seed realistic demo data for dashboards: banks, patterns, features, alerts, rounds."""
    global _demo_seeded
    if _demo_seeded:
        return
    _demo_seeded = True

    lib = _get_library()
    cm = _get_central_model()
    det = _get_detector()
    now = datetime.utcnow()

    # ── Demo Banks ────────────────────────────────────────
    demo_banks = [
        ("demo-bank", "Demo Bank", 112300, 3410, 0.0304, 0.954, 13, 0.41),
        ("bank-nbi", "National Bank of India", 185400, 4250, 0.0229, 0.962, 14, 0.32),
        ("bank-hdfc", "HDFC Financial Services", 142800, 2890, 0.0202, 0.971, 12, 0.28),
        ("bank-icici", "ICICI Digital Banking", 98500, 3720, 0.0378, 0.945, 11, 0.51),
        ("bank-sbc", "State Bank Corp", 224600, 8940, 0.0398, 0.938, 15, 0.62),
        ("bank-axis", "Axis Digital Bank", 67200, 1280, 0.0190, 0.978, 9, 0.22),
    ]
    for bid, name, txns, fraud, fr, acc, rounds, risk in demo_banks:
        profile = BankProfile(
            bank_id=bid,
            display_name=name,
            joined_at=(now - timedelta(days=90)).isoformat(),
            last_active=(now - timedelta(hours=2)).isoformat(),
            total_transactions=txns,
            total_fraud_flagged=fraud,
            avg_fraud_rate=fr,
            model_accuracy=acc,
            rounds_participated=rounds,
            risk_score=risk,
        )
        cm._banks[bid] = profile

    # ── Extra Patterns (emergent, variant, historical) ────
    extra_patterns = [
        PatternEntry(
            pattern_id="PAT-EMG-001",
            name="Real-Time Payment Exploitation",
            description="Attackers exploiting instant payment rails (UPI/IMPS) to withdraw funds before fraud detection systems can react. Micro-delays in settlement used to layer transactions.",
            pattern_type=PatternType.EMERGENT,
            severity=PatternSeverity.CRITICAL,
            status=PatternStatus.UNDER_REVIEW,
            observation_count=320,
            confidence=0.72,
            source_bank_count=4,
            feature_signature={"payment_speed": 0.42, "settlement_gap": 0.35, "amount": 0.18},
            attack_vector="realtime_payment_abuse",
            tags=["upi", "imps", "instant_payment", "emerging"],
            frequency_per_day=12.5,
        ),
        PatternEntry(
            pattern_id="PAT-EMG-002",
            name="Deepfake KYC Bypass",
            description="Use of AI-generated deepfake documents and video to pass KYC verification checks. Detected across multiple onboarding pipelines.",
            pattern_type=PatternType.EMERGENT,
            severity=PatternSeverity.HIGH,
            status=PatternStatus.UNDER_REVIEW,
            observation_count=85,
            confidence=0.64,
            source_bank_count=3,
            feature_signature={"doc_authenticity_score": 0.50, "liveness_anomaly": 0.35, "onboarding_velocity": 0.15},
            attack_vector="deepfake_kyc",
            tags=["deepfake", "kyc", "identity", "ai_generated"],
            frequency_per_day=3.2,
        ),
        PatternEntry(
            pattern_id="PAT-EMG-003",
            name="QR Code Injection Attack",
            description="Malicious QR codes overlaid on legitimate merchant payment terminals redirecting payments to attacker-controlled accounts.",
            pattern_type=PatternType.ZERO_DAY,
            severity=PatternSeverity.HIGH,
            status=PatternStatus.ACTIVE,
            observation_count=42,
            confidence=0.58,
            source_bank_count=2,
            feature_signature={"merchant_mismatch": 0.45, "geo_anomaly": 0.30, "recipient_age": 0.25},
            attack_vector="qr_injection",
            tags=["qr_code", "merchant_fraud", "zero_day"],
            frequency_per_day=1.8,
        ),
        PatternEntry(
            pattern_id="PAT-VAR-001",
            name="ATO via SIM-Swap (Variant)",
            description="Variant of Account Takeover pattern using telecom-assisted SIM swapping to intercept OTPs. Higher success rate than credential stuffing.",
            pattern_type=PatternType.VARIANT,
            severity=PatternSeverity.MEDIUM,
            status=PatternStatus.ACTIVE,
            observation_count=1800,
            confidence=0.81,
            source_bank_count=7,
            feature_signature={"sim_change_recency": 0.40, "otp_failure_rate": 0.30, "device_change": 0.25},
            attack_vector="sim_swap_ato",
            tags=["sim_swap", "ato", "variant", "telecom"],
            related_patterns=["PAT-BASE-002"],
            frequency_per_day=6.4,
        ),
        PatternEntry(
            pattern_id="PAT-HIST-001",
            name="Check Kiting (Legacy)",
            description="Historical pattern of exploiting float time between check deposit and clearance. Largely mitigated by modern clearing systems but still observed in rural banking.",
            pattern_type=PatternType.HISTORICAL,
            severity=PatternSeverity.LOW,
            status=PatternStatus.DEPRECATED,
            observation_count=95000,
            confidence=0.97,
            source_bank_count=15,
            feature_signature={"check_float_time": 0.50, "multi_bank_deposits": 0.30, "amount_pattern": 0.20},
            attack_vector="check_kiting",
            tags=["check", "legacy", "float", "historical"],
            frequency_per_day=0.3,
        ),
        PatternEntry(
            pattern_id="PAT-VAR-002",
            name="Refund Abuse via Chargebacks",
            description="Coordinated friendly fraud exploiting chargeback mechanisms — buy expensive items, file false disputes claiming non-delivery.",
            pattern_type=PatternType.VARIANT,
            severity=PatternSeverity.MEDIUM,
            status=PatternStatus.CONFIRMED,
            observation_count=12500,
            confidence=0.87,
            source_bank_count=9,
            feature_signature={"chargeback_rate": 0.40, "dispute_timing": 0.30, "merchant_category": 0.20, "amount": 0.10},
            attack_vector="chargeback_abuse",
            tags=["chargeback", "friendly_fraud", "refund_abuse"],
            frequency_per_day=18.5,
        ),
    ]
    for p in extra_patterns:
        lib.add_pattern(p)

    # ── Global Feature Importances ────────────────────────
    demo_features = {
        "transaction_amount": [0.142, 0.138, 0.151, 0.145, 0.139, 0.147, 0.143],
        "velocity_1h": [0.128, 0.135, 0.131, 0.127, 0.133, 0.129, 0.130],
        "geo_distance": [0.115, 0.112, 0.118, 0.121, 0.110, 0.116, 0.114],
        "merchant_risk_score": [0.098, 0.102, 0.095, 0.100, 0.097, 0.101, 0.099],
        "device_fingerprint_change": [0.087, 0.091, 0.084, 0.089, 0.086, 0.090, 0.088],
        "time_since_last_txn": [0.076, 0.073, 0.079, 0.075, 0.078, 0.074, 0.077],
        "account_age_days": [0.065, 0.068, 0.062, 0.064, 0.067, 0.063, 0.066],
        "cross_border_flag": [0.058, 0.055, 0.061, 0.057, 0.059, 0.056, 0.060],
        "channel_risk": [0.045, 0.048, 0.042, 0.046, 0.044, 0.047, 0.043],
        "beneficiary_risk": [0.038, 0.041, 0.035, 0.039, 0.037, 0.040, 0.036],
    }
    cm._global_feature_importances = demo_features

    # ── Simulated Round History ───────────────────────────
    cm._round_history = []
    for r in range(1, 16):
        cm._round_history.append({
            "round": r,
            "version": r,
            "timestamp": (now - timedelta(days=30 - r * 2)).isoformat(),
            "banks_participated": min(3 + r // 3, 5),
            "global_accuracy": round(0.91 + r * 0.004, 4),
            "global_loss": round(0.35 - r * 0.018, 4),
        })
    cm._current_version = 15

    # ── Demo Alerts ───────────────────────────────────────
    demo_alerts = [
        EmergentAlert(
            alert_id="ALERT-DEMO-001",
            pattern_id="PAT-EMG-001",
            title="Real-Time Payment Abuse Surge",
            description="4 banks reporting coordinated exploitation of instant payment rails. 320 suspicious transactions detected in last 48 hours with micro-delay layering pattern.",
            severity=PatternSeverity.CRITICAL,
            alert_type="spike",
            confidence=0.88,
            affected_banks=4,
            evidence={"txn_count": 320, "avg_amount": 15400, "peak_hour": "02:00-04:00 IST"},
            recommended_actions=[
                "Enable real-time velocity checks on UPI/IMPS",
                "Apply 30-second settlement delay for flagged accounts",
                "Share attack signatures across participating banks",
            ],
        ),
        EmergentAlert(
            alert_id="ALERT-DEMO-002",
            pattern_id="PAT-EMG-002",
            title="Deepfake KYC Documents Detected",
            description="AI-generated identity documents detected at 3 banks during onboarding. Liveness detection bypassed using generated video. 85 suspicious accounts flagged.",
            severity=PatternSeverity.HIGH,
            alert_type="zero_day",
            confidence=0.76,
            affected_banks=3,
            evidence={"flagged_accounts": 85, "bypass_method": "generated_video", "doc_types": ["Aadhaar", "PAN"]},
            recommended_actions=[
                "Upgrade KYC liveness detection to v3 challenge-based",
                "Enable multi-factor document verification",
                "Review all accounts onboarded in last 30 days",
            ],
        ),
        EmergentAlert(
            alert_id="ALERT-DEMO-003",
            pattern_id="PAT-EMG-003",
            title="QR Code Manipulation Campaign",
            description="Tampered QR codes redirecting UPI payments at merchant locations in 2 cities. Estimated ₹4.2L diverted across 42 transactions.",
            severity=PatternSeverity.HIGH,
            alert_type="zero_day",
            confidence=0.69,
            affected_banks=2,
            evidence={"cities": ["Mumbai", "Pune"], "total_diverted": 420000, "merchant_count": 18},
            recommended_actions=[
                "Alert merchants to verify QR code authenticity",
                "Enable recipient account age check for UPI payments",
                "Coordinate with local law enforcement",
            ],
        ),
        EmergentAlert(
            alert_id="ALERT-DEMO-004",
            pattern_id="PAT-BASE-002",
            title="ATO Spike: Credential Stuffing Wave",
            description="Elevated account takeover attempts across 5 banks. Credential stuffing campaign using leaked database — 2,400 login attempts/hour from rotating proxy network.",
            severity=PatternSeverity.CRITICAL,
            alert_type="spike",
            confidence=0.92,
            affected_banks=5,
            evidence={"login_attempts_per_hr": 2400, "source": "rotating_proxies", "success_rate": 0.031},
            recommended_actions=[
                "Activate enhanced CAPTCHA and rate limiting",
                "Force password reset for accounts with failed attempts",
                "Enable IP reputation scoring",
            ],
        ),
        EmergentAlert(
            alert_id="ALERT-DEMO-005",
            pattern_id="PAT-BASE-004",
            title="Money Mule Network Expansion",
            description="Known mule network has expanded to 3 additional banks. Transfer chain analysis reveals 28 new intermediary accounts with classic amount-decay pattern.",
            severity=PatternSeverity.MEDIUM,
            alert_type="correlation",
            confidence=0.82,
            affected_banks=3,
            evidence={"new_mule_accounts": 28, "chain_depth": 4, "total_volume": 890000},
            recommended_actions=[
                "Flag identified intermediary accounts for enhanced monitoring",
                "Share anonymized mule account signatures",
                "Report to Financial Intelligence Unit",
            ],
        ),
    ]
    det._alerts = demo_alerts

    cm._save_state()
    log.info("Demo data seeded: %d banks, %d extra patterns, %d features, %d alerts",
             len(demo_banks), len(extra_patterns), len(demo_features), len(demo_alerts))


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
    _seed_demo_data()
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
    _seed_demo_data()
    return _get_central_model().get_global_statistics()


@router.get("/knowledge/global/trends", tags=["knowledge"])
async def get_global_trends(window: int = Query(default=10, le=100)):
    """Get trend analysis over recent rounds."""
    _seed_demo_data()
    return _get_central_model().get_trend_analysis(window=window)


@router.get("/knowledge/global/features", tags=["knowledge"])
async def get_global_features(top_n: int = Query(default=20, le=100)):
    """Get globally aggregated feature importances."""
    _seed_demo_data()
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
    _seed_demo_data()
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
