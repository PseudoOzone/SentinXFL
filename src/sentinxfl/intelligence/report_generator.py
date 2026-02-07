"""
SentinXFL - Report Generator
==============================

Generates comprehensive fraud intelligence reports:
- Global cross-bank reports (for SentinXFL employees)
- Bank-specific reports (for client banks)
- Emergent attack briefings
- SAR (Suspicious Activity Report) summaries
- Compliance and audit reports

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sentinxfl.core.logging import get_logger
from sentinxfl.intelligence.central_model import CentralKnowledgeModel
from sentinxfl.intelligence.emergent_detector import EmergentDetector
from sentinxfl.intelligence.pattern_library import PatternLibrary, PatternSeverity, PatternType

logger = get_logger(__name__)


@dataclass
class ReportSection:
    """A section of a generated report."""

    title: str
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    charts: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "info"  # info | warning | critical


@dataclass
class Report:
    """A complete generated report."""

    report_id: str
    report_type: str  # "global" | "bank" | "emergent" | "compliance"
    title: str
    generated_at: str
    generated_for: str  # "all" or bank_id
    summary: str
    sections: list[ReportSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "title": self.title,
            "generated_at": self.generated_at,
            "generated_for": self.generated_for,
            "summary": self.summary,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "data": s.data,
                    "charts": s.charts,
                    "severity": s.severity,
                }
                for s in self.sections
            ],
            "metadata": self.metadata,
        }


class ReportGenerator:
    """
    Generates reports from central knowledge model data.

    Report types:
    1. Global Report - Cross-bank intelligence summary
    2. Bank Report - Bank-specific analysis
    3. Emergent Attack Briefing - Active threats
    4. Compliance Report - Audit trail and regulatory info
    """

    def __init__(
        self,
        library: PatternLibrary,
        central_model: CentralKnowledgeModel,
        detector: EmergentDetector,
    ):
        self.library = library
        self.central_model = central_model
        self.detector = detector
        self._reports: list[Report] = []
        logger.info("ReportGenerator initialized")

    # ----------------------------------------------------------
    # Global Report (Employee Dashboard)
    # ----------------------------------------------------------

    def generate_global_report(self) -> Report:
        """Generate comprehensive global cross-bank intelligence report."""
        now = datetime.utcnow().isoformat()
        stats = self.central_model.get_global_statistics()
        trends = self.central_model.get_trend_analysis()
        top_features = self.central_model.get_global_feature_importance(top_n=15)
        emergent_patterns = self.library.get_emergent_patterns()
        fact_patterns = self.library.get_fact_based_patterns()
        risk_scores = self.central_model.calculate_bank_risk_scores()
        alert_summary = self.detector.get_alert_summary()

        report = Report(
            report_id=f"GLOBAL-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            report_type="global",
            title="Global Fraud Intelligence Report",
            generated_at=now,
            generated_for="all",
            summary=self._build_global_summary(stats, alert_summary),
        )

        # Section 1: Executive Summary
        report.sections.append(ReportSection(
            title="Executive Summary",
            content=(
                f"SentinXFL monitors {stats['total_banks']} participating banks with "
                f"{stats['total_transactions_processed']:,} transactions processed. "
                f"The system has identified {stats['pattern_library']['total']} "
                f"fraud patterns across all institutions."
            ),
            data=stats,
        ))

        # Section 2: Trend Analysis
        report.sections.append(ReportSection(
            title="Performance Trends",
            content=self._build_trend_content(trends),
            data=trends,
            charts=[
                {"type": "line", "label": "Accuracy Over Rounds", "metric": "accuracy"},
                {"type": "line", "label": "Loss Over Rounds", "metric": "loss"},
            ],
        ))

        # Section 3: Top Attack Vectors
        report.sections.append(ReportSection(
            title="Top Attack Vectors",
            content=(
                f"Analysis of {len(top_features)} features across all banks reveals "
                f"the following most significant fraud indicators."
            ),
            data={"features": top_features},
            charts=[{"type": "bar", "label": "Feature Importance", "metric": "importance"}],
        ))

        # Section 4: Emergent Threats
        sev = "critical" if len(emergent_patterns) > 5 else "warning" if emergent_patterns else "info"
        report.sections.append(ReportSection(
            title="Emergent Threat Intelligence",
            content=(
                f"{len(emergent_patterns)} emergent patterns detected that have not yet "
                f"been confirmed as established attack vectors. "
                f"{len(fact_patterns)} patterns are confirmed fact-based threats."
            ),
            data={
                "emergent_count": len(emergent_patterns),
                "fact_based_count": len(fact_patterns),
                "patterns": [p.to_dict() for p in emergent_patterns[:10]],
            },
            severity=sev,
        ))

        # Section 5: Active Alerts
        report.sections.append(ReportSection(
            title="Active Alerts",
            content=f"{alert_summary['total_alerts']} alerts currently active.",
            data=alert_summary,
            severity="critical" if alert_summary["total_alerts"] > 0 else "info",
        ))

        # Section 6: Bank Risk Assessment
        report.sections.append(ReportSection(
            title="Bank Risk Assessment",
            content=f"Risk scores calculated for {len(risk_scores)} participating banks.",
            data={"risk_scores": risk_scores},
            charts=[{"type": "heatmap", "label": "Bank Risk Heatmap"}],
        ))

        self._reports.append(report)
        logger.info("Generated global report: %s", report.report_id)
        return report

    # ----------------------------------------------------------
    # Bank-Specific Report (Client Dashboard)
    # ----------------------------------------------------------

    def generate_bank_report(self, bank_id: str) -> Report | None:
        """Generate a report for a specific client bank."""
        bank = self.central_model.get_bank(bank_id)
        if not bank:
            logger.warning("Bank %s not found", bank_id)
            return None

        now = datetime.utcnow().isoformat()
        fact_patterns = self.library.get_fact_based_patterns()
        emergent_patterns = self.library.get_emergent_patterns()
        stats = self.central_model.get_global_statistics()

        # Filter alerts relevant to this bank
        all_alerts = self.detector.get_alerts(limit=100)
        # Show cross-bank alerts (coordinated attacks) and global emergents
        relevant_alerts = [
            a for a in all_alerts
            if a.affected_banks > 1 or a.alert_type == "correlation"
        ]

        report = Report(
            report_id=f"BANK-{bank_id[:8]}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            report_type="bank",
            title=f"Fraud Intelligence Report - {bank.display_name}",
            generated_at=now,
            generated_for=bank_id,
            summary=(
                f"Report for {bank.display_name}: {bank.total_transactions:,} transactions "
                f"processed, {bank.total_fraud_flagged:,} fraud flagged, "
                f"model accuracy {bank.model_accuracy:.2%}."
            ),
        )

        # Section 1: Bank Overview
        report.sections.append(ReportSection(
            title="Your Bank Overview",
            content=(
                f"Since joining SentinXFL on {bank.joined_at[:10]}, your bank has "
                f"participated in {bank.rounds_participated} federated learning rounds. "
                f"Average fraud rate: {bank.avg_fraud_rate:.4%}."
            ),
            data=bank.to_dict(),
        ))

        # Section 2: Known Threats (fact-based patterns visible to clients)
        report.sections.append(ReportSection(
            title="Confirmed Threat Patterns",
            content=(
                f"The SentinXFL network has confirmed {len(fact_patterns)} fact-based "
                f"fraud patterns across all participating institutions. These are "
                f"validated threats you should be defending against."
            ),
            data={"patterns": [p.to_dict() for p in fact_patterns[:20]]},
            severity="warning" if fact_patterns else "info",
        ))

        # Section 3: Emerging Threats (visible to clients as awareness)
        report.sections.append(ReportSection(
            title="Emerging Threats (Industry Alert)",
            content=(
                f"{len(emergent_patterns)} emerging attack patterns detected across "
                f"the banking network. These are not yet confirmed but warrant "
                f"heightened monitoring."
            ),
            data={
                "emergent_count": len(emergent_patterns),
                "patterns": [
                    {
                        "title": p.name,
                        "description": p.description,
                        "severity": p.severity.value,
                        "confidence": p.confidence,
                        "first_seen": p.first_seen,
                    }
                    for p in emergent_patterns[:10]
                ],
            },
            severity="warning" if emergent_patterns else "info",
        ))

        # Section 4: Global Network Alerts
        if relevant_alerts:
            report.sections.append(ReportSection(
                title="Network-Wide Alerts",
                content=(
                    f"{len(relevant_alerts)} active alerts across the SentinXFL network "
                    f"that may affect your institution."
                ),
                data={"alerts": [a.to_dict() for a in relevant_alerts[:10]]},
                severity="critical",
            ))

        # Section 5: Recommendations
        recommendations = self._generate_bank_recommendations(bank, fact_patterns, emergent_patterns)
        report.sections.append(ReportSection(
            title="Recommended Actions",
            content=recommendations,
        ))

        self._reports.append(report)
        logger.info("Generated bank report for %s: %s", bank_id, report.report_id)
        return report

    # ----------------------------------------------------------
    # Emergent Attack Briefing
    # ----------------------------------------------------------

    def generate_emergent_briefing(self) -> Report:
        """Generate focused briefing on emergent/zero-day attacks."""
        now = datetime.utcnow().isoformat()
        emergent = self.library.get_emergent_patterns()
        zero_days = [p for p in self.library.list_patterns(limit=1000) if p.pattern_type == PatternType.ZERO_DAY]
        alerts = self.detector.get_alerts(limit=50)
        critical_alerts = [a for a in alerts if a.severity == PatternSeverity.CRITICAL]

        report = Report(
            report_id=f"EMERGENT-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            report_type="emergent",
            title="Emergent Attack Intelligence Briefing",
            generated_at=now,
            generated_for="all",
            summary=(
                f"SECURITY BRIEFING: {len(emergent)} emergent patterns, "
                f"{len(zero_days)} potential zero-day attacks, "
                f"{len(critical_alerts)} critical alerts active."
            ),
        )

        if zero_days:
            report.sections.append(ReportSection(
                title="Zero-Day Attack Indicators",
                content=(
                    f"{len(zero_days)} potential zero-day fraud patterns detected. "
                    f"These patterns show entirely novel characteristics not seen before."
                ),
                data={"patterns": [p.to_dict() for p in zero_days[:10]]},
                severity="critical",
            ))

        if critical_alerts:
            report.sections.append(ReportSection(
                title="Critical Alerts",
                content=f"{len(critical_alerts)} critical severity alerts require immediate attention.",
                data={"alerts": [a.to_dict() for a in critical_alerts]},
                severity="critical",
            ))

        self._reports.append(report)
        return report

    # ----------------------------------------------------------
    # Compliance Report
    # ----------------------------------------------------------

    def generate_compliance_report(self, bank_id: str | None = None) -> Report:
        """Generate compliance and audit report."""
        now = datetime.utcnow().isoformat()
        stats = self.central_model.get_global_statistics()
        lib_stats = self.library.get_statistics()

        target = bank_id or "all"
        report = Report(
            report_id=f"COMPLIANCE-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            report_type="compliance",
            title="Privacy & Compliance Audit Report",
            generated_at=now,
            generated_for=target,
            summary=(
                f"Compliance report covering {stats['total_rounds']} federated learning "
                f"rounds with differential privacy protections."
            ),
        )

        report.sections.append(ReportSection(
            title="Data Privacy Compliance",
            content=(
                "All federated learning rounds use Differential Privacy (DP) mechanisms. "
                "No raw data is shared between institutions. Only model gradients with "
                "calibrated noise are exchanged."
            ),
            data={
                "dp_mechanism": "Gaussian",
                "privacy_budget_epsilon": 1.0,
                "privacy_budget_delta": 1e-5,
                "total_rounds": stats["total_rounds"],
            },
        ))

        report.sections.append(ReportSection(
            title="Pattern Library Audit",
            content=(
                f"The pattern library contains {lib_stats['total']} patterns. "
                f"All patterns are derived from aggregated model statistics, not raw data."
            ),
            data=lib_stats,
        ))

        self._reports.append(report)
        return report

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _build_global_summary(self, stats: dict, alert_summary: dict) -> str:
        return (
            f"Global Fraud Intelligence: {stats['total_banks']} banks, "
            f"{stats['total_transactions_processed']:,} transactions, "
            f"{stats['pattern_library']['total']} patterns tracked, "
            f"{alert_summary['total_alerts']} active alerts."
        )

    def _build_trend_content(self, trends: dict) -> str:
        if "message" in trends:
            return trends["message"]
        acc = trends.get("accuracy_trend", {})
        return (
            f"Model accuracy is {acc.get('direction', 'stable')} "
            f"(current: {acc.get('current', 0):.4f}, "
            f"window mean: {acc.get('mean', 0):.4f})."
        )

    def _generate_bank_recommendations(self, bank, fact_patterns, emergent_patterns) -> str:
        recs = []
        if bank.avg_fraud_rate > 0.05:
            recs.append("• Your fraud rate is above industry average. Consider enhanced monitoring.")
        if bank.model_accuracy < 0.9:
            recs.append("• Model accuracy below 90%. Consider uploading more training data.")
        if emergent_patterns:
            recs.append(f"• {len(emergent_patterns)} emerging threats detected. Review your rule engine.")
        if not recs:
            recs.append("• System is operating within normal parameters. Continue regular monitoring.")
        return "\n".join(recs)

    def get_reports(
        self,
        report_type: str | None = None,
        bank_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List generated reports."""
        results = self._reports.copy()
        if report_type:
            results = [r for r in results if r.report_type == report_type]
        if bank_id:
            results = [r for r in results if r.generated_for == bank_id]
        results.sort(key=lambda r: r.generated_at, reverse=True)
        return [r.to_dict() for r in results[:limit]]
