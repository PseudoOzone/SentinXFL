"""
SentinXFL PII Audit Logger
===========================

Immutable audit logging for PII detection and transformation operations.
Required for GDPR, DPDPA, and other compliance frameworks.

This is a PATENT-CORE component.

Author: Anshuman Bakshi
Copyright (c) 2026. All rights reserved.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from enum import Enum

from sentinxfl.core.config import settings
from sentinxfl.core.logging import get_logger
from sentinxfl.privacy.detector import PIIDetectionResult
from sentinxfl.privacy.transformer import TransformationResult
from sentinxfl.privacy.certifier import CertificationResult

log = get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    PII_SCAN_STARTED = "pii_scan_started"
    PII_SCAN_COMPLETED = "pii_scan_completed"
    PII_DETECTED = "pii_detected"
    TRANSFORMATION_STARTED = "transformation_started"
    TRANSFORMATION_COMPLETED = "transformation_completed"
    CERTIFICATION_REQUESTED = "certification_requested"
    CERTIFICATION_GRANTED = "certification_granted"
    CERTIFICATION_DENIED = "certification_denied"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Single audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    actor: str  # User/system that triggered event
    resource: str  # Resource being acted upon (dataset, column, etc.)
    action: str  # Human-readable action description
    details: dict = field(default_factory=dict)
    previous_hash: Optional[str] = None  # For chain integrity
    event_hash: Optional[str] = None  # Hash of this event
    
    def __post_init__(self):
        if not self.event_hash:
            self.event_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate hash of event for integrity verification."""
        data = f"{self.event_id}|{self.event_type}|{self.timestamp}|{self.actor}|{self.resource}|{self.action}|{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class AuditTrail:
    """Complete audit trail for a session."""
    session_id: str
    started_at: str
    events: list[AuditEvent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def event_count(self) -> int:
        return len(self.events)
    
    @property
    def last_event(self) -> Optional[AuditEvent]:
        return self.events[-1] if self.events else None


class PIIAuditLog:
    """
    Immutable audit logger for PII operations.
    
    Features:
    - Chain of custody tracking (hash-linked events)
    - GDPR Article 30 compliant record keeping
    - JSON export for compliance reports
    - Tamper detection via hash verification
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        actor: str = "system",
        persist_path: Optional[Path] = None,
    ):
        """Initialize audit logger.
        
        Args:
            session_id: Unique session identifier (auto-generated if not provided)
            actor: Default actor for events (user/system identifier)
            persist_path: Path to persist audit logs
        """
        self.session_id = session_id or self._generate_session_id()
        self.actor = actor
        self.persist_path = persist_path or settings.get_absolute_path(
            Path("logs") / "audit"
        )
        
        self._trail = AuditTrail(
            session_id=self.session_id,
            started_at=datetime.utcnow().isoformat(),
            metadata={
                "app_version": settings.app_version,
                "environment": settings.environment,
            },
        )
        
        self._event_counter = 0
        
        # Ensure persist directory exists
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        log.info(f"PIIAuditLog initialized: session={self.session_id}")
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        import secrets
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_hex(4)
        return f"AUDIT-{timestamp}-{random_part}"
    
    def _get_previous_hash(self) -> Optional[str]:
        """Get hash of previous event for chain integrity."""
        if self._trail.events:
            return self._trail.events[-1].event_hash
        return None
    
    def _create_event(
        self,
        event_type: AuditEventType,
        resource: str,
        action: str,
        details: dict = None,
    ) -> AuditEvent:
        """Create a new audit event."""
        self._event_counter += 1
        
        event = AuditEvent(
            event_id=f"{self.session_id}-{self._event_counter:05d}",
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            actor=self.actor,
            resource=resource,
            action=action,
            details=details or {},
            previous_hash=self._get_previous_hash(),
        )
        
        self._trail.events.append(event)
        return event
    
    def log_scan_started(
        self,
        dataset_name: str,
        num_columns: int,
        num_rows: int,
    ) -> AuditEvent:
        """Log PII scan initiation."""
        return self._create_event(
            AuditEventType.PII_SCAN_STARTED,
            resource=dataset_name,
            action=f"Initiated PII scan on {num_columns} columns, {num_rows} rows",
            details={
                "num_columns": num_columns,
                "num_rows": num_rows,
            },
        )
    
    def log_scan_completed(
        self,
        dataset_name: str,
        result: PIIDetectionResult,
    ) -> AuditEvent:
        """Log PII scan completion."""
        return self._create_event(
            AuditEventType.PII_SCAN_COMPLETED,
            resource=dataset_name,
            action=f"PII scan completed: {result.columns_with_pii}/{result.total_columns} columns flagged",
            details={
                "total_columns": result.total_columns,
                "columns_with_pii": result.columns_with_pii,
                "passed": result.passed,
                "matches": [
                    {
                        "column": m.column_name,
                        "pii_type": m.pii_type.value,
                        "sensitivity": m.sensitivity.value,
                        "confidence": m.confidence,
                    }
                    for m in result.matches
                ],
            },
        )
    
    def log_pii_detected(
        self,
        column_name: str,
        pii_type: str,
        sensitivity: str,
        confidence: float,
    ) -> AuditEvent:
        """Log individual PII detection."""
        return self._create_event(
            AuditEventType.PII_DETECTED,
            resource=column_name,
            action=f"PII detected: {pii_type} (sensitivity={sensitivity}, confidence={confidence:.2f})",
            details={
                "pii_type": pii_type,
                "sensitivity": sensitivity,
                "confidence": confidence,
            },
        )
    
    def log_transformation_started(
        self,
        dataset_name: str,
        num_columns: int,
    ) -> AuditEvent:
        """Log transformation initiation."""
        return self._create_event(
            AuditEventType.TRANSFORMATION_STARTED,
            resource=dataset_name,
            action=f"Initiated PII transformation on {num_columns} columns",
            details={"num_columns": num_columns},
        )
    
    def log_transformation_completed(
        self,
        dataset_name: str,
        results: list[TransformationResult],
    ) -> AuditEvent:
        """Log transformation completion."""
        successful = sum(1 for r in results if r.success)
        return self._create_event(
            AuditEventType.TRANSFORMATION_COMPLETED,
            resource=dataset_name,
            action=f"Transformation completed: {successful}/{len(results)} columns transformed",
            details={
                "total_columns": len(results),
                "successful": successful,
                "transformations": [
                    {
                        "column": r.column_name,
                        "type": r.transformation_type.value,
                        "rows": r.rows_transformed,
                        "success": r.success,
                    }
                    for r in results
                ],
            },
        )
    
    def log_certification(
        self,
        dataset_name: str,
        result: CertificationResult,
    ) -> AuditEvent:
        """Log certification result."""
        event_type = (
            AuditEventType.CERTIFICATION_GRANTED
            if result.certified
            else AuditEventType.CERTIFICATION_DENIED
        )
        
        return self._create_event(
            event_type,
            resource=dataset_name,
            action=f"Certification {'granted' if result.certified else 'denied'}: {result.certification_level}",
            details={
                "certified": result.certified,
                "level": result.certification_level,
                "score": result.certification_score,
                "gates_passed": [
                    result.gate1_passed,
                    result.gate2_passed,
                    result.gate3_passed,
                    result.gate4_passed,
                    result.gate5_passed,
                ],
                "warnings": result.warnings,
            },
        )
    
    def log_data_access(
        self,
        resource: str,
        purpose: str,
        accessor: Optional[str] = None,
    ) -> AuditEvent:
        """Log data access event (GDPR requirement)."""
        return self._create_event(
            AuditEventType.DATA_ACCESS,
            resource=resource,
            action=f"Data accessed for: {purpose}",
            details={
                "purpose": purpose,
                "accessor": accessor or self.actor,
            },
        )
    
    def log_data_export(
        self,
        resource: str,
        destination: str,
        format: str,
        num_records: int,
    ) -> AuditEvent:
        """Log data export event."""
        return self._create_event(
            AuditEventType.DATA_EXPORT,
            resource=resource,
            action=f"Exported {num_records} records to {destination}",
            details={
                "destination": destination,
                "format": format,
                "num_records": num_records,
            },
        )
    
    def log_error(
        self,
        resource: str,
        error_message: str,
        error_type: str = "unknown",
    ) -> AuditEvent:
        """Log error event."""
        return self._create_event(
            AuditEventType.ERROR,
            resource=resource,
            action=f"Error: {error_message}",
            details={
                "error_type": error_type,
                "error_message": error_message,
            },
        )
    
    def verify_chain_integrity(self) -> tuple[bool, list[str]]:
        """Verify integrity of the audit chain.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        for i, event in enumerate(self._trail.events):
            # Verify hash calculation
            expected_hash = event._calculate_hash()
            if event.event_hash != expected_hash:
                issues.append(f"Event {event.event_id}: Hash mismatch")
            
            # Verify chain linkage
            if i > 0:
                expected_prev = self._trail.events[i - 1].event_hash
                if event.previous_hash != expected_prev:
                    issues.append(f"Event {event.event_id}: Chain broken (previous_hash mismatch)")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            log.info(f"Audit chain integrity verified: {len(self._trail.events)} events")
        else:
            log.error(f"Audit chain integrity FAILED: {len(issues)} issues found")
        
        return is_valid, issues
    
    def export_json(self, filepath: Optional[Path] = None) -> Path:
        """Export audit trail to JSON file.
        
        Args:
            filepath: Optional output path. Defaults to persist_path.
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = self.persist_path / f"audit_{self.session_id}_{timestamp}.json"
        
        # Convert to dict
        trail_dict = {
            "session_id": self._trail.session_id,
            "started_at": self._trail.started_at,
            "metadata": self._trail.metadata,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "timestamp": e.timestamp,
                    "actor": e.actor,
                    "resource": e.resource,
                    "action": e.action,
                    "details": e.details,
                    "previous_hash": e.previous_hash,
                    "event_hash": e.event_hash,
                }
                for e in self._trail.events
            ],
            "event_count": self._trail.event_count,
            "exported_at": datetime.utcnow().isoformat(),
        }
        
        with open(filepath, "w") as f:
            json.dump(trail_dict, f, indent=2)
        
        log.info(f"Audit trail exported to {filepath}")
        return filepath
    
    def generate_compliance_report(self) -> str:
        """Generate GDPR/DPDPA compliance report.
        
        Returns:
            Formatted compliance report string
        """
        is_valid, issues = self.verify_chain_integrity()
        
        lines = [
            "=" * 70,
            "DATA PROCESSING COMPLIANCE REPORT",
            "=" * 70,
            f"Session ID: {self.session_id}",
            f"Report Generated: {datetime.utcnow().isoformat()}",
            f"Audit Chain Integrity: {'✓ VERIFIED' if is_valid else '✗ COMPROMISED'}",
            "-" * 70,
            "",
            "PROCESSING ACTIVITIES:",
        ]
        
        # Summarize by event type
        event_counts = {}
        for event in self._trail.events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
        
        for event_type, count in sorted(event_counts.items()):
            lines.append(f"  • {event_type}: {count} events")
        
        lines.append("")
        lines.append("-" * 70)
        lines.append("DETAILED EVENT LOG:")
        lines.append("")
        
        for event in self._trail.events:
            lines.append(f"  [{event.timestamp[:19]}] {event.event_type.value}")
            lines.append(f"      Resource: {event.resource}")
            lines.append(f"      Action: {event.action}")
            lines.append(f"      Actor: {event.actor}")
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("END OF COMPLIANCE REPORT")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_trail(self) -> AuditTrail:
        """Get the current audit trail."""
        return self._trail
    
    def get_events_by_type(self, event_type: AuditEventType) -> list[AuditEvent]:
        """Get all events of a specific type."""
        return [e for e in self._trail.events if e.event_type == event_type]
    
    def get_events_by_resource(self, resource: str) -> list[AuditEvent]:
        """Get all events for a specific resource."""
        return [e for e in self._trail.events if e.resource == resource]
