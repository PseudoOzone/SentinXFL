"""
SentinXFL Privacy API Routes
=============================

REST API endpoints for PII detection, transformation, and certification.

Author: Anshuman Bakshi
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sentinxfl.core.logging import get_logger
from sentinxfl.data.loader import DataLoader
from sentinxfl.data.schemas import DatasetType
from sentinxfl.privacy.detector import PIIDetector, PIIDetectionResult
from sentinxfl.privacy.transformer import PIITransformer, TransformationType
from sentinxfl.privacy.certifier import PIICertifier
from sentinxfl.privacy.audit import PIIAuditLog

log = get_logger(__name__)
router = APIRouter()


# ============================================
# Request/Response Models
# ============================================
class PIIScanRequest(BaseModel):
    """Request to scan a dataset for PII."""
    dataset_type: DatasetType
    sample_size: int = Field(default=1000, ge=100, le=10000)
    strict_mode: bool = True


class PIIMatchResponse(BaseModel):
    """Single PII match in response."""
    column_name: str
    pii_type: str
    sensitivity: str
    detection_method: str
    confidence: float
    recommendation: str


class PIIScanResponse(BaseModel):
    """Response from PII scan."""
    total_columns: int
    columns_with_pii: int
    passed: bool
    matches: list[PIIMatchResponse]
    scan_timestamp: str


class PIITransformRequest(BaseModel):
    """Request to transform PII in a dataset."""
    dataset_type: DatasetType
    custom_transforms: Optional[dict[str, str]] = None  # column -> transform type


class TransformationResultResponse(BaseModel):
    """Response for a single transformation."""
    column_name: str
    transformation_type: str
    rows_transformed: int
    success: bool


class PIITransformResponse(BaseModel):
    """Response from PII transformation."""
    total_columns: int
    successful: int
    results: list[TransformationResultResponse]


class CertificationResponse(BaseModel):
    """Response from certification."""
    certified: bool
    certification_level: str
    score: float
    gates_passed: list[bool]
    warnings: list[str]
    recommendations: list[str]
    certificate: str


# Global instances
_detector: Optional[PIIDetector] = None
_transformer: Optional[PIITransformer] = None
_certifier: Optional[PIICertifier] = None
_audit_log: Optional[PIIAuditLog] = None
_cached_results: dict[str, PIIDetectionResult] = {}


def get_detector() -> PIIDetector:
    global _detector
    if _detector is None:
        _detector = PIIDetector()
    return _detector


def get_transformer() -> PIITransformer:
    global _transformer
    if _transformer is None:
        _transformer = PIITransformer()
    return _transformer


def get_certifier() -> PIICertifier:
    global _certifier
    if _certifier is None:
        _certifier = PIICertifier()
    return _certifier


def get_audit_log() -> PIIAuditLog:
    global _audit_log
    if _audit_log is None:
        _audit_log = PIIAuditLog(actor="api")
    return _audit_log


# ============================================
# Endpoints
# ============================================
@router.post("/scan", response_model=PIIScanResponse)
async def scan_for_pii(request: PIIScanRequest):
    """
    Scan a dataset for PII using the 5-Gate Pipeline (Gates 1 & 2).

    - **dataset_type**: Type of dataset to scan
    - **sample_size**: Number of rows to sample for analysis
    - **strict_mode**: If true, any PII detection causes failure
    """
    detector = get_detector()
    audit = get_audit_log()
    loader = DataLoader()

    try:
        # Load dataset (small sample for scanning)
        loader.connect()
        if request.dataset_type == DatasetType.BANK_ACCOUNT_FRAUD:
            df = loader.load_bank_account_fraud(sample_frac=0.01)
        elif request.dataset_type == DatasetType.CREDIT_CARD_FRAUD:
            df = loader.load_credit_card_fraud(sample_frac=0.1)
        elif request.dataset_type == DatasetType.PAYSIM:
            df = loader.load_paysim(sample_frac=0.01)
        else:
            raise HTTPException(400, f"Unknown dataset type: {request.dataset_type}")

        # Log scan start
        audit.log_scan_started(
            request.dataset_type.value,
            num_columns=len(df.columns),
            num_rows=len(df),
        )

        # Run detection
        result = detector.detect(df, sample_size=request.sample_size)

        # Cache result for transformation
        _cached_results[request.dataset_type.value] = result

        # Log scan completion
        audit.log_scan_completed(request.dataset_type.value, result)

        return PIIScanResponse(
            total_columns=result.total_columns,
            columns_with_pii=result.columns_with_pii,
            passed=result.passed,
            matches=[
                PIIMatchResponse(
                    column_name=m.column_name,
                    pii_type=m.pii_type.value,
                    sensitivity=m.sensitivity.value,
                    detection_method=m.detection_method,
                    confidence=m.confidence,
                    recommendation=m.recommendation,
                )
                for m in result.matches
            ],
            scan_timestamp=result.scan_timestamp,
        )

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.error(f"Error scanning for PII: {e}")
        audit.log_error(request.dataset_type.value, str(e), "scan_error")
        raise HTTPException(500, f"PII scan failed: {e}")
    finally:
        loader.close()


@router.post("/transform", response_model=PIITransformResponse)
async def transform_pii(request: PIITransformRequest):
    """
    Transform PII in a dataset using appropriate sanitization methods.

    - **dataset_type**: Type of dataset to transform
    - **custom_transforms**: Optional override for transformation types
    """
    transformer = get_transformer()
    audit = get_audit_log()
    loader = DataLoader()

    # Check for cached scan result
    if request.dataset_type.value not in _cached_results:
        raise HTTPException(
            400,
            f"No scan results cached for {request.dataset_type.value}. Run /scan first.",
        )

    detection_result = _cached_results[request.dataset_type.value]

    try:
        # Load full dataset
        loader.connect()
        if request.dataset_type == DatasetType.BANK_ACCOUNT_FRAUD:
            df = loader.load_bank_account_fraud()
        elif request.dataset_type == DatasetType.CREDIT_CARD_FRAUD:
            df = loader.load_credit_card_fraud()
        elif request.dataset_type == DatasetType.PAYSIM:
            df = loader.load_paysim()
        else:
            raise HTTPException(400, f"Unknown dataset type: {request.dataset_type}")

        # Convert custom transforms
        custom_transforms = None
        if request.custom_transforms:
            custom_transforms = {
                col: TransformationType(t) for col, t in request.custom_transforms.items()
            }

        # Log transformation start
        audit.log_transformation_started(
            request.dataset_type.value,
            num_columns=len(detection_result.matches),
        )

        # Run transformation
        transformed_df, results = transformer.transform(
            df, detection_result, custom_transforms
        )

        # Log transformation completion
        audit.log_transformation_completed(request.dataset_type.value, results)

        # Save transformed data
        output_path = loader.save_processed(
            transformed_df,
            f"{request.dataset_type.value}_sanitized",
            format="parquet",
        )

        log.info(f"Transformed data saved to {output_path}")

        successful = sum(1 for r in results if r.success)
        return PIITransformResponse(
            total_columns=len(results),
            successful=successful,
            results=[
                TransformationResultResponse(
                    column_name=r.column_name,
                    transformation_type=r.transformation_type.value,
                    rows_transformed=r.rows_transformed,
                    success=r.success,
                )
                for r in results
            ],
        )

    except Exception as e:
        log.error(f"Error transforming PII: {e}")
        audit.log_error(request.dataset_type.value, str(e), "transform_error")
        raise HTTPException(500, f"PII transformation failed: {e}")
    finally:
        loader.close()


@router.post("/certify/{dataset_type}", response_model=CertificationResponse)
async def certify_dataset(dataset_type: DatasetType):
    """
    Run full 5-Gate certification on a transformed dataset.

    - **dataset_type**: Type of dataset to certify
    """
    certifier = get_certifier()
    audit = get_audit_log()
    loader = DataLoader()

    # Check for cached scan result
    if dataset_type.value not in _cached_results:
        raise HTTPException(
            400,
            f"No scan results cached for {dataset_type.value}. Run /scan first.",
        )

    detection_result = _cached_results[dataset_type.value]

    try:
        # Load transformed data
        from sentinxfl.core.config import settings
        import polars as pl

        processed_path = settings.processed_dir_abs / f"{dataset_type.value}_sanitized.parquet"

        if not processed_path.exists():
            raise HTTPException(
                400,
                f"Transformed data not found. Run /transform first.",
            )

        df = pl.read_parquet(processed_path)

        # Run certification
        result = certifier.certify(df, detection_result)

        # Log certification
        audit.log_certification(dataset_type.value, result)

        # Generate certificate
        certificate = certifier.generate_certificate(result)

        return CertificationResponse(
            certified=result.certified,
            certification_level=result.certification_level,
            score=result.certification_score,
            gates_passed=[
                result.gate1_passed,
                result.gate2_passed,
                result.gate3_passed,
                result.gate4_passed,
                result.gate5_passed,
            ],
            warnings=result.warnings,
            recommendations=result.recommendations,
            certificate=certificate,
        )

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.error(f"Error certifying dataset: {e}")
        audit.log_error(dataset_type.value, str(e), "certification_error")
        raise HTTPException(500, f"Certification failed: {e}")


@router.get("/audit/export")
async def export_audit_trail():
    """Export the current audit trail as JSON."""
    audit = get_audit_log()

    try:
        filepath = audit.export_json()
        return {
            "message": "Audit trail exported",
            "path": str(filepath),
            "event_count": audit.get_trail().event_count,
        }
    except Exception as e:
        log.error(f"Error exporting audit trail: {e}")
        raise HTTPException(500, f"Failed to export audit trail: {e}")


@router.get("/audit/report")
async def get_compliance_report():
    """Get a formatted compliance report."""
    audit = get_audit_log()

    try:
        report = audit.generate_compliance_report()
        is_valid, issues = audit.verify_chain_integrity()

        return {
            "report": report,
            "chain_integrity": is_valid,
            "integrity_issues": issues,
        }
    except Exception as e:
        log.error(f"Error generating compliance report: {e}")
        raise HTTPException(500, f"Failed to generate report: {e}")


@router.get("/transformation-types")
async def list_transformation_types():
    """List available transformation types."""
    return {
        "types": [t.value for t in TransformationType],
        "descriptions": {
            "hash": "One-way SHA-256 hash",
            "hmac": "Keyed hash (deterministic)",
            "mask": "Partial character masking",
            "generalize": "Binning/bucketing",
            "pseudonymize": "Consistent fake replacement",
            "redact": "Complete removal",
            "tokenize": "Format-preserving tokenization",
            "suppress": "Replace with null",
            "noise": "Add differential privacy noise",
        },
    }
