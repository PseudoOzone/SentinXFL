"""
SentinXFL Data Schemas
=======================

Pydantic models for data validation and API responses.

Author: Anshuman Bakshi
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    """Supported dataset types."""
    BANK_ACCOUNT_FRAUD = "bank_account_fraud"
    CREDIT_CARD_FRAUD = "credit_card_fraud"
    PAYSIM = "paysim"


class TransactionType(str, Enum):
    """Transaction types from PaySim."""
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class FraudType(str, Enum):
    """Unified fraud types across datasets."""
    LEGITIMATE = "legitimate"
    ACCOUNT_TAKEOVER = "account_takeover"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    CARD_NOT_PRESENT = "card_not_present"
    APPLICATION_FRAUD = "application_fraud"
    MONEY_MULE = "money_mule"
    UNKNOWN = "unknown"


class TransactionRecord(BaseModel):
    """Unified transaction record schema.

    Normalizes transactions from all three datasets into a common format.
    """
    # Core identifiers
    transaction_id: str = Field(..., description="Unique transaction identifier")
    dataset_source: DatasetType = Field(..., description="Source dataset")

    # Temporal
    timestamp: Optional[datetime] = Field(None, description="Transaction timestamp")
    step: Optional[int] = Field(None, description="Time step (for PaySim)")

    # Account information (anonymized)
    account_id_hash: Optional[str] = Field(None, description="Hashed account ID")
    device_id_hash: Optional[str] = Field(None, description="Hashed device ID")

    # Transaction details
    transaction_type: Optional[str] = Field(None, description="Type of transaction")
    amount: float = Field(..., ge=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code")

    # Balance information
    balance_before: Optional[float] = Field(None, description="Balance before transaction")
    balance_after: Optional[float] = Field(None, description="Balance after transaction")

    # Feature engineering candidates
    velocity_1h: Optional[float] = Field(None, description="Transaction velocity (1 hour)")
    velocity_24h: Optional[float] = Field(None, description="Transaction velocity (24 hours)")
    amount_zscore: Optional[float] = Field(None, description="Amount z-score for account")

    # Labels
    is_fraud: bool = Field(..., description="Fraud label")
    fraud_type: FraudType = Field(default=FraudType.UNKNOWN, description="Type of fraud")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_001",
                "dataset_source": "paysim",
                "amount": 1500.00,
                "is_fraud": False,
                "fraud_type": "legitimate",
            }
        }
    }


class DatasetStats(BaseModel):
    """Statistics for a loaded dataset."""
    name: str
    dataset_type: DatasetType
    total_rows: int = Field(..., ge=0)
    fraud_count: int = Field(..., ge=0)
    legitimate_count: int = Field(..., ge=0)
    fraud_ratio: float = Field(..., ge=0, le=1)
    num_features: int = Field(..., ge=0)
    memory_mb: float = Field(..., ge=0)
    file_path: str
    loaded_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def imbalance_ratio(self) -> float:
        """Calculate class imbalance ratio (legitimate:fraud)."""
        if self.fraud_count == 0:
            return float('inf')
        return self.legitimate_count / self.fraud_count


class UnifiedDatasetStats(BaseModel):
    """Statistics for the combined unified dataset."""
    total_rows: int
    total_fraud: int
    total_legitimate: int
    overall_fraud_ratio: float
    datasets: list[DatasetStats]
    combined_features: int
    total_memory_mb: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DataSplitInfo(BaseModel):
    """Information about train/val/test splits."""
    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_fraud_ratio: float
    val_fraud_ratio: float
    test_fraud_ratio: float
    stratified: bool = True
    random_seed: int


class PIIReport(BaseModel):
    """Report of PII detection results."""
    total_records_scanned: int
    pii_detected: bool
    pii_fields_found: list[str]
    pii_patterns_matched: dict[str, int]
    confidence_scores: dict[str, float]
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow)
    passed_certification: bool = False


class PIITransformationLog(BaseModel):
    """Audit log for PII transformations."""
    record_id: str
    field_name: str
    transformation_type: str  # hash, mask, redact, generalize
    original_entropy: Optional[float] = None
    transformed_entropy: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
