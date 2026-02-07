"""Privacy module for SentinXFL - 5-Gate PII Blocking Pipeline + Differential Privacy."""

from sentinxfl.privacy.detector import PIIDetector
from sentinxfl.privacy.transformer import PIITransformer
from sentinxfl.privacy.certifier import PIICertifier
from sentinxfl.privacy.audit import PIIAuditLog

# Differential Privacy modules
from sentinxfl.privacy.mechanisms import (
    GaussianMechanism,
    LaplaceMechanism,
    ExponentialMechanism,
    GradientClipper,
)
from sentinxfl.privacy.accountant import (
    RDPAccountant,
    get_accountant,
    reset_accountant,
)
from sentinxfl.privacy.dp_trainer import (
    DPSGDConfig,
    DPSGDTrainer,
    DPGradientBoostTrainer,
    compute_dp_params,
)

__all__ = [
    # PII Pipeline
    "PIIDetector",
    "PIITransformer", 
    "PIICertifier",
    "PIIAuditLog",
    # DP Mechanisms
    "GaussianMechanism",
    "LaplaceMechanism",
    "ExponentialMechanism",
    "GradientClipper",
    # Privacy Accountant
    "RDPAccountant",
    "get_accountant",
    "reset_accountant",
    # DP Training
    "DPSGDConfig",
    "DPSGDTrainer",
    "DPGradientBoostTrainer",
    "compute_dp_params",
]
