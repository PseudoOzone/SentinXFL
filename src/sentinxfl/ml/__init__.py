"""
SentinXFL ML Models Module
===========================

Machine learning models for fraud detection:
- XGBoost: Gradient boosting
- LightGBM: Fast gradient boosting  
- IsolationForest: Anomaly detection
- TabNet: Interpretable neural network
- Ensemble: Model combination

Author: Anshuman Bakshi
"""

from sentinxfl.ml.base import BaseModel, ModelMetadata, ModelType
from sentinxfl.ml.metrics import ClassificationMetrics, MetricsCalculator, compare_models
from sentinxfl.ml.registry import (
    ModelRegistry,
    create_model,
    load_model,
    list_saved_models,
)
from sentinxfl.ml.pipeline import TrainingPipeline, TrainingResult

__all__ = [
    # Base
    "BaseModel",
    "ModelMetadata", 
    "ModelType",
    # Metrics
    "ClassificationMetrics",
    "MetricsCalculator",
    "compare_models",
    # Registry
    "ModelRegistry",
    "create_model",
    "load_model",
    "list_saved_models",
    # Pipeline
    "TrainingPipeline",
    "TrainingResult",
]