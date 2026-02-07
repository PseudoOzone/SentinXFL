"""
SentinXFL - ML API Routes
==========================

REST API endpoints for model training, prediction, and management.

Author: Anshuman Bakshi
"""

from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel as PydanticModel, Field

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.registry import ModelRegistry, list_saved_models, load_model
from sentinxfl.ml.base import BaseModel

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# ==========================================
# Loaded models cache
# ==========================================
_loaded_models: dict[str, BaseModel] = {}


# ==========================================
# Request/Response Models
# ==========================================

class PredictionRequest(PydanticModel):
    """Prediction request schema."""
    
    features: list[list[float]] = Field(
        ...,
        description="2D array of features (n_samples x n_features)",
        min_length=1,
    )
    model_name: str = Field(
        default="default",
        description="Name of loaded model to use",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold",
    )


class PredictionResponse(PydanticModel):
    """Prediction response schema."""
    
    predictions: list[int] = Field(..., description="Predicted labels (0=legit, 1=fraud)")
    probabilities: list[float] = Field(..., description="Fraud probabilities")
    model_name: str
    n_samples: int
    n_fraud_predicted: int
    fraud_rate: float


class TrainingRequest(PydanticModel):
    """Training request schema."""
    
    dataset_name: str = Field(
        default="credit_card",
        description="Dataset to train on",
    )
    model_type: Literal["xgboost", "lightgbm", "isolation_forest", "tabnet"] = Field(
        default="xgboost",
        description="Model type to train",
    )
    model_name: str | None = Field(
        default=None,
        description="Custom model name",
    )
    sample_frac: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Fraction of data to use",
    )
    run_cv: bool = Field(
        default=False,
        description="Run cross-validation",
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters",
    )


class TrainingResponse(PydanticModel):
    """Training response schema."""
    
    status: str
    model_name: str
    model_type: str
    training_time_seconds: float
    test_auc_roc: float
    test_auc_pr: float
    test_f1: float
    test_recall: float
    n_samples: int
    model_path: str | None


class ModelInfo(PydanticModel):
    """Model information schema."""
    
    name: str
    model_type: str
    is_loaded: bool
    path: str | None
    created_at: str | None
    trained_at: str | None
    n_features: int
    metrics: dict[str, float]


class FeatureImportanceResponse(PydanticModel):
    """Feature importance response schema."""
    
    model_name: str
    feature_importance: dict[str, float]
    top_features: list[str]


# ==========================================
# Model Management Endpoints
# ==========================================

@router.get("/models", response_model=list[dict[str, Any]])
async def list_models(
    include_loaded: bool = Query(True, description="Include currently loaded models"),
) -> list[dict[str, Any]]:
    """List all available models (saved and loaded)."""
    models = []
    
    # Saved models
    for model_info in list_saved_models():
        model_info["is_loaded"] = model_info["name"] in _loaded_models
        models.append(model_info)
    
    # Loaded-only models (not saved)
    if include_loaded:
        for name, model in _loaded_models.items():
            if not any(m["name"] == name for m in models):
                models.append({
                    "name": name,
                    "model_type": model.metadata.model_type,
                    "is_loaded": True,
                    "path": None,
                    "created_at": model.metadata.created_at.isoformat() if model.metadata.created_at else None,
                })
    
    return models


@router.get("/models/types")
async def list_model_types() -> list[str]:
    """Get available model types."""
    return ModelRegistry.list_types()


@router.post("/models/load")
async def load_model_endpoint(
    path: str,
    name: str | None = None,
) -> dict[str, str]:
    """Load a saved model into memory."""
    from pathlib import Path
    
    model = load_model(Path(path))
    model_name = name or model.name
    
    _loaded_models[model_name] = model
    
    logger.info(f"Loaded model '{model_name}' from {path}")
    return {"status": "loaded", "model_name": model_name}


@router.delete("/models/{model_name}")
async def unload_model(model_name: str) -> dict[str, str]:
    """Unload a model from memory."""
    if model_name not in _loaded_models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    
    del _loaded_models[model_name]
    logger.info(f"Unloaded model '{model_name}'")
    
    return {"status": "unloaded", "model_name": model_name}


@router.get("/models/{model_name}/info", response_model=ModelInfo)
async def get_model_info(model_name: str) -> ModelInfo:
    """Get information about a loaded model."""
    if model_name not in _loaded_models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    
    model = _loaded_models[model_name]
    
    return ModelInfo(
        name=model.name,
        model_type=model.metadata.model_type,
        is_loaded=True,
        path=None,  # Would need to track this
        created_at=model.metadata.created_at.isoformat() if model.metadata.created_at else None,
        trained_at=model.metadata.trained_at.isoformat() if model.metadata.trained_at else None,
        n_features=model.metadata.n_features,
        metrics=model.metadata.metrics,
    )


@router.get("/models/{model_name}/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    model_name: str,
    top_k: int = Query(10, ge=1, le=100),
) -> FeatureImportanceResponse:
    """Get feature importance from a trained model."""
    if model_name not in _loaded_models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    
    model = _loaded_models[model_name]
    
    try:
        importance = model.get_feature_importance()
    except (NotImplementedError, RuntimeError) as e:
        raise HTTPException(400, f"Feature importance not available: {e}")
    
    top_features = list(importance.keys())[:top_k]
    
    return FeatureImportanceResponse(
        model_name=model_name,
        feature_importance=importance,
        top_features=top_features,
    )


# ==========================================
# Prediction Endpoints
# ==========================================

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make fraud predictions using a loaded model.
    
    Returns predicted labels and fraud probabilities.
    """
    if request.model_name not in _loaded_models:
        # Try to find default
        if "default" in _loaded_models:
            model = _loaded_models["default"]
        elif len(_loaded_models) == 1:
            model = list(_loaded_models.values())[0]
        else:
            raise HTTPException(404, f"Model '{request.model_name}' not loaded. Available: {list(_loaded_models.keys())}")
    else:
        model = _loaded_models[request.model_name]
    
    # Convert to numpy
    X = np.array(request.features, dtype=np.float32)
    
    # Predict
    probabilities = model.predict_fraud_probability(X)
    predictions = (probabilities >= request.threshold).astype(int)
    
    n_fraud = int(predictions.sum())
    
    return PredictionResponse(
        predictions=predictions.tolist(),
        probabilities=probabilities.tolist(),
        model_name=model.name,
        n_samples=len(predictions),
        n_fraud_predicted=n_fraud,
        fraud_rate=n_fraud / len(predictions) if len(predictions) > 0 else 0.0,
    )


@router.post("/predict/batch")
async def predict_batch(
    features: list[list[float]],
    model_name: str = "default",
    batch_size: int = 10000,
) -> dict[str, Any]:
    """
    Batch prediction for large datasets.
    
    Processes in chunks to manage memory.
    """
    if model_name not in _loaded_models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    
    model = _loaded_models[model_name]
    X = np.array(features, dtype=np.float32)
    
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        proba = model.predict_fraud_probability(batch)
        pred = (proba >= 0.5).astype(int)
        
        all_predictions.extend(pred.tolist())
        all_probabilities.extend(proba.tolist())
    
    return {
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "n_samples": len(all_predictions),
        "n_fraud_predicted": sum(all_predictions),
    }


# ==========================================
# Training Endpoints
# ==========================================

@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
) -> TrainingResponse:
    """
    Train a new fraud detection model.
    
    For large datasets, training runs in the background.
    """
    from sentinxfl.data.loader import DataLoader
    from sentinxfl.ml.pipeline import TrainingPipeline
    
    # Load data
    loader = DataLoader()
    
    if request.dataset_name == "credit_card":
        df = loader.load_credit_card_fraud(sample_frac=request.sample_frac)
        target_col = "Class"
    elif request.dataset_name == "paysim":
        df = loader.load_paysim(sample_frac=request.sample_frac)
        target_col = "isFraud"
    elif request.dataset_name == "bank_account":
        df = loader.load_bank_account_fraud(sample_frac=request.sample_frac)
        target_col = "fraud_bool"
    else:
        raise HTTPException(400, f"Unknown dataset: {request.dataset_name}")
    
    # Exclude non-numeric columns
    exclude_cols = [c for c in df.columns if df.schema[c] not in [int, float, bool]]
    
    # Train
    pipeline = TrainingPipeline()
    result = pipeline.train_and_evaluate(
        model_type=request.model_type,
        df=df,
        target_col=target_col,
        exclude_cols=exclude_cols,
        run_cv=request.run_cv,
        save_model=True,
        model_name=request.model_name,
        **request.hyperparameters,
    )
    
    # Load trained model
    if result.model_path:
        model = load_model(result.model_path)
        _loaded_models[result.model_name] = model
    
    return TrainingResponse(
        status="completed",
        model_name=result.model_name,
        model_type=result.model_type,
        training_time_seconds=result.training_time_seconds,
        test_auc_roc=result.test_metrics.auc_roc if result.test_metrics else 0.0,
        test_auc_pr=result.test_metrics.auc_pr if result.test_metrics else 0.0,
        test_f1=result.test_metrics.f1_score if result.test_metrics else 0.0,
        test_recall=result.test_metrics.recall if result.test_metrics else 0.0,
        n_samples=result.n_samples_train + result.n_samples_val + result.n_samples_test,
        model_path=str(result.model_path) if result.model_path else None,
    )


# ==========================================
# Model Comparison
# ==========================================

@router.post("/compare")
async def compare_models(
    dataset_name: str = "credit_card",
    model_types: list[str] = ["xgboost", "lightgbm", "isolation_forest"],
    sample_frac: float = 0.1,
) -> dict[str, Any]:
    """
    Train and compare multiple model types on the same dataset.
    """
    from sentinxfl.data.loader import DataLoader
    from sentinxfl.ml.pipeline import TrainingPipeline
    
    # Load data
    loader = DataLoader()
    
    if dataset_name == "credit_card":
        df = loader.load_credit_card_fraud(sample_frac=sample_frac)
        target_col = "Class"
    else:
        raise HTTPException(400, f"Unsupported dataset for comparison: {dataset_name}")
    
    exclude_cols = [c for c in df.columns if df.schema[c] not in [int, float, bool]]
    
    # Compare
    pipeline = TrainingPipeline()
    results = pipeline.compare_models(
        df=df,
        target_col=target_col,
        model_types=model_types,
        exclude_cols=exclude_cols,
    )
    
    # Format response
    comparison = {}
    for model_type, result in results.items():
        if result.test_metrics:
            comparison[model_type] = {
                "auc_roc": result.test_metrics.auc_roc,
                "auc_pr": result.test_metrics.auc_pr,
                "f1_score": result.test_metrics.f1_score,
                "recall": result.test_metrics.recall,
                "precision": result.test_metrics.precision,
                "training_time": result.training_time_seconds,
            }
    
    # Find best
    best_model = max(comparison.keys(), key=lambda k: comparison[k]["auc_pr"])
    
    return {
        "comparison": comparison,
        "best_model": best_model,
        "best_auc_pr": comparison[best_model]["auc_pr"],
        "dataset": dataset_name,
        "sample_frac": sample_frac,
    }
