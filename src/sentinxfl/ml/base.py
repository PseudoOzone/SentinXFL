"""
SentinXFL - Base Model Interface
=================================

Abstract base class defining the interface for all ML models.
Supports serialization, metrics, and federated learning compatibility.

Author: Anshuman Bakshi
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ModelMetadata:
    """Model metadata for tracking and reproducibility."""
    
    name: str
    version: str = "1.0.0"
    model_type: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    trained_at: datetime | None = None
    
    # Training info
    dataset_name: str = ""
    n_samples: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    
    # Hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Federated learning
    fl_round: int | None = None
    client_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "fl_round": self.fl_round,
            "client_id": self.client_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary."""
        data = data.copy()
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("trained_at"):
            data["trained_at"] = datetime.fromisoformat(data["trained_at"])
        return cls(**data)


class BaseModel(ABC):
    """
    Abstract base class for all SentinXFL models.
    
    Provides a consistent interface for:
    - Training (fit)
    - Prediction (predict, predict_proba)
    - Serialization (save, load)
    - Feature importance
    - Federated learning weight extraction
    """
    
    def __init__(self, name: str, **hyperparameters: Any):
        """
        Initialize base model.
        
        Args:
            name: Model name for identification
            **hyperparameters: Model-specific hyperparameters
        """
        self.metadata = ModelMetadata(
            name=name,
            hyperparameters=hyperparameters,
        )
        self._model: Any = None
        self._is_fitted: bool = False
    
    @property
    def name(self) -> str:
        """Model name."""
        return self.metadata.name
    
    @property
    def is_fitted(self) -> bool:
        """Whether model has been trained."""
        return self._is_fitted
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        pass
    
    @abstractmethod
    def fit(
        self,
        X: NDArray[np.float32] | pl.DataFrame,
        y: NDArray[np.int32] | pl.Series,
        X_val: NDArray[np.float32] | pl.DataFrame | None = None,
        y_val: NDArray[np.int32] | pl.Series | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels (0 = legitimate, 1 = fraud)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: NDArray[np.float32] | pl.DataFrame) -> NDArray[np.int32]:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: NDArray[np.float32] | pl.DataFrame) -> NDArray[np.float32]:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        pass
    
    def predict_fraud_probability(self, X: NDArray[np.float32] | pl.DataFrame) -> NDArray[np.float32]:
        """
        Get fraud probability only (convenience method).
        
        Args:
            X: Features to predict
            
        Returns:
            Array of fraud probabilities (column 1)
        """
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba
    
    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def save(self, path: Path | str) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Directory or file path to save to
            
        Returns:
            Path where model was saved
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> Self:
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model instance
        """
        pass
    
    # ==========================================
    # Federated Learning Support
    # ==========================================
    
    def get_weights(self) -> dict[str, NDArray[np.float32]]:
        """
        Get model weights for federated learning.
        
        Returns:
            Dictionary of parameter name to weight array
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support federated weight extraction"
        )
    
    def set_weights(self, weights: dict[str, NDArray[np.float32]]) -> None:
        """
        Set model weights from federated aggregation.
        
        Args:
            weights: Dictionary of parameter name to weight array
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support federated weight setting"
        )
    
    # ==========================================
    # Helper Methods
    # ==========================================
    
    def _to_numpy(
        self,
        X: NDArray[np.float32] | pl.DataFrame,
        y: NDArray[np.int32] | pl.Series | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32] | None]:
        """Convert inputs to numpy arrays if needed."""
        if isinstance(X, pl.DataFrame):
            # Store feature names
            self.metadata.feature_names = X.columns
            X = X.to_numpy().astype(np.float32)
        
        if y is not None and isinstance(y, pl.Series):
            y = y.to_numpy().astype(np.int32)
        
        return X, y
    
    def _validate_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before prediction")
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"


class ModelType(str):
    """Model type constants."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ISOLATION_FOREST = "isolation_forest"
    TABNET = "tabnet"
    ENSEMBLE = "ensemble"
