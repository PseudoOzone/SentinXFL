"""
SentinXFL - Isolation Forest Model
===================================

Isolation Forest for unsupervised anomaly detection.
Useful when fraud labels are unavailable or unreliable.

Author: Anshuman Bakshi
"""

import json
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.base import BaseModel, ModelMetadata, ModelType

logger = get_logger(__name__)
settings = get_settings()


class IsolationForestModel(BaseModel):
    """
    Isolation Forest for anomaly-based fraud detection.
    
    Key advantages:
    - Unsupervised: doesn't require fraud labels
    - Fast: O(n log n) complexity
    - Low memory footprint
    - Handles high-dimensional data well
    
    Note: Anomaly scores are converted to probabilities using
    the decision_function, scaled to [0, 1] range.
    """
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "max_samples": "auto",
        "contamination": 0.01,  # Expected fraud rate
        "max_features": 1.0,
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": 42,
        "warm_start": False,
    }
    
    def __init__(
        self,
        name: str = "isolation_forest_fraud",
        **hyperparameters: Any,
    ):
        """
        Initialize Isolation Forest model.
        
        Args:
            name: Model name
            **hyperparameters: Override default hyperparameters
        """
        # Merge with defaults
        params = self.DEFAULT_PARAMS.copy()
        params.update(hyperparameters)
        
        super().__init__(name=name, **params)
        self.metadata.model_type = ModelType.ISOLATION_FOREST
        
        # Store score range for probability conversion
        self._score_min: float = 0.0
        self._score_max: float = 1.0
    
    def _create_model(self) -> IsolationForest:
        """Create Isolation Forest model."""
        params = self.metadata.hyperparameters.copy()
        return IsolationForest(**params)
    
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32] | None = None,  # Not used, included for interface compatibility
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Train Isolation Forest.
        
        Note: y is ignored (unsupervised learning).
        
        Args:
            X: Training features
            y: Ignored (for interface compatibility)
            X_val: Validation features (used for score calibration)
            y_val: Validation labels (used for threshold tuning)
            **kwargs: Additional arguments
        """
        X, _ = self._to_numpy(X)
        
        # Create and train model
        self._model = self._create_model()
        
        logger.info(f"Training IsolationForest: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        self._model.fit(X)
        
        # Calibrate score range using decision function
        scores = self._model.decision_function(X)
        self._score_min = float(scores.min())
        self._score_max = float(scores.max())
        
        logger.info(f"Score range: [{self._score_min:.4f}, {self._score_max:.4f}]")
        
        # Update metadata
        self._is_fitted = True
        self.metadata.trained_at = __import__("datetime").datetime.now()
        self.metadata.n_samples = X.shape[0]
        self.metadata.n_features = X.shape[1]
        
        # If validation labels provided, tune threshold
        if X_val is not None and y_val is not None:
            self._tune_threshold(X_val, y_val)
        
        return self
    
    def _tune_threshold(self, X_val: NDArray[np.float32], y_val: NDArray[np.int32]) -> None:
        """Tune decision threshold using validation data and labels."""
        from sklearn.metrics import f1_score
        
        X_val, y_val = self._to_numpy(X_val, y_val)
        
        # Get anomaly scores
        proba = self.predict_fraud_probability(X_val)
        
        # Grid search for best threshold
        best_f1 = 0.0
        best_threshold = 0.5
        
        for threshold in np.linspace(0.1, 0.9, 17):
            y_pred = (proba >= threshold).astype(np.int32)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.metadata.hyperparameters["optimal_threshold"] = best_threshold
        logger.info(f"Tuned threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
    
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Predict class labels (0 = normal, 1 = anomaly/fraud).
        
        Uses contamination-based threshold from IsolationForest.
        """
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        
        # IsolationForest returns -1 for anomalies, 1 for normal
        raw_pred = self._model.predict(X)
        
        # Convert to 0/1 (1 = fraud/anomaly)
        return (raw_pred == -1).astype(np.int32)
    
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Predict class probabilities.
        
        Converts anomaly scores to probabilities using:
        1. decision_function (higher = more normal)
        2. Negate and scale to [0, 1]
        3. Return as 2-column array [P(normal), P(fraud)]
        """
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        
        # Get anomaly scores (negative = anomaly, positive = normal)
        scores = self._model.decision_function(X)
        
        # Convert to fraud probability (negate and scale)
        # More negative score = more anomalous = higher fraud probability
        range_val = self._score_max - self._score_min
        if range_val > 0:
            fraud_proba = 1 - (scores - self._score_min) / range_val
        else:
            fraud_proba = np.ones_like(scores) * 0.5
        
        # Clip to [0, 1]
        fraud_proba = np.clip(fraud_proba, 0, 1)
        
        # Return 2-column array
        normal_proba = 1 - fraud_proba
        return np.column_stack([normal_proba, fraud_proba]).astype(np.float32)
    
    def get_anomaly_score(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Get raw anomaly scores.
        
        Returns decision_function values where:
        - Negative values indicate anomalies
        - Positive values indicate normal points
        - Values close to zero are near the boundary
        """
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        return self._model.decision_function(X).astype(np.float32)
    
    def get_feature_importance(self) -> dict[str, float]:
        """
        Feature importance is not directly available for Isolation Forest.
        
        Returns uniform importance as placeholder.
        Use SHAP values for proper feature importance.
        """
        self._validate_fitted()
        
        n_features = self.metadata.n_features
        feature_names = self.metadata.feature_names or [f"f{i}" for i in range(n_features)]
        
        # Uniform importance (Isolation Forest doesn't have native importance)
        importance = 1.0 / n_features
        
        return {name: importance for name in feature_names}
    
    def save(self, path: Path | str) -> Path:
        """Save model to disk."""
        self._validate_fitted()
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn model
        model_path = path / "model.joblib"
        joblib.dump(self._model, model_path)
        
        # Save metadata
        meta_path = path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save score range
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "score_min": self._score_min,
                "score_max": self._score_max,
            }, f, indent=2)
        
        logger.info(f"Saved IsolationForest model to {path}")
        return path
    
    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load model from disk."""
        path = Path(path)
        
        # Load config
        config_path = path / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Load metadata
        meta_path = path / "metadata.json"
        with open(meta_path) as f:
            meta_dict = json.load(f)
        
        # Create instance
        instance = cls(
            name=meta_dict["name"],
            **meta_dict.get("hyperparameters", {}),
        )
        instance.metadata = ModelMetadata.from_dict(meta_dict)
        instance._score_min = config.get("score_min", 0.0)
        instance._score_max = config.get("score_max", 1.0)
        
        # Load sklearn model
        model_path = path / "model.joblib"
        instance._model = joblib.load(model_path)
        instance._is_fitted = True
        
        logger.info(f"Loaded IsolationForest model from {path}")
        return instance
