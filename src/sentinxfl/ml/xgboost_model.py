"""
SentinXFL - XGBoost Model
==========================

XGBoost implementation optimized for fraud detection.
Supports GPU acceleration (RTX 3050) and class imbalance handling.

Author: Anshuman Bakshi
"""

import json
from pathlib import Path
from typing import Any, Self

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.base import BaseModel, ModelMetadata, ModelType

logger = get_logger(__name__)
settings = get_settings()


class XGBoostModel(BaseModel):
    """
    XGBoost model for fraud detection.
    
    Optimized for:
    - Highly imbalanced datasets (scale_pos_weight)
    - GPU acceleration when available
    - Early stopping to prevent overfitting
    """
    
    # Default hyperparameters optimized for fraud detection
    DEFAULT_PARAMS = {
        # Booster parameters
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        
        # Regularization
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "gamma": 0.1,  # Min split loss
        
        # Training
        "early_stopping_rounds": 50,
        "eval_metric": ["auc", "aucpr"],
        
        # Performance
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 1,
    }
    
    def __init__(
        self,
        name: str = "xgboost_fraud",
        use_gpu: bool = True,
        **hyperparameters: Any,
    ):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
            use_gpu: Whether to use GPU (CUDA) acceleration
            **hyperparameters: Override default hyperparameters
        """
        # Merge with defaults
        params = self.DEFAULT_PARAMS.copy()
        params.update(hyperparameters)
        
        super().__init__(name=name, **params)
        
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.metadata.model_type = ModelType.XGBOOST
        
        # Store for later
        self._scale_pos_weight: float | None = None
    
    def _check_gpu_available(self) -> bool:
        """Check if CUDA GPU is available for XGBoost."""
        try:
            # Try creating a small GPU booster
            import xgboost as xgb
            dtest = xgb.DMatrix(np.array([[1.0, 2.0]]))
            params = {"device": "cuda"}
            booster = xgb.train(params, dtest, num_boost_round=1)
            return True
        except Exception:
            logger.debug("GPU not available for XGBoost, using CPU")
            return False
    
    def _create_model(self, enable_early_stopping: bool = True) -> xgb.XGBClassifier:
        """Create XGBoost classifier."""
        params = self.metadata.hyperparameters.copy()
        
        # Remove training-specific params
        early_stopping = params.pop("early_stopping_rounds", 50)
        
        # Set device
        if self.use_gpu:
            params["device"] = "cuda"
            params["tree_method"] = "hist"  # GPU-optimized
            logger.info(f"XGBoost using GPU acceleration")
        else:
            params["device"] = "cpu"
            params["tree_method"] = "hist"
        
        # Add scale_pos_weight if computed
        if self._scale_pos_weight is not None:
            params["scale_pos_weight"] = self._scale_pos_weight
        
        # Only set early_stopping_rounds if a validation set will be provided
        es_kwargs = {}
        if enable_early_stopping:
            es_kwargs["early_stopping_rounds"] = early_stopping
        
        return xgb.XGBClassifier(
            **params,
            **es_kwargs,
            enable_categorical=False,
        )
    
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        auto_balance: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Train XGBoost model.
        
        Args:
            X: Training features
            y: Training labels (0 = legitimate, 1 = fraud)
            X_val: Validation features for early stopping
            y_val: Validation labels
            auto_balance: Automatically compute scale_pos_weight
            **kwargs: Additional fit arguments
        """
        X, y = self._to_numpy(X, y)
        
        # Compute class weight for imbalanced data
        if auto_balance:
            n_negative = np.sum(y == 0)
            n_positive = np.sum(y == 1)
            if n_positive > 0:
                self._scale_pos_weight = n_negative / n_positive
                logger.info(f"Auto-balance: scale_pos_weight={self._scale_pos_weight:.2f}")
        
        # Prepare validation set
        eval_set = None
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val, y_val = self._to_numpy(X_val, y_val)
            eval_set = [(X_val, y_val)]
        
        # Create model (disable early stopping if no validation set)
        self._model = self._create_model(enable_early_stopping=has_val)
        
        # Train
        logger.info(f"Training XGBoost: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=kwargs.get("verbose", 100),
        )
        
        # Update metadata
        self._is_fitted = True
        self.metadata.trained_at = __import__("datetime").datetime.now()
        self.metadata.n_samples = X.shape[0]
        self.metadata.n_features = X.shape[1]
        
        # Log best iteration
        if hasattr(self._model, "best_iteration"):
            logger.info(f"Best iteration: {self._model.best_iteration}")
        
        return self
    
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict class labels."""
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        return self._model.predict(X).astype(np.int32)
    
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities."""
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        return self._model.predict_proba(X).astype(np.float32)
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores (gain-based)."""
        self._validate_fitted()
        
        importance = self._model.feature_importances_
        feature_names = self.metadata.feature_names or [f"f{i}" for i in range(len(importance))]
        
        return dict(sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ))
    
    def save(self, path: Path | str) -> Path:
        """Save model to disk."""
        self._validate_fitted()
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = path / "model.ubj"
        self._model.save_model(str(model_path))
        
        # Save metadata
        meta_path = path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save additional config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "use_gpu": self.use_gpu,
                "scale_pos_weight": self._scale_pos_weight,
            }, f, indent=2)
        
        logger.info(f"Saved XGBoost model to {path}")
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
            use_gpu=config.get("use_gpu", False),
            **meta_dict.get("hyperparameters", {}),
        )
        instance.metadata = ModelMetadata.from_dict(meta_dict)
        instance._scale_pos_weight = config.get("scale_pos_weight")
        
        # Load XGBoost model
        model_path = path / "model.ubj"
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(str(model_path))
        instance._is_fitted = True
        
        logger.info(f"Loaded XGBoost model from {path}")
        return instance
    
    # ==========================================
    # Federated Learning Support
    # ==========================================
    
    def get_weights(self) -> dict[str, NDArray[np.float32]]:
        """
        Get model weights for federated averaging.
        
        For tree models, we extract the tree structure as JSON.
        Not directly averageable, but can be used for model transfer.
        """
        self._validate_fitted()
        
        # Get booster and dump to JSON
        booster = self._model.get_booster()
        trees_json = booster.save_raw(raw_format="json")
        
        # Return as bytes array for compatibility
        return {
            "trees_json": np.frombuffer(trees_json, dtype=np.uint8),
            "n_estimators": np.array([self._model.n_estimators], dtype=np.int32),
        }
    
    def get_booster(self) -> xgb.Booster:
        """Get underlying XGBoost booster."""
        self._validate_fitted()
        return self._model.get_booster()
