"""
SentinXFL - LightGBM Model
===========================

LightGBM implementation optimized for fraud detection.
Faster training than XGBoost with comparable accuracy.

Author: Anshuman Bakshi
"""

import json
from pathlib import Path
from typing import Any, Self

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.base import BaseModel, ModelMetadata, ModelType

logger = get_logger(__name__)
settings = get_settings()


class LightGBMModel(BaseModel):
    """
    LightGBM model for fraud detection.
    
    Advantages over XGBoost:
    - Faster training (histogram-based)
    - Lower memory usage
    - Handles categorical features natively
    - Better for large datasets
    """
    
    # Default hyperparameters optimized for fraud detection
    DEFAULT_PARAMS = {
        # Booster parameters
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        
        # Regularization
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "min_split_gain": 0.1,  # Min gain to split
        
        # Training
        "early_stopping_rounds": 50,
        "metric": ["auc", "average_precision"],
        
        # Performance
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
        "force_col_wise": True,  # Better for medium datasets
    }
    
    def __init__(
        self,
        name: str = "lightgbm_fraud",
        use_gpu: bool = False,  # LightGBM GPU requires specific build
        **hyperparameters: Any,
    ):
        """
        Initialize LightGBM model.
        
        Args:
            name: Model name
            use_gpu: Whether to use GPU (requires GPU build)
            **hyperparameters: Override default hyperparameters
        """
        # Merge with defaults
        params = self.DEFAULT_PARAMS.copy()
        params.update(hyperparameters)
        
        super().__init__(name=name, **params)
        
        self.use_gpu = use_gpu
        self.metadata.model_type = ModelType.LIGHTGBM
        
        # Class weight
        self._class_weight: dict[int, float] | None = None
    
    def _create_model(self) -> lgb.LGBMClassifier:
        """Create LightGBM classifier."""
        params = self.metadata.hyperparameters.copy()
        
        # Remove training-specific params
        early_stopping = params.pop("early_stopping_rounds", 50)
        params.pop("metric", None)  # Handled separately
        
        # GPU settings
        if self.use_gpu:
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0
            logger.info("LightGBM using GPU acceleration")
        
        # Add class weights if computed
        if self._class_weight is not None:
            params["class_weight"] = self._class_weight
        
        return lgb.LGBMClassifier(**params)
    
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
        Train LightGBM model.
        
        Args:
            X: Training features
            y: Training labels (0 = legitimate, 1 = fraud)
            X_val: Validation features for early stopping
            y_val: Validation labels
            auto_balance: Automatically compute class weights
            **kwargs: Additional fit arguments
        """
        X, y = self._to_numpy(X, y)
        
        # Compute class weights for imbalanced data
        if auto_balance:
            n_negative = np.sum(y == 0)
            n_positive = np.sum(y == 1)
            total = len(y)
            
            if n_positive > 0:
                # Use balanced weights
                self._class_weight = {
                    0: total / (2 * n_negative),
                    1: total / (2 * n_positive),
                }
                logger.info(f"Auto-balance: class_weight={self._class_weight}")
        
        # Create model
        self._model = self._create_model()
        
        # Prepare validation set and callbacks
        fit_params: dict[str, Any] = {}
        callbacks = []
        
        if X_val is not None and y_val is not None:
            X_val, y_val = self._to_numpy(X_val, y_val)
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["eval_names"] = ["valid"]
            
            # Add early stopping callback
            early_stopping_rounds = self.metadata.hyperparameters.get("early_stopping_rounds", 50)
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
            callbacks.append(lgb.log_evaluation(period=100))
        
        if callbacks:
            fit_params["callbacks"] = callbacks
        
        # Train
        logger.info(f"Training LightGBM: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        self._model.fit(X, y, **fit_params)
        
        # Update metadata
        self._is_fitted = True
        self.metadata.trained_at = __import__("datetime").datetime.now()
        self.metadata.n_samples = X.shape[0]
        self.metadata.n_features = X.shape[1]
        
        # Log best iteration
        if hasattr(self._model, "best_iteration_"):
            logger.info(f"Best iteration: {self._model.best_iteration_}")
        
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
    
    def get_feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'gain' (default), 'split', or 'shap'
        """
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
        
        # Save LightGBM model
        model_path = path / "model.txt"
        self._model.booster_.save_model(str(model_path))
        
        # Save metadata
        meta_path = path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save additional config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "use_gpu": self.use_gpu,
                "class_weight": self._class_weight,
            }, f, indent=2)
        
        logger.info(f"Saved LightGBM model to {path}")
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
        instance._class_weight = config.get("class_weight")
        
        # Load LightGBM model
        model_path = path / "model.txt"
        booster = lgb.Booster(model_file=str(model_path))
        instance._model = lgb.LGBMClassifier()
        instance._model._Booster = booster
        instance._model._n_classes = 2
        instance._model._n_features = booster.num_feature()
        instance._is_fitted = True
        
        logger.info(f"Loaded LightGBM model from {path}")
        return instance
    
    def get_booster(self) -> lgb.Booster:
        """Get underlying LightGBM booster."""
        self._validate_fitted()
        return self._model.booster_
