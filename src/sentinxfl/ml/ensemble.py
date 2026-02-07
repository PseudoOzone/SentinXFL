"""
SentinXFL - Ensemble Model
===========================

Weighted ensemble combining multiple fraud detection models.
Supports weighted averaging, stacking, and probability calibration.

Author: Anshuman Bakshi
"""

import json
from pathlib import Path
from typing import Any, Literal, Self

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.base import BaseModel, ModelMetadata, ModelType
from sentinxfl.ml.metrics import MetricsCalculator

logger = get_logger(__name__)
settings = get_settings()


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple fraud detectors.
    
    Ensemble strategies:
    - weighted_average: Simple weighted average of probabilities
    - stacking: Meta-learner on top of base predictions
    - voting: Majority voting with optional weights
    
    Calibration:
    - Platt scaling (logistic regression on scores)
    - Isotonic regression
    """
    
    def __init__(
        self,
        name: str = "ensemble_fraud",
        strategy: Literal["weighted_average", "stacking", "voting"] = "weighted_average",
        calibrate: bool = True,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        **hyperparameters: Any,
    ):
        """
        Initialize ensemble model.
        
        Args:
            name: Model name
            strategy: Ensemble combination strategy
            calibrate: Whether to calibrate final probabilities
            calibration_method: 'sigmoid' (Platt) or 'isotonic'
            **hyperparameters: Additional parameters
        """
        super().__init__(
            name=name,
            strategy=strategy,
            calibrate=calibrate,
            calibration_method=calibration_method,
            **hyperparameters,
        )
        
        self.strategy = strategy
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.metadata.model_type = ModelType.ENSEMBLE
        
        # Base models
        self._models: list[BaseModel] = []
        self._model_names: list[str] = []
        self._weights: NDArray[np.float32] | None = None
        
        # Stacking meta-learner
        self._meta_learner: LogisticRegression | None = None
        
        # Calibrator
        self._calibrator: Any | None = None
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> Self:
        """
        Add a base model to the ensemble.
        
        Args:
            model: Trained base model
            weight: Initial weight (will be optimized if strategy is weighted_average)
            
        Returns:
            Self for method chaining
        """
        if not model.is_fitted:
            raise ValueError(f"Model '{model.name}' must be fitted before adding to ensemble")
        
        self._models.append(model)
        self._model_names.append(model.name)
        
        # Initialize or update weights
        if self._weights is None:
            self._weights = np.array([weight], dtype=np.float32)
        else:
            self._weights = np.append(self._weights, weight)
        
        logger.info(f"Added model '{model.name}' to ensemble (weight={weight})")
        return self
    
    def _create_model(self) -> None:
        """Not used - ensemble wraps existing models."""
        pass
    
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        optimize_weights: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Fit ensemble (optimize weights and/or train meta-learner).
        
        Note: Base models must already be fitted before calling this.
        
        Args:
            X: Features for weight optimization
            y: Labels
            X_val: Validation features for calibration
            y_val: Validation labels
            optimize_weights: Whether to optimize weights (for weighted_average)
            **kwargs: Additional arguments
        """
        if len(self._models) == 0:
            raise ValueError("No models in ensemble. Use add_model() first.")
        
        X, y = self._to_numpy(X, y)
        
        logger.info(f"Fitting ensemble with {len(self._models)} models, strategy={self.strategy}")
        
        # Get base model predictions
        base_proba = self._get_base_predictions(X)  # Shape: (n_samples, n_models)
        
        if self.strategy == "weighted_average":
            if optimize_weights:
                self._optimize_weights(base_proba, y)
        
        elif self.strategy == "stacking":
            self._fit_stacking(base_proba, y)
        
        # Calibrate if requested
        if self.calibrate and X_val is not None and y_val is not None:
            self._fit_calibrator(X_val, y_val)
        
        # Update metadata
        self._is_fitted = True
        self.metadata.trained_at = __import__("datetime").datetime.now()
        self.metadata.n_samples = X.shape[0]
        self.metadata.n_features = X.shape[1]
        self.metadata.hyperparameters["model_names"] = self._model_names
        self.metadata.hyperparameters["weights"] = self._weights.tolist() if self._weights is not None else None
        
        # Log weights
        logger.info(f"Ensemble weights: {dict(zip(self._model_names, self._weights.tolist()))}")
        
        return self
    
    def _get_base_predictions(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get fraud probability predictions from all base models."""
        predictions = []
        
        for model in self._models:
            proba = model.predict_fraud_probability(X)
            predictions.append(proba)
        
        return np.column_stack(predictions).astype(np.float32)
    
    def _optimize_weights(
        self,
        base_proba: NDArray[np.float32],
        y: NDArray[np.int32],
        metric: str = "auc_pr",
    ) -> None:
        """
        Optimize ensemble weights using scipy.minimize.
        
        Maximizes the specified metric (default: AUPRC).
        """
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        n_models = base_proba.shape[1]
        
        def objective(weights: NDArray[np.float32]) -> float:
            # Normalize weights to sum to 1
            weights = np.abs(weights) / np.abs(weights).sum()
            
            # Combine predictions
            combined = (base_proba * weights).sum(axis=1)
            
            # Calculate metric (negative because minimize)
            if metric == "auc_pr":
                return -average_precision_score(y, combined)
            else:
                return -roc_auc_score(y, combined)
        
        # Initial weights (equal)
        x0 = np.ones(n_models) / n_models
        
        # Bounds (non-negative)
        bounds = [(0.01, 1.0)] * n_models
        
        # Constraint: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )
        
        # Normalize final weights
        self._weights = (np.abs(result.x) / np.abs(result.x).sum()).astype(np.float32)
        
        logger.info(f"Optimized weights (by {metric}): {result.fun:.4f}")
    
    def _fit_stacking(
        self,
        base_proba: NDArray[np.float32],
        y: NDArray[np.int32],
    ) -> None:
        """Fit stacking meta-learner on base predictions."""
        self._meta_learner = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        
        self._meta_learner.fit(base_proba, y)
        
        # Use meta-learner coefficients as weights for interpretability
        coef = np.abs(self._meta_learner.coef_[0])
        self._weights = (coef / coef.sum()).astype(np.float32)
        
        logger.info("Fitted stacking meta-learner")
    
    def _fit_calibrator(
        self,
        X_val: NDArray[np.float32],
        y_val: NDArray[np.int32],
    ) -> None:
        """Fit probability calibrator on validation data."""
        X_val, y_val = self._to_numpy(X_val, y_val)
        
        # Get raw ensemble predictions
        raw_proba = self._predict_raw(X_val)
        
        # Fit calibration using Platt scaling
        if self.calibration_method == "sigmoid":
            # Logistic regression on raw probabilities
            self._calibrator = LogisticRegression(max_iter=1000)
            self._calibrator.fit(raw_proba.reshape(-1, 1), y_val)
        else:
            # Isotonic regression (non-parametric)
            from sklearn.isotonic import IsotonicRegression
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_proba, y_val)
        
        logger.info(f"Fitted {self.calibration_method} calibrator")
    
    def _predict_raw(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get raw (uncalibrated) fraud probabilities."""
        base_proba = self._get_base_predictions(X)
        
        if self.strategy == "stacking" and self._meta_learner is not None:
            return self._meta_learner.predict_proba(base_proba)[:, 1].astype(np.float32)
        
        elif self.strategy == "voting":
            # Majority voting
            threshold = 0.5
            votes = (base_proba > threshold).astype(np.float32)
            if self._weights is not None:
                return (votes * self._weights).sum(axis=1) / self._weights.sum()
            return votes.mean(axis=1).astype(np.float32)
        
        else:  # weighted_average
            if self._weights is not None:
                return (base_proba * self._weights).sum(axis=1).astype(np.float32)
            return base_proba.mean(axis=1).astype(np.float32)
    
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict class labels."""
        self._validate_fitted()
        proba = self.predict_fraud_probability(X)
        
        # Use optimal threshold if available
        threshold = self.metadata.hyperparameters.get("optimal_threshold", 0.5)
        return (proba >= threshold).astype(np.int32)
    
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities."""
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        
        fraud_proba = self.predict_fraud_probability(X)
        normal_proba = 1 - fraud_proba
        
        return np.column_stack([normal_proba, fraud_proba]).astype(np.float32)
    
    def predict_fraud_probability(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get calibrated fraud probabilities."""
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        
        raw_proba = self._predict_raw(X)
        
        # Apply calibration if available
        if self._calibrator is not None:
            if self.calibration_method == "sigmoid":
                calibrated = self._calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
            else:
                calibrated = self._calibrator.predict(raw_proba)
            return calibrated.astype(np.float32)
        
        return raw_proba
    
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get aggregated feature importance across all base models.
        
        Weighted by model weights.
        """
        self._validate_fitted()
        
        aggregated: dict[str, float] = {}
        
        for i, model in enumerate(self._models):
            try:
                importance = model.get_feature_importance()
                weight = self._weights[i] if self._weights is not None else 1.0
                
                for feature, score in importance.items():
                    if feature not in aggregated:
                        aggregated[feature] = 0.0
                    aggregated[feature] += score * weight
            except (NotImplementedError, RuntimeError):
                continue
        
        # Normalize
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}
        
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_contributions(self, X: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
        """
        Get per-sample contribution from each base model.
        
        Useful for understanding which model contributed most to each prediction.
        """
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        
        contributions = {}
        base_proba = self._get_base_predictions(X)
        
        for i, name in enumerate(self._model_names):
            weight = self._weights[i] if self._weights is not None else 1.0
            contributions[name] = (base_proba[:, i] * weight).astype(np.float32)
        
        return contributions
    
    def save(self, path: Path | str) -> Path:
        """Save ensemble to disk (metadata only, base models saved separately)."""
        self._validate_fitted()
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta_path = path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "strategy": self.strategy,
                "calibrate": self.calibrate,
                "calibration_method": self.calibration_method,
                "model_names": self._model_names,
                "weights": self._weights.tolist() if self._weights is not None else None,
            }, f, indent=2)
        
        # Save meta-learner if exists
        if self._meta_learner is not None:
            meta_learner_path = path / "meta_learner.joblib"
            joblib.dump(self._meta_learner, meta_learner_path)
        
        # Save calibrator if exists
        if self._calibrator is not None:
            calibrator_path = path / "calibrator.joblib"
            joblib.dump(self._calibrator, calibrator_path)
        
        logger.info(f"Saved ensemble config to {path}")
        logger.warning("Base models must be saved separately and loaded before ensemble.load()")
        
        return path
    
    @classmethod
    def load(
        cls,
        path: Path | str,
        models: list[BaseModel] | None = None,
    ) -> Self:
        """
        Load ensemble from disk.
        
        Args:
            path: Path to saved ensemble
            models: Pre-loaded base models (required in same order as saved)
        """
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
            strategy=config.get("strategy", "weighted_average"),
            calibrate=config.get("calibrate", True),
            calibration_method=config.get("calibration_method", "sigmoid"),
        )
        instance.metadata = ModelMetadata.from_dict(meta_dict)
        instance._model_names = config.get("model_names", [])
        instance._weights = np.array(config.get("weights", []), dtype=np.float32) if config.get("weights") else None
        
        # Load base models
        if models is not None:
            instance._models = models
        else:
            logger.warning("No base models provided - ensemble predictions will fail")
        
        # Load meta-learner
        meta_learner_path = path / "meta_learner.joblib"
        if meta_learner_path.exists():
            instance._meta_learner = joblib.load(meta_learner_path)
        
        # Load calibrator
        calibrator_path = path / "calibrator.joblib"
        if calibrator_path.exists():
            instance._calibrator = joblib.load(calibrator_path)
        
        instance._is_fitted = True
        
        logger.info(f"Loaded ensemble from {path}")
        return instance
