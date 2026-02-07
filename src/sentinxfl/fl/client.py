"""
SentinXFL - Federated Learning Client
======================================

Flower-based FL client with local training and DP integration.

Author: Anshuman Bakshi
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.privacy.accountant import RDPAccountant
from sentinxfl.privacy.mechanisms import GaussianMechanism, GradientClipper

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ClientConfig:
    """Configuration for FL client."""
    
    # Client identity
    client_id: str = "client_0"
    
    # Training settings
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Privacy (local DP)
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.1
    
    # Model type
    model_type: str = "xgboost"  # xgboost, lightgbm, tabnet


@dataclass
class LocalTrainingResult:
    """Result of local training round."""
    
    weights: list[NDArray[np.float32]]
    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    privacy_spent: float = 0.0


class FraudDetectionClient(fl.client.NumPyClient):
    """
    Flower client for federated fraud detection.
    
    Implements local training with optional differential privacy
    and reports model updates to the FL server.
    """
    
    def __init__(
        self,
        client_id: str,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        config: ClientConfig | None = None,
        model_factory: Callable | None = None,
    ):
        """
        Initialize FL client.
        
        Args:
            client_id: Unique client identifier
            X_train: Local training features
            y_train: Local training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            config: Client configuration
            model_factory: Function to create model instances
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.config = config or ClientConfig(client_id=client_id)
        self.model_factory = model_factory
        
        # Initialize model placeholder
        self.model = None
        
        # Privacy components
        if self.config.dp_enabled:
            self.clipper = GradientClipper(max_norm=self.config.dp_clip_norm)
            self.accountant = RDPAccountant(
                epsilon_budget=self.config.dp_epsilon,
                delta=self.config.dp_delta,
            )
            self.noise_mechanism = GaussianMechanism(
                epsilon=self.config.dp_epsilon,
                delta=self.config.dp_delta,
                sensitivity=self.config.dp_clip_norm,
            )
        else:
            self.clipper = None
            self.accountant = None
            self.noise_mechanism = None
        
        # Training history
        self.round_count = 0
        self.training_history: list[LocalTrainingResult] = []
        
        logger.info(
            f"Client {client_id}: {len(X_train)} samples, "
            f"DP={'ON' if self.config.dp_enabled else 'OFF'}"
        )
    
    def get_parameters(self, config: dict[str, Any]) -> list[NDArray]:
        """Return current model parameters."""
        if self.model is None:
            return []
        return self._get_model_weights()
    
    def set_parameters(self, parameters: list[NDArray]) -> None:
        """Set model parameters from server."""
        if parameters:
            self._set_model_weights(parameters)
    
    def fit(
        self,
        parameters: list[NDArray],
        config: dict[str, Any],
    ) -> tuple[list[NDArray], int, dict[str, float]]:
        """
        Train model locally and return updated weights.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_weights, num_samples, metrics)
        """
        self.round_count += 1
        
        # Set global model
        self.set_parameters(parameters)
        
        # Get training config from server or use local
        local_epochs = config.get("local_epochs", self.config.local_epochs)
        batch_size = config.get("batch_size", self.config.batch_size)
        
        # Train locally
        result = self._train_local(
            epochs=local_epochs,
            batch_size=batch_size,
        )
        
        # Apply local DP if enabled
        if self.config.dp_enabled:
            result.weights = self._apply_local_dp(result.weights)
        
        # Record history
        self.training_history.append(result)
        
        logger.debug(
            f"Client {self.client_id} round {self.round_count}: "
            f"loss={result.metrics.get('loss', 'N/A'):.4f}"
        )
        
        return result.weights, result.num_samples, result.metrics
    
    def evaluate(
        self,
        parameters: list[NDArray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """
        Evaluate model on local validation data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        if self.X_val is None or self.y_val is None:
            # Use training data if no validation set
            X_eval, y_eval = self.X_train, self.y_train
        else:
            X_eval, y_eval = self.X_val, self.y_val
        
        loss, metrics = self._evaluate_local(X_eval, y_eval)
        
        return loss, len(X_eval), metrics
    
    def _get_model_weights(self) -> list[NDArray[np.float32]]:
        """Extract weights from current model."""
        # Implementation depends on model type
        if self.model is None:
            return []
        
        # For sklearn-like models, try to get weights
        if hasattr(self.model, "get_weights"):
            return self.model.get_weights()
        
        # For XGBoost/LightGBM, extract booster weights
        if hasattr(self.model, "get_booster"):
            # Placeholder: actual implementation needs model-specific handling
            return self._extract_tree_weights()
        
        # Default: return empty
        logger.warning(f"Cannot extract weights from {type(self.model)}")
        return []
    
    def _set_model_weights(self, weights: list[NDArray[np.float32]]) -> None:
        """Set weights on current model."""
        if self.model is None and self.model_factory is not None:
            self.model = self.model_factory()
        
        if hasattr(self.model, "set_weights"):
            self.model.set_weights(weights)
    
    def _train_local(
        self,
        epochs: int,
        batch_size: int,
    ) -> LocalTrainingResult:
        """
        Perform local training.
        
        This is a placeholder - actual implementation depends on model type.
        """
        num_samples = len(self.X_train)
        
        # Initialize model if needed
        if self.model is None and self.model_factory is not None:
            self.model = self.model_factory()
        
        # Training logic depends on model type
        metrics = {}
        
        if self.model is not None:
            # Fit model
            if hasattr(self.model, "fit"):
                self.model.fit(self.X_train, self.y_train)
                
                # Get training metrics
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(self.X_train)
                    if proba.ndim == 2:
                        proba = proba[:, 1]
                    
                    # Log loss
                    from sklearn.metrics import log_loss, f1_score
                    metrics["loss"] = log_loss(self.y_train, proba)
                    
                    # F1
                    preds = (proba > 0.5).astype(int)
                    metrics["f1"] = f1_score(self.y_train, preds, zero_division=0)
        
        # Get updated weights
        weights = self._get_model_weights()
        
        return LocalTrainingResult(
            weights=weights,
            num_samples=num_samples,
            metrics=metrics,
        )
    
    def _evaluate_local(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
    ) -> tuple[float, dict[str, float]]:
        """Evaluate model on local data."""
        metrics = {}
        loss = 0.0
        
        if self.model is not None and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            from sklearn.metrics import log_loss, f1_score, roc_auc_score
            
            loss = log_loss(y, proba)
            metrics["loss"] = loss
            
            preds = (proba > 0.5).astype(int)
            metrics["f1"] = f1_score(y, preds, zero_division=0)
            
            if len(np.unique(y)) > 1:
                metrics["auc"] = roc_auc_score(y, proba)
        
        return loss, metrics
    
    def _apply_local_dp(
        self,
        weights: list[NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        """Apply local differential privacy to weights."""
        if not self.config.dp_enabled:
            return weights
        
        noisy_weights = []
        for w in weights:
            # Clip
            norm = np.linalg.norm(w)
            if norm > self.config.dp_clip_norm:
                w = w * (self.config.dp_clip_norm / norm)
            
            # Add noise
            noise = np.random.normal(
                0,
                self.config.dp_noise_multiplier * self.config.dp_clip_norm,
                w.shape,
            ).astype(np.float32)
            
            noisy_weights.append(w + noise)
        
        # Account for privacy spent
        if self.accountant:
            self.accountant.accumulate_gaussian(
                sigma=self.config.dp_noise_multiplier,
                sensitivity=self.config.dp_clip_norm,
                operation=f"local_dp_round_{self.round_count}",
            )
        
        return noisy_weights
    
    def _extract_tree_weights(self) -> list[NDArray[np.float32]]:
        """Extract weights from tree-based model."""
        # Placeholder for tree weight extraction
        # Real implementation would serialize tree structure
        return []


# ==========================================
# XGBoost-specific Client
# ==========================================

class XGBoostFLClient(FraudDetectionClient):
    """
    FL client specifically for XGBoost models.
    
    Uses XGBoost's native FL support or manual weight extraction.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xgb_model = None
    
    def _create_model(self):
        """Create XGBoost model."""
        try:
            import xgboost as xgb
            
            # Calculate class weight
            n_pos = np.sum(self.y_train == 1)
            n_neg = np.sum(self.y_train == 0)
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
            
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=self.config.learning_rate,
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                device="cuda" if settings.model_config.get("use_gpu", False) else "cpu",
                random_state=42,
            )
            
            return self.xgb_model
        except ImportError:
            logger.error("XGBoost not installed")
            return None
    
    def _train_local(self, epochs: int, batch_size: int) -> LocalTrainingResult:
        """Train XGBoost locally."""
        if self.xgb_model is None:
            self._create_model()
        
        if self.xgb_model is not None:
            self.xgb_model.fit(self.X_train, self.y_train)
            
            # Get predictions for metrics
            proba = self.xgb_model.predict_proba(self.X_train)[:, 1]
            
            from sklearn.metrics import log_loss, f1_score
            
            metrics = {
                "loss": log_loss(self.y_train, proba),
                "f1": f1_score(self.y_train, (proba > 0.5).astype(int), zero_division=0),
            }
        else:
            metrics = {}
        
        # For XGBoost, we export the model as bytes for aggregation
        # This is a simplified approach - real FL with trees is complex
        weights = self._get_xgb_weights()
        
        return LocalTrainingResult(
            weights=weights,
            num_samples=len(self.X_train),
            metrics=metrics,
        )
    
    def _get_xgb_weights(self) -> list[NDArray[np.float32]]:
        """Get XGBoost model as serializable weights."""
        if self.xgb_model is None:
            return []
        
        # Export model to JSON and return as bytes
        # This enables tree-based FL aggregation
        try:
            booster = self.xgb_model.get_booster()
            model_bytes = booster.save_raw()
            # Convert to float array for Flower compatibility
            return [np.frombuffer(model_bytes, dtype=np.uint8).astype(np.float32)]
        except Exception as e:
            logger.warning(f"Failed to extract XGBoost weights: {e}")
            return []


# ==========================================
# Client Factory
# ==========================================

def create_client(
    client_id: str,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int32],
    X_val: NDArray[np.float32] | None = None,
    y_val: NDArray[np.int32] | None = None,
    model_type: str = "xgboost",
    config: ClientConfig | None = None,
) -> FraudDetectionClient:
    """
    Create FL client based on model type.
    
    Args:
        client_id: Unique client identifier
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of model (xgboost, lightgbm, etc.)
        config: Client configuration
        
    Returns:
        Configured FL client
    """
    config = config or ClientConfig(client_id=client_id, model_type=model_type)
    
    if model_type.lower() == "xgboost":
        return XGBoostFLClient(
            client_id=client_id,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
        )
    
    # Default to generic client
    return FraudDetectionClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
    )


def start_client(
    server_address: str,
    client: FraudDetectionClient,
) -> None:
    """
    Start FL client and connect to server.
    
    Args:
        server_address: Server address (host:port)
        client: Configured FL client
    """
    logger.info(f"Starting client {client.client_id}, connecting to {server_address}")
    
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )
