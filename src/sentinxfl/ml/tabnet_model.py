"""
SentinXFL - TabNet Model
=========================

TabNet neural network for tabular data with attention mechanisms.
Provides interpretable feature importance through attention masks.

VRAM Optimized for RTX 3050 4GB (~1GB target usage).

Author: Anshuman Bakshi
"""

import json
import gc
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.base import BaseModel, ModelMetadata, ModelType

logger = get_logger(__name__)
settings = get_settings()


class TabNetModel(BaseModel):
    """
    TabNet model for interpretable fraud detection.
    
    Key features:
    - Attention-based feature selection (interpretability)
    - Sparsemax for sparse feature masks
    - Built-in feature importance
    - Supports GPU with VRAM management
    
    VRAM Budget: ~1GB (optimized for RTX 3050 4GB shared with LLM)
    """
    
    # Default hyperparameters OPTIMIZED FOR 4GB VRAM
    DEFAULT_PARAMS = {
        # Architecture (reduced for VRAM)
        "n_d": 16,  # Width of decision prediction layer (default: 8, we use 16 for better accuracy)
        "n_a": 16,  # Width of attention embedding layer
        "n_steps": 4,  # Number of decision steps (default: 3)
        "gamma": 1.5,  # Coefficient for feature reusage
        "n_independent": 2,  # Number of independent GLU layers
        "n_shared": 2,  # Number of shared GLU layers
        
        # Training
        "max_epochs": 100,
        "patience": 15,
        "batch_size": 1024,  # Reduced for VRAM
        "virtual_batch_size": 256,  # Ghost batch normalization size
        "momentum": 0.02,
        "lambda_sparse": 1e-3,  # Sparsity regularization
        
        # Optimizer
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": 2e-2},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "scheduler_params": {"step_size": 10, "gamma": 0.9},
        
        # Masking
        "mask_type": "sparsemax",  # sparsemax or entmax
        
        # Performance
        "seed": 42,
        "verbose": 1,
    }
    
    def __init__(
        self,
        name: str = "tabnet_fraud",
        device: str = "auto",
        **hyperparameters: Any,
    ):
        """
        Initialize TabNet model.
        
        Args:
            name: Model name
            device: 'auto', 'cuda', or 'cpu'
            **hyperparameters: Override default hyperparameters
        """
        # Merge with defaults
        params = self.DEFAULT_PARAMS.copy()
        params.update(hyperparameters)
        
        super().__init__(name=name, **params)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.metadata.model_type = ModelType.TABNET
        
        # Feature importance from attention
        self._attention_weights: NDArray[np.float32] | None = None
    
    def _create_model(self) -> Any:
        """Create TabNet classifier."""
        from pytorch_tabnet.tab_model import TabNetClassifier
        
        params = self.metadata.hyperparameters.copy()
        
        # Extract non-TabNet params
        max_epochs = params.pop("max_epochs", 100)
        patience = params.pop("patience", 15)
        batch_size = params.pop("batch_size", 1024)
        virtual_batch_size = params.pop("virtual_batch_size", 256)
        
        # Create model
        model = TabNetClassifier(
            n_d=params.get("n_d", 16),
            n_a=params.get("n_a", 16),
            n_steps=params.get("n_steps", 4),
            gamma=params.get("gamma", 1.5),
            n_independent=params.get("n_independent", 2),
            n_shared=params.get("n_shared", 2),
            lambda_sparse=params.get("lambda_sparse", 1e-3),
            momentum=params.get("momentum", 0.02),
            mask_type=params.get("mask_type", "sparsemax"),
            optimizer_fn=params.get("optimizer_fn", torch.optim.Adam),
            optimizer_params=params.get("optimizer_params", {"lr": 2e-2}),
            scheduler_fn=params.get("scheduler_fn"),
            scheduler_params=params.get("scheduler_params"),
            seed=params.get("seed", 42),
            verbose=params.get("verbose", 1),
            device_name=self.device,
        )
        
        # Store training params
        self._max_epochs = max_epochs
        self._patience = patience
        self._batch_size = batch_size
        self._virtual_batch_size = virtual_batch_size
        
        return model
    
    def _clear_gpu_memory(self) -> None:
        """Clear CUDA cache to free VRAM."""
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
    
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Train TabNet model.
        
        Args:
            X: Training features
            y: Training labels (0 = legitimate, 1 = fraud)
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional fit arguments
        """
        # Clear GPU memory before training
        self._clear_gpu_memory()
        
        X, y = self._to_numpy(X, y)
        
        # Create model
        self._model = self._create_model()
        
        # Log VRAM usage
        if self.device == "cuda":
            vram_before = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"VRAM before training: {vram_before:.2f} GB")
        
        logger.info(f"Training TabNet: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        # Prepare validation
        eval_set = None
        eval_name = None
        if X_val is not None and y_val is not None:
            X_val, y_val = self._to_numpy(X_val, y_val)
            eval_set = [(X_val, y_val)]
            eval_name = ["valid"]
        
        # Compute class weights for imbalanced data
        n_classes = len(np.unique(y))
        n_samples = len(y)
        class_counts = np.bincount(y)
        weights = n_samples / (n_classes * class_counts)
        
        # Train
        try:
            self._model.fit(
                X_train=X,
                y_train=y,
                eval_set=eval_set,
                eval_name=eval_name,
                max_epochs=self._max_epochs,
                patience=self._patience,
                batch_size=self._batch_size,
                virtual_batch_size=self._virtual_batch_size,
                weights=1,  # 1 = automatic class balancing
                drop_last=False,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM! Reduce batch_size or n_d/n_a parameters")
                self._clear_gpu_memory()
                raise
            raise
        
        # Log VRAM usage after training
        if self.device == "cuda":
            vram_after = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"VRAM after training: {vram_after:.2f} GB")
        
        # Store attention weights for interpretability
        self._store_attention_weights(X)
        
        # Update metadata
        self._is_fitted = True
        self.metadata.trained_at = __import__("datetime").datetime.now()
        self.metadata.n_samples = X.shape[0]
        self.metadata.n_features = X.shape[1]
        
        return self
    
    def _store_attention_weights(self, X: NDArray[np.float32]) -> None:
        """Store feature attention weights for interpretability."""
        # Get attention masks from TabNet
        # Uses a sample of data to compute average attention
        sample_size = min(10000, len(X))
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        
        # TabNet returns explain tuple (masks, bias)
        explain_tuple = self._model.explain(X_sample)
        masks = explain_tuple[0]  # Shape: (n_samples, n_features)
        
        # Average across samples
        self._attention_weights = masks.mean(axis=0).astype(np.float32)
        
        logger.debug(f"Stored attention weights: {self._attention_weights.shape}")
    
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
        """
        Get feature importance from attention weights.
        
        TabNet's attention mechanism provides interpretable
        feature importance scores.
        """
        self._validate_fitted()
        
        if self._attention_weights is None:
            raise RuntimeError("Attention weights not computed. Fit model first.")
        
        feature_names = self.metadata.feature_names or [
            f"f{i}" for i in range(len(self._attention_weights))
        ]
        
        # Normalize to sum to 1
        importance = self._attention_weights / self._attention_weights.sum()
        
        return dict(sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ))
    
    def explain(self, X: NDArray[np.float32]) -> tuple[NDArray[np.float32], float]:
        """
        Get per-sample feature explanations.
        
        Args:
            X: Features to explain
            
        Returns:
            Tuple of (attention_masks, aggregate_bias)
            - attention_masks: Shape (n_samples, n_features)
            - aggregate_bias: Scalar bias term
        """
        self._validate_fitted()
        X, _ = self._to_numpy(X)
        
        masks, bias = self._model.explain(X)
        return masks.astype(np.float32), float(bias)
    
    def save(self, path: Path | str) -> Path:
        """Save model to disk."""
        self._validate_fitted()
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save TabNet model (pytorch)
        model_path = path / "model.pt"
        self._model.save_model(str(model_path))
        
        # Save metadata
        meta_path = path / "metadata.json"
        # Remove non-serializable params
        meta_dict = self.metadata.to_dict()
        hp = meta_dict.get("hyperparameters", {})
        hp.pop("optimizer_fn", None)
        hp.pop("scheduler_fn", None)
        
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2)
        
        # Save attention weights
        if self._attention_weights is not None:
            weights_path = path / "attention_weights.npy"
            np.save(weights_path, self._attention_weights)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "device": self.device,
                "max_epochs": self._max_epochs,
                "patience": self._patience,
                "batch_size": self._batch_size,
                "virtual_batch_size": self._virtual_batch_size,
            }, f, indent=2)
        
        logger.info(f"Saved TabNet model to {path}")
        return path
    
    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load model from disk."""
        from pytorch_tabnet.tab_model import TabNetClassifier
        
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
            device=config.get("device", "auto"),
            **meta_dict.get("hyperparameters", {}),
        )
        instance.metadata = ModelMetadata.from_dict(meta_dict)
        instance._max_epochs = config.get("max_epochs", 100)
        instance._patience = config.get("patience", 15)
        instance._batch_size = config.get("batch_size", 1024)
        instance._virtual_batch_size = config.get("virtual_batch_size", 256)
        
        # Load TabNet model
        model_path = path / "model.pt"
        instance._model = TabNetClassifier()
        instance._model.load_model(str(model_path))
        instance._is_fitted = True
        
        # Load attention weights
        weights_path = path / "attention_weights.npy"
        if weights_path.exists():
            instance._attention_weights = np.load(weights_path)
        
        logger.info(f"Loaded TabNet model from {path}")
        return instance
    
    # ==========================================
    # Federated Learning Support
    # ==========================================
    
    def get_weights(self) -> dict[str, NDArray[np.float32]]:
        """Get model weights for federated averaging."""
        self._validate_fitted()
        
        weights = {}
        for name, param in self._model.network.named_parameters():
            weights[name] = param.detach().cpu().numpy().astype(np.float32)
        
        return weights
    
    def set_weights(self, weights: dict[str, NDArray[np.float32]]) -> None:
        """Set model weights from federated aggregation."""
        self._validate_fitted()
        
        with torch.no_grad():
            for name, param in self._model.network.named_parameters():
                if name in weights:
                    param.copy_(torch.from_numpy(weights[name]).to(self.device))
