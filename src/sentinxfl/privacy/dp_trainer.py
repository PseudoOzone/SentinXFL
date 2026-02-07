"""
SentinXFL - DP-SGD Trainer
===========================

Differentially Private Stochastic Gradient Descent trainer.
Provides privacy-preserving model training with formal guarantees.

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.privacy.accountant import RDPAccountant, get_accountant
from sentinxfl.privacy.mechanisms import GaussianMechanism, GradientClipper

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class DPSGDConfig:
    """Configuration for DP-SGD training."""
    
    # Privacy parameters
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0  # Gradient clipping bound
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.01
    epochs: int = 10
    
    # Noise computation (auto-computed if None)
    noise_multiplier: float | None = None
    
    # Flags
    strict_budget: bool = True  # Stop if budget exhausted
    verbose: bool = True


@dataclass
class DPTrainingResult:
    """Results from DP training run."""
    
    epochs_completed: int
    final_epsilon: float
    delta: float
    noise_multiplier: float
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    privacy_per_epoch: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "epochs_completed": self.epochs_completed,
            "final_epsilon": self.final_epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "training_loss": self.training_loss,
            "privacy_per_epoch": self.privacy_per_epoch,
        }


class DPSGDTrainer:
    """
    Differentially Private SGD trainer.
    
    Implements the DP-SGD algorithm from Abadi et al. 2016:
    1. Per-sample gradient computation
    2. Gradient clipping
    3. Gaussian noise addition
    4. Privacy accounting via RDP
    """
    
    def __init__(
        self,
        config: DPSGDConfig | None = None,
        accountant: RDPAccountant | None = None,
    ):
        """
        Initialize DP-SGD trainer.
        
        Args:
            config: Training configuration
            accountant: Privacy accountant (uses global if None)
        """
        self.config = config or DPSGDConfig()
        self.accountant = accountant or get_accountant()
        
        # Components
        self.clipper = GradientClipper(max_norm=self.config.max_grad_norm)
        
        # Auto-compute noise multiplier if not specified
        if self.config.noise_multiplier is None:
            self._auto_compute_noise()
        
        self.mechanism = GaussianMechanism(
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            sensitivity=self.config.max_grad_norm,
        )
        
        logger.info(
            f"DP-SGD Trainer: ε={self.config.epsilon}, δ={self.config.delta}, "
            f"σ={self.config.noise_multiplier:.2f}, C={self.config.max_grad_norm}"
        )
    
    def _auto_compute_noise(self) -> None:
        """Auto-compute noise multiplier for target privacy."""
        # Estimate number of steps
        # Assume dataset size for now (will be updated with actual data)
        estimated_dataset_size = 10000
        sampling_rate = self.config.batch_size / estimated_dataset_size
        steps_per_epoch = int(1 / sampling_rate)
        total_steps = steps_per_epoch * self.config.epochs
        
        # Find noise multiplier
        self.config.noise_multiplier = self.accountant.get_noise_multiplier(
            target_epsilon=self.config.epsilon,
            sampling_rate=sampling_rate,
            n_steps=total_steps,
        )
    
    def private_gradient(
        self,
        per_sample_gradients: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Compute private gradient from per-sample gradients.
        
        Steps:
        1. Clip each per-sample gradient to bounded L2 norm
        2. Sum clipped gradients
        3. Add Gaussian noise calibrated to sensitivity
        4. Divide by batch size for average
        
        Args:
            per_sample_gradients: Shape (batch_size, n_params)
            
        Returns:
            Private gradient estimate
        """
        batch_size = len(per_sample_gradients)
        
        # 1. Clip per-sample gradients
        clipped = self.clipper.clip_batch(per_sample_gradients)
        
        # 2. Sum
        summed = clipped.sum(axis=0)
        
        # 3. Add noise (scaled for batch)
        # Noise std = sigma * C (where C is clip norm, sigma is noise multiplier)
        noise_std = self.config.noise_multiplier * self.config.max_grad_norm
        noise = np.random.normal(0, noise_std, summed.shape).astype(np.float32)
        noisy_sum = summed + noise
        
        # 4. Average
        private_grad = noisy_sum / batch_size
        
        return private_grad
    
    def train_step_numpy(
        self,
        weights: NDArray[np.float32],
        X_batch: NDArray[np.float32],
        y_batch: NDArray[np.int32],
        loss_fn: Callable,
        grad_fn: Callable,
    ) -> tuple[NDArray[np.float32], float]:
        """
        Single DP-SGD training step using numpy operations.
        
        This is a simplified version for demonstration.
        Real usage should integrate with PyTorch/TensorFlow.
        
        Args:
            weights: Model weights
            X_batch: Batch features
            y_batch: Batch labels
            loss_fn: Loss function(weights, X, y) -> loss
            grad_fn: Gradient function(weights, x, y) -> grad (per-sample)
            
        Returns:
            Tuple of (updated_weights, batch_loss)
        """
        batch_size = len(X_batch)
        
        # Compute per-sample gradients
        per_sample_grads = np.array([
            grad_fn(weights, X_batch[i:i+1], y_batch[i:i+1])
            for i in range(batch_size)
        ])
        
        # Privatize gradient
        private_grad = self.private_gradient(per_sample_grads)
        
        # Update weights
        updated_weights = weights - self.config.learning_rate * private_grad
        
        # Compute loss (for monitoring)
        loss = loss_fn(weights, X_batch, y_batch)
        
        return updated_weights, loss
    
    def train_epoch(
        self,
        weights: NDArray[np.float32],
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        loss_fn: Callable,
        grad_fn: Callable,
    ) -> tuple[NDArray[np.float32], float]:
        """
        Train for one epoch with DP-SGD.
        
        Returns:
            Tuple of (updated_weights, average_epoch_loss)
        """
        n_samples = len(X)
        n_batches = n_samples // self.config.batch_size
        
        # Track sampling rate for privacy accounting
        sampling_rate = self.config.batch_size / n_samples
        
        # Shuffle data
        idx = np.random.permutation(n_samples)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        
        epoch_losses = []
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.config.batch_size
            end = start + self.config.batch_size
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            weights, batch_loss = self.train_step_numpy(
                weights, X_batch, y_batch, loss_fn, grad_fn
            )
            
            epoch_losses.append(batch_loss)
        
        # Account for privacy spent in this epoch
        self.accountant.accumulate_subsampled_gaussian(
            sigma=self.config.noise_multiplier,
            sampling_rate=sampling_rate,
            n_steps=n_batches,
            sensitivity=self.config.max_grad_norm,
            operation=f"dp_sgd_epoch",
        )
        
        return weights, float(np.mean(epoch_losses))
    
    def check_budget(self) -> bool:
        """Check if privacy budget is exhausted."""
        if self.config.strict_budget and self.accountant.budget_exhausted:
            logger.warning("Privacy budget exhausted!")
            return False
        return True


class DPGradientBoostTrainer:
    """
    DP training wrapper for gradient boosting models.
    
    Adds noise to histogram/leaf outputs instead of gradients.
    This is a simpler approach suitable for tree ensembles.
    """
    
    def __init__(
        self,
        epsilon: float | None = None,
        delta: float | None = None,
        sensitivity: float = 1.0,
    ):
        """
        Initialize DP trainer for gradient boosting.
        
        Args:
            epsilon: Privacy parameter
            delta: Failure probability
            sensitivity: Sensitivity of leaf outputs
        """
        self.epsilon = epsilon or settings.dp_epsilon
        self.delta = delta or settings.dp_delta
        self.sensitivity = sensitivity
        
        self.mechanism = GaussianMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity,
        )
        
        self.accountant = get_accountant()
    
    def privatize_leaf_values(
        self,
        leaf_values: NDArray[np.float32],
        n_trees: int,
    ) -> NDArray[np.float32]:
        """
        Add noise to leaf values for privacy.
        
        Splits privacy budget across trees.
        
        Args:
            leaf_values: Leaf value predictions
            n_trees: Number of trees (for budget splitting)
            
        Returns:
            Noisy leaf values
        """
        # Split epsilon across trees (basic composition)
        per_tree_epsilon = self.epsilon / np.sqrt(n_trees)
        
        tree_mechanism = GaussianMechanism(
            epsilon=per_tree_epsilon,
            delta=self.delta / n_trees,
            sensitivity=self.sensitivity,
        )
        
        noisy_values = tree_mechanism.add_noise(leaf_values)
        
        # Account for privacy
        self.accountant.accumulate_gaussian(
            sigma=tree_mechanism.sigma,
            sensitivity=self.sensitivity,
            operation="dp_leaf_noise",
            metadata={"n_trees": n_trees},
        )
        
        return noisy_values
    
    def privatize_histogram(
        self,
        histogram: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Add noise to histogram for private split finding.
        
        Args:
            histogram: Gradient histogram
            
        Returns:
            Noisy histogram
        """
        return self.mechanism.add_noise(histogram)


# ==========================================
# Utility Functions
# ==========================================

def compute_dp_params(
    target_epsilon: float,
    target_delta: float,
    dataset_size: int,
    batch_size: int,
    epochs: int,
) -> dict[str, float]:
    """
    Compute DP-SGD parameters for target privacy.
    
    Args:
        target_epsilon: Target privacy budget
        target_delta: Target failure probability
        dataset_size: Number of training samples
        batch_size: Batch size
        epochs: Number of training epochs
        
    Returns:
        Dictionary with computed parameters
    """
    sampling_rate = batch_size / dataset_size
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Create temporary accountant
    temp_accountant = RDPAccountant(
        epsilon_budget=target_epsilon,
        delta=target_delta,
    )
    
    noise_multiplier = temp_accountant.get_noise_multiplier(
        target_epsilon=target_epsilon,
        sampling_rate=sampling_rate,
        n_steps=total_steps,
    )
    
    return {
        "noise_multiplier": noise_multiplier,
        "sampling_rate": sampling_rate,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "effective_sigma": noise_multiplier,
    }
