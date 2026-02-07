"""
SentinXFL - Differential Privacy Mechanisms
=============================================

Core DP mechanisms for adding calibrated noise to preserve privacy.
Supports Gaussian (smooth sensitivity) and Laplace (L1 sensitivity).

Author: Anshuman Bakshi
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    
    mechanism: Literal["gaussian", "laplace"] = "gaussian"
    epsilon: float = 1.0  # Privacy parameter
    delta: float = 1e-5  # Failure probability (for Gaussian)
    sensitivity: float = 1.0  # Query sensitivity (L1 or L2)
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta < 0 or self.delta >= 1:
            raise ValueError("Delta must be in [0, 1)")
        if self.sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")


class GaussianMechanism:
    """
    Gaussian mechanism for (ε, δ)-differential privacy.
    
    Adds Gaussian noise calibrated to the L2 sensitivity.
    Provides better utility than Laplace for high-dimensional queries.
    
    Noise scale: σ = sensitivity * √(2 * ln(1.25/δ)) / ε
    """
    
    def __init__(
        self,
        epsilon: float | None = None,
        delta: float | None = None,
        sensitivity: float = 1.0,
    ):
        """
        Initialize Gaussian mechanism.
        
        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Failure probability (smaller = more private)
            sensitivity: L2 sensitivity of the query
        """
        self.epsilon = epsilon or settings.dp_epsilon
        self.delta = delta or settings.dp_delta
        self.sensitivity = sensitivity
        
        # Compute noise scale
        self.sigma = self._compute_sigma()
        
        logger.debug(
            f"GaussianMechanism: ε={self.epsilon}, δ={self.delta}, "
            f"Δ={self.sensitivity}, σ={self.sigma:.4f}"
        )
    
    def _compute_sigma(self) -> float:
        """
        Compute Gaussian noise standard deviation.
        
        Using the standard formula: σ = Δ * √(2 * ln(1.25/δ)) / ε
        """
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise(self, value: float | NDArray[np.float32]) -> float | NDArray[np.float32]:
        """
        Add calibrated Gaussian noise to a value or array.
        
        Args:
            value: Original value(s)
            
        Returns:
            Noisy value(s)
        """
        if isinstance(value, (int, float)):
            noise = np.random.normal(0, self.sigma)
            return float(value + noise)
        
        noise = np.random.normal(0, self.sigma, value.shape)
        return (value + noise).astype(np.float32)
    
    def __call__(self, value: float | NDArray[np.float32]) -> float | NDArray[np.float32]:
        """Alias for add_noise."""
        return self.add_noise(value)


class LaplaceMechanism:
    """
    Laplace mechanism for ε-differential privacy.
    
    Adds Laplace noise calibrated to the L1 sensitivity.
    Classic mechanism, provides pure DP (no δ).
    
    Noise scale: b = sensitivity / ε
    """
    
    def __init__(
        self,
        epsilon: float | None = None,
        sensitivity: float = 1.0,
    ):
        """
        Initialize Laplace mechanism.
        
        Args:
            epsilon: Privacy parameter (smaller = more private)
            sensitivity: L1 sensitivity of the query
        """
        self.epsilon = epsilon or settings.dp_epsilon
        self.sensitivity = sensitivity
        
        # Compute scale parameter
        self.scale = self.sensitivity / self.epsilon
        
        logger.debug(
            f"LaplaceMechanism: ε={self.epsilon}, Δ={self.sensitivity}, b={self.scale:.4f}"
        )
    
    def add_noise(self, value: float | NDArray[np.float32]) -> float | NDArray[np.float32]:
        """
        Add calibrated Laplace noise to a value or array.
        
        Args:
            value: Original value(s)
            
        Returns:
            Noisy value(s)
        """
        if isinstance(value, (int, float)):
            noise = np.random.laplace(0, self.scale)
            return float(value + noise)
        
        noise = np.random.laplace(0, self.scale, value.shape)
        return (value + noise).astype(np.float32)
    
    def __call__(self, value: float | NDArray[np.float32]) -> float | NDArray[np.float32]:
        """Alias for add_noise."""
        return self.add_noise(value)


class ExponentialMechanism:
    """
    Exponential mechanism for selecting from discrete options.
    
    Useful for DP model selection, hyperparameter tuning.
    Probability of selecting option r: P(r) ∝ exp(ε * u(r) / (2 * Δu))
    """
    
    def __init__(
        self,
        epsilon: float | None = None,
        sensitivity: float = 1.0,
    ):
        """
        Initialize Exponential mechanism.
        
        Args:
            epsilon: Privacy parameter
            sensitivity: Sensitivity of utility function
        """
        self.epsilon = epsilon or settings.dp_epsilon
        self.sensitivity = sensitivity
    
    def select(
        self,
        options: list,
        utilities: NDArray[np.float64],
    ) -> tuple[int, any]:
        """
        Select an option with probability proportional to exp(utility).
        
        Args:
            options: List of options to choose from
            utilities: Utility score for each option
            
        Returns:
            Tuple of (selected_index, selected_option)
        """
        # Compute selection probabilities
        scaled_utilities = (self.epsilon * utilities) / (2 * self.sensitivity)
        
        # Numerical stability: subtract max before exp
        scaled_utilities = scaled_utilities - scaled_utilities.max()
        probs = np.exp(scaled_utilities)
        probs = probs / probs.sum()
        
        # Sample
        idx = np.random.choice(len(options), p=probs)
        
        return idx, options[idx]


class GradientClipper:
    """
    Clips gradients to bound sensitivity for DP-SGD.
    
    Essential for DP training: clips per-sample gradients
    to have bounded L2 norm before aggregation.
    """
    
    def __init__(self, max_norm: float | None = None):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum L2 norm for gradients
        """
        self.max_norm = max_norm or settings.dp_max_grad_norm
    
    def clip(self, gradients: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Clip gradients to have bounded L2 norm.
        
        Args:
            gradients: Gradient array (can be flattened or per-sample)
            
        Returns:
            Clipped gradients
        """
        # Compute L2 norm
        norm = np.linalg.norm(gradients)
        
        if norm > self.max_norm:
            # Scale down
            gradients = gradients * (self.max_norm / norm)
            logger.debug(f"Clipped gradient from norm {norm:.4f} to {self.max_norm}")
        
        return gradients
    
    def clip_batch(
        self,
        per_sample_gradients: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Clip per-sample gradients independently.
        
        Args:
            per_sample_gradients: Shape (batch_size, n_params)
            
        Returns:
            Clipped gradients, same shape
        """
        # Compute per-sample norms
        norms = np.linalg.norm(per_sample_gradients, axis=1, keepdims=True)
        
        # Clip factor: min(1, max_norm / norm)
        clip_factor = np.minimum(1.0, self.max_norm / (norms + 1e-8))
        
        return (per_sample_gradients * clip_factor).astype(np.float32)


def create_mechanism(
    mechanism_type: Literal["gaussian", "laplace", "exponential"] = "gaussian",
    **kwargs,
) -> GaussianMechanism | LaplaceMechanism | ExponentialMechanism:
    """
    Factory function for creating DP mechanisms.
    
    Args:
        mechanism_type: Type of mechanism
        **kwargs: Mechanism-specific arguments
        
    Returns:
        DP mechanism instance
    """
    if mechanism_type == "gaussian":
        return GaussianMechanism(**kwargs)
    elif mechanism_type == "laplace":
        return LaplaceMechanism(**kwargs)
    elif mechanism_type == "exponential":
        return ExponentialMechanism(**kwargs)
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")


# ==========================================
# Utility Functions
# ==========================================

def compute_sensitivity(
    query_type: Literal["sum", "count", "mean", "median", "histogram"],
    n_samples: int | None = None,
    data_range: tuple[float, float] | None = None,
) -> float:
    """
    Compute sensitivity for common query types.
    
    Args:
        query_type: Type of statistical query
        n_samples: Number of samples in dataset
        data_range: (min, max) range of data values
        
    Returns:
        L1 sensitivity (for Laplace) or L2 sensitivity (for Gaussian)
    """
    if query_type == "count":
        # Adding/removing one person changes count by 1
        return 1.0
    
    elif query_type == "sum":
        # Sensitivity = range of possible values
        if data_range is None:
            raise ValueError("data_range required for sum sensitivity")
        return data_range[1] - data_range[0]
    
    elif query_type == "mean":
        # Sensitivity = range / n
        if data_range is None or n_samples is None:
            raise ValueError("data_range and n_samples required for mean sensitivity")
        return (data_range[1] - data_range[0]) / n_samples
    
    elif query_type == "histogram":
        # Adding/removing one changes one bin by 1
        return 1.0
    
    elif query_type == "median":
        # Median has unbounded sensitivity in general
        # Use smooth sensitivity or restrict to bounded data
        if data_range is None:
            raise ValueError("data_range required for bounded median")
        return data_range[1] - data_range[0]
    
    else:
        raise ValueError(f"Unknown query type: {query_type}")
