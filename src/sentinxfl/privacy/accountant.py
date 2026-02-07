"""
SentinXFL - Privacy Accountant
===============================

Tracks cumulative privacy loss using Rényi Differential Privacy (RDP).
RDP provides tighter composition bounds than basic composition theorems.

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class PrivacySpent:
    """Record of privacy spent in a single operation."""
    
    operation: str
    epsilon: float
    delta: float
    rdp_alphas: list[float] = field(default_factory=list)
    rdp_epsilons: list[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class RDPAccountant:
    """
    Rényi Differential Privacy Accountant.
    
    Uses RDP composition which is tighter than basic composition.
    Tracks privacy loss as RDP at multiple orders α, then converts
    to (ε, δ)-DP when needed.
    
    RDP Definition: D_α(M(x) || M(x')) ≤ ρ
    where D_α is the Rényi divergence of order α.
    """
    
    # Standard α values for RDP accounting
    DEFAULT_ALPHAS = [
        1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10,
        12, 14, 16, 20, 24, 28, 32, 64, 256
    ]
    
    def __init__(
        self,
        epsilon_budget: float | None = None,
        delta: float | None = None,
        alphas: list[float] | None = None,
    ):
        """
        Initialize RDP accountant.
        
        Args:
            epsilon_budget: Total privacy budget (max ε)
            delta: Target δ for converting RDP to (ε,δ)-DP
            alphas: RDP orders to track
        """
        self.epsilon_budget = epsilon_budget or settings.dp_epsilon
        self.delta = delta or settings.dp_delta
        self.alphas = np.array(alphas or self.DEFAULT_ALPHAS)
        
        # Cumulative RDP at each alpha
        self._rdp_epsilons = np.zeros(len(self.alphas))
        
        # History of privacy-consuming operations
        self._history: list[PrivacySpent] = []
        
        logger.info(f"RDP Accountant initialized: budget ε={self.epsilon_budget}, δ={self.delta}")
    
    @property
    def spent_epsilon(self) -> float:
        """Get current spent epsilon (converted from RDP)."""
        return self._rdp_to_eps_delta(self._rdp_epsilons, self.delta)
    
    @property
    def remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon_budget - self.spent_epsilon)
    
    @property
    def budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.spent_epsilon >= self.epsilon_budget
    
    @property
    def history(self) -> list[PrivacySpent]:
        """Get history of privacy-consuming operations."""
        return self._history
    
    def get_privacy_spent(self) -> tuple[float, float]:
        """
        Get current total privacy spent.
        
        Returns:
            Tuple of (epsilon_spent, delta)
        """
        return (self.spent_epsilon, self.delta)
    
    def _rdp_to_eps_delta(self, rdp_eps: np.ndarray, delta: float) -> float:
        """
        Convert RDP to (ε, δ)-DP.
        
        Uses the formula: ε = min_α { ρ_α + log(1/δ) / (α-1) }
        """
        # Handle edge cases
        if delta == 0:
            return float('inf')
        
        # For each alpha, compute the implied epsilon
        eps = rdp_eps + np.log(1 / delta) / (self.alphas - 1)
        
        # Return the minimum (tightest bound)
        return float(np.min(eps))
    
    def _compute_gaussian_rdp(
        self,
        sigma: float,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """
        Compute RDP of Gaussian mechanism at all tracked orders.
        
        RDP of Gaussian: ρ_α = α * (Δ / σ)² / 2
        """
        return self.alphas * (sensitivity / sigma) ** 2 / 2
    
    def _compute_subsampled_gaussian_rdp(
        self,
        sigma: float,
        sampling_rate: float,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """
        Compute RDP of subsampled Gaussian mechanism.
        
        Privacy amplification by subsampling provides significant savings.
        """
        rdp = np.zeros(len(self.alphas))
        
        for i, alpha in enumerate(self.alphas):
            if alpha == 1:
                continue  # Skip α=1
            
            # Use Mironov's bound for subsampled Gaussian
            q = sampling_rate
            base_rdp = alpha * (sensitivity / sigma) ** 2 / 2
            
            # Amplification factor (simplified bound)
            if q < 0.5:
                # Tight bound for small q
                rdp[i] = self._compute_subsampled_rdp_order(
                    alpha, q, sensitivity / sigma
                )
            else:
                rdp[i] = base_rdp
        
        return rdp
    
    def _compute_subsampled_rdp_order(
        self,
        alpha: float,
        q: float,
        noise_multiplier: float,
    ) -> float:
        """
        Compute subsampled Gaussian RDP at a specific order.
        
        Using the analytic moments accountant bound.
        """
        # Base RDP (without subsampling)
        base_rdp = alpha * noise_multiplier ** (-2) / 2
        
        if q == 0:
            return 0
        if q == 1:
            return base_rdp
        
        # Subsampling amplification (simplified tight bound)
        # log(1 + q²(exp(2*base_rdp) - 1)) / (alpha - 1)
        if alpha > 1:
            # Use the tighter Poisson subsampling bound
            rdp = np.log1p(q**2 * (np.exp((alpha - 1) * base_rdp) - 1)) / (alpha - 1)
            return min(rdp, base_rdp)
        
        return base_rdp
    
    def accumulate_gaussian(
        self,
        sigma: float,
        sensitivity: float = 1.0,
        operation: str = "gaussian_mechanism",
        metadata: dict | None = None,
    ) -> PrivacySpent:
        """
        Account for a Gaussian mechanism query.
        
        Args:
            sigma: Noise standard deviation
            sensitivity: Query sensitivity
            operation: Name of the operation
            metadata: Optional metadata
            
        Returns:
            PrivacySpent record
        """
        # Compute RDP
        step_rdp = self._compute_gaussian_rdp(sigma, sensitivity)
        
        # Accumulate (RDP composes additively)
        self._rdp_epsilons += step_rdp
        
        # Create record
        spent = PrivacySpent(
            operation=operation,
            epsilon=self.spent_epsilon,
            delta=self.delta,
            rdp_alphas=self.alphas.tolist(),
            rdp_epsilons=step_rdp.tolist(),
            metadata=metadata or {},
        )
        
        self._history.append(spent)
        
        logger.debug(f"Accumulated privacy: ε={self.spent_epsilon:.4f} (budget: {self.epsilon_budget})")
        
        return spent
    
    def accumulate_subsampled_gaussian(
        self,
        sigma: float,
        sampling_rate: float,
        n_steps: int = 1,
        sensitivity: float = 1.0,
        operation: str = "dp_sgd_step",
        metadata: dict | None = None,
    ) -> PrivacySpent:
        """
        Account for subsampled Gaussian mechanism (DP-SGD).
        
        Args:
            sigma: Noise multiplier
            sampling_rate: Batch size / dataset size
            n_steps: Number of training steps
            sensitivity: Gradient sensitivity (after clipping)
            operation: Name of the operation
            metadata: Optional metadata
            
        Returns:
            PrivacySpent record
        """
        # Compute single-step RDP
        step_rdp = self._compute_subsampled_gaussian_rdp(sigma, sampling_rate, sensitivity)
        
        # RDP composes additively across steps
        total_rdp = step_rdp * n_steps
        
        # Accumulate
        self._rdp_epsilons += total_rdp
        
        # Create record
        spent = PrivacySpent(
            operation=operation,
            epsilon=self.spent_epsilon,
            delta=self.delta,
            rdp_alphas=self.alphas.tolist(),
            rdp_epsilons=total_rdp.tolist(),
            metadata={
                "sigma": sigma,
                "sampling_rate": sampling_rate,
                "n_steps": n_steps,
                **(metadata or {}),
            },
        )
        
        self._history.append(spent)
        
        logger.info(
            f"DP-SGD: {n_steps} steps, σ={sigma:.2f}, q={sampling_rate:.4f} → "
            f"ε={self.spent_epsilon:.4f}"
        )
        
        return spent
    
    def can_spend(self, epsilon_estimate: float) -> bool:
        """Check if we can spend an estimated amount without exceeding budget."""
        return self.spent_epsilon + epsilon_estimate <= self.epsilon_budget
    
    def get_noise_multiplier(
        self,
        target_epsilon: float,
        sampling_rate: float,
        n_steps: int,
    ) -> float:
        """
        Compute required noise multiplier to achieve target epsilon.
        
        Uses binary search to find σ such that n_steps of DP-SGD
        yields cumulative privacy (ε, δ).
        
        Args:
            target_epsilon: Target privacy budget to spend
            sampling_rate: Batch size / dataset size
            n_steps: Number of training steps
            
        Returns:
            Required noise multiplier σ
        """
        # Binary search for sigma
        sigma_low, sigma_high = 0.01, 100.0
        
        for _ in range(100):  # Max iterations
            sigma_mid = (sigma_low + sigma_high) / 2
            
            # Compute RDP for this sigma
            rdp = self._compute_subsampled_gaussian_rdp(sigma_mid, sampling_rate) * n_steps
            eps = self._rdp_to_eps_delta(rdp, self.delta)
            
            if abs(eps - target_epsilon) < 0.01:
                return sigma_mid
            elif eps > target_epsilon:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
        
        return sigma_mid
    
    def get_expected_epochs(
        self,
        sigma: float,
        sampling_rate: float,
        target_epsilon: float | None = None,
    ) -> int:
        """
        Compute how many epochs can be run within budget.
        
        Args:
            sigma: Noise multiplier
            sampling_rate: Batch size / dataset size
            target_epsilon: Target budget (default: remaining budget)
            
        Returns:
            Maximum number of epochs
        """
        target = target_epsilon or self.remaining_budget
        
        # Single step RDP
        step_rdp = self._compute_subsampled_gaussian_rdp(sigma, sampling_rate)
        
        # Binary search for max steps
        steps_low, steps_high = 1, 100000
        
        while steps_low < steps_high:
            steps_mid = (steps_low + steps_high + 1) // 2
            total_rdp = step_rdp * steps_mid
            eps = self._rdp_to_eps_delta(total_rdp, self.delta)
            
            if eps <= target:
                steps_low = steps_mid
            else:
                steps_high = steps_mid - 1
        
        # Convert steps to epochs
        steps_per_epoch = int(1 / sampling_rate)
        return max(1, steps_low // steps_per_epoch)
    
    def get_history(self) -> list[dict]:
        """Get privacy consumption history."""
        return [s.to_dict() for s in self._history]
    
    def get_summary(self) -> dict:
        """Get accountant summary."""
        return {
            "budget_epsilon": self.epsilon_budget,
            "spent_epsilon": self.spent_epsilon,
            "remaining_epsilon": self.remaining_budget,
            "delta": self.delta,
            "n_operations": len(self._history),
            "budget_exhausted": self.budget_exhausted,
        }
    
    def reset(self) -> None:
        """Reset accountant (clear history and accumulated privacy)."""
        self._rdp_epsilons = np.zeros(len(self.alphas))
        self._history = []
        logger.info("Privacy accountant reset")


# Global accountant instance (singleton pattern)
_global_accountant: RDPAccountant | None = None


def get_accountant() -> RDPAccountant:
    """Get or create global privacy accountant."""
    global _global_accountant
    
    if _global_accountant is None:
        _global_accountant = RDPAccountant()
    
    return _global_accountant


def reset_accountant() -> None:
    """Reset global privacy accountant."""
    global _global_accountant
    _global_accountant = RDPAccountant()
