"""
SentinXFL - Byzantine-Resilient Aggregators
=============================================

Standalone aggregation algorithms for Byzantine fault tolerance
in federated learning. Can be used with or without Flower.

Author: Anshuman Bakshi
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AggregationResult:
    """Result of model aggregation."""
    
    aggregated_weights: list[NDArray[np.float32]]
    num_selected: int
    num_total: int
    selected_indices: list[int] | None = None
    scores: NDArray[np.float32] | None = None
    metadata: dict[str, Any] | None = None


class BaseAggregator(ABC):
    """Abstract base class for aggregation strategies."""
    
    @abstractmethod
    def aggregate(
        self,
        client_weights: list[list[NDArray[np.float32]]],
        num_samples: list[int] | None = None,
    ) -> AggregationResult:
        """
        Aggregate client weights.
        
        Args:
            client_weights: List of client weight arrays
            num_samples: Number of samples per client (for weighting)
            
        Returns:
            Aggregation result
        """
        pass
    
    @staticmethod
    def flatten_weights(weights: list[NDArray]) -> NDArray[np.float32]:
        """Flatten list of weight arrays to single vector."""
        return np.concatenate([w.flatten() for w in weights])
    
    @staticmethod
    def unflatten_weights(
        flat: NDArray,
        shapes: list[tuple],
    ) -> list[NDArray[np.float32]]:
        """Unflatten vector back to list of weight arrays."""
        weights = []
        offset = 0
        for shape in shapes:
            size = np.prod(shape)
            weights.append(flat[offset:offset + size].reshape(shape))
            offset += size
        return weights


class FedAvgAggregator(BaseAggregator):
    """
    Standard Federated Averaging.
    
    Weighted average of client updates based on number of samples.
    """
    
    def aggregate(
        self,
        client_weights: list[list[NDArray[np.float32]]],
        num_samples: list[int] | None = None,
    ) -> AggregationResult:
        """Compute weighted average of client weights."""
        n_clients = len(client_weights)
        
        if num_samples is None:
            num_samples = [1] * n_clients
        
        total_samples = sum(num_samples)
        
        # Initialize with zeros
        aggregated = [
            np.zeros_like(w, dtype=np.float32)
            for w in client_weights[0]
        ]
        
        # Weighted sum
        for client_w, n in zip(client_weights, num_samples):
            weight = n / total_samples
            for i, w in enumerate(client_w):
                aggregated[i] += w.astype(np.float32) * weight
        
        return AggregationResult(
            aggregated_weights=aggregated,
            num_selected=n_clients,
            num_total=n_clients,
            selected_indices=list(range(n_clients)),
        )


class MultiKrumAggregator(BaseAggregator):
    """
    Multi-Krum Byzantine-resilient aggregation.
    
    Selects clients whose updates are closest to their neighbors,
    filtering out outliers that may be malicious.
    
    Reference: Blanchard et al. "Machine Learning with Adversaries" (NeurIPS 2017)
    """
    
    def __init__(
        self,
        num_byzantine: int = 0,
        num_to_select: int | None = None,
    ):
        """
        Initialize Multi-Krum aggregator.
        
        Args:
            num_byzantine: Expected number of Byzantine clients (f)
            num_to_select: Number of clients to select (default: n - f - 1)
        """
        self.num_byzantine = num_byzantine
        self.num_to_select = num_to_select
    
    def aggregate(
        self,
        client_weights: list[list[NDArray[np.float32]]],
        num_samples: list[int] | None = None,
    ) -> AggregationResult:
        """Aggregate using Multi-Krum selection."""
        n = len(client_weights)
        f = self.num_byzantine
        
        # Check if we have enough clients
        if n < 2 * f + 3:
            logger.warning(
                f"Multi-Krum requires n >= 2f + 3. Got n={n}, f={f}. "
                "Falling back to FedAvg."
            )
            return FedAvgAggregator().aggregate(client_weights, num_samples)
        
        # Default selection count
        m = self.num_to_select or (n - f - 1)
        m = min(m, n)
        
        # Flatten weights for distance computation
        flat_weights = [self.flatten_weights(w) for w in client_weights]
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(flat_weights)
        
        # Compute Krum scores
        scores = self._compute_krum_scores(distances, n, f)
        
        # Select top m clients (lowest scores)
        selected_indices = np.argsort(scores)[:m].tolist()
        
        logger.debug(
            f"Multi-Krum: selected {len(selected_indices)}/{n} clients, "
            f"scores range: [{scores.min():.2f}, {scores.max():.2f}]"
        )
        
        # Aggregate selected clients (equal weight or by samples)
        selected_weights = [client_weights[i] for i in selected_indices]
        selected_samples = (
            [num_samples[i] for i in selected_indices]
            if num_samples else None
        )
        
        agg_result = FedAvgAggregator().aggregate(selected_weights, selected_samples)
        
        return AggregationResult(
            aggregated_weights=agg_result.aggregated_weights,
            num_selected=len(selected_indices),
            num_total=n,
            selected_indices=selected_indices,
            scores=scores,
            metadata={"krum_scores": scores.tolist()},
        )
    
    def _compute_pairwise_distances(
        self,
        flat_weights: list[NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Compute pairwise Euclidean distances."""
        n = len(flat_weights)
        distances = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(flat_weights[i] - flat_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _compute_krum_scores(
        self,
        distances: NDArray[np.float32],
        n: int,
        f: int,
    ) -> NDArray[np.float32]:
        """
        Compute Krum scores.
        
        Score(i) = sum of squared distances to n-f-2 closest neighbors
        """
        scores = np.zeros(n, dtype=np.float32)
        num_neighbors = max(1, n - f - 2)
        
        for i in range(n):
            dists = distances[i]
            sorted_dists = np.sort(dists)
            # Skip index 0 (self-distance = 0)
            scores[i] = np.sum(sorted_dists[1:num_neighbors + 1] ** 2)
        
        return scores


class TrimmedMeanAggregator(BaseAggregator):
    """
    Coordinate-wise trimmed mean aggregation.
    
    For each weight coordinate, removes the highest and lowest
    values before averaging. Robust to bounded Byzantine attacks.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        """
        Initialize Trimmed Mean aggregator.
        
        Args:
            trim_ratio: Fraction to trim from each end (default: 10%)
        """
        self.trim_ratio = trim_ratio
    
    def aggregate(
        self,
        client_weights: list[list[NDArray[np.float32]]],
        num_samples: list[int] | None = None,
    ) -> AggregationResult:
        """Aggregate using coordinate-wise trimmed mean."""
        n = len(client_weights)
        trim_count = max(1, int(n * self.trim_ratio))
        
        if n <= 2 * trim_count:
            logger.warning(
                f"Not enough clients for trimming: n={n}, trim={trim_count}. "
                "Falling back to FedAvg."
            )
            return FedAvgAggregator().aggregate(client_weights, num_samples)
        
        aggregated = []
        
        for layer_idx in range(len(client_weights[0])):
            # Stack all client weights for this layer
            stacked = np.stack(
                [w[layer_idx].astype(np.float32) for w in client_weights],
                axis=0,
            )
            
            # Sort along client axis
            sorted_weights = np.sort(stacked, axis=0)
            
            # Trim extremes
            trimmed = sorted_weights[trim_count:-trim_count]
            
            # Mean of remaining
            layer_mean = np.mean(trimmed, axis=0).astype(np.float32)
            aggregated.append(layer_mean)
        
        logger.debug(
            f"Trimmed Mean: aggregated {n} clients, trimmed {trim_count} from each end"
        )
        
        return AggregationResult(
            aggregated_weights=aggregated,
            num_selected=n - 2 * trim_count,
            num_total=n,
            metadata={"trim_count": trim_count},
        )


class CoordinateMedianAggregator(BaseAggregator):
    """
    Coordinate-wise median aggregation.
    
    Takes the median of each weight coordinate across clients.
    Very robust to outliers but may converge slower.
    """
    
    def aggregate(
        self,
        client_weights: list[list[NDArray[np.float32]]],
        num_samples: list[int] | None = None,
    ) -> AggregationResult:
        """Aggregate using coordinate-wise median."""
        n = len(client_weights)
        
        aggregated = []
        
        for layer_idx in range(len(client_weights[0])):
            stacked = np.stack(
                [w[layer_idx].astype(np.float32) for w in client_weights],
                axis=0,
            )
            layer_median = np.median(stacked, axis=0).astype(np.float32)
            aggregated.append(layer_median)
        
        logger.debug(f"Coordinate Median: aggregated {n} clients")
        
        return AggregationResult(
            aggregated_weights=aggregated,
            num_selected=n,
            num_total=n,
        )


class BulyanAggregator(BaseAggregator):
    """
    Bulyan Byzantine-resilient aggregation.
    
    Combines Krum selection with trimmed mean for stronger guarantees.
    First selects 2(n-2f-2) clients using Krum, then applies trimmed mean.
    
    Reference: Guerraoui et al. "The Hidden Vulnerability of Distributed Learning" (ICML 2018)
    """
    
    def __init__(
        self,
        num_byzantine: int = 0,
        trim_ratio: float = 0.25,
    ):
        """
        Initialize Bulyan aggregator.
        
        Args:
            num_byzantine: Expected number of Byzantine clients
            trim_ratio: Trim ratio for final aggregation
        """
        self.num_byzantine = num_byzantine
        self.trim_ratio = trim_ratio
    
    def aggregate(
        self,
        client_weights: list[list[NDArray[np.float32]]],
        num_samples: list[int] | None = None,
    ) -> AggregationResult:
        """Aggregate using Bulyan (Krum + trimmed mean)."""
        n = len(client_weights)
        f = self.num_byzantine
        
        # Bulyan requires n >= 4f + 3
        if n < 4 * f + 3:
            logger.warning(
                f"Bulyan requires n >= 4f + 3. Got n={n}, f={f}. "
                "Falling back to Multi-Krum."
            )
            return MultiKrumAggregator(f).aggregate(client_weights, num_samples)
        
        # Step 1: Krum selection to get 2(n - 2f - 2) clients
        krum_select_count = 2 * (n - 2 * f - 2)
        
        krum = MultiKrumAggregator(
            num_byzantine=f,
            num_to_select=krum_select_count,
        )
        krum_result = krum.aggregate(client_weights, num_samples)
        
        # Step 2: Apply trimmed mean to Krum-selected clients
        selected_weights = [
            client_weights[i] for i in krum_result.selected_indices
        ]
        selected_samples = (
            [num_samples[i] for i in krum_result.selected_indices]
            if num_samples else None
        )
        
        trimmed = TrimmedMeanAggregator(self.trim_ratio)
        final_result = trimmed.aggregate(selected_weights, selected_samples)
        
        logger.debug(
            f"Bulyan: Krum selected {len(selected_weights)}, "
            f"then trimmed mean aggregated"
        )
        
        return AggregationResult(
            aggregated_weights=final_result.aggregated_weights,
            num_selected=final_result.num_selected,
            num_total=n,
            selected_indices=krum_result.selected_indices,
            metadata={
                "krum_selected": len(krum_result.selected_indices),
                "final_aggregated": final_result.num_selected,
            },
        )


# ==========================================
# Byzantine Attack Simulators (for testing)
# ==========================================

class ByzantineAttack:
    """Simulate Byzantine attacks for testing robustness."""
    
    @staticmethod
    def random_attack(
        weights: list[NDArray[np.float32]],
        scale: float = 10.0,
    ) -> list[NDArray[np.float32]]:
        """Replace weights with random noise."""
        return [
            np.random.randn(*w.shape).astype(np.float32) * scale
            for w in weights
        ]
    
    @staticmethod
    def sign_flip_attack(
        weights: list[NDArray[np.float32]],
        scale: float = 1.0,
    ) -> list[NDArray[np.float32]]:
        """Flip signs of weights and optionally scale."""
        return [-w * scale for w in weights]
    
    @staticmethod
    def scaling_attack(
        weights: list[NDArray[np.float32]],
        scale: float = 100.0,
    ) -> list[NDArray[np.float32]]:
        """Scale weights by large factor."""
        return [w * scale for w in weights]
    
    @staticmethod
    def label_flip_gradient(
        honest_grad: list[NDArray[np.float32]],
        scale: float = 1.0,
    ) -> list[NDArray[np.float32]]:
        """Simulate gradient from flipped labels."""
        return [-g * scale for g in honest_grad]


# ==========================================
# Factory Function
# ==========================================

def create_aggregator(
    strategy: str,
    num_byzantine: int = 0,
    **kwargs,
) -> BaseAggregator:
    """
    Create aggregator by strategy name.
    
    Args:
        strategy: Aggregation strategy name
        num_byzantine: Expected Byzantine clients
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        Configured aggregator instance
    """
    strategy = strategy.lower()
    
    if strategy == "fedavg":
        return FedAvgAggregator()
    
    elif strategy in ("krum", "multikrum", "multi-krum"):
        return MultiKrumAggregator(
            num_byzantine=num_byzantine,
            num_to_select=kwargs.get("num_to_select"),
        )
    
    elif strategy in ("trimmed_mean", "trimmed-mean", "trimmedmean"):
        return TrimmedMeanAggregator(
            trim_ratio=kwargs.get("trim_ratio", 0.1),
        )
    
    elif strategy == "median":
        return CoordinateMedianAggregator()
    
    elif strategy == "bulyan":
        return BulyanAggregator(
            num_byzantine=num_byzantine,
            trim_ratio=kwargs.get("trim_ratio", 0.25),
        )
    
    else:
        logger.warning(f"Unknown strategy '{strategy}', using FedAvg")
        return FedAvgAggregator()
