"""
SentinXFL - Federated Learning Server
======================================

Flower-based FL server with custom aggregation strategies
and Byzantine-resilient aggregation.

Author: Anshuman Bakshi
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ServerConfig:
    """Configuration for FL server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    num_rounds: int = 10
    
    # Client settings  
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    
    # Aggregation
    aggregation_strategy: str = "fedavg"  # fedavg, krum, trimmed_mean
    
    # Privacy (optional DP at aggregation)
    dp_enabled: bool = True
    dp_noise_multiplier: float = 0.1
    dp_clip_norm: float = 1.0
    
    # Byzantine resilience
    byzantine_clients: int = 0  # Expected malicious clients


@dataclass
class RoundResult:
    """Result of a single FL round."""
    
    round_num: int
    num_clients: int
    aggregated_loss: float | None = None
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    client_contributions: list[int] = field(default_factory=list)


class DPFedAvg(FedAvg):
    """
    FedAvg with differential privacy noise at aggregation.
    
    Adds Gaussian noise to the aggregated model updates to
    provide server-side DP guarantees.
    """
    
    def __init__(
        self,
        noise_multiplier: float = 0.1,
        clip_norm: float = 1.0,
        **kwargs,
    ):
        """
        Initialize DP-FedAvg strategy.
        
        Args:
            noise_multiplier: Noise scale (sigma)
            clip_norm: L2 norm bound for client updates
            **kwargs: Arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self._round_num = 0
        
        logger.info(
            f"DP-FedAvg: σ={noise_multiplier}, C={clip_norm}"
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate with DP noise addition.
        
        Steps:
        1. Clip each client's update to bounded norm
        2. Aggregate (weighted average)
        3. Add calibrated Gaussian noise
        """
        if not results:
            return None, {}
        
        self._round_num = server_round
        
        # Extract weights and number of samples from each client
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Clip each client's update
        clipped_weights = []
        for client_weights, num_examples in weights_results:
            clipped = self._clip_update(client_weights)
            clipped_weights.append((clipped, num_examples))
        
        # Aggregate (weighted average)
        aggregated = self._aggregate_weights(clipped_weights)
        
        # Add DP noise
        noisy_aggregated = self._add_noise(aggregated)
        
        logger.debug(
            f"Round {server_round}: Aggregated {len(results)} clients with DP"
        )
        
        # Collect metrics
        metrics = {}
        for key in ["loss", "accuracy", "f1"]:
            values = [
                fit_res.metrics.get(key)
                for _, fit_res in results
                if fit_res.metrics and key in fit_res.metrics
            ]
            if values:
                metrics[f"avg_{key}"] = sum(values) / len(values)
        
        return ndarrays_to_parameters(noisy_aggregated), metrics
    
    def _clip_update(
        self, 
        weights: list[NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]:
        """Clip client update to bounded L2 norm."""
        # Flatten weights
        flat = np.concatenate([w.flatten() for w in weights])
        
        # Compute norm
        norm = np.linalg.norm(flat)
        
        # Clip if necessary
        if norm > self.clip_norm:
            scale = self.clip_norm / norm
            return [w * scale for w in weights]
        
        return weights
    
    def _aggregate_weights(
        self,
        weights_results: list[tuple[list[NDArray], int]],
    ) -> list[NDArray[np.float32]]:
        """Weighted average of client weights."""
        total_samples = sum(n for _, n in weights_results)
        
        # Initialize with zeros
        aggregated = [
            np.zeros_like(w) for w in weights_results[0][0]
        ]
        
        # Weighted sum
        for client_weights, num_examples in weights_results:
            weight = num_examples / total_samples
            for i, w in enumerate(client_weights):
                aggregated[i] += w * weight
        
        return aggregated
    
    def _add_noise(
        self,
        weights: list[NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        """Add calibrated Gaussian noise for DP."""
        noise_std = self.noise_multiplier * self.clip_norm
        
        noisy_weights = []
        for w in weights:
            noise = np.random.normal(0, noise_std, w.shape).astype(np.float32)
            noisy_weights.append(w + noise)
        
        return noisy_weights


class MultiKrumStrategy(FedAvg):
    """
    Multi-Krum aggregation for Byzantine resilience.
    
    Selects the updates that are closest to their neighbors,
    filtering out potentially malicious outliers.
    
    Reference: Blanchard et al. "Machine Learning with Adversaries" (NeurIPS 2017)
    """
    
    def __init__(
        self,
        num_to_select: int | None = None,
        num_byzantine: int = 0,
        **kwargs,
    ):
        """
        Initialize Multi-Krum strategy.
        
        Args:
            num_to_select: Number of clients to select (default: n - f - 1)
            num_byzantine: Expected number of Byzantine clients (f)
            **kwargs: Arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.num_to_select = num_to_select
        self.num_byzantine = num_byzantine
        
        logger.info(
            f"Multi-Krum: f={num_byzantine}, m={num_to_select}"
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate using Multi-Krum selection."""
        if not results:
            return None, {}
        
        n = len(results)
        f = self.num_byzantine
        
        # Must have enough clients for Byzantine tolerance
        if n < 2 * f + 3:
            logger.warning(
                f"Not enough clients for Krum: n={n}, f={f}"
            )
            return super().aggregate_fit(server_round, results, failures)
        
        # Default: select all except byzantine + 1
        m = self.num_to_select or (n - f - 1)
        m = min(m, n)
        
        # Extract flattened weights
        client_weights = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            client_weights.append(flat)
        
        # Compute pairwise distances
        distances = self._compute_distances(client_weights)
        
        # Compute Krum scores
        scores = self._compute_krum_scores(distances, n, f)
        
        # Select top m clients (lowest scores)
        selected_indices = np.argsort(scores)[:m]
        
        logger.debug(
            f"Round {server_round}: Krum selected {len(selected_indices)}/{n} clients"
        )
        
        # Aggregate selected clients
        selected_results = [results[i] for i in selected_indices]
        
        # Use parent's aggregation on selected subset
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in selected_results
        ]
        
        # Average selected weights
        total_samples = sum(n for _, n in weights_results)
        aggregated = [
            np.zeros_like(w) for w in weights_results[0][0]
        ]
        
        for client_weights_shaped, num_examples in weights_results:
            weight = num_examples / total_samples
            for i, w in enumerate(client_weights_shaped):
                aggregated[i] += w * weight
        
        return ndarrays_to_parameters(aggregated), {}
    
    def _compute_distances(
        self,
        client_weights: list[NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Compute pairwise Euclidean distances."""
        n = len(client_weights)
        distances = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(client_weights[i] - client_weights[j])
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
        num_neighbors = n - f - 2
        
        for i in range(n):
            # Get distances from client i to all others
            dists = distances[i]
            # Sort and sum smallest n-f-2 (excluding self which is 0)
            sorted_dists = np.sort(dists)
            scores[i] = np.sum(sorted_dists[1:num_neighbors + 1] ** 2)
        
        return scores


class TrimmedMeanStrategy(FedAvg):
    """
    Coordinate-wise trimmed mean aggregation.
    
    For each weight coordinate, removes the highest and lowest
    values before averaging. Robust to outliers.
    """
    
    def __init__(
        self,
        trim_ratio: float = 0.1,
        **kwargs,
    ):
        """
        Initialize Trimmed Mean strategy.
        
        Args:
            trim_ratio: Fraction to trim from each end (default: 10%)
            **kwargs: Arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio
        
        logger.info(f"Trimmed Mean: trim_ratio={trim_ratio}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate using trimmed mean."""
        if not results:
            return None, {}
        
        n = len(results)
        trim_count = max(1, int(n * self.trim_ratio))
        
        if n <= 2 * trim_count:
            logger.warning(
                f"Not enough clients for trimming: n={n}, trim={trim_count}"
            )
            return super().aggregate_fit(server_round, results, failures)
        
        # Extract weights from all clients
        all_weights = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        
        # Coordinate-wise trimmed mean
        aggregated = []
        for layer_idx in range(len(all_weights[0])):
            # Stack all client weights for this layer
            stacked = np.stack([w[layer_idx] for w in all_weights], axis=0)
            
            # Sort along client axis
            sorted_weights = np.sort(stacked, axis=0)
            
            # Trim extremes
            trimmed = sorted_weights[trim_count:-trim_count]
            
            # Mean of remaining
            layer_mean = np.mean(trimmed, axis=0).astype(np.float32)
            aggregated.append(layer_mean)
        
        logger.debug(
            f"Round {server_round}: Trimmed mean with {n} clients, trim={trim_count}"
        )
        
        return ndarrays_to_parameters(aggregated), {}


class CoordinateMedianStrategy(FedAvg):
    """
    Coordinate-wise median aggregation.
    
    Takes the median of each weight coordinate across clients.
    Very robust to outliers but may converge slower.
    """
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate using coordinate-wise median."""
        if not results:
            return None, {}
        
        # Extract weights
        all_weights = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        
        # Coordinate-wise median
        aggregated = []
        for layer_idx in range(len(all_weights[0])):
            stacked = np.stack([w[layer_idx] for w in all_weights], axis=0)
            layer_median = np.median(stacked, axis=0).astype(np.float32)
            aggregated.append(layer_median)
        
        logger.debug(
            f"Round {server_round}: Coordinate median with {len(results)} clients"
        )
        
        return ndarrays_to_parameters(aggregated), {}


# ==========================================
# Server Factory
# ==========================================

def create_strategy(config: ServerConfig) -> fl.server.strategy.Strategy:
    """
    Create FL strategy based on configuration.
    
    Args:
        config: Server configuration
        
    Returns:
        Flower strategy instance
    """
    base_kwargs = {
        "min_fit_clients": config.min_fit_clients,
        "min_evaluate_clients": config.min_evaluate_clients,
        "min_available_clients": config.min_available_clients,
    }
    
    strategy_name = config.aggregation_strategy.lower()
    
    if strategy_name == "fedavg":
        if config.dp_enabled:
            return DPFedAvg(
                noise_multiplier=config.dp_noise_multiplier,
                clip_norm=config.dp_clip_norm,
                **base_kwargs,
            )
        else:
            return FedAvg(**base_kwargs)
    
    elif strategy_name == "krum" or strategy_name == "multikrum":
        return MultiKrumStrategy(
            num_byzantine=config.byzantine_clients,
            **base_kwargs,
        )
    
    elif strategy_name == "trimmed_mean":
        return TrimmedMeanStrategy(**base_kwargs)
    
    elif strategy_name == "median":
        return CoordinateMedianStrategy(**base_kwargs)
    
    else:
        logger.warning(f"Unknown strategy '{strategy_name}', using FedAvg")
        return FedAvg(**base_kwargs)


def start_server(config: ServerConfig | None = None) -> None:
    """
    Start the FL server.
    
    Args:
        config: Server configuration
    """
    config = config or ServerConfig()
    
    strategy = create_strategy(config)
    
    logger.info(
        f"Starting FL server on {config.host}:{config.port} "
        f"with {config.aggregation_strategy} strategy"
    )
    
    fl.server.start_server(
        server_address=f"{config.host}:{config.port}",
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
    )


# ==========================================
# Server State Manager
# ==========================================

class FLServerManager:
    """
    Manages FL server state and provides control interface.
    
    Used for simulation and testing scenarios.
    """
    
    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self.strategy = create_strategy(self.config)
        self.round_results: list[RoundResult] = []
        self.current_round = 0
        self.global_model: list[NDArray] | None = None
    
    def initialize_model(
        self,
        model_weights: list[NDArray[np.float32]],
    ) -> None:
        """Initialize global model weights."""
        self.global_model = model_weights
        logger.info(f"Initialized global model with {len(model_weights)} layers")
    
    def get_model_parameters(self) -> Parameters | None:
        """Get current global model as Flower Parameters."""
        if self.global_model is None:
            return None
        return ndarrays_to_parameters(self.global_model)
    
    def aggregate_round(
        self,
        client_results: list[tuple[list[NDArray], int, dict]],
    ) -> list[NDArray]:
        """
        Manually aggregate a round's results.
        
        Used for simulation without network.
        
        Args:
            client_results: List of (weights, num_samples, metrics) tuples
            
        Returns:
            Aggregated model weights
        """
        self.current_round += 1
        
        # Convert to Flower format
        class MockClientProxy(ClientProxy):
            def __init__(self, cid: str):
                self._cid = cid
            
            @property
            def cid(self) -> str:
                return self._cid
            
            def get_properties(self, ins, timeout=None):
                raise NotImplementedError()
            
            def get_parameters(self, ins, timeout=None):
                raise NotImplementedError()
            
            def fit(self, ins, timeout=None):
                raise NotImplementedError()
            
            def evaluate(self, ins, timeout=None):
                raise NotImplementedError()
            
            def reconnect(self, ins, timeout=None):
                raise NotImplementedError()
        
        results = []
        for i, (weights, num_samples, metrics) in enumerate(client_results):
            proxy = MockClientProxy(cid=str(i))
            fit_res = FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message=""),
                parameters=ndarrays_to_parameters(weights),
                num_examples=num_samples,
                metrics=metrics,
            )
            results.append((proxy, fit_res))
        
        # Aggregate
        aggregated_params, metrics = self.strategy.aggregate_fit(
            server_round=self.current_round,
            results=results,
            failures=[],
        )
        
        if aggregated_params is not None:
            self.global_model = parameters_to_ndarrays(aggregated_params)
        
        # Record result
        round_result = RoundResult(
            round_num=self.current_round,
            num_clients=len(client_results),
            aggregated_metrics=dict(metrics),
            client_contributions=[n for _, n, _ in client_results],
        )
        self.round_results.append(round_result)
        
        logger.info(
            f"Round {self.current_round}: Aggregated {len(client_results)} clients"
        )
        
        return self.global_model
