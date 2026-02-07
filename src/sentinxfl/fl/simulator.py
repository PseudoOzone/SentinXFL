"""
SentinXFL - Federated Learning Simulator
=========================================

Single-machine FL simulation for development and testing.
Simulates multiple clients without network overhead.

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.fl.aggregators import (
    AggregationResult,
    BaseAggregator,
    ByzantineAttack,
    create_aggregator,
)
from sentinxfl.privacy.accountant import RDPAccountant

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SimulatedClient:
    """Simulated FL client with local data."""
    
    client_id: str
    X_train: NDArray[np.float32]
    y_train: NDArray[np.int32]
    X_val: NDArray[np.float32] | None = None
    y_val: NDArray[np.int32] | None = None
    is_byzantine: bool = False
    attack_type: str | None = None  # random, sign_flip, scaling
    
    # Local training state
    local_model_weights: list[NDArray[np.float32]] | None = None
    training_history: list[dict] = field(default_factory=list)
    
    @property
    def num_samples(self) -> int:
        return len(self.X_train)
    
    @property
    def fraud_ratio(self) -> float:
        """Ratio of fraud cases in local data."""
        return np.mean(self.y_train == 1)


@dataclass
class RoundMetrics:
    """Metrics for a single FL round."""
    
    round_num: int
    num_clients: int
    avg_train_loss: float | None = None
    avg_val_loss: float | None = None
    avg_f1: float | None = None
    avg_auc: float | None = None
    global_val_loss: float | None = None
    global_f1: float | None = None
    global_auc: float | None = None
    clients_selected: int | None = None
    privacy_spent: float | None = None
    
    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "num_clients": self.num_clients,
            "avg_train_loss": self.avg_train_loss,
            "avg_val_loss": self.avg_val_loss,
            "avg_f1": self.avg_f1,
            "global_val_loss": self.global_val_loss,
            "global_f1": self.global_f1,
            "privacy_spent": self.privacy_spent,
        }


@dataclass
class SimulationConfig:
    """Configuration for FL simulation."""
    
    # FL settings
    num_rounds: int = 10
    clients_per_round: int | None = None  # None = all clients
    
    # Aggregation
    aggregation_strategy: str = "fedavg"
    num_byzantine: int = 0
    
    # Local training
    local_epochs: int = 1
    local_batch_size: int = 32
    local_learning_rate: float = 0.01
    
    # Privacy
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_noise_multiplier: float = 0.1
    dp_clip_norm: float = 1.0
    
    # Model
    model_type: str = "xgboost"


class FLSimulator:
    """
    Federated Learning Simulator.
    
    Simulates FL training without network communication.
    Useful for rapid prototyping and algorithm development.
    """
    
    def __init__(
        self,
        config: SimulationConfig | None = None,
        model_factory: Callable | None = None,
    ):
        """
        Initialize FL simulator.
        
        Args:
            config: Simulation configuration
            model_factory: Factory function to create model instances
        """
        self.config = config or SimulationConfig()
        self.model_factory = model_factory
        
        # Clients
        self.clients: list[SimulatedClient] = []
        
        # Global state
        self.global_model_weights: list[NDArray[np.float32]] | None = None
        self.round_history: list[RoundMetrics] = []
        self.current_round = 0
        
        # Aggregator
        self.aggregator = create_aggregator(
            self.config.aggregation_strategy,
            num_byzantine=self.config.num_byzantine,
        )
        
        # Privacy accountant (global)
        if self.config.dp_enabled:
            self.accountant = RDPAccountant(
                epsilon_budget=self.config.dp_epsilon,
                delta=self.config.dp_delta,
            )
        else:
            self.accountant = None
        
        logger.info(
            f"FL Simulator: {self.config.aggregation_strategy} aggregation, "
            f"DP={'ON' if self.config.dp_enabled else 'OFF'}"
        )
    
    def add_client(
        self,
        client_id: str,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        is_byzantine: bool = False,
        attack_type: str | None = None,
    ) -> None:
        """
        Add a simulated client.
        
        Args:
            client_id: Unique identifier
            X_train: Local training features
            y_train: Local training labels
            X_val: Validation features
            y_val: Validation labels
            is_byzantine: Whether client is malicious
            attack_type: Type of Byzantine attack
        """
        client = SimulatedClient(
            client_id=client_id,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            is_byzantine=is_byzantine,
            attack_type=attack_type,
        )
        
        self.clients.append(client)
        
        logger.debug(
            f"Added client {client_id}: {client.num_samples} samples, "
            f"fraud_ratio={client.fraud_ratio:.2%}"
            + (f", BYZANTINE ({attack_type})" if is_byzantine else "")
        )
    
    def setup_iid_split(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        num_clients: int,
        val_ratio: float = 0.2,
        num_byzantine: int = 0,
        attack_type: str = "random",
    ) -> None:
        """
        Create IID data split across clients.
        
        Args:
            X: Full feature matrix
            y: Full labels
            num_clients: Number of clients
            val_ratio: Validation split ratio
            num_byzantine: Number of Byzantine clients
            attack_type: Attack type for Byzantine clients
        """
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        # Split into client chunks
        chunk_size = n_samples // num_clients
        
        for i in range(num_clients):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_clients - 1 else n_samples
            
            client_indices = indices[start:end]
            X_client = X[client_indices]
            y_client = y[client_indices]
            
            # Split into train/val
            val_size = int(len(X_client) * val_ratio)
            X_train = X_client[val_size:]
            y_train = y_client[val_size:]
            X_val = X_client[:val_size]
            y_val = y_client[:val_size]
            
            # Determine if Byzantine
            is_byzantine = i < num_byzantine
            
            self.add_client(
                client_id=f"client_{i}",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                is_byzantine=is_byzantine,
                attack_type=attack_type if is_byzantine else None,
            )
        
        logger.info(
            f"Created IID split: {num_clients} clients, "
            f"{num_byzantine} Byzantine"
        )
    
    def setup_non_iid_split(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        num_clients: int,
        alpha: float = 0.5,
        val_ratio: float = 0.2,
    ) -> None:
        """
        Create non-IID split using Dirichlet distribution.
        
        Args:
            X: Full feature matrix
            y: Full labels  
            num_clients: Number of clients
            alpha: Dirichlet concentration (lower = more non-IID)
            val_ratio: Validation split ratio
        """
        # Separate by class
        class_0_idx = np.where(y == 0)[0]
        class_1_idx = np.where(y == 1)[0]
        
        # Sample proportions from Dirichlet
        proportions_0 = np.random.dirichlet([alpha] * num_clients)
        proportions_1 = np.random.dirichlet([alpha] * num_clients)
        
        # Distribute samples
        np.random.shuffle(class_0_idx)
        np.random.shuffle(class_1_idx)
        
        # Calculate split points
        splits_0 = (proportions_0 * len(class_0_idx)).astype(int)
        splits_1 = (proportions_1 * len(class_1_idx)).astype(int)
        
        # Ensure total matches
        splits_0[-1] = len(class_0_idx) - splits_0[:-1].sum()
        splits_1[-1] = len(class_1_idx) - splits_1[:-1].sum()
        
        offset_0 = 0
        offset_1 = 0
        
        for i in range(num_clients):
            # Get this client's indices
            client_0_idx = class_0_idx[offset_0:offset_0 + splits_0[i]]
            client_1_idx = class_1_idx[offset_1:offset_1 + splits_1[i]]
            
            offset_0 += splits_0[i]
            offset_1 += splits_1[i]
            
            # Combine and shuffle
            client_indices = np.concatenate([client_0_idx, client_1_idx])
            np.random.shuffle(client_indices)
            
            if len(client_indices) == 0:
                continue
            
            X_client = X[client_indices]
            y_client = y[client_indices]
            
            # Split into train/val
            val_size = max(1, int(len(X_client) * val_ratio))
            X_train = X_client[val_size:]
            y_train = y_client[val_size:]
            X_val = X_client[:val_size]
            y_val = y_client[:val_size]
            
            self.add_client(
                client_id=f"client_{i}",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
        
        logger.info(
            f"Created non-IID split (α={alpha}): {len(self.clients)} clients"
        )
    
    def initialize_global_model(
        self,
        model_weights: list[NDArray[np.float32]] | None = None,
    ) -> None:
        """
        Initialize global model.
        
        Args:
            model_weights: Initial weights (if None, train on first client)
        """
        if model_weights is not None:
            self.global_model_weights = model_weights
        elif self.model_factory is not None and self.clients:
            # Train initial model on first client
            model = self.model_factory()
            client = self.clients[0]
            model.fit(client.X_train, client.y_train)
            
            if hasattr(model, "get_weights"):
                self.global_model_weights = model.get_weights()
            else:
                # Placeholder weights for tree models
                self.global_model_weights = [np.zeros(100, dtype=np.float32)]
        else:
            self.global_model_weights = [np.zeros(100, dtype=np.float32)]
        
        logger.info("Global model initialized")
    
    def _train_client_local(
        self,
        client: SimulatedClient,
    ) -> tuple[list[NDArray], dict]:
        """Train a single client locally."""
        if self.model_factory is None:
            # Return dummy weights
            return self.global_model_weights.copy(), {"loss": 0.5}
        
        # Create and train local model
        model = self.model_factory()
        
        # Set global weights if supported
        if hasattr(model, "set_weights") and self.global_model_weights:
            model.set_weights(self.global_model_weights)
        
        # Train locally
        model.fit(client.X_train, client.y_train)
        
        # Get metrics
        metrics = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(client.X_train)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            from sklearn.metrics import log_loss, f1_score
            
            metrics["loss"] = log_loss(client.y_train, proba)
            preds = (proba > 0.5).astype(int)
            metrics["f1"] = f1_score(client.y_train, preds, zero_division=0)
        
        # Get updated weights
        if hasattr(model, "get_weights"):
            weights = model.get_weights()
        else:
            weights = self.global_model_weights.copy()
        
        # Apply Byzantine attack if applicable
        if client.is_byzantine and client.attack_type:
            weights = self._apply_attack(weights, client.attack_type)
        
        # Apply local DP
        if self.config.dp_enabled:
            weights = self._apply_local_dp(weights)
        
        return weights, metrics
    
    def _apply_attack(
        self,
        weights: list[NDArray],
        attack_type: str,
    ) -> list[NDArray]:
        """Apply Byzantine attack to weights."""
        attack = ByzantineAttack()
        
        if attack_type == "random":
            return attack.random_attack(weights)
        elif attack_type == "sign_flip":
            return attack.sign_flip_attack(weights)
        elif attack_type == "scaling":
            return attack.scaling_attack(weights)
        else:
            return weights
    
    def _apply_local_dp(
        self,
        weights: list[NDArray],
    ) -> list[NDArray]:
        """Apply local DP to weights."""
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
        
        return noisy_weights
    
    def run_round(self) -> RoundMetrics:
        """
        Execute one FL round.
        
        Returns:
            Metrics for this round
        """
        self.current_round += 1
        
        # Select clients for this round
        if self.config.clients_per_round and self.config.clients_per_round < len(self.clients):
            selected_clients = np.random.choice(
                self.clients,
                self.config.clients_per_round,
                replace=False,
            ).tolist()
        else:
            selected_clients = self.clients
        
        # Train locally on each client
        client_weights = []
        client_samples = []
        all_metrics = []
        
        for client in selected_clients:
            weights, metrics = self._train_client_local(client)
            
            client_weights.append(weights)
            client_samples.append(client.num_samples)
            all_metrics.append(metrics)
            
            # Update client state
            client.local_model_weights = weights
            client.training_history.append(metrics)
        
        # Aggregate
        agg_result = self.aggregator.aggregate(client_weights, client_samples)
        self.global_model_weights = agg_result.aggregated_weights
        
        # Account for privacy
        privacy_spent = None
        if self.accountant:
            self.accountant.accumulate_gaussian(
                sigma=self.config.dp_noise_multiplier,
                sensitivity=self.config.dp_clip_norm,
                operation=f"round_{self.current_round}",
            )
            privacy_spent = self.accountant.get_privacy_spent()[0]
        
        # Compute average metrics
        avg_loss = np.mean([m.get("loss", 0) for m in all_metrics])
        avg_f1 = np.mean([m.get("f1", 0) for m in all_metrics])
        
        round_metrics = RoundMetrics(
            round_num=self.current_round,
            num_clients=len(selected_clients),
            avg_train_loss=float(avg_loss),
            avg_f1=float(avg_f1),
            clients_selected=agg_result.num_selected,
            privacy_spent=privacy_spent,
        )
        
        self.round_history.append(round_metrics)
        
        logger.info(
            f"Round {self.current_round}: {len(selected_clients)} clients, "
            f"loss={avg_loss:.4f}, f1={avg_f1:.4f}"
            + (f", ε={privacy_spent:.2f}" if privacy_spent else "")
        )
        
        return round_metrics
    
    def run(
        self,
        num_rounds: int | None = None,
    ) -> list[RoundMetrics]:
        """
        Run full FL simulation.
        
        Args:
            num_rounds: Number of rounds (uses config if None)
            
        Returns:
            List of round metrics
        """
        num_rounds = num_rounds or self.config.num_rounds
        
        if not self.clients:
            raise ValueError("No clients added. Use add_client() or setup_*_split() first.")
        
        if self.global_model_weights is None:
            self.initialize_global_model()
        
        logger.info(f"Starting FL simulation: {num_rounds} rounds, {len(self.clients)} clients")
        
        for _ in range(num_rounds):
            self.run_round()
            
            # Check privacy budget
            if self.accountant and self.accountant.budget_exhausted:
                logger.warning("Privacy budget exhausted. Stopping early.")
                break
        
        return self.round_history
    
    def get_summary(self) -> dict:
        """Get simulation summary."""
        return {
            "total_rounds": self.current_round,
            "num_clients": len(self.clients),
            "aggregation_strategy": self.config.aggregation_strategy,
            "dp_enabled": self.config.dp_enabled,
            "final_epsilon": (
                self.accountant.get_privacy_spent()[0]
                if self.accountant else None
            ),
            "final_avg_loss": (
                self.round_history[-1].avg_train_loss
                if self.round_history else None
            ),
            "final_avg_f1": (
                self.round_history[-1].avg_f1
                if self.round_history else None
            ),
        }


# ==========================================
# Quick Simulation Functions
# ==========================================

def quick_simulate(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    num_clients: int = 5,
    num_rounds: int = 10,
    aggregation: str = "fedavg",
    dp_enabled: bool = True,
    non_iid: bool = False,
    num_byzantine: int = 0,
) -> dict:
    """
    Run quick FL simulation with minimal setup.
    
    Args:
        X: Feature matrix
        y: Labels
        num_clients: Number of clients
        num_rounds: Number of FL rounds
        aggregation: Aggregation strategy
        dp_enabled: Enable differential privacy
        non_iid: Use non-IID data split
        num_byzantine: Number of Byzantine clients
        
    Returns:
        Simulation summary
    """
    config = SimulationConfig(
        num_rounds=num_rounds,
        aggregation_strategy=aggregation,
        dp_enabled=dp_enabled,
        num_byzantine=num_byzantine,
    )
    
    simulator = FLSimulator(config)
    
    if non_iid:
        simulator.setup_non_iid_split(X, y, num_clients)
    else:
        simulator.setup_iid_split(
            X, y, num_clients,
            num_byzantine=num_byzantine,
        )
    
    simulator.run()
    
    return simulator.get_summary()
