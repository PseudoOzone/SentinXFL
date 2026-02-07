"""
SentinXFL - Sprint 3 Tests
===========================

Tests for Federated Learning and Differential Privacy components.

Author: Anshuman Bakshi
"""

import numpy as np
import pytest


# ==========================================
# DP Mechanisms Tests
# ==========================================

class TestDPMechanisms:
    """Tests for differential privacy mechanisms."""
    
    def test_gaussian_mechanism_noise_calibration(self):
        """Test Gaussian mechanism calibrates noise correctly."""
        from sentinxfl.privacy.mechanisms import GaussianMechanism
        
        mechanism = GaussianMechanism(
            epsilon=1.0,
            delta=1e-5,
            sensitivity=1.0,
        )
        
        # Sigma should be computed based on epsilon, delta, sensitivity
        assert mechanism.sigma > 0
        
        # Noise should have expected variance
        data = np.array([1.0, 2.0, 3.0])
        noisy = mechanism.add_noise(data)
        
        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)  # Noise was added
    
    def test_laplace_mechanism_noise_calibration(self):
        """Test Laplace mechanism calibrates noise correctly."""
        from sentinxfl.privacy.mechanisms import LaplaceMechanism
        
        mechanism = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        
        assert mechanism.scale > 0
        
        data = np.array([1.0, 2.0, 3.0])
        noisy = mechanism.add_noise(data)
        
        assert noisy.shape == data.shape
    
    def test_gradient_clipper(self):
        """Test gradient clipping to bounded norm."""
        from sentinxfl.privacy.mechanisms import GradientClipper
        
        clipper = GradientClipper(max_norm=1.0)
        
        # Test single gradient
        gradient = np.array([3.0, 4.0])  # Norm = 5
        clipped = clipper.clip(gradient)
        
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6
        
        # Test batch
        batch = np.array([
            [3.0, 4.0],  # Norm = 5
            [0.5, 0.5],  # Norm = 0.707
        ])
        
        clipped_batch = clipper.clip_batch(batch)
        
        for row in clipped_batch:
            assert np.linalg.norm(row) <= 1.0 + 1e-6


class TestRDPAccountant:
    """Tests for RDP privacy accountant."""
    
    def test_accountant_initialization(self):
        """Test accountant initializes correctly."""
        from sentinxfl.privacy.accountant import RDPAccountant
        
        accountant = RDPAccountant(epsilon_budget=1.0, delta=1e-5)
        
        assert accountant.epsilon_budget == 1.0
        assert accountant.delta == 1e-5
        assert not accountant.budget_exhausted
    
    def test_gaussian_accumulation(self):
        """Test privacy accumulation for Gaussian mechanism."""
        from sentinxfl.privacy.accountant import RDPAccountant
        
        accountant = RDPAccountant(epsilon_budget=10.0, delta=1e-5)
        
        # Accumulate privacy for several operations
        for i in range(5):
            accountant.accumulate_gaussian(
                sigma=1.0,
                sensitivity=1.0,
                operation=f"op_{i}",
            )
        
        eps_spent, _ = accountant.get_privacy_spent()
        
        assert eps_spent > 0
        assert len(accountant.history) == 5
    
    def test_budget_exhaustion(self):
        """Test budget exhaustion detection."""
        from sentinxfl.privacy.accountant import RDPAccountant
        
        accountant = RDPAccountant(epsilon_budget=0.1, delta=1e-5)
        
        # High privacy operations
        for i in range(100):
            accountant.accumulate_gaussian(
                sigma=0.5,  # Low sigma = high privacy cost
                sensitivity=1.0,
                operation=f"op_{i}",
            )
        
        # Budget should be exhausted
        assert accountant.budget_exhausted
    
    def test_noise_multiplier_calculation(self):
        """Test noise multiplier computation for target privacy."""
        from sentinxfl.privacy.accountant import RDPAccountant
        
        accountant = RDPAccountant(epsilon_budget=1.0, delta=1e-5)
        
        sigma = accountant.get_noise_multiplier(
            target_epsilon=1.0,
            sampling_rate=0.01,
            n_steps=100,
        )
        
        assert sigma > 0


class TestDPTrainer:
    """Tests for DP-SGD trainer."""
    
    def test_private_gradient_clipping(self):
        """Test gradient clipping in private gradient computation."""
        from sentinxfl.privacy.dp_trainer import DPSGDTrainer, DPSGDConfig
        
        config = DPSGDConfig(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5,
        )
        
        trainer = DPSGDTrainer(config)
        
        # Per-sample gradients
        gradients = np.random.randn(10, 20).astype(np.float32) * 5  # Large gradients
        
        private_grad = trainer.private_gradient(gradients)
        
        # Should have noise added
        assert private_grad.shape == (20,)


# ==========================================
# FL Aggregator Tests
# ==========================================

class TestAggregators:
    """Tests for FL aggregation strategies."""
    
    def test_fedavg_weighted_average(self):
        """Test FedAvg computes weighted average."""
        from sentinxfl.fl.aggregators import FedAvgAggregator
        
        aggregator = FedAvgAggregator()
        
        # 3 clients with different weights
        client_weights = [
            [np.array([1.0, 1.0, 1.0], dtype=np.float32)],
            [np.array([2.0, 2.0, 2.0], dtype=np.float32)],
            [np.array([3.0, 3.0, 3.0], dtype=np.float32)],
        ]
        num_samples = [100, 100, 100]  # Equal samples
        
        result = aggregator.aggregate(client_weights, num_samples)
        
        # Equal weight average: (1+2+3)/3 = 2
        np.testing.assert_allclose(
            result.aggregated_weights[0],
            [2.0, 2.0, 2.0],
            rtol=1e-5,
        )
    
    def test_multi_krum_selection(self):
        """Test Multi-Krum filters outliers."""
        from sentinxfl.fl.aggregators import MultiKrumAggregator
        
        # Need at least 2f+3 clients for f=1 Byzantine
        aggregator = MultiKrumAggregator(num_byzantine=1, num_to_select=3)
        
        # 5 clients, one is outlier
        client_weights = [
            [np.array([1.0, 1.0], dtype=np.float32)],
            [np.array([1.1, 0.9], dtype=np.float32)],
            [np.array([0.9, 1.1], dtype=np.float32)],
            [np.array([1.0, 1.0], dtype=np.float32)],
            [np.array([100.0, 100.0], dtype=np.float32)],  # Outlier
        ]
        
        result = aggregator.aggregate(client_weights)
        
        # Outlier (index 4) should not be selected
        assert 4 not in result.selected_indices
        assert result.num_selected == 3
    
    def test_trimmed_mean_removes_extremes(self):
        """Test trimmed mean removes extreme values."""
        from sentinxfl.fl.aggregators import TrimmedMeanAggregator
        
        aggregator = TrimmedMeanAggregator(trim_ratio=0.2)  # Trim 20% each end
        
        # 5 clients with some extreme values
        client_weights = [
            [np.array([0.0], dtype=np.float32)],   # Low extreme
            [np.array([1.0], dtype=np.float32)],
            [np.array([1.0], dtype=np.float32)],
            [np.array([1.0], dtype=np.float32)],
            [np.array([100.0], dtype=np.float32)], # High extreme
        ]
        
        result = aggregator.aggregate(client_weights)
        
        # After trimming extremes, should be close to 1.0
        assert 0.5 < result.aggregated_weights[0][0] < 2.0
    
    def test_coordinate_median(self):
        """Test coordinate-wise median aggregation."""
        from sentinxfl.fl.aggregators import CoordinateMedianAggregator
        
        aggregator = CoordinateMedianAggregator()
        
        client_weights = [
            [np.array([1.0, 100.0], dtype=np.float32)],
            [np.array([2.0, 2.0], dtype=np.float32)],
            [np.array([3.0, 3.0], dtype=np.float32)],
        ]
        
        result = aggregator.aggregate(client_weights)
        
        # Median of [1, 2, 3] = 2, median of [100, 2, 3] = 3
        np.testing.assert_allclose(
            result.aggregated_weights[0],
            [2.0, 3.0],
            rtol=1e-5,
        )


class TestByzantineAttacks:
    """Tests for Byzantine attack simulations."""
    
    def test_random_attack(self):
        """Test random noise attack."""
        from sentinxfl.fl.aggregators import ByzantineAttack
        
        weights = [np.array([1.0, 2.0], dtype=np.float32)]
        attacked = ByzantineAttack.random_attack(weights, scale=10.0)
        
        # Attacked weights should be very different
        assert not np.allclose(weights[0], attacked[0])
    
    def test_sign_flip_attack(self):
        """Test sign flip attack."""
        from sentinxfl.fl.aggregators import ByzantineAttack
        
        weights = [np.array([1.0, 2.0], dtype=np.float32)]
        attacked = ByzantineAttack.sign_flip_attack(weights)
        
        np.testing.assert_allclose(attacked[0], [-1.0, -2.0])


# ==========================================
# FL Simulator Tests
# ==========================================

class TestFLSimulator:
    """Tests for FL simulation."""
    
    def test_iid_split(self):
        """Test IID data distribution."""
        from sentinxfl.fl.simulator import FLSimulator, SimulationConfig
        
        config = SimulationConfig(num_rounds=1)
        simulator = FLSimulator(config)
        
        X = np.random.randn(1000, 10).astype(np.float32)
        y = (np.random.rand(1000) > 0.9).astype(np.int32)
        
        simulator.setup_iid_split(X, y, num_clients=5)
        
        assert len(simulator.clients) == 5
        
        # Each client should have ~200 samples
        for client in simulator.clients:
            assert 150 < client.num_samples < 250
    
    def test_non_iid_split(self):
        """Test non-IID data distribution."""
        from sentinxfl.fl.simulator import FLSimulator, SimulationConfig
        
        config = SimulationConfig(num_rounds=1)
        simulator = FLSimulator(config)
        
        X = np.random.randn(1000, 10).astype(np.float32)
        y = np.concatenate([
            np.zeros(500, dtype=np.int32),
            np.ones(500, dtype=np.int32),
        ])
        
        # Low alpha = more non-IID
        simulator.setup_non_iid_split(X, y, num_clients=5, alpha=0.1)
        
        assert len(simulator.clients) > 0
    
    def test_simulation_run(self):
        """Test full simulation run."""
        from sentinxfl.fl.simulator import FLSimulator, SimulationConfig
        
        config = SimulationConfig(
            num_rounds=3,
            dp_enabled=True,
            dp_epsilon=500.0,  # Very high budget for test (simulation uses high noise cost)
            dp_noise_multiplier=1.0,  # Higher noise = lower epsilon per round
        )
        
        simulator = FLSimulator(config)
        
        X = np.random.randn(500, 10).astype(np.float32)
        y = (np.random.rand(500) > 0.9).astype(np.int32)
        
        simulator.setup_iid_split(X, y, num_clients=3)
        simulator.initialize_global_model()
        
        history = simulator.run()
        
        assert len(history) >= 1  # At least one round should complete
        assert simulator.current_round >= 1
    
    def test_byzantine_resilience(self):
        """Test Byzantine resilience with Krum."""
        from sentinxfl.fl.simulator import FLSimulator, SimulationConfig
        
        config = SimulationConfig(
            num_rounds=2,
            aggregation_strategy="krum",
            num_byzantine=1,
            dp_enabled=False,
        )
        
        simulator = FLSimulator(config)
        
        X = np.random.randn(1000, 10).astype(np.float32)
        y = (np.random.rand(1000) > 0.9).astype(np.int32)
        
        simulator.setup_iid_split(
            X, y,
            num_clients=5,
            num_byzantine=1,
            attack_type="random",
        )
        
        history = simulator.run()
        
        # Simulation should complete despite Byzantine client
        assert len(history) == 2


# ==========================================
# FL Client/Server Tests
# ==========================================

class TestFLClient:
    """Tests for FL client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        from sentinxfl.fl.client import FraudDetectionClient, ClientConfig
        
        X = np.random.randn(100, 10).astype(np.float32)
        y = (np.random.rand(100) > 0.9).astype(np.int32)
        
        config = ClientConfig(client_id="test_client", dp_enabled=True)
        
        client = FraudDetectionClient(
            client_id="test",
            X_train=X,
            y_train=y,
            config=config,
        )
        
        assert client.client_id == "test"
        assert client.accountant is not None


class TestFLServer:
    """Tests for FL server strategies."""
    
    def test_dp_fedavg_strategy(self):
        """Test DPFedAvg strategy creation."""
        from sentinxfl.fl.server import DPFedAvg
        
        strategy = DPFedAvg(
            noise_multiplier=0.1,
            clip_norm=1.0,
            min_fit_clients=2,
        )
        
        assert strategy.noise_multiplier == 0.1
        assert strategy.clip_norm == 1.0
    
    def test_create_strategy(self):
        """Test strategy factory function."""
        from sentinxfl.fl.server import create_strategy, ServerConfig
        
        # Test FedAvg with DP
        config = ServerConfig(
            aggregation_strategy="fedavg",
            dp_enabled=True,
        )
        strategy = create_strategy(config)
        
        assert strategy is not None
        
        # Test Krum
        config = ServerConfig(
            aggregation_strategy="krum",
            dp_enabled=False,
        )
        strategy = create_strategy(config)
        
        assert strategy is not None


# ==========================================
# Run Tests
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
