"""
Federated Learning Framework Setup
Client-server architecture for distributed fraud detection training
"""

from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime
import numpy as np

try:
    import flwr as fl
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    # Client config
    num_clients: int = 3
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Server config
    num_rounds: int = 10
    min_clients_available: int = 2
    
    # Communication
    compression_ratio: float = 0.1  # Compress models to 10% size
    quantization_bits: int = 32  # 32-bit precision
    
    # Privacy
    use_differential_privacy: bool = False
    epsilon: float = 5.0  # DP epsilon value
    delta: float = 1e-5   # DP delta value
    
    # Aggregation
    aggregation_method: str = 'FedAvg'  # FedAvg, FedProx, FedAttentive
    learning_rate_schedule: str = 'constant'  # constant, decay
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'num_clients': self.num_clients,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_rounds': self.num_rounds,
            'min_clients_available': self.min_clients_available,
            'compression_ratio': self.compression_ratio,
            'use_differential_privacy': self.use_differential_privacy,
            'epsilon': self.epsilon,
            'aggregation_method': self.aggregation_method
        }


class FederatedEmbeddingModel(nn.Module):
    """DistilBERT model for federated learning"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = 768
        
        # Simplified embedding layer for demonstration
        # In production, use actual DistilBERT
        self.embedding = nn.Embedding(30522, self.embedding_dim)  # BERT vocab size
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        embeddings = self.embedding(input_ids)  # [batch, seq_len, 768]
        
        # Global average pooling over sequence
        embeddings = embeddings.transpose(1, 2)  # [batch, 768, seq_len]
        pooled = self.pooling(embeddings)  # [batch, 768, 1]
        output = pooled.squeeze(2)  # [batch, 768]
        
        return output


class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: str, model: nn.Module, 
                 train_loader: DataLoader, config: FederatedConfig):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_loader: Training data loader
            config: Federated configuration
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'rounds': [],
            'losses': [],
            'accuracies': []
        }
        
        logger.info(f"Client {client_id} initialized on {self.device}")
    
    def train_local(self) -> Dict[str, float]:
        """Train model locally for one or more epochs"""
        logger.info(f"Client {self.client_id} starting local training...")
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(data.long())
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Client {self.client_id} Epoch {epoch + 1}/{self.config.local_epochs} "
                       f"Loss: {avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        avg_loss = total_loss / self.config.local_epochs
        self.history['losses'].append(avg_loss)
        
        return {
            'client_id': self.client_id,
            'loss': avg_loss,
            'num_samples': len(self.train_loader.dataset),
            'num_updates': num_batches
        }
    
    def get_model_weights(self) -> List[np.ndarray]:
        """Extract model weights as numpy arrays"""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy())
        return weights
    
    def set_model_weights(self, weights: List[np.ndarray]):
        """Set model weights from numpy arrays"""
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data = torch.from_numpy(weight).to(param.device)
    
    def compress_weights(self) -> List[np.ndarray]:
        """Compress weights for efficient transmission"""
        weights = self.get_model_weights()
        
        # Simple compression: reduce precision
        if self.config.quantization_bits < 32:
            compressed = []
            for w in weights:
                # Quantization to lower precision
                w_min, w_max = w.min(), w.max()
                w_scaled = (w - w_min) / (w_max - w_min + 1e-8)
                w_quantized = (w_scaled * (2 ** self.config.quantization_bits - 1)).astype(np.int32)
                compressed.append((w_quantized, w_min, w_max))
            return compressed
        
        return weights


class FederatedServer:
    """Federated learning server (coordinator)"""
    
    def __init__(self, model: nn.Module, config: FederatedConfig):
        """
        Initialize federated server
        
        Args:
            model: Initial global model
            config: Federated configuration
        """
        self.model = model
        self.config = config
        self.global_weights = None
        self.client_weights_history = []
        self.aggregation_history = []
        
        logger.info(f"Federated Server initialized with {config.aggregation_method}")
    
    def aggregate_weights(self, clients_weights: List[Tuple[List[np.ndarray], int]]
                         ) -> List[np.ndarray]:
        """
        Aggregate weights from multiple clients using FedAvg
        
        Args:
            clients_weights: List of (weights, num_samples) tuples from each client
        
        Returns:
            Aggregated weights
        """
        if not clients_weights:
            return self.get_model_weights()
        
        # Extract weights and sample counts
        all_weights = [w for w, _ in clients_weights]
        sample_counts = [n for _, n in clients_weights]
        total_samples = sum(sample_counts)
        
        # FedAvg: weighted average by number of samples
        aggregated = []
        num_layers = len(all_weights[0])
        
        for layer_idx in range(num_layers):
            layer_weights = [w[layer_idx] for w in all_weights]
            
            # Weighted average
            weighted_sum = np.zeros_like(layer_weights[0], dtype=np.float32)
            for w, n in zip(layer_weights, sample_counts):
                weighted_sum += w * (n / total_samples)
            
            aggregated.append(weighted_sum)
        
        return aggregated
    
    def get_model_weights(self) -> List[np.ndarray]:
        """Get current global model weights"""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy())
        return weights
    
    def set_model_weights(self, weights: List[np.ndarray]):
        """Set global model weights"""
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data = torch.from_numpy(weight).to(param.device)
    
    def log_round(self, round_num: int, aggregated_weights: List[np.ndarray],
                  clients_metrics: List[Dict]):
        """Log aggregation round results"""
        round_info = {
            'round': round_num,
            'timestamp': datetime.utcnow().isoformat(),
            'num_clients': len(clients_metrics),
            'clients_metrics': clients_metrics,
            'model_magnitude': float(np.linalg.norm([w.flatten() for w in aggregated_weights]))
        }
        self.aggregation_history.append(round_info)
        
        # Log to file
        avg_loss = np.mean([m.get('loss', 0) for m in clients_metrics])
        logger.info(f"Round {round_num}: Avg Loss = {avg_loss:.4f}, "
                   f"Clients = {len(clients_metrics)}")


class FederatedFramework:
    """Main federated learning orchestrator"""
    
    def __init__(self, config: FederatedConfig):
        """Initialize federated framework"""
        self.config = config
        self.server = None
        self.clients = []
        self.execution_log = {
            'config': config.to_dict(),
            'start_time': datetime.utcnow().isoformat(),
            'rounds': []
        }
        
        logger.info(f"FederatedFramework initialized with config: {config}")
    
    def create_clients(self, data_loaders: List[DataLoader]):
        """Create federated clients with data"""
        model = FederatedEmbeddingModel()
        self.server = FederatedServer(model, self.config)
        
        for idx, loader in enumerate(data_loaders):
            client_id = f"client_{idx}"
            client_model = FederatedEmbeddingModel()
            client = FederatedClient(client_id, client_model, loader, self.config)
            self.clients.append(client)
        
        logger.info(f"Created {len(self.clients)} federated clients")
    
    def train_round(self, round_num: int) -> Dict:
        """Execute one round of federated training"""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEDERATED TRAINING ROUND {round_num}/{self.config.num_rounds}")
        logger.info(f"{'='*60}")
        
        # Get current global weights
        global_weights = self.server.get_model_weights()
        
        # Distribute to clients
        clients_metrics = []
        for client in self.clients:
            client.set_model_weights(global_weights)
            
            # Train locally
            metrics = client.train_local()
            clients_metrics.append(metrics)
        
        # Aggregate weights
        clients_weights = [
            (client.get_model_weights(), client.train_loader.dataset.__len__())
            for client in self.clients
        ]
        aggregated_weights = self.server.aggregate_weights(clients_weights)
        self.server.set_model_weights(aggregated_weights)
        
        # Log round
        self.server.log_round(round_num, aggregated_weights, clients_metrics)
        
        # Return round statistics
        avg_loss = np.mean([m['loss'] for m in clients_metrics])
        total_samples = sum(m['num_samples'] for m in clients_metrics)
        
        round_stats = {
            'round': round_num,
            'avg_loss': float(avg_loss),
            'total_samples_processed': total_samples,
            'num_clients': len(self.clients),
            'clients_metrics': clients_metrics
        }
        
        self.execution_log['rounds'].append(round_stats)
        
        return round_stats
    
    def train(self):
        """Execute full federated training"""
        logger.info(f"\nStarting Federated Training with {len(self.clients)} clients")
        logger.info(f"Total rounds: {self.config.num_rounds}")
        
        for round_num in range(1, self.config.num_rounds + 1):
            self.train_round(round_num)
        
        self.execution_log['end_time'] = datetime.utcnow().isoformat()
        logger.info("\nFederated Training Complete!")
        
        return self.get_training_summary()
    
    def get_training_summary(self) -> Dict:
        """Get training summary"""
        return {
            'config': self.config.to_dict(),
            'num_rounds': len(self.execution_log['rounds']),
            'total_samples': sum(r.get('total_samples_processed', 0) 
                                for r in self.execution_log['rounds']),
            'avg_loss_history': [r['avg_loss'] for r in self.execution_log['rounds']],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def save_checkpoint(self, output_dir: Optional[Path] = None):
        """Save federated training checkpoint"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'models' / 'federated'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save global model
        model_path = output_dir / 'global_model.pt'
        torch.save(self.server.model.state_dict(), model_path)
        
        # Save execution log
        log_path = output_dir / 'federated_training_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved to {output_dir}")


if __name__ == "__main__":
    print("Federated Learning Framework")
    print("This module is designed to be integrated with the pipeline")
    print("\nExample usage:")
    print("  config = FederatedConfig(num_clients=3, num_rounds=10)")
    print("  framework = FederatedFramework(config)")
    print("  framework.create_clients([loader1, loader2, loader3])")
    print("  framework.train()")
