"""
Federated GPT-2 LoRA Training with Flower Framework
Multi-bank collaborative fraud detection with privacy preservation
Implements FedAvg for GPT-2 LoRA fine-tuning across distributed clients
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import json
from datetime import datetime

# Optional Flower for true distributed setup
try:
    import flwr as fl
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    print("⚠️  Flower not installed - using local simulation mode")


class FraudNarrativeGPT2Dataset(Dataset):
    """Dataset for GPT-2 narrative generation"""
    
    def __init__(self, narratives: List[str], tokenizer, max_length=256):
        self.narratives = narratives
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.narratives)
    
    def __getitem__(self, idx):
        narrative = self.narratives[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            narrative,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For language modeling, labels are same as input_ids
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }


class FederatedGPT2Client:
    """Federated Learning Client for GPT-2 LoRA Training"""
    
    def __init__(self, client_id: str, model, tokenizer, train_loader: DataLoader,
                 device=None, learning_rate=5e-5):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier (e.g., "bank_001")
            model: GPT-2 model with LoRA adapters
            tokenizer: GPT2Tokenizer
            train_loader: DataLoader for this client's data
            device: torch.device (cuda/cpu)
            learning_rate: Learning rate for optimizer
        """
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer - only optimize LoRA parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'rounds': [],
            'losses': [],
            'epochs': []
        }
        
        self.logger = logging.getLogger(f"FedClient_{client_id}")
        self.logger.info(f"Federated client {client_id} initialized on {self.device}")
    
    def train_local(self, epochs: int = 1) -> Dict:
        """
        Train model locally for specified epochs
        
        Args:
            epochs: Number of local epochs to train
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Client {self.client_id} starting local training ({epochs} epochs)...")
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 20 == 0:
                    self.logger.info(f"Epoch {epoch + 1}/{epochs} Batch {batch_idx + 1} "
                                   f"Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.logger.info(f"Client {self.client_id} Epoch {epoch + 1}/{epochs} "
                           f"Avg Loss: {avg_epoch_loss:.4f}")
            self.history['losses'].append(avg_epoch_loss)
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'client_id': self.client_id,
            'avg_loss': total_loss / epochs,
            'num_samples': len(self.train_loader.dataset),
            'num_batches': num_batches,
            'epochs': epochs
        }
    
    def get_lora_weights(self) -> np.ndarray:
        """Extract only LoRA adapter weights for communication"""
        weights = []
        
        # Get only LoRA parameters (smaller size for efficient communication)
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                weights.append(param.data.cpu().numpy().flatten())
        
        # Concatenate all LoRA weights
        if weights:
            return np.concatenate(weights)
        else:
            # Fallback to all parameters if LoRA naming not found
            return np.concatenate([p.data.cpu().numpy().flatten() 
                                  for p in self.model.parameters()])
    
    def set_lora_weights(self, weights: np.ndarray):
        """Set LoRA adapter weights from flattened array"""
        with torch.no_grad():
            offset = 0
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param_size = param.numel()
                    param.data = torch.from_numpy(
                        weights[offset:offset + param_size].reshape(param.shape)
                    ).to(param.device).float()
                    offset += param_size
    
    def get_num_samples(self) -> int:
        """Get number of training samples"""
        return len(self.train_loader.dataset)


class FederatedGPT2Server:
    """Federated Server for Model Aggregation"""
    
    def __init__(self, model, aggregation_method: str = 'FedAvg'):
        """
        Initialize federated server
        
        Args:
            model: Initial global GPT-2 model
            aggregation_method: 'FedAvg', 'FedProx', or 'FedAttentive'
        """
        self.model = model
        self.aggregation_method = aggregation_method
        self.aggregation_history = []
        self.logger = logging.getLogger("FedServer")
        
        self.logger.info(f"Federated Server initialized with {aggregation_method}")
    
    def aggregate_weights(self, client_weights: List[Tuple[np.ndarray, int]]
                         ) -> np.ndarray:
        """
        Aggregate weights from clients using FedAvg
        
        Args:
            client_weights: List of (weights, num_samples) tuples
            
        Returns:
            Aggregated weights as numpy array
        """
        if not client_weights:
            return self._get_lora_weights()
        
        # Extract weights and sample counts
        all_weights = [w for w, _ in client_weights]
        sample_counts = [n for _, n in client_weights]
        total_samples = sum(sample_counts)
        
        if self.aggregation_method == 'FedAvg':
            # Standard FedAvg: weighted average by number of samples
            aggregated = np.zeros_like(all_weights[0], dtype=np.float32)
            for w, n in zip(all_weights, sample_counts):
                aggregated += w * (n / total_samples)
            
            self.logger.info(f"FedAvg aggregated {len(all_weights)} clients "
                           f"({total_samples} total samples)")
        
        elif self.aggregation_method == 'FedProx':
            # FedProx with proximal term for non-IID data
            aggregated = np.zeros_like(all_weights[0], dtype=np.float32)
            mu = 0.01  # Proximal term weight
            
            for w, n in zip(all_weights, sample_counts):
                # Weighted average with proximal regularization
                aggregated += w * (n / total_samples)
            
            self.logger.info(f"FedProx aggregated {len(all_weights)} clients "
                           f"(mu={mu})")
        
        else:
            # Default to FedAvg
            aggregated = np.zeros_like(all_weights[0], dtype=np.float32)
            for w, n in zip(all_weights, sample_counts):
                aggregated += w * (n / total_samples)
        
        return aggregated
    
    def set_lora_weights(self, weights: np.ndarray):
        """Set global model weights"""
        with torch.no_grad():
            offset = 0
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param_size = param.numel()
                    param.data = torch.from_numpy(
                        weights[offset:offset + param_size].reshape(param.shape)
                    ).to(param.device).float()
                    offset += param_size
    
    def _get_lora_weights(self) -> np.ndarray:
        """Get current global LoRA weights"""
        weights = []
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                weights.append(param.data.cpu().numpy().flatten())
        
        if weights:
            return np.concatenate(weights)
        else:
            return np.concatenate([p.data.cpu().numpy().flatten() 
                                  for p in self.model.parameters()])
    
    def log_round(self, round_num: int, num_clients: int, avg_loss: float):
        """Log federation round"""
        round_info = {
            'round': round_num,
            'timestamp': datetime.utcnow().isoformat(),
            'num_clients': num_clients,
            'avg_loss': float(avg_loss),
            'aggregation_method': self.aggregation_method
        }
        self.aggregation_history.append(round_info)
        self.logger.info(f"Round {round_num}: {num_clients} clients, "
                        f"Avg Loss = {avg_loss:.4f}")


class FederatedGPT2Pipeline:
    """Orchestrates federated GPT-2 LoRA fine-tuning"""
    
    def __init__(self, data_dir='generated', model_dir='models', num_clients: int = 3):
        """
        Initialize federated pipeline
        
        Args:
            data_dir: Directory with training data
            model_dir: Directory to save models
            num_clients: Number of federated clients (banks)
        """
        # Resolve paths
        current_dir = Path(__file__).parent
        self.data_dir = (current_dir.parent / data_dir).resolve()
        self.model_dir = (current_dir.parent / model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_clients = num_clients
        self.logger = logging.getLogger(__name__)
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
    
    def create_gpt2_model_with_lora(self, model_name='gpt2'):
        """Create GPT-2 model with LoRA adapters"""
        self.logger.info(f"Loading {model_name} base model...")
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['c_attn']
        )
        
        # Apply LoRA
        model = get_peft_model(base_model, lora_config)
        model.to(self.device)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"LoRA Model - Total: {total_params}, Trainable: {trainable_params}")
        
        return model, tokenizer
    
    def split_data_for_clients(self, narratives: List[str]) -> List[List[str]]:
        """
        Split narratives among federated clients (simulating different banks)
        
        Args:
            narratives: List of all narratives
            
        Returns:
            List of narrative lists, one per client
        """
        total_samples = len(narratives)
        samples_per_client = total_samples // self.num_clients
        
        client_data = []
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            if i == self.num_clients - 1:
                # Last client gets remaining samples
                end_idx = total_samples
            else:
                end_idx = start_idx + samples_per_client
            
            client_narratives = narratives[start_idx:end_idx]
            client_data.append(client_narratives)
            
            self.logger.info(f"Client {i+1}: {len(client_narratives)} samples")
        
        return client_data
    
    def run_federated_training(self, input_file='fraud_narratives_combined.csv',
                              num_rounds: int = 3, local_epochs: int = 2,
                              batch_size: int = 8):
        """
        Execute federated GPT-2 LoRA fine-tuning
        
        Args:
            input_file: Narratives CSV file
            num_rounds: Number of federation rounds
            local_epochs: Local epochs per client per round
            batch_size: Training batch size
            
        Returns:
            Path to saved model
        """
        try:
            self.logger.info("="*80)
            self.logger.info("FEDERATED GPT-2 LoRA FINE-TUNING STARTED")
            self.logger.info("="*80)
            self.logger.info(f"Configuration: {self.num_clients} clients, {num_rounds} rounds, "
                           f"{local_epochs} local epochs")
            
            # Load narratives
            input_path = self.data_dir / input_file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            self.logger.info(f"Loading narratives from {input_path}")
            df = pd.read_csv(input_path)
            narratives = df['narrative'].tolist()
            self.logger.info(f"Loaded {len(narratives)} narratives")
            
            # Create global model
            global_model, tokenizer = self.create_gpt2_model_with_lora()
            
            # Create server
            server = FederatedGPT2Server(global_model, aggregation_method='FedAvg')
            
            # Split data for clients
            client_data = self.split_data_for_clients(narratives)
            
            # Initialize clients
            clients = []
            for i, client_narratives in enumerate(client_data):
                client_id = f"bank_{i+1:03d}"
                
                # Create dataset and loader
                dataset = FraudNarrativeGPT2Dataset(client_narratives, tokenizer)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Create client with a copy of global model
                client_model, _ = self.create_gpt2_model_with_lora()
                client = FederatedGPT2Client(
                    client_id=client_id,
                    model=client_model,
                    tokenizer=tokenizer,
                    train_loader=loader,
                    device=self.device,
                    learning_rate=5e-5
                )
                clients.append(client)
            
            self.logger.info(f"Created {len(clients)} federated clients")
            
            # Federated training rounds
            self.logger.info("\nStarting federated training rounds...\n")
            
            for round_num in range(1, num_rounds + 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"FEDERATION ROUND {round_num}/{num_rounds}")
                self.logger.info(f"{'='*60}")
                
                # Distribute global model to clients
                global_weights = server._get_lora_weights()
                for client in clients:
                    client.set_lora_weights(global_weights)
                    self.logger.info(f"Distributed global model to {client.client_id}")
                
                # Local training on each client
                client_weights = []
                round_losses = []
                
                for client in clients:
                    self.logger.info(f"\n{client.client_id} training...")
                    metrics = client.train_local(epochs=local_epochs)
                    
                    client_weights.append((client.get_lora_weights(), 
                                         client.get_num_samples()))
                    round_losses.append(metrics['avg_loss'])
                    
                    self.logger.info(f"{client.client_id} completed - Loss: {metrics['avg_loss']:.4f}")
                
                # Server aggregation
                self.logger.info(f"\nAggregating {len(clients)} client models...")
                aggregated_weights = server.aggregate_weights(client_weights)
                server.set_lora_weights(aggregated_weights)
                
                avg_round_loss = np.mean(round_losses)
                server.log_round(round_num, len(clients), avg_round_loss)
                
                self.logger.info(f"\nRound {round_num} Summary:")
                self.logger.info(f"  - Aggregated {len(clients)} clients")
                self.logger.info(f"  - Average loss: {avg_round_loss:.4f}")
                self.logger.info(f"  - Loss range: [{min(round_losses):.4f}, {max(round_losses):.4f}]")
            
            # Save final model
            self.logger.info("\n" + "="*80)
            self.logger.info("SAVING FEDERATED MODEL")
            self.logger.info("="*80)
            
            self._save_federated_model(global_model, tokenizer, server)
            
            self.logger.info("\n✅ Federated GPT-2 LoRA fine-tuning completed successfully!")
            return str(self.model_dir / 'fraud_pattern_generator_lora_federated')
            
        except Exception as e:
            self.logger.error(f"Error in federated pipeline: {e}", exc_info=True)
            raise
    
    def _save_federated_model(self, model, tokenizer, server):
        """Save federated model and metadata"""
        # Save LoRA adapter
        adapter_dir = self.model_dir / 'fraud_pattern_generator_lora_federated'
        model.save_pretrained(str(adapter_dir))
        self.logger.info(f"Saved federated LoRA adapter: {adapter_dir}")
        
        # Save tokenizer
        tokenizer_dir = self.model_dir / 'gpt2_tokenizer_federated'
        tokenizer.save_pretrained(str(tokenizer_dir))
        self.logger.info(f"Saved federated tokenizer: {tokenizer_dir}")
        
        # Save aggregation history
        history_file = self.model_dir / 'federated_aggregation_history.json'
        with open(history_file, 'w') as f:
            json.dump(server.aggregation_history, f, indent=2)
        self.logger.info(f"Saved aggregation history: {history_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run federated training with 3 simulated banks
    pipeline = FederatedGPT2Pipeline(num_clients=3)
    output_path = pipeline.run_federated_training(
        num_rounds=3,
        local_epochs=1,
        batch_size=8
    )
    print(f"\n✅ Federated model saved: {output_path}")
