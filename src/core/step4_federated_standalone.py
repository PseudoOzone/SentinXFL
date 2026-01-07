"""
STEP 4: Federated GPT-2 LoRA Fine-tuning - Standalone Execution
Direct federated training without pipeline dependencies
"""

import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import logging
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step4_federated_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FraudNarrativeDataset(Dataset):
    """Dataset for GPT-2 training"""
    
    def __init__(self, narratives: List[str], tokenizer, max_length=256):
        self.narratives = narratives
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.narratives)
    
    def __getitem__(self, idx):
        narrative = self.narratives[idx]
        encoding = self.tokenizer(
            narrative,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class FederatedClient:
    """Single federated client (bank)"""
    
    def __init__(self, client_id: str, model, tokenizer, train_loader, device):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.device = device
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        
    def train_local(self, epochs: int = 1) -> Dict:
        """Train locally for specified epochs"""
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if (batch_idx + 1) % 20 == 0:
                    logger.info(f"{self.client_id} Epoch {epoch+1}/{epochs} Batch {batch_idx+1} Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(self.train_loader)
            logger.info(f"{self.client_id} Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            total_loss += avg_loss
        
        return {
            'client_id': self.client_id,
            'avg_loss': total_loss / epochs,
            'num_samples': len(self.train_loader.dataset)
        }
    
    def get_weights(self) -> np.ndarray:
        """Extract LoRA weights"""
        weights = []
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights) if weights else np.array([])
    
    def set_weights(self, weights: np.ndarray):
        """Set LoRA weights"""
        with torch.no_grad():
            offset = 0
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param_size = param.numel()
                    param.data = torch.from_numpy(
                        weights[offset:offset + param_size].reshape(param.shape)
                    ).to(param.device).float()
                    offset += param_size


class FederatedServer:
    """Server for model aggregation"""
    
    def __init__(self, model):
        self.model = model
        self.history = []
    
    def aggregate_weights(self, client_weights: List[Tuple[np.ndarray, int]]) -> np.ndarray:
        """FedAvg aggregation"""
        all_weights = [w for w, _ in client_weights]
        sample_counts = [n for _, n in client_weights]
        total_samples = sum(sample_counts)
        
        aggregated = np.zeros_like(all_weights[0], dtype=np.float32)
        for w, n in zip(all_weights, sample_counts):
            aggregated += w * (n / total_samples)
        
        return aggregated
    
    def set_weights(self, weights: np.ndarray):
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
    
    def get_weights(self) -> np.ndarray:
        """Get global model weights"""
        weights = []
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights) if weights else np.array([])
    
    def log_round(self, round_num: int, num_clients: int, avg_loss: float):
        """Log federation round"""
        round_info = {
            'round': round_num,
            'timestamp': datetime.utcnow().isoformat(),
            'num_clients': num_clients,
            'avg_loss': float(avg_loss)
        }
        self.history.append(round_info)
        logger.info(f"Round {round_num}: {num_clients} clients, Avg Loss: {avg_loss:.4f}")


def run_federated_training(num_clients: int = 3, num_rounds: int = 3, 
                           local_epochs: int = 2, batch_size: int = 8):
    """Execute federated training"""
    
    logger.info("="*100)
    logger.info("STEP 4: FEDERATED GPT-2 LORA FINE-TUNING STARTED")
    logger.info(f"Configuration: {num_clients} banks, {num_rounds} rounds, {local_epochs} local epochs")
    logger.info("="*100)
    
    # Setup device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! RTX 3050 required.")
    
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load narratives
    narrative_file = Path('generated/fraud_narratives_combined.csv')
    if not narrative_file.exists():
        raise FileNotFoundError(f"Narratives not found: {narrative_file}")
    
    logger.info(f"Loading narratives from {narrative_file}")
    df = pd.read_csv(narrative_file)
    narratives = df['narrative'].tolist()
    logger.info(f"Loaded {len(narratives)} narratives")
    
    # Create global model
    logger.info("Creating GPT-2 with LoRA adapters...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['c_attn']
    )
    
    global_model = get_peft_model(base_model, lora_config)
    global_model.to(device)
    logger.info("[OK] Global model created with LoRA adapters")
    
    # Create server
    server = FederatedServer(global_model)
    
    # Split data for clients
    samples_per_client = len(narratives) // num_clients
    client_data = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = len(narratives) if i == num_clients - 1 else (i + 1) * samples_per_client
        client_data.append(narratives[start:end])
        logger.info(f"Bank {i+1}: {len(client_data[-1])} samples")
    
    # Create clients
    clients = []
    for i, data in enumerate(client_data):
        client_id = f"bank_{i+1:03d}"
        dataset = FraudNarrativeDataset(data, tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create new base model for each client
        base_client_model = GPT2LMHeadModel.from_pretrained('gpt2')
        client_model = get_peft_model(base_client_model, lora_config)
        
        client = FederatedClient(client_id, client_model, tokenizer, loader, device)
        clients.append(client)
    
    logger.info(f"[OK] Created {len(clients)} federated clients")
    
    # Federated training rounds
    logger.info("\n" + "="*100)
    logger.info("STARTING FEDERATION ROUNDS")
    logger.info("="*100)
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"FEDERATION ROUND {round_num}/{num_rounds}")
        logger.info(f"{'='*60}")
        
        # Distribute global model
        global_weights = server.get_weights()
        for client in clients:
            client.set_weights(global_weights)
        
        # Local training
        client_weights = []
        round_losses = []
        
        for client in clients:
            logger.info(f"\n{client.client_id} training...")
            metrics = client.train_local(epochs=local_epochs)
            
            client_weights.append((client.get_weights(), client.train_loader.dataset.__len__()))
            round_losses.append(metrics['avg_loss'])
            
            logger.info(f"{client.client_id} completed - Loss: {metrics['avg_loss']:.4f}")
        
        # Server aggregation
        logger.info(f"\nAggregating {len(clients)} client models...")
        aggregated_weights = server.aggregate_weights(client_weights)
        server.set_weights(aggregated_weights)
        
        avg_round_loss = np.mean(round_losses)
        server.log_round(round_num, len(clients), avg_round_loss)
        
        logger.info(f"\nRound {round_num} Summary:")
        logger.info(f"  - Clients: {len(clients)}")
        logger.info(f"  - Avg loss: {avg_round_loss:.4f}")
        logger.info(f"  - Loss range: [{min(round_losses):.4f}, {max(round_losses):.4f}]")
    
    # Save model
    logger.info("\n" + "="*100)
    logger.info("[SAVING] FEDERATED MODEL")
    logger.info("="*100)
    
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Save LoRA adapter
    adapter_dir = model_dir / 'fraud_pattern_generator_lora_federated'
    global_model.save_pretrained(str(adapter_dir))
    logger.info(f"[OK] Saved LoRA adapter: {adapter_dir}")
    
    # Save tokenizer
    tokenizer_dir = model_dir / 'gpt2_tokenizer_federated'
    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info(f"[OK] Saved tokenizer: {tokenizer_dir}")
    
    # Save aggregation history
    history_file = model_dir / 'federated_aggregation_history.json'
    with open(history_file, 'w') as f:
        json.dump(server.history, f, indent=2)
    logger.info(f"[OK] Saved aggregation history: {history_file}")
    
    logger.info("\n" + "="*100)
    logger.info("[SUCCESS] FEDERATED TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*100)
    logger.info(f"\n[RESULTS]:")
    logger.info(f"  - Total rounds: {num_rounds}")
    logger.info(f"  - Total clients: {num_clients}")
    logger.info(f"  - Final avg loss: {server.history[-1]['avg_loss']:.4f}")
    logger.info(f"  - Model location: {adapter_dir}")
    
    return True


if __name__ == "__main__":
    try:
        success = run_federated_training(
            num_clients=3,
            num_rounds=3,
            local_epochs=2,
            batch_size=8
        )
        
        if success:
            logger.info("\n[SUCCESS] Training pipeline executed successfully!")
            sys.exit(0)
        else:
            logger.error("\n[ERROR] Training failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
