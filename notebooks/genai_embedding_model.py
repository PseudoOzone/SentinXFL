"""
Step 3: Fraud Embedding Model
Trains DistilBERT model on fraud narratives with GPU support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from typing import Tuple, List


class FraudNarrativeDataset(Dataset):
    """Custom Dataset for fraud narratives"""
    
    def __init__(self, narratives: List[str], labels: List[int], tokenizer, max_length=256):
        self.narratives = narratives
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.narratives)
    
    def __getitem__(self, idx):
        narrative = self.narratives[idx]
        label = self.labels[idx]
        
        # Tokenize
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
            'label': torch.tensor(label, dtype=torch.long)
        }


class FraudEmbeddingModel(nn.Module):
    """DistilBERT-based model for fraud detection with embedding extraction"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        super(FraudEmbeddingModel, self).__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.embedding_dim = self.distilbert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            logits, embeddings
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        
        return logits, cls_embedding


class EmbeddingTrainer:
    """Trains the fraud embedding model"""
    
    def __init__(self, model_name='distilbert-base-uncased', device=None):
        self.logger = logging.getLogger(__name__)
        
        # Device detection - Force CUDA on RTX 3050
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')  # Force GPU 0
                torch.cuda.set_device(0)
            else:
                raise RuntimeError("CUDA not available! RTX 3050 not detected. Please check GPU drivers.")
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Print GPU info
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            # Enable memory optimization for 6GB VRAM RTX 3050
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cache cleared for optimization")
        
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        # Initialize model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = FraudEmbeddingModel(model_name=model_name)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader
            epoch: Epoch number
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch} [{batch_idx + 1}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def get_embeddings(self, narratives: List[str], batch_size=32):
        """
        Get embeddings for narratives
        
        Args:
            narratives: List of narrative strings
            batch_size: Batch size
            
        Returns:
            numpy array of embeddings
        """
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(narratives), batch_size):
                batch = narratives[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch,
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                _, embeddings = self.model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def save_model(self, output_dir):
        """
        Save model and tokenizer
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / 'fraud_embedding_model.pt'
        torch.save(self.model.state_dict(), model_file)
        self.logger.info(f"Saved model: {model_file}")
        
        # Save tokenizer
        tokenizer_dir = output_path / 'embedding_tokenizer'
        self.tokenizer.save_pretrained(str(tokenizer_dir))
        self.logger.info(f"Saved tokenizer: {tokenizer_dir}")


class EmbeddingPipeline:
    """Orchestrates embedding model training"""
    
    def __init__(self, data_dir='generated', model_dir='models'):
        # Resolve paths relative to parent directory (project root)
        current_dir = Path(__file__).parent
        self.data_dir = (current_dir.parent / data_dir).resolve()
        self.model_dir = (current_dir.parent / model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Device setup - Force CUDA on RTX 3050
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        else:
            raise RuntimeError("CUDA not available! RTX 3050 not detected. Please check GPU drivers.")
    
    def run(self, input_file='fraud_narratives_combined.csv', epochs=3, batch_size=16):
        """
        Execute embedding training pipeline
        
        Args:
            input_file: Narratives CSV file
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Path to embeddings pickle file
        """
        try:
            # Load narratives
            input_path = self.data_dir / input_file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            self.logger.info(f"Loading narratives from {input_path}")
            df = pd.read_csv(input_path)
            
            narratives = df['narrative'].tolist()
            labels = df['fraud_label'].tolist()
            
            self.logger.info(f"Loaded {len(narratives)} narratives")
            
            # Initialize trainer
            trainer = EmbeddingTrainer(device=self.device)
            
            # Create dataset and dataloader
            dataset = FraudNarrativeDataset(narratives, labels, trainer.tokenizer)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Train model
            self.logger.info(f"Training model for {epochs} epochs with batch_size={batch_size}")
            for epoch in range(1, epochs + 1):
                avg_loss = trainer.train_epoch(train_loader, epoch)
                self.logger.info(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")
            
            # Get embeddings
            self.logger.info("Extracting embeddings for all narratives")
            embeddings = trainer.get_embeddings(narratives, batch_size=batch_size)
            
            # Save embeddings
            embeddings_file = self.data_dir / 'fraud_embeddings.pkl'
            with open(embeddings_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'labels': labels,
                    'narratives': narratives
                }, f)
            self.logger.info(f"Saved embeddings: {embeddings_file}")
            
            # Save model
            trainer.save_model(self.model_dir)
            
            return str(embeddings_file)
            
        except Exception as e:
            self.logger.error(f"Error in embedding pipeline: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = EmbeddingPipeline()
    # This will run after Step 2 is complete
    output_path = pipeline.run()
    print(f"Embeddings saved: {output_path}")
