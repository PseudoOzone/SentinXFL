"""
Fraud Detection Inference API - Quick Integration
Enables fraud detection inference using trained models
Works with both single and federated models
Integrated with Llama for enhanced explanations and pattern analysis
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

# Try to import Llama integration (optional, with graceful fallback)
try:
    from ollama_integration import LlamaFraudAnalyzer
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    LlamaFraudAnalyzer = None

logger = logging.getLogger(__name__)


class FraudDetectionInference:
    """Inference engine for fraud detection"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load embedding model if available"""
        embedding_path = self.model_dir / 'fraud_embedding_model.pt'
        
        if not embedding_path.exists():
            logger.warning(f"Embedding model not found: {embedding_path}")
            return False
        
        try:
            self.embedding_model = torch.load(embedding_path, map_location=self.device)
            self.embedding_model.eval()
            logger.info(f"✅ Loaded embedding model from {embedding_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def extract_embeddings(self, narratives: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for narratives
        
        Args:
            narratives: List of text narratives
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (N, 768)
        """
        if self.embedding_model is None:
            logger.warning("Embedding model not loaded - cannot extract embeddings")
            return None
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(narratives), batch_size):
                batch = narratives[i:i + batch_size]
                
                # Simple tokenization (word-level for demo)
                tokens = [len(text.split()) for text in batch]
                max_len = max(tokens) if tokens else 0
                
                # Create dummy input for embedding extraction
                input_ids = torch.randint(0, 30522, (len(batch), min(max_len, 256)))
                input_ids = input_ids.to(self.device)
                
                # Extract embeddings
                outputs = self.embedding_model(input_ids)
                batch_embeddings = outputs.cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings) if embeddings else None
    
    def detect_fraud_similarity(self, transaction_narrative: str, 
                               reference_embeddings: np.ndarray,
                               threshold: float = 0.7) -> Dict:
        """
        Detect fraud using embedding similarity
        
        Args:
            transaction_narrative: Description of transaction
            reference_embeddings: Pre-computed fraud embeddings
            threshold: Similarity threshold for fraud flag
            
        Returns:
            Dictionary with fraud score and risk assessment
        """
        if self.embedding_model is None or reference_embeddings is None:
            return {
                'fraud_score': 0.0,
                'risk_level': 'UNKNOWN',
                'status': 'Model not loaded'
            }
        
        try:
            # Extract embedding for transaction
            tx_embedding = self.extract_embeddings([transaction_narrative])
            
            if tx_embedding is None:
                return {
                    'fraud_score': 0.0,
                    'risk_level': 'ERROR',
                    'status': 'Failed to extract embedding'
                }
            
            # Compute similarity to fraud patterns
            tx_emb = tx_embedding[0]  # Take first (only) embedding
            
            # Cosine similarity
            similarities = np.dot(reference_embeddings, tx_emb) / (
                np.linalg.norm(reference_embeddings, axis=1) * 
                np.linalg.norm(tx_emb) + 1e-8
            )
            
            max_similarity = float(np.max(similarities))
            mean_similarity = float(np.mean(similarities))
            
            # Risk assessment
            if max_similarity > threshold:
                risk_level = 'HIGH'
            elif max_similarity > (threshold - 0.2):
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'fraud_score': max_similarity,
                'mean_similarity': mean_similarity,
                'risk_level': risk_level,
                'threshold': threshold,
                'status': 'Success'
            }
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'fraud_score': 0.0,
                'risk_level': 'ERROR',
                'status': str(e)
            }


class FederatedModelClient:
    """Client for accessing federated models"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.federated_model_path = self.model_dir / 'fraud_pattern_generator_lora_federated'
        self.single_model_path = self.model_dir / 'fraud_pattern_generator_lora'
    
    def list_available_models(self) -> Dict:
        """List available trained models"""
        models = {
            'federated': {
                'available': self.federated_model_path.exists(),
                'path': str(self.federated_model_path) if self.federated_model_path.exists() else None
            },
            'single_bank': {
                'available': self.single_model_path.exists(),
                'path': str(self.single_model_path) if self.single_model_path.exists() else None
            }
        }
        
        # Check aggregation history
        history_file = self.model_dir / 'federated_aggregation_history.json'
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                    models['federated']['rounds_completed'] = len(history)
                    if history:
                        models['federated']['last_round_loss'] = history[-1]['avg_loss']
            except:
                pass
        
        return models
    
    def get_model_stats(self) -> Dict:
        """Get statistics about trained models"""
        models = self.list_available_models()
        
        stats = {
            'timestamp': str(Path(self.model_dir).stat().st_mtime),
            'models': models,
            'inference_engine': 'FraudDetectionInference',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        return stats


class SimpleInferencePipeline:
    """Simple pipeline for fraud detection inference with Llama enhancement"""
    
    def __init__(self, model_dir='models', use_llama: bool = True):
        self.inference = FraudDetectionInference(model_dir)
        self.client = FederatedModelClient(model_dir)
        self.use_llama = use_llama and LLAMA_AVAILABLE
        
        # Initialize Llama if available
        if self.use_llama:
            try:
                self.analyzer = LlamaFraudAnalyzer()
                logger.info("✅ Llama integration initialized for enhanced explanations")
            except Exception as e:
                logger.warning(f"Llama initialization failed: {e} - running without Llama")
                self.use_llama = False
        else:
            self.analyzer = None
    
    def analyze_transaction(self, transaction_data: Dict) -> Dict:
        """
        Analyze a transaction for fraud risk with optional Llama enhancement
        
        Args:
            transaction_data: Dictionary with transaction details
                - 'narrative': Description of transaction
                - 'amount': Transaction amount (optional)
                - 'type': Transaction type (optional)
        
        Returns:
            Fraud analysis result with Llama explanation if available
        """
        narrative = transaction_data.get('narrative', '')
        
        # For demo, use random reference embeddings
        # In production, would load pre-computed fraud embeddings
        reference_embeddings = np.random.randn(100, 768)  # Dummy embeddings
        
        result = self.inference.detect_fraud_similarity(
            narrative,
            reference_embeddings,
            threshold=0.65
        )
        
        # Add transaction data
        result['transaction'] = {
            'amount': transaction_data.get('amount'),
            'type': transaction_data.get('type'),
            'narrative': narrative[:100] + ('...' if len(narrative) > 100 else '')
        }
        
        # Enhance with Llama if available
        if self.use_llama and self.analyzer:
            try:
                fraud_score = result.get('fraud_score', 0)
                
                # Get analysis from Llama
                analysis = self.analyzer.analyze_fraud_pattern(narrative)
                result['llama_analysis'] = analysis
                
                result['llama_enhanced'] = True
                
            except Exception as e:
                logger.debug(f"Llama enhancement failed: {e} - continuing with base result")
                result['llama_enhanced'] = False
        else:
            result['llama_enhanced'] = False
        
        return result


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    pipeline = SimpleInferencePipeline()
    
    # Check available models
    print("📊 Available Models:")
    print(json.dumps(pipeline.client.list_available_models(), indent=2))
    
    # Test inference (will use dummy embeddings for now)
    test_transaction = {
        'narrative': 'Large wire transfer to unknown account in offshore jurisdiction',
        'amount': 50000,
        'type': 'Wire Transfer'
    }
    
    print("\n🔍 Inference Test:")
    result = pipeline.analyze_transaction(test_transaction)
    print(json.dumps(result, indent=2))
