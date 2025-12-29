"""
Ollama Integration for Fraud Detection System
- Manages Ollama service and model lifecycle
- Integrates Llama for fraud pattern generation and monitoring
- Provides real-time fraud detection with LLM augmentation
"""

import subprocess
import requests
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaService:
    """Manages Ollama service lifecycle"""
    
    def __init__(self):
        self.ollama_path = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
        self.api_url = "http://localhost:11434"
        self.process = None
        self.is_running = False
        
    def start(self) -> bool:
        """Start Ollama service"""
        try:
            if self.is_running:
                logger.info("✓ Ollama already running")
                return True
            
            logger.info("🚀 Starting Ollama service...")
            self.process = subprocess.Popen(
                str(self.ollama_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for service to be ready
            for attempt in range(30):
                try:
                    response = requests.get(f"{self.api_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        self.is_running = True
                        logger.info("✅ Ollama service started successfully")
                        return True
                except:
                    time.sleep(1)
                    
            logger.error("❌ Ollama service failed to start")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error starting Ollama: {e}")
            return False
    
    def stop(self):
        """Stop Ollama service"""
        if self.process:
            self.process.terminate()
            self.is_running = False
            logger.info("⏹️ Ollama service stopped")
    
    def pull_model(self, model_name: str = "llama2") -> bool:
        """Download model from Ollama registry"""
        try:
            logger.info(f"📥 Pulling {model_name}...")
            cmd = [str(self.ollama_path), "pull", model_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"✅ {model_name} pulled successfully")
                return True
            else:
                logger.error(f"❌ Failed to pull {model_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pulling model: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


class LlamaFraudAnalyzer:
    """Uses Llama to analyze fraud patterns and generate randomizer training data"""
    
    def __init__(self, model_name: str = "llama2", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        self.context_history = []
        
    def generate_fraud_narrative(self, transaction_data: Dict) -> str:
        """Generate realistic fraud narrative using Llama"""
        prompt = f"""Based on this transaction data, generate a realistic fraud scenario narrative:
        
Transaction: {json.dumps(transaction_data, indent=2)}

Generate a concise fraud narrative explaining how this transaction exhibits fraudulent patterns. 
Format: "Fraud Pattern: [pattern type]. Risk Indicators: [indicators]. Severity: [high/medium/low]"
"""
        
        return self._call_llama(prompt, max_tokens=150)
    
    def analyze_fraud_pattern(self, transaction_narrative: str) -> Dict:
        """Analyze transaction narrative and identify fraud patterns"""
        prompt = f"""Analyze this transaction narrative for fraud patterns:

"{transaction_narrative}"

Identify:
1. Fraud Type (if any)
2. Risk Level (high/medium/low)
3. Key Risk Indicators
4. Confidence Score (0-1)

Format as JSON."""
        
        response = self._call_llama(prompt, max_tokens=200)
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except:
            pass
        
        return {
            'fraud_type': 'unknown',
            'risk_level': 'medium',
            'indicators': [],
            'confidence': 0.5
        }
    
    def generate_randomizer_training(self, fraud_df: pd.DataFrame, num_samples: int = 100) -> List[Dict]:
        """Generate randomizer training data from real fraud patterns"""
        logger.info(f"🔄 Generating {num_samples} randomizer training samples...")
        
        training_samples = []
        
        # Sample fraud cases
        sample_indices = np.random.choice(len(fraud_df), min(num_samples, len(fraud_df)), replace=False)
        
        for idx in sample_indices:
            row = fraud_df.iloc[idx]
            
            # Convert row to dict
            transaction_data = row.to_dict()
            
            # Generate fraud narrative
            narrative = self.generate_fraud_narrative(transaction_data)
            
            # Analyze pattern
            analysis = self.analyze_fraud_pattern(narrative)
            
            training_sample = {
                'original_data': transaction_data,
                'generated_narrative': narrative,
                'pattern_analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            training_samples.append(training_sample)
            
            if (len(training_samples) % 10) == 0:
                logger.info(f"  Generated {len(training_samples)}/{num_samples} samples")
        
        logger.info(f"✅ Generated {len(training_samples)} training samples")
        return training_samples
    
    def _call_llama(self, prompt: str, max_tokens: int = 200) -> str:
        """Call Llama model via Ollama API"""
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': max_tokens,
                        'temperature': 0.7,
                        'top_p': 0.9,
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Llama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Llama: {e}")
            return ""


class LlamaMonitoringSystem:
    """Real-time fraud detection with Llama monitoring"""
    
    def __init__(self, analyzer: LlamaFraudAnalyzer, model_dir: Path = Path('models')):
        self.analyzer = analyzer
        self.model_dir = model_dir
        self.monitoring_log = []
        self.accuracy_history = []
        self.detection_threshold = 0.65
        
    def monitor_transaction(self, transaction_narrative: str, 
                          embedding_score: Optional[float] = None) -> Dict:
        """Monitor single transaction with Llama + embedding analysis"""
        
        # Get Llama analysis
        llama_analysis = self.analyzer.analyze_fraud_pattern(transaction_narrative)
        
        # Combine with embedding score if available
        combined_score = self._combine_scores(llama_analysis, embedding_score)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'narrative': transaction_narrative,
            'llama_analysis': llama_analysis,
            'embedding_score': embedding_score,
            'combined_fraud_score': combined_score,
            'is_fraud': combined_score > self.detection_threshold,
            'confidence': llama_analysis.get('confidence', 0.5)
        }
        
        self.monitoring_log.append(result)
        return result
    
    def batch_monitor(self, transactions: List[str], 
                     embedding_scores: Optional[List[float]] = None) -> List[Dict]:
        """Monitor batch of transactions"""
        results = []
        
        for i, narrative in enumerate(transactions):
            embedding_score = embedding_scores[i] if embedding_scores else None
            result = self.monitor_transaction(narrative, embedding_score)
            results.append(result)
        
        return results
    
    def evaluate_accuracy(self, monitoring_results: List[Dict], 
                         ground_truth: List[int]) -> Dict:
        """Evaluate monitoring accuracy and update threshold"""
        
        if len(monitoring_results) != len(ground_truth):
            logger.warning("Results and ground truth length mismatch")
            return {}
        
        predictions = [1 if r['is_fraud'] else 0 for r in monitoring_results]
        
        # Calculate metrics
        tp = sum((p == 1) and (g == 1) for p, g in zip(predictions, ground_truth))
        fp = sum((p == 1) and (g == 0) for p, g in zip(predictions, ground_truth))
        tn = sum((p == 0) and (g == 0) for p, g in zip(predictions, ground_truth))
        fn = sum((p == 0) and (g == 1) for p, g in zip(predictions, ground_truth))
        
        accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
        
        self.accuracy_history.append(metrics)
        
        # Adapt threshold if accuracy is low
        if accuracy < 0.8 and len(self.accuracy_history) > 1:
            self._adapt_threshold()
        
        logger.info(f"📊 Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2f}")
        
        return metrics
    
    def _combine_scores(self, llama_analysis: Dict, embedding_score: Optional[float] = None) -> float:
        """Combine Llama and embedding scores"""
        llama_score = llama_analysis.get('confidence', 0.5)
        
        if embedding_score is not None:
            # Weighted average: 60% Llama, 40% embedding
            return 0.6 * llama_score + 0.4 * embedding_score
        
        return llama_score
    
    def _adapt_threshold(self):
        """Adaptive threshold adjustment based on accuracy history"""
        if len(self.accuracy_history) < 2:
            return
        
        current_accuracy = self.accuracy_history[-1]['accuracy']
        previous_accuracy = self.accuracy_history[-2]['accuracy']
        
        # If accuracy is decreasing, adjust threshold
        if current_accuracy < previous_accuracy:
            self.detection_threshold *= 0.95  # Lower threshold slightly
            logger.info(f"📈 Adjusted threshold to {self.detection_threshold:.3f}")
    
    def get_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        
        if not self.accuracy_history:
            return {'status': 'No monitoring data available'}
        
        latest_metrics = self.accuracy_history[-1]
        
        # Calculate trends
        accuracies = [m['accuracy'] for m in self.accuracy_history]
        accuracy_trend = accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
        
        return {
            'total_monitored': len(self.monitoring_log),
            'current_accuracy': latest_metrics['accuracy'],
            'accuracy_trend': accuracy_trend,
            'precision': latest_metrics['precision'],
            'recall': latest_metrics['recall'],
            'f1_score': latest_metrics['f1_score'],
            'detection_threshold': self.detection_threshold,
            'timestamp': datetime.now().isoformat(),
            'history_count': len(self.accuracy_history)
        }


def setup_ollama_integration(model_name: str = "llama2") -> Tuple[Optional[OllamaService], Optional[LlamaFraudAnalyzer]]:
    """Complete setup for Ollama integration"""
    
    logger.info("=" * 60)
    logger.info("🚀 OLLAMA FRAUD DETECTION INTEGRATION SETUP")
    logger.info("=" * 60)
    
    # Initialize Ollama service
    service = OllamaService()
    
    # Start service
    if not service.start():
        logger.error("Failed to start Ollama service")
        return None, None
    
    # Check and pull model
    available_models = service.list_models()
    logger.info(f"Available models: {available_models}")
    
    if not any(model_name in model for model in available_models):
        logger.info(f"Model {model_name} not found, pulling...")
        if not service.pull_model(model_name):
            logger.error(f"Failed to pull {model_name}")
            return None, None
    else:
        logger.info(f"✅ {model_name} is available")
    
    # Initialize analyzer
    analyzer = LlamaFraudAnalyzer(model_name)
    
    logger.info("✅ Ollama integration setup complete!")
    logger.info("=" * 60)
    
    return service, analyzer


if __name__ == "__main__":
    # Test setup
    service, analyzer = setup_ollama_integration()
    
    if service and analyzer:
        # Test with sample transaction
        sample_transaction = {
            'amount': 5000,
            'merchant': 'Luxury Jewelry',
            'location': 'Singapore',
            'time': '02:30 AM',
            'device': 'mobile',
            'velocity_6h': 15000
        }
        
        logger.info("\n📝 Testing fraud narrative generation...")
        narrative = analyzer.generate_fraud_narrative(sample_transaction)
        logger.info(f"Generated: {narrative}")
        
        logger.info("\n🔍 Testing pattern analysis...")
        analysis = analyzer.analyze_fraud_pattern(narrative)
        logger.info(f"Analysis: {json.dumps(analysis, indent=2)}")
