"""
Llama-based Fraud Randomizer Training
- Trains Llama to generate realistic fraud variations
- Uses real-world fraud patterns from datasets
- Increases accuracy through continuous feedback
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ollama_integration import LlamaFraudAnalyzer, LlamaMonitoringSystem, OllamaService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaRandomizerTrainer:
    """Trains Llama to generate fraud variations for randomizer"""
    
    def __init__(self, analyzer: LlamaFraudAnalyzer):
        self.analyzer = analyzer
        self.training_data = []
        self.patterns_learned = {}
        self.variance_factors = {}
        
    def load_real_fraud_data(self, filepath: Path) -> pd.DataFrame:
        """Load real fraud data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"✅ Loaded {len(df)} fraud records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def extract_fraud_patterns(self, fraud_df: pd.DataFrame) -> Dict:
        """Extract and learn fraud patterns from real data"""
        patterns = {
            'amount_patterns': {},
            'location_patterns': {},
            'time_patterns': {},
            'merchant_patterns': {},
            'velocity_patterns': {}
        }
        
        try:
            # Amount patterns
            amount_col = [c for c in fraud_df.columns if 'amount' in c.lower()][0] if any('amount' in c.lower() for c in fraud_df.columns) else None
            if amount_col:
                patterns['amount_patterns'] = {
                    'mean': fraud_df[amount_col].mean(),
                    'std': fraud_df[amount_col].std(),
                    'min': fraud_df[amount_col].min(),
                    'max': fraud_df[amount_col].max(),
                    'quantiles': fraud_df[amount_col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            
            # Velocity patterns
            velocity_cols = [c for c in fraud_df.columns if 'velocity' in c.lower()]
            for vcol in velocity_cols:
                if vcol in fraud_df.columns:
                    patterns['velocity_patterns'][vcol] = {
                        'mean': fraud_df[vcol].mean(),
                        'std': fraud_df[vcol].std(),
                    }
            
            logger.info(f"✅ Extracted patterns: {json.dumps({k: 'OK' for k in patterns.keys()}, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
        
        return patterns
    
    def generate_variations(self, base_transaction: Dict, num_variations: int = 5) -> List[Dict]:
        """Generate realistic fraud variations using Llama"""
        variations = []
        
        logger.info(f"🔄 Generating {num_variations} variations for transaction...")
        
        for i in range(num_variations):
            # Create variation with different characteristics
            variation = base_transaction.copy()
            
            # Apply random variations to specific fields
            if 'amount' in variation:
                factor = np.random.uniform(0.8, 1.5)  # 80-150% of original
                variation['amount'] = variation['amount'] * factor
            
            if 'velocity_6h' in variation:
                factor = np.random.uniform(1.0, 2.5)  # 100-250% increase
                variation['velocity_6h'] = variation['velocity_6h'] * factor
            
            # Add variation identifier
            variation['variation_id'] = f"v{i+1}"
            
            # Generate narrative for this variation
            narrative = self.analyzer.generate_fraud_narrative(variation)
            
            # Get Llama analysis
            analysis = self.analyzer.analyze_fraud_pattern(narrative)
            
            variation_record = {
                'base_transaction': base_transaction,
                'variation': variation,
                'narrative': narrative,
                'analysis': analysis,
                'similarity_to_base': self._calculate_similarity(base_transaction, variation)
            }
            
            variations.append(variation_record)
        
        logger.info(f"✅ Generated {len(variations)} variations")
        return variations
    
    def train_on_real_world_frauds(self, fraud_df: pd.DataFrame, 
                                   sample_size: int = 50) -> Dict:
        """Train randomizer on real-world fraud patterns"""
        
        logger.info(f"\n{'='*70}")
        logger.info("🎓 TRAINING LLAMA RANDOMIZER ON REAL-WORLD FRAUDS")
        logger.info(f"{'='*70}\n")
        
        # Extract patterns
        self.patterns_learned = self.extract_fraud_patterns(fraud_df)
        
        # Sample fraud cases
        sample_size = min(sample_size, len(fraud_df))
        sample_indices = np.random.choice(len(fraud_df), sample_size, replace=False)
        
        all_variations = []
        training_metrics = {
            'total_base_frauds': sample_size,
            'total_variations_generated': 0,
            'variations_per_fraud': 5,
            'training_start': datetime.now().isoformat(),
            'variations_by_type': {}
        }
        
        for idx, sample_idx in enumerate(sample_indices):
            row = fraud_df.iloc[sample_idx]
            
            # Convert to transaction dict
            transaction = row.to_dict()
            
            # Generate variations
            variations = self.generate_variations(transaction, num_variations=5)
            all_variations.extend(variations)
            
            # Track by fraud type if available
            fraud_type = transaction.get('fraud_type', 'unknown')
            if fraud_type not in training_metrics['variations_by_type']:
                training_metrics['variations_by_type'][fraud_type] = 0
            training_metrics['variations_by_type'][fraud_type] += len(variations)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{sample_size} fraud cases...")
        
        training_metrics['total_variations_generated'] = len(all_variations)
        training_metrics['training_end'] = datetime.now().isoformat()
        
        # Save training results
        self.training_data = all_variations
        
        logger.info(f"\n✅ TRAINING COMPLETE")
        logger.info(f"   Base frauds: {training_metrics['total_base_frauds']}")
        logger.info(f"   Total variations: {training_metrics['total_variations_generated']}")
        logger.info(f"   Variations per fraud: {training_metrics['variations_per_fraud']}")
        
        return training_metrics
    
    def save_training_results(self, output_dir: Path = Path('models')) -> Path:
        """Save trained randomizer results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save variations
        variations_file = output_dir / 'llama_randomizer_variations.json'
        with open(variations_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            training_data = []
            for item in self.training_data:
                item_copy = item.copy()
                # Handle nested dicts
                training_data.append(self._convert_to_serializable(item_copy))
            json.dump(training_data, f, indent=2, default=str)
        
        logger.info(f"✅ Saved variations to {variations_file}")
        
        # Save patterns
        patterns_file = output_dir / 'llama_learned_patterns.json'
        with open(patterns_file, 'w') as f:
            json.dump(self._convert_to_serializable(self.patterns_learned), f, indent=2, default=str)
        
        logger.info(f"✅ Saved patterns to {patterns_file}")
        
        return output_dir
    
    def _calculate_similarity(self, base: Dict, variation: Dict) -> float:
        """Calculate similarity between base and variation"""
        numeric_keys = [k for k in base.keys() if isinstance(base.get(k), (int, float))]
        
        if not numeric_keys:
            return 1.0
        
        differences = []
        for key in numeric_keys:
            if key in variation:
                base_val = base[key]
                var_val = variation[key]
                if base_val != 0:
                    diff = abs(var_val - base_val) / abs(base_val)
                    differences.append(diff)
        
        if not differences:
            return 1.0
        
        # Similarity = 1 - average percentage difference
        avg_diff = np.mean(differences)
        return max(0.0, 1.0 - avg_diff)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


class AccuracyImprover:
    """Improves model accuracy through Llama-based feedback"""
    
    def __init__(self, monitoring_system: LlamaMonitoringSystem):
        self.monitoring = monitoring_system
        self.improvement_history = []
        self.feedback_log = []
        
    def analyze_false_positives(self, predictions: List[int], 
                               ground_truth: List[int]) -> List[Dict]:
        """Analyze and learn from false positives"""
        fp_cases = []
        
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            if pred == 1 and true == 0:  # False positive
                fp_cases.append({
                    'index': i,
                    'predicted_fraud': True,
                    'actual_fraud': False,
                    'feedback': 'Model needs refinement on this pattern'
                })
        
        logger.info(f"🔴 Found {len(fp_cases)} false positives")
        return fp_cases
    
    def analyze_false_negatives(self, predictions: List[int], 
                               ground_truth: List[int]) -> List[Dict]:
        """Analyze and learn from false negatives"""
        fn_cases = []
        
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            if pred == 0 and true == 1:  # False negative
                fn_cases.append({
                    'index': i,
                    'predicted_fraud': False,
                    'actual_fraud': True,
                    'feedback': 'Model missed this fraud pattern'
                })
        
        logger.info(f"🔴 Found {len(fn_cases)} false negatives")
        return fn_cases
    
    def get_improvement_recommendations(self, metrics: Dict) -> List[str]:
        """Get recommendations to improve accuracy"""
        recommendations = []
        
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        if accuracy < 0.80:
            recommendations.append("📈 Accuracy below 80% - more training needed")
        
        if precision < 0.75:
            recommendations.append("⚠️ Precision low - reduce false positives (check FP analysis)")
        
        if recall < 0.85:
            recommendations.append("⚠️ Recall low - increase sensitivity to catch more frauds (check FN analysis)")
        
        if precision > 0.95 and recall < 0.80:
            recommendations.append("⚡ Model is too conservative - lower threshold to increase recall")
        
        return recommendations


def main():
    """Main training pipeline"""
    
    # Setup Ollama
    logger.info("🚀 Starting Ollama integration setup...")
    service = OllamaService()
    
    if not service.start():
        logger.error("Failed to start Ollama")
        return
    
    # Check model availability
    models = service.list_models()
    logger.info(f"Available models: {models}")
    
    if not any("llama" in m for m in models):
        logger.info("Pulling llama2 model...")
        service.pull_model("llama2")
    
    # Initialize analyzer
    analyzer = LlamaFraudAnalyzer()
    
    # Create trainer
    trainer = LlamaRandomizerTrainer(analyzer)
    
    # Load fraud data
    data_dir = Path(__file__).parent.parent / 'generated'
    fraud_files = list(data_dir.glob('*_clean.csv'))
    
    if not fraud_files:
        logger.error(f"No fraud data found in {data_dir}")
        service.stop()
        return
    
    # Train on available fraud data
    for fraud_file in fraud_files[:2]:  # Use first 2 variant files
        logger.info(f"\n📂 Processing {fraud_file.name}...")
        
        fraud_df = trainer.load_real_fraud_data(fraud_file)
        if fraud_df.empty:
            continue
        
        # Train on subset
        metrics = trainer.train_on_real_world_frauds(fraud_df, sample_size=20)
        logger.info(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    # Save results
    output_dir = trainer.save_training_results()
    logger.info(f"\n✅ Training complete! Results saved to {output_dir}")
    
    # Setup monitoring
    logger.info("\n🔍 Setting up monitoring system...")
    monitoring = LlamaMonitoringSystem(analyzer)
    
    # Test monitoring on some transactions
    if trainer.training_data:
        test_narratives = [t['narrative'] for t in trainer.training_data[:10]]
        test_ground_truth = [1] * len(test_narratives)  # All are fraud
        
        logger.info(f"\n📊 Testing monitoring on {len(test_narratives)} transactions...")
        results = monitoring.batch_monitor(test_narratives)
        
        # Evaluate
        predictions = [1 if r['is_fraud'] else 0 for r in results]
        metrics = monitoring.evaluate_accuracy(results, test_ground_truth)
        
        logger.info(f"\n📈 Monitoring Metrics:")
        logger.info(json.dumps(metrics, indent=2))
        
        # Get report
        report = monitoring.get_monitoring_report()
        logger.info(f"\n📋 Monitoring Report:")
        logger.info(json.dumps(report, indent=2))
    
    # Cleanup
    service.stop()
    logger.info("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
