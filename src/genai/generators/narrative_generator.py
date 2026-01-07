"""
Step 2: Fraud Narrative Generator
Converts transaction data into narrative descriptions for LLM training
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
import random


class NarrativeGenerator:
    """Generates fraud/legitimate narratives from transaction data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Risk factors and patterns
        self.high_risk_merchants = [
            'Jewelry', 'Luxury', 'Diamond', 'VIP', 'Premium', 'Elite', 
            'Private', 'Exclusive', 'Fine', 'Haute', 'International'
        ]
        
        self.fraud_indicators = [
            'unusual location', 'high amount', 'quick succession', 
            'after hours', 'overseas', 'premium merchant', 'high-value'
        ]
    
    def generate_narrative(self, row):
        """
        Generate a narrative for a financial profile
        
        Args:
            row: pandas Series representing a customer profile
            
        Returns:
            Narrative string
        """
        # Extract features from financial data
        age = row.get('customer_age', 30)
        income = row.get('income', 50000)
        risk_score = row.get('credit_risk_score', 150)
        velocity_6h = row.get('velocity_6h', 0)
        velocity_24h = row.get('velocity_24h', 0)
        velocity_4w = row.get('velocity_4w', 0)
        days_since_request = row.get('days_since_request', 0)
        employment = row.get('employment_status', 'unknown')
        housing = row.get('housing_status', 'unknown')
        payment_type = row.get('payment_type', 'unknown')
        device_fraud_count = row.get('device_fraud_count', 0)
        
        # Build narrative from financial profile
        narrative = f"Customer profile: "
        
        # Age and employment
        if age < 25:
            narrative += f"Young ({age} years old) "
        elif age > 65:
            narrative += f"Elderly ({age} years old) "
        else:
            narrative += f"Age {age}. "
        
        # Income and employment status
        narrative += f"Income level ${income:.0f}. Employment: {employment}. Housing: {housing}. "
        
        # Risk assessment
        if risk_score > 200:
            narrative += "High credit risk score. "
        elif risk_score < 100:
            narrative += "Low credit risk score. "
        
        # Transaction velocity patterns
        if velocity_6h > 5000:
            narrative += f"High 6-hour velocity (${velocity_6h:.0f}). "
        if velocity_24h > 10000:
            narrative += f"High 24-hour velocity (${velocity_24h:.0f}). "
        if velocity_4w > 50000:
            narrative += f"High 4-week velocity (${velocity_4w:.0f}). "
        
        # Days since request
        if days_since_request < 7:
            narrative += "Recent request. "
        
        # Payment type
        narrative += f"Payment type: {payment_type}. "
        
        # Device fraud history
        if device_fraud_count > 0:
            narrative += f"Device has {device_fraud_count} fraud incidents. "
        
        return narrative.strip()
    
    def generate_batch_narratives(self, df, sample_size=None):
        """
        Generate narratives for a batch of customer profiles
        
        Args:
            df: DataFrame with financial data
            sample_size: Optional limit on number of narratives to generate
            
        Returns:
            List of dictionaries with narrative, label, and features
        """
        narratives = []
        
        # Use sample if specified, otherwise use all rows
        data = df.sample(n=min(sample_size, len(df))) if sample_size else df
        
        for idx, row in data.iterrows():
            narrative = self.generate_narrative(row)
            fraud_label = int(row.get('fraud_bool', 0))
            income = float(row.get('income', 0))
            risk_score = float(row.get('credit_risk_score', 0))
            
            narratives.append({
                'narrative': narrative,
                'fraud_label': fraud_label,
                'income': income,
                'risk_score': risk_score,
                'age': int(row.get('customer_age', 0)),
                'velocity_4w': float(row.get('velocity_4w', 0))
            })
        
        return narratives
    
    def augment_narratives(self, narratives: List[Dict]) -> List[Dict]:
        """
        Augment narratives with additional variations and context
        
        Args:
            narratives: List of narrative dictionaries
            
        Returns:
            Augmented list of narratives
        """
        augmented = narratives.copy()
        
        # Add fraud patterns to fraud transactions
        for narr in augmented:
            if narr['fraud_label'] == 1:
                # Add fraud indicators
                indicators = random.sample(self.fraud_indicators, k=min(2, len(self.fraud_indicators)))
                for indicator in indicators:
                    narr['narrative'] += f" {indicator} detected."
        
        return augmented
    
    def save_narratives(self, narratives: List[Dict], output_path):
        """
        Save narratives to CSV file
        
        Args:
            narratives: List of narrative dictionaries
            output_path: Path to save CSV
        """
        df = pd.DataFrame(narratives)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(narratives)} narratives to {output_path}")


class NarrativeGeneratorPipeline:
    """Orchestrates narrative generation from cleaned data"""
    
    def __init__(self, data_dir='generated', output_dir='generated'):
        # Resolve paths relative to parent directory (project root)
        current_dir = Path(__file__).parent
        self.data_dir = (current_dir.parent / data_dir).resolve()
        self.output_dir = (current_dir.parent / output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.generator = NarrativeGenerator()
    
    def run(self, input_file='fraud_data_combined_clean.csv', sample_size=5000):
        """
        Execute narrative generation pipeline
        
        Args:
            input_file: Name of the combined clean CSV file
            sample_size: Number of narratives to generate
            
        Returns:
            Path to generated narratives CSV
        """
        try:
            # Load combined clean data
            input_path = self.data_dir / input_file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            self.logger.info(f"Loading data from {input_path}")
            df = pd.read_csv(input_path)
            
            # Generate narratives
            self.logger.info(f"Generating narratives for {min(sample_size, len(df))} transactions")
            narratives = self.generator.generate_batch_narratives(df, sample_size=sample_size)
            
            # Augment narratives with fraud patterns
            self.logger.info("Augmenting narratives with fraud patterns")
            narratives = self.generator.augment_narratives(narratives)
            
            # Save narratives
            output_file = self.output_dir / 'fraud_narratives_combined.csv'
            self.generator.save_narratives(narratives, output_file)
            
            self.logger.info(f"Narrative generation complete: {len(narratives)} narratives")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error in narrative generation pipeline: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = NarrativeGeneratorPipeline()
    # This will run after Step 1 is complete
    output_path = pipeline.run()
    print(f"Narratives generated: {output_path}")
