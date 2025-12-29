"""
Step 1: PII Cleaning Module
Combines all datasets and removes/masks PII
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from security.pii_guard import PIIGuard


class PIICleaner:
    """Orchestrates PII cleaning for all datasets"""
    
    def __init__(self, data_dir='data', output_dir='generated'):
        # Resolve paths relative to parent directory (project root)
        current_dir = Path(__file__).parent
        self.data_dir = (current_dir.parent / data_dir).resolve()
        self.output_dir = (current_dir.parent / output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.pii_guard = PIIGuard()
        
    def load_all_datasets(self):
        """
        Load all CSV datasets from data directory
        
        Returns:
            Dictionary with dataset names as keys and DataFrames as values
        """
        datasets = {}
        
        # Expected datasets - note: actual files use space in "Variant I"
        dataset_names = ['Base.csv', 'Variant I.csv', 'Variant II.csv', 
                        'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
        
        for dataset_name in dataset_names:
            file_path = self.data_dir / dataset_name
            if file_path.exists():
                try:
                    # Sample to reduce memory for large datasets
                    df = pd.read_csv(file_path, nrows=50000)
                    datasets[dataset_name.replace('.csv', '')] = df
                    self.logger.info(f"Loaded {dataset_name}: {len(df)} rows (sampled)")
                except Exception as e:
                    self.logger.error(f"Error loading {dataset_name}: {e}")
            else:
                self.logger.warning(f"Dataset not found: {file_path}")
        
        return datasets
    
    def clean_dataset(self, df, dataset_name):
        """
        Clean PII from a single dataset
        
        Args:
            df: DataFrame to clean
            dataset_name: Name of the dataset
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning PII from {dataset_name}")
        cleaned_df = self.pii_guard.clean_dataframe(df)
        return cleaned_df
    
    def combine_datasets(self, datasets_dict):
        """
        Combine all cleaned datasets
        
        Args:
            datasets_dict: Dictionary of DataFrames
            
        Returns:
            Combined DataFrame
        """
        combined_dfs = []
        for name, df in datasets_dict.items():
            combined_dfs.append(df)
        
        combined = pd.concat(combined_dfs, ignore_index=True)
        self.logger.info(f"Combined datasets into {len(combined)} rows")
        return combined
    
    def run(self):
        """
        Execute full PII cleaning pipeline
        
        Returns:
            Path to combined cleaned CSV
        """
        try:
            # Load all datasets
            self.logger.info("Loading all datasets...")
            datasets = self.load_all_datasets()
            
            if not datasets:
                raise FileNotFoundError("No datasets found in data directory")
            
            # Clean each dataset
            self.logger.info("Cleaning PII from all datasets...")
            cleaned_datasets = {}
            for name, df in datasets.items():
                cleaned_df = self.clean_dataset(df, name)
                cleaned_datasets[name] = cleaned_df
                
                # Save individual cleaned dataset
                output_file = self.output_dir / f"{name}_clean.csv"
                cleaned_df.to_csv(output_file, index=False)
                self.logger.info(f"Saved cleaned dataset: {output_file}")
            
            # Combine all cleaned datasets
            self.logger.info("Combining all cleaned datasets...")
            combined = self.combine_datasets(cleaned_datasets)
            
            # Save combined dataset
            output_file = self.output_dir / "fraud_data_combined_clean.csv"
            combined.to_csv(output_file, index=False)
            self.logger.info(f"Saved combined clean dataset: {output_file}")
            
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error in PII cleaning pipeline: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    cleaner = PIICleaner()
    output_path = cleaner.run()
    print(f"PII cleaning complete. Output: {output_path}")
