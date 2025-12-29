"""
PII Detection and Masking Module
Identifies and removes/masks Personally Identifiable Information (PII) from datasets
"""

import re
import pandas as pd
import logging


class PIIGuard:
    """Detects and masks PII in transaction data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def mask_email(email):
        """Mask email addresses"""
        if pd.isna(email):
            return email
        return re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[MASKED_EMAIL]', str(email))
    
    @staticmethod
    def mask_phone(phone):
        """Mask phone numbers"""
        if pd.isna(phone):
            return phone
        return re.sub(r'\d{3}-\d{4}', '***-****', str(phone))
    
    @staticmethod
    def mask_ssn(ssn):
        """Mask SSN"""
        if pd.isna(ssn):
            return ssn
        return re.sub(r'\d{3}-\d{2}-\d{4}', '***-**-****', str(ssn))
    
    @staticmethod
    def mask_name(name):
        """Mask customer name"""
        if pd.isna(name):
            return name
        name_str = str(name)
        if len(name_str) > 2:
            return name_str[0] + '*' * (len(name_str) - 2) + name_str[-1]
        return '[MASKED_NAME]'
    
    def clean_dataframe(self, df):
        """
        Clean PII from dataframe
        
        Args:
            df: pandas DataFrame
            
        Returns:
            DataFrame with masked PII columns (if any exist)
        """
        df_clean = df.copy()
        
        # Mask email column if exists
        if 'customer_email' in df_clean.columns:
            df_clean['customer_email'] = df_clean['customer_email'].apply(self.mask_email)
        
        # Mask phone column if exists
        if 'customer_phone' in df_clean.columns:
            df_clean['customer_phone'] = df_clean['customer_phone'].apply(self.mask_phone)
        
        # Mask SSN column if exists
        if 'customer_ssn' in df_clean.columns:
            df_clean['customer_ssn'] = df_clean['customer_ssn'].apply(self.mask_ssn)
        
        # Mask name columns if exist
        for col in ['customer_name', 'merchant_name', 'name_email_similarity']:
            if col in df_clean.columns:
                # Don't mask merchant_name for fraud detection
                if col != 'merchant_name' and col != 'name_email_similarity':
                    df_clean[col] = df_clean[col].apply(self.mask_name)
        
        # For financial data, no direct PII columns found in actual dataset
        # But we log that cleaning was performed
        
        return df_clean
    
    def detect_pii_fields(self, df):
        """
        Detect columns that contain PII
        
        Args:
            df: pandas DataFrame
            
        Returns:
            List of column names containing PII
        """
        pii_fields = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pii_term in col_lower for pii_term in ['email', 'phone', 'ssn', 'name', 'address']):
                pii_fields.append(col)
        
        return pii_fields


if __name__ == "__main__":
    # Test the PII Guard
    test_data = {
        'customer_email': ['john@example.com', 'jane@example.com'],
        'customer_phone': ['555-1234', '555-5678'],
        'customer_ssn': ['123-45-6789', '987-65-4321']
    }
    test_df = pd.DataFrame(test_data)
    
    guard = PIIGuard()
    cleaned = guard.clean_dataframe(test_df)
    print("Original:")
    print(test_df)
    print("\nCleaned:")
    print(cleaned)
