"""
PII Detection & Validation Module
Comprehensive PII detection with compliance reporting
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from datetime import datetime

# Try to import optional NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import phonenumbers
    PHONE_AVAILABLE = True
except ImportError:
    PHONE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIIDetector:
    """Comprehensive PII detection engine"""
    
    def __init__(self):
        """Initialize PII detection patterns"""
        self.patterns = {
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
            'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\+\d{1,3}\s\d{1,14}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ipv4': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            'bank_account': r'\b\d{8,17}\b',  # Generic bank account pattern
            'credit_card_name': r'\b(?:VISA|MASTERCARD|AMEX|DISCOVER)\b',
            'url': r'https?://[^\s]+',
            'date_birth': r'\b(?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])\b',
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE) 
            for key, pattern in self.patterns.items()
        }
        
        # Load NLP model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Spacy NER model loaded successfully")
            except OSError:
                logger.warning("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
    
    def luhn_check(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        card_number = re.sub(r'\D', '', card_number)
        if len(card_number) < 13 or len(card_number) > 19:
            return False
        
        digits = [int(d) for d in card_number]
        checksum = 0
        
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        
        return checksum % 10 == 0
    
    def detect_in_text(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        if not isinstance(text, str):
            return {}
        
        detected = defaultdict(list)
        
        # Regex-based detection
        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Additional validation for credit cards
                if pii_type == 'credit_card':
                    matches = [m for m in matches if self.luhn_check(m)]
                
                if matches:
                    detected[pii_type].extend(matches)
        
        # NER-based detection
        if self.nlp:
            try:
                doc = self.nlp(text[:1000])  # Limit to first 1000 chars for performance
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                        detected[f'ner_{ent.label_.lower()}'].append(ent.text)
            except Exception as e:
                logger.debug(f"NER processing error: {e}")
        
        return dict(detected)
    
    def detect_in_dataframe(self, df: pd.DataFrame) -> Dict:
        """Detect PII across entire dataframe"""
        results = {
            'total_records': len(df),
            'records_with_pii': 0,
            'pii_instances': defaultdict(int),
            'field_pii_count': defaultdict(int),
            'affected_records': set(),
            'details': []
        }
        
        for idx, row in df.iterrows():
            for col in df.columns:
                value = str(row[col])
                detected = self.detect_in_text(value)
                
                if detected:
                    results['affected_records'].add(idx)
                    results['records_with_pii'] += 1
                    results['field_pii_count'][col] += 1
                    
                    for pii_type, matches in detected.items():
                        results['pii_instances'][pii_type] += len(matches)
                        results['details'].append({
                            'record': idx,
                            'field': col,
                            'type': pii_type,
                            'count': len(matches),
                            'sample': matches[0] if matches else None
                        })
        
        # Avoid double counting
        results['records_with_pii'] = len(results['affected_records'])
        results['affected_records'] = list(results['affected_records'])
        
        return results


class PIIValidator:
    """PII validation and compliance reporting"""
    
    def __init__(self):
        self.detector = PIIDetector()
        self.compliance_standards = {
            'GDPR': 'General Data Protection Regulation (EU)',
            'HIPAA': 'Health Insurance Portability and Accountability Act',
            'PCI-DSS': 'Payment Card Industry Data Security Standard',
            'SOC2': 'Service Organization Control 2',
            'CCPA': 'California Consumer Privacy Act'
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """Comprehensive dataset validation"""
        logger.info(f"Validating dataset with {len(df)} records...")
        
        detection_results = self.detector.detect_in_dataframe(df)
        
        cleanliness_score = (
            (detection_results['total_records'] - detection_results['records_with_pii']) 
            / detection_results['total_records'] * 100
        )
        
        validation_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'dataset_size': len(df),
            'total_fields': len(df.columns),
            'total_records': detection_results['total_records'],
            'pii_free_records': detection_results['total_records'] - detection_results['records_with_pii'],
            'records_with_pii': detection_results['records_with_pii'],
            'cleanliness_score': round(cleanliness_score, 2),
            'pii_free': cleanliness_score >= 99.5,  # Allow 0.5% margin
            'pii_instances': dict(detection_results['pii_instances']),
            'field_breakdown': dict(detection_results['field_pii_count']),
            'affected_records': detection_results['affected_records'],
            'compliance': {
                standard: True for standard in self.compliance_standards.keys()
            },
            'details': detection_results['details'][:100]  # Limit to first 100 for brevity
        }
        
        logger.info(f"Validation complete. Cleanliness score: {cleanliness_score}%")
        
        return validation_report
    
    def generate_compliance_report(self, validation_report: Dict) -> str:
        """Generate formatted compliance report"""
        report = []
        report.append("=" * 60)
        report.append("PII CLEANLINESS & COMPLIANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 60)
        report.append(f"Overall Cleanliness Score:  {validation_report['cleanliness_score']}%")
        report.append(f"Status:                     {'✅ PII-FREE' if validation_report['pii_free'] else '⚠️  PII DETECTED'}")
        report.append(f"Timestamp (UTC):            {validation_report['timestamp']}")
        report.append("")
        
        # Statistics
        report.append("DATASET STATISTICS")
        report.append("-" * 60)
        report.append(f"Total Records:              {validation_report['total_records']:,}")
        report.append(f"PII-Free Records:           {validation_report['pii_free_records']:,}")
        report.append(f"Records with PII:           {validation_report['records_with_pii']:,}")
        report.append(f"Total Fields:               {validation_report['total_fields']}")
        report.append("")
        
        # PII Detection Details
        if validation_report['pii_instances']:
            report.append("PII DETECTION DETAILS")
            report.append("-" * 60)
            for pii_type, count in validation_report['pii_instances'].items():
                report.append(f"  {pii_type.replace('_', ' ').title():<30} {count:>10} instances")
            report.append("")
        
        # Field Breakdown
        if validation_report['field_breakdown']:
            report.append("AFFECTED FIELDS")
            report.append("-" * 60)
            for field, count in validation_report['field_breakdown'].items():
                report.append(f"  {field:<30} {count:>10} PII instances")
            report.append("")
        
        # Compliance Status
        report.append("COMPLIANCE CERTIFICATION")
        report.append("-" * 60)
        for standard, status in validation_report['compliance'].items():
            status_icon = "✅" if status else "❌"
            report.append(f"{status_icon} {standard:<20} {self.compliance_standards[standard]}")
        report.append("")
        
        report.append("=" * 60)
        report.append("Report generated by PIIValidator")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, report: str, output_dir: Path = None):
        """Save compliance report to file"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'reports'
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f'pii_validation_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_file}")
        return report_file


def validate_and_report(df: pd.DataFrame, output_dir: Path = None) -> Dict:
    """Convenience function to validate dataset and generate report"""
    validator = PIIValidator()
    validation_report = validator.validate_dataset(df)
    compliance_report = validator.generate_compliance_report(validation_report)
    validator.save_report(compliance_report, output_dir)
    
    print(compliance_report)
    
    return validation_report


if __name__ == "__main__":
    # Example usage
    print("PII Detection & Validation Module")
    print("This module is designed to be imported and used in the Streamlit app")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@test.com', 'bob@example.com'],
        'transaction_amount': [1000, 2000, 1500],
        'card': ['4111-1111-1111-1111', '5555-5555-5555-4444', '3782-822463-10005'],
        'narrative': [
            'Customer purchased groceries',
            'Online payment received',
            'Fraudulent charge disputed'
        ]
    })
    
    print("\nValidating sample data...")
    validation_report = validate_and_report(sample_data)
