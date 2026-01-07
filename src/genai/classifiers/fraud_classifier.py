"""
Attack Pattern Analysis & Extraction Module
Identifies and scores fraud attack patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import logging
import json
from datetime import datetime

try:
    from nltk.tokenize import word_tokenize
    from nltk import ngrams
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackPatternAnalyzer:
    """Extract and analyze fraud attack patterns"""
    
    def __init__(self, min_pattern_frequency: int = 3):
        """
        Initialize pattern analyzer
        
        Args:
            min_pattern_frequency: Minimum occurrences to be considered a pattern
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.nlp = None
        self.patterns = {}
        
        # Load NLP model
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Spacy model loaded for pattern analysis")
            except OSError:
                logger.warning("Spacy model not found")
        
        # Download NLTK data if needed
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        
        # Attack pattern categories
        self.attack_categories = {
            'card_fraud': {
                'keywords': ['stolen', 'card', 'unauthorized', 'charged', 'fraudulent'],
                'severity': 'high',
                'avg_loss': 2500
            },
            'identity_theft': {
                'keywords': ['identity', 'account', 'unauthorized access', 'impersonation'],
                'severity': 'critical',
                'avg_loss': 5000
            },
            'phishing': {
                'keywords': ['link', 'email', 'clicked', 'suspicious', 'credential'],
                'severity': 'high',
                'avg_loss': 1500
            },
            'account_takeover': {
                'keywords': ['password', 'login', 'breach', 'compromise', 'takeover'],
                'severity': 'critical',
                'avg_loss': 4000
            },
            'transaction_anomaly': {
                'keywords': ['unusual', 'anomaly', 'pattern', 'suspicious', 'abnormal'],
                'severity': 'medium',
                'avg_loss': 1000
            },
            'social_engineering': {
                'keywords': ['tricked', 'deceived', 'pretended', 'impersonated', 'social'],
                'severity': 'high',
                'avg_loss': 2000
            },
            'data_breach': {
                'keywords': ['breach', 'leaked', 'exposed', 'compromised', 'stolen data'],
                'severity': 'critical',
                'avg_loss': 10000
            },
            'malware': {
                'keywords': ['malware', 'virus', 'infection', 'ransomware', 'trojan'],
                'severity': 'critical',
                'avg_loss': 8000
            }
        }
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """Extract n-grams from text"""
        if not NLTK_AVAILABLE or not isinstance(text, str):
            return []
        
        try:
            tokens = word_tokenize(text.lower())
            # Filter out very short tokens
            tokens = [t for t in tokens if len(t) > 2]
            return list(ngrams(tokens, n))
        except:
            return []
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using NER"""
        entities = defaultdict(list)
        
        if not self.nlp or not isinstance(text, str):
            return entities
        
        try:
            doc = self.nlp(text[:2000])  # Limit for performance
            for ent in doc.ents:
                entities[ent.label_].append(ent.text)
        except:
            pass
        
        return dict(entities)
    
    def categorize_fraud_type(self, text: str) -> List[Tuple[str, float]]:
        """Categorize fraud narrative by attack type"""
        text_lower = text.lower()
        scores = []
        
        for category, config in self.attack_categories.items():
            keyword_matches = sum(
                text_lower.count(keyword) 
                for keyword in config['keywords']
            )
            
            if keyword_matches > 0:
                # Score based on keyword frequency
                score = min(keyword_matches / len(config['keywords']), 1.0)
                scores.append((category, score))
        
        # Return sorted by score
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def analyze_narratives(self, narratives: List[str], 
                          amounts: List[float] = None) -> Dict:
        """Analyze fraud narratives for patterns"""
        logger.info(f"Analyzing {len(narratives)} fraud narratives...")
        
        analysis = {
            'total_narratives': len(narratives),
            'ngram_patterns': defaultdict(int),
            'entity_patterns': defaultdict(int),
            'attack_types': defaultdict(lambda: {
                'count': 0, 'avg_score': 0, 'avg_loss': 0, 'narratives': []
            }),
            'keywords': Counter(),
            'entities': defaultdict(int),
            'top_patterns': []
        }
        
        # Process each narrative
        for idx, narrative in enumerate(narratives):
            if not isinstance(narrative, str):
                continue
            
            # Extract n-grams
            bigrams = self.extract_ngrams(narrative, n=2)
            trigrams = self.extract_ngrams(narrative, n=3)
            
            for gram in bigrams:
                analysis['ngram_patterns'][' '.join(gram)] += 1
            for gram in trigrams:
                analysis['ngram_patterns'][' '.join(gram)] += 1
            
            # Extract entities
            entities = self.extract_entities(narrative)
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    analysis['entities'][entity_type] += 1
            
            # Categorize fraud type
            fraud_types = self.categorize_fraud_type(narrative)
            for fraud_type, score in fraud_types:
                loss = amounts[idx] if amounts and idx < len(amounts) else 0
                analysis['attack_types'][fraud_type]['count'] += 1
                analysis['attack_types'][fraud_type]['avg_score'] += score
                analysis['attack_types'][fraud_type]['avg_loss'] += loss
                analysis['attack_types'][fraud_type]['narratives'].append(idx)
            
            # Extract keywords
            words = [w.lower() for w in narrative.split() if len(w) > 3]
            analysis['keywords'].update(words)
        
        # Aggregate and normalize
        for attack_type, data in analysis['attack_types'].items():
            if data['count'] > 0:
                data['avg_score'] = round(data['avg_score'] / data['count'], 3)
                data['avg_loss'] = round(data['avg_loss'] / data['count'], 2)
                data['severity'] = self.attack_categories[attack_type]['severity']
        
        # Get top patterns
        analysis['top_patterns'] = self._extract_top_patterns(
            analysis['ngram_patterns']
        )
        
        # Clean up defaultdict for serialization
        analysis['ngram_patterns'] = dict(analysis['ngram_patterns'])
        analysis['entities'] = dict(analysis['entities'])
        analysis['attack_types'] = dict(analysis['attack_types'])
        
        return analysis
    
    def _extract_top_patterns(self, patterns: Dict[str, int], 
                             top_n: int = 20) -> List[Dict]:
        """Extract top N patterns with metadata"""
        # Filter by minimum frequency
        filtered = {p: c for p, c in patterns.items() 
                   if c >= self.min_pattern_frequency}
        
        # Sort by frequency
        sorted_patterns = sorted(filtered.items(), 
                                key=lambda x: x[1], reverse=True)[:top_n]
        
        return [
            {
                'pattern': pattern,
                'frequency': count,
                'percentage': round(count / sum(patterns.values()) * 100, 2)
            }
            for pattern, count in sorted_patterns
        ]
    
    def generate_threat_score(self, narrative: str, 
                             amount: float = 0) -> Dict:
        """Generate threat score for a narrative"""
        fraud_types = self.categorize_fraud_type(narrative)
        
        # Calculate composite score
        base_score = fraud_types[0][1] if fraud_types else 0
        
        # Increase score based on attack severity
        severity_multipliers = {
            'critical': 1.5,
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8
        }
        
        top_attack_type = fraud_types[0][0] if fraud_types else 'unknown'
        severity = self.attack_categories[top_attack_type]['severity']
        multiplier = severity_multipliers.get(severity, 1.0)
        
        # Consider amount (higher amounts = higher threat)
        amount_factor = min(amount / 5000, 2.0) if amount > 0 else 1.0
        
        threat_score = min(base_score * multiplier * amount_factor, 1.0)
        
        return {
            'threat_score': round(threat_score, 3),
            'primary_attack_type': top_attack_type,
            'attack_confidence': round(fraud_types[0][1], 3) if fraud_types else 0,
            'severity': severity,
            'risk_level': self._get_risk_level(threat_score),
            'contributing_factors': [ft[0] for ft in fraud_types[:3]]
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from threat score"""
        if score >= 0.75:
            return 'CRITICAL'
        elif score >= 0.5:
            return 'HIGH'
        elif score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'


class AttackPatternReporter:
    """Generate visualizable reports from pattern analysis"""
    
    def __init__(self, analysis_results: Dict):
        self.results = analysis_results
    
    def get_top_patterns_table(self, top_n: int = 15) -> pd.DataFrame:
        """Get top patterns as DataFrame for visualization"""
        patterns = self.results['top_patterns'][:top_n]
        return pd.DataFrame(patterns)
    
    def get_attack_type_summary(self) -> pd.DataFrame:
        """Get attack type distribution"""
        data = []
        for attack_type, stats in self.results['attack_types'].items():
            data.append({
                'Attack Type': attack_type.replace('_', ' ').title(),
                'Count': stats['count'],
                'Avg Confidence': stats['avg_score'],
                'Avg Loss ($)': f"${stats['avg_loss']:,.2f}",
                'Severity': stats['severity'].upper()
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Count', ascending=False)
    
    def get_threat_distribution(self) -> Dict:
        """Get distribution of threat levels"""
        return {
            'critical': sum(1 for s in self.results['attack_types'].values() 
                           if s['severity'] == 'critical'),
            'high': sum(1 for s in self.results['attack_types'].values() 
                       if s['severity'] == 'high'),
            'medium': sum(1 for s in self.results['attack_types'].values() 
                         if s['severity'] == 'medium'),
            'low': sum(1 for s in self.results['attack_types'].values() 
                      if s['severity'] == 'low'),
        }
    
    def generate_summary_report(self) -> str:
        """Generate text summary of findings"""
        report = []
        report.append("=" * 70)
        report.append("FRAUD ATTACK PATTERN ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Summary
        report.append("OVERVIEW")
        report.append("-" * 70)
        report.append(f"Total Narratives Analyzed:  {self.results['total_narratives']:,}")
        report.append(f"Unique N-gram Patterns:     {len(self.results['ngram_patterns']):,}")
        report.append(f"Attack Types Detected:      {len(self.results['attack_types'])}")
        report.append(f"Named Entities Extracted:   {len(self.results['entities'])}")
        report.append("")
        
        # Top Patterns
        if self.results['top_patterns']:
            report.append("TOP 10 FRAUD PATTERNS")
            report.append("-" * 70)
            for idx, pattern in enumerate(self.results['top_patterns'][:10], 1):
                report.append(f"{idx:2}. {pattern['pattern']:<40} "
                            f"({pattern['frequency']:>4} occurrences, {pattern['percentage']:>5.1f}%)")
            report.append("")
        
        # Attack Types
        if self.results['attack_types']:
            report.append("ATTACK TYPE BREAKDOWN")
            report.append("-" * 70)
            sorted_attacks = sorted(
                self.results['attack_types'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            for attack_type, stats in sorted_attacks:
                report.append(f"{attack_type.title():<20} "
                            f"Count: {stats['count']:>5}  "
                            f"Avg Loss: ${stats['avg_loss']:>10,.2f}  "
                            f"Severity: {stats['severity'].upper()}")
            report.append("")
        
        # Most common keywords
        if self.results['keywords']:
            report.append("TOP KEYWORDS IN FRAUD NARRATIVES")
            report.append("-" * 70)
            top_keywords = self.results['keywords'].most_common(15)
            for keyword, count in top_keywords:
                report.append(f"  {keyword:<25} {count:>5} occurrences")
            report.append("")
        
        report.append("=" * 70)
        report.append(f"Report generated: {datetime.utcnow().isoformat()}")
        report.append("=" * 70)
        
        return "\n".join(report)


def analyze_fraud_patterns(df: pd.DataFrame, 
                          narrative_column: str = 'narrative',
                          amount_column: str = 'amount') -> Dict:
    """Convenience function to analyze fraud patterns"""
    analyzer = AttackPatternAnalyzer()
    
    narratives = df[narrative_column].tolist() if narrative_column in df.columns else []
    amounts = df[amount_column].tolist() if amount_column in df.columns else None
    
    analysis = analyzer.analyze_narratives(narratives, amounts)
    
    reporter = AttackPatternReporter(analysis)
    report = reporter.generate_summary_report()
    print(report)
    
    return analysis, reporter


if __name__ == "__main__":
    print("Attack Pattern Analysis Module")
    print("This module is designed to be imported for fraud pattern detection")
