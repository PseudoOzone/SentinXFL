"""
SentinXFL - Central Knowledge & DP Pattern Library
====================================================

Continuously-evolving fraud pattern knowledge base that learns
from federated training rounds and provides fact-based emergent
attack intelligence to all connected banks.

Modules:
    - pattern_library: Core pattern storage and retrieval
    - pattern_miner: Automated pattern extraction from FL rounds
    - emergent_detector: Zero-day and emergent attack detection
    - central_model: Central knowledge model with versioning
    - report_generator: Industry-standard report generation

Author: Anshuman Bakshi
"""

from sentinxfl.intelligence.pattern_library import PatternLibrary, PatternEntry, PatternType, PatternSeverity, PatternStatus
from sentinxfl.intelligence.emergent_detector import EmergentDetector, EmergentAlert
from sentinxfl.intelligence.central_model import CentralKnowledgeModel, KnowledgeSnapshot, BankProfile
from sentinxfl.intelligence.report_generator import ReportGenerator, Report
from sentinxfl.intelligence.pattern_miner import PatternMiner, MiningResult

__all__ = [
    "PatternLibrary",
    "PatternEntry",
    "PatternType",
    "PatternSeverity",
    "PatternStatus",
    "EmergentDetector",
    "EmergentAlert",
    "CentralKnowledgeModel",
    "KnowledgeSnapshot",
    "BankProfile",
    "ReportGenerator",
    "Report",
    "PatternMiner",
    "MiningResult",
]
