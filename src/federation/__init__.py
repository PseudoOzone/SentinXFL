"""
SentinXFL v2.0 - Federated Learning Module

This module handles multi-organization federated learning operations:
- Organization management and registration
- Async aggregation protocols
- Checkpoint and recovery mechanisms
"""

from .multi_org_manager import OrganizationManager, Organization

__all__ = [
    'OrganizationManager',
    'Organization',
]

__version__ = '2.0.0'
