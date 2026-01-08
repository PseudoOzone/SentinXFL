"""
Database module for SentinXFL v2.0

Provides persistent storage and caching:
- SQLite for transaction logging
- Redis for caching (optional)
"""

__all__ = [
    'LocalDatabase',
    'CacheManager',
]

__version__ = '2.0.0'
