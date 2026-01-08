"""
Local database implementation using SQLite for SentinXFL v2.0

Provides:
- Transaction logging
- Model version tracking
- Organization profiles
- Audit trail storage
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LocalDatabase:
    """
    SQLite database wrapper for SentinXFL.
    
    Features:
    - Transaction logging
    - Model version history
    - Organization profiles
    - 7-year audit trail storage
    """
    
    def __init__(self, db_path: str = "data/sentinxfl.db"):
        """
        Initialize local database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_connection()
        self._init_tables()
        logger.info(f"LocalDatabase initialized: {db_path}")
    
    def _init_connection(self) -> None:
        """Initialize database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.debug(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _init_tables(self) -> None:
        """Create required tables if they don't exist."""
        if self.conn is None:
            raise RuntimeError("Database connection not initialized")
        
        cursor = self.conn.cursor()
        
        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                amount REAL,
                org_id TEXT,
                fraud_score REAL,
                decision TEXT,
                latency_ms REAL
            )
        """)
        
        # Model versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                round INTEGER PRIMARY KEY,
                version TEXT UNIQUE,
                accuracy REAL,
                f1_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Organization profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                org_id TEXT PRIMARY KEY,
                region TEXT,
                public_key TEXT,
                registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME
            )
        """)
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                org_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        
        self.conn.commit()
        logger.info("Database tables initialized")
    
    def log_transaction(self,
                       txn_id: str,
                       amount: float,
                       org_id: str,
                       fraud_score: float,
                       decision: str,
                       latency_ms: float = 0.0) -> bool:
        """
        Log a transaction to database.
        
        Args:
            txn_id: Transaction ID
            amount: Transaction amount
            org_id: Organization ID
            fraud_score: Fraud probability (0-1)
            decision: Decision (APPROVED/BLOCKED)
            latency_ms: Processing latency in milliseconds
            
        Returns:
            bool: True if successful
        """
        if self.conn is None:
            logger.error("Database not connected")
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (id, amount, org_id, fraud_score, decision, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (txn_id, amount, org_id, fraud_score, decision, latency_ms))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log transaction: {e}")
            return False
    
    def get_org_transactions(self, org_id: str, limit: int = 1000) -> List[Tuple]:
        """
        Retrieve organization's recent transactions.
        
        Args:
            org_id: Organization ID
            limit: Max transactions to retrieve
            
        Returns:
            List[Tuple]: List of transactions
        """
        if self.conn is None:
            logger.error("Database not connected")
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM transactions
                WHERE org_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (org_id, limit))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to retrieve transactions: {e}")
            return []
    
    def log_model_version(self,
                         round_num: int,
                         version: str,
                         accuracy: float,
                         f1_score: float) -> bool:
        """
        Log a model version.
        
        Args:
            round_num: FL round number
            version: Version identifier
            accuracy: Model accuracy
            f1_score: Model F1 score
            
        Returns:
            bool: True if successful
        """
        if self.conn is None:
            logger.error("Database not connected")
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO model_versions (round, version, accuracy, f1_score)
                VALUES (?, ?, ?, ?)
            """, (round_num, version, accuracy, f1_score))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log model version: {e}")
            return False
    
    def log_audit(self,
                 action: str,
                 org_id: str,
                 details: str = "") -> bool:
        """
        Log audit event.
        
        Args:
            action: Action description
            org_id: Organization ID
            details: Additional details
            
        Returns:
            bool: True if successful
        """
        if self.conn is None:
            logger.error("Database not connected")
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (action, org_id, details)
                VALUES (?, ?, ?)
            """, (action, org_id, details))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            dict: Statistics (transaction count, model versions, etc.)
        """
        if self.conn is None:
            return {}
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM transactions")
            txn_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM model_versions")
            model_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM organizations")
            org_count = cursor.fetchone()[0]
            
            return {
                'transactions': txn_count,
                'model_versions': model_count,
                'organizations': org_count,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"LocalDatabase(txns={stats.get('transactions', 0)}, models={stats.get('model_versions', 0)})"
