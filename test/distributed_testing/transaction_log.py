#!/usr/bin/env python3
"""
Transaction Log for Distributed Testing Framework

This module provides a transaction log for recording operations and enabling
recovery in distributed systems.
"""

import time
import json
import logging
import anyio
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionLog:
    """
    Transaction log for recording operations and enabling recovery
    """
    
    def __init__(self, log_id: str = "default", persistence_path: str = None, max_entries: int = 1000):
        """
        Initialize transaction log
        
        Args:
            log_id: Unique identifier for this log
            persistence_path: Optional path for log persistence
            max_entries: Maximum number of entries to keep in memory
        """
        self.log_id = log_id
        self.persistence_path = persistence_path
        self.max_entries = max_entries
        
        # Transaction storage
        self.transactions = []
        
        # Create logger
        self.logger = logging.getLogger(f"transaction_log.{log_id}")
        self.logger.info(f"Transaction log {log_id} initialized")
    
    async def append(self, transaction: Dict[str, Any]) -> bool:
        """
        Append a transaction to the log
        
        Args:
            transaction: Transaction to record
            
        Returns:
            Append success
        """
        # Add timestamp if not present
        if "timestamp" not in transaction:
            transaction["timestamp"] = time.time()
        
        # Add transaction ID if not present
        if "transaction_id" not in transaction:
            transaction["transaction_id"] = f"{self.log_id}_{len(self.transactions) + 1}"
        
        # Append to log
        self.transactions.append(transaction)
        
        # Trim log if needed
        if len(self.transactions) > self.max_entries:
            self.transactions = self.transactions[-self.max_entries:]
        
        # Persist log if path is set
        if self.persistence_path:
            await self._persist_log()
        
        return True
    
    async def get_latest(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get latest transactions
        
        Args:
            count: Number of latest transactions to return
            
        Returns:
            List of latest transactions
        """
        return self.transactions[-count:] if self.transactions and count > 0 else []
    
    async def get_transactions(self, start_time: float = None, end_time: float = None, 
                             action_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get transactions matching criteria
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            action_type: Optional action type filter
            limit: Maximum number of transactions to return
            
        Returns:
            List of matching transactions
        """
        # Apply filters
        filtered = self.transactions
        
        if start_time is not None:
            filtered = [t for t in filtered if t.get("timestamp", 0) >= start_time]
        
        if end_time is not None:
            filtered = [t for t in filtered if t.get("timestamp", 0) <= end_time]
        
        if action_type is not None:
            filtered = [t for t in filtered if t.get("action") == action_type]
        
        # Apply limit
        filtered = filtered[-limit:] if limit > 0 else filtered
        
        return filtered
    
    async def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the transaction log
        
        Returns:
            Dictionary with log statistics
        """
        # Count actions
        action_counts = {}
        for transaction in self.transactions:
            action = transaction.get("action", "unknown")
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        # Get time range
        if self.transactions:
            first_time = min(t.get("timestamp", 0) for t in self.transactions)
            last_time = max(t.get("timestamp", 0) for t in self.transactions)
            time_range = last_time - first_time
        else:
            first_time = 0
            last_time = 0
            time_range = 0
        
        # Generate stats
        stats = {
            "log_id": self.log_id,
            "total_transactions": len(self.transactions),
            "action_distribution": action_counts,
            "first_transaction_time": first_time,
            "last_transaction_time": last_time,
            "time_range_seconds": time_range,
            "transactions_per_second": len(self.transactions) / time_range if time_range > 0 else 0
        }
        
        return stats
    
    async def clear(self) -> bool:
        """
        Clear the transaction log
        
        Returns:
            Clear success
        """
        self.transactions = []
        
        # Persist empty log if path is set
        if self.persistence_path:
            await self._persist_log()
        
        self.logger.info(f"Transaction log {self.log_id} cleared")
        return True
    
    async def _persist_log(self) -> bool:
        """
        Persist log to disk
        
        Returns:
            Persistence success
        """
        if not self.persistence_path:
            return False
        
        try:
            # Create log snapshot
            snapshot = {
                "log_id": self.log_id,
                "timestamp": time.time(),
                "transactions": self.transactions
            }
            
            # Write to file
            async with anyio.Lock():
                with open(f"{self.persistence_path}/{self.log_id}_transactions.json", "w") as f:
                    json.dump(snapshot, f)
            
            return True
        except Exception as e:
            self.logger.error(f"Error persisting transaction log: {e}")
            return False
    
    async def load_log_from_disk(self) -> bool:
        """
        Load log from disk
        
        Returns:
            Load success
        """
        if not self.persistence_path:
            return False
        
        try:
            # Read from file
            with open(f"{self.persistence_path}/{self.log_id}_transactions.json", "r") as f:
                snapshot = json.load(f)
            
            # Restore log
            self.transactions = snapshot["transactions"]
            
            self.logger.info(f"Transaction log loaded from disk with {len(self.transactions)} entries")
            return True
        except FileNotFoundError:
            self.logger.warning(f"No transaction log file found at {self.persistence_path}/{self.log_id}_transactions.json")
            return False
        except Exception as e:
            self.logger.error(f"Error loading transaction log: {e}")
            return False