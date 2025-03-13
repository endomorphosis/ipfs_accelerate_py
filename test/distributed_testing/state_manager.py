#!/usr/bin/env python3
"""
State Manager for Distributed Testing Framework

This module manages state across distributed components, providing a consistent
view of shared state with transactions and conflict resolution.
"""

import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StateManager:
    """
    Manager for distributed state with transactions and conflict resolution
    """
    
    def __init__(self, state_id: str = "default", persistence_path: str = None):
        """
        Initialize state manager
        
        Args:
            state_id: Unique identifier for this state instance
            persistence_path: Optional path for state persistence
        """
        self.state_id = state_id
        self.persistence_path = persistence_path
        
        # State storage
        self.state = {}
        self.state_versions = {}  # key -> version
        self.state_timestamps = {}  # key -> timestamp
        
        # Transaction log
        self.transactions = []
        
        # Create logger
        self.logger = logging.getLogger(f"state_manager.{state_id}")
        self.logger.info(f"State manager {state_id} initialized")
    
    async def update_state(self, key: str, value: Any) -> bool:
        """
        Update state for a key
        
        Args:
            key: State key
            value: New value
            
        Returns:
            Update success
        """
        # Record current time
        current_time = time.time()
        
        # Check if key exists
        is_new = key not in self.state
        
        # Update version
        if key in self.state_versions:
            self.state_versions[key] += 1
        else:
            self.state_versions[key] = 1
        
        # Update state
        self.state[key] = value
        self.state_timestamps[key] = current_time
        
        # Record transaction
        transaction = {
            "action": "create" if is_new else "update",
            "key": key,
            "version": self.state_versions[key],
            "timestamp": current_time
        }
        self.transactions.append(transaction)
        
        # Persist state if path is set
        if self.persistence_path:
            await self._persist_state()
        
        self.logger.info(f"State {key} {'created' if is_new else 'updated'} to version {self.state_versions[key]}")
        return True
    
    async def get_state(self, key: str) -> Optional[Any]:
        """
        Get current state for a key
        
        Args:
            key: State key
            
        Returns:
            Current state value or None if not found
        """
        if key not in self.state:
            return None
        
        return self.state[key]
    
    async def get_state_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a state key
        
        Args:
            key: State key
            
        Returns:
            State metadata or None if not found
        """
        if key not in self.state:
            return None
        
        return {
            "key": key,
            "version": self.state_versions.get(key, 0),
            "last_updated": self.state_timestamps.get(key, 0),
            "time_since_update": time.time() - self.state_timestamps.get(key, 0)
        }
    
    async def delete_state(self, key: str) -> bool:
        """
        Delete state for a key
        
        Args:
            key: State key to delete
            
        Returns:
            Deletion success
        """
        if key not in self.state:
            self.logger.warning(f"State {key} not found for deletion")
            return False
        
        # Record current time
        current_time = time.time()
        
        # Remove state
        del self.state[key]
        
        # Record transaction
        transaction = {
            "action": "delete",
            "key": key,
            "version": self.state_versions.get(key, 0),
            "timestamp": current_time
        }
        self.transactions.append(transaction)
        
        # Remove metadata
        if key in self.state_versions:
            del self.state_versions[key]
        
        if key in self.state_timestamps:
            del self.state_timestamps[key]
        
        # Persist state if path is set
        if self.persistence_path:
            await self._persist_state()
        
        self.logger.info(f"State {key} deleted")
        return True
    
    async def get_all_state(self) -> Dict[str, Any]:
        """
        Get all current state
        
        Returns:
            Dictionary of all state
        """
        return dict(self.state)
    
    async def get_all_state_with_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all state with metadata
        
        Returns:
            Dictionary of state keys to objects with value and metadata
        """
        result = {}
        
        for key in self.state:
            metadata = await self.get_state_metadata(key)
            result[key] = {
                "value": self.state[key],
                "metadata": metadata
            }
        
        return result
    
    async def get_transaction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get transaction history
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions
        """
        return self.transactions[-limit:] if limit > 0 else list(self.transactions)
    
    async def _persist_state(self) -> bool:
        """
        Persist state to disk
        
        Returns:
            Persistence success
        """
        if not self.persistence_path:
            return False
        
        try:
            # Create state snapshot
            snapshot = {
                "state_id": self.state_id,
                "timestamp": time.time(),
                "state": self.state,
                "metadata": {
                    "versions": self.state_versions,
                    "timestamps": self.state_timestamps
                }
            }
            
            # Write to file
            async with asyncio.Lock():
                with open(f"{self.persistence_path}/{self.state_id}_state.json", "w") as f:
                    json.dump(snapshot, f)
            
            return True
        except Exception as e:
            self.logger.error(f"Error persisting state: {e}")
            return False
    
    async def load_state_from_disk(self) -> bool:
        """
        Load state from disk
        
        Returns:
            Load success
        """
        if not self.persistence_path:
            return False
        
        try:
            # Read from file
            with open(f"{self.persistence_path}/{self.state_id}_state.json", "r") as f:
                snapshot = json.load(f)
            
            # Restore state
            self.state = snapshot["state"]
            self.state_versions = snapshot["metadata"]["versions"]
            self.state_timestamps = snapshot["metadata"]["timestamps"]
            
            self.logger.info(f"State loaded from disk with {len(self.state)} keys")
            return True
        except FileNotFoundError:
            self.logger.warning(f"No state file found at {self.persistence_path}/{self.state_id}_state.json")
            return False
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False