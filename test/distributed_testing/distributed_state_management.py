#!/usr/bin/env python3
"""
Distributed Testing Framework - Distributed State Management Module

This module implements distributed state management for the distributed testing framework.
It provides a reliable way to maintain consistent state across coordinator nodes,
with support for state replication, consistency checking, and recovery.

Key features:
- State replication between coordinators
- State partitioning for efficiency
- Consistency checking and conflict resolution
- Automatic state recovery after failures
- Transaction-based state updates
- Support for partial updates and delta synchronization

Usage:
    Import this module in coordinator.py to enable distributed state management.
"""

import anyio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import aiohttp
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dist_state_manager")

class StatePartition:
    """Represents a partition of the distributed state."""
    
    def __init__(self, name: str, priority: int = 0):
        """
        Initialize a state partition.
        
        Args:
            name: Partition name
            priority: Replication priority (higher values replicate first)
        """
        self.name = name
        self.priority = priority
        self.data = {}
        self.version = 0
        self.last_modified = time.time()
        self.checksum = ""
        self.transaction_log = []
        self.max_transaction_log = 1000  # Maximum number of transactions to keep
    
    def update(self, key: str, value: Any, transaction_id: str = None) -> str:
        """
        Update a value in the partition.
        
        Args:
            key: Key to update
            value: New value
            transaction_id: Optional transaction ID
            
        Returns:
            Transaction ID
        """
        # Generate transaction ID if not provided
        transaction_id = transaction_id or f"tx-{uuid.uuid4().hex[:8]}"
        
        # Record transaction
        transaction = {
            "id": transaction_id,
            "type": "update",
            "key": key,
            "old_value": self.data.get(key),
            "new_value": value,
            "timestamp": time.time()
        }
        
        # Update data
        self.data[key] = value
        self.version += 1
        self.last_modified = time.time()
        
        # Add transaction to log
        self.transaction_log.append(transaction)
        
        # Trim transaction log if needed
        if len(self.transaction_log) > self.max_transaction_log:
            self.transaction_log = self.transaction_log[-self.max_transaction_log:]
        
        # Update checksum
        self._update_checksum()
        
        return transaction_id
    
    def delete(self, key: str, transaction_id: str = None) -> str:
        """
        Delete a value from the partition.
        
        Args:
            key: Key to delete
            transaction_id: Optional transaction ID
            
        Returns:
            Transaction ID
        """
        # Generate transaction ID if not provided
        transaction_id = transaction_id or f"tx-{uuid.uuid4().hex[:8]}"
        
        # Record transaction
        transaction = {
            "id": transaction_id,
            "type": "delete",
            "key": key,
            "old_value": self.data.get(key),
            "timestamp": time.time()
        }
        
        # Delete key if it exists
        if key in self.data:
            del self.data[key]
            self.version += 1
            self.last_modified = time.time()
            
            # Add transaction to log
            self.transaction_log.append(transaction)
            
            # Trim transaction log if needed
            if len(self.transaction_log) > self.max_transaction_log:
                self.transaction_log = self.transaction_log[-self.max_transaction_log:]
            
            # Update checksum
            self._update_checksum()
        
        return transaction_id
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the partition.
        
        Args:
            key: Key to get
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        return self.data.get(key, default)
    
    def contains(self, key: str) -> bool:
        """
        Check if partition contains a key.
        
        Args:
            key: Key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.data
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all data in the partition.
        
        Returns:
            Dictionary with all data
        """
        return self.data.copy()
    
    def clear(self, transaction_id: str = None) -> str:
        """
        Clear all data in the partition.
        
        Args:
            transaction_id: Optional transaction ID
            
        Returns:
            Transaction ID
        """
        # Generate transaction ID if not provided
        transaction_id = transaction_id or f"tx-{uuid.uuid4().hex[:8]}"
        
        # Record transaction
        transaction = {
            "id": transaction_id,
            "type": "clear",
            "old_data": self.data.copy(),
            "timestamp": time.time()
        }
        
        # Clear data
        self.data = {}
        self.version += 1
        self.last_modified = time.time()
        
        # Add transaction to log
        self.transaction_log.append(transaction)
        
        # Trim transaction log if needed
        if len(self.transaction_log) > self.max_transaction_log:
            self.transaction_log = self.transaction_log[-self.max_transaction_log:]
        
        # Update checksum
        self._update_checksum()
        
        return transaction_id
    
    def update_batch(self, updates: Dict[str, Any], transaction_id: str = None) -> str:
        """
        Update multiple values in the partition.
        
        Args:
            updates: Dictionary of key-value pairs to update
            transaction_id: Optional transaction ID
            
        Returns:
            Transaction ID
        """
        # Generate transaction ID if not provided
        transaction_id = transaction_id or f"tx-{uuid.uuid4().hex[:8]}"
        
        # Record transaction
        transaction = {
            "id": transaction_id,
            "type": "batch_update",
            "updates": updates,
            "old_values": {key: self.data.get(key) for key in updates},
            "timestamp": time.time()
        }
        
        # Update data
        for key, value in updates.items():
            self.data[key] = value
        
        self.version += 1
        self.last_modified = time.time()
        
        # Add transaction to log
        self.transaction_log.append(transaction)
        
        # Trim transaction log if needed
        if len(self.transaction_log) > self.max_transaction_log:
            self.transaction_log = self.transaction_log[-self.max_transaction_log:]
        
        # Update checksum
        self._update_checksum()
        
        return transaction_id
    
    def apply_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Apply a transaction to the partition.
        
        Args:
            transaction: Transaction to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            transaction_type = transaction.get("type")
            transaction_id = transaction.get("id")
            
            # Check if transaction already applied
            for tx in self.transaction_log:
                if tx.get("id") == transaction_id:
                    logger.debug(f"Transaction {transaction_id} already applied, skipping")
                    return True
            
            if transaction_type == "update":
                key = transaction.get("key")
                value = transaction.get("new_value")
                self.update(key, value, transaction_id)
                return True
            elif transaction_type == "delete":
                key = transaction.get("key")
                self.delete(key, transaction_id)
                return True
            elif transaction_type == "clear":
                self.clear(transaction_id)
                return True
            elif transaction_type == "batch_update":
                updates = transaction.get("updates", {})
                self.update_batch(updates, transaction_id)
                return True
            else:
                logger.warning(f"Unknown transaction type: {transaction_type}")
                return False
        except Exception as e:
            logger.error(f"Error applying transaction: {e}")
            return False
    
    def get_transactions_since(self, timestamp: float) -> List[Dict[str, Any]]:
        """
        Get transactions since a timestamp.
        
        Args:
            timestamp: Timestamp to get transactions since
            
        Returns:
            List of transactions
        """
        return [tx for tx in self.transaction_log if tx.get("timestamp", 0) > timestamp]
    
    def size(self) -> int:
        """
        Get the size of the partition.
        
        Returns:
            Number of key-value pairs
        """
        return len(self.data)
    
    def merge(self, other: 'StatePartition') -> List[str]:
        """
        Merge another partition into this one.
        
        Args:
            other: Other partition to merge
            
        Returns:
            List of conflicting keys
        """
        conflicts = []
        
        # Apply all transactions from other partition
        for transaction in other.transaction_log:
            # Skip already applied transactions
            if any(tx.get("id") == transaction.get("id") for tx in self.transaction_log):
                continue
            
            # Apply transaction
            self.apply_transaction(transaction)
        
        # Check for conflicts
        for key, value in other.data.items():
            if key in self.data and self.data[key] != value:
                conflicts.append(key)
        
        # If conflicts, use more recent values
        if conflicts and other.last_modified > self.last_modified:
            for key in conflicts:
                self.data[key] = other.data[key]
        
        # Update version and checksum
        self.version = max(self.version, other.version) + 1
        self.last_modified = time.time()
        self._update_checksum()
        
        return conflicts
    
    def _update_checksum(self):
        """Update the checksum of the partition."""
        try:
            # Convert data to sorted JSON string
            data_str = json.dumps(self.data, sort_keys=True)
            
            # Calculate checksum
            self.checksum = hashlib.sha256(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error updating checksum: {e}")

class DistributedStateManager:
    """Manages distributed state across coordinator nodes."""
    
    def __init__(
        self, 
        coordinator, 
        cluster_nodes: List[str], 
        node_id: str,
        state_dir: str = None,
        sync_interval: float = 5.0,
        partition_config: Dict[str, int] = None,
        max_sync_delay: float = 1.0
    ):
        """
        Initialize the distributed state manager.
        
        Args:
            coordinator: Reference to the coordinator instance
            cluster_nodes: List of node URLs in the cluster
            node_id: Unique identifier for this node
            state_dir: Directory to store state files (default: ./state)
            sync_interval: Interval between state synchronization in seconds
            partition_config: Partition names and priorities
            max_sync_delay: Maximum delay between syncs when changes are detected
        """
        self.coordinator = coordinator
        self.cluster_nodes = cluster_nodes
        self.node_id = node_id
        self.node_url = self._get_node_url(node_id, cluster_nodes)
        
        # State directory
        self.state_dir = state_dir or os.path.join(os.path.dirname(coordinator.db_path) if coordinator.db_path else ".", "state")
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Timing parameters
        self.sync_interval = sync_interval
        self.max_sync_delay = max_sync_delay
        
        # Default partition configuration
        default_partitions = {
            "workers": 10,
            "tasks": 9,
            "task_history": 7,
            "system_health": 8,
            "configuration": 10
        }
        
        # Use provided partition config or default
        self.partition_config = partition_config or default_partitions
        
        # Initialize partitions
        self.partitions: Dict[str, StatePartition] = {}
        for name, priority in self.partition_config.items():
            self.partitions[name] = StatePartition(name, priority)
        
        # Synchronization state
        self.last_sync_time = 0
        self.sync_in_progress = False
        self.changes_pending = False
        
        # Connection management
        self.session = None
        self.running = False
        self.tasks = set()
        self._task_group = None
        
        # Load state from disk
        self._load_state_from_disk()
        
        logger.info(f"DistributedStateManager initialized with {len(self.partitions)} partitions")
    
    def _get_node_url(self, node_id: str, cluster_nodes: List[str]) -> str:
        """Get the URL for a node.
        
        Args:
            node_id: Node ID
            cluster_nodes: List of node URLs
            
        Returns:
            URL for the node
        """
        for node in cluster_nodes:
            if node_id in node:
                return node
        
        # If node_id is not found in URLs, use the first one
        return cluster_nodes[0] if cluster_nodes else None
    
    def _load_state_from_disk(self):
        """Load state from disk."""
        for partition_name in self.partitions:
            file_path = os.path.join(self.state_dir, f"{partition_name}.json")
            
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # Restore partition data
                        partition = self.partitions[partition_name]
                        partition.data = data.get("data", {})
                        partition.version = data.get("version", 0)
                        partition.last_modified = data.get("last_modified", time.time())
                        partition.checksum = data.get("checksum", "")
                        partition.transaction_log = data.get("transaction_log", [])
                        
                        logger.info(f"Loaded partition {partition_name} from disk: {len(partition.data)} items, version {partition.version}")
            except Exception as e:
                logger.error(f"Error loading partition {partition_name} from disk: {e}")
    
    def _save_state_to_disk(self, partition_name: str = None):
        """
        Save state to disk.
        
        Args:
            partition_name: Name of specific partition to save, or None for all
        """
        if partition_name:
            partitions = [partition_name]
        else:
            partitions = self.partitions.keys()
        
        for name in partitions:
            if name not in self.partitions:
                continue
                
            partition = self.partitions[name]
            file_path = os.path.join(self.state_dir, f"{name}.json")
            
            try:
                # Prepare data
                data = {
                    "data": partition.data,
                    "version": partition.version,
                    "last_modified": partition.last_modified,
                    "checksum": partition.checksum,
                    "transaction_log": partition.transaction_log
                }
                
                # Write to file
                with open(file_path, 'w') as f:
                    json.dump(data, f)
                    
                logger.debug(f"Saved partition {name} to disk: {len(partition.data)} items, version {partition.version}")
            except Exception as e:
                logger.error(f"Error saving partition {name} to disk: {e}")
    
    async def start(self):
        """Start the distributed state manager."""
        if self.running:
            return
            
        self.running = True
        
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        
        # Start sync loop (anyio task group)
        if self._task_group is None:
            self._task_group = anyio.create_task_group()
            await self._task_group.__aenter__()
        self._task_group.start_soon(self._sync_loop)
        
        logger.info(f"DistributedStateManager started for node {self.node_id}")
    
    async def stop(self):
        """Stop the distributed state manager."""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel background tasks
        if self._task_group is not None:
            self._task_group.cancel_scope.cancel()
            await self._task_group.__aexit__(None, None, None)
            self._task_group = None
        self.tasks.clear()
        
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Save state to disk
        self._save_state_to_disk()
            
        logger.info(f"DistributedStateManager stopped for node {self.node_id}")
    
    async def _sync_loop(self):
        """Synchronization loop."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if sync is needed
                if (current_time - self.last_sync_time >= self.sync_interval or 
                    (self.changes_pending and current_time - self.last_sync_time >= self.max_sync_delay)):
                    
                    if not self.sync_in_progress:
                        self.sync_in_progress = True
                        self.changes_pending = False
                        await self._synchronize_state()
                        self.last_sync_time = time.time()
                        self.sync_in_progress = False
                
            except anyio.get_cancelled_exc_class():
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                self.sync_in_progress = False
                
            # Sleep until next check
            await anyio.sleep(0.5)
    
    async def _synchronize_state(self):
        """Synchronize state with other nodes."""
        logger.debug("Starting state synchronization")
        
        # Get sorted list of partitions by priority
        partition_names = sorted(
            self.partitions.keys(),
            key=lambda name: self.partitions[name].priority,
            reverse=True
        )
        
        # Synchronize each partition
        for partition_name in partition_names:
            await self._sync_partition(partition_name)
            
        # Save state to disk
        self._save_state_to_disk()
        
        logger.debug("State synchronization completed")
    
    async def _sync_partition(self, partition_name: str):
        """
        Synchronize a specific partition.
        
        Args:
            partition_name: Name of partition to synchronize
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return
            
        logger.debug(f"Synchronizing partition {partition_name}")
        
        # Get checksums from all nodes
        checksums = await self._get_partition_checksums(partition_name)
        
        # Check if all checksums match
        if checksums and all(checksum == partition.checksum for node_id, checksum in checksums.items() if node_id != self.node_id):
            logger.debug(f"Partition {partition_name} is consistent across all nodes")
            return
            
        # Get nodes with different checksums
        nodes_to_sync = [node_id for node_id, checksum in checksums.items() 
                          if checksum != partition.checksum and node_id != self.node_id]
        
        if not nodes_to_sync:
            logger.debug(f"No nodes to synchronize for partition {partition_name}")
            return
            
        # Synchronize with each node
        for node_id in nodes_to_sync:
            node_url = self._get_node_url(node_id, self.cluster_nodes)
            if not node_url:
                continue
                
            await self._sync_with_node(node_url, partition_name)
    
    async def _get_partition_checksums(self, partition_name: str) -> Dict[str, str]:
        """
        Get checksums for a partition from all nodes.
        
        Args:
            partition_name: Name of partition
            
        Returns:
            Dictionary mapping node IDs to checksums
        """
        checksums = {self.node_id: self.partitions[partition_name].checksum}
        
        for node in self.cluster_nodes:
            if self.node_id in node:
                continue  # Skip self
                
            try:
                async with self.session.get(
                    f"{node}/state/checksum/{partition_name}",
                    timeout=2
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        node_id = result.get("node_id")
                        checksum = result.get("checksum")
                        
                        if node_id and checksum:
                            checksums[node_id] = checksum
            except Exception as e:
                logger.debug(f"Error getting checksum from {node}: {e}")
                
        return checksums
    
    async def _sync_with_node(self, node_url: str, partition_name: str):
        """
        Synchronize a partition with another node.
        
        Args:
            node_url: URL of node to sync with
            partition_name: Name of partition to sync
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return
            
        try:
            # First, try to get only transactions since our last modification
            since = partition.last_modified
            
            async with self.session.get(
                f"{node_url}/state/transactions/{partition_name}",
                params={"since": since},
                timeout=5
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    transactions = result.get("transactions", [])
                    remote_version = result.get("version", 0)
                    
                    # Apply transactions
                    applied = 0
                    for transaction in transactions:
                        if partition.apply_transaction(transaction):
                            applied += 1
                    
                    if applied > 0:
                        logger.info(f"Applied {applied} transactions from {node_url} for partition {partition_name}")
                        
                        # Update version if remote is higher
                        if remote_version > partition.version:
                            partition.version = remote_version
                        else:
                            partition.version += 1
                            
                        partition.last_modified = time.time()
                        partition._update_checksum()
                        
                        # Save partition to disk
                        self._save_state_to_disk(partition_name)
                        return
            
            # If that didn't work, get full partition data
            async with self.session.get(
                f"{node_url}/state/partition/{partition_name}",
                timeout=5
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("data", {})
                    version = result.get("version", 0)
                    last_modified = result.get("last_modified", time.time())
                    
                    # Compare versions
                    if version > partition.version or (version == partition.version and last_modified > partition.last_modified):
                        # Remote version is newer, update our data
                        partition.data = data
                        partition.version = version
                        partition.last_modified = time.time()
                        partition._update_checksum()
                        
                        logger.info(f"Updated partition {partition_name} from {node_url}")
                        
                        # Save partition to disk
                        self._save_state_to_disk(partition_name)
                    elif version < partition.version or (version == partition.version and last_modified < partition.last_modified):
                        # Our version is newer, push our data
                        await self._push_partition(node_url, partition_name)
        except Exception as e:
            logger.error(f"Error syncing partition {partition_name} with {node_url}: {e}")
    
    async def _push_partition(self, node_url: str, partition_name: str):
        """
        Push a partition to another node.
        
        Args:
            node_url: URL of node to push to
            partition_name: Name of partition to push
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return
            
        try:
            # Prepare data
            data = {
                "partition_name": partition_name,
                "data": partition.data,
                "version": partition.version,
                "last_modified": partition.last_modified,
                "checksum": partition.checksum,
                "source_node": self.node_id
            }
            
            async with self.session.post(
                f"{node_url}/state/update",
                json=data,
                timeout=5
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully pushed partition {partition_name} to {node_url}")
                else:
                    logger.warning(f"Failed to push partition {partition_name} to {node_url}: {response.status}")
        except Exception as e:
            logger.error(f"Error pushing partition {partition_name} to {node_url}: {e}")
    
    async def handle_checksum_request(self, partition_name: str):
        """
        Handle a checksum request.
        
        Args:
            partition_name: Name of requested partition
            
        Returns:
            Response with checksum information
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return {
                "error": f"Partition {partition_name} not found",
                "node_id": self.node_id
            }
            
        return {
            "node_id": self.node_id,
            "partition_name": partition_name,
            "checksum": partition.checksum,
            "version": partition.version,
            "last_modified": partition.last_modified
        }
    
    async def handle_transactions_request(self, partition_name: str, since: float):
        """
        Handle a transactions request.
        
        Args:
            partition_name: Name of requested partition
            since: Timestamp to get transactions since
            
        Returns:
            Response with transactions information
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return {
                "error": f"Partition {partition_name} not found",
                "node_id": self.node_id
            }
            
        transactions = partition.get_transactions_since(since)
            
        return {
            "node_id": self.node_id,
            "partition_name": partition_name,
            "version": partition.version,
            "last_modified": partition.last_modified,
            "transactions": transactions,
            "transaction_count": len(transactions)
        }
    
    async def handle_partition_request(self, partition_name: str):
        """
        Handle a partition request.
        
        Args:
            partition_name: Name of requested partition
            
        Returns:
            Response with partition information
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return {
                "error": f"Partition {partition_name} not found",
                "node_id": self.node_id
            }
            
        return {
            "node_id": self.node_id,
            "partition_name": partition_name,
            "data": partition.data,
            "version": partition.version,
            "last_modified": partition.last_modified,
            "checksum": partition.checksum,
            "size": partition.size()
        }
    
    async def handle_update_request(self, request):
        """
        Handle a partition update request.
        
        Args:
            request: Update request
            
        Returns:
            Response with update status
        """
        partition_name = request.get("partition_name")
        data = request.get("data")
        version = request.get("version")
        last_modified = request.get("last_modified")
        checksum = request.get("checksum")
        source_node = request.get("source_node")
        
        partition = self.partitions.get(partition_name)
        if not partition:
            return {
                "status": "error",
                "error": f"Partition {partition_name} not found",
                "node_id": self.node_id
            }
            
        # Check if update is newer
        if version > partition.version or (version == partition.version and last_modified > partition.last_modified):
            # Update partition
            partition.data = data
            partition.version = version + 1  # Increment version
            partition.last_modified = time.time()
            partition._update_checksum()
            
            # Save partition to disk
            self._save_state_to_disk(partition_name)
            
            logger.info(f"Updated partition {partition_name} from node {source_node}")
            
            return {
                "status": "success",
                "node_id": self.node_id,
                "partition_name": partition_name,
                "new_version": partition.version
            }
        else:
            # Our version is newer or same
            return {
                "status": "rejected",
                "node_id": self.node_id,
                "partition_name": partition_name,
                "reason": "local_version_newer",
                "local_version": partition.version,
                "received_version": version
            }
    
    def get(self, partition_name: str, key: str, default: Any = None) -> Any:
        """
        Get a value from a partition.
        
        Args:
            partition_name: Name of partition
            key: Key to get
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return default
            
        return partition.get(key, default)
    
    def update(self, partition_name: str, key: str, value: Any) -> bool:
        """
        Update a value in a partition.
        
        Args:
            partition_name: Name of partition
            key: Key to update
            value: New value
            
        Returns:
            True if successful, False otherwise
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return False
            
        partition.update(key, value)
        self.changes_pending = True
        
        return True
    
    def delete(self, partition_name: str, key: str) -> bool:
        """
        Delete a value from a partition.
        
        Args:
            partition_name: Name of partition
            key: Key to delete
            
        Returns:
            True if successful, False otherwise
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return False
            
        partition.delete(key)
        self.changes_pending = True
        
        return True
    
    def update_batch(self, partition_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update multiple values in a partition.
        
        Args:
            partition_name: Name of partition
            updates: Dictionary of key-value pairs to update
            
        Returns:
            True if successful, False otherwise
        """
        partition = self.partitions.get(partition_name)
        if not partition:
            return False
            
        partition.update_batch(updates)
        self.changes_pending = True
        
        return True
    
    def register_worker(self, worker_id: str, worker_data: Dict[str, Any]) -> bool:
        """
        Register a worker in the distributed state.
        
        Args:
            worker_id: Worker ID
            worker_data: Worker data
            
        Returns:
            True if successful, False otherwise
        """
        return self.update("workers", worker_id, worker_data)
    
    def update_worker_status(self, worker_id: str, status: str) -> bool:
        """
        Update worker status in the distributed state.
        
        Args:
            worker_id: Worker ID
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        worker_data = self.get("workers", worker_id)
        if not worker_data:
            return False
            
        worker_data["status"] = status
        worker_data["last_updated"] = time.time()
        
        return self.update("workers", worker_id, worker_data)
    
    def get_worker(self, worker_id: str) -> Dict[str, Any]:
        """
        Get worker data from the distributed state.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Worker data or None
        """
        return self.get("workers", worker_id)
    
    def get_all_workers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all workers from the distributed state.
        
        Returns:
            Dictionary mapping worker IDs to worker data
        """
        partition = self.partitions.get("workers")
        if not partition:
            return {}
            
        return partition.get_all()
    
    def register_task(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """
        Register a task in the distributed state.
        
        Args:
            task_id: Task ID
            task_data: Task data
            
        Returns:
            True if successful, False otherwise
        """
        return self.update("tasks", task_id, task_data)
    
    def update_task_status(self, task_id: str, status: str, worker_id: str = None) -> bool:
        """
        Update task status in the distributed state.
        
        Args:
            task_id: Task ID
            status: New status
            worker_id: Worker ID (if assigned)
            
        Returns:
            True if successful, False otherwise
        """
        task_data = self.get("tasks", task_id)
        if not task_data:
            return False
            
        task_data["status"] = status
        task_data["last_updated"] = time.time()
        
        if worker_id:
            task_data["worker_id"] = worker_id
            
        if status == "completed" or status == "failed":
            task_data["end_time"] = time.time()
            
            # Add to task history
            task_history = task_data.copy()
            self.update("task_history", task_id, task_history)
            
        return self.update("tasks", task_id, task_data)
    
    def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Get task data from the distributed state.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task data or None
        """
        return self.get("tasks", task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tasks from the distributed state.
        
        Returns:
            Dictionary mapping task IDs to task data
        """
        partition = self.partitions.get("tasks")
        if not partition:
            return {}
            
        return partition.get_all()
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health information from the distributed state.
        
        Returns:
            System health information
        """
        return self.get("system_health", "current", {})
    
    def update_system_health(self, health_data: Dict[str, Any]) -> bool:
        """
        Update system health information in the distributed state.
        
        Args:
            health_data: Health data
            
        Returns:
            True if successful, False otherwise
        """
        health_data["last_updated"] = time.time()
        
        # Keep history
        timestamp = datetime.now().isoformat()
        self.update("system_health", f"history_{timestamp}", health_data)
        
        # Update current
        return self.update("system_health", "current", health_data)
    
    def get_configuration(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value from the distributed state.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.get("configuration", key, default)
    
    def update_configuration(self, key: str, value: Any) -> bool:
        """
        Update configuration value in the distributed state.
        
        Args:
            key: Configuration key
            value: New value
            
        Returns:
            True if successful, False otherwise
        """
        return self.update("configuration", key, value)
    
    def get_all_configuration(self) -> Dict[str, Any]:
        """
        Get all configuration values from the distributed state.
        
        Returns:
            Dictionary mapping configuration keys to values
        """
        partition = self.partitions.get("configuration")
        if not partition:
            return {}
            
        return partition.get_all()
    
    def create_snapshot(self) -> str:
        """
        Create a snapshot of the distributed state.
        
        Returns:
            Path to snapshot file
        """
        snapshot_dir = os.path.join(self.state_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        snapshot_file = os.path.join(snapshot_dir, f"snapshot_{timestamp}.json")
        
        snapshot = {}
        
        for name, partition in self.partitions.items():
            snapshot[name] = {
                "data": partition.data,
                "version": partition.version,
                "last_modified": partition.last_modified,
                "checksum": partition.checksum
            }
        
        try:
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f)
                
            logger.info(f"Created state snapshot at {snapshot_file}")
            return snapshot_file
        except Exception as e:
            logger.error(f"Error creating state snapshot: {e}")
            return None
    
    def restore_snapshot(self, snapshot_file: str) -> bool:
        """
        Restore a snapshot of the distributed state.
        
        Args:
            snapshot_file: Path to snapshot file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(snapshot_file, 'r') as f:
                snapshot = json.load(f)
                
            for name, data in snapshot.items():
                if name in self.partitions:
                    partition = self.partitions[name]
                    partition.data = data.get("data", {})
                    partition.version = data.get("version", 0)
                    partition.last_modified = data.get("last_modified", time.time())
                    partition.checksum = data.get("checksum", "")
                    partition._update_checksum()
            
            # Save restored state to disk
            self._save_state_to_disk()
            
            # Mark changes pending
            self.changes_pending = True
            
            logger.info(f"Restored state from snapshot {snapshot_file}")
            return True
        except Exception as e:
            logger.error(f"Error restoring state from snapshot: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the distributed state.
        
        Returns:
            Dictionary with state metrics
        """
        metrics = {
            "partitions": {},
            "total_items": 0,
            "last_sync_time": self.last_sync_time,
            "sync_in_progress": self.sync_in_progress,
            "changes_pending": self.changes_pending
        }
        
        for name, partition in self.partitions.items():
            size = partition.size()
            metrics["partitions"][name] = {
                "size": size,
                "version": partition.version,
                "last_modified": partition.last_modified
            }
            metrics["total_items"] += size
        
        return metrics