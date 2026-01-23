#!/usr/bin/env python3
"""
Unit tests for fault tolerance components.

This module contains unit tests for the enhanced fault tolerance components,
including distributed state management, error recovery strategies, and 
coordinator redundancy.
"""

import anyio
import json
import logging
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration_mode import integration_enabled, integration_opt_in_message

if not integration_enabled():
    pytest.skip(integration_opt_in_message(), allow_module_level=True)

pytest.importorskip("aiohttp")

from distributed_state_management import DistributedStateManager, StatePartition
from error_recovery_strategies import (
    EnhancedErrorRecoveryManager,
    ErrorCategory,
    RecoveryLevel,
    RetryStrategy,
    WorkerRecoveryStrategy,
    DatabaseRecoveryStrategy,
)
from coordinator_redundancy import RedundancyManager, NodeRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("fault_tolerance_test")

class TestStatePartition(unittest.TestCase):
    """Test cases for StatePartition class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.partition = StatePartition("test_partition", priority=5)
    
    def test_update(self):
        """Test update operation."""
        # Update a value
        transaction_id = self.partition.update("key1", "value1")
        self.assertIsNotNone(transaction_id)
        self.assertEqual(self.partition.get("key1"), "value1")
        self.assertEqual(self.partition.version, 1)
        self.assertEqual(len(self.partition.transaction_log), 1)
        
        # Update the same key
        transaction_id = self.partition.update("key1", "value2")
        self.assertIsNotNone(transaction_id)
        self.assertEqual(self.partition.get("key1"), "value2")
        self.assertEqual(self.partition.version, 2)
        self.assertEqual(len(self.partition.transaction_log), 2)
    
    def test_delete(self):
        """Test delete operation."""
        # Add a value
        self.partition.update("key1", "value1")
        
        # Delete the value
        transaction_id = self.partition.delete("key1")
        self.assertIsNotNone(transaction_id)
        self.assertIsNone(self.partition.get("key1"))
        self.assertEqual(self.partition.version, 2)
        self.assertEqual(len(self.partition.transaction_log), 2)
        
        # Delete non-existent key (should not change version)
        current_version = self.partition.version
        transaction_id = self.partition.delete("key2")
        self.assertIsNotNone(transaction_id)
        self.assertEqual(self.partition.version, current_version)
    
    def test_get(self):
        """Test get operation."""
        # Add a value
        self.partition.update("key1", "value1")
        
        # Get the value
        self.assertEqual(self.partition.get("key1"), "value1")
        
        # Get non-existent key
        self.assertIsNone(self.partition.get("key2"))
        
        # Get non-existent key with default
        self.assertEqual(self.partition.get("key2", "default"), "default")
    
    def test_update_batch(self):
        """Test batch update operation."""
        # Update multiple values
        updates = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        transaction_id = self.partition.update_batch(updates)
        self.assertIsNotNone(transaction_id)
        self.assertEqual(self.partition.get("key1"), "value1")
        self.assertEqual(self.partition.get("key2"), "value2")
        self.assertEqual(self.partition.get("key3"), "value3")
        self.assertEqual(self.partition.version, 1)
        self.assertEqual(len(self.partition.transaction_log), 1)
    
    def test_clear(self):
        """Test clear operation."""
        # Add some values
        self.partition.update("key1", "value1")
        self.partition.update("key2", "value2")
        
        # Clear all data
        transaction_id = self.partition.clear()
        self.assertIsNotNone(transaction_id)
        self.assertEqual(self.partition.size(), 0)
        self.assertEqual(self.partition.version, 3)
        self.assertEqual(len(self.partition.transaction_log), 3)
    
    def test_apply_transaction(self):
        """Test applying a transaction."""
        # Create a transaction
        transaction = {
            "id": "tx-1",
            "type": "update",
            "key": "key1",
            "new_value": "value1",
            "timestamp": datetime.now().timestamp()
        }
        
        # Apply the transaction
        result = self.partition.apply_transaction(transaction)
        self.assertTrue(result)
        self.assertEqual(self.partition.get("key1"), "value1")
        
        # Apply the same transaction again (should be ignored)
        current_version = self.partition.version
        result = self.partition.apply_transaction(transaction)
        self.assertTrue(result)
        self.assertEqual(self.partition.version, current_version)  # Version shouldn't change
    
    def test_get_transactions_since(self):
        """Test getting transactions since a timestamp."""
        # Add some values
        self.partition.update("key1", "value1")
        timestamp = datetime.now().timestamp()
        self.partition.update("key2", "value2")
        
        # Get transactions since timestamp
        transactions = self.partition.get_transactions_since(timestamp)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["key"], "key2")
    
    def test_merge(self):
        """Test merging partitions."""
        # Create another partition
        other_partition = StatePartition("other_partition", priority=5)
        
        # Add some values to both partitions
        self.partition.update("key1", "value1")
        self.partition.update("key2", "value2")
        
        other_partition.update("key2", "value2-other")  # Conflict
        other_partition.update("key3", "value3")
        
        # Make the other partition older
        other_partition.last_modified = self.partition.last_modified - 1
        
        # Merge
        conflicts = self.partition.merge(other_partition)
        
        # Check results
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0], "key2")
        self.assertEqual(self.partition.get("key1"), "value1")
        self.assertEqual(self.partition.get("key2"), "value2")  # Our value wins (newer)
        self.assertEqual(self.partition.get("key3"), "value3")
        
        # Make the other partition newer and merge again
        other_partition.last_modified = self.partition.last_modified + 10
        conflicts = self.partition.merge(other_partition)
        
        # Check results
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0], "key2")
        self.assertEqual(self.partition.get("key2"), "value2-other")  # Their value wins (newer)

class TestDistributedStateManager(unittest.TestCase):
    """Test cases for DistributedStateManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for state files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock coordinator
        self.coordinator = MagicMock()
        
        # Create cluster nodes
        self.cluster_nodes = ["http://localhost:8080", "http://localhost:8081"]
        
        # Create state manager
        self.state_manager = DistributedStateManager(
            coordinator=self.coordinator,
            cluster_nodes=self.cluster_nodes,
            node_id="node-1",
            state_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization."""
        # Check that partitions were created
        self.assertTrue("workers" in self.state_manager.partitions)
        self.assertTrue("tasks" in self.state_manager.partitions)
        self.assertTrue("task_history" in self.state_manager.partitions)
        self.assertTrue("system_health" in self.state_manager.partitions)
        self.assertTrue("configuration" in self.state_manager.partitions)
    
    def test_update_and_get(self):
        """Test update and get operations."""
        # Update some state
        self.state_manager.update("workers", "worker-1", {"status": "active"})
        
        # Get the state
        worker = self.state_manager.get("workers", "worker-1")
        self.assertIsNotNone(worker)
        self.assertEqual(worker["status"], "active")
        
        # Get non-existent state
        worker = self.state_manager.get("workers", "worker-2")
        self.assertIsNone(worker)
        
        # Get with default
        worker = self.state_manager.get("workers", "worker-2", {"status": "unknown"})
        self.assertEqual(worker["status"], "unknown")
    
    def test_update_batch(self):
        """Test batch update operation."""
        # Update multiple values
        updates = {
            "worker-1": {"status": "active"},
            "worker-2": {"status": "idle"}
        }
        
        result = self.state_manager.update_batch("workers", updates)
        self.assertTrue(result)
        
        # Check values
        self.assertEqual(self.state_manager.get("workers", "worker-1")["status"], "active")
        self.assertEqual(self.state_manager.get("workers", "worker-2")["status"], "idle")
    
    def test_delete(self):
        """Test delete operation."""
        # Add a value
        self.state_manager.update("workers", "worker-1", {"status": "active"})
        
        # Delete the value
        result = self.state_manager.delete("workers", "worker-1")
        self.assertTrue(result)
        
        # Check it's gone
        self.assertIsNone(self.state_manager.get("workers", "worker-1"))
    
    def test_create_snapshot(self):
        """Test creating a snapshot."""
        # Add some state
        self.state_manager.update("workers", "worker-1", {"status": "active"})
        self.state_manager.update("tasks", "task-1", {"status": "running"})
        
        # Create snapshot
        snapshot_file = self.state_manager.create_snapshot()
        self.assertIsNotNone(snapshot_file)
        
        # Check file exists
        self.assertTrue(os.path.exists(snapshot_file))
        
        # Check file content
        with open(snapshot_file, 'r') as f:
            snapshot = json.load(f)
            self.assertTrue("workers" in snapshot)
            self.assertTrue("tasks" in snapshot)
    
    def test_restore_snapshot(self):
        """Test restoring from a snapshot."""
        # Add some state
        self.state_manager.update("workers", "worker-1", {"status": "active"})
        
        # Create snapshot
        snapshot_file = self.state_manager.create_snapshot()
        
        # Update state
        self.state_manager.update("workers", "worker-1", {"status": "idle"})
        
        # Restore snapshot
        result = self.state_manager.restore_snapshot(snapshot_file)
        self.assertTrue(result)
        
        # Check state was restored
        self.assertEqual(self.state_manager.get("workers", "worker-1")["status"], "active")
    
    def test_get_metrics(self):
        """Test getting metrics."""
        # Add some state
        self.state_manager.update("workers", "worker-1", {"status": "active"})
        self.state_manager.update("tasks", "task-1", {"status": "running"})
        
        # Get metrics
        metrics = self.state_manager.get_metrics()
        self.assertTrue("partitions" in metrics)
        self.assertTrue("workers" in metrics["partitions"])
        self.assertTrue("tasks" in metrics["partitions"])
        self.assertEqual(metrics["partitions"]["workers"]["size"], 1)
        self.assertEqual(metrics["partitions"]["tasks"]["size"], 1)
        self.assertEqual(metrics["total_items"], 2)

class TestErrorRecoveryStrategies(unittest.TestCase):
    """Test cases for error recovery strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock coordinator
        self.coordinator = MagicMock()
        
        # Create recovery manager
        self.recovery_manager = EnhancedErrorRecoveryManager(self.coordinator)
    
    def test_retry_strategy(self):
        """Test retry strategy."""
        # Create a retry strategy
        retry_strategy = RetryStrategy(self.coordinator, max_retries=3)
        
        # Create a test operation
        attempt_count = 0
        def test_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Test error")
            return "Success"
        
        # Execute strategy with an error
        error_info = {
            "operation": test_operation,
            "args": [],
            "kwargs": {}
        }
        
        async def run_test():
            return await retry_strategy.execute(error_info)
        
        result = anyio.run(run_test)
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(attempt_count, 3)
    
    def test_worker_recovery_strategy(self):
        """Test worker recovery strategy."""
        # Create a worker recovery strategy
        worker_strategy = WorkerRecoveryStrategy(self.coordinator)
        
        # Set up mock coordinator
        self.coordinator.workers = {
            "worker-1": {"worker_id": "worker-1", "status": "active"}
        }
        self.coordinator.worker_connections = {}
        self.coordinator.tasks = {}
        self.coordinator.running_tasks = {}
        self.coordinator.db = MagicMock()
        
        # Execute strategy
        error_info = {
            "worker_id": "worker-1",
            "category": ErrorCategory.WORKER_OFFLINE.value
        }
        
        async def run_test():
            return await worker_strategy.execute(error_info)

        result = anyio.run(run_test)
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(self.coordinator.workers["worker-1"]["status"], "recovery_pending")
    
    def test_error_categorization(self):
        """Test error categorization."""
        # Test various error types
        connection_error = ConnectionError("Failed to connect")
        timeout_error = TimeoutError("Operation timed out")
        value_error = ValueError("Invalid value")
        
        # Categorize errors
        connection_info = self.recovery_manager.categorize_error(connection_error)
        timeout_info = self.recovery_manager.categorize_error(timeout_error)
        value_info = self.recovery_manager.categorize_error(value_error)
        
        # Check categorization
        self.assertEqual(connection_info["category"], ErrorCategory.CONNECTION.value)
        self.assertEqual(timeout_info["category"], ErrorCategory.TIMEOUT.value)
        self.assertEqual(value_info["category"], ErrorCategory.UNKNOWN.value)
        
        # Test with context
        worker_error = Exception("Worker crashed")
        worker_info = self.recovery_manager.categorize_error(worker_error, {"component": "worker"})
        self.assertEqual(worker_info["category"], ErrorCategory.WORKER_OFFLINE.value)
    
    def test_strategy_selection(self):
        """Test strategy selection."""
        # Test strategy selection for different error categories
        connection_strategy = self.recovery_manager.get_strategy_for_error({"category": ErrorCategory.CONNECTION.value})
        worker_strategy = self.recovery_manager.get_strategy_for_error({"category": ErrorCategory.WORKER_OFFLINE.value})
        unknown_strategy = self.recovery_manager.get_strategy_for_error({"category": ErrorCategory.UNKNOWN.value})
        
        # Check strategy selection
        self.assertEqual(connection_strategy.name, "retry")
        self.assertEqual(worker_strategy.name, "worker")
        self.assertEqual(unknown_strategy.name, "retry")
    
    @patch('asyncio.sleep')
    async def test_recovery_process(self, mock_sleep):
        """Test the full recovery process."""
        # Mock sleep to avoid delays in test
        mock_sleep.return_value = None
        
        # Set up mock coordinator
        self.coordinator.workers = {
            "worker-1": {"worker_id": "worker-1", "status": "active"}
        }
        self.coordinator.worker_connections = {}
        self.coordinator.tasks = {}
        self.coordinator.running_tasks = {}
        self.coordinator.db = MagicMock()
        
        # Create an error
        error = ConnectionError("Worker disconnected")
        context = {"component": "worker", "worker_id": "worker-1"}
        
        # Attempt recovery
        success, info = await self.recovery_manager.recover(error, context)
        
        # Check results
        self.assertTrue(success)
        self.assertEqual(self.coordinator.workers["worker-1"]["status"], "recovery_pending")
        self.assertTrue(info["success"])
        self.assertEqual(info["error_info"]["category"], ErrorCategory.CONNECTION.value)

class TestCoordinatorRedundancy(unittest.TestCase):
    """Test cases for coordinator redundancy."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock coordinator
        self.coordinator = MagicMock()
        
        # Create cluster nodes
        self.cluster_nodes = ["http://localhost:8080", "http://localhost:8081"]
        
        # Create a temporary directory for state files
        self.temp_dir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.temp_dir, "redundancy_state.json")
        
        # Create redundancy manager
        self.redundancy_manager = RedundancyManager(
            coordinator=self.coordinator,
            cluster_nodes=self.cluster_nodes,
            node_id="node-1",
            state_path=self.state_path,
            use_state_manager=False
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization."""
        # Check initial state
        self.assertEqual(self.redundancy_manager.current_role, NodeRole.FOLLOWER)
        self.assertEqual(self.redundancy_manager.current_term, 0)
        self.assertIsNone(self.redundancy_manager.voted_for)
        self.assertIsNone(self.redundancy_manager.leader_id)
        self.assertEqual(self.redundancy_manager.commit_index, 0)
    
    def test_election_timeout(self):
        """Test election timeout calculation."""
        # Get random election timeout
        timeout = self.redundancy_manager._get_random_election_timeout()
        
        # Check range
        self.assertTrue(timeout >= self.redundancy_manager.election_timeout_min)
        self.assertTrue(timeout <= self.redundancy_manager.election_timeout_max)
    
    @patch('time.time')
    def test_become_leader(self, mock_time):
        """Test becoming leader."""
        # Mock time.time
        mock_time.return_value = 12345.0
        
        async def run_test():
            # Become a candidate first
            self.redundancy_manager.current_role = NodeRole.CANDIDATE
            self.redundancy_manager.current_term = 1
            self.redundancy_manager.voted_for = "node-1"
            
            # Add votes
            self.redundancy_manager.votes_received = {"node-1", "node-2"}
            
            # Become leader
            await self.redundancy_manager._become_leader()
            
            # Check state
            self.assertEqual(self.redundancy_manager.current_role, NodeRole.LEADER)
            self.assertEqual(self.redundancy_manager.leader_id, "node-1")
            
            # Check next and match index
            self.assertEqual(self.redundancy_manager.next_index["http://localhost:8080"], 1)
            self.assertEqual(self.redundancy_manager.match_index["http://localhost:8080"], 0)
        
        anyio.run(run_test)
    
    @patch('time.time')
    def test_become_follower(self, mock_time):
        """Test becoming follower."""
        # Mock time.time
        mock_time.return_value = 12345.0
        
        async def run_test():
            # Start as a leader
            self.redundancy_manager.current_role = NodeRole.LEADER
            self.redundancy_manager.current_term = 1
            self.redundancy_manager.leader_id = "node-1"
            
            # Become follower with higher term
            await self.redundancy_manager._become_follower(2)
            
            # Check state
            self.assertEqual(self.redundancy_manager.current_role, NodeRole.FOLLOWER)
            self.assertEqual(self.redundancy_manager.current_term, 2)
            self.assertIsNone(self.redundancy_manager.voted_for)
        
        anyio.run(run_test)
    
    def test_persistent_state(self):
        """Test persistent state save and load."""
        async def run_test():
            # Set some state
            self.redundancy_manager.current_term = 5
            self.redundancy_manager.voted_for = "node-2"
            self.redundancy_manager.log_entries = [{"term": 1, "command": "test"}]

            # Save state
            await self.redundancy_manager._save_persistent_state()

            # Reset state
            self.redundancy_manager.current_term = 0
            self.redundancy_manager.voted_for = None
            self.redundancy_manager.log_entries = []

            # Load state
            await self.redundancy_manager._load_persistent_state()

            # Check state was restored
            self.assertEqual(self.redundancy_manager.current_term, 5)
            self.assertEqual(self.redundancy_manager.voted_for, "node-2")
            self.assertEqual(len(self.redundancy_manager.log_entries), 1)

        anyio.run(run_test)
        self.assertEqual(self.redundancy_manager.log_entries[0]["term"], 1)

if __name__ == "__main__":
    unittest.main()