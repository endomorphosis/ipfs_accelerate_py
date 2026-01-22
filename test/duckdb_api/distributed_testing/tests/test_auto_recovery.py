#!/usr/bin/env python3
"""
Tests for the Auto Recovery System of the Distributed Testing Framework.

This module tests the coordinator redundancy and failover capabilities
of the auto recovery system.
"""

import os
import sys
import unittest
import tempfile
import threading
import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from duckdb_api.distributed_testing.auto_recovery import (
    AutoRecovery,
    COORDINATOR_STATUS_LEADER,
    COORDINATOR_STATUS_FOLLOWER,
    COORDINATOR_STATUS_CANDIDATE
)

class TestAutoRecovery(unittest.TestCase):
    """Test cases for the Auto Recovery system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = MagicMock()
        self.coordinator_manager = MagicMock()
        self.task_scheduler = MagicMock()
        
        # Create test auto recovery instance
        self.auto_recovery = AutoRecovery(
            coordinator_id="test-coordinator-1",
            db_manager=self.db_manager,
            coordinator_manager=self.coordinator_manager,
            task_scheduler=self.task_scheduler
        )
        
        # Configure for testing
        self.auto_recovery.configure({
            "heartbeat_interval": 0.1,  # 100ms
            "election_timeout_min": 50,  # 50ms
            "election_timeout_max": 100,  # 100ms
            "leader_check_interval": 0.1,  # 100ms
            "failover_enabled": True,
            "auto_leader_election": True,
            "auto_discover_coordinators": False,  # Disable for testing
            "state_persistence_enabled": False,  # Disable for testing
            "coordinator_addresses": [],
            "visualization_enabled": False  # Disable for testing
        })
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop auto recovery if running
        if hasattr(self, 'auto_recovery'):
            self.auto_recovery.stop()
            
        # Remove temp directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_initialization(self):
        """Test initialization of AutoRecovery."""
        # Check initial state
        self.assertEqual(self.auto_recovery.coordinator_id, "test-coordinator-1")
        self.assertEqual(self.auto_recovery.status, "follower")
        self.assertEqual(self.auto_recovery.term, 0)
        self.assertIsNone(self.auto_recovery.voted_for)
        self.assertIsNone(self.auto_recovery.leader_id)
        self.assertEqual(self.auto_recovery.commit_index, 0)
        self.assertEqual(self.auto_recovery.last_applied, 0)
        self.assertEqual(len(self.auto_recovery.log_entries), 0)
        
    def test_coordinator_registration(self):
        """Test coordinator registration."""
        # Register a coordinator
        success = self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu", "cuda"]}
        )
        
        # Check registration
        self.assertTrue(success)
        self.assertIn("test-coordinator-2", self.auto_recovery.coordinators)
        
        # Check coordinator info
        coordinator = self.auto_recovery.coordinators["test-coordinator-2"]
        self.assertEqual(coordinator["coordinator_id"], "test-coordinator-2")
        self.assertEqual(coordinator["address"], "localhost")
        self.assertEqual(coordinator["port"], 8081)
        self.assertEqual(coordinator["capabilities"]["hardware_types"], ["cpu", "cuda"])
        self.assertEqual(coordinator["status"], "follower")
        
        # Update existing coordinator
        success = self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8082,  # Change port
            {"hardware_types": ["cpu"]}  # Change capabilities
        )
        
        # Check update
        self.assertTrue(success)
        self.assertIn("test-coordinator-2", self.auto_recovery.coordinators)
        
        # Check updated coordinator info
        coordinator = self.auto_recovery.coordinators["test-coordinator-2"]
        self.assertEqual(coordinator["coordinator_id"], "test-coordinator-2")
        self.assertEqual(coordinator["address"], "localhost")
        self.assertEqual(coordinator["port"], 8082)  # Updated port
        self.assertEqual(coordinator["capabilities"]["hardware_types"], ["cpu"])  # Updated capabilities
        
    def test_unregister_coordinator(self):
        """Test coordinator unregistration."""
        # Register coordinators
        self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu", "cuda"]}
        )
        
        self.auto_recovery.register_coordinator(
            "test-coordinator-3",
            "localhost",
            8082,
            {"hardware_types": ["cpu"]}
        )
        
        # Check registration
        self.assertIn("test-coordinator-2", self.auto_recovery.coordinators)
        self.assertIn("test-coordinator-3", self.auto_recovery.coordinators)
        
        # Unregister a coordinator
        success = self.auto_recovery.unregister_coordinator("test-coordinator-2")
        
        # Check unregistration
        self.assertTrue(success)
        self.assertNotIn("test-coordinator-2", self.auto_recovery.coordinators)
        self.assertIn("test-coordinator-3", self.auto_recovery.coordinators)
        
        # Try to unregister non-existent coordinator
        success = self.auto_recovery.unregister_coordinator("non-existent")
        
        # Check failure
        self.assertFalse(success)
        
    def test_update_coordinator_heartbeat(self):
        """Test coordinator heartbeat updates."""
        # Register a coordinator
        self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu", "cuda"]}
        )
        
        # Get initial heartbeat
        initial_heartbeat = self.auto_recovery.coordinators["test-coordinator-2"]["last_heartbeat"]
        
        # Wait a bit
        time.sleep(0.1)
        
        # Update heartbeat
        success = self.auto_recovery.update_coordinator_heartbeat("test-coordinator-2")
        
        # Check update
        self.assertTrue(success)
        self.assertGreater(
            self.auto_recovery.coordinators["test-coordinator-2"]["last_heartbeat"],
            initial_heartbeat
        )
        
        # Try to update non-existent coordinator
        success = self.auto_recovery.update_coordinator_heartbeat("non-existent")
        
        # Check failure
        self.assertFalse(success)
        
    def test_leader_election_single_node(self):
        """Test leader election process in a single-node cluster."""
        # Start auto recovery
        self.auto_recovery.start()
        
        # Wait for initial election to complete
        time.sleep(0.2)
        
        # In a single-node cluster, should become leader
        self.assertEqual(self.auto_recovery.status, COORDINATOR_STATUS_LEADER)
        self.assertEqual(self.auto_recovery.leader_id, "test-coordinator-1")
        self.assertEqual(self.auto_recovery.term, 1)
        
        # Stop auto recovery
        self.auto_recovery.stop()
        
    def test_leader_election_callbacks(self):
        """Test leader election callback functions."""
        # Set up callbacks
        become_leader_called = False
        leader_changed_old = None
        leader_changed_new = None
        
        def on_become_leader():
            nonlocal become_leader_called
            become_leader_called = True
            
        def on_leader_changed(old_leader, new_leader):
            nonlocal leader_changed_old, leader_changed_new
            leader_changed_old = old_leader
            leader_changed_new = new_leader
            
        # Register callbacks
        self.auto_recovery.on_become_leader(on_become_leader)
        self.auto_recovery.on_leader_changed(on_leader_changed)
        
        # Start auto recovery
        self.auto_recovery.start()
        
        # Wait for initial election to complete
        time.sleep(0.2)
        
        # Check callbacks
        self.assertTrue(become_leader_called)
        self.assertIsNone(leader_changed_old)
        self.assertEqual(leader_changed_new, "test-coordinator-1")
        
        # Stop auto recovery
        self.auto_recovery.stop()
        
    def test_append_log_entry(self):
        """Test appending log entries."""
        # Start as leader
        self.auto_recovery._become_leader()
        
        # Append a log entry
        success = self.auto_recovery.append_log_entry(
            "test",
            {"test_data": "value"},
            "test_component",
            "test_action"
        )
        
        # Check success
        self.assertTrue(success)
        self.assertEqual(len(self.auto_recovery.log_entries), 1)
        
        # Check log entry
        entry = self.auto_recovery.log_entries[0]
        self.assertEqual(entry["type"], "test")
        self.assertEqual(entry["data"], {"test_data": "value"})
        self.assertEqual(entry["component"], "test_component")
        self.assertEqual(entry["action"], "test_action")
        self.assertEqual(entry["term"], 0)
        
        # Try appending while not leader
        self.auto_recovery.status = COORDINATOR_STATUS_FOLLOWER
        success = self.auto_recovery.append_log_entry(
            "test2",
            {"test_data": "value2"}
        )
        
        # Check failure
        self.assertFalse(success)
        self.assertEqual(len(self.auto_recovery.log_entries), 1)  # No new entry
        
    def test_heartbeat_handling(self):
        """Test handling of leader heartbeats."""
        # Register another coordinator
        self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu"]}
        )
        
        # Create heartbeat data (as if from a leader)
        heartbeat_data = {
            "coordinator_id": "test-coordinator-2",
            "term": 1,  # Higher term
            "leader_id": "test-coordinator-2",
            "commit_index": 0,
            "prev_log_index": 0,
            "prev_log_term": 0,
            "entries": []
        }
        
        # Process heartbeat
        response = self.auto_recovery.handle_heartbeat(heartbeat_data)
        
        # Check response
        self.assertTrue(response["success"])
        self.assertEqual(response["term"], 1)
        
        # Check state update
        self.assertEqual(self.auto_recovery.status, COORDINATOR_STATUS_FOLLOWER)
        self.assertEqual(self.auto_recovery.term, 1)
        self.assertEqual(self.auto_recovery.leader_id, "test-coordinator-2")
        
        # Now try with a lower term
        heartbeat_data = {
            "coordinator_id": "test-coordinator-3",
            "term": 0,  # Lower term
            "leader_id": "test-coordinator-3",
            "commit_index": 0,
            "prev_log_index": 0,
            "prev_log_term": 0,
            "entries": []
        }
        
        # Process heartbeat
        response = self.auto_recovery.handle_heartbeat(heartbeat_data)
        
        # Check response (should reject due to lower term)
        self.assertFalse(response["success"])
        self.assertEqual(response["term"], 1)
        
        # Check state (should not change)
        self.assertEqual(self.auto_recovery.leader_id, "test-coordinator-2")
        
    def test_vote_request_handling(self):
        """Test handling of vote requests."""
        # Create vote request data
        vote_request = {
            "coordinator_id": "test-coordinator-2",
            "term": 1,  # Higher term
            "last_log_index": 0,
            "last_log_term": 0
        }
        
        # Process vote request
        response = self.auto_recovery.handle_vote_request(vote_request)
        
        # Check response (should grant vote)
        self.assertTrue(response["vote_granted"])
        self.assertEqual(response["term"], 1)
        
        # Check state update
        self.assertEqual(self.auto_recovery.status, COORDINATOR_STATUS_FOLLOWER)
        self.assertEqual(self.auto_recovery.term, 1)
        self.assertEqual(self.auto_recovery.voted_for, "test-coordinator-2")
        
        # Now try with another coordinator in same term
        vote_request = {
            "coordinator_id": "test-coordinator-3",
            "term": 1,  # Same term
            "last_log_index": 0,
            "last_log_term": 0
        }
        
        # Process vote request
        response = self.auto_recovery.handle_vote_request(vote_request)
        
        # Check response (should not grant vote, already voted)
        self.assertFalse(response["vote_granted"])
        self.assertEqual(response["term"], 1)
        
        # Check state (should not change vote)
        self.assertEqual(self.auto_recovery.voted_for, "test-coordinator-2")
        
        # Now try with a lower term
        vote_request = {
            "coordinator_id": "test-coordinator-4",
            "term": 0,  # Lower term
            "last_log_index": 0,
            "last_log_term": 0
        }
        
        # Process vote request
        response = self.auto_recovery.handle_vote_request(vote_request)
        
        # Check response (should reject due to lower term)
        self.assertFalse(response["vote_granted"])
        self.assertEqual(response["term"], 1)
        
    def test_state_snapshot(self):
        """Test creating and retrieving state snapshots."""
        # Start as leader
        self.auto_recovery._become_leader()
        
        # Add some test data
        self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu"]}
        )
        
        # Add a log entry
        self.auto_recovery.append_log_entry(
            "test",
            {"test_data": "value"}
        )
        
        # Create a state snapshot
        self.auto_recovery._create_state_snapshot()
        
        # Get the snapshot
        snapshot = self.auto_recovery.get_state_snapshot()
        
        # Check snapshot content
        self.assertEqual(snapshot["coordinator_id"], "test-coordinator-1")
        self.assertEqual(snapshot["term"], 0)
        self.assertEqual(snapshot["commit_index"], 0)
        self.assertEqual(snapshot["log_entries_count"], 1)
        self.assertEqual(len(snapshot["coordinators"]), 1)
        self.assertEqual(snapshot["coordinators"][0]["coordinator_id"], "test-coordinator-2")
        
    def test_leader_status_checking(self):
        """Test leader status checking methods."""
        # Initially not leader
        self.assertFalse(self.auto_recovery.is_leader())
        
        # Become leader
        self.auto_recovery._become_leader()
        
        # Now is leader
        self.assertTrue(self.auto_recovery.is_leader())
        self.assertEqual(self.auto_recovery.get_leader_id(), "test-coordinator-1")
        
        # Check status
        status = self.auto_recovery.get_status()
        self.assertEqual(status["coordinator_id"], "test-coordinator-1")
        self.assertEqual(status["status"], COORDINATOR_STATUS_LEADER)
        self.assertEqual(status["term"], 0)
        self.assertEqual(status["leader_id"], "test-coordinator-1")
        
        # Become follower
        self.auto_recovery._become_follower(1, "test-coordinator-2")
        
        # Now not leader
        self.assertFalse(self.auto_recovery.is_leader())
        self.assertEqual(self.auto_recovery.get_leader_id(), "test-coordinator-2")
        
        # Check status
        status = self.auto_recovery.get_status()
        self.assertEqual(status["coordinator_id"], "test-coordinator-1")
        self.assertEqual(status["status"], COORDINATOR_STATUS_FOLLOWER)
        self.assertEqual(status["term"], 1)
        self.assertEqual(status["leader_id"], "test-coordinator-2")
        
    @patch('requests.post')
    def test_sync_with_leader(self, mock_post):
        """Test synchronization with a leader coordinator."""
        # Register a coordinator as leader
        self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu"]}
        )
        
        # Set as leader
        self.auto_recovery.leader_id = "test-coordinator-2"
        self.auto_recovery.status = COORDINATOR_STATUS_FOLLOWER
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "snapshot": {
                "timestamp": datetime.now().isoformat(),
                "coordinator_id": "test-coordinator-2",
                "term": 1,
                "commit_index": 2,
                "last_applied": 2,
                "log_entries_count": 3,
                "tasks": [],
                "workers": [],
                "coordinators": []
            }
        }
        mock_post.return_value = mock_response
        
        # Sync with leader
        success = self.auto_recovery.sync_with_leader()
        
        # Check sync
        self.assertTrue(success)
        mock_post.assert_called_once_with(
            'http://localhost:8081/api/v1/coordinator/sync',
            json={'coordinator_id': 'test-coordinator-1'},
            timeout=5.0
        )
        
    def test_handle_sync_request(self):
        """Test handling of state sync requests."""
        # Start as leader
        self.auto_recovery._become_leader()
        
        # Add some test data
        self.auto_recovery.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu"]}
        )
        
        # Handle sync request
        response = self.auto_recovery.handle_sync_request({
            "coordinator_id": "test-coordinator-3"
        })
        
        # Check response
        self.assertTrue(response["success"])
        self.assertIn("snapshot", response)
        
        # Check snapshot
        snapshot = response["snapshot"]
        self.assertEqual(snapshot["coordinator_id"], "test-coordinator-1")
        self.assertEqual(snapshot["term"], 0)
        
        # Now test as follower
        self.auto_recovery.status = COORDINATOR_STATUS_FOLLOWER
        
        # Handle sync request
        response = self.auto_recovery.handle_sync_request({
            "coordinator_id": "test-coordinator-3"
        })
        
        # Check response (should fail as not leader)
        self.assertFalse(response["success"])
        
if __name__ == '__main__':
    unittest.main()