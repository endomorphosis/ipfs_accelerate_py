#!/usr/bin/env python3
"""
Test scenarios for the coordinator redundancy implementation in the Distributed Testing Framework.
Tests the Raft consensus algorithm, leader election, state replication, and failover mechanisms.
"""

import anyio
import os
import sys
import unittest
import tempfile
import time
import uuid
import json
import logging
import random
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration_mode import integration_enabled, integration_opt_in_message

if not integration_enabled():
    pytest.skip(integration_opt_in_message(), allow_module_level=True)

pytest.importorskip("aiohttp")

from coordinator_redundancy import RedundancyManager, NodeRole
from coordinator import DistributedTestingCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockCoordinator:
    """Mock implementation of DistributedTestingCoordinator for testing RedundancyManager."""
    
    def __init__(self, node_id="node-1", host="localhost", port=8080):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.state = {
            "workers": {},
            "tasks": {},
            "status": {},
            "results": {}
        }
        self.on_state_change_callback = None
        self.on_leadership_change_callback = None
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_db.duckdb")
        
    async def apply_state_change(self, state_change):
        """Apply a state change to the coordinator's state."""
        change_type = state_change.get("type")
        if change_type == "add_worker":
            worker_id = state_change["worker_id"]
            self.state["workers"][worker_id] = state_change["worker_data"]
        elif change_type == "update_worker":
            worker_id = state_change["worker_id"]
            if worker_id in self.state["workers"]:
                self.state["workers"][worker_id].update(state_change["worker_data"])
        elif change_type == "add_task":
            task_id = state_change["task_id"]
            self.state["tasks"][task_id] = state_change["task_data"]
        elif change_type == "update_task":
            task_id = state_change["task_id"]
            if task_id in self.state["tasks"]:
                self.state["tasks"][task_id].update(state_change["task_data"])
        
        if self.on_state_change_callback:
            await self.on_state_change_callback(state_change)
            
    def get_full_state(self):
        """Get the full state of the coordinator."""
        return self.state
    
    async def http_request(self, url, method="GET", data=None, timeout=5):
        """Mock HTTP request implementation."""
        return {"success": True, "data": {}}
    
    def register_on_state_change(self, callback):
        """Register a callback for state changes."""
        self.on_state_change_callback = callback
        
    def register_on_leadership_change(self, callback):
        """Register a callback for leadership changes."""
        self.on_leadership_change_callback = callback
        
    async def on_leadership_acquired(self):
        """Called when this node becomes the leader."""
        logger.info(f"Node {self.node_id} became the leader")
        
    async def on_leadership_lost(self):
        """Called when this node loses leadership."""
        logger.info(f"Node {self.node_id} lost leadership")
        
    def get_db_path(self):
        """Get the path to the DuckDB database."""
        return self.db_path
    

class TestRedundancyManager(unittest.TestCase):
    """Test suite for the RedundancyManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for the nodes
        self.temp_dirs = [tempfile.mkdtemp() for _ in range(3)]
        
        # Create mock coordinators
        self.coordinators = [
            MockCoordinator(f"node-{i+1}", "localhost", 8080+i) 
            for i in range(3)
        ]
        
        # Create redundancy managers with the coordinators
        self.managers = []
        for i, coordinator in enumerate(self.coordinators):
            peers = [
                {"id": f"node-{j+1}", "host": "localhost", "port": 8080+j}
                for j in range(3) if j != i
            ]
            
            manager = RedundancyManager(
                coordinator.node_id,
                coordinator.host,
                coordinator.port,
                peers,
                data_dir=self.temp_dirs[i],
                coordinator=coordinator
            )
            self.managers.append(manager)
            
    def tearDown(self):
        """Clean up after tests."""
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir: {e}")
        

            
    def test_init(self):
        """Test initialization of RedundancyManager."""
        manager = self.managers[0]
        self.assertEqual(manager.node_id, "node-1")
        self.assertEqual(manager.role, NodeRole.FOLLOWER)
        self.assertEqual(manager.current_term, 0)
        self.assertIsNone(manager.voted_for)
        self.assertEqual(len(manager.peers), 2)
        
    def test_become_follower(self):
        """Test transition to follower role."""
        manager = self.managers[0]
        
        # Force manager to be in a different role first
        manager.role = NodeRole.CANDIDATE
        
        # Call _become_follower
        self.event_loop.run_until_complete(manager._become_follower(1, "node-2"))
        
        self.assertEqual(manager.role, NodeRole.FOLLOWER)
        self.assertEqual(manager.current_term, 1)
        self.assertEqual(manager.voted_for, None)  # Reset on new term
        self.assertEqual(manager.current_leader, "node-2")
        
    def test_start_election(self):
        """Test election process initiation."""
        manager = self.managers[0]
        
        # Patch the request_vote method to return successful votes
        with patch.object(manager, 'request_vote', new_callable=AsyncMock) as mock_request_vote:
            mock_request_vote.return_value = {"term": 1, "vote_granted": True}
            
            # Run the election
            self.event_loop.run_until_complete(manager._start_election())
            
            # Verify the manager became a candidate and incremented term
            self.assertEqual(manager.role, NodeRole.CANDIDATE)
            self.assertEqual(manager.current_term, 1)
            self.assertEqual(manager.voted_for, manager.node_id)  # Vote for self
            
            # Verify request_vote was called for each peer
            self.assertEqual(mock_request_vote.call_count, 2)
            
    def test_become_leader(self):
        """Test transition to leader role."""
        manager = self.managers[0]
        
        # Set up as candidate first
        manager.role = NodeRole.CANDIDATE
        manager.current_term = 1
        
        # Patch methods called during leader transition
        with patch.object(manager, '_reset_leader_state') as mock_reset, \
             patch.object(manager, '_sync_state_to_followers', new_callable=AsyncMock) as mock_sync, \
             patch.object(manager, '_schedule_heartbeats') as mock_schedule:
            
            # Call become_leader
            self.event_loop.run_until_complete(manager._become_leader())
            
            # Verify the transition
            self.assertEqual(manager.role, NodeRole.LEADER)
            self.assertEqual(manager.current_leader, manager.node_id)
            
            # Verify methods were called
            mock_reset.assert_called_once()
            mock_sync.assert_called_once()
            mock_schedule.assert_called_once()
            
    def test_handle_request_vote(self):
        """Test handling of vote requests."""
        manager = self.managers[0]
        manager.current_term = 1
        
        # Case 1: Request with lower term - reject
        request = {
            "term": 0,
            "candidate_id": "node-2",
            "last_log_index": 0,
            "last_log_term": 0
        }
        response = self.event_loop.run_until_complete(manager.handle_request_vote(request))
        self.assertFalse(response["vote_granted"])
        
        # Case 2: Request with higher term - grant
        request = {
            "term": 2,
            "candidate_id": "node-2",
            "last_log_index": 0,
            "last_log_term": 0
        }
        response = self.event_loop.run_until_complete(manager.handle_request_vote(request))
        self.assertTrue(response["vote_granted"])
        self.assertEqual(manager.current_term, 2)
        self.assertEqual(manager.voted_for, "node-2")
        
    def test_handle_append_entries(self):
        """Test handling of append entries (heartbeat) requests."""
        manager = self.managers[0]
        manager.current_term = 1
        
        # Case 1: Request with lower term - reject
        request = {
            "term": 0,
            "leader_id": "node-2",
            "prev_log_index": 0,
            "prev_log_term": 0,
            "entries": [],
            "leader_commit": 0
        }
        response = self.event_loop.run_until_complete(manager.handle_append_entries(request))
        self.assertFalse(response["success"])
        
        # Case 2: Request with valid term - accept
        request = {
            "term": 1,
            "leader_id": "node-2",
            "prev_log_index": 0,
            "prev_log_term": 0,
            "entries": [],
            "leader_commit": 0
        }
        response = self.event_loop.run_until_complete(manager.handle_append_entries(request))
        self.assertTrue(response["success"])
        self.assertEqual(manager.current_leader, "node-2")
        
    def test_append_log(self):
        """Test appending log entries and replication."""
        manager = self.managers[0]
        manager.role = NodeRole.LEADER
        manager.current_term = 1
        
        # Patch the append_entries method to simulate successful replication
        with patch.object(manager, 'append_entries', new_callable=AsyncMock) as mock_append:
            mock_append.return_value = {"term": 1, "success": True}
            
            # Create a log entry
            entry = {
                "type": "add_worker",
                "worker_id": "worker-1",
                "worker_data": {"host": "worker-host", "port": 9000}
            }
            
            # Append the log entry
            result = self.event_loop.run_until_complete(manager.append_log(entry))
            
            # Verify the entry was added to the log
            self.assertTrue(result)
            self.assertEqual(len(manager.log), 1)
            self.assertEqual(manager.log[0]["command"], entry)
            
            # Verify append_entries was called for each peer
            self.assertEqual(mock_append.call_count, 2)
            
    def test_state_synchronization(self):
        """Test state synchronization between nodes."""
        leader = self.managers[0]
        follower = self.managers[1]
        
        # Set roles
        leader.role = NodeRole.LEADER
        leader.current_term = 1
        follower.role = NodeRole.FOLLOWER
        follower.current_term = 1
        follower.current_leader = leader.node_id
        
        # Add some state to the leader
        leader.coordinator.state["workers"]["worker-1"] = {"host": "worker-host", "port": 9000}
        leader.coordinator.state["tasks"]["task-1"] = {"type": "benchmark", "model": "bert"}
        
        # Patch the http_request method to simulate state sync
        with patch.object(follower, 'http_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "state": leader.coordinator.state,
                "term": leader.current_term,
                "last_applied": 0
            }
            
            # Trigger state sync
            anyio.run(follower._sync_state_from_leader)
            
            # Verify the follower's state was updated
            self.assertEqual(follower.coordinator.state, leader.coordinator.state)
            
    @patch('coordinator_redundancy.anyio.sleep', return_value=None)
    def test_failover(self, mock_sleep):
        """Test leader failover when the current leader fails."""
        # Setup three managers in a stable state
        leader = self.managers[0]
        follower1 = self.managers[1]
        follower2 = self.managers[2]
        
        # Set initial state
        leader.role = NodeRole.LEADER
        leader.current_term = 1
        follower1.role = NodeRole.FOLLOWER
        follower1.current_term = 1
        follower1.current_leader = leader.node_id
        follower2.role = NodeRole.FOLLOWER
        follower2.current_term = 1
        follower2.current_leader = leader.node_id
        
        # Add some log entries
        log_entry = {
            "term": 1,
            "command": {
                "type": "add_worker",
                "worker_id": "worker-1",
                "worker_data": {"host": "worker-host", "port": 9000}
            }
        }
        leader.log.append(log_entry)
        follower1.log.append(log_entry)
        follower2.log.append(log_entry)
        
        # Simulate leader failure by making heartbeats fail
        with patch.object(follower1, 'http_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")
            
            # Patch the election methods
            with patch.object(follower1, '_start_election', new_callable=AsyncMock) as mock_election:
                mock_election.return_value = True  # Election successful
                
                # Trigger a heartbeat timeout
                self.event_loop.run_until_complete(follower1._check_leader_heartbeat())
                
                # Verify election was started
                mock_election.assert_called_once()
                
    def test_recovery_from_crash(self):
        """Test recovery of state after a crash (restart)."""
        manager = self.managers[0]
        manager.current_term = 5
        manager.voted_for = "node-2"
        
        # Add some log entries
        log_entry = {
            "term": 5,
            "command": {
                "type": "add_worker",
                "worker_id": "worker-1",
                "worker_data": {"host": "worker-host", "port": 9000}
            }
        }
        manager.log.append(log_entry)
        
        # Save state
        self.event_loop.run_until_complete(manager._save_persistent_state())
        
        # Create a new manager with the same data dir to simulate restart
        new_manager = RedundancyManager(
            manager.node_id,
            manager.host,
            manager.port,
            manager.peers,
            data_dir=self.temp_dirs[0],
            coordinator=self.coordinators[0]
        )
        
        # Load state
        self.event_loop.run_until_complete(new_manager._load_persistent_state())
        
        # Verify state was recovered
        self.assertEqual(new_manager.current_term, 5)
        self.assertEqual(new_manager.voted_for, "node-2")
        self.assertEqual(len(new_manager.log), 1)
        self.assertEqual(new_manager.log[0]["command"]["worker_id"], "worker-1")
        
    def test_forward_to_leader(self):
        """Test forwarding requests to the leader."""
        manager = self.managers[0]
        manager.role = NodeRole.FOLLOWER
        manager.current_term = 1
        manager.current_leader = "node-2"
        
        # Patch the http_request method to simulate forwarding
        with patch.object(manager, 'http_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"success": True, "result": "success"}
            
            # Create a request to forward
            request = {
                "type": "add_worker",
                "worker_id": "worker-1",
                "worker_data": {"host": "worker-host", "port": 9000}
            }
            
            # Forward the request
            response = self.event_loop.run_until_complete(manager._forward_to_leader(
                "/api/workers/register", "POST", request
            ))
            
            # Verify the request was forwarded
            self.assertEqual(response, {"success": True, "result": "success"})
            mock_request.assert_called_once()
            
    async def run_managers(self, duration=5):
        """Run multiple managers concurrently to test interaction."""
        # Start all managers
        start_tasks = [manager.start() for manager in self.managers]
        for task in start_tasks:
            await task
            
        # Let them run for a while
        await anyio.sleep(duration)
        
        # Stop all managers
        stop_tasks = [manager.stop() for manager in self.managers]
        for task in stop_tasks:
            await task
            
    def test_cluster_operation(self):
        """Test operation of a complete cluster."""
        # This is a more complex test that runs the full cluster
        # Patching is needed to avoid actual network calls
        
        # Patch the network methods in all managers
        for manager in self.managers:
            manager.http_request = AsyncMock(return_value={"success": True})
            manager.append_entries = AsyncMock(return_value={"term": 1, "success": True})
            manager.request_vote = AsyncMock(return_value={"term": 1, "vote_granted": True})
            
        # Run the simulation
        self.event_loop.run_until_complete(self.run_managers(duration=2))
        
        # Verify a leader was elected
        leader_count = sum(1 for manager in self.managers if manager.role == NodeRole.LEADER)
        self.assertEqual(leader_count, 1, "There should be exactly one leader")
        
        # Verify all nodes have the same term
        terms = [manager.current_term for manager in self.managers]
        self.assertEqual(len(set(terms)), 1, "All nodes should have the same term")


if __name__ == "__main__":
    unittest.main()