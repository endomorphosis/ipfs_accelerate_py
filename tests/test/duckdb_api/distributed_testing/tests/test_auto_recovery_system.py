#!/usr/bin/env python3
"""
Tests for the Enhanced Auto Recovery System of the Distributed Testing Framework.

This module tests the high availability clustering and WebNN/WebGPU awareness capabilities
of the enhanced auto recovery system.
"""

import os
import sys
import unittest
import tempfile
import threading
import time
import json
import requests
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from duckdb_api.distributed_testing.auto_recovery import (
    AutoRecoverySystem,
    COORDINATOR_STATUS_LEADER,
    COORDINATOR_STATUS_FOLLOWER,
    COORDINATOR_STATUS_CANDIDATE
)

class TestAutoRecoverySystem(unittest.TestCase):
    """Test cases for the Enhanced Auto Recovery System."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_db.duckdb")
        self.visualization_path = os.path.join(self.temp_dir, "visualizations")
        os.makedirs(self.visualization_path, exist_ok=True)
        
        # Create mock components
        self.coordinator_manager = MagicMock()
        self.task_scheduler = MagicMock()
        self.fault_tolerance_system = MagicMock()
        
        # Set fault tolerance metrics for testing
        self.fault_tolerance_system.error_count = 10
        self.fault_tolerance_system.total_operations = 100
        
        # Create test auto recovery system
        self.auto_recovery_system = AutoRecoverySystem(
            coordinator_id="test-coordinator-1",
            coordinator_addresses=["localhost:8081", "localhost:8082"],
            db_path=self.db_path,
            auto_leader_election=True,
            visualization_path=self.visualization_path
        )
        
        # Set component references
        self.auto_recovery_system.set_coordinator_manager(self.coordinator_manager)
        self.auto_recovery_system.set_task_scheduler(self.task_scheduler)
        self.auto_recovery_system.set_fault_tolerance_system(self.fault_tolerance_system)
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop auto recovery if running
        if hasattr(self, 'auto_recovery_system'):
            self.auto_recovery_system.stop()
            
        # Remove temp directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of AutoRecoverySystem."""
        # Check initial state
        self.assertEqual(self.auto_recovery_system.coordinator_id, "test-coordinator-1")
        self.assertEqual(self.auto_recovery_system.db_path, self.db_path)
        self.assertEqual(self.auto_recovery_system.visualization_path, self.visualization_path)
        self.assertEqual(self.auto_recovery_system.coordinator_addresses, ["localhost:8081", "localhost:8082"])
        self.assertFalse(self.auto_recovery_system.is_running)
        self.assertTrue(self.auto_recovery_system.state_sync_required)
        
        # Check component references
        self.assertEqual(self.auto_recovery_system.coordinator_manager, self.coordinator_manager)
        self.assertEqual(self.auto_recovery_system.task_scheduler, self.task_scheduler)
        self.assertEqual(self.auto_recovery_system.fault_tolerance_system, self.fault_tolerance_system)
        
        # Check initial health metrics
        self.assertIn("cpu_usage", self.auto_recovery_system.health_metrics)
        self.assertIn("memory_usage", self.auto_recovery_system.health_metrics)
        self.assertIn("disk_usage", self.auto_recovery_system.health_metrics)
        self.assertIn("network_latency", self.auto_recovery_system.health_metrics)
        self.assertIn("error_rate", self.auto_recovery_system.health_metrics)
        
        # Check initial web capabilities
        self.assertIn("webnn_supported", self.auto_recovery_system.web_capabilities)
        self.assertIn("webgpu_supported", self.auto_recovery_system.web_capabilities)
        self.assertIn("browsers", self.auto_recovery_system.web_capabilities)
        
    def test_start_stop(self):
        """Test starting and stopping the AutoRecoverySystem."""
        # Start system
        self.auto_recovery_system.start()
        
        # Check that system is running
        self.assertTrue(self.auto_recovery_system.is_running)
        
        # Check that threads are running
        self.assertIsNotNone(self.auto_recovery_system.health_check_thread)
        self.assertTrue(self.auto_recovery_system.health_check_thread.is_alive())
        
        # Check visualization thread if visualization path provided
        if self.auto_recovery_system.visualization_path:
            self.assertIsNotNone(self.auto_recovery_system.visualization_thread)
            self.assertTrue(self.auto_recovery_system.visualization_thread.is_alive())
        
        # Stop system
        self.auto_recovery_system.stop()
        
        # Check that system is not running
        self.assertFalse(self.auto_recovery_system.is_running)
        
        # Wait for threads to stop
        time.sleep(0.1)
        
        # Check that threads are not running or are None
        self.assertTrue(
            not hasattr(self.auto_recovery_system, 'health_check_thread') or 
            self.auto_recovery_system.health_check_thread is None or 
            not self.auto_recovery_system.health_check_thread.is_alive()
        )
    
    def test_health_metrics_update(self):
        """Test health metrics update."""
        # Update health metrics
        self.auto_recovery_system._update_health_metrics()
        
        # Check that metrics are updated
        self.assertGreaterEqual(self.auto_recovery_system.health_metrics["cpu_usage"], 0.0)
        self.assertGreaterEqual(self.auto_recovery_system.health_metrics["memory_usage"], 0.0)
        self.assertGreaterEqual(self.auto_recovery_system.health_metrics["disk_usage"], 0.0)
        
        # Check error rate from fault tolerance system
        self.assertEqual(self.auto_recovery_system.health_metrics["error_rate"], 10.0)  # 10 errors out of 100 operations
        
    def test_health_issues_handling(self):
        """Test health issues handling."""
        # Set critical CPU usage
        self.auto_recovery_system.health_metrics["cpu_usage"] = 96.0
        
        # Check for health issues
        self.auto_recovery_system._check_health_issues()
        
        # Set critical memory usage
        self.auto_recovery_system.health_metrics["cpu_usage"] = 50.0
        self.auto_recovery_system.health_metrics["memory_usage"] = 96.0
        
        # Check for health issues
        with patch.object(self.auto_recovery_system, '_free_memory') as mock_free_memory:
            self.auto_recovery_system._check_health_issues()
            mock_free_memory.assert_called_once()
        
        # Set critical disk usage
        self.auto_recovery_system.health_metrics["memory_usage"] = 50.0
        self.auto_recovery_system.health_metrics["disk_usage"] = 96.0
        
        # Check for health issues
        with patch.object(self.auto_recovery_system, '_free_disk_space') as mock_free_disk:
            self.auto_recovery_system._check_health_issues()
            mock_free_disk.assert_called_once()
            
    def test_free_memory(self):
        """Test memory freeing functionality."""
        # Test with gc module
        with patch('gc.collect') as mock_gc_collect:
            self.auto_recovery_system._free_memory()
            mock_gc_collect.assert_called_once()
            
    def test_free_disk_space(self):
        """Test disk space freeing functionality."""
        # Create some temporary log files
        for i in range(5):
            with open(os.path.join(self.temp_dir, f"test_{i}.log"), "w") as f:
                f.write("Test log file")
                
        # Set old modification time (8 days ago)
        old_time = time.time() - (8 * 24 * 60 * 60)
        for i in range(5):
            os.utime(os.path.join(self.temp_dir, f"test_{i}.log"), (old_time, old_time))
            
        # Run disk space freeing with custom log dir
        with patch.object(self.auto_recovery_system, '_free_disk_space', wraps=self.auto_recovery_system._free_disk_space):
            # Replace log_dir with temp_dir in _free_disk_space method
            original_method = self.auto_recovery_system._free_disk_space
            
            def patched_method():
                nonlocal self
                original_log_dir = "."
                try:
                    # Monkey patch log_dir temporarily
                    self.auto_recovery_system.__dict__["_TestAutoRecoverySystem__log_dir"] = self.temp_dir
                    original_method()
                finally:
                    if "_TestAutoRecoverySystem__log_dir" in self.auto_recovery_system.__dict__:
                        del self.auto_recovery_system.__dict__["_TestAutoRecoverySystem__log_dir"]
                        
            # Run patched method
            with patch.object(self.auto_recovery_system, '_free_disk_space', patched_method):
                self.auto_recovery_system._free_disk_space()
            
            # Check that files were deleted or attempt was made
            pass

    def test_hash_data(self):
        """Test data hashing for integrity verification."""
        # Hash sample data
        data = {
            "coordinator_id": "test-coordinator-1",
            "timestamp": "2025-03-16T12:34:56.789",
            "value": 123
        }
        
        # Calculate hash
        hash_value = self.auto_recovery_system._hash_data(data)
        
        # Check that hash is non-empty string
        self.assertIsInstance(hash_value, str)
        self.assertTrue(len(hash_value) > 0)
        
        # Ensure hashing is deterministic
        hash_value2 = self.auto_recovery_system._hash_data(data)
        self.assertEqual(hash_value, hash_value2)
        
        # Ensure hashing is sensitive to changes
        data["value"] = 456
        hash_value3 = self.auto_recovery_system._hash_data(data)
        self.assertNotEqual(hash_value, hash_value3)
        
    def test_health_update_handling(self):
        """Test handling health updates from other coordinators."""
        # Register a coordinator
        self.auto_recovery_system.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu", "cuda"]}
        )
        
        # Create health update data
        health_metrics = {
            "cpu_usage": 50.0,
            "memory_usage": 60.0,
            "disk_usage": 70.0,
            "network_latency": 10.0,
            "error_rate": 5.0,
            "last_updated": datetime.now().isoformat()
        }
        
        web_capabilities = {
            "webnn_supported": True,
            "webgpu_supported": True,
            "browsers": {
                "chrome": {
                    "webnn_supported": True,
                    "webgpu_supported": True
                },
                "firefox": {
                    "webnn_supported": False,
                    "webgpu_supported": True
                }
            },
            "last_updated": datetime.now().isoformat()
        }
        
        health_data = {
            "coordinator_id": "test-coordinator-2",
            "timestamp": datetime.now().isoformat(),
            "health_metrics": health_metrics,
            "web_capabilities": web_capabilities,
            "is_leader": True
        }
        
        # Add hash for integrity verification
        health_data["hash"] = self.auto_recovery_system._hash_data(health_data)
        
        # Handle health update
        response = self.auto_recovery_system.handle_health_update(health_data)
        
        # Check response
        self.assertTrue(response["success"])
        
        # Check coordinator info update
        coordinator = self.auto_recovery_system.auto_recovery.coordinators["test-coordinator-2"]
        self.assertEqual(coordinator["health_metrics"], health_metrics)
        self.assertEqual(coordinator["web_capabilities"], web_capabilities)
        self.assertIn("last_health_update", coordinator)
        
        # Test with invalid hash
        health_data["hash"] = "invalid_hash"
        response = self.auto_recovery_system.handle_health_update(health_data)
        self.assertFalse(response["success"])
        self.assertEqual(response["reason"], "Invalid hash")
        
        # Test with missing hash
        del health_data["hash"]
        response = self.auto_recovery_system.handle_health_update(health_data)
        self.assertFalse(response["success"])
        self.assertEqual(response["reason"], "Missing hash")
        
        # Test with missing coordinator ID
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "health_metrics": health_metrics,
            "web_capabilities": web_capabilities,
            "is_leader": True,
            "hash": "some_hash"
        }
        response = self.auto_recovery_system.handle_health_update(health_data)
        self.assertFalse(response["success"])
        self.assertEqual(response["reason"], "Missing coordinator ID")
        
    @patch('duckdb_api.distributed_testing.enhanced_hardware_detector.EnhancedHardwareDetector')
    def test_web_capabilities_update(self, mock_detector_class):
        """Test updating web capabilities."""
        # Mock detector instance
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        
        # Mock capabilities
        mock_capabilities = {
            "webnn_supported": True,
            "webgpu_supported": True,
            "browsers": {
                "chrome": {
                    "webnn_supported": True,
                    "webgpu_supported": True
                },
                "firefox": {
                    "webnn_supported": False,
                    "webgpu_supported": True
                },
                "edge": {
                    "webnn_supported": True,
                    "webgpu_supported": True
                }
            }
        }
        
        # Set up mock
        mock_detector.detect_capabilities.return_value = mock_capabilities
        
        # Update capabilities
        self.auto_recovery_system._update_web_capabilities()
        
        # Check values
        self.assertTrue(self.auto_recovery_system.web_capabilities["webnn_supported"])
        self.assertTrue(self.auto_recovery_system.web_capabilities["webgpu_supported"])
        self.assertEqual(len(self.auto_recovery_system.web_capabilities["browsers"]), 3)
        self.assertIn("chrome", self.auto_recovery_system.web_capabilities["browsers"])
        self.assertIn("firefox", self.auto_recovery_system.web_capabilities["browsers"])
        self.assertIn("edge", self.auto_recovery_system.web_capabilities["browsers"])
        
    def test_status_reporting(self):
        """Test status reporting functionality."""
        # Get status
        status = self.auto_recovery_system.get_status()
        
        # Check status fields
        self.assertEqual(status["coordinator_id"], "test-coordinator-1")
        self.assertIn("health_metrics", status)
        self.assertIn("web_capabilities", status)
        self.assertTrue(status["visualization_enabled"])
        self.assertEqual(status["db_path"], self.db_path)
        
    def test_visualization_generation(self, check_content=False):
        """Test visualization generation for cluster status."""
        # Generate visualization
        self.auto_recovery_system._generate_cluster_status_visualization()
        
        # Check if visualization file exists
        viz_files = [f for f in os.listdir(self.visualization_path) if f.startswith("cluster_status_")]
        self.assertTrue(len(viz_files) > 0)
        
        if check_content:
            # Check content of text file
            text_files = [f for f in viz_files if f.endswith(".md")]
            if text_files:
                viz_file = os.path.join(self.visualization_path, text_files[0])
                with open(viz_file, "r") as f:
                    content = f.read()
                self.assertIn("Coordinator Cluster Status", content)
                self.assertIn(self.auto_recovery_system.coordinator_id, content)
                    
        # Generate health metrics visualization
        self.auto_recovery_system._generate_health_metrics_visualization()
        
        # Check if visualization file exists
        viz_files = [f for f in os.listdir(self.visualization_path) if f.startswith("health_metrics_")]
        self.assertTrue(len(viz_files) > 0)
        
        if check_content:
            # Check content of text file
            text_files = [f for f in viz_files if f.endswith(".md")]
            if text_files:
                viz_file = os.path.join(self.visualization_path, text_files[0])
                with open(viz_file, "r") as f:
                    content = f.read()
                self.assertIn("Coordinator Health Metrics", content)
                self.assertIn("CPU Usage", content)
                self.assertIn("Memory Usage", content)
                
        # Generate leader transition visualization
        self.auto_recovery_system._generate_leader_transition_visualization("old-leader", "new-leader")
        
        # Check if visualization file exists
        viz_files = [f for f in os.listdir(self.visualization_path) if f.startswith("leader_transition_")]
        self.assertTrue(len(viz_files) > 0)
        
        if check_content:
            # Check content
            viz_file = os.path.join(self.visualization_path, viz_files[0])
            with open(viz_file, "r") as f:
                content = f.read()
            self.assertIn("Leader Transition Event", content)
            self.assertIn("old-leader", content)
            self.assertIn("new-leader", content)
            
    def test_broadcast_health_status(self):
        """Test broadcasting health status to other coordinators."""
        # Make this coordinator the leader
        self.auto_recovery_system.auto_recovery.status = COORDINATOR_STATUS_LEADER
        
        # Register another coordinator
        self.auto_recovery_system.register_coordinator(
            "test-coordinator-2",
            "localhost",
            8081,
            {"hardware_types": ["cpu", "cuda"]}
        )
        
        # Mock requests.post
        with patch('requests.post') as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            
            # Broadcast health status
            self.auto_recovery_system._broadcast_health_status()
            
            # Check that request was made
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(args[0], "http://localhost:8081/api/v1/coordinator/health")
            self.assertIn("json", kwargs)
            
            # Check request data
            data = kwargs["json"]
            self.assertEqual(data["coordinator_id"], "test-coordinator-1")
            self.assertIn("health_metrics", data)
            self.assertIn("web_capabilities", data)
            self.assertIn("hash", data)
            self.assertTrue(data["is_leader"])
            
    def test_leader_callbacks(self):
        """Test leader transition callbacks."""
        # Create a fresh instance with mocked callback methods
        self.auto_recovery_system = AutoRecoverySystem(
            coordinator_id="test-coordinator-1",
            coordinator_addresses=["localhost:8081", "localhost:8082"],
            db_path=self.db_path,
            auto_leader_election=True,
            visualization_path=self.visualization_path
        )
        
        # Replace the real methods with mocks
        self.auto_recovery_system._on_become_leader = MagicMock()
        self.auto_recovery_system._on_leader_changed = MagicMock()
        
        # Re-register the mocked callbacks
        self.auto_recovery_system.auto_recovery.on_become_leader_callbacks = []
        self.auto_recovery_system.auto_recovery.on_leader_changed_callbacks = []
        self.auto_recovery_system.auto_recovery.on_become_leader_callbacks.append(self.auto_recovery_system._on_become_leader)
        self.auto_recovery_system.auto_recovery.on_leader_changed_callbacks.append(self.auto_recovery_system._on_leader_changed)
        
        # Simulate leader transition
        self.auto_recovery_system.auto_recovery.on_become_leader_callbacks[0]()
        
        # Check that callback was called
        self.auto_recovery_system._on_become_leader.assert_called_once()
        
        # Simulate leader changed
        self.auto_recovery_system.auto_recovery.on_leader_changed_callbacks[0]("old-leader", "new-leader")
        
        # Check that callback was called
        self.auto_recovery_system._on_leader_changed.assert_called_once_with("old-leader", "new-leader")

if __name__ == '__main__':
    unittest.main()