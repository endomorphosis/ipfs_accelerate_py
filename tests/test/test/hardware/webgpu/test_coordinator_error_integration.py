#!/usr/bin/env python3
"""
Test module for coordinator error integration functionality.
"""

import unittest
import logging
import json
from unittest.mock import MagicMock, patch
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from duckdb_api.distributed_testing.distributed_error_handler import (
    DistributedErrorHandler,
    ErrorCategory
)
from duckdb_api.distributed_testing.coordinator_error_integration import (
    integrate_error_handler,
    reschedule_task,
    execute_recovery_action,
    request_resource_cleanup,
    mark_resource_unavailable,
    reallocate_task,
    increase_timeout,
    request_worker_reconnect,
    mark_hardware_unavailable,
    reallocate_to_alternative_hardware,
    mark_worker_unavailable,
    reassign_task,
    record_test_failure
)

# Disable logging for tests
logging.disable(logging.CRITICAL)


class MockCoordinator:
    """Mock coordinator for testing integration."""
    
    def __init__(self):
        """Initialize mock coordinator."""
        self.tasks = {}
        self.workers = {}
        self.current_time = 1000
        
        # Add some test tasks
        self.tasks["task1"] = {
            "id": "task1",
            "status": "running",
            "worker_id": "worker1",
            "type": "test",
            "timeout_seconds": 600,
            "requirements": {
                "hardware": ["cuda"],
                "min_cuda_compute": 7.0,
                "min_memory_gb": 4.0
            }
        }
        
        self.tasks["task2"] = {
            "id": "task2",
            "status": "pending",
            "type": "benchmark",
            "timeout_seconds": 1200
        }
        
        # Add some test workers
        self.workers["worker1"] = {
            "id": "worker1",
            "status": "active",
            "capabilities": {
                "hardware_types": ["cuda", "cpu"],
                "cuda_compute": 7.5,
                "memory_gb": 16
            }
        }
        
        self.workers["worker2"] = {
            "id": "worker2",
            "status": "active",
            "capabilities": {
                "hardware_types": ["cpu"],
                "memory_gb": 8
            }
        }
        
        # Mock methods
        self.handle_task_error = MagicMock(return_value={"original": True})
        self.handle_worker_error = MagicMock(return_value={"original": True})
        self.send_message_to_worker = MagicMock()
        self.get_backup_coordinator_url = MagicMock(return_value="http://backup-coordinator:8080")
    
    def get_current_time(self):
        """Get current time."""
        return self.current_time


class TestCoordinatorErrorIntegration(unittest.TestCase):
    """Test coordinator error integration."""
    
    def setUp(self):
        """Set up test case."""
        self.coordinator = MockCoordinator()
        
        # Integrate error handler
        self.coordinator_with_error_handler = integrate_error_handler(self.coordinator)
    
    def test_integration_adds_error_handler(self):
        """Test that integration adds an error handler to the coordinator."""
        self.assertIsInstance(self.coordinator_with_error_handler.error_handler, DistributedErrorHandler)
    
    def test_enhanced_task_error_handler(self):
        """Test enhanced task error handling."""
        # Create a test error
        error = {
            "type": "ConnectionError",
            "message": "Connection refused",
            "traceback": "...",
            "timestamp": 1000
        }
        
        # Call enhanced handler
        result = self.coordinator_with_error_handler.handle_task_error("task1", error, "worker1")
        
        # Verify result has expected keys
        self.assertIn("error_category", result)
        self.assertIn("retry", result)
        self.assertIn("original", result)
        
        # Verify the original handler was called
        self.coordinator.handle_task_error.assert_called_once_with("task1", error, "worker1")
        
        # Verify the error was categorized
        self.assertEqual(result["error_category"], ErrorCategory.NETWORK_CONNECTION_ERROR)
    
    def test_enhanced_worker_error_handler(self):
        """Test enhanced worker error handling."""
        # Create a test error
        error = {
            "type": "WorkerCrashError",
            "message": "Worker crashed",
            "traceback": "...",
            "timestamp": 1000
        }
        
        # Call enhanced handler
        result = self.coordinator_with_error_handler.handle_worker_error("worker1", error)
        
        # Verify result has expected keys
        self.assertIn("error_category", result)
        self.assertIn("original", result)
        
        # Verify the original handler was called
        self.coordinator.handle_worker_error.assert_called_once_with("worker1", error)
    
    def test_reschedule_task(self):
        """Test task rescheduling functionality."""
        # Test rescheduling a task
        result = reschedule_task(self.coordinator, "task1", 30)
        
        # Verify the task was rescheduled
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks["task1"]["status"], "pending")
        self.assertEqual(self.coordinator.tasks["task1"]["attempt_count"], 2)
        self.assertEqual(self.coordinator.tasks["task1"]["scheduled_time"], 1030)
        self.assertNotIn("worker_id", self.coordinator.tasks["task1"])
        
        # Test rescheduling a non-existent task
        result = reschedule_task(self.coordinator, "nonexistent", 30)
        self.assertFalse(result)
    
    def test_execute_recovery_action_resource(self):
        """Test executing resource recovery actions."""
        # Mock implementation
        with patch("duckdb_api.distributed_testing.coordinator_error_integration.request_resource_cleanup") as mock_action:
            mock_action.return_value = True
            
            # Test resource cleanup action
            result = execute_recovery_action(self.coordinator, "request_resource_cleanup", None, "worker1")
            
            # Verify the action was executed
            self.assertTrue(result)
            mock_action.assert_called_once_with(self.coordinator, "worker1")
    
    def test_execute_recovery_action_hardware(self):
        """Test executing hardware recovery actions."""
        # Mock implementation
        with patch("duckdb_api.distributed_testing.coordinator_error_integration.mark_hardware_unavailable") as mock_action:
            mock_action.return_value = True
            
            # Test hardware action
            result = execute_recovery_action(self.coordinator, "mark_hardware_unavailable:cuda:worker1", None, "worker1")
            
            # Verify the action was executed
            self.assertTrue(result)
            mock_action.assert_called_once_with(self.coordinator, "worker1", "cuda")
    
    def test_request_resource_cleanup(self):
        """Test requesting resource cleanup on a worker."""
        # Test resource cleanup request
        result = request_resource_cleanup(self.coordinator, "worker1")
        
        # Verify the request was sent
        self.assertTrue(result)
        self.coordinator.send_message_to_worker.assert_called_once()
        
        # Check the message content
        args, kwargs = self.coordinator.send_message_to_worker.call_args
        self.assertEqual(args[0], "worker1")
        self.assertEqual(args[1]["type"], "command")
        self.assertEqual(args[1]["command"], "cleanup_resources")
    
    def test_mark_resource_unavailable(self):
        """Test marking resources as unavailable on a worker."""
        # Test marking resources unavailable
        result = mark_resource_unavailable(self.coordinator, "worker1")
        
        # Verify the resource status was updated
        self.assertTrue(result)
        self.assertEqual(self.coordinator.workers["worker1"]["resource_status"], "limited")
    
    def test_reallocate_task(self):
        """Test reallocating a task."""
        # Test task reallocation
        result = reallocate_task(self.coordinator, "task1")
        
        # Verify the task was marked for reallocation
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks["task1"]["status"], "pending")
        self.assertTrue(self.coordinator.tasks["task1"]["needs_reallocation"])
        self.assertNotIn("worker_id", self.coordinator.tasks["task1"])
    
    def test_increase_timeout(self):
        """Test increasing timeout for a task."""
        # Test timeout increase
        result = increase_timeout(self.coordinator, "task1")
        
        # Verify the timeout was increased
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks["task1"]["timeout_seconds"], 900)  # 600 * 1.5
    
    def test_request_worker_reconnect(self):
        """Test requesting worker to reconnect."""
        # Test reconnect request
        result = request_worker_reconnect(self.coordinator, "worker1")
        
        # Verify the request was sent
        self.assertTrue(result)
        self.coordinator.send_message_to_worker.assert_called_once()
        
        # Check the message content
        args, kwargs = self.coordinator.send_message_to_worker.call_args
        self.assertEqual(args[0], "worker1")
        self.assertEqual(args[1]["type"], "command")
        self.assertEqual(args[1]["command"], "reconnect")
    
    def test_mark_hardware_unavailable(self):
        """Test marking hardware as unavailable on a worker."""
        # Test marking hardware unavailable
        result = mark_hardware_unavailable(self.coordinator, "worker1", "cuda")
        
        # Verify the hardware status was updated
        self.assertTrue(result)
        self.assertEqual(self.coordinator.workers["worker1"]["hardware_status"]["cuda"], "unavailable")
    
    def test_reallocate_to_alternative_hardware(self):
        """Test reallocating task to alternative hardware."""
        # Test reallocating to alternative hardware
        result = reallocate_to_alternative_hardware(self.coordinator, "task1")
        
        # Verify the hardware requirements were updated
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks["task1"]["requirements"]["hardware"], ["rocm", "mps", "webgpu", "cpu", "cuda"])
    
    def test_mark_worker_unavailable(self):
        """Test marking a worker as unavailable."""
        # Test marking worker unavailable
        result = mark_worker_unavailable(self.coordinator, "worker1")
        
        # Verify the worker status was updated
        self.assertTrue(result)
        self.assertEqual(self.coordinator.workers["worker1"]["status"], "unavailable")
    
    def test_reassign_task(self):
        """Test reassigning a task."""
        # Test task reassignment
        result = reassign_task(self.coordinator, "task1")
        
        # Verify the task was marked for reassignment
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks["task1"]["status"], "pending")
        self.assertTrue(self.coordinator.tasks["task1"]["needs_reassignment"])
        self.assertNotIn("worker_id", self.coordinator.tasks["task1"])
    
    def test_record_test_failure(self):
        """Test recording a test failure."""
        # Test recording test failure
        result = record_test_failure(self.coordinator, "task1")
        
        # Verify the failure was recorded
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks["task1"]["status"], "failed")
        self.assertEqual(self.coordinator.tasks["task1"]["failure_type"], "assertion")


if __name__ == "__main__":
    unittest.main()