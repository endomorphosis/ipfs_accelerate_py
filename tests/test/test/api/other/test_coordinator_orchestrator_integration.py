#!/usr/bin/env python3
"""
Test Coordinator-Orchestrator Integration

This module provides tests for the integration between the CoordinatorServer
and the MultiDeviceOrchestrator, validating proper task orchestration across
multiple worker nodes.
"""

import os
import sys
import json
import time
import uuid
import unittest
import asyncio
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from pathlib import Path

# Ensure parent directory is in the path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import components to test
from duckdb_api.distributed_testing.multi_device_orchestrator import (
    MultiDeviceOrchestrator, SplitStrategy, TaskStatus, SubtaskStatus
)
from duckdb_api.distributed_testing.coordinator_orchestrator_integration import (
    CoordinatorOrchestratorIntegration, integrate_orchestrator_with_coordinator
)

class MockCoordinator:
    """Mock implementation of a CoordinatorServer for testing."""
    
    def __init__(self):
        self.api_handlers = {}
        self.task_manager = MagicMock()
        self.worker_manager = MagicMock()
        self.resource_manager = MagicMock()
        self.callback_handlers = {}
    
    def register_callback_handler(self, callback_type, handler):
        """Register a callback handler."""
        self.callback_handlers[callback_type] = handler


class TestCoordinatorOrchestratorIntegration(unittest.TestCase):
    """Tests for the CoordinatorOrchestratorIntegration class."""
    
    def setUp(self):
        """Set up a test environment."""
        self.coordinator = MockCoordinator()
        self.integration = CoordinatorOrchestratorIntegration(self.coordinator)
    
    def test_initialization(self):
        """Test that the integration is initialized correctly."""
        self.assertIsNotNone(self.integration)
        self.assertEqual(self.integration.coordinator, self.coordinator)
        self.assertEqual(self.integration.task_manager, self.coordinator.task_manager)
        self.assertEqual(self.integration.worker_manager, self.coordinator.worker_manager)
        self.assertIsNotNone(self.integration.orchestrator)
        self.assertEqual(len(self.coordinator.api_handlers), 4)
        self.assertIn("subtask_result", self.coordinator.callback_handlers)
    
    def test_orchestrate_task(self):
        """Test orchestrating a task."""
        # Mock the orchestrator's orchestrate_task method
        self.integration.orchestrator.orchestrate_task = MagicMock(return_value="task-123")
        
        # Create task data and strategy
        task_data = {"type": "test", "input_data": [1, 2, 3]}
        strategy = SplitStrategy.DATA_PARALLEL
        
        # Call the method
        task_id = self.integration.orchestrate_task(task_data, strategy)
        
        # Verify the result
        self.assertEqual(task_id, "task-123")
        self.assertIn(task_id, self.integration.orchestrated_tasks)
        self.assertEqual(self.integration.orchestrated_tasks[task_id]["task_data"], task_data)
        self.assertEqual(self.integration.orchestrated_tasks[task_id]["strategy"], strategy)
        self.assertIn("creation_time", self.integration.orchestrated_tasks[task_id])
        self.assertEqual(self.integration.orchestrated_tasks[task_id]["status"], "orchestrating")
        
        # Verify orchestrator method was called
        self.integration.orchestrator.orchestrate_task.assert_called_once_with(task_data, strategy)
    
    async def _test_orchestrate_request(self):
        """Test the orchestrate request API handler."""
        # Mock the orchestrator's orchestrate_task method
        self.integration.orchestrator.orchestrate_task = MagicMock(return_value="task-456")
        
        # Create request data
        request_data = {
            "task_data": {"type": "benchmark", "input_data": [1, 2, 3]},
            "strategy": "data_parallel"
        }
        
        # Call the handler
        response = await self.integration._handle_orchestrate_request(request_data)
        
        # Verify the response
        self.assertTrue(response["success"])
        self.assertEqual(response["task_id"], "task-456")
        self.assertIn("message", response)
        
        # Verify orchestrator method was called
        self.integration.orchestrator.orchestrate_task.assert_called_once()
    
    def test_orchestrate_request(self):
        """Run the async test for orchestrate request."""
        asyncio.run(self._test_orchestrate_request())
    
    async def _test_invalid_orchestrate_request(self):
        """Test the orchestrate request API handler with invalid data."""
        # Create invalid request data (missing strategy)
        request_data = {
            "task_data": {"type": "benchmark", "input_data": [1, 2, 3]}
        }
        
        # Call the handler
        response = await self.integration._handle_orchestrate_request(request_data)
        
        # Verify the response
        self.assertFalse(response["success"])
        self.assertIn("error", response)
    
    def test_invalid_orchestrate_request(self):
        """Run the async test for invalid orchestrate request."""
        asyncio.run(self._test_invalid_orchestrate_request())
    
    async def _test_orchestrated_task_request(self):
        """Test the orchestrated task request API handler."""
        # Mock the orchestrator's get_task_status method
        task_status = {
            "status": TaskStatus.IN_PROGRESS,
            "completion_percentage": 50,
            "subtasks": []
        }
        self.integration.orchestrator.get_task_status = MagicMock(return_value=task_status)
        
        # Create request data
        request_data = {"task_id": "task-123"}
        
        # Call the handler
        response = await self.integration._handle_orchestrated_task_request(request_data)
        
        # Verify the response
        self.assertTrue(response["success"])
        self.assertEqual(response["task_status"], task_status)
        
        # Verify orchestrator method was called
        self.integration.orchestrator.get_task_status.assert_called_once_with("task-123")
    
    def test_orchestrated_task_request(self):
        """Run the async test for orchestrated task request."""
        asyncio.run(self._test_orchestrated_task_request())
    
    async def _test_list_orchestrated_tasks(self):
        """Test the list orchestrated tasks API handler."""
        # Add some orchestrated tasks
        self.integration.orchestrated_tasks = {
            "task-1": {"creation_time": datetime.now(), "status": "orchestrating"},
            "task-2": {"creation_time": datetime.now(), "status": "orchestrating"},
            "task-3": {"creation_time": datetime.now(), "status": "orchestrating"}
        }
        
        # Mock the orchestrator's get_task_status method
        def mock_get_task_status(task_id):
            index = int(task_id.split("-")[1])
            return {
                "status": TaskStatus.IN_PROGRESS if index < 3 else TaskStatus.COMPLETED,
                "completion_percentage": index * 25,
                "strategy": SplitStrategy.DATA_PARALLEL
            }
        
        self.integration.orchestrator.get_task_status = MagicMock(side_effect=mock_get_task_status)
        
        # Create request data
        request_data = {
            "filters": {
                "limit": 2,
                "offset": 0
            }
        }
        
        # Call the handler
        response = await self.integration._handle_list_orchestrated_tasks(request_data)
        
        # Verify the response
        self.assertTrue(response["success"])
        self.assertEqual(len(response["tasks"]), 2)
        self.assertEqual(response["total"], 3)
        self.assertEqual(response["returned"], 2)
        self.assertEqual(response["offset"], 0)
        self.assertEqual(response["limit"], 2)
        
        # Check that tasks contain required fields
        for task in response["tasks"]:
            self.assertIn("task_id", task)
            self.assertIn("status", task)
            self.assertIn("creation_time", task)
            self.assertIn("strategy", task)
            self.assertIn("completion_percentage", task)
    
    def test_list_orchestrated_tasks(self):
        """Run the async test for list orchestrated tasks."""
        asyncio.run(self._test_list_orchestrated_tasks())
    
    async def _test_cancel_orchestrated_task(self):
        """Test the cancel orchestrated task API handler."""
        # Mock the orchestrator's cancel_task method
        self.integration.orchestrator.cancel_task = MagicMock(return_value=True)
        
        # Create request data
        request_data = {"task_id": "task-123"}
        
        # Call the handler
        response = await self.integration._handle_cancel_orchestrated_task(request_data)
        
        # Verify the response
        self.assertTrue(response["success"])
        self.assertIn("message", response)
        
        # Verify orchestrator method was called
        self.integration.orchestrator.cancel_task.assert_called_once_with("task-123")
    
    def test_cancel_orchestrated_task(self):
        """Run the async test for cancel orchestrated task."""
        asyncio.run(self._test_cancel_orchestrated_task())
    
    def test_handle_subtask_result(self):
        """Test handling a subtask result."""
        # Mock the orchestrator's process_subtask_result method
        self.integration.orchestrator.process_subtask_result = MagicMock()
        
        # Create result data
        task_id = "task-123"
        worker_id = "worker-456"
        result_data = {
            "subtask_id": "subtask-789",
            "success": True,
            "result": {"output": "test output"}
        }
        
        # Call the handler
        self.integration._handle_subtask_result(task_id, worker_id, result_data)
        
        # Verify orchestrator method was called
        self.integration.orchestrator.process_subtask_result.assert_called_once_with(
            "subtask-789", 
            {"output": "test output"}, 
            True
        )
    
    def test_get_task_result(self):
        """Test getting a task result."""
        # Mock the orchestrator's get_task_result method
        result = {"output": "test output"}
        self.integration.orchestrator.get_task_result = MagicMock(return_value=result)
        
        # Call the method
        task_result = self.integration.get_task_result("task-123")
        
        # Verify the result
        self.assertEqual(task_result, result)
        
        # Verify orchestrator method was called
        self.integration.orchestrator.get_task_result.assert_called_once_with("task-123")
    
    def test_get_task_status(self):
        """Test getting a task status."""
        # Mock the orchestrator's get_task_status method
        status = {"status": TaskStatus.COMPLETED}
        self.integration.orchestrator.get_task_status = MagicMock(return_value=status)
        
        # Call the method
        task_status = self.integration.get_task_status("task-123")
        
        # Verify the result
        self.assertEqual(task_status, status)
        
        # Verify orchestrator method was called
        self.integration.orchestrator.get_task_status.assert_called_once_with("task-123")
    
    def test_stop(self):
        """Test stopping the integration."""
        # Mock the orchestrator's stop method
        self.integration.orchestrator.stop = MagicMock()
        
        # Call the method
        self.integration.stop()
        
        # Verify orchestrator method was called
        self.integration.orchestrator.stop.assert_called_once()
    
    def test_integrate_orchestrator_with_coordinator(self):
        """Test the integration helper function."""
        coordinator = MockCoordinator()
        
        # Call the function
        integration = integrate_orchestrator_with_coordinator(coordinator)
        
        # Verify the result
        self.assertIsInstance(integration, CoordinatorOrchestratorIntegration)
        self.assertEqual(integration.coordinator, coordinator)
        self.assertTrue(hasattr(coordinator, 'orchestrator_integration'))
        self.assertEqual(coordinator.orchestrator_integration, integration)


if __name__ == "__main__":
    unittest.main()