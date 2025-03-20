#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for the Dynamic Resource Management (DRM) system.

These tests validate the DRM integration with the Distributed Testing Framework,
including coordinator, workers, task scheduling, and cloud provider integration.
"""

import unittest
import os
import sys
import json
import time
import asyncio
import threading
import tempfile
import logging
from unittest.mock import MagicMock, patch
import multiprocessing
import signal
import requests
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from coordinator import CoordinatorServer
from worker import WorkerNode
from dynamic_resource_manager import DynamicResourceManager
from resource_performance_predictor import ResourcePerformancePredictor
from cloud_provider_integration import CloudProviderManager


class MockCloudProvider:
    """Mock cloud provider for testing."""
    
    def __init__(self, name):
        """Initialize provider with name."""
        self.name = name
        self.workers = {}
        self.worker_counter = 0
    
    def create_worker(self, resources=None, worker_type=None):
        """Create a worker instance."""
        worker_id = f"{self.name}-worker-{self.worker_counter}"
        self.worker_counter += 1
        self.workers[worker_id] = {
            "status": "running",
            "resources": resources,
            "worker_type": worker_type
        }
        return {
            "worker_id": worker_id,
            "status": "running",
            "provider": self.name,
            "endpoint": f"http://localhost:808{self.worker_counter}"
        }
    
    def terminate_worker(self, worker_id):
        """Terminate a worker instance."""
        if worker_id in self.workers:
            self.workers[worker_id]["status"] = "terminated"
            return True
        return False
    
    def get_worker_status(self, worker_id):
        """Get the status of a worker instance."""
        if worker_id in self.workers:
            return {
                "worker_id": worker_id,
                "status": self.workers[worker_id]["status"],
                "provider": self.name
            }
        return None
    
    def get_available_resources(self):
        """Get available resources on this provider."""
        return {
            "max_workers": 5,
            "available_workers": 5 - len([w for w in self.workers.values() if w["status"] == "running"])
        }


class IntegrationTestBase(unittest.TestCase):
    """Base class for DRM integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create cloud provider configuration
        self.cloud_config = {
            "test_provider": {
                "enabled": True,
                "max_workers": 5
            }
        }
        self.cloud_config_path = os.path.join(self.temp_dir.name, "cloud_config.json")
        with open(self.cloud_config_path, "w") as f:
            json.dump(self.cloud_config, f)
        
        # Set up mock cloud provider
        self.mock_provider = MockCloudProvider("test_provider")
        
        # Create coordinator server with patcher to avoid actually starting server
        self.start_server_patcher = patch.object(CoordinatorServer, 'start_server')
        self.mock_start_server = self.start_server_patcher.start()
        
        # Initialize coordinator with DRM
        self.coordinator = CoordinatorServer(
            host="localhost",
            port=8080,
            db_path=":memory:",
            enable_dynamic_resource_management=True,
            cloud_config=self.cloud_config_path
        )
        
        # Patch cloud provider registration to use our mock provider
        self.coordinator.cloud_provider_manager.providers["test_provider"] = self.mock_provider
        
        # Create resources for mock workers
        self.cpu_resources = {
            "cpu": {
                "cores": 8,
                "physical_cores": 4,
                "available_cores": 7.5
            },
            "memory": {
                "total_mb": 16384,
                "available_mb": 12288
            },
            "gpu": {
                "devices": 0,
                "available_devices": 0,
                "total_memory_mb": 0,
                "available_memory_mb": 0
            }
        }
        
        self.gpu_resources = {
            "cpu": {
                "cores": 16,
                "physical_cores": 8,
                "available_cores": 14.0
            },
            "memory": {
                "total_mb": 32768,
                "available_mb": 24576
            },
            "gpu": {
                "devices": 2,
                "available_devices": 2,
                "total_memory_mb": 16384,
                "available_memory_mb": 14336
            }
        }
        
        # Sample tasks with different resource requirements
        self.small_task = {
            "task_id": "small_task",
            "type": "benchmark",
            "config": {
                "model": "bert-tiny",
                "batch_size": 1
            },
            "requirements": {
                "cpu_cores": 2,
                "memory_mb": 4096
            }
        }
        
        self.large_task = {
            "task_id": "large_task",
            "type": "benchmark",
            "config": {
                "model": "llama-7b",
                "batch_size": 4
            },
            "requirements": {
                "cpu_cores": 8,
                "memory_mb": 16384,
                "gpu_memory_mb": 8192
            }
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop coordinator
        if hasattr(self, 'coordinator'):
            self.coordinator.stop()
        
        # Clean up patchers
        self.start_server_patcher.stop()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()


class TestDRMCoordinatorIntegration(IntegrationTestBase):
    """Test suite for DRM integration with coordinator."""
    
    def test_coordinator_initialization_with_drm(self):
        """Test coordinator initialization with DRM components."""
        # Verify that DRM components are initialized
        self.assertIsNotNone(self.coordinator.dynamic_resource_manager)
        self.assertIsInstance(self.coordinator.dynamic_resource_manager, DynamicResourceManager)
        
        self.assertIsNotNone(self.coordinator.resource_performance_predictor)
        self.assertIsInstance(self.coordinator.resource_performance_predictor, ResourcePerformancePredictor)
        
        self.assertIsNotNone(self.coordinator.cloud_provider_manager)
        self.assertIsInstance(self.coordinator.cloud_provider_manager, CloudProviderManager)
        
        # Verify that cloud providers are registered
        self.assertIn("test_provider", self.coordinator.cloud_provider_manager.providers)

    def test_worker_registration_with_resources(self):
        """Test worker registration with resource information."""
        # Create mock WebSocket for worker registration
        mock_websocket = MagicMock()
        
        # Mock worker registration message
        worker_id = "test-worker-1"
        register_message = {
            "type": "register",
            "worker_id": worker_id,
            "hostname": "localhost",
            "capabilities": ["cpu", "memory"],
            "resources": self.cpu_resources
        }
        
        # Process registration message
        self.coordinator._process_message(register_message, mock_websocket)
        
        # Verify worker is registered with resource information
        self.assertIn(worker_id, self.coordinator.worker_manager.workers)
        
        # Verify resources are registered with dynamic resource manager
        self.assertIn(worker_id, self.coordinator.dynamic_resource_manager.worker_resources)
        
        # Verify correct resource information is stored
        worker_resources = self.coordinator.dynamic_resource_manager.worker_resources[worker_id]
        self.assertEqual(worker_resources["cpu"]["cores"], 8)
        self.assertEqual(worker_resources["memory"]["total_mb"], 16384)

    def test_worker_heartbeat_with_resource_updates(self):
        """Test worker heartbeat with resource updates."""
        # Register worker first
        mock_websocket = MagicMock()
        worker_id = "test-worker-2"
        register_message = {
            "type": "register",
            "worker_id": worker_id,
            "hostname": "localhost",
            "capabilities": ["cpu", "memory"],
            "resources": self.cpu_resources
        }
        self.coordinator._process_message(register_message, mock_websocket)
        
        # Create updated resources with lower availability
        updated_resources = {
            "cpu": {
                "cores": 8,
                "physical_cores": 4,
                "available_cores": 5.0  # Less available cores
            },
            "memory": {
                "total_mb": 16384,
                "available_mb": 8192  # Less available memory
            },
            "gpu": {
                "devices": 0,
                "available_devices": 0,
                "total_memory_mb": 0,
                "available_memory_mb": 0
            }
        }
        
        # Send heartbeat with updated resources
        heartbeat_message = {
            "type": "heartbeat",
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat(),
            "resources": updated_resources
        }
        self.coordinator._process_message(heartbeat_message, mock_websocket)
        
        # Verify resource information is updated
        worker_resources = self.coordinator.dynamic_resource_manager.worker_resources[worker_id]
        self.assertEqual(worker_resources["cpu"]["available_cores"], 5.0)
        self.assertEqual(worker_resources["memory"]["available_mb"], 8192)

    def test_resource_aware_task_scheduling(self):
        """Test resource-aware task scheduling."""
        # Register CPU and GPU workers
        mock_websocket_cpu = MagicMock()
        mock_websocket_gpu = MagicMock()
        
        cpu_worker_id = "cpu-worker"
        gpu_worker_id = "gpu-worker"
        
        # Register CPU worker
        register_cpu_message = {
            "type": "register",
            "worker_id": cpu_worker_id,
            "hostname": "localhost",
            "capabilities": ["cpu", "memory"],
            "resources": self.cpu_resources
        }
        self.coordinator._process_message(register_cpu_message, mock_websocket_cpu)
        
        # Register GPU worker
        register_gpu_message = {
            "type": "register",
            "worker_id": gpu_worker_id,
            "hostname": "localhost",
            "capabilities": ["cpu", "memory", "gpu"],
            "resources": self.gpu_resources
        }
        self.coordinator._process_message(register_gpu_message, mock_websocket_gpu)
        
        # Create a patch for the _send_message method to capture scheduled tasks
        scheduled_tasks = []
        
        def mock_send_message(message, websocket):
            if message.get("type") == "task":
                scheduled_tasks.append({
                    "task": message.get("task"),
                    "worker_id": message.get("worker_id")
                })
        
        with patch.object(self.coordinator, '_send_message', side_effect=mock_send_message):
            # Add tasks to queue
            self.coordinator.task_manager.add_task(self.small_task)
            self.coordinator.task_manager.add_task(self.large_task)
            
            # Mock get_next_task for CPU worker
            next_cpu_task = self.coordinator.task_manager.get_next_task(
                worker_id=cpu_worker_id,
                capabilities=["cpu", "memory"]
            )
            
            # CPU worker should get the small task
            self.assertEqual(next_cpu_task["task_id"], "small_task")
            
            # Mock get_next_task for GPU worker
            next_gpu_task = self.coordinator.task_manager.get_next_task(
                worker_id=gpu_worker_id,
                capabilities=["cpu", "memory", "gpu"]
            )
            
            # GPU worker should get the large task
            self.assertEqual(next_gpu_task["task_id"], "large_task")

    def test_resource_reservation_and_release(self):
        """Test resource reservation and release during task scheduling."""
        # Register worker
        mock_websocket = MagicMock()
        worker_id = "test-worker-3"
        register_message = {
            "type": "register",
            "worker_id": worker_id,
            "hostname": "localhost",
            "capabilities": ["cpu", "memory", "gpu"],
            "resources": self.gpu_resources
        }
        self.coordinator._process_message(register_message, mock_websocket)
        
        # Add task to queue
        self.coordinator.task_manager.add_task(self.large_task)
        
        # Get next task (should reserve resources)
        next_task = self.coordinator.task_manager.get_next_task(
            worker_id=worker_id,
            capabilities=["cpu", "memory", "gpu"]
        )
        
        # Verify resources are reserved
        self.assertIn("resource_reservation_id", next_task)
        reservation_id = next_task["resource_reservation_id"]
        
        # Check reservation in dynamic resource manager
        self.assertIn(reservation_id, self.coordinator.dynamic_resource_manager.resource_reservations)
        
        # Verify worker's available resources are reduced
        worker_resources = self.coordinator.dynamic_resource_manager.worker_resources[worker_id]
        self.assertGreaterEqual(worker_resources["cpu"].get("reserved_cores", 0), self.large_task["requirements"]["cpu_cores"])
        self.assertGreaterEqual(worker_resources["memory"].get("reserved_mb", 0), self.large_task["requirements"]["memory_mb"])
        self.assertGreaterEqual(worker_resources["gpu"].get("reserved_memory_mb", 0), self.large_task["requirements"]["gpu_memory_mb"])
        
        # Complete task (should release resources)
        self.coordinator.task_manager.complete_task(next_task["task_id"])
        
        # Verify resources are released
        self.assertNotIn(reservation_id, self.coordinator.dynamic_resource_manager.resource_reservations)
        
        # Verify worker's available resources are restored
        worker_resources = self.coordinator.dynamic_resource_manager.worker_resources[worker_id]
        self.assertEqual(worker_resources["cpu"].get("reserved_cores", 0), 0)
        self.assertEqual(worker_resources["memory"].get("reserved_mb", 0), 0)
        self.assertEqual(worker_resources["gpu"].get("reserved_memory_mb", 0), 0)

    def test_resource_recording_for_prediction(self):
        """Test recording of resource usage data for prediction."""
        # Register worker
        mock_websocket = MagicMock()
        worker_id = "test-worker-4"
        register_message = {
            "type": "register",
            "worker_id": worker_id,
            "hostname": "localhost",
            "capabilities": ["cpu", "memory"],
            "resources": self.cpu_resources
        }
        self.coordinator._process_message(register_message, mock_websocket)
        
        # Add task to queue
        self.coordinator.task_manager.add_task(self.small_task)
        
        # Get next task
        next_task = self.coordinator.task_manager.get_next_task(
            worker_id=worker_id,
            capabilities=["cpu", "memory"]
        )
        
        # Create mock result with resource usage
        result_message = {
            "type": "task_result",
            "worker_id": worker_id,
            "task_id": next_task["task_id"],
            "timestamp": datetime.now().isoformat(),
            "result": {
                "status": "completed",
                "output": "Task completed successfully"
            },
            "execution_metrics": {
                "execution_time_seconds": 45.2,
                "peak_memory_mb": 3840,
                "peak_cpu_percent": 75.5
            }
        }
        
        # Mock record_task_execution to verify it's called
        with patch.object(self.coordinator.resource_performance_predictor, 'record_task_execution') as mock_record:
            # Process result message
            self.coordinator._process_message(result_message, mock_websocket)
            
            # Verify record_task_execution was called
            mock_record.assert_called_once()
            
            # Verify correct data was passed
            call_args = mock_record.call_args[1]
            self.assertEqual(call_args["success"], True)
            self.assertEqual(call_args["resource_usage"]["execution_time_seconds"], 45.2)

    def test_adaptive_scaling_worker_pool(self):
        """Test adaptive scaling of worker pool based on resource utilization."""
        # Create a high utilization scenario
        for i in range(3):
            worker_id = f"busy-worker-{i}"
            # Register worker with high utilization
            busy_resources = {
                "cpu": {
                    "cores": 8,
                    "physical_cores": 4,
                    "available_cores": 1.5  # High CPU utilization
                },
                "memory": {
                    "total_mb": 16384,
                    "available_mb": 2048  # High memory utilization
                },
                "gpu": {
                    "devices": 1,
                    "available_devices": 1,
                    "total_memory_mb": 8192,
                    "available_memory_mb": 1024  # High GPU utilization
                }
            }
            
            # Register the busy worker
            mock_websocket = MagicMock()
            register_message = {
                "type": "register",
                "worker_id": worker_id,
                "hostname": "localhost",
                "capabilities": ["cpu", "memory", "gpu"],
                "resources": busy_resources
            }
            self.coordinator._process_message(register_message, mock_websocket)
        
        # Register some tasks beyond capacity
        for i in range(10):
            self.coordinator.task_manager.add_task({
                "task_id": f"overflow-task-{i}",
                "type": "benchmark",
                "config": {
                    "model": "bert-base",
                    "batch_size": 4
                },
                "requirements": {
                    "cpu_cores": 4,
                    "memory_mb": 8192,
                    "gpu_memory_mb": 4096
                }
            })
        
        # Force an evaluation cycle
        with patch.object(self.mock_provider, 'create_worker') as mock_create:
            # Configure scaling decision conditions
            self.coordinator.dynamic_resource_manager.utilization_history = [
                {"timestamp": datetime.now() - timedelta(minutes=4), "cpu": 0.85, "memory": 0.88, "gpu": 0.87},
                {"timestamp": datetime.now() - timedelta(minutes=3), "cpu": 0.87, "memory": 0.89, "gpu": 0.89},
                {"timestamp": datetime.now() - timedelta(minutes=2), "cpu": 0.89, "memory": 0.92, "gpu": 0.90},
                {"timestamp": datetime.now() - timedelta(minutes=1), "cpu": 0.92, "memory": 0.94, "gpu": 0.93}
            ]
            
            # Set last scale time to be outside cooldown
            self.coordinator.dynamic_resource_manager.last_scale_up_time = time.time() - 7200
            
            # Manually trigger scaling evaluation
            self.coordinator._evaluate_scaling()
            
            # Verify cloud provider was called to create new workers
            mock_create.assert_called()

    def test_cloud_provider_integration(self):
        """Test cloud provider integration for worker scaling."""
        # Test creating worker through cloud provider
        new_worker = self.coordinator.cloud_provider_manager.create_worker(
            provider="test_provider",
            resources={"cpu_cores": 8, "memory_mb": 16384},
            worker_type="cpu"
        )
        
        # Verify worker was created
        self.assertIsNotNone(new_worker)
        self.assertIn("worker_id", new_worker)
        self.assertEqual(new_worker["provider"], "test_provider")
        
        # Verify worker status can be retrieved
        status = self.coordinator.cloud_provider_manager.get_worker_status(
            provider="test_provider",
            worker_id=new_worker["worker_id"]
        )
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "running")
        
        # Test terminating worker
        result = self.coordinator.cloud_provider_manager.terminate_worker(
            provider="test_provider",
            worker_id=new_worker["worker_id"]
        )
        self.assertTrue(result)
        
        # Verify worker is terminated
        status = self.coordinator.cloud_provider_manager.get_worker_status(
            provider="test_provider",
            worker_id=new_worker["worker_id"]
        )
        self.assertEqual(status["status"], "terminated")


class TestEndToEndDRMWorkflow(IntegrationTestBase):
    """Test end-to-end workflow with DRM system."""
    
    @patch('worker.WorkerNode.connect')
    @patch('worker.WorkerNode._start_websocket_client')
    def test_complete_workflow(self, mock_start_websocket, mock_connect):
        """Test complete workflow from registration to task execution with DRM."""
        # Start coordinator in a separate thread
        coordinator_thread = threading.Thread(target=self.coordinator.start)
        coordinator_thread.daemon = True
        coordinator_thread.start()
        
        # Wait for coordinator to initialize
        time.sleep(0.5)
        
        try:
            # Create mock worker
            worker = WorkerNode(
                coordinator_url="ws://localhost:8080",
                api_key="test-key",
                worker_id="test-worker",
                work_dir=self.temp_dir.name
            )
            
            # Mock methods to avoid actual connection
            mock_connect.return_value = True
            
            # Inject hardware resources
            worker.hardware_metrics = {
                "resources": self.gpu_resources,
                "cpu_percent": 20.5,
                "memory_percent": 35.2,
                "gpu_utilization": 15.8
            }
            
            # Replace _get_hardware_metrics to return our mock data
            worker._get_hardware_metrics = MagicMock(return_value=worker.hardware_metrics)
            
            # Mock registration
            with patch.object(self.coordinator.dynamic_resource_manager, 'register_worker') as mock_register:
                # Simulate worker registration
                worker._send_registration()
                
                # Verify register_worker was called with correct resources
                mock_register.assert_called_once()
                call_args = mock_register.call_args
                self.assertEqual(call_args[0][0], "test-worker")  # worker_id
                self.assertEqual(call_args[0][1], self.gpu_resources)  # resources
            
            # Mock heartbeat
            with patch.object(self.coordinator.dynamic_resource_manager, 'update_worker_resources') as mock_update:
                # Simulate worker heartbeat
                worker._send_heartbeat()
                
                # Verify update_worker_resources was called with correct resources
                mock_update.assert_called_once()
                call_args = mock_update.call_args
                self.assertEqual(call_args[0][0], "test-worker")  # worker_id
                self.assertEqual(call_args[0][1], self.gpu_resources)  # resources
            
            # Add task to coordinator
            self.coordinator.task_manager.add_task(self.large_task)
            
            # Mock task execution
            with patch.object(worker, '_execute_task') as mock_execute:
                # Create result for mock execution
                task_result = {
                    "status": "completed",
                    "output": "Task completed successfully",
                    "execution_metrics": {
                        "execution_time_seconds": 95.2,
                        "peak_memory_mb": 14336,
                        "peak_cpu_percent": 85.5,
                        "peak_gpu_memory_mb": 7168,
                        "peak_gpu_utilization": 92.3
                    }
                }
                mock_execute.return_value = task_result
                
                # Mock receiving task message
                task_message = {
                    "type": "task",
                    "task_id": self.large_task["task_id"],
                    "task": self.large_task
                }
                
                # Patch resource recording to verify it's called
                with patch.object(self.coordinator.resource_performance_predictor, 'record_task_execution') as mock_record:
                    # Simulate worker receiving task
                    worker._handle_task(task_message)
                    
                    # Verify task was executed
                    mock_execute.assert_called_once()
                    
                    # Verify result was sent (would trigger recording resource usage)
                    mock_record.assert_called_once()
                    
                    # Verify resource usage was recorded correctly
                    call_kwargs = mock_record.call_args[1]
                    self.assertEqual(call_kwargs["success"], True)
                    self.assertEqual(call_kwargs["resource_usage"]["execution_time_seconds"], 95.2)
                    self.assertEqual(call_kwargs["resource_usage"]["peak_memory_mb"], 14336)
                    self.assertEqual(call_kwargs["resource_usage"]["peak_gpu_memory_mb"], 7168)
            
            # Verify resource reservation was released
            self.assertEqual(len(self.coordinator.dynamic_resource_manager.resource_reservations), 0)
            
        finally:
            # Stop coordinator
            self.coordinator.stop()
            
            # Wait for thread to complete
            coordinator_thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()