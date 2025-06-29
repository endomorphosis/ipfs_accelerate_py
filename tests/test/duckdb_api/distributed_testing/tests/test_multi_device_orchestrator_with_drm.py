#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for the Multi-Device Orchestrator with Dynamic Resource Management.
"""

import unittest
import os
import sys
import json
import time
import uuid
import logging
import threading
from unittest.mock import MagicMock, patch
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
from multi_device_orchestrator import (
    MultiDeviceOrchestrator,
    TaskStatus,
    SubtaskStatus,
    SplitStrategy
)
from dynamic_resource_manager import DynamicResourceManager
from resource_performance_predictor import ResourcePerformancePredictor


class MockTaskManager:
    """Mock task manager for testing."""

    def __init__(self):
        """Initialize mock task manager."""
        self.tasks = {}
        self.task_counter = 0
        self.task_finished_callback = None

    def add_task(self, task_data, priority=5):
        """Mock adding a task."""
        task_id = task_data.get("task_id", f"task_{self.task_counter}")
        self.task_counter += 1
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "data": task_data,
            "priority": priority,
            "status": "pending",
            "subtask_id": task_data.get("subtask_id"),
            "parent_task_id": task_data.get("parent_task_id"),
            "callback": task_data.get("callback")
        }
        
        # Simulate task execution in background thread
        threading.Thread(
            target=self._execute_task,
            args=(task_id,),
            daemon=True
        ).start()
        
        return task_id

    def cancel_task(self, task_id):
        """Mock cancelling a task."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "cancelled"
            return True
        return False

    def get_task_status(self, task_id):
        """Mock getting task status."""
        if task_id in self.tasks:
            return self.tasks[task_id]["status"]
        return "not_found"

    def set_task_finished_callback(self, callback):
        """Set callback function for when tasks finish."""
        self.task_finished_callback = callback

    def _execute_task(self, task_id):
        """Simulate task execution."""
        if task_id not in self.tasks:
            return
        
        # Simulate task execution time
        task_data = self.tasks[task_id]["data"]
        execution_time = task_data.get("execution_time_ms", 500) / 1000.0
        time.sleep(execution_time)
        
        # Check if task was cancelled
        if self.tasks[task_id]["status"] == "cancelled":
            return
        
        # Update task status
        self.tasks[task_id]["status"] = "completed"
        
        # Generate result
        result = {
            "status": "completed",
            "task_id": task_id,
            "execution_time_ms": execution_time * 1000,
            "output": f"Output for task {task_id}"
        }
        
        # Add specific results based on task type
        if "benchmark" in task_data.get("type", ""):
            result["metrics"] = {
                "latency_ms": execution_time * 1000,
                "throughput": 1000 / execution_time,
                "memory_mb": task_data.get("config", {}).get("memory_mb", 1024)
            }
            
            # Add input-specific results if data parallel
            if "input_data" in task_data:
                result["results"] = [
                    {"input": item, "output": f"Result for {item}"}
                    for item in task_data["input_data"]
                ]
        
        # If this is a subtask, call the callback
        callback = self.tasks[task_id].get("callback")
        if callback and callback.get("type") == "subtask_result":
            subtask_id = callback.get("subtask_id")
            if subtask_id and self.task_finished_callback:
                self.task_finished_callback(subtask_id, result, True)


class MockWorkerManager:
    """Mock worker manager for testing."""

    def __init__(self):
        """Initialize mock worker manager."""
        self.workers = {}

    def add_worker(self, worker_id, capabilities):
        """Add a mock worker."""
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "capabilities": capabilities,
            "status": "active",
            "last_seen": datetime.now()
        }

    def get_worker_status(self, worker_id):
        """Get mock worker status."""
        if worker_id in self.workers:
            return self.workers[worker_id]["status"]
        return "not_found"

    def get_worker_capabilities(self, worker_id):
        """Get mock worker capabilities."""
        if worker_id in self.workers:
            return self.workers[worker_id]["capabilities"]
        return {}


class TestMultiDeviceOrchestratorWithDRM(unittest.TestCase):
    """Integration test suite for MultiDeviceOrchestrator with DynamicResourceManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.task_manager = MockTaskManager()
        self.worker_manager = MockWorkerManager()
        
        # Create resource manager
        self.resource_manager = DynamicResourceManager(
            target_utilization=0.7,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3
        )
        
        # Set up orchestrator to use the task_manager callback
        self.orchestrator = MultiDeviceOrchestrator(
            task_manager=self.task_manager,
            worker_manager=self.worker_manager,
            resource_manager=self.resource_manager
        )
        
        # Set up callback from task manager to orchestrator
        self.task_manager.set_task_finished_callback(self.orchestrator.process_subtask_result)
        
        # Add mock workers with different capabilities
        self.worker_manager.add_worker("cpu_worker", {
            "hardware_type": "cpu",
            "cpu_cores": 8,
            "memory_gb": 16,
            "gpu": None
        })
        
        self.worker_manager.add_worker("gpu_worker", {
            "hardware_type": "gpu",
            "cpu_cores": 8,
            "memory_gb": 32,
            "gpu": "nvidia-t4",
            "gpu_memory_gb": 16
        })
        
        self.worker_manager.add_worker("tpu_worker", {
            "hardware_type": "tpu",
            "cpu_cores": 4,
            "memory_gb": 16,
            "tpu_cores": 8,
            "tpu_memory_gb": 32
        })
        
        # Register workers with resource manager
        for worker_id, worker_info in self.worker_manager.workers.items():
            capabilities = worker_info["capabilities"]
            
            # Convert to resource manager format
            resources = {
                "cpu": {
                    "cores": capabilities.get("cpu_cores", 1),
                    "available_cores": capabilities.get("cpu_cores", 1)
                },
                "memory": {
                    "total_mb": capabilities.get("memory_gb", 1) * 1024,
                    "available_mb": capabilities.get("memory_gb", 1) * 1024
                }
            }
            
            # Add GPU if present
            if capabilities.get("gpu"):
                resources["gpu"] = {
                    "devices": 1,
                    "memory_mb": capabilities.get("gpu_memory_gb", 1) * 1024,
                    "available_memory_mb": capabilities.get("gpu_memory_gb", 1) * 1024
                }
            
            # Add TPU if present
            if capabilities.get("tpu_cores"):
                resources["tpu"] = {
                    "devices": capabilities.get("tpu_cores", 1),
                    "memory_mb": capabilities.get("tpu_memory_gb", 1) * 1024,
                    "available_memory_mb": capabilities.get("tpu_memory_gb", 1) * 1024
                }
            
            self.resource_manager.register_worker(worker_id, resources)
        
        # Sample task data for tests
        self.sample_task = {
            "task_id": "test_drm_task_1",
            "type": "benchmark",
            "config": {
                "model": "bert-base-uncased",
                "batch_size": 4,
                "memory_mb": 4096,
                "execution_time_ms": 200
            },
            "input_data": [f"input_{i}" for i in range(10)]
        }

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop orchestrator
        self.orchestrator.stop()
        
        # Clean up resource manager
        self.resource_manager.cleanup()

    def test_data_parallel_orchestration_with_resource_management(self):
        """Test data parallel orchestration with resource management."""
        # Create a task with resource requirements
        task_data = self.sample_task.copy()
        task_data["resource_requirements"] = {
            "cpu_cores": 2,
            "memory_mb": 4096,
            "gpu_memory_mb": 0  # No GPU required
        }
        
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=task_data,
            strategy=SplitStrategy.DATA_PARALLEL
        )
        
        # Let the orchestration complete
        start_time = time.time()
        max_wait = 5.0  # Maximum wait time in seconds
        
        while time.time() - start_time < max_wait:
            status = self.orchestrator.get_task_status(task_id)
            if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            time.sleep(0.1)
        
        # Verify task was completed
        status = self.orchestrator.get_task_status(task_id)
        self.assertEqual(status["status"], TaskStatus.COMPLETED)
        
        # Verify resource reservations were properly released
        worker_utilization = self.resource_manager.get_worker_utilization()
        for worker_id, metrics in worker_utilization["workers"].items():
            self.assertAlmostEqual(metrics["utilization"]["cpu"], 0.0)
            self.assertAlmostEqual(metrics["utilization"]["memory"], 0.0)
            
            # Check GPU utilization for GPU worker
            if worker_id == "gpu_worker":
                self.assertAlmostEqual(metrics["utilization"]["gpu"], 0.0)
        
        # Get task result
        result = self.orchestrator.get_task_result(task_id)
        self.assertIsNotNone(result)
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 10)  # All inputs processed

    def test_model_parallel_orchestration_with_gpu_resources(self):
        """Test model parallel orchestration with GPU resource allocation."""
        # Create a task that requires GPU for some components
        task_data = {
            "task_id": "test_drm_task_2",
            "type": "benchmark",
            "config": {
                "model": "bert-large-uncased",
                "batch_size": 8,
                "memory_mb": 8192,
                "execution_time_ms": 300
            },
            "model_components": [
                "embedding",  # CPU component
                "encoder",    # GPU component
                "decoder"     # GPU component
            ],
            "component_resources": {
                "embedding": {
                    "cpu_cores": 2,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 0  # No GPU required
                },
                "encoder": {
                    "cpu_cores": 2,
                    "memory_mb": 8192,
                    "gpu_memory_mb": 4096  # Requires GPU
                },
                "decoder": {
                    "cpu_cores": 2,
                    "memory_mb": 8192,
                    "gpu_memory_mb": 4096  # Requires GPU
                }
            }
        }
        
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=task_data,
            strategy=SplitStrategy.MODEL_PARALLEL
        )
        
        # Let the orchestration complete
        start_time = time.time()
        max_wait = 5.0  # Maximum wait time in seconds
        
        while time.time() - start_time < max_wait:
            status = self.orchestrator.get_task_status(task_id)
            if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            time.sleep(0.1)
        
        # Verify task was completed
        status = self.orchestrator.get_task_status(task_id)
        self.assertEqual(status["status"], TaskStatus.COMPLETED)
        
        # Verify all components were processed
        result = self.orchestrator.get_task_result(task_id)
        self.assertIsNotNone(result)
        self.assertIn("component_results", result)
        self.assertEqual(len(result["component_results"]), 3)
        
        # Verify all components are included
        self.assertIn("embedding", result["component_results"])
        self.assertIn("encoder", result["component_results"])
        self.assertIn("decoder", result["component_results"])

    def test_ensemble_orchestration_with_mixed_resources(self):
        """Test ensemble orchestration with mixed resource requirements."""
        # Create a task with ensemble configurations
        task_data = {
            "task_id": "test_drm_task_3",
            "type": "benchmark",
            "config": {
                "model": "bert-base-uncased",
                "execution_time_ms": 250
            },
            "ensemble_configs": [
                {
                    "variant": "fp32_cpu",
                    "precision": "fp32",
                    "hardware": "cpu",
                    "batch_size": 1,
                    "resource_requirements": {
                        "cpu_cores": 4,
                        "memory_mb": 4096,
                        "gpu_memory_mb": 0
                    }
                },
                {
                    "variant": "fp16_gpu",
                    "precision": "fp16",
                    "hardware": "gpu",
                    "batch_size": 8,
                    "resource_requirements": {
                        "cpu_cores": 2,
                        "memory_mb": 8192,
                        "gpu_memory_mb": 8192
                    }
                },
                {
                    "variant": "int8_tpu",
                    "precision": "int8",
                    "hardware": "tpu",
                    "batch_size": 16,
                    "resource_requirements": {
                        "cpu_cores": 1,
                        "memory_mb": 4096,
                        "tpu_cores": 4
                    }
                }
            ]
        }
        
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=task_data,
            strategy=SplitStrategy.ENSEMBLE
        )
        
        # Let the orchestration complete
        start_time = time.time()
        max_wait = 5.0  # Maximum wait time in seconds
        
        while time.time() - start_time < max_wait:
            status = self.orchestrator.get_task_status(task_id)
            if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            time.sleep(0.1)
        
        # Verify task was completed
        status = self.orchestrator.get_task_status(task_id)
        self.assertEqual(status["status"], TaskStatus.COMPLETED)
        
        # Verify all variants were processed
        result = self.orchestrator.get_task_result(task_id)
        self.assertIsNotNone(result)
        self.assertIn("variant_results", result)
        self.assertEqual(len(result["variant_results"]), 3)
        
        # Verify all variants are included
        self.assertIn("fp32_cpu", result["variant_results"])
        self.assertIn("fp16_gpu", result["variant_results"])
        self.assertIn("int8_tpu", result["variant_results"])
        
        # Verify ensemble metrics were calculated
        self.assertIn("ensemble_metrics", result)

    def test_resource_oversubscription_handling(self):
        """Test handling of resource oversubscription."""
        # Create tasks that collectively require more resources than available
        tasks = []
        for i in range(5):
            task_data = self.sample_task.copy()
            task_data["task_id"] = f"oversubscription_task_{i}"
            task_data["resource_requirements"] = {
                "cpu_cores": 4,  # Each uses half the CPU cores
                "memory_mb": 8192,  # Each uses half the memory
                "gpu_memory_mb": 0  # No GPU required
            }
            tasks.append(task_data)
        
        # Orchestrate all tasks
        task_ids = []
        for task_data in tasks:
            task_id = self.orchestrator.orchestrate_task(
                task_data=task_data,
                strategy=SplitStrategy.DATA_PARALLEL
            )
            task_ids.append(task_id)
        
        # Let the orchestration run
        time.sleep(2.0)
        
        # Check resource utilization
        worker_utilization = self.resource_manager.get_worker_utilization()
        
        # Verify high CPU utilization on CPU worker
        cpu_worker_util = worker_utilization["workers"].get("cpu_worker", {}).get("utilization", {})
        self.assertGreater(cpu_worker_util.get("cpu", 0), 0.5)  # At least 50% utilization
        
        # Wait for all tasks to complete
        start_time = time.time()
        max_wait = 10.0  # Maximum wait time in seconds
        
        while time.time() - start_time < max_wait:
            all_completed = True
            for task_id in task_ids:
                status = self.orchestrator.get_task_status(task_id)
                if status["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    all_completed = False
                    break
            if all_completed:
                break
            time.sleep(0.5)
        
        # Verify all tasks eventually completed
        for task_id in task_ids:
            status = self.orchestrator.get_task_status(task_id)
            self.assertEqual(status["status"], TaskStatus.COMPLETED)
        
        # Verify resources were properly released
        worker_utilization = self.resource_manager.get_worker_utilization()
        for worker_id, metrics in worker_utilization["workers"].items():
            self.assertAlmostEqual(metrics["utilization"]["cpu"], 0.0)
            self.assertAlmostEqual(metrics["utilization"]["memory"], 0.0)

    def test_fault_tolerance_with_subtask_failure(self):
        """Test fault tolerance with subtask failure."""
        # Create a task where one subtask will fail
        task_data = self.sample_task.copy()
        task_data["fail_on_first_error"] = False  # Don't fail the entire task on first error
        task_data["input_data"] = [f"input_{i}" for i in range(5)]
        
        # Add a special input that will cause failure
        task_data["input_data"].append("fail_this_input")
        
        # Patch the MockTaskManager._execute_task method to simulate failure
        original_execute = self.task_manager._execute_task
        
        def mock_execute_with_failure(task_id):
            task_data = self.task_manager.tasks[task_id]["data"]
            
            # Check if this task contains the failure input
            if "input_data" in task_data and "fail_this_input" in task_data["input_data"]:
                # Update task status
                self.task_manager.tasks[task_id]["status"] = "failed"
                
                # Generate error result
                result = {
                    "status": "failed",
                    "task_id": task_id,
                    "error": "Simulated failure for testing"
                }
                
                # Call callback if this is a subtask
                callback = self.task_manager.tasks[task_id].get("callback")
                if callback and callback.get("type") == "subtask_result":
                    subtask_id = callback.get("subtask_id")
                    if subtask_id and self.task_manager.task_finished_callback:
                        self.task_manager.task_finished_callback(subtask_id, result, False)
                
                return
            
            # Otherwise, execute normally
            return original_execute(task_id)
        
        # Apply the patch
        with patch.object(self.task_manager, '_execute_task', mock_execute_with_failure):
            # Orchestrate task
            task_id = self.orchestrator.orchestrate_task(
                task_data=task_data,
                strategy=SplitStrategy.DATA_PARALLEL
            )
            
            # Let the orchestration complete
            start_time = time.time()
            max_wait = 5.0  # Maximum wait time in seconds
            
            while time.time() - start_time < max_wait:
                status = self.orchestrator.get_task_status(task_id)
                if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    break
                time.sleep(0.1)
            
            # Verify task was completed despite the failure
            status = self.orchestrator.get_task_status(task_id)
            self.assertEqual(status["status"], TaskStatus.COMPLETED)
            
            # Verify some subtasks failed
            failed_subtasks = [s for s in status["subtasks"] if s["status"] == SubtaskStatus.FAILED]
            self.assertGreater(len(failed_subtasks), 0)
            
            # Verify result contains only successful subtask results
            result = self.orchestrator.get_task_result(task_id)
            self.assertIsNotNone(result)
            self.assertIn("results", result)
            self.assertLess(len(result["results"]), 6)  # Less than total inputs due to failure


if __name__ == '__main__':
    unittest.main()