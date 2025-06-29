#!/usr/bin/env python3
"""
Tests for Hardware-Aware Fault Tolerance System

This module tests the hardware-aware fault tolerance system for the
Distributed Testing Framework. It validates the system's ability to
handle failures in different hardware environments and apply appropriate
recovery strategies.
"""

import os
import sys
import json
import time
import unittest
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import components to test
from duckdb_api.distributed_testing.hardware_aware_fault_tolerance import (
    HardwareAwareFaultToleranceManager,
    FailureType,
    RecoveryStrategy,
    FailureContext,
    RecoveryAction,
    apply_recovery_action,
    create_recovery_manager
)
from duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    HardwareCapabilityProfile,
    MemoryProfile
)


class MockDBManager:
    """Mock database manager for testing state persistence."""
    
    def __init__(self):
        self.data = {}
        self.checkpoints = {}
        
    def save_fault_tolerance_state(self, state):
        """Save fault tolerance state."""
        self.data["state"] = state
        return True
        
    def get_fault_tolerance_state(self):
        """Get fault tolerance state."""
        return self.data.get("state")
        
    def save_checkpoint(self, checkpoint_id, checkpoint):
        """Save a checkpoint."""
        self.checkpoints[checkpoint_id] = checkpoint
        return True
        
    def get_all_checkpoints(self):
        """Get all checkpoints."""
        return self.checkpoints


class MockCoordinator:
    """Mock coordinator for testing fault tolerance integration."""
    
    def __init__(self):
        self.tasks = {}
        self.workers = {}
        self.running_tasks = []
        self.retried_tasks = []
        self.escalated_tasks = []
        self.restarted_browsers = []
        self.reset_worker_states = []
        self.task_manager = self
        self.worker_manager = self
        
    def add_task(self, task_id, config, requirements=None, retry_count=0):
        """Add a task to the mock coordinator."""
        self.tasks[task_id] = {
            "task_id": task_id,
            "config": config,
            "requirements": requirements or {},
            "retry_count": retry_count,
            "status": "queued"
        }
        return task_id
        
    def add_worker(self, worker_id, hardware_profile=None, status="active"):
        """Add a worker to the mock coordinator."""
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "hardware_profile": hardware_profile,
            "status": status
        }
        return worker_id
        
    def get_task(self, task_id):
        """Get a task by ID."""
        return self.tasks.get(task_id)
        
    def get_worker(self, worker_id):
        """Get a worker by ID."""
        return self.workers.get(worker_id)
        
    def get_running_tasks(self):
        """Get list of running task IDs."""
        return self.running_tasks
        
    def retry_task(self, task_id, worker_id=None, exclude_workers=None):
        """Retry a task."""
        self.retried_tasks.append({
            "task_id": task_id,
            "worker_id": worker_id,
            "exclude_workers": exclude_workers
        })
        return True
        
    def update_task_requirements(self, task_id, requirements):
        """Update task requirements."""
        if task_id in self.tasks:
            self.tasks[task_id]["requirements"] = requirements
        return True
        
    def update_task_config(self, task_id, config):
        """Update task configuration."""
        if task_id in self.tasks:
            self.tasks[task_id]["config"] = config
        return True
        
    def escalate_task(self, task_id, message):
        """Escalate a task to human operator."""
        self.escalated_tasks.append({
            "task_id": task_id,
            "message": message
        })
        return True
        
    def restart_worker_browser(self, worker_id):
        """Restart a worker's browser."""
        self.restarted_browsers.append(worker_id)
        return True
        
    def reset_worker_state(self, worker_id):
        """Reset a worker's state."""
        self.reset_worker_states.append(worker_id)
        return True


def create_hardware_profile(hardware_class, architecture=None, vendor=None):
    """Create a hardware capability profile for testing."""
    class_map = {
        "CPU": HardwareClass.CPU,
        "GPU": HardwareClass.GPU,
        "TPU": HardwareClass.TPU,
        "NPU": HardwareClass.NPU,
        "WEBGPU": HardwareClass.GPU,  # WebGPU is a type of GPU
        "WEBNN": HardwareClass.HYBRID  # WebNN can use various hardware
    }
    
    hw_class = class_map.get(hardware_class, HardwareClass.CPU)
    
    # Create memory profile (8GB)
    memory = MemoryProfile(
        total_bytes=8 * 1024 * 1024 * 1024,
        available_bytes=6 * 1024 * 1024 * 1024,
        memory_type="DDR4" if hw_class == HardwareClass.CPU else "GDDR6"
    )
    
    # Create a set of supported precision types
    supported_precisions = {PrecisionType.FP32, PrecisionType.FP16}
    
    # Create a basic profile
    return HardwareCapabilityProfile(
        hardware_class=hw_class,
        architecture=architecture,
        vendor=vendor,
        memory=memory,
        compute_units=4,
        supported_precisions=supported_precisions
    )


class HardwareAwareFaultToleranceTest(unittest.TestCase):
    """Tests for the Hardware-Aware Fault Tolerance System."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock objects
        self.db_manager = MockDBManager()
        self.coordinator = MockCoordinator()
        
        # Create test tasks with different configurations
        self.task_cpu = self.coordinator.add_task(
            "task_cpu_1",
            {"batch_size": 4, "precision": "fp32"},
            {"hardware": ["cpu"]}
        )
        
        self.task_gpu = self.coordinator.add_task(
            "task_gpu_1",
            {"batch_size": 8, "precision": "fp16"},
            {"hardware": ["cuda"]}
        )
        
        self.task_webgpu = self.coordinator.add_task(
            "task_webgpu_1",
            {"batch_size": 2, "precision": "fp16"},
            {"hardware": ["webgpu"], "browser": "chrome"}
        )
        
        # Create test workers with different capabilities
        self.worker_cpu = self.coordinator.add_worker(
            "worker_cpu_1",
            create_hardware_profile("CPU")
        )
        
        self.worker_gpu = self.coordinator.add_worker(
            "worker_gpu_1",
            create_hardware_profile("GPU")
        )
        
        self.worker_webgpu = self.coordinator.add_worker(
            "worker_webgpu_1",
            create_hardware_profile("WEBGPU")
        )
        
        # Create fault tolerance manager for testing
        self.manager = HardwareAwareFaultToleranceManager(
            db_manager=self.db_manager,
            coordinator=self.coordinator
        )
        
        # Customize for testing
        self.manager.config["failure_pattern_threshold"] = 2  # Lower threshold for testing
        self.manager.config["checkpoint_interval"] = 1  # Shorter interval for testing
        
        # Start the manager
        self.manager.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the manager
        self.manager.stop()
    
    def test_basic_initialization(self):
        """Test that the manager initializes correctly."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.db_manager, self.db_manager)
        self.assertEqual(self.manager.coordinator, self.coordinator)
        self.assertIsNotNone(self.manager.config)
        self.assertIsNotNone(self.manager.failure_history)
        self.assertIsNotNone(self.manager.failure_patterns)
        self.assertIsNotNone(self.manager.task_states)
        self.assertIsNotNone(self.manager.checkpoints)
        self.assertIsNotNone(self.manager.recovery_history)
        self.assertIsNotNone(self.manager.checkpoint_thread)
    
    def test_error_categorization(self):
        """Test that errors are correctly categorized."""
        # Test OOM error
        error_info = {"message": "CUDA out of memory"}
        error_type = self.manager._categorize_error(error_info)
        self.assertEqual(error_type, FailureType.RESOURCE_EXHAUSTION)
        
        # Test CUDA error
        error_info = {"message": "CUDA error: device-side assert triggered"}
        error_type = self.manager._categorize_error(error_info)
        self.assertEqual(error_type, FailureType.HARDWARE_ERROR)
        
        # Test browser crash
        error_info = {"message": "browser crash detected"}
        error_type = self.manager._categorize_error(error_info)
        self.assertEqual(error_type, FailureType.BROWSER_FAILURE)
        
        # Test WebGPU context lost
        error_info = {"message": "webgpu context lost"}
        error_type = self.manager._categorize_error(error_info)
        self.assertEqual(error_type, FailureType.BROWSER_FAILURE)
        
        # Test worker crash
        error_info = {"message": "worker crash detected"}
        error_type = self.manager._categorize_error(error_info)
        self.assertEqual(error_type, FailureType.WORKER_CRASH)
        
        # Test timeout
        error_info = {"message": "operation timed out"}
        error_type = self.manager._categorize_error(error_info)
        self.assertEqual(error_type, FailureType.TIMEOUT)
        
        # Test default categorization
        error_info = {"message": "some random error"}
        error_type = self.manager._categorize_error(error_info)
        self.assertEqual(error_type, FailureType.SOFTWARE_ERROR)
    
    def test_handle_failure_cpu(self):
        """Test handling a CPU task failure."""
        # Simulate a CPU task failure
        error_info = {
            "message": "CPU runtime error: segmentation fault",
            "type": "runtime_error",
            "stacktrace": "test stack trace"
        }
        
        # Handle the failure
        recovery_action = self.manager.handle_failure(
            self.task_cpu, self.worker_cpu, error_info
        )
        
        # Check that the recovery action is appropriate for a CPU task
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.DELAYED_RETRY)
        self.assertEqual(recovery_action.worker_id, self.worker_cpu)
        self.assertGreater(recovery_action.delay, 0)
        
        # Check that the task state was updated
        task_state = self.manager.get_task_state(self.task_cpu)
        self.assertIn("last_failure", task_state)
        self.assertIn("retry_count", task_state)
        self.assertEqual(task_state["retry_count"], 1)
        
        # Check that the failure was added to history
        self.assertEqual(len(self.manager.failure_history), 1)
        failure = self.manager.failure_history[0]
        self.assertEqual(failure.task_id, self.task_cpu)
        self.assertEqual(failure.worker_id, self.worker_cpu)
        self.assertEqual(failure.error_type, FailureType.SOFTWARE_ERROR)
    
    def test_handle_failure_gpu(self):
        """Test handling a GPU task failure."""
        # Simulate a GPU task failure
        error_info = {
            "message": "CUDA error: device-side assert triggered",
            "type": "cuda_error",
            "stacktrace": "test stack trace"
        }
        
        # Handle the failure
        recovery_action = self.manager.handle_failure(
            self.task_gpu, self.worker_gpu, error_info
        )
        
        # Check the recovery action
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.DIFFERENT_WORKER)
        self.assertIsNone(recovery_action.worker_id)  # Different worker, so no specific worker
    
    def test_handle_failure_webgpu(self):
        """Test handling a WebGPU task failure."""
        # Simulate a WebGPU context lost error
        error_info = {
            "message": "webgpu context lost",
            "type": "webgpu_error",
            "stacktrace": "test stack trace"
        }
        
        # Handle the failure
        recovery_action = self.manager.handle_failure(
            self.task_webgpu, self.worker_webgpu, error_info
        )
        
        # Check the recovery action
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.BROWSER_RESTART)
    
    def test_handle_oom_error(self):
        """Test handling an out-of-memory error."""
        # Simulate an OOM error
        error_info = {
            "message": "CUDA out of memory",
            "type": "resource_error",
            "stacktrace": "test stack trace"
        }
        
        # Handle the failure
        recovery_action = self.manager.handle_failure(
            self.task_gpu, self.worker_gpu, error_info
        )
        
        # Check the recovery action
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.REDUCED_BATCH_SIZE)
        self.assertIsNotNone(recovery_action.modified_task)
        
        # Check that batch size was reduced
        batch_size = recovery_action.modified_task.get("config", {}).get("batch_size")
        self.assertEqual(batch_size, 4)  # Should be reduced from 8 to 4
    
    def test_reduced_precision(self):
        """Test creating a reduced precision task."""
        # Create a task with fp32 precision
        task_id = self.coordinator.add_task(
            "fp32_task",
            {"precision": "fp32", "batch_size": 4}
        )
        
        # Create reduced precision version
        modified_task = self.manager._create_reduced_precision_task(task_id)
        
        # Check that precision was reduced
        self.assertEqual(modified_task["config"]["precision"], "fp16")
        
        # Test reducing from fp16 to int8
        task_id = self.coordinator.add_task(
            "fp16_task",
            {"precision": "fp16", "batch_size": 4}
        )
        modified_task = self.manager._create_reduced_precision_task(task_id)
        self.assertEqual(modified_task["config"]["precision"], "int8")
    
    def test_reduced_batch_size(self):
        """Test creating a reduced batch size task."""
        # Create a task with batch_size field
        task_id = self.coordinator.add_task(
            "batch_task",
            {"batch_size": 16}
        )
        
        # Create reduced batch size version
        modified_task = self.manager._create_reduced_batch_task(task_id)
        
        # Check that batch size was reduced
        self.assertEqual(modified_task["config"]["batch_size"], 8)  # Reduced by half
        
        # Test reducing batch_sizes array
        task_id = self.coordinator.add_task(
            "batch_sizes_task",
            {"batch_sizes": [16, 32, 64]}
        )
        modified_task = self.manager._create_reduced_batch_task(task_id)
        self.assertEqual(modified_task["config"]["batch_sizes"], [8, 16, 32])
    
    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        # Create a checkpoint
        checkpoint_data = {
            "progress": 50,
            "completed_steps": 10,
            "results": [1, 2, 3, 4, 5]
        }
        
        checkpoint_id = self.manager.create_checkpoint(self.task_cpu, checkpoint_data)
        
        # Check that checkpoint was created
        self.assertIsNotNone(checkpoint_id)
        
        # Check that task state was updated
        task_state = self.manager.get_task_state(self.task_cpu)
        self.assertIn("last_checkpoint", task_state)
        self.assertEqual(task_state["last_checkpoint"]["checkpoint_id"], checkpoint_id)
        
        # Check that checkpoint is retrievable
        latest_checkpoint = self.manager.get_latest_checkpoint(self.task_cpu)
        self.assertEqual(latest_checkpoint, checkpoint_data)
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculation with exponential backoff."""
        # First attempt - base delay
        delay1 = self.manager._calculate_retry_delay(1)
        self.assertGreaterEqual(delay1, self.manager.config["base_delay"] * 0.8)  # Allow for jitter
        self.assertLessEqual(delay1, self.manager.config["base_delay"] * 1.2)
        
        # Second attempt - delay should increase
        delay2 = self.manager._calculate_retry_delay(2)
        self.assertGreater(delay2, delay1)
        
        # Third attempt - even more delay
        delay3 = self.manager._calculate_retry_delay(3)
        self.assertGreater(delay3, delay2)
        
        # Ensure max delay is respected
        delay10 = self.manager._calculate_retry_delay(10)
        self.assertLessEqual(delay10, self.manager.config["max_delay"])
    
    def test_failure_pattern_detection(self):
        """Test detection of failure patterns."""
        # Create multiple similar failures
        error_info = {
            "message": "CUDA error: device-side assert triggered",
            "type": "cuda_error"
        }
        
        # Create a second GPU task
        task_gpu_2 = self.coordinator.add_task(
            "task_gpu_2",
            {"batch_size": 8, "precision": "fp16"},
            {"hardware": ["cuda"]}
        )
        
        # Simulate multiple failures on the same GPU worker
        self.manager.handle_failure(self.task_gpu, self.worker_gpu, error_info)
        self.manager.handle_failure(task_gpu_2, self.worker_gpu, error_info)
        
        # Check that a pattern was detected
        self.assertGreaterEqual(len(self.manager.failure_patterns), 1)
        
        # Get the pattern and check its properties
        pattern = next(iter(self.manager.failure_patterns.values()))
        self.assertIn(pattern["type"], ["worker_id", "hardware_class", "error_type"])
        self.assertGreaterEqual(pattern["count"], 2)
        self.assertIn("recommended_action", pattern)
        
        # Create a third GPU task
        task_gpu_3 = self.coordinator.add_task(
            "task_gpu_3",
            {"batch_size": 8, "precision": "fp16"},
            {"hardware": ["cuda"]}
        )
        
        # Create another failure that should match the pattern
        recovery_action = self.manager.handle_failure(task_gpu_3, self.worker_gpu, error_info)
        
        # Check that the recovery action came from the pattern and is a valid pattern-based strategy
        self.assertIn(recovery_action.strategy, [
            RecoveryStrategy.DIFFERENT_WORKER,
            RecoveryStrategy.DIFFERENT_HARDWARE_CLASS,
            RecoveryStrategy.ESCALATION,
            RecoveryStrategy.REDUCED_BATCH_SIZE
        ])
        self.assertIn("pattern", recovery_action.message.lower())
    
    def test_apply_recovery_action(self):
        """Test applying a recovery action."""
        # Create a recovery action
        action = RecoveryAction(
            strategy=RecoveryStrategy.DIFFERENT_WORKER,
            message="Test recovery action"
        )
        
        # Apply the action
        success = apply_recovery_action(
            self.task_cpu,
            action,
            coordinator=self.coordinator
        )
        
        # Check that the action was applied
        self.assertTrue(success)
        self.assertEqual(len(self.coordinator.retried_tasks), 1)
        retried_task = self.coordinator.retried_tasks[0]
        self.assertEqual(retried_task["task_id"], self.task_cpu)
        
        # Test browser restart
        action = RecoveryAction(
            strategy=RecoveryStrategy.BROWSER_RESTART,
            worker_id=self.worker_webgpu,
            message="Restart browser"
        )
        
        success = apply_recovery_action(
            self.task_webgpu,
            action,
            coordinator=self.coordinator
        )
        
        self.assertTrue(success)
        self.assertIn(self.worker_webgpu, self.coordinator.restarted_browsers)
        
        # Test escalation
        action = RecoveryAction(
            strategy=RecoveryStrategy.ESCALATION,
            message="Escalate to human operator"
        )
        
        success = apply_recovery_action(
            self.task_cpu,
            action,
            coordinator=self.coordinator
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.coordinator.escalated_tasks), 1)
        escalated_task = self.coordinator.escalated_tasks[0]
        self.assertEqual(escalated_task["task_id"], self.task_cpu)
    
    def test_max_retries(self):
        """Test that max retries is respected."""
        # Set retry count to max_retries
        task_id = self.coordinator.add_task(
            "max_retry_task",
            {"batch_size": 4},
            retry_count=self.manager.config["max_retries"]
        )
        
        # Try to handle a failure
        error_info = {"message": "Test error"}
        action = self.manager.handle_failure(task_id, self.worker_cpu, error_info)
        
        # Should get escalation due to max retries
        self.assertEqual(action.strategy, RecoveryStrategy.ESCALATION)
        self.assertIn("maximum retry count", action.message)
    
    def test_helper_functions(self):
        """Test helper functions."""
        # Test create_recovery_manager
        manager = create_recovery_manager(
            coordinator=self.coordinator,
            db_manager=self.db_manager
        )
        
        self.assertIsInstance(manager, HardwareAwareFaultToleranceManager)
        manager.stop()  # Clean up
    
    def test_state_persistence(self):
        """Test state persistence to database."""
        # Create some state
        self.manager.handle_failure(
            self.task_cpu,
            self.worker_cpu,
            {"message": "Test error 1"}
        )
        
        self.manager.handle_failure(
            self.task_gpu,
            self.worker_gpu,
            {"message": "Test error 2"}
        )
        
        # Persist state
        self.manager._persist_state()
        
        # Check that state was saved
        state = self.db_manager.data.get("state")
        self.assertIsNotNone(state)
        self.assertIn("task_states", state)
        self.assertIn("failure_patterns", state)
        self.assertIn("recovery_history", state)
        
        # Create a new manager and load the state
        new_manager = HardwareAwareFaultToleranceManager(
            db_manager=self.db_manager,
            coordinator=self.coordinator
        )
        
        # Load the state
        new_manager._load_persisted_state()
        
        # Check that state was loaded
        self.assertEqual(len(new_manager.task_states), len(self.manager.task_states))
        
        # Clean up
        new_manager.stop()


if __name__ == "__main__":
    unittest.main()