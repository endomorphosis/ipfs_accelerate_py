#!/usr/bin/env python3
"""
Enhanced Error Handling Demo

This script demonstrates the enhanced error handling system in action.
It simulates a coordinator with workers and tasks, and shows how various errors are handled.
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from duckdb_api.distributed_testing.distributed_error_handler import (
    DistributedErrorHandler,
    ErrorCategory
)
from duckdb_api.distributed_testing.coordinator_error_integration import (
    integrate_error_handler,
    execute_recovery_action
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)

# Enable more detailed logging for the demo
logging.getLogger("distributed_error_handler").setLevel(logging.DEBUG)
logging.getLogger("coordinator_error_integration").setLevel(logging.DEBUG)

logger = logging.getLogger("error_handling_demo")


class MockCoordinator:
    """Mock coordinator for demonstrating error handling."""
    
    def __init__(self):
        """Initialize mock coordinator."""
        self.tasks = {}
        self.workers = {}
        self.current_time = int(time.time())
        self.messages = []
        
        # Initialize with some test tasks and workers
        self._initialize_tasks()
        self._initialize_workers()
    
    def _initialize_tasks(self):
        """Initialize test tasks."""
        # Task running on worker1
        self.tasks["task1"] = {
            "id": "task1",
            "status": "running",
            "worker_id": "worker1",
            "type": "test",
            "test_file": "test_bert.py",
            "start_time": self.current_time - 60,
            "timeout_seconds": 600,
            "attempt_count": 1,
            "requirements": {
                "hardware": ["cuda"],
                "min_cuda_compute": 7.0,
                "min_memory_gb": 4.0
            }
        }
        
        # Task pending assignment
        self.tasks["task2"] = {
            "id": "task2",
            "status": "pending",
            "type": "benchmark",
            "benchmark_file": "benchmark_vit.py",
            "timeout_seconds": 1200
        }
        
        # Task running on worker2
        self.tasks["task3"] = {
            "id": "task3",
            "status": "running",
            "worker_id": "worker2",
            "type": "test",
            "test_file": "test_whisper.py",
            "start_time": self.current_time - 30,
            "timeout_seconds": 900,
            "attempt_count": 1,
            "requirements": {
                "hardware": ["cpu"],
                "min_memory_gb": 2.0
            }
        }
    
    def _initialize_workers(self):
        """Initialize test workers."""
        # CUDA-capable worker
        self.workers["worker1"] = {
            "id": "worker1",
            "status": "active",
            "last_seen": self.current_time,
            "capabilities": {
                "hardware_types": ["cuda", "cpu"],
                "cuda_compute": 7.5,
                "memory_gb": 16
            }
        }
        
        # CPU-only worker
        self.workers["worker2"] = {
            "id": "worker2",
            "status": "active",
            "last_seen": self.current_time,
            "capabilities": {
                "hardware_types": ["cpu"],
                "memory_gb": 8
            }
        }
        
        # WebGPU-capable worker
        self.workers["worker3"] = {
            "id": "worker3",
            "status": "active",
            "last_seen": self.current_time,
            "capabilities": {
                "hardware_types": ["webgpu", "cpu"],
                "browser": "chrome",
                "memory_gb": 4
            }
        }
    
    def handle_task_error(self, task_id, error, worker_id=None):
        """Original task error handler.
        
        This would be replaced by the enhanced handler.
        """
        logger.info(f"Original task error handler called for task {task_id}")
        return {"original": True, "task_id": task_id}
    
    def handle_worker_error(self, worker_id, error):
        """Original worker error handler.
        
        This would be replaced by the enhanced handler.
        """
        logger.info(f"Original worker error handler called for worker {worker_id}")
        return {"original": True, "worker_id": worker_id}
    
    def get_current_time(self):
        """Get current time."""
        return self.current_time
    
    def send_message_to_worker(self, worker_id, message):
        """Send a message to a worker.
        
        In this demo, just logs the message.
        """
        logger.info(f"Sending message to worker {worker_id}: {message}")
        self.messages.append({"worker_id": worker_id, "message": message})
    
    def get_backup_coordinator_url(self):
        """Get URL of backup coordinator.
        
        In this demo, returns a mock URL.
        """
        return "http://backup-coordinator:8080"


def simulate_task_errors(coordinator):
    """Simulate various task errors and demonstrate error handling."""
    logger.info("\n=== Simulating Task Errors ===\n")
    
    # 1. Connection error
    logger.info("\n--- Simulating Connection Error ---\n")
    error_connection = {
        "type": "ConnectionError",
        "message": "Connection refused by remote host",
        "traceback": "...stack trace...",
        "timestamp": coordinator.get_current_time()
    }
    
    result_connection = coordinator.handle_task_error("task1", error_connection, "worker1")
    logger.info(f"Connection Error Result: {json.dumps(result_connection, indent=2, default=str)}")
    
    # 2. Memory error
    logger.info("\n--- Simulating Memory Error ---\n")
    error_memory = {
        "type": "MemoryError",
        "message": "Out of memory",
        "traceback": "...stack trace...",
        "timestamp": coordinator.get_current_time()
    }
    
    result_memory = coordinator.handle_task_error("task1", error_memory, "worker1")
    logger.info(f"Memory Error Result: {json.dumps(result_memory, indent=2, default=str)}")
    
    # 3. Test assertion error
    logger.info("\n--- Simulating Test Assertion Error ---\n")
    error_assertion = {
        "type": "AssertionError",
        "message": "Test assertion failed: expected True, got False",
        "traceback": "...stack trace...",
        "timestamp": coordinator.get_current_time()
    }
    
    result_assertion = coordinator.handle_task_error("task3", error_assertion, "worker2")
    logger.info(f"Assertion Error Result: {json.dumps(result_assertion, indent=2, default=str)}")


def simulate_worker_errors(coordinator):
    """Simulate various worker errors and demonstrate error handling."""
    logger.info("\n=== Simulating Worker Errors ===\n")
    
    # 1. Worker disconnect error
    logger.info("\n--- Simulating Worker Disconnect Error ---\n")
    error_disconnect = {
        "type": "WorkerDisconnectedError",
        "message": "Worker disconnected unexpectedly",
        "traceback": "...stack trace...",
        "timestamp": coordinator.get_current_time()
    }
    
    result_disconnect = coordinator.handle_worker_error("worker1", error_disconnect)
    logger.info(f"Worker Disconnect Result: {json.dumps(result_disconnect, indent=2, default=str)}")
    
    # 2. Worker timeout error
    logger.info("\n--- Simulating Worker Timeout Error ---\n")
    error_timeout = {
        "type": "WorkerTimeoutError",
        "message": "Worker timed out during task execution",
        "traceback": "...stack trace...",
        "timestamp": coordinator.get_current_time()
    }
    
    result_timeout = coordinator.handle_worker_error("worker2", error_timeout)
    logger.info(f"Worker Timeout Result: {json.dumps(result_timeout, indent=2, default=str)}")


def simulate_hardware_errors(coordinator):
    """Simulate various hardware errors and demonstrate error handling."""
    logger.info("\n=== Simulating Hardware Errors ===\n")
    
    # 1. Hardware not available error
    logger.info("\n--- Simulating Hardware Not Available Error ---\n")
    error_hardware = {
        "type": "HardwareError",
        "message": "CUDA device not available",
        "traceback": "...stack trace...",
        "timestamp": coordinator.get_current_time(),
        "hardware_context": {
            "hardware_type": "cuda",
            "device_id": 0
        }
    }
    
    result_hardware = coordinator.handle_task_error("task1", error_hardware, "worker1")
    logger.info(f"Hardware Error Result: {json.dumps(result_hardware, indent=2, default=str)}")


def show_task_state(coordinator):
    """Show the current state of tasks after error handling."""
    logger.info("\n=== Task State After Error Handling ===\n")
    
    for task_id, task in coordinator.tasks.items():
        logger.info(f"Task {task_id}:")
        logger.info(f"  Status: {task.get('status')}")
        logger.info(f"  Worker: {task.get('worker_id', 'None')}")
        logger.info(f"  Attempt: {task.get('attempt_count', 1)}")
        
        # Show additional fields added by error handling
        for key in sorted(task.keys()):
            if key not in ["id", "status", "worker_id", "type", "start_time", 
                          "timeout_seconds", "attempt_count", "requirements"]:
                logger.info(f"  {key}: {task[key]}")
        
        logger.info("")


def show_worker_state(coordinator):
    """Show the current state of workers after error handling."""
    logger.info("\n=== Worker State After Error Handling ===\n")
    
    for worker_id, worker in coordinator.workers.items():
        logger.info(f"Worker {worker_id}:")
        logger.info(f"  Status: {worker.get('status')}")
        logger.info(f"  Last Seen: {datetime.fromtimestamp(worker.get('last_seen', 0))}")
        
        # Show additional fields added by error handling
        for key in sorted(worker.keys()):
            if key not in ["id", "status", "last_seen", "capabilities"]:
                logger.info(f"  {key}: {worker[key]}")
        
        logger.info("")


def show_messages(coordinator):
    """Show messages sent to workers during error handling."""
    logger.info("\n=== Messages Sent During Error Handling ===\n")
    
    for i, message in enumerate(coordinator.messages):
        logger.info(f"Message {i+1}:")
        logger.info(f"  To: Worker {message['worker_id']}")
        logger.info(f"  Type: {message['message'].get('type')}")
        logger.info(f"  Command: {message['message'].get('command')}")
        logger.info(f"  Timestamp: {datetime.fromtimestamp(message['message'].get('timestamp', 0))}")
        logger.info("")


def main():
    """Main function to run the error handling demo."""
    logger.info("Starting Enhanced Error Handling Demo")
    
    # Create a mock coordinator
    coordinator = MockCoordinator()
    
    # Integrate error handler
    coordinator = integrate_error_handler(coordinator)
    
    # Simulate various errors
    simulate_task_errors(coordinator)
    simulate_worker_errors(coordinator)
    simulate_hardware_errors(coordinator)
    
    # Show state after error handling
    show_task_state(coordinator)
    show_worker_state(coordinator)
    show_messages(coordinator)
    
    logger.info("Enhanced Error Handling Demo Complete")


if __name__ == "__main__":
    main()