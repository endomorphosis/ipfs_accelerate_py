#!/usr/bin/env python3
"""
Test the task scheduler component of the distributed testing framework.

This script tests the TaskScheduler's ability to match tasks to workers
based on hardware requirements and priorities.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from duckdb_api.distributed_testing.coordinator import DatabaseManager
from duckdb_api.distributed_testing.task_scheduler import TaskScheduler


class TaskSchedulerTest(unittest.TestCase):
    """
    Tests for the TaskScheduler component.
    
    Tests the matching of tasks to workers based on hardware requirements,
    priorities, and other criteria.
    """
    
    def setUp(self):
        """Set up test environment with TaskScheduler and test data."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".duckdb")
        
        # Create database manager
        self.db_manager = DatabaseManager(self.db_path)
        
        # Create task scheduler
        self.task_scheduler = TaskScheduler(self.db_manager)
        
        # Define test workers with different capabilities
        self.cpu_worker = {
            "worker_id": "cpu_worker",
            "capabilities": {
                "hardware_types": ["cpu"],
                "memory_gb": 8
            }
        }
        
        self.cuda_worker = {
            "worker_id": "cuda_worker",
            "capabilities": {
                "hardware_types": ["cpu", "cuda"],
                "memory_gb": 16,
                "cuda_compute": 7.5
            }
        }
        
        self.rocm_worker = {
            "worker_id": "rocm_worker",
            "capabilities": {
                "hardware_types": ["cpu", "rocm"],
                "memory_gb": 16
            }
        }
        
        self.browser_worker = {
            "worker_id": "browser_worker",
            "capabilities": {
                "hardware_types": ["cpu", "webgpu", "webnn"],
                "browsers": ["chrome", "firefox", "edge"],
                "memory_gb": 8
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Close the database connection
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()
        
        # Remove temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_cpu_task_matching(self):
        """Test matching CPU tasks to appropriate workers."""
        # Create a CPU task
        cpu_task = {
            "task_id": "cpu_task",
            "priority": 1,
            "create_time": datetime.now(),
            "requirements": {"hardware": ["cpu"]}
        }
        
        # Add task to scheduler
        self.task_scheduler.add_task(
            cpu_task["task_id"], 
            "test", 
            cpu_task["priority"], 
            {"test_type": "cpu_test"}, 
            cpu_task["requirements"]
        )
        
        # Task should match all workers
        for worker in [self.cpu_worker, self.cuda_worker, self.rocm_worker, self.browser_worker]:
            self.assertTrue(
                self.task_scheduler._worker_meets_requirements(
                    worker["capabilities"],
                    cpu_task["requirements"]
                ),
                f"CPU task should match {worker['worker_id']}"
            )
    
    def test_cuda_task_matching(self):
        """Test matching CUDA tasks to CUDA-capable workers."""
        # Create a CUDA task
        cuda_task = {
            "task_id": "cuda_task",
            "priority": 1,
            "create_time": datetime.now(),
            "requirements": {"hardware": ["cuda"]}
        }
        
        # Add task to scheduler
        self.task_scheduler.add_task(
            cuda_task["task_id"], 
            "test", 
            cuda_task["priority"], 
            {"test_type": "cuda_test"}, 
            cuda_task["requirements"]
        )
        
        # Task should match only CUDA worker
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.cpu_worker["capabilities"],
                cuda_task["requirements"]
            ),
            "CUDA task should not match CPU worker"
        )
        
        self.assertTrue(
            self.task_scheduler._worker_meets_requirements(
                self.cuda_worker["capabilities"],
                cuda_task["requirements"]
            ),
            "CUDA task should match CUDA worker"
        )
        
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.rocm_worker["capabilities"],
                cuda_task["requirements"]
            ),
            "CUDA task should not match ROCm worker"
        )
        
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.browser_worker["capabilities"],
                cuda_task["requirements"]
            ),
            "CUDA task should not match browser worker"
        )
    
    def test_browser_task_matching(self):
        """Test matching browser tasks to browser-capable workers."""
        # Create a browser task
        browser_task = {
            "task_id": "browser_task",
            "priority": 1,
            "create_time": datetime.now(),
            "requirements": {"hardware": ["webgpu"], "browser": "firefox"}
        }
        
        # Add task to scheduler
        self.task_scheduler.add_task(
            browser_task["task_id"], 
            "test", 
            browser_task["priority"], 
            {"test_type": "browser_test"}, 
            browser_task["requirements"]
        )
        
        # Task should match only browser worker
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.cpu_worker["capabilities"],
                browser_task["requirements"]
            ),
            "Browser task should not match CPU worker"
        )
        
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.cuda_worker["capabilities"],
                browser_task["requirements"]
            ),
            "Browser task should not match CUDA worker"
        )
        
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.rocm_worker["capabilities"],
                browser_task["requirements"]
            ),
            "Browser task should not match ROCm worker"
        )
        
        self.assertTrue(
            self.task_scheduler._worker_meets_requirements(
                self.browser_worker["capabilities"],
                browser_task["requirements"]
            ),
            "Browser task should match browser worker"
        )
    
    def test_memory_requirement_matching(self):
        """Test matching tasks with memory requirements."""
        # Create a high-memory task
        high_memory_task = {
            "task_id": "high_memory_task",
            "priority": 1,
            "create_time": datetime.now(),
            "requirements": {"hardware": ["cpu"], "min_memory_gb": 12}
        }
        
        # Add task to scheduler
        self.task_scheduler.add_task(
            high_memory_task["task_id"], 
            "test", 
            high_memory_task["priority"], 
            {"test_type": "memory_test"}, 
            high_memory_task["requirements"]
        )
        
        # Task should match only workers with enough memory
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.cpu_worker["capabilities"],
                high_memory_task["requirements"]
            ),
            "High memory task should not match CPU worker with insufficient memory"
        )
        
        self.assertTrue(
            self.task_scheduler._worker_meets_requirements(
                self.cuda_worker["capabilities"],
                high_memory_task["requirements"]
            ),
            "High memory task should match CUDA worker with sufficient memory"
        )
        
        self.assertTrue(
            self.task_scheduler._worker_meets_requirements(
                self.rocm_worker["capabilities"],
                high_memory_task["requirements"]
            ),
            "High memory task should match ROCm worker with sufficient memory"
        )
        
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.browser_worker["capabilities"],
                high_memory_task["requirements"]
            ),
            "High memory task should not match browser worker with insufficient memory"
        )
    
    def test_cuda_compute_requirement_matching(self):
        """Test matching tasks with CUDA compute capability requirements."""
        # Create a task requiring high CUDA compute capability
        cuda_compute_task = {
            "task_id": "cuda_compute_task",
            "priority": 1,
            "create_time": datetime.now(),
            "requirements": {"hardware": ["cuda"], "min_cuda_compute": 7.0}
        }
        
        # Create another task requiring even higher CUDA compute capability
        high_cuda_compute_task = {
            "task_id": "high_cuda_compute_task",
            "priority": 1,
            "create_time": datetime.now(),
            "requirements": {"hardware": ["cuda"], "min_cuda_compute": 8.0}
        }
        
        # Add tasks to scheduler
        self.task_scheduler.add_task(
            cuda_compute_task["task_id"], 
            "test", 
            cuda_compute_task["priority"], 
            {"test_type": "cuda_compute_test"}, 
            cuda_compute_task["requirements"]
        )
        
        self.task_scheduler.add_task(
            high_cuda_compute_task["task_id"], 
            "test", 
            high_cuda_compute_task["priority"], 
            {"test_type": "high_cuda_compute_test"}, 
            high_cuda_compute_task["requirements"]
        )
        
        # First task should match CUDA worker
        self.assertTrue(
            self.task_scheduler._worker_meets_requirements(
                self.cuda_worker["capabilities"],
                cuda_compute_task["requirements"]
            ),
            "CUDA compute task should match CUDA worker with sufficient compute capability"
        )
        
        # Second task should not match CUDA worker
        self.assertFalse(
            self.task_scheduler._worker_meets_requirements(
                self.cuda_worker["capabilities"],
                high_cuda_compute_task["requirements"]
            ),
            "High CUDA compute task should not match CUDA worker with insufficient compute capability"
        )
    
    def test_priority_based_scheduling(self):
        """Test that tasks are scheduled based on priority."""
        # Add tasks with different priorities
        self.task_scheduler.add_task(
            "low_priority", 
            "test", 
            10, 
            {"test_type": "low_priority_test"}, 
            {"hardware": ["cpu"]}
        )
        
        self.task_scheduler.add_task(
            "medium_priority", 
            "test", 
            5, 
            {"test_type": "medium_priority_test"}, 
            {"hardware": ["cpu"]}
        )
        
        self.task_scheduler.add_task(
            "high_priority", 
            "test", 
            1, 
            {"test_type": "high_priority_test"}, 
            {"hardware": ["cpu"]}
        )
        
        # Get next task (should be high priority)
        next_task = self.task_scheduler.get_next_task(
            "cpu_worker", 
            self.cpu_worker["capabilities"]
        )
        
        self.assertIsNotNone(next_task, "Should have a task to assign")
        self.assertEqual(next_task["task_id"], "high_priority", "High priority task should be assigned first")
        
        # Get next task (should be medium priority)
        next_task = self.task_scheduler.get_next_task(
            "cpu_worker", 
            self.cpu_worker["capabilities"]
        )
        
        self.assertIsNotNone(next_task, "Should have a task to assign")
        self.assertEqual(next_task["task_id"], "medium_priority", "Medium priority task should be assigned second")
        
        # Get next task (should be low priority)
        next_task = self.task_scheduler.get_next_task(
            "cpu_worker", 
            self.cpu_worker["capabilities"]
        )
        
        self.assertIsNotNone(next_task, "Should have a task to assign")
        self.assertEqual(next_task["task_id"], "low_priority", "Low priority task should be assigned last")
        
        # No more tasks to assign
        next_task = self.task_scheduler.get_next_task(
            "cpu_worker", 
            self.cpu_worker["capabilities"]
        )
        
        self.assertIsNone(next_task, "Should have no more tasks to assign")


if __name__ == "__main__":
    unittest.main()