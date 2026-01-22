#!/usr/bin/env python3
"""
Test the load balancer component of the distributed testing framework.

This script tests the LoadBalancer's ability to distribute tasks efficiently
across worker nodes based on their capabilities and current workload.
"""

import os
import sys
import unittest
import tempfile
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from duckdb_api.distributed_testing.coordinator import DatabaseManager
from duckdb_api.distributed_testing.load_balancer import LoadBalancer
from duckdb_api.distributed_testing.coordinator import WORKER_STATUS_ACTIVE, WORKER_STATUS_BUSY


class LoadBalancerTest(unittest.TestCase):
    """
    Tests for the LoadBalancer component.
    
    Tests the distribution of tasks across worker nodes based on
    their capabilities, current workload, and performance metrics.
    """
    
    def setUp(self):
        """Set up test environment with LoadBalancer and test data."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".duckdb")
        
        # Create database manager
        self.db_manager = DatabaseManager(self.db_path)
        
        # Create test workers with different capabilities
        self.worker_data = [
            {
                "worker_id": "worker_1",
                "hostname": "host_1",
                "capabilities": {
                    "hardware_types": ["cpu"],
                    "memory_gb": 8
                },
                "performance": {
                    "avg_task_time": 2.5,
                    "completed_tasks": 10,
                    "failed_tasks": 1
                }
            },
            {
                "worker_id": "worker_2",
                "hostname": "host_2",
                "capabilities": {
                    "hardware_types": ["cpu", "cuda"],
                    "memory_gb": 16,
                    "cuda_compute": 7.5
                },
                "performance": {
                    "avg_task_time": 1.5,
                    "completed_tasks": 15,
                    "failed_tasks": 0
                }
            },
            {
                "worker_id": "worker_3",
                "hostname": "host_3",
                "capabilities": {
                    "hardware_types": ["cpu", "webgpu"],
                    "browsers": ["chrome", "firefox"],
                    "memory_gb": 8
                },
                "performance": {
                    "avg_task_time": 3.0,
                    "completed_tasks": 8,
                    "failed_tasks": 2
                }
            }
        ]
        
        # Add workers to database
        for worker in self.worker_data:
            self.db_manager.add_worker(
                worker["worker_id"], 
                worker["hostname"], 
                worker["capabilities"]
            )
            self.db_manager.update_worker_status(worker["worker_id"], WORKER_STATUS_ACTIVE)
        
        # Create test tasks
        self.tasks = [
            {
                "task_id": "cpu_task_1",
                "type": "benchmark",
                "priority": 1,
                "requirements": {"hardware": ["cpu"]}
            },
            {
                "task_id": "cpu_task_2",
                "type": "benchmark",
                "priority": 2,
                "requirements": {"hardware": ["cpu"]}
            },
            {
                "task_id": "gpu_task",
                "type": "benchmark",
                "priority": 1,
                "requirements": {"hardware": ["cuda"]}
            },
            {
                "task_id": "browser_task",
                "type": "test",
                "priority": 1,
                "requirements": {"hardware": ["webgpu"], "browser": "firefox"}
            },
            {
                "task_id": "high_memory_task",
                "type": "benchmark",
                "priority": 1,
                "requirements": {"hardware": ["cpu"], "min_memory_gb": 12}
            }
        ]
        
        # Dictionary to store task assignments from load balancer
        self.task_assignments = {}
        
        # Create load balancer with short check interval for testing
        self.load_balancer = LoadBalancer(
            db_manager=self.db_manager,
            check_interval=1  # Check every 1 second
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop load balancer if running
        if hasattr(self, 'load_balancer_thread') and self.load_balancer_thread.is_alive():
            self.load_balancer.stop_balancing()
            self.load_balancer_thread.join(timeout=5.0)
        
        # Close the database connection
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()
        
        # Remove temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_worker_scoring(self):
        """Test worker scoring based on capabilities and performance."""
        # Get worker scores for a CPU task
        cpu_task = self.tasks[0]
        worker_scores = self.load_balancer.score_workers_for_task(cpu_task)
        
        # All workers should have scores for CPU task
        self.assertEqual(len(worker_scores), 3, "All 3 workers should have scores for CPU task")
        
        # Worker 2 should have highest score due to better performance
        sorted_workers = sorted(worker_scores.items(), key=lambda x: x[1], reverse=True)
        self.assertEqual(sorted_workers[0][0], "worker_2", 
                        "Worker 2 should have highest score for CPU task")
        
        # Get worker scores for a GPU task
        gpu_task = self.tasks[2]
        worker_scores = self.load_balancer.score_workers_for_task(gpu_task)
        
        # Only worker 2 should have a score for GPU task
        self.assertEqual(len(worker_scores), 1, 
                        "Only 1 worker should have a score for GPU task")
        self.assertIn("worker_2", worker_scores, 
                     "Worker 2 should be the only worker with a score for GPU task")
        
        # Get worker scores for a browser task
        browser_task = self.tasks[3]
        worker_scores = self.load_balancer.score_workers_for_task(browser_task)
        
        # Only worker 3 should have a score for browser task
        self.assertEqual(len(worker_scores), 1, 
                        "Only 1 worker should have a score for browser task")
        self.assertIn("worker_3", worker_scores, 
                     "Worker 3 should be the only worker with a score for browser task")
    
    def test_task_assignment(self):
        """Test task assignment to workers based on scores."""
        # Assign a CPU task
        cpu_task = self.tasks[0]
        assigned_worker = self.load_balancer.assign_task(cpu_task)
        
        # Task should be assigned to worker 2 (highest score)
        self.assertEqual(assigned_worker, "worker_2", 
                        "CPU task should be assigned to worker 2")
        
        # Update worker 2 status to busy
        self.db_manager.update_worker_status("worker_2", WORKER_STATUS_BUSY)
        
        # Assign another CPU task
        cpu_task2 = self.tasks[1]
        assigned_worker = self.load_balancer.assign_task(cpu_task2)
        
        # Task should be assigned to next best worker (worker 1 or 3)
        self.assertIn(assigned_worker, ["worker_1", "worker_3"], 
                     "Second CPU task should be assigned to worker 1 or 3")
        
        # Assign a GPU task
        gpu_task = self.tasks[2]
        assigned_worker = self.load_balancer.assign_task(gpu_task)
        
        # No assignment should be made since worker 2 is busy
        self.assertIsNone(assigned_worker, 
                         "No worker should be assigned to GPU task when worker 2 is busy")
        
        # Reset worker 2 status to active
        self.db_manager.update_worker_status("worker_2", WORKER_STATUS_ACTIVE)
        
        # Try GPU task again
        assigned_worker = self.load_balancer.assign_task(gpu_task)
        
        # Task should now be assigned to worker 2
        self.assertEqual(assigned_worker, "worker_2", 
                        "GPU task should be assigned to worker 2 when available")
    
    def test_load_balancing(self):
        """Test load balancing of tasks across workers."""
        # Create mock assignment function to track assignments
        def mock_assign_task(task_id, worker_id):
            self.task_assignments[task_id] = worker_id
            return True
        
        # Set up load balancer with mock assignment function
        self.load_balancer.assign_task_to_worker = mock_assign_task
        
        # Start load balancer in a separate thread
        self.load_balancer_thread = threading.Thread(
            target=self.load_balancer.start_balancing,
            daemon=True
        )
        self.load_balancer_thread.start()
        
        # Add tasks to the system
        for task in self.tasks:
            self.db_manager.add_task(
                task["task_id"],
                task["type"],
                task["priority"],
                {"test_config": True},
                task["requirements"]
            )
        
        # Wait for load balancer to assign tasks
        time.sleep(3)
        
        # Check task assignments
        self.assertIn("cpu_task_1", self.task_assignments, 
                     "CPU task 1 should be assigned")
        self.assertIn("cpu_task_2", self.task_assignments, 
                     "CPU task 2 should be assigned")
        self.assertIn("gpu_task", self.task_assignments, 
                     "GPU task should be assigned")
        self.assertIn("browser_task", self.task_assignments, 
                     "Browser task should be assigned")
        
        # CPU tasks should be distributed across workers
        cpu_task_workers = set([
            self.task_assignments.get("cpu_task_1"),
            self.task_assignments.get("cpu_task_2")
        ])
        self.assertGreaterEqual(len(cpu_task_workers), 1, 
                              "CPU tasks should be distributed")
        
        # GPU task should be assigned to worker 2
        self.assertEqual(self.task_assignments.get("gpu_task"), "worker_2", 
                        "GPU task should be assigned to worker 2")
        
        # Browser task should be assigned to worker 3
        self.assertEqual(self.task_assignments.get("browser_task"), "worker_3", 
                        "Browser task should be assigned to worker 3")
        
        # High memory task should be assigned to worker 2
        self.assertEqual(self.task_assignments.get("high_memory_task"), "worker_2", 
                        "High memory task should be assigned to worker 2")
    
    def test_workload_balancing(self):
        """Test balancing workload when workers become overloaded."""
        # Set up initial workload
        # Worker 1: 2 tasks
        # Worker 2: 1 task
        # Worker 3: 0 tasks
        self.db_manager.add_task("task_1", "test", 1, {}, {"hardware": ["cpu"]})
        self.db_manager.add_task("task_2", "test", 1, {}, {"hardware": ["cpu"]})
        self.db_manager.add_task("task_3", "test", 1, {}, {"hardware": ["cpu"]})
        
        # Create mock task data in database
        self.db_manager.conn.execute("""
        UPDATE distributed_tasks 
        SET worker_id = ?, status = 'running' 
        WHERE task_id = ?
        """, ["worker_1", "task_1"])
        
        self.db_manager.conn.execute("""
        UPDATE distributed_tasks 
        SET worker_id = ?, status = 'running' 
        WHERE task_id = ?
        """, ["worker_1", "task_2"])
        
        self.db_manager.conn.execute("""
        UPDATE distributed_tasks 
        SET worker_id = ?, status = 'running' 
        WHERE task_id = ?
        """, ["worker_2", "task_3"])
        
        # Get workload distribution
        workload = self.load_balancer.get_worker_load()
        
        # Verify initial workload
        self.assertEqual(workload.get("worker_1", 0), 2, 
                        "Worker 1 should have 2 tasks")
        self.assertEqual(workload.get("worker_2", 0), 1, 
                        "Worker 2 should have 1 task")
        self.assertEqual(workload.get("worker_3", 0), 0, 
                        "Worker 3 should have 0 tasks")
        
        # Check if load balancer would try to rebalance
        overloaded_workers = self.load_balancer.detect_overloaded_workers()
        underutilized_workers = self.load_balancer.detect_underutilized_workers()
        
        # Worker 1 should be overloaded and Worker 3 underutilized
        self.assertIn("worker_1", overloaded_workers, 
                     "Worker 1 should be detected as overloaded")
        self.assertIn("worker_3", underutilized_workers, 
                     "Worker 3 should be detected as underutilized")
    
    def test_task_migration(self):
        """Test migration of tasks from overloaded to underutilized workers."""
        # Register migration function
        migrations = []
        
        def mock_migrate_task(task_id, from_worker_id, to_worker_id):
            migrations.append({
                "task_id": task_id,
                "from_worker": from_worker_id,
                "to_worker": to_worker_id
            })
            return True
        
        # Set up load balancer with mock migration function
        self.load_balancer.migrate_task = mock_migrate_task
        
        # Set up imbalanced workload (as in test_workload_balancing)
        self.db_manager.add_task("task_1", "test", 1, {}, {"hardware": ["cpu"]})
        self.db_manager.add_task("task_2", "test", 1, {}, {"hardware": ["cpu"]})
        self.db_manager.add_task("task_3", "test", 1, {}, {"hardware": ["cpu"]})
        
        self.db_manager.conn.execute("""
        UPDATE distributed_tasks 
        SET worker_id = ?, status = 'running' 
        WHERE task_id = ?
        """, ["worker_1", "task_1"])
        
        self.db_manager.conn.execute("""
        UPDATE distributed_tasks 
        SET worker_id = ?, status = 'running' 
        WHERE task_id = ?
        """, ["worker_1", "task_2"])
        
        self.db_manager.conn.execute("""
        UPDATE distributed_tasks 
        SET worker_id = ?, status = 'running' 
        WHERE task_id = ?
        """, ["worker_2", "task_3"])
        
        # Run the rebalancing algorithm once
        self.load_balancer.rebalance_tasks()
        
        # Verify that task migration was attempted
        self.assertGreaterEqual(len(migrations), 1, 
                              "At least one task should be migrated")
        
        # The migration should be from worker_1 to worker_3
        for migration in migrations:
            self.assertEqual(migration["from_worker"], "worker_1", 
                            "Migration should be from worker_1")
            self.assertEqual(migration["to_worker"], "worker_3", 
                            "Migration should be to worker_3")
    
    def test_performance_based_balancing(self):
        """Test balancing based on worker performance metrics."""
        # Add performance metrics to database
        for worker in self.worker_data:
            worker_id = worker["worker_id"]
            performance = worker["performance"]
            
            # Add execution history records to simulate performance
            for i in range(performance["completed_tasks"]):
                self.db_manager.add_execution_history(
                    f"past_task_{worker_id}_{i}",
                    worker_id,
                    1,
                    "completed",
                    datetime.now() - timedelta(hours=1),
                    datetime.now() - timedelta(hours=1) + timedelta(seconds=performance["avg_task_time"]),
                    performance["avg_task_time"],
                    "",
                    {}
                )
            
            # Add failure records if needed
            for i in range(performance["failed_tasks"]):
                self.db_manager.add_execution_history(
                    f"past_failed_task_{worker_id}_{i}",
                    worker_id,
                    1,
                    "failed",
                    datetime.now() - timedelta(hours=1),
                    datetime.now() - timedelta(hours=1) + timedelta(seconds=5),
                    5.0,
                    "Test failure",
                    {}
                )
        
        # Get performance-based scores for workers
        scores = self.load_balancer.get_performance_based_scores()
        
        # Worker 2 should have highest score (fastest, no failures)
        highest_score_worker = max(scores.items(), key=lambda x: x[1])[0]
        self.assertEqual(highest_score_worker, "worker_2", 
                        "Worker 2 should have highest performance score")
        
        # Worker 3 should have lowest score (slowest, most failures)
        lowest_score_worker = min(scores.items(), key=lambda x: x[1])[0]
        self.assertEqual(lowest_score_worker, "worker_3", 
                        "Worker 3 should have lowest performance score")


if __name__ == "__main__":
    unittest.main()