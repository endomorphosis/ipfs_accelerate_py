#!/usr/bin/env python3
"""
Integration test for the distributed testing framework.

This script tests the integration between all components of the framework:
- coordinator.py: Central server managing tasks and workers
- worker.py: Worker nodes that execute tasks
- task_scheduler.py: Intelligent task scheduling and distribution
- load_balancer.py: Adaptive workload distribution
- health_monitor.py: Worker health monitoring and recovery
- dashboard_server.py: Web-based monitoring interface

The test validates that all components work together seamlessly in different scenarios.
"""

import os
import sys
import json
import time
import anyio
import unittest
import tempfile
import threading
import subprocess
import websockets
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from data.duckdb.distributed_testing.coordinator import CoordinatorServer, DatabaseManager, SecurityManager
from data.duckdb.distributed_testing.task_scheduler import TaskScheduler
from data.duckdb.distributed_testing.load_balancer import LoadBalancer
from data.duckdb.distributed_testing.health_monitor import HealthMonitor
from data.duckdb.distributed_testing.dashboard_server import DashboardServer
from data.duckdb.distributed_testing.run_test import generate_security_config


class DistributedFrameworkIntegrationTest(unittest.TestCase):
    """
    Integration tests for the distributed testing framework.
    
    This test suite validates that all components of the distributed testing
    framework work together seamlessly.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with coordinator, dashboard, and workers."""
        print("\nSetting up distributed testing environment...")
        
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {cls.temp_dir}")
        
        # Create a temporary database file
        cls.db_path = os.path.join(cls.temp_dir, "test_db.duckdb")
        print(f"Using database path: {cls.db_path}")
        
        # Create security configuration
        cls.security_config_path = os.path.join(cls.temp_dir, "security_config.json")
        cls.security_config = generate_security_config(cls.security_config_path)
        cls.worker_api_key = cls.security_config["api_keys"]["worker"]
        print(f"Created security configuration with worker API key")
        
        # Set up coordinator
        cls.coordinator_host = "localhost"
        cls.coordinator_port = 8079  # Use non-standard port for testing
        cls.coordinator_url = f"ws://{cls.coordinator_host}:{cls.coordinator_port}"
        
        # Set up dashboard
        cls.dashboard_host = "localhost"
        cls.dashboard_port = 8082  # Different port for dashboard
        cls.dashboard_url = f"http://{cls.dashboard_host}:{cls.dashboard_port}"
        
        # Start the coordinator in a separate thread
        cls.coordinator_started = anyio.Event()
        cls.coordinator_stopped = anyio.Event()
        cls.coordinator_thread = threading.Thread(
            target=cls._run_coordinator,
            daemon=True
        )
        cls.coordinator_thread.start()
        
        # Wait for coordinator to start
        timeout = 10
        start_time = time.time()
        while not cls.coordinator_started.is_set() and (time.time() - start_time) < timeout:
            time.sleep(0.5)
        
        if not cls.coordinator_started.is_set():
            raise TimeoutError("Coordinator failed to start within timeout")
        
        print(f"Coordinator started at {cls.coordinator_url}")
        
        # Start the dashboard in a separate thread
        cls.dashboard_started = anyio.Event()
        cls.dashboard_thread = threading.Thread(
            target=cls._run_dashboard,
            daemon=True
        )
        cls.dashboard_thread.start()
        
        # Wait for dashboard to start
        timeout = 5
        start_time = time.time()
        while not cls.dashboard_started.is_set() and (time.time() - start_time) < timeout:
            time.sleep(0.5)
        
        if not cls.dashboard_started.is_set():
            raise TimeoutError("Dashboard failed to start within timeout")
        
        print(f"Dashboard started at {cls.dashboard_url}")
        
        # Start worker processes
        cls.worker_processes = []
        cls._start_workers(2)
        
        # Wait for workers to connect and register
        time.sleep(3)
        print("Setup complete")

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        print("\nCleaning up distributed testing environment...")
        
        # Stop all worker processes
        for worker_process in cls.worker_processes:
            print(f"Terminating worker process {worker_process.pid}")
            worker_process.terminate()
            worker_process.wait()
        
        # Stop the coordinator
        print("Stopping coordinator")
        cls.coordinator_stopped.set()
        cls.coordinator_thread.join(timeout=5.0)
        
        # Stop the dashboard
        print("Stopping dashboard")
        if hasattr(cls, 'dashboard_server'):
            cls.dashboard_server.stop()
            cls.dashboard_thread.join(timeout=5.0)
        
        # Clean up temporary directory
        if os.path.exists(cls.temp_dir):
            for file in os.listdir(cls.temp_dir):
                os.remove(os.path.join(cls.temp_dir, file))
            os.rmdir(cls.temp_dir)
            print(f"Removed temporary directory: {cls.temp_dir}")
    
    @classmethod
    def _run_coordinator(cls):
        """Run the coordinator in a separate thread."""
        try:
            async def coordinator_task():
                # Create database manager
                db_manager = DatabaseManager(cls.db_path)

                # Create security manager with the database
                security_manager = SecurityManager(
                    db_manager,
                    token_secret=cls.security_config["token_secret"],
                )

                # Create coordinator
                cls.coordinator = CoordinatorServer(
                    host=cls.coordinator_host,
                    port=cls.coordinator_port,
                    db_path=cls.db_path,
                    token_secret=cls.security_config["token_secret"],
                    heartbeat_timeout=5,  # Short timeout for testing
                )

                # Save references to components for testing
                cls.db_manager = db_manager
                cls.task_scheduler = cls.coordinator.task_manager
                cls.worker_manager = cls.coordinator.worker_manager
                cls.security_manager = security_manager

                # Create helper function to know when coordinator is ready
                async def on_coordinator_start():
                    cls.coordinator_started.set()
                    await cls.coordinator_stopped.wait()
                    await cls.coordinator.stop()

                # Run coordinator and wait for stop signal
                async with anyio.create_task_group() as tg:
                    tg.start_soon(cls.coordinator.start)
                    await on_coordinator_start()
                    tg.cancel_scope.cancel()

            anyio.run(coordinator_task)
        except Exception as e:
            print(f"Error in coordinator thread: {e}")
            import traceback
            traceback.print_exc()
    
    @classmethod
    def _run_dashboard(cls):
        """Run the dashboard in a separate thread."""
        try:
            # Create dashboard server
            cls.dashboard_server = DashboardServer(
                host=cls.dashboard_host,
                port=cls.dashboard_port,
                coordinator_url=f"http://{cls.coordinator_host}:{cls.coordinator_port}",
                auto_open=False
            )
            
            # Start dashboard
            cls.dashboard_server.start()
            cls.dashboard_started.set()
            
            # Keep running until application exits
            while True:
                time.sleep(1)
                if not cls.coordinator_started.is_set():
                    break
                
        except Exception as e:
            print(f"Error in dashboard thread: {e}")
            import traceback
            traceback.print_exc()
    
    @classmethod
    def _start_workers(cls, count=2):
        """Start multiple worker processes."""
        for i in range(count):
            worker_id = f"test_worker_{i}"
            worker_dir = os.path.join(cls.temp_dir, f"worker_{i}")
            os.makedirs(worker_dir, exist_ok=True)
            
            worker_cmd = [
                sys.executable,
                os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
                "--coordinator", cls.coordinator_url,
                "--api-key", cls.worker_api_key,
                "--worker-id", worker_id,
                "--work-dir", worker_dir,
                "--reconnect-interval", "2",
                "--heartbeat-interval", "3"
            ]
            
            print(f"Starting worker {worker_id}")
            process = subprocess.Popen(worker_cmd)
            cls.worker_processes.append(process)
    
    def test_01_coordinator_initialization(self):
        """Test that coordinator components are properly initialized."""
        # Check if coordinator is running
        self.assertTrue(self.coordinator.running)
        
        # Check if database manager is initialized
        self.assertIsNotNone(self.db_manager)
        self.assertEqual(self.db_manager.db_path, self.db_path)
        
        # Check if task scheduler is initialized
        self.assertIsNotNone(self.task_scheduler)
        
        # Check if worker manager is initialized
        self.assertIsNotNone(self.worker_manager)
        
        # Check if security manager is initialized
        self.assertIsNotNone(self.security_manager)
    
    def test_02_worker_registration(self):
        """Test that workers are registered with coordinator."""
        # Get workers from worker manager
        workers = self.worker_manager.workers
        
        # Verify that our test workers are registered
        self.assertIn("test_worker_0", workers)
        self.assertIn("test_worker_1", workers)
        
        # Check worker status
        for worker_id in ["test_worker_0", "test_worker_1"]:
            worker = workers[worker_id]
            self.assertIn(worker["status"], ["active", "busy", "registered"])
            
            # Verify capabilities are detected
            self.assertIn("hardware_types", worker["capabilities"])
            self.assertIn("cpu", worker["capabilities"]["hardware_types"])
            
            # Verify platform information
            self.assertIn("platform", worker["capabilities"])
            self.assertIn("system", worker["capabilities"]["platform"])
    
    def test_03_adding_and_distributing_tasks(self):
        """Test adding tasks and distributing them to workers."""
        # Add test tasks
        task_ids = []
        for i in range(4):  # Add 4 tasks
            task_id = self.coordinator.add_task(
                "command",
                {"command": ["echo", f"Hello from task {i}"]},
                {"hardware": ["cpu"]},
                priority=i+1
            )
            task_ids.append(task_id)
            print(f"Added task {task_id} with priority {i+1}")
        
        # Verify tasks are added
        for task_id in task_ids:
            task_status = self.task_scheduler.get_task_status(task_id)
            self.assertIsNotNone(task_status, f"Task {task_id} not found")
        
        # Wait for tasks to be processed
        time.sleep(10)
        
        # Check task statuses (should all be completed or running)
        completed_tasks = 0
        for task_id in task_ids:
            task_status = self.task_scheduler.get_task_status(task_id)
            self.assertIsNotNone(task_status, f"Task {task_id} not found")
            print(f"Task {task_id} status: {task_status.get('status', 'unknown')}")
            if task_status.get("status") == "completed":
                completed_tasks += 1
        
        # At least some tasks should be completed
        self.assertGreater(completed_tasks, 0, "No tasks were completed")
    
    def test_04_task_prioritization(self):
        """Test that tasks are executed according to priority."""
        # Add test tasks with different priorities
        high_priority_task = self.coordinator.add_task(
            "command",
            {"command": ["sleep", "1"], "name": "high_priority"},
            {"hardware": ["cpu"]},
            priority=1  # Highest priority
        )
        
        low_priority_task = self.coordinator.add_task(
            "command",
            {"command": ["sleep", "1"], "name": "low_priority"},
            {"hardware": ["cpu"]},
            priority=10  # Lower priority
        )
        
        medium_priority_task = self.coordinator.add_task(
            "command",
            {"command": ["sleep", "1"], "name": "medium_priority"},
            {"hardware": ["cpu"]},
            priority=5  # Medium priority
        )
        
        # Get the task queue
        with self.task_scheduler.task_lock:
            queue = self.task_scheduler.task_queue.copy()
        
        # Check priority order
        priorities = [item[0] for item in queue]
        
        # The priorities should be in ascending order (lower number = higher priority)
        self.assertEqual(sorted(priorities), priorities, 
                         "Tasks are not correctly ordered by priority")
    
    def test_05_task_requirements_matching(self):
        """Test that tasks are assigned based on worker capabilities."""
        # Add a task requiring CPU
        cpu_task = self.coordinator.add_task(
            "command",
            {"command": ["echo", "CPU task"]},
            {"hardware": ["cpu"]},
            priority=1
        )
        
        # Add a task requiring GPU (which our test workers don't have)
        gpu_task = self.coordinator.add_task(
            "command",
            {"command": ["echo", "GPU task"]},
            {"hardware": ["cuda"]},
            priority=1
        )
        
        # Wait for processing
        time.sleep(5)
        
        # Check statuses
        cpu_status = self.task_scheduler.get_task_status(cpu_task)
        gpu_status = self.task_scheduler.get_task_status(gpu_task)
        
        # CPU task should be assigned or completed
        self.assertIn(cpu_status.get("status"), 
                     ["running", "completed", "assigned"],
                     "CPU task was not processed")
        
        # GPU task should still be queued (no worker has GPU)
        self.assertEqual(gpu_status.get("status"), "queued", 
                        "GPU task should remain queued as no worker has GPU capability")
    
    def test_06_task_result_reporting(self):
        """Test that task results are correctly reported and stored."""
        # Add a benchmark task
        benchmark_task = self.coordinator.add_task(
            "benchmark",
            {
                "model": "bert-base-uncased",
                "batch_sizes": [1, 2],
                "precision": "fp16",
                "iterations": 2
            },
            {"hardware": ["cpu"]},
            priority=1
        )
        
        # Wait for task to complete
        time.sleep(10)
        
        # Check if task is completed
        task_status = self.task_scheduler.get_task_status(benchmark_task)
        self.assertEqual(task_status.get("status"), "completed", 
                        f"Benchmark task status is {task_status.get('status')}, expected 'completed'")
        
        # Query results from database
        if self.db_manager:
            result = self.db_manager.conn.execute("""
            SELECT * FROM task_results WHERE task_id = ?
            """, [benchmark_task]).fetchone()
            
            self.assertIsNotNone(result, "Task result not found in database")
            
            # Verify result structure
            results = json.loads(result[4])  # results column
            self.assertIn("model", results)
            self.assertEqual(results["model"], "bert-base-uncased")
            self.assertIn("batch_sizes", results)
    
    def test_07_worker_heartbeat_and_status_updates(self):
        """Test worker heartbeat mechanism and status updates."""
        # Get initial worker statuses
        workers = self.worker_manager.get_available_workers()
        self.assertGreaterEqual(len(workers), 1, "No available workers found")
        
        # Check last heartbeat times
        for worker in workers:
            self.assertIsNotNone(worker["last_heartbeat"])
            
            # Last heartbeat should be recent
            last_heartbeat = worker["last_heartbeat"]
            if isinstance(last_heartbeat, str):
                last_heartbeat = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                
            self.assertLess((datetime.now() - last_heartbeat).total_seconds(), 10,
                           "Worker heartbeat is too old")
            
            # Status should be active or busy
            self.assertIn(worker["status"], ["active", "busy", "registered"])
    
    def test_08_dashboard_connectivity(self):
        """Test dashboard connection to coordinator."""
        # Verify dashboard is accessible
        try:
            response = requests.get(f"{self.dashboard_url}/api/status")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
        except requests.RequestException as e:
            self.fail(f"Dashboard API request failed: {e}")
        
        # Verify dashboard can access coordinator data
        try:
            response = requests.get(f"{self.dashboard_url}/api/dashboard")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Check for worker data
            self.assertIn("workers", data)
            self.assertGreaterEqual(data["workers"]["total"], 2)
            
            # Check for task data
            self.assertIn("tasks", data)
        except requests.RequestException as e:
            self.fail(f"Dashboard API request failed: {e}")
    
    def test_09_worker_failure_handling(self):
        """Test handling of worker failure and recovery."""
        # Get initial active workers
        initial_workers = self.worker_manager.get_available_workers()
        self.assertGreaterEqual(len(initial_workers), 2, "Need at least 2 workers for this test")
        
        # Choose a worker to "fail"
        worker_to_fail = initial_workers[0]["worker_id"]
        print(f"Testing failure recovery for worker {worker_to_fail}")
        
        # Stop the worker process
        for i, process in enumerate(self.worker_processes):
            if i == 0:  # Fail the first worker
                print(f"Stopping worker process {process.pid}")
                process.terminate()
                process.wait()
                self.worker_processes.pop(i)
                break
        
        # Wait for health monitoring to detect failure
        time.sleep(10)
        
        # Check that the worker is marked as unavailable
        worker = self.worker_manager.get_worker(worker_to_fail)
        self.assertIn(worker["status"], ["unavailable", "disconnected"])
        
        # Add tasks that were meant for the failed worker
        for i in range(3):
            self.coordinator.add_task(
                "command",
                {"command": ["echo", f"Recovery task {i}"]},
                {"hardware": ["cpu"]},
                priority=1
            )
        
        # Start a new worker
        self._start_workers(1)
        
        # Allow time for the new worker to connect and register
        time.sleep(5)
        
        # Verify we still have workers available
        available_workers = self.worker_manager.get_available_workers()
        self.assertGreaterEqual(len(available_workers), 1, 
                              "No available workers after recovery")
        
        # Verify tasks are being processed
        time.sleep(5)  # Give time for task processing
    
    def test_10_concurrent_task_processing(self):
        """Test concurrent processing of multiple tasks."""
        # Add several tasks simultaneously
        task_ids = []
        for i in range(6):  # Add 6 tasks
            task_id = self.coordinator.add_task(
                "command",
                {"command": ["sleep", "2"], "name": f"concurrent_task_{i}"},
                {"hardware": ["cpu"]},
                priority=1
            )
            task_ids.append(task_id)
        
        # Wait a bit for task assignment
        time.sleep(3)
        
        # Check if tasks are running concurrently
        running_count = 0
        assigned_count = 0
        
        for task_id in task_ids:
            status = self.task_scheduler.get_task_status(task_id)
            if status.get("status") == "running":
                running_count += 1
            elif status.get("status") == "assigned":
                assigned_count += 1
        
        # We should have at least 1 running task per worker
        available_workers = len(self.worker_manager.get_available_workers())
        self.assertGreaterEqual(running_count + assigned_count, min(2, available_workers), 
                              "Tasks are not being processed concurrently")
        
        # Wait for all tasks to complete
        time.sleep(10)
        
        # Verify all tasks completed
        completed_count = 0
        for task_id in task_ids:
            status = self.task_scheduler.get_task_status(task_id)
            if status.get("status") == "completed":
                completed_count += 1
        
        self.assertGreaterEqual(completed_count, len(task_ids) - 1,
                              f"Only {completed_count} of {len(task_ids)} tasks completed")

if __name__ == "__main__":
    unittest.main()