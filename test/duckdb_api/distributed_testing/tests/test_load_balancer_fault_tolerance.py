#!/usr/bin/env python3
"""
Fault Tolerance Integration Test for Load Balancer

This module tests the fault tolerance capabilities of the Load Balancer component
in the Distributed Testing Framework. It verifies that the system can handle
worker failures, recover from them, and continue operating effectively.
"""

import os
import sys
import json
import time
import signal
import unittest
import tempfile
import threading
import subprocess
import anyio
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from duckdb_api.distributed_testing.coordinator import CoordinatorServer
from duckdb_api.distributed_testing.coordinator_load_balancer_integration import CoordinatorLoadBalancerIntegration
from duckdb_api.distributed_testing.run_test import generate_security_config


class LoadBalancerFaultToleranceTest(unittest.TestCase):
    """
    Integration Tests for Load Balancer Fault Tolerance.
    
    This test suite validates that the load balancer can handle worker failures,
    task migration, and system recovery scenarios.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with coordinator, load balancer, and workers."""
        print("\nSetting up fault tolerance test environment...")
        
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
        cls.coordinator_port = 8889  # Use non-standard port for testing
        cls.coordinator_url = f"ws://{cls.coordinator_host}:{cls.coordinator_port}"
        
        # Start the coordinator with load balancer in a separate thread
        cls.coordinator_started = anyio.Event()
        cls.coordinator_stopped = anyio.Event()
        cls.coordinator_thread = threading.Thread(
            target=cls._run_coordinator_with_load_balancer,
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
        
        print(f"Coordinator with load balancer started at {cls.coordinator_url}")
        
        # Start mock workers with different capabilities
        cls.worker_processes = []
        cls._start_workers(6)  # Start 6 workers with different capabilities
        
        # Wait for workers to connect and register
        time.sleep(5)
        print("Setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        print("\nCleaning up fault tolerance test environment...")
        
        # Stop all worker processes
        for worker_process in cls.worker_processes:
            print(f"Terminating worker process {worker_process['process'].pid}")
            worker_process["process"].terminate()
            worker_process["process"].wait()
        
        # Stop the coordinator
        print("Stopping coordinator with load balancer")
        cls.coordinator_stopped.set()
        cls.coordinator_thread.join(timeout=5.0)
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
            print(f"Removed temporary directory: {cls.temp_dir}")
    
    @classmethod
    def _run_coordinator_with_load_balancer(cls):
        """Run the coordinator with load balancer in a separate thread."""
        try:
            # Setup the event loop
            loop = # TODO: Remove event loop management - asyncio.new_event_loop()
            # TODO: Remove event loop management - asyncio.set_event_loop(loop)
            
            # Import and apply patches
            from duckdb_api.distributed_testing.coordinator_patch import apply_patches, remove_patches
            apply_patches()
            
            # Load balancer configuration with fault tolerance enabled
            load_balancer_config = {
                "db_path": cls.db_path,
                "monitoring_interval": 2,  # Short interval for testing
                "rebalance_interval": 5,   # Short interval for testing
                "worker_concurrency": 2,
                "enable_work_stealing": True,
                "fault_tolerance": {
                    "enabled": True,
                    "max_recovery_attempts": 3,
                    "recovery_timeout": 10,
                    "recovery_strategies": ["immediate", "progressive", "coordinated"],
                    "task_prioritization": True
                },
                "default_scheduler": {
                    "type": "performance_based"
                }
            }
            
            # Create coordinator with load balancer
            cls.coordinator = CoordinatorServer(
                host=cls.coordinator_host,
                port=cls.coordinator_port,
                db_path=cls.db_path,
                token_secret=cls.security_config["token_secret"],
                heartbeat_timeout=5,  # Short timeout for testing
                enable_load_balancer=True,
                load_balancer_config=load_balancer_config
            )
            
            # Create helper function to know when coordinator is ready
            async def on_coordinator_start():
                cls.coordinator_started.set()
                await cls.coordinator_stopped.wait()
                # Stop coordinator
                await cls.coordinator.stop()
            
            # Run both tasks: the coordinator and the signal handler
            loop.create_task(cls.coordinator.start())
            loop.run_until_complete(on_coordinator_start())
            
            # Cleanup
            loop.close()
            remove_patches()
        except Exception as e:
            print(f"Error in coordinator thread: {e}")
            import traceback
            traceback.print_exc()
    
    @classmethod
    def _start_workers(cls, count=6):
        """Start multiple worker processes with different capabilities."""
        # Default capabilities for different worker types
        default_capabilities = [
            # Basic CPU worker
            {
                "hardware_types": ["cpu"],
                "memory_gb": 4
            },
            # CPU + GPU worker
            {
                "hardware_types": ["cpu", "cuda"],
                "cuda_compute": 7.5,
                "memory_gb": 16
            },
            # CPU + WebGPU worker
            {
                "hardware_types": ["cpu", "webgpu"],
                "browsers": ["chrome", "firefox"],
                "memory_gb": 8
            },
            # CPU + TPU worker
            {
                "hardware_types": ["cpu", "tpu"],
                "tpu_version": "v3",
                "memory_gb": 32
            },
            # Low-powered CPU worker
            {
                "hardware_types": ["cpu"],
                "cpu_cores": 2,
                "memory_gb": 2
            },
            # High-powered CPU worker
            {
                "hardware_types": ["cpu"],
                "cpu_cores": 64,
                "memory_gb": 512
            }
        ]
        
        for i in range(count):
            worker_id = f"test_worker_{i}"
            worker_dir = os.path.join(cls.temp_dir, f"worker_{i}")
            os.makedirs(worker_dir, exist_ok=True)
            
            # Select capability set (cycling through the available options)
            capability_index = i % len(default_capabilities)
            capabilities_json = json.dumps(default_capabilities[capability_index])
            
            worker_cmd = [
                sys.executable,
                os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
                "--coordinator", cls.coordinator_url,
                "--api-key", cls.worker_api_key,
                "--worker-id", worker_id,
                "--work-dir", worker_dir,
                "--reconnect-interval", "2",
                "--heartbeat-interval", "3",
                "--capabilities", capabilities_json
            ]
            
            print(f"Starting worker {worker_id} with capabilities: {default_capabilities[capability_index]}")
            process = subprocess.Popen(worker_cmd)
            cls.worker_processes.append({
                "process": process,
                "worker_id": worker_id,
                "work_dir": worker_dir,
                "capabilities": default_capabilities[capability_index]
            })
    
    def _add_test_tasks(self, count=10):
        """Add test tasks with various requirements."""
        task_types = [
            {
                "type": "command",
                "config": {"command": ["sleep", "5"]},
                "requirements": {"hardware": ["cpu"]},
                "priority": 1
            },
            {
                "type": "command",
                "config": {"command": ["echo", "GPU task"]},
                "requirements": {"hardware": ["cuda"]},
                "priority": 2
            },
            {
                "type": "command",
                "config": {"command": ["echo", "WebGPU task"]},
                "requirements": {"hardware": ["webgpu"]},
                "priority": 3
            },
            {
                "type": "benchmark",
                "config": {
                    "model": "bert-base-uncased",
                    "batch_sizes": [1, 2, 4],
                    "precision": "fp16",
                    "iterations": 3
                },
                "requirements": {"hardware": ["cpu"], "min_memory_gb": 8},
                "priority": 1
            },
            {
                "type": "benchmark",
                "config": {
                    "model": "t5-small",
                    "batch_sizes": [1],
                    "precision": "fp32",
                    "iterations": 2
                },
                "requirements": {"hardware": ["cpu"]},
                "priority": 4
            }
        ]
        
        task_ids = []
        for i in range(count):
            task_type = task_types[i % len(task_types)]
            task_id = self.coordinator.add_task(
                task_type["type"],
                task_type["config"],
                task_type["requirements"],
                task_type["priority"]
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def _get_running_tasks_by_worker(self):
        """Get a mapping of worker_id to list of running task_ids."""
        result = {}
        
        # Query tasks from the database
        tasks = self.coordinator.db_manager.conn.execute("""
        SELECT task_id, worker_id FROM distributed_tasks 
        WHERE status = 'running' OR status = 'assigned'
        """).fetchall()
        
        # Group by worker
        for task_id, worker_id in tasks:
            if worker_id not in result:
                result[worker_id] = []
            result[worker_id].append(task_id)
        
        return result
    
    def _kill_worker(self, worker_index):
        """Kill a worker process and return its worker_id."""
        if worker_index >= len(self.worker_processes):
            raise ValueError(f"Worker index {worker_index} out of range (0-{len(self.worker_processes)-1})")
        
        worker = self.worker_processes[worker_index]
        worker_id = worker["worker_id"]
        process = worker["process"]
        
        print(f"Killing worker {worker_id} (PID: {process.pid})")
        process.terminate()
        process.wait()
        
        # Remove from list
        self.worker_processes.pop(worker_index)
        
        return worker_id
    
    def _restart_worker(self, worker_id, capabilities=None):
        """Restart a worker with the given worker_id and optional capabilities."""
        # Find worker info from the killed worker
        worker_index = None
        for i, worker in enumerate(self.worker_processes):
            if worker["worker_id"] == worker_id:
                worker_index = i
                break
        
        # Create worker directory
        worker_dir = os.path.join(self.temp_dir, f"worker_{worker_id}")
        os.makedirs(worker_dir, exist_ok=True)
        
        # Use provided capabilities or default CPU capabilities
        if capabilities is None:
            capabilities = {
                "hardware_types": ["cpu"],
                "memory_gb": 4
            }
        
        capabilities_json = json.dumps(capabilities)
        
        worker_cmd = [
            sys.executable,
            os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
            "--coordinator", self.coordinator_url,
            "--api-key", self.worker_api_key,
            "--worker-id", worker_id,
            "--work-dir", worker_dir,
            "--reconnect-interval", "2",
            "--heartbeat-interval", "3",
            "--capabilities", capabilities_json
        ]
        
        print(f"Restarting worker {worker_id} with capabilities: {capabilities}")
        process = subprocess.Popen(worker_cmd)
        
        # Add to worker processes list
        self.worker_processes.append({
            "process": process,
            "worker_id": worker_id,
            "work_dir": worker_dir,
            "capabilities": capabilities
        })
        
        return worker_id
    
    def test_01_basic_initialization(self):
        """Test that fault tolerance is properly configured."""
        # Verify that load balancer is enabled
        self.assertTrue(hasattr(self.coordinator, 'load_balancer'))
        self.assertIsNotNone(self.coordinator.load_balancer)
        
        # Verify that fault tolerance is enabled
        load_balancer = self.coordinator.load_balancer
        self.assertTrue(hasattr(load_balancer, 'config'))
        self.assertIn('fault_tolerance', load_balancer.config)
        self.assertTrue(load_balancer.config['fault_tolerance'].get('enabled', False))
    
    def test_02_worker_registration(self):
        """Test that workers are registered with the coordinator."""
        # Get worker information from coordinator
        workers = self.coordinator.worker_manager.get_workers()
        
        # Verify we have all our test workers
        self.assertEqual(len(workers), 6, "Expected 6 test workers to be registered")
        
        # Verify worker capabilities are correctly registered
        for worker in workers:
            self.assertIn('capabilities', worker)
            self.assertIn('hardware_types', worker['capabilities'])
    
    def test_03_task_assignment_before_failure(self):
        """Test that tasks are properly assigned to workers before any failures."""
        # Add test tasks
        task_ids = self._add_test_tasks(count=12)
        
        # Wait for tasks to be assigned and start running
        time.sleep(5)
        
        # Get task assignments
        task_assignments = {}
        for task_id in task_ids:
            task = self.coordinator.task_manager.get_task(task_id)
            if 'worker_id' in task:
                task_assignments[task_id] = task['worker_id']
        
        # Verify that tasks are assigned
        self.assertGreater(len(task_assignments), 0, "No tasks were assigned to workers")
        
        # Verify that GPU tasks are assigned to GPU workers
        for task_id, worker_id in task_assignments.items():
            task = self.coordinator.task_manager.get_task(task_id)
            
            if 'requirements' in task and 'hardware' in task['requirements']:
                if 'cuda' in task['requirements']['hardware']:
                    # Get worker capabilities
                    worker = self.coordinator.worker_manager.get_worker(worker_id)
                    self.assertIn('cuda', worker['capabilities']['hardware_types'], 
                                 f"GPU task {task_id} assigned to non-GPU worker {worker_id}")
    
    def test_04_worker_failure_detection(self):
        """Test that worker failures are detected."""
        # Kill a CPU worker
        worker_index = 0  # CPU worker
        worker_id = self._kill_worker(worker_index)
        
        # Wait for health monitor to detect failure
        time.sleep(10)
        
        # Verify worker is marked as unavailable
        worker = self.coordinator.worker_manager.get_worker(worker_id)
        self.assertIsNotNone(worker, f"Worker {worker_id} not found")
        self.assertIn(worker['status'], ['unavailable', 'disconnected'], 
                     f"Worker {worker_id} should be marked as unavailable or disconnected")
    
    def test_05_task_reassignment_after_failure(self):
        """Test that tasks are reassigned after worker failure."""
        # Add new tasks
        task_ids = self._add_test_tasks(count=6)
        
        # Wait for tasks to be initially assigned
        time.sleep(5)
        
        # Get initial task assignments
        initial_assignments = {}
        for task_id in task_ids:
            task = self.coordinator.task_manager.get_task(task_id)
            if 'worker_id' in task:
                initial_assignments[task_id] = task['worker_id']
        
        # Find a worker with at least one running task
        workers_with_tasks = self._get_running_tasks_by_worker()
        worker_to_kill = None
        for worker_id, tasks in workers_with_tasks.items():
            if len(tasks) > 0:
                # Find the worker in our process list
                for i, worker in enumerate(self.worker_processes):
                    if worker["worker_id"] == worker_id:
                        worker_to_kill = (i, worker_id, tasks)
                        break
                if worker_to_kill:
                    break
        
        # Skip test if no suitable worker found
        if not worker_to_kill:
            self.skipTest("No worker with running tasks found")
            return
        
        # Get the tasks that were assigned to this worker
        worker_index, worker_id, affected_tasks = worker_to_kill
        print(f"Worker {worker_id} has {len(affected_tasks)} tasks: {affected_tasks}")
        
        # Kill the worker
        self._kill_worker(worker_index)
        
        # Wait for failure detection and task reassignment
        time.sleep(15)
        
        # Check if affected tasks were reassigned or marked for retry
        for task_id in affected_tasks:
            task = self.coordinator.task_manager.get_task(task_id)
            
            # Task should either be reassigned to a new worker or marked for retry
            if task['status'] in ['assigned', 'running']:
                # Task was reassigned
                self.assertIn('worker_id', task)
                self.assertNotEqual(task['worker_id'], worker_id, 
                                  f"Task {task_id} should not still be assigned to failed worker {worker_id}")
            elif task['status'] in ['queued', 'pending']:
                # Task was marked for retry
                pass
            else:
                # Task could have completed before worker failure
                self.assertIn(task['status'], ['completed', 'failed'], 
                             f"Task {task_id} has unexpected status {task['status']}")
    
    def test_06_worker_recovery(self):
        """Test worker recovery and task continuity."""
        # Add new tasks
        task_ids = self._add_test_tasks(count=8)
        
        # Wait for tasks to be assigned
        time.sleep(5)
        
        # Get task assignments
        workers_with_tasks = self._get_running_tasks_by_worker()
        
        # Find a worker with at least 2 running tasks
        worker_to_kill = None
        for worker_id, tasks in workers_with_tasks.items():
            if len(tasks) >= 2:
                # Find the worker in our process list
                for i, worker in enumerate(self.worker_processes):
                    if worker["worker_id"] == worker_id:
                        worker_to_kill = (i, worker_id, tasks)
                        break
                if worker_to_kill:
                    break
        
        # Skip test if no suitable worker found
        if not worker_to_kill:
            self.skipTest("No worker with multiple running tasks found")
            return
        
        # Get the tasks that were assigned to this worker
        worker_index, worker_id, affected_tasks = worker_to_kill
        capabilities = next((w["capabilities"] for w in self.worker_processes if w["worker_id"] == worker_id), None)
        print(f"Worker {worker_id} has {len(affected_tasks)} tasks: {affected_tasks}")
        
        # Kill the worker
        self._kill_worker(worker_index)
        
        # Wait for failure detection
        time.sleep(8)
        
        # Restart the worker with same worker_id and capabilities
        self._restart_worker(worker_id, capabilities)
        
        # Wait for worker to reconnect and task reassignment
        time.sleep(10)
        
        # Verify that the worker is back and active
        worker = self.coordinator.worker_manager.get_worker(worker_id)
        self.assertEqual(worker['status'], 'active', f"Worker {worker_id} should be active after reconnection")
        
        # Check task status for previously affected tasks
        completed_or_running = 0
        for task_id in affected_tasks:
            task = self.coordinator.task_manager.get_task(task_id)
            status = task.get('status')
            worker_assignment = task.get('worker_id')
            
            print(f"Task {task_id} status: {status}, assigned to: {worker_assignment}")
            
            # Task should either be reassigned, running, or completed
            if status in ['running', 'completed', 'assigned']:
                completed_or_running += 1
        
        # At least some tasks should be running or completed
        self.assertGreater(completed_or_running, 0, "No affected tasks are running or completed after worker recovery")
    
    def test_07_simultaneous_worker_failures(self):
        """Test handling of multiple simultaneous worker failures."""
        # Add a larger batch of tasks
        task_ids = self._add_test_tasks(count=15)
        
        # Wait for tasks to be assigned
        time.sleep(5)
        
        # Get running workers
        available_workers = [i for i, w in enumerate(self.worker_processes)]
        
        # Choose 2 workers to kill simultaneously
        if len(available_workers) < 2:
            self.skipTest("Need at least 2 workers for this test")
            return
        
        # Kill 2 random workers
        to_kill = random.sample(available_workers, 2)
        killed_workers = []
        
        # Kill in reverse order to avoid index issues
        for index in sorted(to_kill, reverse=True):
            killed_workers.append(self._kill_worker(index))
        
        print(f"Killed workers: {killed_workers}")
        
        # Wait for failure detection and recovery
        time.sleep(15)
        
        # Verify task distribution to remaining workers
        remaining_workers = self.coordinator.worker_manager.get_available_workers()
        self.assertGreaterEqual(len(remaining_workers), len(self.worker_processes), 
                              "Should have at least as many available workers as worker processes")
        
        # Check that tasks are being processed
        running_or_completed = 0
        for task_id in task_ids:
            task = self.coordinator.task_manager.get_task(task_id)
            if task.get('status') in ['running', 'completed', 'assigned']:
                running_or_completed += 1
                
                # If assigned or running, verify it's on a non-killed worker
                if task.get('status') in ['running', 'assigned'] and 'worker_id' in task:
                    self.assertNotIn(task['worker_id'], killed_workers, 
                                   f"Task {task_id} should not be assigned to a killed worker")
        
        # At least some tasks should be running or completed
        self.assertGreater(running_or_completed, 0, "No tasks are running or completed after multiple worker failures")
    
    def test_08_work_stealing_after_recovery(self):
        """Test that work stealing redistributes load after worker recovery."""
        # Ensure we have at least 4 workers
        if len(self.worker_processes) < 4:
            # Start additional workers if needed
            additional_needed = 4 - len(self.worker_processes)
            self._start_workers(additional_needed)
            time.sleep(5)  # Wait for workers to register
        
        # Add a large number of CPU tasks
        cpu_intensive_tasks = []
        for i in range(12):
            task_id = self.coordinator.add_task(
                "command",
                {"command": ["sleep", "10"]},  # Long-running CPU task
                {"hardware": ["cpu"]},
                priority=1
            )
            cpu_intensive_tasks.append(task_id)
        
        # Wait for tasks to be assigned
        time.sleep(5)
        
        # Get initial task distribution
        initial_distribution = self._get_running_tasks_by_worker()
        print(f"Initial task distribution: {initial_distribution}")
        
        # Find the worker with the most tasks
        most_loaded_worker = max(initial_distribution.items(), key=lambda x: len(x[1]), default=(None, []))
        
        if most_loaded_worker[0] is None or len(most_loaded_worker[1]) < 2:
            self.skipTest("No sufficiently loaded worker found for this test")
            return
        
        # Add a new CPU worker with high capacity
        new_worker_id = f"test_worker_highcap_{int(time.time())}"
        high_capacity_capabilities = {
            "hardware_types": ["cpu"],
            "cpu_cores": 128,
            "memory_gb": 1024,
            "priority_score": 100  # High priority for work stealing
        }
        
        # Start the new worker
        worker_dir = os.path.join(self.temp_dir, f"worker_{new_worker_id}")
        os.makedirs(worker_dir, exist_ok=True)
        
        worker_cmd = [
            sys.executable,
            os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
            "--coordinator", self.coordinator_url,
            "--api-key", self.worker_api_key,
            "--worker-id", new_worker_id,
            "--work-dir", worker_dir,
            "--reconnect-interval", "2",
            "--heartbeat-interval", "3",
            "--capabilities", json.dumps(high_capacity_capabilities)
        ]
        
        print(f"Starting high-capacity worker {new_worker_id} with capabilities: {high_capacity_capabilities}")
        process = subprocess.Popen(worker_cmd)
        self.worker_processes.append({
            "process": process,
            "worker_id": new_worker_id,
            "work_dir": worker_dir,
            "capabilities": high_capacity_capabilities
        })
        
        # Wait for worker to connect and work stealing to occur
        time.sleep(20)
        
        # Get new task distribution
        new_distribution = self._get_running_tasks_by_worker()
        print(f"New task distribution: {new_distribution}")
        
        # Verify work stealing occurred
        self.assertIn(new_worker_id, new_distribution, 
                     "New high-capacity worker should have stolen some tasks")
        self.assertGreaterEqual(len(new_distribution.get(new_worker_id, [])), 1, 
                              "New worker should have at least 1 task")
    
    def test_09_fault_tolerant_task_completion(self):
        """Test that tasks complete successfully despite worker failures."""
        # Add a batch of quick tasks
        quick_tasks = []
        for i in range(10):
            task_id = self.coordinator.add_task(
                "command",
                {"command": ["echo", f"Quick task {i}"]},
                {"hardware": ["cpu"]},
                priority=1
            )
            quick_tasks.append(task_id)
        
        # Wait for tasks to start processing
        time.sleep(3)
        
        # Kill a random worker
        if not self.worker_processes:
            self.skipTest("No workers available for this test")
            return
        
        worker_index = random.randint(0, len(self.worker_processes) - 1)
        killed_worker_id = self._kill_worker(worker_index)
        
        # Wait for tasks to complete despite the failure
        time.sleep(15)
        
        # Verify all tasks completed
        completed_count = 0
        for task_id in quick_tasks:
            task = self.coordinator.task_manager.get_task(task_id)
            if task.get('status') == 'completed':
                completed_count += 1
        
        # Most quick tasks should complete despite worker failure
        self.assertGreaterEqual(completed_count, 7, 
                              f"Only {completed_count} out of 10 quick tasks completed")
    
    def test_10_resiliency_to_repeated_failures(self):
        """Test system resiliency to repeated worker failures."""
        # Start with a clean set of workers if needed
        if len(self.worker_processes) < 3:
            self._start_workers(3 - len(self.worker_processes))
            time.sleep(5)  # Wait for registration
        
        # Add long-running tasks
        long_tasks = []
        for i in range(5):
            task_id = self.coordinator.add_task(
                "command",
                {"command": ["sleep", "15"]},  # Long-running tasks
                {"hardware": ["cpu"]},
                priority=1
            )
            long_tasks.append(task_id)
        
        # Wait for tasks to be assigned
        time.sleep(3)
        
        # Perform repeated kill/restart cycles
        for i in range(3):
            if not self.worker_processes:
                break
                
            # Kill a random worker
            worker_index = random.randint(0, len(self.worker_processes) - 1)
            worker = self.worker_processes[worker_index]
            worker_id = worker["worker_id"]
            capabilities = worker["capabilities"]
            
            self._kill_worker(worker_index)
            print(f"Cycle {i+1}: Killed worker {worker_id}")
            
            # Wait briefly
            time.sleep(3)
            
            # Restart the worker
            self._restart_worker(worker_id, capabilities)
            print(f"Cycle {i+1}: Restarted worker {worker_id}")
            
            # Wait before next cycle
            time.sleep(4)
        
        # Wait for tasks to complete
        time.sleep(20)
        
        # Verify task completion
        completed_count = 0
        for task_id in long_tasks:
            task = self.coordinator.task_manager.get_task(task_id)
            if task.get('status') == 'completed':
                completed_count += 1
        
        # At least some tasks should complete despite repeated failures
        self.assertGreater(completed_count, 0, 
                          f"No tasks completed after repeated worker failures")


if __name__ == "__main__":
    unittest.main()