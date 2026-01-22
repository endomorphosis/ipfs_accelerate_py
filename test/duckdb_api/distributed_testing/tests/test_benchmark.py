#!/usr/bin/env python3
"""
Benchmark tests for the distributed testing framework.

This script provides performance benchmarks for key components of the
distributed testing framework, measuring response times, throughput,
and scalability under various loads.
"""

import os
import sys
import time
import json
import random
import unittest
import tempfile
import threading
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from duckdb_api.distributed_testing.coordinator import (
    DatabaseManager, CoordinatorServer, SecurityManager
)
from duckdb_api.distributed_testing.task_scheduler import TaskScheduler
from duckdb_api.distributed_testing.load_balancer import LoadBalancer
from duckdb_api.distributed_testing.health_monitor import HealthMonitor


class BenchmarkTimer:
    """Simple class for timing benchmark operations."""
    
    def __init__(self, name: str):
        """
        Initialize the timer with a name.
        
        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.name}: {self.duration:.6f} seconds")
        return False  # Don't suppress exceptions


class DistributedFrameworkBenchmark(unittest.TestCase):
    """
    Performance benchmarks for the distributed testing framework.
    
    Tests the performance of key components under various loads.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with database and components."""
        # Create a temporary database file
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix=".duckdb")
        
        # Create database manager
        cls.db_manager = DatabaseManager(cls.db_path)
        
        # Create task scheduler
        cls.task_scheduler = TaskScheduler(cls.db_manager)
        
        # Create load balancer
        cls.load_balancer = LoadBalancer(cls.db_manager)
        
        # Create health monitor
        cls.health_monitor = HealthMonitor(cls.db_manager)
        
        # Create security manager
        cls.security_manager = SecurityManager(cls.db_manager)
        
        # Create coordinator (but don't start it)
        cls.coordinator = CoordinatorServer(
            host="localhost",
            port=8099,  # Use a port unlikely to be in use
            db_path=cls.db_path,
            heartbeat_timeout=5
        )
        
        # Initialize test data
        cls._init_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Close database connection
        if hasattr(cls, 'db_manager') and cls.db_manager:
            cls.db_manager.close()
        
        # Remove temporary database
        os.close(cls.db_fd)
        os.unlink(cls.db_path)
    
    @classmethod
    def _init_test_data(cls):
        """Initialize test data for benchmarks."""
        # Create worker nodes with different capabilities
        worker_types = [
            {"type": "cpu", "hardware_types": ["cpu"], "memory_gb": 8},
            {"type": "cuda", "hardware_types": ["cpu", "cuda"], "memory_gb": 16, "cuda_compute": 7.5},
            {"type": "rocm", "hardware_types": ["cpu", "rocm"], "memory_gb": 16},
            {"type": "webgpu", "hardware_types": ["cpu", "webgpu"], "browsers": ["chrome", "firefox"], "memory_gb": 8}
        ]
        
        # Number of workers of each type
        cls.num_workers_per_type = 10
        cls.total_workers = cls.num_workers_per_type * len(worker_types)
        
        # Create workers
        for i in range(cls.total_workers):
            worker_type = worker_types[i % len(worker_types)]
            worker_id = f"bench_worker_{i}"
            hostname = f"bench_host_{i}"
            
            # Add some variation to capabilities
            capabilities = worker_type.copy()
            capabilities["memory_gb"] += random.randint(-2, 2)
            
            # Add worker to database
            cls.db_manager.add_worker(worker_id, hostname, capabilities)
    
    def _generate_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """
        Generate a list of test tasks.
        
        Args:
            num_tasks: Number of tasks to generate
            
        Returns:
            List of task dictionaries
        """
        task_types = ["benchmark", "test", "command"]
        hardware_reqs = [
            {"hardware": ["cpu"]},
            {"hardware": ["cuda"]},
            {"hardware": ["rocm"]},
            {"hardware": ["webgpu"], "browser": "chrome"},
            {"hardware": ["webgpu"], "browser": "firefox"},
            {"hardware": ["cpu"], "min_memory_gb": 12}
        ]
        
        tasks = []
        for i in range(num_tasks):
            task_id = f"bench_task_{i}"
            task_type = random.choice(task_types)
            priority = random.randint(1, 10)
            requirements = random.choice(hardware_reqs)
            
            task = {
                "task_id": task_id,
                "type": task_type,
                "priority": priority,
                "requirements": requirements
            }
            tasks.append(task)
        
        return tasks
    
    def test_00_benchmark_db_initialization(self):
        """Benchmark database initialization time."""
        with BenchmarkTimer("Database initialization"):
            db_fd, db_path = tempfile.mkstemp(suffix=".duckdb")
            db_manager = DatabaseManager(db_path)
            
        # Clean up
        db_manager.close()
        os.close(db_fd)
        os.unlink(db_path)
    
    def test_01_benchmark_worker_registration(self):
        """Benchmark worker registration performance."""
        # Prepare test workers
        num_workers = 100
        test_workers = []
        for i in range(num_workers):
            test_workers.append({
                "worker_id": f"perf_worker_{i}",
                "hostname": f"perf_host_{i}",
                "capabilities": {
                    "hardware_types": ["cpu"],
                    "memory_gb": 8 + (i % 8)
                }
            })
        
        # Measure time to register workers
        with BenchmarkTimer(f"Registering {num_workers} workers"):
            for worker in test_workers:
                self.db_manager.add_worker(
                    worker["worker_id"],
                    worker["hostname"],
                    worker["capabilities"]
                )
        
        # Time to retrieve workers
        with BenchmarkTimer(f"Retrieving {num_workers} workers"):
            for worker in test_workers:
                self.db_manager.get_worker(worker["worker_id"])
    
    def test_02_benchmark_task_creation(self):
        """Benchmark task creation performance."""
        # Number of tasks to create
        task_counts = [10, 100, 1000]
        
        for count in task_counts:
            tasks = self._generate_tasks(count)
            
            # Measure time to create tasks
            with BenchmarkTimer(f"Creating {count} tasks"):
                for task in tasks:
                    self.db_manager.add_task(
                        task["task_id"],
                        task["type"],
                        task["priority"],
                        {"benchmark": True},
                        task["requirements"]
                    )
    
    def test_03_benchmark_task_assignment(self):
        """Benchmark task assignment performance with different numbers of workers and tasks."""
        # Generate scenarios with different numbers of workers and tasks
        scenarios = [
            {"workers": 10, "tasks": 20},
            {"workers": 50, "tasks": 100},
            {"workers": 100, "tasks": 200}
        ]
        
        for scenario in scenarios:
            num_workers = scenario["workers"]
            num_tasks = scenario["tasks"]
            
            print(f"\nScenario: {num_workers} workers, {num_tasks} tasks")
            
            # Create temporary test database
            test_db_fd, test_db_path = tempfile.mkstemp(suffix=".duckdb")
            test_db = DatabaseManager(test_db_path)
            
            # Create scheduler
            scheduler = TaskScheduler(test_db)
            
            # Create test workers
            for i in range(num_workers):
                worker_type = i % 4  # 0=CPU, 1=CUDA, 2=ROCm, 3=WebGPU
                capabilities = {
                    "hardware_types": ["cpu"],
                    "memory_gb": 8
                }
                
                if worker_type == 1:
                    capabilities["hardware_types"].append("cuda")
                    capabilities["cuda_compute"] = 7.5
                elif worker_type == 2:
                    capabilities["hardware_types"].append("rocm")
                elif worker_type == 3:
                    capabilities["hardware_types"].append("webgpu")
                    capabilities["browsers"] = ["chrome", "firefox"]
                
                test_db.add_worker(f"test_worker_{i}", f"test_host_{i}", capabilities)
            
            # Create test tasks
            tasks = self._generate_tasks(num_tasks)
            for task in tasks:
                test_db.add_task(
                    task["task_id"],
                    task["type"],
                    task["priority"],
                    {"benchmark": True},
                    task["requirements"]
                )
            
            # Benchmark task assignment
            assignment_times = []
            with BenchmarkTimer(f"Assigning {num_tasks} tasks to {num_workers} workers"):
                for _ in range(10):  # Run multiple iterations for more stable timing
                    start = time.time()
                    pending_tasks = test_db.get_pending_tasks(limit=num_tasks)
                    for task in pending_tasks:
                        worker = scheduler.get_next_task(f"bench_worker_{random.randint(0, num_workers-1)}", 
                                                      {"hardware_types": ["cpu"]})
                    end = time.time()
                    assignment_times.append(end - start)
            
            # Report statistics
            avg_time = statistics.mean(assignment_times)
            tasks_per_second = num_tasks / avg_time if avg_time > 0 else 0
            print(f"Average assignment time: {avg_time:.6f} seconds")
            print(f"Tasks per second: {tasks_per_second:.2f}")
            
            # Clean up
            test_db.close()
            os.close(test_db_fd)
            os.unlink(test_db_path)
    
    def test_04_benchmark_load_balancing(self):
        """Benchmark load balancing performance with different workloads."""
        # Generate scenarios with different imbalance levels
        scenarios = [
            {"name": "Slight imbalance", "distribution": [2, 2, 2, 3, 3]},
            {"name": "Moderate imbalance", "distribution": [1, 2, 3, 5, 9]},
            {"name": "Severe imbalance", "distribution": [0, 1, 2, 3, 14]}
        ]
        
        for scenario in scenarios:
            name = scenario["name"]
            distribution = scenario["distribution"]
            print(f"\nScenario: {name}")
            
            # Create temporary test database
            test_db_fd, test_db_path = tempfile.mkstemp(suffix=".duckdb")
            test_db = DatabaseManager(test_db_path)
            
            # Create load balancer
            balancer = LoadBalancer(test_db)
            
            # Create 5 test workers
            for i in range(5):
                test_db.add_worker(f"lb_worker_{i}", f"lb_host_{i}", {"hardware_types": ["cpu"]})
            
            # Create tasks with the specified distribution
            for worker_idx, num_tasks in enumerate(distribution):
                for i in range(num_tasks):
                    task_id = f"lb_task_w{worker_idx}_t{i}"
                    test_db.add_task(task_id, "test", 1, {}, {"hardware": ["cpu"]})
                    
                    # Assign task to worker
                    test_db.conn.execute("""
                    UPDATE distributed_tasks 
                    SET worker_id = ?, status = 'running' 
                    WHERE task_id = ?
                    """, [f"lb_worker_{worker_idx}", task_id])
            
            # Benchmark load balancing operation
            with BenchmarkTimer(f"Load balancing - {name}"):
                # Get initial workload
                initial_load = balancer.get_worker_load()
                print(f"Initial workload: {initial_load}")
                
                # Detect imbalances
                overloaded = balancer.detect_overloaded_workers()
                underutilized = balancer.detect_underutilized_workers()
                print(f"Overloaded workers: {overloaded}")
                print(f"Underutilized workers: {underutilized}")
                
                # Perform rebalancing (with mock migration)
                migrations = []
                
                def mock_migrate(task_id, from_worker, to_worker):
                    migrations.append((task_id, from_worker, to_worker))
                    return True
                
                balancer.migrate_task = mock_migrate
                balancer.rebalance_tasks()
                
                print(f"Migrations: {len(migrations)}")
                if migrations:
                    print(f"First 3 migrations: {migrations[:3]}")
            
            # Clean up
            test_db.close()
            os.close(test_db_fd)
            os.unlink(test_db_path)
    
    def test_05_benchmark_health_monitoring(self):
        """Benchmark health monitoring performance with different numbers of workers."""
        # Worker counts to test
        worker_counts = [10, 50, 100, 200]
        
        for count in worker_counts:
            print(f"\nScenario: {count} workers")
            
            # Create temporary test database
            test_db_fd, test_db_path = tempfile.mkstemp(suffix=".duckdb")
            test_db = DatabaseManager(test_db_path)
            
            # Create workers with different states
            for i in range(count):
                worker_id = f"health_worker_{i}"
                test_db.add_worker(worker_id, f"health_host_{i}", {"hardware_types": ["cpu"]})
                
                # Set some workers' last heartbeat to be old
                if i % 5 == 0:  # 20% of workers will be "unhealthy"
                    # Set old heartbeat directly in database
                    old_time = datetime.now() - timedelta(minutes=10)
                    test_db.conn.execute("""
                    UPDATE worker_nodes 
                    SET last_heartbeat = ? 
                    WHERE worker_id = ?
                    """, [old_time, worker_id])
            
            # Create health monitor
            recovery_actions = []
            
            def mock_recovery(worker_id, alert_type, alert_data):
                recovery_actions.append((worker_id, alert_type))
                return True
            
            monitor = HealthMonitor(
                db_manager=test_db,
                recovery_action=mock_recovery
            )
            
            # Benchmark health check operation
            with BenchmarkTimer(f"Health check - {count} workers"):
                # Perform health check
                monitor.check_worker_health()
                
                # Get health status
                status = monitor.get_health_status()
                
                # Get alerts
                alerts = monitor.get_active_alerts()
                
                print(f"Unhealthy workers: {count - status['summary']['healthy_workers']}")
                print(f"Alerts generated: {len(alerts)}")
                
                # Check recovery actions
                print(f"Recovery actions: {len(recovery_actions)}")
            
            # Clean up
            test_db.close()
            os.close(test_db_fd)
            os.unlink(test_db_path)
    
    def test_06_benchmark_concurrent_operations(self):
        """Benchmark concurrent operations with multiple components running."""
        # Create temporary test database
        test_db_fd, test_db_path = tempfile.mkstemp(suffix=".duckdb")
        test_db = DatabaseManager(test_db_path)
        
        # Create basic test data
        num_workers = 50
        num_tasks = 100
        
        # Create workers
        for i in range(num_workers):
            worker_id = f"concurrent_worker_{i}"
            test_db.add_worker(worker_id, f"concurrent_host_{i}", {"hardware_types": ["cpu"]})
        
        # Create tasks
        tasks = self._generate_tasks(num_tasks)
        for task in tasks:
            test_db.add_task(
                task["task_id"],
                task["type"],
                task["priority"],
                {"benchmark": True},
                task["requirements"]
            )
        
        # Create components
        scheduler = TaskScheduler(test_db)
        balancer = LoadBalancer(test_db)
        monitor = HealthMonitor(test_db)
        
        # Create thread for each component
        stop_event = threading.Event()
        threads = []
        
        # Scheduler thread
        def scheduler_thread():
            for _ in range(20):
                if stop_event.is_set():
                    break
                # Get and assign tasks
                for i in range(5):
                    worker_id = f"concurrent_worker_{random.randint(0, num_workers-1)}"
                    scheduler.get_next_task(worker_id, {"hardware_types": ["cpu"]})
                time.sleep(0.1)
        
        # Balancer thread
        def balancer_thread():
            for _ in range(10):
                if stop_event.is_set():
                    break
                # Check load balance
                balancer.get_worker_load()
                balancer.detect_overloaded_workers()
                balancer.detect_underutilized_workers()
                time.sleep(0.2)
        
        # Monitor thread
        def monitor_thread():
            for _ in range(5):
                if stop_event.is_set():
                    break
                # Check health
                monitor.check_worker_health()
                monitor.get_health_status()
                time.sleep(0.5)
        
        # Database thread (simulates other operations)
        def db_thread():
            for _ in range(30):
                if stop_event.is_set():
                    break
                # Random database operations
                op_type = random.randint(0, 2)
                if op_type == 0:
                    # Get a random worker
                    worker_id = f"concurrent_worker_{random.randint(0, num_workers-1)}"
                    test_db.get_worker(worker_id)
                elif op_type == 1:
                    # Update a worker heartbeat
                    worker_id = f"concurrent_worker_{random.randint(0, num_workers-1)}"
                    test_db.update_worker_heartbeat(worker_id)
                else:
                    # Get tasks
                    test_db.get_pending_tasks(limit=10)
                time.sleep(0.05)
        
        # Start threads
        threads.append(threading.Thread(target=scheduler_thread))
        threads.append(threading.Thread(target=balancer_thread))
        threads.append(threading.Thread(target=monitor_thread))
        threads.append(threading.Thread(target=db_thread))
        
        # Benchmark concurrent operations
        with BenchmarkTimer(f"Concurrent operations - {len(threads)} threads"):
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for threads to complete
            time.sleep(5)  # Allow threads to run for 5 seconds
            
            # Stop threads
            stop_event.set()
            for thread in threads:
                thread.join(timeout=1.0)
        
        # Clean up
        test_db.close()
        os.close(test_db_fd)
        os.unlink(test_db_path)


if __name__ == "__main__":
    unittest.main()