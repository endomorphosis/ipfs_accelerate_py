#!/usr/bin/env python3
"""
Stress testing for the Adaptive Load Balancer.

This script tests the load balancer under high concurrency conditions to:
1. Verify performance and scalability with large numbers of workers
2. Assess scheduling throughput with many concurrent test submissions
3. Test stability under various load patterns
4. Benchmark resource utilization
5. Evaluate worker thermal management effectiveness
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
import random
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import queue

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import load balancer components
from duckdb_api.distributed_testing.load_balancer import (
    LoadBalancerService,
    WorkerCapabilities,
    WorkerLoad,
    TestRequirements,
    WorkerAssignment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("load_balancer_stress_test")


# Test configuration parameters
DEFAULT_NUM_WORKERS = 20
DEFAULT_NUM_TESTS = 100
DEFAULT_TEST_DURATION = 60  # seconds
DEFAULT_WORKER_MEMORY = [4, 8, 16, 32]  # GB, randomly selected per worker
DEFAULT_WORKER_CUDA = [0, 1, 2, 4]  # Number of CUDA devices, randomly selected
DEFAULT_TEST_MEMORY = [0.5, 1, 2, 4, 8]  # GB, randomly selected per test
DEFAULT_TEST_CUDA = [0, 0, 0, 1, 1, 2]  # More weighted toward CPU-only tests
DEFAULT_TEST_PRIORITIES = [1, 2, 2, 3, 3, 3, 4, 4, 5]  # Weighted distribution


def generate_worker_capabilities(worker_id: str, memory_gb: float = 8.0,
                                cuda_devices: int = 1) -> WorkerCapabilities:
    """Generate worker capabilities with specific memory and GPU configuration."""
    return WorkerCapabilities(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        hardware_specs={
            "platform": "Linux",
            "cpu": {"cores_physical": 4, "cores_logical": 8, "frequency_mhz": 3000},
            "memory": {"total_gb": memory_gb, "available_gb": memory_gb * 0.9},
            "gpu": {"cuda_available": cuda_devices > 0, "device_count": cuda_devices}
        },
        software_versions={
            "python": "3.10.0",
            "torch": "2.0.1"
        },
        supported_backends=["cpu"] + (["cuda"] if cuda_devices > 0 else []),
        network_bandwidth=1000.0,
        storage_capacity=500.0,
        available_accelerators={"cuda": cuda_devices} if cuda_devices > 0 else {},
        available_memory=memory_gb * 0.9,
        available_disk=100.0,
        cpu_cores=4,
        cpu_threads=8
    )


def create_test_requirements(test_id: str, memory_required: float = 2.0,
                           cuda_required: int = 0, priority: int = 3) -> TestRequirements:
    """Create test requirements with specific memory and CUDA requirements."""
    cuda_requirements = {}
    required_backend = None
    
    if cuda_required > 0:
        cuda_requirements = {"cuda": cuda_required}
        required_backend = "cuda"
        
    return TestRequirements(
        test_id=test_id,
        model_id=f"model-{test_id}",
        model_family="test",
        test_type="unit",
        minimum_memory=memory_required,
        required_backend=required_backend,
        expected_duration=10.0,
        priority=priority,
        required_accelerators=cuda_requirements
    )


# Metrics tracking
class TestMetrics:
    """Track and analyze test metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self.lock = threading.RLock()
        self.scheduled_times = {}  # test_id -> scheduled time
        self.completion_times = {}  # test_id -> completion time
        self.latencies = []  # scheduling latencies in seconds
        self.throughput_per_second = {}  # second -> count
        self.successfully_scheduled = 0
        self.failed_scheduling = 0
        self.assignments_per_worker = {}  # worker_id -> count
        self.schedule_attempts = {}  # test_id -> attempts
        self.start_time = datetime.now()
        
    def record_submission(self, test_id: str) -> None:
        """Record test submission time."""
        with self.lock:
            self.scheduled_times[test_id] = datetime.now()
            
    def record_completion(self, assignment: WorkerAssignment) -> None:
        """Record test completion."""
        with self.lock:
            test_id = assignment.test_id
            worker_id = assignment.worker_id
            now = datetime.now()
            
            # Record completion time
            self.completion_times[test_id] = now
            
            # Calculate scheduling latency if we have submission time
            if test_id in self.scheduled_times:
                latency = (assignment.assigned_at - self.scheduled_times[test_id]).total_seconds()
                self.latencies.append(latency)
                
            # Update throughput counter
            second = int((now - self.start_time).total_seconds())
            if second not in self.throughput_per_second:
                self.throughput_per_second[second] = 0
            self.throughput_per_second[second] += 1
            
            # Update success/failure count
            if assignment.status == "completed" or assignment.status == "running":
                self.successfully_scheduled += 1
                # Update worker assignment count
                if worker_id not in self.assignments_per_worker:
                    self.assignments_per_worker[worker_id] = 0
                self.assignments_per_worker[worker_id] += 1
            else:
                self.failed_scheduling += 1
                
    def record_scheduling_attempt(self, test_id: str) -> None:
        """Record a scheduling attempt for a test."""
        with self.lock:
            if test_id not in self.schedule_attempts:
                self.schedule_attempts[test_id] = 0
            self.schedule_attempts[test_id] += 1
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self.lock:
            total_tests = self.successfully_scheduled + self.failed_scheduling
            success_rate = (self.successfully_scheduled / total_tests * 100) if total_tests > 0 else 0
            
            # Calculate average latency
            avg_latency = statistics.mean(self.latencies) if self.latencies else 0
            max_latency = max(self.latencies) if self.latencies else 0
            min_latency = min(self.latencies) if self.latencies else 0
            
            # Calculate peak throughput (tests per second)
            throughput_values = list(self.throughput_per_second.values())
            peak_throughput = max(throughput_values) if throughput_values else 0
            avg_throughput = statistics.mean(throughput_values) if throughput_values else 0
            
            # Calculate worker distribution stats
            worker_assignments = list(self.assignments_per_worker.values())
            worker_assignment_stddev = statistics.stdev(worker_assignments) if len(worker_assignments) > 1 else 0
            worker_assignment_range = max(worker_assignments) - min(worker_assignments) if worker_assignments else 0
            worker_utilization = len(self.assignments_per_worker) / len(self.schedule_attempts) if self.schedule_attempts else 0
            
            # Calculate average scheduling attempts
            avg_attempts = statistics.mean(list(self.schedule_attempts.values())) if self.schedule_attempts else 0
            
            # Return summary
            return {
                "total_tests": total_tests,
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "max_latency": max_latency,
                "min_latency": min_latency,
                "peak_throughput": peak_throughput,
                "avg_throughput": avg_throughput,
                "worker_assignment_stddev": worker_assignment_stddev,
                "worker_assignment_range": worker_assignment_range,
                "worker_utilization": worker_utilization,
                "avg_scheduling_attempts": avg_attempts
            }


class LoadBalancerStressTest:
    """Stress test for the Adaptive Load Balancer."""
    
    def __init__(self, 
                 num_workers: int = DEFAULT_NUM_WORKERS, 
                 num_tests: int = DEFAULT_NUM_TESTS,
                 duration: int = DEFAULT_TEST_DURATION,
                 worker_memory: List[int] = None,
                 worker_cuda: List[int] = None,
                 test_memory: List[float] = None,
                 test_cuda: List[int] = None,
                 burst_mode: bool = False,
                 dynamic_workers: bool = False):
        """Initialize stress test configuration.
        
        Args:
            num_workers: Number of workers to simulate
            num_tests: Number of tests to submit
            duration: Duration of the test in seconds
            worker_memory: List of possible memory values (GB) for workers
            worker_cuda: List of possible CUDA device counts for workers
            test_memory: List of possible memory requirements (GB) for tests
            test_cuda: List of possible CUDA device requirements for tests
            burst_mode: Submit tests in bursts rather than even distribution
            dynamic_workers: Add/remove workers during the test
        """
        self.num_workers = num_workers
        self.num_tests = num_tests
        self.duration = duration
        self.worker_memory = worker_memory or DEFAULT_WORKER_MEMORY
        self.worker_cuda = worker_cuda or DEFAULT_WORKER_CUDA
        self.test_memory = test_memory or DEFAULT_TEST_MEMORY
        self.test_cuda = test_cuda or DEFAULT_TEST_CUDA
        self.burst_mode = burst_mode
        self.dynamic_workers = dynamic_workers
        
        # Initialize load balancer
        self.load_balancer = LoadBalancerService()
        
        # Metrics
        self.metrics = TestMetrics()
        
        # Worker state
        self.active_workers = set()
        self.worker_lock = threading.Lock()
        
        # Completion signals
        self.stop_event = threading.Event()
        self.worker_update_thread = None
        self.test_submit_thread = None
        
    def assignment_callback(self, assignment: WorkerAssignment) -> None:
        """Callback for test assignments."""
        self.metrics.record_completion(assignment)
        
        if assignment.status == "completed":
            logger.debug(f"✅ Test {assignment.test_id} completed on {assignment.worker_id}")
        elif assignment.status == "failed":
            if assignment.worker_id == "none":
                logger.debug(f"❌ Test {assignment.test_id} failed to schedule - {assignment.result.get('error', 'Unknown error')}")
            else:
                logger.debug(f"❌ Test {assignment.test_id} failed on {assignment.worker_id}")
                
    def register_worker(self) -> str:
        """Register a new worker with random capabilities."""
        worker_id = f"worker-{uuid.uuid4()}"
        memory_gb = random.choice(self.worker_memory)
        cuda_devices = random.choice(self.worker_cuda)
        
        capabilities = generate_worker_capabilities(worker_id, memory_gb, cuda_devices)
        self.load_balancer.register_worker(worker_id, capabilities)
        self.load_balancer.update_worker_load(worker_id, WorkerLoad(worker_id=worker_id))
        
        logger.debug(f"Registered worker {worker_id} with {memory_gb}GB memory, {cuda_devices} CUDA devices")
        
        return worker_id
        
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        self.load_balancer.unregister_worker(worker_id)
        logger.debug(f"Unregistered worker {worker_id}")
        
    def submit_test(self) -> str:
        """Submit a test with random requirements."""
        test_id = f"test-{uuid.uuid4()}"
        memory_required = random.choice(self.test_memory)
        cuda_required = random.choice(self.test_cuda)
        priority = random.choice(DEFAULT_TEST_PRIORITIES)
        
        requirements = create_test_requirements(
            test_id, memory_required, cuda_required, priority
        )
        
        self.load_balancer.submit_test(requirements)
        self.metrics.record_submission(test_id)
        
        logger.debug(f"Submitted test {test_id} with {memory_required}GB memory, {cuda_required} CUDA devices, priority {priority}")
        
        return test_id
        
    def worker_update_loop(self) -> None:
        """Worker management loop."""
        while not self.stop_event.is_set():
            try:
                with self.worker_lock:
                    # Simulate worker load changes
                    for worker_id in self.active_workers:
                        # Skip with some probability
                        if random.random() < 0.7:  # Only update 30% of workers each cycle
                            continue
                            
                        # Get current load or create new
                        load = WorkerLoad(worker_id=worker_id)
                        
                        # Randomize utilization values
                        load.cpu_utilization = random.uniform(10, 90)
                        load.memory_utilization = random.uniform(20, 80)
                        load.gpu_utilization = random.uniform(0, 100) if random.random() < 0.7 else 0.0
                        load.io_utilization = random.uniform(5, 50)
                        load.network_utilization = random.uniform(10, 60)
                        
                        # Simulate thermal state changes occasionally
                        if random.random() < 0.05:  # 5% chance to start warming
                            load.start_warming()
                        elif random.random() < 0.05:  # 5% chance to start cooling
                            load.start_cooling()
                            
                        # Update load in load balancer
                        self.load_balancer.update_worker_load(worker_id, load)
                    
                    # If dynamic workers enabled, add or remove workers
                    if self.dynamic_workers:
                        # 10% chance to add a worker if below max
                        if random.random() < 0.1 and len(self.active_workers) < self.num_workers * 1.5:
                            worker_id = self.register_worker()
                            self.active_workers.add(worker_id)
                            
                        # 5% chance to remove a worker if above min
                        if random.random() < 0.05 and len(self.active_workers) > self.num_workers * 0.5:
                            # Select a random worker to remove
                            if self.active_workers:
                                worker_id = random.choice(list(self.active_workers))
                                self.unregister_worker(worker_id)
                                self.active_workers.remove(worker_id)
            except Exception as e:
                logger.error(f"Error in worker update loop: {e}")
                
            # Sleep for a random interval
            sleep_time = random.uniform(0.2, 1.0)  # 200ms to 1s
            self.stop_event.wait(sleep_time)
            
    def test_submit_loop(self) -> None:
        """Test submission loop."""
        tests_per_second = self.num_tests / self.duration
        interval = 1.0 / tests_per_second if tests_per_second > 0 else 1.0
        
        # Calculate parameters for burst mode
        if self.burst_mode:
            burst_size = min(int(self.num_tests / 10), 50)  # 10 bursts or max 50 tests per burst
            burst_interval = self.duration / (self.num_tests / burst_size)
        
        tests_submitted = 0
        start_time = time.time()
        
        while tests_submitted < self.num_tests and not self.stop_event.is_set():
            try:
                elapsed = time.time() - start_time
                if elapsed > self.duration:
                    logger.info(f"Test duration exceeded, stopping at {tests_submitted} tests")
                    break
                
                if self.burst_mode:
                    # Submit a burst of tests
                    actual_burst = min(burst_size, self.num_tests - tests_submitted)
                    logger.info(f"Submitting burst of {actual_burst} tests")
                    
                    for _ in range(actual_burst):
                        self.submit_test()
                        tests_submitted += 1
                        
                    # Wait until next burst
                    next_burst_time = start_time + (tests_submitted / self.num_tests) * self.duration
                    sleep_time = max(0, next_burst_time - time.time())
                    self.stop_event.wait(sleep_time)
                else:
                    # Submit at regular intervals
                    self.submit_test()
                    tests_submitted += 1
                    
                    # Sleep for interval
                    self.stop_event.wait(interval)
            except Exception as e:
                logger.error(f"Error in test submit loop: {e}")
                time.sleep(0.1)  # Brief pause on error
        
        logger.info(f"Completed submitting {tests_submitted} tests")
        
    def run(self) -> Dict[str, Any]:
        """Run the stress test."""
        logger.info(f"Starting stress test with {self.num_workers} workers, {self.num_tests} tests over {self.duration}s")
        
        # Register assignment callback
        self.load_balancer.register_assignment_callback(self.assignment_callback)
        
        # Start load balancer
        self.load_balancer.start()
        
        try:
            # Register initial workers
            for _ in range(self.num_workers):
                worker_id = self.register_worker()
                self.active_workers.add(worker_id)
                
            logger.info(f"Registered {len(self.active_workers)} workers")
            
            # Start worker update loop
            self.worker_update_thread = threading.Thread(
                target=self.worker_update_loop,
                daemon=True
            )
            self.worker_update_thread.start()
            
            # Start test submission loop
            self.test_submit_thread = threading.Thread(
                target=self.test_submit_loop,
                daemon=True
            )
            self.test_submit_thread.start()
            
            # Wait for test duration plus a buffer for processing
            time.sleep(self.duration + 10)  # Add 10 second buffer for processing
            
            # Stop threads
            self.stop_event.set()
            
            if self.worker_update_thread and self.worker_update_thread.is_alive():
                self.worker_update_thread.join(timeout=5)
                
            if self.test_submit_thread and self.test_submit_thread.is_alive():
                self.test_submit_thread.join(timeout=5)
                
            # Get metrics summary
            metrics = self.metrics.get_summary()
            
            # Add active worker counts
            with self.worker_lock:
                metrics["final_worker_count"] = len(self.active_workers)
                
            # Get balancer stats
            with self.load_balancer.lock:
                metrics["pending_tests"] = len(self.load_balancer.pending_tests)
                metrics["assigned_tests"] = len(self.load_balancer.test_assignments)
                
            return metrics
            
        finally:
            # Stop load balancer
            self.load_balancer.stop()
            logger.info("Stress test completed")


def run_single_stress_test(args: argparse.Namespace) -> None:
    """Run a single stress test with the specified parameters."""
    # Create the test with specified parameters
    test = LoadBalancerStressTest(
        num_workers=args.workers,
        num_tests=args.tests,
        duration=args.duration,
        burst_mode=args.burst,
        dynamic_workers=args.dynamic
    )
    
    # Run the test
    logger.info("Starting stress test...")
    metrics = test.run()
    
    # Print results
    print("\n========== Stress Test Results ==========")
    print(f"Test Configuration:")
    print(f"  Workers: {args.workers}")
    print(f"  Tests: {args.tests}")
    print(f"  Duration: {args.duration}s")
    print(f"  Burst Mode: {args.burst}")
    print(f"  Dynamic Workers: {args.dynamic}")
    
    print("\nPerformance Metrics:")
    print(f"  Success Rate: {metrics['success_rate']:.2f}%")
    print(f"  Average Latency: {metrics['avg_latency']:.2f}s")
    print(f"  Min/Max Latency: {metrics['min_latency']:.2f}s / {metrics['max_latency']:.2f}s")
    print(f"  Peak Throughput: {metrics['peak_throughput']} tests/second")
    print(f"  Average Throughput: {metrics['avg_throughput']:.2f} tests/second")
    
    print("\nWorker Distribution:")
    print(f"  Worker Assignment Std Dev: {metrics['worker_assignment_stddev']:.2f}")
    print(f"  Worker Assignment Range: {metrics['worker_assignment_range']}")
    print(f"  Worker Utilization: {metrics['worker_utilization'] * 100:.2f}%")
    print(f"  Average Scheduling Attempts: {metrics['avg_scheduling_attempts']:.2f}")
    
    print("\nFinal State:")
    print(f"  Final Worker Count: {metrics['final_worker_count']}")
    print(f"  Pending Tests: {metrics['pending_tests']}")
    print(f"  Assigned Tests: {metrics['assigned_tests']}")
    
    print("========================================\n")
    
    # Save results to file if specified
    if args.output:
        result_data = {
            "configuration": {
                "workers": args.workers,
                "tests": args.tests,
                "duration": args.duration,
                "burst_mode": args.burst,
                "dynamic_workers": args.dynamic,
                "timestamp": datetime.now().isoformat()
            },
            "metrics": metrics
        }
        
        with open(args.output, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        print(f"Results saved to {args.output}")


def run_benchmark_suite(args: argparse.Namespace) -> None:
    """Run a suite of benchmarks with varying configurations."""
    logger.info("Starting benchmark suite...")
    
    # Define configurations to test
    worker_counts = [5, 10, 20, 50, 100]  # Number of workers
    test_counts = [100, 500, 1000]        # Number of tests
    
    # Prepare results container
    results = []
    
    # Run benchmarks
    for workers in worker_counts:
        for tests in test_counts:
            # Skip very large combinations to save time
            if workers * tests > 20000 and not args.full:
                continue
                
            # Calculate duration based on test count
            duration = min(tests // 10, 120)  # 10 tests per second, max 2 minutes
            
            logger.info(f"Running benchmark with {workers} workers, {tests} tests, {duration}s duration")
            
            # Create test instance
            test = LoadBalancerStressTest(
                num_workers=workers,
                num_tests=tests,
                duration=duration,
                burst_mode=False,
                dynamic_workers=False
            )
            
            # Run the test
            metrics = test.run()
            
            # Add configuration to results
            result = {
                "workers": workers,
                "tests": tests,
                "duration": duration,
                "metrics": metrics
            }
            
            results.append(result)
            
            # Print abbreviated results
            print(f"\nBenchmark w={workers}, t={tests}, d={duration}s:")
            print(f"  Throughput: {metrics['peak_throughput']} tests/s, Latency: {metrics['avg_latency']:.2f}s")
            print(f"  Success Rate: {metrics['success_rate']:.2f}%")
            
    # Save benchmark results
    output_file = args.output or f"load_balancer_benchmark_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "benchmark_results": results,
            "timestamp": datetime.now().isoformat(),
            "full_suite": args.full
        }, f, indent=2)
        
    print(f"\nBenchmark results saved to {output_file}")
    
    # Generate summary
    print("\n========== Benchmark Summary ==========")
    print("Workers | Tests | Throughput | Latency | Success Rate")
    print("--------|-------|------------|---------|-------------")
    for result in results:
        workers = result["workers"]
        tests = result["tests"]
        throughput = result["metrics"]["peak_throughput"]
        latency = result["metrics"]["avg_latency"]
        success = result["metrics"]["success_rate"]
        print(f"{workers:7d} | {tests:5d} | {throughput:10d} | {latency:7.2f} | {success:11.2f}%")


def simulate_load_spike(args: argparse.Namespace) -> None:
    """Simulate a load spike scenario."""
    logger.info("Starting load spike simulation...")
    
    # Create test instance with dynamic workers enabled
    test = LoadBalancerStressTest(
        num_workers=args.workers,
        num_tests=args.tests,
        duration=args.duration,
        burst_mode=True,  # Use burst mode for spikes
        dynamic_workers=True  # Enable dynamic workers
    )
    
    # Register callback to monitor metrics
    throughput_queue = queue.Queue()
    latency_queue = queue.Queue()
    
    original_callback = test.assignment_callback
    
    def enhanced_callback(assignment: WorkerAssignment):
        # Call original callback
        original_callback(assignment)
        
        # Get current metrics
        if hasattr(test, 'metrics'):
            summary = test.metrics.get_summary()
            if summary["total_tests"] > 0:
                throughput_queue.put((time.time(), summary["peak_throughput"]))
                latency_queue.put((time.time(), summary["avg_latency"]))
    
    test.assignment_callback = enhanced_callback
    
    # Run the test
    logger.info("Running load spike simulation...")
    metrics = test.run()
    
    # Print results
    print("\n========== Load Spike Simulation Results ==========")
    print(f"Test Configuration:")
    print(f"  Initial Workers: {args.workers}")
    print(f"  Final Workers: {metrics['final_worker_count']}")
    print(f"  Tests: {args.tests}")
    print(f"  Duration: {args.duration}s")
    
    print("\nPerformance Under Load Spike:")
    print(f"  Success Rate: {metrics['success_rate']:.2f}%")
    print(f"  Average Latency: {metrics['avg_latency']:.2f}s")
    print(f"  Peak Throughput: {metrics['peak_throughput']} tests/second")
    print(f"  Worker Utilization: {metrics['worker_utilization'] * 100:.2f}%")
    
    # Save results
    if args.output:
        # Extract time series data from queues
        throughput_data = []
        while not throughput_queue.empty():
            throughput_data.append(throughput_queue.get())
            
        latency_data = []
        while not latency_queue.empty():
            latency_data.append(latency_queue.get())
            
        result_data = {
            "configuration": {
                "initial_workers": args.workers,
                "final_workers": metrics['final_worker_count'],
                "tests": args.tests,
                "duration": args.duration,
                "timestamp": datetime.now().isoformat()
            },
            "metrics": metrics,
            "time_series": {
                "throughput": throughput_data,
                "latency": latency_data
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        print(f"Results saved to {args.output}")


def load_config(config_file_path: str = None) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    if not config_file_path:
        # Use default path
        config_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "load_balancer_stress_config.json"
        )
    
    if not os.path.exists(config_file_path):
        logger.warning(f"Config file {config_file_path} not found, using default configuration")
        return {}
    
    try:
        with open(config_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}


def get_scenario_configuration(config: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """Get configuration for a specific test scenario."""
    if not config or 'scenario_configurations' not in config:
        return {}
    
    scenarios = config.get('scenario_configurations', {})
    if scenario_name not in scenarios:
        logger.warning(f"Scenario {scenario_name} not found in configuration")
        return {}
    
    scenario = scenarios[scenario_name]
    
    # Get test configuration
    test_config_name = scenario.get('test_config')
    test_configs = config.get('test_configurations', {})
    test_config = test_configs.get(test_config_name, {})
    
    # Get worker profile
    worker_profile_name = scenario.get('worker_profile')
    worker_profiles = config.get('worker_profiles', {})
    worker_profile = worker_profiles.get(worker_profile_name, {})
    
    # Get test profile
    test_profile_name = scenario.get('test_profile')
    test_profiles = config.get('test_profiles', {})
    test_profile = test_profiles.get(test_profile_name, {})
    
    # Combine all configurations
    result = {
        "workers": test_config.get('workers', DEFAULT_NUM_WORKERS),
        "tests": test_config.get('tests', DEFAULT_NUM_TESTS),
        "duration": test_config.get('duration', DEFAULT_TEST_DURATION),
        "worker_memory": worker_profile.get('memory_distribution'),
        "worker_cuda": worker_profile.get('cuda_distribution'),
        "test_memory": test_profile.get('memory_requirements'),
        "test_cuda": test_profile.get('cuda_requirements'),
        "priority_distribution": test_profile.get('priority_distribution'),
        "burst_mode": scenario.get('burst_mode', False),
        "dynamic_workers": scenario.get('dynamic_workers', False),
        "description": scenario.get('description', "")
    }
    
    return result


def list_available_scenarios(config: Dict[str, Any]) -> None:
    """List all available test scenarios in the configuration."""
    if not config or 'scenario_configurations' not in config:
        print("No scenario configurations found")
        return
    
    scenarios = config.get('scenario_configurations', {})
    if not scenarios:
        print("No scenario configurations found")
        return
    
    print("\n=== Available Test Scenarios ===\n")
    print(f"{'Scenario Name':<20} | {'Description':<60}")
    print("-" * 20 + "-+-" + "-" * 60)
    
    for name, scenario in scenarios.items():
        description = scenario.get('description', "No description")
        print(f"{name:<20} | {description:<60}")
    
    print("\nUse '--scenario <name>' to run a specific scenario")


def run_scenario(scenario_name: str, output_file: str, config: Dict[str, Any]) -> None:
    """Run a specific test scenario from configuration."""
    scenario_config = get_scenario_configuration(config, scenario_name)
    if not scenario_config:
        logger.error(f"Failed to load configuration for scenario {scenario_name}")
        return
    
    logger.info(f"Running scenario: {scenario_name}")
    logger.info(f"Description: {scenario_config.get('description', 'No description')}")
    
    # Create stress test with scenario configuration
    test = LoadBalancerStressTest(
        num_workers=scenario_config.get('workers', DEFAULT_NUM_WORKERS),
        num_tests=scenario_config.get('tests', DEFAULT_NUM_TESTS),
        duration=scenario_config.get('duration', DEFAULT_TEST_DURATION),
        worker_memory=scenario_config.get('worker_memory'),
        worker_cuda=scenario_config.get('worker_cuda'),
        test_memory=scenario_config.get('test_memory'),
        test_cuda=scenario_config.get('test_cuda'),
        burst_mode=scenario_config.get('burst_mode', False),
        dynamic_workers=scenario_config.get('dynamic_workers', False)
    )
    
    # Override DEFAULT_TEST_PRIORITIES if priority_distribution is provided
    if 'priority_distribution' in scenario_config and scenario_config['priority_distribution']:
        global DEFAULT_TEST_PRIORITIES
        DEFAULT_TEST_PRIORITIES = scenario_config['priority_distribution']
    
    # Run the test
    logger.info(f"Starting scenario test with {scenario_config.get('workers')} workers, "
                f"{scenario_config.get('tests')} tests, {scenario_config.get('duration')}s duration")
    metrics = test.run()
    
    # Print results
    print(f"\n========== Scenario Test Results: {scenario_name} ==========")
    print(f"Test Configuration:")
    print(f"  Description: {scenario_config.get('description', 'No description')}")
    print(f"  Workers: {scenario_config.get('workers')}")
    print(f"  Tests: {scenario_config.get('tests')}")
    print(f"  Duration: {scenario_config.get('duration')}s")
    print(f"  Burst Mode: {scenario_config.get('burst_mode')}")
    print(f"  Dynamic Workers: {scenario_config.get('dynamic_workers')}")
    
    print("\nPerformance Metrics:")
    print(f"  Success Rate: {metrics['success_rate']:.2f}%")
    print(f"  Average Latency: {metrics['avg_latency']:.2f}s")
    print(f"  Min/Max Latency: {metrics['min_latency']:.2f}s / {metrics['max_latency']:.2f}s")
    print(f"  Peak Throughput: {metrics['peak_throughput']} tests/second")
    print(f"  Average Throughput: {metrics['avg_throughput']:.2f} tests/second")
    
    print("\nWorker Distribution:")
    print(f"  Worker Assignment Std Dev: {metrics['worker_assignment_stddev']:.2f}")
    print(f"  Worker Assignment Range: {metrics['worker_assignment_range']}")
    print(f"  Worker Utilization: {metrics['worker_utilization'] * 100:.2f}%")
    print(f"  Average Scheduling Attempts: {metrics['avg_scheduling_attempts']:.2f}")
    
    print("\nFinal State:")
    print(f"  Final Worker Count: {metrics['final_worker_count']}")
    print(f"  Pending Tests: {metrics['pending_tests']}")
    print(f"  Assigned Tests: {metrics['assigned_tests']}")
    
    print("========================================\n")
    
    # Save results to file if specified
    if output_file:
        result_data = {
            "scenario": scenario_name,
            "configuration": scenario_config,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        print(f"Results saved to {output_file}")


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Load Balancer Stress Test Tool")
    
    # Common arguments
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"Number of workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--tests", type=int, default=DEFAULT_NUM_TESTS,
                        help=f"Number of tests (default: {DEFAULT_NUM_TESTS})")
    parser.add_argument("--duration", type=int, default=DEFAULT_TEST_DURATION,
                        help=f"Test duration in seconds (default: {DEFAULT_TEST_DURATION})")
    parser.add_argument("--output", type=str, default="",
                        help="Output file for results (JSON format)")
    parser.add_argument("--config", type=str, default="",
                        help="Path to configuration file (JSON format)")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="List available test scenarios from configuration")
    parser.add_argument("--scenario", type=str, default="",
                        help="Run a specific test scenario from configuration")
    
    # Create subparsers for different test modes
    subparsers = parser.add_subparsers(dest="mode", help="Test mode")
    
    # Single stress test
    stress_parser = subparsers.add_parser("stress", help="Run a single stress test")
    stress_parser.add_argument("--burst", action="store_true", 
                              help="Submit tests in bursts rather than evenly distributed")
    stress_parser.add_argument("--dynamic", action="store_true",
                              help="Add/remove workers during the test")
    
    # Benchmark suite
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark suite with varying configurations")
    benchmark_parser.add_argument("--full", action="store_true",
                                 help="Run the full benchmark suite (can take a long time)")
    
    # Load spike simulation
    spike_parser = subparsers.add_parser("spike", help="Simulate load spikes")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # List scenarios if requested
    if args.list_scenarios:
        list_available_scenarios(config)
        return
        
    # Run scenario if specified
    if args.scenario:
        run_scenario(args.scenario, args.output, config)
        return
    
    # Default to stress test if no mode specified
    if not args.mode:
        args.mode = "stress"
        args.burst = False
        args.dynamic = False
    
    # Run appropriate test
    if args.mode == "stress":
        run_single_stress_test(args)
    elif args.mode == "benchmark":
        run_benchmark_suite(args)
    elif args.mode == "spike":
        simulate_load_spike(args)


if __name__ == "__main__":
    main()