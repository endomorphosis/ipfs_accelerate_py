#!/usr/bin/env python3
"""
Basic example of using the Adaptive Load Balancer.

This script demonstrates the basic usage of the load balancer component
with mock worker data and test assignments.
"""

import os
import sys
import time
import logging
import threading
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

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
    WorkerAssignment,
    WorkerCapabilityDetector
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("load_balancer_example")


def generate_mock_worker_capabilities(worker_id: str) -> WorkerCapabilities:
    """Generate mock capabilities for a worker."""
    worker_types = {
        "worker1": {
            "hardware_specs": {
                "platform": "Linux",
                "cpu": {"cores_physical": 8, "cores_logical": 16, "frequency_mhz": 3200},
                "memory": {"total_gb": 32, "available_gb": 28},
                "gpu": {"cuda_available": True, "device_count": 2}
            },
            "supported_backends": ["cpu", "cuda", "tensorrt"],
            "available_accelerators": {"cuda": 2},
            "cpu_cores": 8,
            "cpu_threads": 16,
            "available_memory": 28.0
        },
        "worker2": {
            "hardware_specs": {
                "platform": "Linux",
                "cpu": {"cores_physical": 4, "cores_logical": 8, "frequency_mhz": 2800},
                "memory": {"total_gb": 16, "available_gb": 12},
                "gpu": {"cuda_available": True, "device_count": 1}
            },
            "supported_backends": ["cpu", "cuda"],
            "available_accelerators": {"cuda": 1},
            "cpu_cores": 4,
            "cpu_threads": 8,
            "available_memory": 12.0
        },
        "worker3": {
            "hardware_specs": {
                "platform": "Darwin",
                "cpu": {"cores_physical": 10, "cores_logical": 10, "frequency_mhz": 3500},
                "memory": {"total_gb": 32, "available_gb": 24},
                "gpu": {"mps_available": True}
            },
            "supported_backends": ["cpu", "mps"],
            "available_accelerators": {"mps": 1},
            "cpu_cores": 10,
            "cpu_threads": 10,
            "available_memory": 24.0
        }
    }
    
    # Use predefined worker or create random one
    if worker_id in worker_types:
        config = worker_types[worker_id]
    else:
        # Random worker configuration
        has_gpu = random.random() > 0.3
        num_cores = random.randint(2, 16)
        config = {
            "hardware_specs": {
                "platform": random.choice(["Linux", "Darwin", "Windows"]),
                "cpu": {
                    "cores_physical": num_cores,
                    "cores_logical": num_cores * 2,
                    "frequency_mhz": random.randint(2000, 4000)
                },
                "memory": {
                    "total_gb": random.choice([8, 16, 32, 64]),
                    "available_gb": random.choice([6, 12, 24, 48])
                },
                "gpu": {
                    "cuda_available": has_gpu,
                    "device_count": random.randint(1, 4) if has_gpu else 0
                }
            },
            "supported_backends": ["cpu"] + (["cuda"] if has_gpu else []),
            "available_accelerators": {"cuda": random.randint(1, 4)} if has_gpu else {},
            "cpu_cores": num_cores,
            "cpu_threads": num_cores * 2,
            "available_memory": float(random.choice([6, 12, 24, 48]))
        }
    
    # Create capabilities object
    capabilities = WorkerCapabilities(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        hardware_specs=config["hardware_specs"],
        software_versions={
            "python": "3.10.0",
            "torch": "2.0.1",
            "tensorflow": "2.12.0",
            "onnxruntime": "1.14.0"
        },
        supported_backends=config["supported_backends"],
        network_bandwidth=1000.0,
        storage_capacity=500.0,
        available_accelerators=config["available_accelerators"],
        available_memory=config["available_memory"],
        available_disk=100.0,
        cpu_cores=config["cpu_cores"],
        cpu_threads=config["cpu_threads"]
    )
    
    return capabilities


def generate_mock_tests(num_tests: int) -> List[TestRequirements]:
    """Generate mock test requirements."""
    test_types = ["performance", "compatibility", "integration"]
    model_families = ["transformer", "diffusion", "vision", "audio"]
    model_ids = {
        "transformer": ["bert-base-uncased", "gpt2", "t5-base"],
        "diffusion": ["stable-diffusion-v1-5", "sd-turbo"],
        "vision": ["vit-base-patch16-224", "resnet50"],
        "audio": ["whisper-tiny", "wav2vec2-base"]
    }
    
    tests = []
    for i in range(num_tests):
        test_type = random.choice(test_types)
        model_family = random.choice(model_families)
        model_id = random.choice(model_ids[model_family])
        priority = random.randint(1, 5)
        
        # Determine resource requirements based on model
        if model_family == "transformer" and "gpt" in model_id:
            min_memory = 8.0
            expected_duration = 120.0
            required_backend = "cuda"
            required_accelerators = {"cuda": 1}
        elif model_family == "diffusion":
            min_memory = 12.0
            expected_duration = 180.0
            required_backend = "cuda"
            required_accelerators = {"cuda": 1}
        else:
            min_memory = 2.0
            expected_duration = 60.0
            required_backend = None
            required_accelerators = {}
            
        # Create test requirements
        test = TestRequirements(
            test_id=f"test-{i+1}",
            model_id=model_id,
            model_family=model_family,
            test_type=test_type,
            minimum_memory=min_memory,
            preferred_backend="cuda" if "cuda" in model_id else None,
            required_backend=required_backend,
            expected_duration=expected_duration,
            priority=priority,
            required_accelerators=required_accelerators
        )
        
        tests.append(test)
        
    return tests


def assignment_callback(assignment: WorkerAssignment) -> None:
    """Callback for assignment status changes."""
    if assignment.status == "completed":
        logger.info(f"Test {assignment.test_id} completed successfully on worker {assignment.worker_id} "
                  f"in {assignment.execution_time:.2f}s")
    elif assignment.status == "failed":
        logger.info(f"Test {assignment.test_id} failed on worker {assignment.worker_id}")


def worker_simulation(worker_id: str, load_balancer: LoadBalancerService,
                    stop_event: threading.Event) -> None:
    """Simulate worker behavior."""
    logger.info(f"Worker {worker_id} starting")
    
    # Register worker
    capabilities = generate_mock_worker_capabilities(worker_id)
    load_balancer.register_worker(worker_id, capabilities)
    
    # Initial load
    load = WorkerLoad(worker_id=worker_id)
    load_balancer.update_worker_load(worker_id, load)
    
    # Worker loop
    while not stop_event.is_set():
        # Get next assignment
        assignment = load_balancer.get_next_assignment(worker_id)
        
        if assignment:
            # Process assignment
            logger.info(f"Worker {worker_id} processing test {assignment.test_id}")
            
            # Mark as running
            load_balancer.update_assignment_status(assignment.test_id, "running")
            
            # Simulate work
            duration = random.uniform(0.5, 2.0) * assignment.test_requirements.expected_duration
            start_time = time.time()
            
            # Simulate load increase
            load.active_tests += 1
            load.cpu_utilization = min(100.0, load.cpu_utilization + 20.0)
            load.memory_utilization = min(100.0, load.memory_utilization + 15.0)
            if "cuda" in capabilities.supported_backends:
                load.gpu_utilization = min(100.0, load.gpu_utilization + 30.0)
            load_balancer.update_worker_load(worker_id, load)
            
            # Wait for "execution"
            remaining = max(0.1, duration - (time.time() - start_time))
            stop_event.wait(remaining)
            
            if stop_event.is_set():
                break
                
            # Simulate success or failure
            success = random.random() > 0.1  # 90% success rate
            status = "completed" if success else "failed"
            result = {"output": "test result data", "success": success}
            
            # Update status
            load_balancer.update_assignment_status(assignment.test_id, status, result)
            
            # Simulate load decrease
            load.active_tests = max(0, load.active_tests - 1)
            load.cpu_utilization = max(0.0, load.cpu_utilization - 20.0)
            load.memory_utilization = max(0.0, load.memory_utilization - 15.0)
            if "cuda" in capabilities.supported_backends:
                load.gpu_utilization = max(0.0, load.gpu_utilization - 30.0)
            load_balancer.update_worker_load(worker_id, load)
        else:
            # No assignment, wait a bit
            stop_event.wait(1.0)
            
            # Update load (simulate background work)
            load.cpu_utilization = max(0.0, min(100.0, load.cpu_utilization + random.uniform(-5.0, 5.0)))
            load.memory_utilization = max(0.0, min(100.0, load.memory_utilization + random.uniform(-3.0, 3.0)))
            if "cuda" in capabilities.supported_backends:
                load.gpu_utilization = max(0.0, min(100.0, load.gpu_utilization + random.uniform(-10.0, 10.0)))
            load_balancer.update_worker_load(worker_id, load)
            
    logger.info(f"Worker {worker_id} stopping")


def main():
    """Run the example."""
    print("========== Adaptive Load Balancer Example ==========")
    
    # Create load balancer
    load_balancer = LoadBalancerService()
    
    # Register callback
    load_balancer.register_assignment_callback(assignment_callback)
    
    # Start load balancer
    load_balancer.start()
    print("Load balancer started")
    
    # Create stop event
    stop_event = threading.Event()
    
    # Start worker threads
    worker_threads = []
    for i in range(1, 4):  # 3 workers
        worker_id = f"worker{i}"
        thread = threading.Thread(
            target=worker_simulation,
            args=(worker_id, load_balancer, stop_event),
            daemon=True
        )
        thread.start()
        worker_threads.append(thread)
        
    print(f"Started {len(worker_threads)} worker threads")
    
    try:
        # Generate and submit tests
        print("\n----- Submitting test batch 1 -----")
        tests = generate_mock_tests(5)  # 5 tests in first batch
        for test in tests:
            test_id = load_balancer.submit_test(test)
            print(f"Submitted test {test_id}: {test.model_id} (priority {test.priority})")
            
        # Wait a bit for tests to start processing
        time.sleep(10)
        
        # Submit more tests
        print("\n----- Submitting test batch 2 -----")
        tests = generate_mock_tests(10)  # 10 more tests
        for test in tests:
            test_id = load_balancer.submit_test(test)
            print(f"Submitted test {test_id}: {test.model_id} (priority {test.priority})")
            
        # Wait for tests to be processed
        print("\n----- Waiting for tests to complete -----")
        wait_time = 60  # Wait up to 60 seconds
        for _ in range(wait_time):
            time.sleep(1)
            
            # Check if all tests are processed
            with load_balancer.lock:
                if not load_balancer.pending_tests and all(
                    assignment.status in ["completed", "failed"]
                    for assignment in load_balancer.test_assignments.values()
                ):
                    break
                    
        # Submit high-priority test
        print("\n----- Submitting high-priority test -----")
        test = TestRequirements(
            test_id="high-priority-test",
            model_id="gpt2-xl",
            model_family="transformer",
            test_type="performance",
            minimum_memory=8.0,
            required_backend="cuda",
            expected_duration=180.0,
            priority=1,  # Highest priority
            required_accelerators={"cuda": 1}
        )
        test_id = load_balancer.submit_test(test)
        print(f"Submitted high-priority test {test_id}: {test.model_id} (priority {test.priority})")
        
        # Wait a bit more for high-priority test
        time.sleep(10)
        
        # Trigger rebalance
        print("\n----- Triggering rebalance -----")
        load_balancer.rebalance()
        
        # Wait a bit more
        time.sleep(10)
        
        # Print results
        print("\n----- Final results -----")
        with load_balancer.lock:
            total_tests = len(load_balancer.test_assignments)
            completed_tests = sum(1 for a in load_balancer.test_assignments.values() 
                                if a.status == "completed")
            failed_tests = sum(1 for a in load_balancer.test_assignments.values() 
                             if a.status == "failed")
            pending_tests = len(load_balancer.pending_tests)
            
            print(f"Total tests: {total_tests}")
            print(f"Completed tests: {completed_tests}")
            print(f"Failed tests: {failed_tests}")
            print(f"Pending tests: {pending_tests}")
            
            # Print assignments by worker
            assignments_by_worker = {}
            for worker_id, assignments in load_balancer.active_assignments.items():
                assignments_by_worker[worker_id] = len(assignments)
                
            print("\nAssignments by worker:")
            for worker_id, count in assignments_by_worker.items():
                print(f"  {worker_id}: {count}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Stop worker threads
        stop_event.set()
        
        # Stop load balancer
        load_balancer.stop()
        
        print("\n========== Example Complete ==========")


if __name__ == "__main__":
    main()