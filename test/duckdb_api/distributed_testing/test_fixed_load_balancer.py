#!/usr/bin/env python3
"""
Verify the fixed Adaptive Load Balancer implementation.

This script tests the fixed load balancer with a focus on:
1. Proper handling of resource capacity checks
2. Prevention of infinite requeuing
3. Graceful handling of tests that can't be scheduled
"""

import os
import sys
import time
import logging
import threading
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

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
logger = logging.getLogger("load_balancer_test")


def generate_test_worker_capabilities(worker_id: str, memory_gb: float = 8.0, 
                                    cuda_devices: int = 1) -> WorkerCapabilities:
    """Generate test worker capabilities with specific memory and GPU configuration."""
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


def test_assignment_callback(assignment: WorkerAssignment) -> None:
    """Test callback for assignment status changes."""
    if assignment.status == "completed":
        logger.info(f"✅ Test {assignment.test_id} completed successfully on {assignment.worker_id}")
    elif assignment.status == "failed":
        if assignment.worker_id == "none":
            logger.warning(f"❌ Test {assignment.test_id} could not be scheduled - {assignment.result.get('error', 'Unknown error')}")
        else:
            logger.warning(f"❌ Test {assignment.test_id} failed on {assignment.worker_id}")


def test_capacity_checks():
    """Test worker capacity checks for memory and accelerators."""
    print("\n--- Testing worker capacity checks ---")
    
    # Create load balancer
    load_balancer = LoadBalancerService()
    load_balancer.register_assignment_callback(test_assignment_callback)
    load_balancer.max_requeue_attempts = 3  # Set lower for faster testing
    load_balancer.start()
    
    try:
        # Register workers with varying capabilities
        worker1 = generate_test_worker_capabilities("worker1", memory_gb=8.0, cuda_devices=1)
        worker2 = generate_test_worker_capabilities("worker2", memory_gb=16.0, cuda_devices=2)
        worker3 = generate_test_worker_capabilities("worker3", memory_gb=4.0, cuda_devices=0)
        
        load_balancer.register_worker("worker1", worker1)
        load_balancer.register_worker("worker2", worker2)
        load_balancer.register_worker("worker3", worker3)
        
        # Initialize loads
        for worker_id in ["worker1", "worker2", "worker3"]:
            load_balancer.update_worker_load(worker_id, WorkerLoad(worker_id=worker_id))
            
        # Submit tests that should be schedulable
        print("\nSubmitting schedulable tests:")
        test1 = create_test_requirements("test1", memory_required=2.0, cuda_required=0, priority=3)
        test2 = create_test_requirements("test2", memory_required=4.0, cuda_required=1, priority=2)
        test3 = create_test_requirements("test3", memory_required=6.0, cuda_required=0, priority=1)
        
        load_balancer.submit_test(test1)
        load_balancer.submit_test(test2)
        load_balancer.submit_test(test3)
        
        # Give time for scheduling
        time.sleep(2)
        
        # Submit tests that should not be schedulable
        print("\nSubmitting tests that should fail scheduling:")
        test4 = create_test_requirements("test4", memory_required=20.0, cuda_required=0, priority=3)
        test5 = create_test_requirements("test5", memory_required=4.0, cuda_required=3, priority=1)
        
        load_balancer.submit_test(test4)
        load_balancer.submit_test(test5)
        
        # Wait for max requeue attempts to be reached
        time.sleep(3)
        
        # Print results
        print("\nTest assignments:")
        with load_balancer.lock:
            for test_id, assignment in load_balancer.test_assignments.items():
                print(f"  {test_id}: worker={assignment.worker_id}, status={assignment.status}")
                
            print("\nPending tests:")
            for test_id in load_balancer.pending_tests:
                print(f"  {test_id}")
                
            # Verify no requeue counts left
            print("\nRequeue counts:")
            for test_id, count in load_balancer.test_requeue_count.items():
                print(f"  {test_id}: {count}")
                
    finally:
        # Stop load balancer
        load_balancer.stop()
        
    print("\n--- Capacity check tests complete ---")


def test_requeue_limit():
    """Test maximum requeue limit to prevent infinite requeuing."""
    print("\n--- Testing requeue limit ---")
    
    # Create load balancer
    load_balancer = LoadBalancerService()
    load_balancer.register_assignment_callback(test_assignment_callback)
    load_balancer.max_requeue_attempts = 3  # Set lower for faster testing
    load_balancer.start()
    
    try:
        # Register one worker
        worker = generate_test_worker_capabilities("worker1", memory_gb=4.0, cuda_devices=0)
        load_balancer.register_worker("worker1", worker)
        load_balancer.update_worker_load("worker1", WorkerLoad(worker_id="worker1"))
        
        # Submit tests that require resources the worker doesn't have
        print("\nSubmitting tests that require unavailable resources:")
        test1 = create_test_requirements("gpu_test1", memory_required=2.0, cuda_required=1, priority=1)
        test2 = create_test_requirements("gpu_test2", memory_required=2.0, cuda_required=1, priority=2)
        test3 = create_test_requirements("memory_test", memory_required=8.0, cuda_required=0, priority=3)
        
        load_balancer.submit_test(test1)
        load_balancer.submit_test(test2)
        load_balancer.submit_test(test3)
        
        # Wait for requeue attempts to reach max
        time.sleep(3)
        
        # Verify all tests have failed with appropriate error
        print("\nVerifying test statuses:")
        with load_balancer.lock:
            for test_id, assignment in load_balancer.test_assignments.items():
                error = assignment.result.get("error", "") if assignment.result else ""
                print(f"  {test_id}: status={assignment.status}, error={error}")
                
            # Verify no pending tests remain
            print(f"\nPending tests count: {len(load_balancer.pending_tests)}")
            print(f"Requeue count entries: {len(load_balancer.test_requeue_count)}")
            
            assert len(load_balancer.pending_tests) == 0, "Should have no pending tests"
            assert len(load_balancer.test_requeue_count) == 0, "Should have no requeue counts"
            
    finally:
        # Stop load balancer
        load_balancer.stop()
        
    print("\n--- Requeue limit tests complete ---")


def main():
    """Run the test suite."""
    print("========== Testing Fixed Load Balancer ==========")
    
    # Test worker capacity checks
    test_capacity_checks()
    
    # Test requeue limit
    test_requeue_limit()
    
    print("\n========== All Tests Complete ==========")


if __name__ == "__main__":
    main()