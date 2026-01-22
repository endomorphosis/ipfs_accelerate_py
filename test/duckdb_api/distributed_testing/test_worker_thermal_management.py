#!/usr/bin/env python3
"""
Test worker thermal management functionality (warming/cooling).

This script tests the worker warming and cooling features in the load balancer
to verify proper behavior and impact on scheduling decisions.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
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
logger = logging.getLogger("thermal_management_test")


def test_thermal_state_transitions():
    """Test thermal state transitions and performance level changes."""
    print("\n--- Testing Thermal State Transitions ---")
    
    # Create worker load
    worker_load = WorkerLoad(worker_id="test_worker")
    
    # Initial state
    print(f"Initial state: warming={worker_load.warming_state}, cooling={worker_load.cooling_state}, "
          f"performance={worker_load.performance_level:.2f}")
    
    # Start warming
    print("\nStarting warm-up (5 seconds)")
    worker_load.start_warming(duration_seconds=5.0)
    print(f"After start_warming: warming={worker_load.warming_state}, cooling={worker_load.cooling_state}, "
          f"performance={worker_load.performance_level:.2f}")
    
    # Update state a few times during warming
    for i in range(3):
        time.sleep(1.0)
        worker_load.update_thermal_state()
        print(f"During warming ({i+1}s): warming={worker_load.warming_state}, cooling={worker_load.cooling_state}, "
              f"performance={worker_load.performance_level:.2f}")
    
    # Wait until warming is complete
    time.sleep(3.0)
    worker_load.update_thermal_state()
    print(f"After warming: warming={worker_load.warming_state}, cooling={worker_load.cooling_state}, "
          f"performance={worker_load.performance_level:.2f}")
    
    # Start cooling
    print("\nStarting cool-down (5 seconds)")
    worker_load.start_cooling(duration_seconds=5.0)
    print(f"After start_cooling: warming={worker_load.warming_state}, cooling={worker_load.cooling_state}, "
          f"performance={worker_load.performance_level:.2f}")
    
    # Update state a few times during cooling
    for i in range(3):
        time.sleep(1.0)
        worker_load.update_thermal_state()
        print(f"During cooling ({i+1}s): warming={worker_load.warming_state}, cooling={worker_load.cooling_state}, "
              f"performance={worker_load.performance_level:.2f}")
    
    # Wait until cooling is complete
    time.sleep(3.0)
    worker_load.update_thermal_state()
    print(f"After cooling: warming={worker_load.warming_state}, cooling={worker_load.cooling_state}, "
          f"performance={worker_load.performance_level:.2f}")


def test_effective_load_calculation():
    """Test the impact of thermal states on effective load calculation."""
    print("\n--- Testing Effective Load Calculation ---")
    
    # Create worker load with some base utilization
    worker_load = WorkerLoad(
        worker_id="test_worker",
        cpu_utilization=50.0,
        memory_utilization=40.0
    )
    
    # Get base load score
    base_score = worker_load.calculate_load_score()
    effective_score = worker_load.get_effective_load_score()
    print(f"Normal state: base_score={base_score:.2f}, effective_score={effective_score:.2f}")
    
    # Test during warming
    worker_load.start_warming()
    worker_load.performance_level = 0.7  # Mid-warming
    
    warming_base_score = worker_load.calculate_load_score()
    warming_effective_score = worker_load.get_effective_load_score()
    print(f"Warming state: base_score={warming_base_score:.2f}, effective_score={warming_effective_score:.2f}")
    
    # Test during cooling
    worker_load.start_cooling()
    worker_load.performance_level = 0.8  # Mid-cooling
    
    cooling_base_score = worker_load.calculate_load_score()
    cooling_effective_score = worker_load.get_effective_load_score()
    print(f"Cooling state: base_score={cooling_base_score:.2f}, effective_score={cooling_effective_score:.2f}")


def test_thermal_management_in_load_balancer():
    """Test thermal management within the load balancer service."""
    print("\n--- Testing Thermal Management in Load Balancer ---")
    
    # Create load balancer
    load_balancer = LoadBalancerService()
    load_balancer.monitoring_interval = 1  # Faster for testing
    load_balancer.work_steal_interval = 2  # Faster for testing
    load_balancer.start()
    
    try:
        # Register workers
        worker1 = WorkerCapabilities(
            worker_id="worker1",
            hostname="host1",
            supported_backends=["cpu"],
            available_memory=8.0,
            cpu_cores=4,
            cpu_threads=8
        )
        worker2 = WorkerCapabilities(
            worker_id="worker2",
            hostname="host2",
            supported_backends=["cpu"],
            available_memory=8.0,
            cpu_cores=4,
            cpu_threads=8
        )
        
        load_balancer.register_worker("worker1", worker1)
        load_balancer.register_worker("worker2", worker2)
        
        # Initialize loads
        load1 = WorkerLoad(worker_id="worker1")
        load2 = WorkerLoad(worker_id="worker2")
        
        # Worker 1 is idle
        load_balancer.update_worker_load("worker1", load1)
        
        # Worker 2 is highly loaded
        load2.cpu_utilization = 90.0
        load2.memory_utilization = 85.0
        load2.active_tests = 5
        load_balancer.update_worker_load("worker2", load2)
        
        # Let monitoring run to trigger thermal management
        print("Waiting for thermal management to trigger...")
        time.sleep(3)
        
        # Check worker thermal states
        with load_balancer.lock:
            load1 = load_balancer.worker_loads.get("worker1")
            load2 = load_balancer.worker_loads.get("worker2")
            
            if load1:
                print(f"Worker1: warming={load1.warming_state}, cooling={load1.cooling_state}, "
                      f"performance={load1.performance_level:.2f}")
            
            if load2:
                print(f"Worker2: warming={load2.warming_state}, cooling={load2.cooling_state}, "
                      f"performance={load2.performance_level:.2f}")
                
        # Submit a test that could go to either worker
        print("\nSubmitting test to both workers...")
        test_req = TestRequirements(
            test_id="test1",
            model_id="test-model",
            minimum_memory=2.0,
            expected_duration=5.0
        )
        
        test_id = load_balancer.submit_test(test_req)
        
        # Wait for scheduling
        time.sleep(2)
        
        # Check assignment
        with load_balancer.lock:
            assignment = load_balancer.test_assignments.get(test_id)
            if assignment:
                print(f"Test assigned to: {assignment.worker_id}")
                
                # The idle worker should be warming up, so the test should go to the busy worker
                # unless the busy worker is cooling down
                load1 = load_balancer.worker_loads.get("worker1")
                load2 = load_balancer.worker_loads.get("worker2")
                
                if load1:
                    print(f"Worker1: warming={load1.warming_state}, cooling={load1.cooling_state}, "
                          f"performance={load1.performance_level:.2f}, "
                          f"effective_load={load1.get_effective_load_score():.2f}")
                
                if load2:
                    print(f"Worker2: warming={load2.warming_state}, cooling={load2.cooling_state}, "
                          f"performance={load2.performance_level:.2f}, "
                          f"effective_load={load2.get_effective_load_score():.2f}")
                
    finally:
        # Stop load balancer
        load_balancer.stop()


def main():
    """Run all tests."""
    print("========== Testing Worker Thermal Management ==========")
    
    # Test thermal state transitions
    test_thermal_state_transitions()
    
    # Test effective load calculation
    test_effective_load_calculation()
    
    # Test thermal management in load balancer
    test_thermal_management_in_load_balancer()
    
    print("\n========== All Tests Complete ==========")


if __name__ == "__main__":
    main()