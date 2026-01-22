#!/usr/bin/env python3
"""
Load Balancer Integration Example

This script demonstrates how to integrate the Hardware-Aware Workload Management 
system with the Load Balancer component, using the HardwareAwareScheduler.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
import random
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import load balancer components
from duckdb_api.distributed_testing.load_balancer.models import (
    TestRequirements, WorkerCapabilities, WorkerLoad, WorkerPerformance, WorkerAssignment
)
from duckdb_api.distributed_testing.load_balancer.scheduling_algorithms import SchedulingAlgorithm
from duckdb_api.distributed_testing.load_balancer.service import LoadBalancerService

# Import hardware workload management components
from distributed_testing.hardware_workload_management import (
    HardwareWorkloadManager, WorkloadProfile, WorkloadType, WorkloadProfileMetric,
    create_workload_profile, HardwareTaxonomy
)

# Import hardware-aware scheduler
from distributed_testing.hardware_aware_scheduler import HardwareAwareScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("load_balancer_integration_example")


def create_sample_worker_capabilities(worker_id: str, worker_type: str = "generic") -> WorkerCapabilities:
    """
    Create sample worker capabilities for demonstration.
    
    Args:
        worker_id: Worker identifier
        worker_type: Type of worker (generic, gpu, tpu, browser, mobile)
        
    Returns:
        WorkerCapabilities instance
    """
    # Base worker capabilities
    capabilities = WorkerCapabilities(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        hardware_specs={
            "cpu": {"model": "Intel Core i7", "cores": 8, "threads": 16},
            "memory": {"total_gb": 32, "type": "DDR4"},
            "os": {"name": "Linux", "version": "Ubuntu 22.04"}
        },
        software_versions={
            "python": "3.9.5",
            "pytorch": "2.0.0",
            "tensorflow": "2.10.0",
            "cuda": "11.7"
        },
        supported_backends=["cpu", "pytorch", "tensorflow"],
        network_bandwidth=1000.0,
        storage_capacity=512.0,
        available_accelerators={},
        available_memory=28.0,
        available_disk=450.0,
        cpu_cores=8,
        cpu_threads=16
    )
    
    # Customize based on worker type
    if worker_type == "gpu":
        capabilities.supported_backends.extend(["cuda", "gpu"])
        capabilities.available_accelerators["gpu"] = 1
        capabilities.hardware_specs["gpu"] = {
            "model": "NVIDIA RTX 3080",
            "memory": "12GB",
            "cuda_cores": 8704,
            "tensor_cores": 272
        }
    
    elif worker_type == "tpu":
        capabilities.supported_backends.extend(["tpu", "tensorflow"])
        capabilities.available_accelerators["tpu"] = 1
        capabilities.hardware_specs["tpu"] = {
            "model": "Google TPU v3",
            "cores": 8,
            "memory": "16GB"
        }
    
    elif worker_type == "browser":
        capabilities.supported_backends.extend(["webgpu", "webnn"])
        capabilities.hardware_specs["browser"] = {
            "name": "Chrome",
            "version": "110.0",
            "webgpu": True,
            "webnn": True
        }
    
    elif worker_type == "mobile":
        capabilities.supported_backends.extend(["npu", "qnn"])
        capabilities.available_accelerators["npu"] = 1
        capabilities.hardware_specs["mobile"] = {
            "chipset": "Qualcomm Snapdragon 8 Gen 2",
            "npu": "Hexagon 690",
            "gpu": "Adreno 740"
        }
        capabilities.available_memory = 8.0
        capabilities.storage_capacity = 128.0
        capabilities.cpu_cores = 4
        capabilities.cpu_threads = 8
    
    return capabilities


def create_sample_test_requirements(test_id: str, test_type: str, model_id: str) -> TestRequirements:
    """
    Create sample test requirements for demonstration.
    
    Args:
        test_id: Test identifier
        test_type: Type of test (vision, nlp, audio, etc.)
        model_id: Identifier of the model to test
        
    Returns:
        TestRequirements instance
    """
    # Determine requirements based on test type
    memory_req = 1.0  # Default 1GB
    backend_req = None
    duration = 60.0  # Default 60 seconds
    priority = 3  # Default medium priority
    
    if "vision" in test_type.lower():
        memory_req = 2.0  # Vision models need more memory
        backend_req = "gpu" if random.random() > 0.3 else None
        duration = 45.0
    
    elif "nlp" in test_type.lower() or "text" in test_type.lower():
        memory_req = 4.0  # NLP models need even more memory
        if "large" in model_id.lower():
            memory_req = 8.0  # Large language models need a lot of memory
            duration = 120.0
        backend_req = "gpu" if random.random() > 0.2 else None
    
    elif "audio" in test_type.lower() or "speech" in test_type.lower():
        memory_req = 3.0
        duration = 90.0
        backend_req = "cpu" if random.random() > 0.7 else None
    
    # Create test requirements
    requirements = TestRequirements(
        test_id=test_id,
        model_id=model_id,
        model_family=test_type.split("_")[0] if "_" in test_type else test_type,
        test_type=test_type,
        minimum_memory=memory_req,
        required_memory_limit=32.0,
        preferred_backend=backend_req,
        required_backend=None,
        expected_duration=duration,
        priority=priority,
        required_accelerators={},
        timeout=duration * 5.0
    )
    
    # Add accelerator requirements for certain test types
    if "gpu" in test_type.lower() or backend_req == "gpu":
        requirements.required_accelerators["gpu"] = 1
    
    if "tpu" in test_type.lower():
        requirements.required_accelerators["tpu"] = 1
    
    if "npu" in test_type.lower() or "mobile" in test_type.lower():
        requirements.required_accelerators["npu"] = 1
    
    # Add custom properties for multi-device execution if appropriate
    if "distributed" in test_type.lower() or "multi_device" in test_type.lower():
        requirements.custom_properties = {
            "is_shardable": True,
            "min_shards": 2,
            "max_shards": 4,
            "allocation_strategy": "sharded"
        }
    
    elif "replicated" in test_type.lower() or "fault_tolerant" in test_type.lower():
        requirements.custom_properties = {
            "is_shardable": True,
            "min_shards": 2,
            "max_shards": 3,
            "allocation_strategy": "replicated"
        }
    
    else:
        requirements.custom_properties = {
            "is_shardable": False,
            "min_shards": 1,
            "max_shards": 1,
            "allocation_strategy": "single"
        }
    
    return requirements


def simulate_worker_load(worker_id: str, current_tests: List[str]) -> WorkerLoad:
    """
    Simulate worker load for demonstration.
    
    Args:
        worker_id: Worker identifier
        current_tests: List of test IDs currently assigned to the worker
        
    Returns:
        WorkerLoad instance
    """
    test_count = len(current_tests)
    
    # Base load increases with number of tests
    base_load = min(0.8, 0.1 + (test_count * 0.15))
    
    # Random variation
    variation = random.uniform(-0.1, 0.1)
    
    # Calculate usage percentages
    cpu_usage = min(95.0, 20.0 + (base_load * 80.0) + (variation * 10.0))
    memory_usage = min(95.0, 30.0 + (base_load * 70.0) + (variation * 5.0))
    gpu_usage = min(95.0, 10.0 + (base_load * 85.0) + (variation * 15.0))
    io_usage = min(95.0, 5.0 + (base_load * 30.0) + (variation * 10.0))
    network_usage = min(95.0, 10.0 + (base_load * 40.0) + (variation * 10.0))
    
    # Create worker load
    load = WorkerLoad(
        worker_id=worker_id,
        active_tests=test_count,
        queued_tests=0,
        cpu_utilization=cpu_usage,
        memory_utilization=memory_usage,
        gpu_utilization=gpu_usage,
        io_utilization=io_usage,
        network_utilization=network_usage,
        queue_depth=0,
        active_test_ids=set(current_tests),
        reserved_memory=test_count * 2.0,  # Assume each test uses ~2GB
        reserved_accelerators={}
    )
    
    # Add thermal state based on load
    if base_load > 0.7:
        # High load, start cooling
        load.cooling_state = True
        load.cooling_until = time.time() + 120.0
        load.performance_level = 0.8
    elif base_load < 0.2 and test_count > 0:
        # Low load with tests, start warming
        load.warming_state = True
        load.warming_until = time.time() + 60.0
        load.performance_level = 0.9
    
    return load


def run_load_balancer_example():
    """Run the load balancer integration example."""
    # Create hardware taxonomy
    hardware_taxonomy = HardwareTaxonomy()
    
    # Create hardware workload manager
    workload_manager = HardwareWorkloadManager(hardware_taxonomy)
    
    # Create hardware-aware scheduler
    scheduler = HardwareAwareScheduler(workload_manager, hardware_taxonomy)
    
    # Create load balancer service
    load_balancer = LoadBalancerService()
    
    # Set custom scheduler
    load_balancer.default_scheduler = scheduler
    
    # Start services
    workload_manager.start()
    load_balancer.start()
    
    # Register workers
    worker_types = {
        "worker1": "generic",
        "worker2": "gpu",
        "worker3": "tpu",
        "worker4": "browser",
        "worker5": "mobile"
    }
    
    workers = {}
    worker_assignments = {}
    
    for worker_id, worker_type in worker_types.items():
        capabilities = create_sample_worker_capabilities(worker_id, worker_type)
        load_balancer.register_worker(worker_id, capabilities)
        workers[worker_id] = capabilities
        worker_assignments[worker_id] = []
        logger.info(f"Registered worker {worker_id} of type {worker_type}")
    
    # Create test types
    test_types = [
        "vision_classification", 
        "nlp_text_classification", 
        "audio_speech_recognition",
        "vision_object_detection",
        "nlp_large_language_model",
        "nlp_text_embedding",
        "audio_speech_synthesis",
        "vision_segmentation",
        "distributed_nlp_model",
        "replicated_vision_model",
        "multi_device_training",
        "mobile_nlp_inference",
        "browser_vision_inference",
        "tpu_accelerated_training"
    ]
    
    # Create model IDs
    model_ids = {
        "vision": ["vit-base", "resnet50", "yolov5", "swin-transformer", "efficientnet"],
        "nlp": ["bert-base", "t5-large", "roberta", "gpt2", "llama-7b"],
        "audio": ["whisper-small", "wav2vec2", "hubert", "conformer", "encodec"]
    }
    
    # Submit tests
    test_count = 20
    submitted_tests = []
    test_to_type_map = {}
    
    logger.info(f"Submitting {test_count} tests...")
    
    for i in range(test_count):
        # Select random test type
        test_type = random.choice(test_types)
        
        # Determine model category
        model_category = "vision"
        if "nlp" in test_type.lower() or "text" in test_type.lower():
            model_category = "nlp"
        elif "audio" in test_type.lower() or "speech" in test_type.lower():
            model_category = "audio"
        
        # Select random model ID from appropriate category
        model_id = random.choice(model_ids[model_category])
        
        # Create test ID
        test_id = f"test_{i+1}_{uuid.uuid4().hex[:8]}"
        
        # Create test requirements
        requirements = create_sample_test_requirements(test_id, test_type, model_id)
        
        # Submit test
        load_balancer.submit_test(requirements)
        submitted_tests.append(test_id)
        test_to_type_map[test_id] = test_type
        
        logger.info(f"Submitted test {test_id} of type {test_type} with model {model_id}")
        
        # Small delay to simulate realistic submission pattern
        time.sleep(0.1)
    
    # Wait a bit for scheduling to complete
    logger.info("Waiting for scheduling to complete...")
    time.sleep(2)
    
    # Process assignments
    logger.info("Processing assignments...")
    
    # First update loads based on initial assignments
    for worker_id in workers:
        assignments = load_balancer.get_worker_assignments(worker_id)
        assigned_test_ids = [a.test_id for a in assignments]
        worker_assignments[worker_id] = assigned_test_ids
        
        # Update worker load
        load = simulate_worker_load(worker_id, assigned_test_ids)
        load_balancer.update_worker_load(worker_id, load)
    
    # Check assignments and simulate execution
    completed_tests = []
    
    for test_id in submitted_tests:
        assignment = load_balancer.get_assignment(test_id)
        
        if assignment:
            worker_id = assignment.worker_id
            test_type = test_to_type_map.get(test_id, "unknown")
            
            logger.info(f"Test {test_id} ({test_type}) assigned to worker {worker_id}")
            
            # Mark as running
            load_balancer.update_assignment_status(test_id, "running")
            
            # Simulate execution (in a real system, this would happen asynchronously)
            # Here we just instantly mark it as completed for demonstration
            success = random.random() > 0.1  # 90% success rate
            status = "completed" if success else "failed"
            result = {"output": f"Test result for {test_id}", "success": success}
            
            # Add a small delay to simulate execution time
            time.sleep(0.2)
            
            # Mark as completed
            load_balancer.update_assignment_status(test_id, status, result)
            
            if success:
                completed_tests.append(test_id)
            
            # Update worker load after test completion
            worker_assignments[worker_id].remove(test_id)
            load = simulate_worker_load(worker_id, worker_assignments[worker_id])
            load_balancer.update_worker_load(worker_id, load)
    
    # Wait a bit to allow processing
    time.sleep(1)
    
    # Print summary
    logger.info("\n=== Summary ===")
    logger.info(f"Submitted tests: {len(submitted_tests)}")
    logger.info(f"Completed tests: {len(completed_tests)}")
    logger.info(f"Success rate: {len(completed_tests) / len(submitted_tests) * 100:.1f}%")
    
    # Print worker assignments
    logger.info("\n=== Worker Assignments ===")
    for worker_id, worker_type in worker_types.items():
        assignments = load_balancer.get_worker_assignments(worker_id)
        logger.info(f"Worker {worker_id} ({worker_type}): {len(assignments)} active tests")
    
    # Stop services
    workload_manager.stop()
    load_balancer.stop()
    
    logger.info("Example completed")


if __name__ == "__main__":
    run_load_balancer_example()