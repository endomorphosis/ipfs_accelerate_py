"""
Test script for the heterogeneous scheduler and enhanced hardware detection.

This script demonstrates the capabilities of the enhanced hardware detection
and heterogeneous scheduling system by simulating a distributed test environment
with multiple worker nodes having different hardware profiles.
"""

import argparse
import json
import logging
import os
import random
import time
import uuid
from typing import Dict, List, Any, Optional

from .enhanced_hardware_detector import EnhancedHardwareDetector, get_enhanced_hardware_info
from .hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    AcceleratorFeature,
    HardwareCapabilityProfile,
    HardwareTaxonomy
)
from .heterogeneous_scheduler import (
    HeterogeneousScheduler,
    WorkloadProfile,
    TestTask,
    WorkerState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_hardware_profile(worker_id: str, profile_type: str) -> Dict[str, Any]:
    """
    Create a sample hardware profile for a simulated worker.
    
    Args:
        worker_id: ID of the worker
        profile_type: Type of profile (cpu, gpu, mixed, browser, mobile)
        
    Returns:
        Dict with hardware information
    """
    # Base profile
    profile = {
        "worker_id": worker_id,
        "hardware_profiles": [],
        "platform_info": {
            "os": "Linux",
            "os_version": "5.15.0",
            "python_version": "3.10.12",
            "architecture": "x86_64"
        },
        "browser_info": {},
        "cpu_info": {},
        "memory_info": {},
        "gpu_info": [],
        "specialized_hardware": {
            "tpu": [],
            "npu": [],
            "fpga": [],
            "dsp": []
        }
    }
    
    # Add CPU profile
    cpu_profile = {
        "hardware_class": "cpu",
        "architecture": "x86_64",
        "vendor": "intel",
        "model_name": "Intel Core i9-13900K",
        "supported_backends": ["pytorch", "tensorflow", "onnx", "openvino"],
        "supported_precisions": ["fp32", "fp16", "int8"],
        "features": ["avx", "avx2", "avx512", "simd"],
        "memory_total_gb": 64.0,
        "memory_available_gb": 48.0,
        "compute_units": 32,
        "clock_speed_mhz": 5200,
        "performance_profile": {
            "fp32_matmul": 16640.0,
            "fp32_conv": 8320.0,
            "int8_matmul": 49920.0,
            "int8_conv": 24960.0
        }
    }
    profile["hardware_profiles"].append(cpu_profile)
    
    # Add profiles based on type
    if profile_type == "gpu" or profile_type == "mixed":
        # NVIDIA GPU profile
        gpu_profile = {
            "hardware_class": "gpu",
            "architecture": "gpu_cuda",
            "vendor": "nvidia",
            "model_name": "NVIDIA RTX 4090",
            "supported_backends": ["pytorch", "tensorflow", "onnx", "cuda", "tensorrt"],
            "supported_precisions": ["fp32", "fp16", "int8", "int4"],
            "features": ["tensor_cores", "ray_tracing", "compute_shaders", "simt"],
            "memory_total_gb": 24.0,
            "memory_available_gb": 20.0,
            "compute_units": 128,
            "clock_speed_mhz": 2520,
            "performance_profile": {
                "fp32_matmul": 161280.0,
                "fp32_conv": 96768.0,
                "fp16_matmul": 322560.0,
                "fp16_conv": 193536.0,
                "int8_matmul": 2322432.0,
                "int8_conv": 1548288.0
            }
        }
        profile["hardware_profiles"].append(gpu_profile)
        profile["gpu_info"].append({
            "type": "cuda",
            "name": "NVIDIA RTX 4090",
            "compute_capability": "8.9",
            "compute_units": 128,
            "memory_total": 24 * 1024 * 1024 * 1024,
            "memory_available": 20 * 1024 * 1024 * 1024,
            "clock_rate_mhz": 2520,
            "vendor": "nvidia",
            "has_tensor_cores": True,
            "has_ray_tracing": True
        })
    
    if profile_type == "browser" or profile_type == "mixed":
        # Browser profiles
        profile["browser_info"] = {
            "chrome": {"available": True, "webgpu": True, "webnn": True, "version": "121.0.6167.140"},
            "firefox": {"available": True, "webgpu": True, "webnn": False, "version": "121.0"},
            "edge": {"available": True, "webgpu": True, "webnn": True, "version": "121.0.2277.128"},
            "safari": {"available": False, "webgpu": False, "webnn": False, "version": None}
        }
        
        # Chrome browser profile
        chrome_profile = {
            "hardware_class": "hybrid",
            "architecture": "gpu_webgpu",
            "vendor": "google",
            "model_name": "Chrome Browser",
            "supported_backends": ["webgpu", "webnn"],
            "supported_precisions": ["fp32", "fp16"],
            "features": ["compute_shaders"],
            "memory_total_gb": 4.0,
            "memory_available_gb": 2.0,
            "compute_units": 8,
            "clock_speed_mhz": 1000,
            "performance_profile": {
                "fp32_matmul": 80.0,
                "fp32_conv": 64.0,
                "fp16_matmul": 120.0,
                "fp16_conv": 96.0,
                "int8_matmul": 140.0,
                "int8_conv": 105.0,
                "fp32_audio": 84.0
            }
        }
        profile["hardware_profiles"].append(chrome_profile)
        
        # Firefox browser profile
        firefox_profile = {
            "hardware_class": "hybrid",
            "architecture": "gpu_webgpu",
            "vendor": "other",
            "model_name": "Firefox Browser",
            "supported_backends": ["webgpu"],
            "supported_precisions": ["fp32", "fp16"],
            "features": ["compute_shaders"],
            "memory_total_gb": 4.0,
            "memory_available_gb": 2.0,
            "compute_units": 4,
            "clock_speed_mhz": 1000,
            "performance_profile": {
                "fp32_matmul": 64.0,
                "fp32_conv": 51.2,
                "fp16_matmul": 96.0,
                "fp16_conv": 76.8,
                "fp32_audio": 120.0
            }
        }
        profile["hardware_profiles"].append(firefox_profile)
    
    if profile_type == "mobile" or profile_type == "mixed":
        # NPU profile for mobile
        npu_profile = {
            "hardware_class": "npu",
            "architecture": "npu_qualcomm",
            "vendor": "qualcomm",
            "model_name": "Qualcomm Hexagon NPU",
            "supported_backends": ["onnx", "qnn"],
            "supported_precisions": ["fp32", "fp16", "int8", "int4"],
            "features": ["neural_engine", "quantization"],
            "memory_total_gb": 8.0,
            "memory_available_gb": 6.0,
            "compute_units": 8,
            "clock_speed_mhz": 1000,
            "performance_profile": {
                "fp32_matmul": 2400.0,
                "fp32_conv": 1600.0,
                "fp16_matmul": 4800.0,
                "fp16_conv": 3200.0,
                "int8_matmul": 9600.0,
                "int8_conv": 6400.0,
                "int4_matmul": 16000.0,
                "int4_conv": 12000.0
            }
        }
        profile["hardware_profiles"].append(npu_profile)
        profile["specialized_hardware"]["npu"].append({
            "type": "npu",
            "vendor": "qualcomm",
            "name": "Qualcomm Hexagon NPU",
            "compute_units": 8,
            "memory_total": 8 * 1024 * 1024 * 1024,
            "memory_available": 6 * 1024 * 1024 * 1024,
            "clock_rate_mhz": 1000,
            "has_quantization": True,
            "tdp_w": 5.0
        })
    
    if profile_type == "tpu" or profile_type == "mixed":
        # TPU profile
        tpu_profile = {
            "hardware_class": "tpu",
            "architecture": "tpu",
            "vendor": "google",
            "model_name": "Google TPU v4",
            "supported_backends": ["tensorflow", "jax"],
            "supported_precisions": ["fp32", "fp16", "bf16", "int8"],
            "features": ["tensor_cores", "quantization", "sparsity"],
            "memory_total_gb": 32.0,
            "memory_available_gb": 28.0,
            "compute_units": 2,
            "clock_speed_mhz": 1100,
            "performance_profile": {
                "fp32_matmul": 88000.0,
                "fp32_conv": 44000.0,
                "fp16_matmul": 176000.0,
                "fp16_conv": 88000.0,
                "int8_matmul": 352000.0,
                "int8_conv": 176000.0
            }
        }
        profile["hardware_profiles"].append(tpu_profile)
        profile["specialized_hardware"]["tpu"].append({
            "type": "tpu",
            "vendor": "google",
            "name": "Google TPU v4",
            "compute_units": 2,
            "memory_total": 32 * 1024 * 1024 * 1024,
            "memory_available": 28 * 1024 * 1024 * 1024,
            "clock_rate_mhz": 1100,
            "has_quantization": True,
            "tdp_w": 175.0
        })
    
    # Add optimal hardware specializations
    profile["optimal_hardware"] = {}
    
    if profile_type == "gpu" or profile_type == "mixed":
        profile["optimal_hardware"]["nlp"] = {
            "hardware_class": "gpu",
            "architecture": "gpu_cuda",
            "vendor": "nvidia",
            "model_name": "NVIDIA RTX 4090",
            "effectiveness_score": 0.95,
            "supported_backends": ["pytorch", "tensorflow", "onnx", "cuda", "tensorrt"],
            "supported_precisions": ["fp32", "fp16", "int8", "int4"],
            "features": ["tensor_cores", "compute_shaders", "simt"],
            "memory_total_gb": 24.0,
            "compute_units": 128,
            "performance_profile": {
                "fp32_matmul": 161280.0,
                "fp32_conv": 96768.0,
                "fp16_matmul": 322560.0,
                "fp16_conv": 193536.0,
                "int8_matmul": 2322432.0,
                "int8_conv": 1548288.0
            }
        }
        
        profile["optimal_hardware"]["vision"] = {
            "hardware_class": "gpu",
            "architecture": "gpu_cuda", 
            "vendor": "nvidia",
            "model_name": "NVIDIA RTX 4090",
            "effectiveness_score": 0.98,
            "supported_backends": ["pytorch", "tensorflow", "onnx", "cuda", "tensorrt"],
            "supported_precisions": ["fp32", "fp16", "int8", "int4"],
            "features": ["tensor_cores", "compute_shaders", "simt"],
            "memory_total_gb": 24.0,
            "compute_units": 128,
            "performance_profile": {
                "fp32_matmul": 161280.0,
                "fp32_conv": 96768.0,
                "fp16_matmul": 322560.0,
                "fp16_conv": 193536.0,
                "int8_matmul": 2322432.0,
                "int8_conv": 1548288.0
            }
        }
    
    if profile_type == "browser" or profile_type == "mixed":
        profile["optimal_hardware"]["audio"] = {
            "hardware_class": "hybrid",
            "architecture": "gpu_webgpu",
            "vendor": "other",
            "model_name": "Firefox Browser",
            "effectiveness_score": 0.92,
            "supported_backends": ["webgpu"],
            "supported_precisions": ["fp32", "fp16"],
            "features": ["compute_shaders"],
            "memory_total_gb": 4.0,
            "compute_units": 4,
            "performance_profile": {
                "fp32_matmul": 64.0,
                "fp32_conv": 51.2,
                "fp16_matmul": 96.0,
                "fp16_conv": 76.8,
                "fp32_audio": 120.0
            }
        }
    
    if profile_type == "mobile" or profile_type == "mixed":
        profile["optimal_hardware"]["edge_vision"] = {
            "hardware_class": "npu",
            "architecture": "npu_qualcomm",
            "vendor": "qualcomm",
            "model_name": "Qualcomm Hexagon NPU",
            "effectiveness_score": 0.90,
            "supported_backends": ["onnx", "qnn"],
            "supported_precisions": ["fp32", "fp16", "int8", "int4"],
            "features": ["neural_engine", "quantization"],
            "memory_total_gb": 8.0,
            "compute_units": 8,
            "performance_profile": {
                "fp32_matmul": 2400.0,
                "fp32_conv": 1600.0,
                "fp16_matmul": 4800.0,
                "fp16_conv": 3200.0,
                "int8_matmul": 9600.0,
                "int8_conv": 6400.0,
                "int4_matmul": 16000.0,
                "int4_conv": 12000.0
            }
        }
    
    if profile_type == "tpu" or profile_type == "mixed":
        profile["optimal_hardware"]["large_batch_nlp"] = {
            "hardware_class": "tpu",
            "architecture": "tpu",
            "vendor": "google",
            "model_name": "Google TPU v4",
            "effectiveness_score": 0.96,
            "supported_backends": ["tensorflow", "jax"],
            "supported_precisions": ["fp32", "fp16", "bf16", "int8"],
            "features": ["tensor_cores", "quantization", "sparsity"],
            "memory_total_gb": 32.0,
            "compute_units": 2,
            "performance_profile": {
                "fp32_matmul": 88000.0,
                "fp32_conv": 44000.0,
                "fp16_matmul": 176000.0,
                "fp16_conv": 88000.0,
                "int8_matmul": 352000.0,
                "int8_conv": 176000.0
            }
        }
    
    return profile


def create_workload_profile(workload_type: str) -> WorkloadProfile:
    """
    Create a sample workload profile for testing.
    
    Args:
        workload_type: Type of workload (nlp, vision, audio, etc.)
        
    Returns:
        WorkloadProfile object
    """
    if workload_type == "nlp":
        return WorkloadProfile(
            workload_type="nlp",
            operation_types=["matmul", "attention", "softmax"],
            precision_types=["fp16", "int8"],
            min_memory_gb=4.0,
            preferred_memory_gb=8.0,
            required_features=["tensor_cores"],
            required_backends=["pytorch", "onnx"],
            batch_size_options=[1, 4, 8, 16, 32, 64],
            optimal_batch_size=16,
            priority=2,
            max_execution_time_ms=5000,
            is_latency_sensitive=False,
            is_throughput_sensitive=True
        )
    elif workload_type == "vision":
        return WorkloadProfile(
            workload_type="vision",
            operation_types=["conv", "matmul", "pooling"],
            precision_types=["fp16", "int8"],
            min_memory_gb=2.0,
            preferred_memory_gb=6.0,
            required_features=["tensor_cores"],
            required_backends=["pytorch", "onnx"],
            batch_size_options=[1, 8, 16, 32],
            optimal_batch_size=32,
            priority=2,
            max_execution_time_ms=2000,
            is_latency_sensitive=False,
            is_throughput_sensitive=True
        )
    elif workload_type == "audio":
        return WorkloadProfile(
            workload_type="audio",
            operation_types=["conv1d", "matmul", "fft"],
            precision_types=["fp32", "fp16"],
            min_memory_gb=2.0,
            preferred_memory_gb=4.0,
            required_features=[],
            required_backends=["webgpu"],
            batch_size_options=[1, 2, 4, 8],
            optimal_batch_size=4,
            priority=2,
            max_execution_time_ms=3000,
            is_latency_sensitive=True,
            is_throughput_sensitive=False
        )
    elif workload_type == "edge_vision":
        return WorkloadProfile(
            workload_type="edge_vision",
            operation_types=["conv", "matmul", "pooling"],
            precision_types=["int8", "int4"],
            min_memory_gb=0.5,
            preferred_memory_gb=1.0,
            required_features=["quantization"],
            required_backends=["onnx"],
            batch_size_options=[1, 2, 4],
            optimal_batch_size=1,
            priority=1,
            max_execution_time_ms=1000,
            is_latency_sensitive=True,
            is_throughput_sensitive=False,
            is_power_sensitive=True
        )
    elif workload_type == "large_batch_nlp":
        return WorkloadProfile(
            workload_type="large_batch_nlp",
            operation_types=["matmul", "attention", "softmax"],
            precision_types=["fp16", "bf16"],
            min_memory_gb=16.0,
            preferred_memory_gb=24.0,
            required_features=["tensor_cores"],
            required_backends=["tensorflow", "jax"],
            batch_size_options=[32, 64, 128, 256],
            optimal_batch_size=128,
            priority=3,
            max_execution_time_ms=10000,
            is_latency_sensitive=False,
            is_throughput_sensitive=True
        )
    else:
        # Default generic workload
        return WorkloadProfile(
            workload_type=workload_type,
            operation_types=["matmul", "conv"],
            precision_types=["fp32", "fp16"],
            min_memory_gb=1.0,
            preferred_memory_gb=2.0,
            required_features=[],
            required_backends=[],
            batch_size_options=[1, 2, 4, 8],
            optimal_batch_size=4,
            priority=1,
            max_execution_time_ms=5000,
            is_latency_sensitive=False,
            is_throughput_sensitive=False
        )


def create_test_task(workload_type: str, batch_size: Optional[int] = None, priority: Optional[int] = None) -> TestTask:
    """
    Create a test task for a specific workload type.
    
    Args:
        workload_type: Type of workload (nlp, vision, audio, etc.)
        batch_size: Optional batch size override
        priority: Optional priority override
        
    Returns:
        TestTask object
    """
    # Create workload profile
    profile = create_workload_profile(workload_type)
    
    # Override batch size if provided
    if batch_size is not None:
        if batch_size in profile.batch_size_options:
            profile.optimal_batch_size = batch_size
        else:
            # Add to options and set as optimal
            profile.batch_size_options.append(batch_size)
            profile.optimal_batch_size = batch_size
    
    # Override priority if provided
    if priority is not None:
        profile.priority = priority
    
    # Create inputs based on workload type
    inputs = {}
    if workload_type == "nlp":
        inputs = {
            "text": "This is a sample text for natural language processing.",
            "max_length": 64,
            "return_attention": True
        }
    elif workload_type == "vision":
        inputs = {
            "image_size": [224, 224],
            "normalize": True,
            "batch_size": profile.optimal_batch_size
        }
    elif workload_type == "audio":
        inputs = {
            "audio_length": 10.0,
            "sample_rate": 16000,
            "channels": 1
        }
    else:
        inputs = {
            "batch_size": profile.optimal_batch_size,
            "generic_param": True
        }
    
    # Create task
    return TestTask(
        task_id=f"{workload_type}_{uuid.uuid4().hex[:8]}",
        workload_profile=profile,
        inputs=inputs,
        batch_size=profile.optimal_batch_size,
        timeout_ms=profile.max_execution_time_ms,
        priority=profile.priority
    )


def simulate_task_execution(worker_state: WorkerState, task: TestTask) -> Dict[str, Any]:
    """
    Simulate the execution of a task on a worker.
    
    Args:
        worker_state: State of the worker executing the task
        task: Task to execute
        
    Returns:
        Dict with execution results
    """
    # Choose the most suitable hardware for this workload
    hardware_class = None
    hardware_model = None
    execution_time_ms = None
    
    workload_type = task.workload_profile.workload_type
    
    # First check workload specializations
    if workload_type in worker_state.workload_specializations:
        for profile in worker_state.hardware_profiles:
            if (profile.get("hardware_class") in task.workload_profile.hardware_class_affinity and
                task.workload_profile.hardware_class_affinity[profile.get("hardware_class")] > 0.5):
                hardware_class = profile.get("hardware_class")
                hardware_model = profile.get("model_name")
                
                # Simulate execution time based on performance profile
                # First operation in operation_types with first precision in precision_types
                if task.workload_profile.operation_types and task.workload_profile.precision_types:
                    op_type = task.workload_profile.operation_types[0]
                    precision = task.workload_profile.precision_types[0]
                    perf_key = f"{precision}_{op_type}"
                    
                    if (perf_key in profile.get("performance_profile", {}) and 
                        profile["performance_profile"][perf_key] > 0):
                        # Higher performance means lower execution time
                        # This is a simplified model
                        base_execution_time = 1000 * 1000 / profile["performance_profile"][perf_key]
                        
                        # Scale by batch size
                        batch_factor = task.batch_size / 8 if task.batch_size > 0 else 1
                        
                        # Add some random variation
                        execution_time_ms = base_execution_time * batch_factor * random.uniform(0.8, 1.2)
                        break
    
    # If no suitable hardware found based on specialization, use a default
    if hardware_class is None:
        # Just use the first hardware profile
        if worker_state.hardware_profiles:
            profile = worker_state.hardware_profiles[0]
            hardware_class = profile.get("hardware_class")
            hardware_model = profile.get("model_name")
            execution_time_ms = random.uniform(500, 5000)  # Random execution time
    
    # Simulate success most of the time, occasional failure
    success = random.random() > 0.05  # 95% success rate
    
    # Create result
    if success:
        result = {
            "status": "success",
            "hardware_class": hardware_class,
            "hardware_model": hardware_model,
            "execution_time_ms": execution_time_ms,
            "workload_type": workload_type,
            "batch_size": task.batch_size,
            "output": {
                "result_shape": [task.batch_size, 768] if workload_type == "nlp" else [task.batch_size, 1000],
                "success": True,
                "metrics": {
                    "latency_ms": execution_time_ms,
                    "throughput_items_per_sec": task.batch_size / (execution_time_ms / 1000) if execution_time_ms else None
                }
            }
        }
    else:
        result = {
            "status": "error",
            "hardware_class": hardware_class,
            "hardware_model": hardware_model,
            "error": "Simulated task failure for testing",
            "workload_type": workload_type,
            "batch_size": task.batch_size
        }
    
    # Simulate execution time
    if execution_time_ms:
        sleep_time = min(execution_time_ms / 1000, 0.1)  # Don't sleep too long in simulation
        time.sleep(sleep_time)
    
    return result


def run_simulation(
    num_workers: int = 3,
    num_tasks: int = 50,
    scheduler_strategy: str = "adaptive",
    visualization: bool = True,
    output_file: Optional[str] = None
):
    """
    Run a simulation of the heterogeneous scheduler.
    
    Args:
        num_workers: Number of simulated workers
        num_tasks: Number of tasks to schedule
        scheduler_strategy: Scheduling strategy (adaptive, resource_aware, performance_aware, round_robin)
        visualization: Whether to visualize the results
        output_file: File to save the results
    """
    # Create scheduler
    scheduler = HeterogeneousScheduler(
        strategy=scheduler_strategy,
        thermal_management=True,
        enable_workload_learning=True
    )
    
    # Create workers with different hardware profiles
    worker_types = ["cpu", "gpu", "browser", "mobile", "tpu", "mixed"]
    for i in range(num_workers):
        worker_id = f"worker_{i+1}"
        worker_type = worker_types[i % len(worker_types)]
        worker_profile = create_sample_hardware_profile(worker_id, worker_type)
        
        # Register worker
        scheduler.register_worker(worker_id, worker_profile)
        
        # Log worker registration
        logger.info(f"Registered worker {worker_id} with profile type {worker_type}")
        logger.info(f"  Hardware classes: {scheduler.workers[worker_id].hardware_classes}")
        logger.info(f"  Backends: {scheduler.workers[worker_id].supported_backends}")
        logger.info(f"  Workload specializations: {scheduler.workers[worker_id].workload_specializations}")
    
    # Create and submit tasks
    workload_types = ["nlp", "vision", "audio", "edge_vision", "large_batch_nlp", "generic"]
    submitted_tasks = []
    
    for i in range(num_tasks):
        workload_type = workload_types[i % len(workload_types)]
        
        # Occasional high-priority tasks
        priority = 3 if random.random() < 0.1 else None
        
        # Create task
        task = create_test_task(workload_type, priority=priority)
        
        # Submit task
        scheduler.submit_task(task)
        submitted_tasks.append(task)
        
        # Log task submission
        logger.info(f"Submitted task {task.task_id} of type {task.workload_profile.workload_type} with priority {task.priority}")
    
    # Run scheduler iterations until all tasks are complete or failed
    iteration = 0
    max_iterations = 20
    
    while (scheduler.pending_tasks or scheduler.scheduled_tasks) and iteration < max_iterations:
        # Schedule pending tasks
        scheduler.schedule_tasks()
        
        # Log scheduled tasks
        logger.info(f"Iteration {iteration}: Scheduled {scheduler.stats['tasks_scheduled']} tasks, {len(scheduler.pending_tasks)} pending")
        
        # Simulate workers executing tasks
        for worker_id, worker in scheduler.workers.items():
            # Skip offline workers
            if worker.status == "offline" or worker.status == "cooling":
                continue
            
            # Process active tasks
            active_tasks = worker.active_tasks.copy()  # Copy to avoid modification during iteration
            for task in active_tasks:
                # Simulate task execution
                result = simulate_task_execution(worker, task)
                
                # Report completion or failure
                if result["status"] == "success":
                    scheduler.report_task_completion(
                        worker_id, 
                        task.task_id, 
                        result["output"], 
                        {"hardware_class": result["hardware_class"], "hardware_model": result["hardware_model"]}
                    )
                    logger.info(f"Task {task.task_id} completed on {worker_id} in {result.get('execution_time_ms', 'unknown')}ms")
                else:
                    scheduler.report_task_failure(worker_id, task.task_id, result["error"])
                    logger.info(f"Task {task.task_id} failed on {worker_id}: {result['error']}")
            
            # Update thermal state
            worker.update_thermal_state()
        
        # Perform load balancing every few iterations
        if iteration % 3 == 0:
            scheduler.perform_load_balancing()
        
        # Check worker heartbeats
        scheduler.check_worker_heartbeats(timeout_seconds=10.0)
        
        # Update iteration counter
        iteration += 1
        
        # Small delay between iterations
        time.sleep(0.1)
    
    # Print final statistics
    print(f"\n===== Simulation Complete =====")
    print(f"- Strategy: {scheduler_strategy}")
    print(f"- Workers: {num_workers}")
    print(f"- Tasks: {num_tasks}")
    print(f"- Iterations: {iteration}")
    print(f"- Tasks completed: {scheduler.stats['tasks_completed']}")
    print(f"- Tasks failed: {scheduler.stats['tasks_failed']}")
    print(f"- Tasks pending: {len(scheduler.pending_tasks)}")
    print(f"- Average queue time: {scheduler.stats['avg_queue_time_ms']:.2f}ms")
    print(f"- Average execution time: {scheduler.stats['avg_execution_time_ms']:.2f}ms")
    
    # Workload performance by hardware class
    print(f"\n===== Workload Performance by Hardware Class =====")
    workload_stats = {}
    for workload_type in workload_types:
        stats = scheduler.get_workload_stats(workload_type)
        if stats:
            workload_stats[workload_type] = stats
            print(f"- {workload_type}: {stats['completed_count']} completed, {stats['failed_count']} failed")
            if stats["performance_by_hardware"]:
                print(f"  Performance by hardware:")
                for hw_class, avg_time in stats["performance_by_hardware"].items():
                    print(f"  - {hw_class}: {avg_time:.2f}ms")
    
    # Generate visualization
    if visualization:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Plot 1: Task completion by worker
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # Worker task distribution
            worker_tasks = {}
            for worker_id, worker in scheduler.workers.items():
                worker_tasks[worker_id] = len(worker.completed_tasks)
            
            worker_ids = list(worker_tasks.keys())
            task_counts = list(worker_tasks.values())
            
            axs[0, 0].bar(worker_ids, task_counts)
            axs[0, 0].set_title('Tasks Completed by Worker')
            axs[0, 0].set_xlabel('Worker ID')
            axs[0, 0].set_ylabel('Number of Tasks')
            
            # Workload to hardware class mapping
            workload_hardware = {}
            for task in scheduler.completed_tasks:
                workload_type = task.workload_profile.workload_type
                hardware_class = task.executed_on_hardware_class
                
                if workload_type not in workload_hardware:
                    workload_hardware[workload_type] = {}
                
                if hardware_class not in workload_hardware[workload_type]:
                    workload_hardware[workload_type][hardware_class] = 0
                
                workload_hardware[workload_type][hardware_class] += 1
            
            # Create a stacked bar chart
            workload_types = list(workload_hardware.keys())
            hardware_classes = set()
            for workload in workload_hardware.values():
                hardware_classes.update(workload.keys())
            hardware_classes = list(hardware_classes)
            
            # Prepare data for stacked bar chart
            data = np.zeros((len(workload_types), len(hardware_classes)))
            for i, workload_type in enumerate(workload_types):
                for j, hardware_class in enumerate(hardware_classes):
                    data[i, j] = workload_hardware[workload_type].get(hardware_class, 0)
            
            # Create bottom positions for each bar segment
            bottoms = np.zeros(len(workload_types))
            for j in range(len(hardware_classes)):
                axs[0, 1].bar(workload_types, data[:, j], bottom=bottoms, label=hardware_classes[j])
                bottoms += data[:, j]
            
            axs[0, 1].set_title('Workload to Hardware Class Mapping')
            axs[0, 1].set_xlabel('Workload Type')
            axs[0, 1].set_ylabel('Number of Tasks')
            axs[0, 1].legend()
            
            # Execution time by hardware class
            hardware_execution_times = {}
            for task in scheduler.completed_tasks:
                hardware_class = task.executed_on_hardware_class
                if hardware_class not in hardware_execution_times:
                    hardware_execution_times[hardware_class] = []
                
                if task.execution_time_ms:
                    hardware_execution_times[hardware_class].append(task.execution_time_ms)
            
            # Calculate average execution time
            avg_execution_times = {}
            for hw_class, times in hardware_execution_times.items():
                avg_execution_times[hw_class] = sum(times) / len(times) if times else 0
            
            hw_classes = list(avg_execution_times.keys())
            avg_times = list(avg_execution_times.values())
            
            axs[1, 0].bar(hw_classes, avg_times)
            axs[1, 0].set_title('Average Execution Time by Hardware Class')
            axs[1, 0].set_xlabel('Hardware Class')
            axs[1, 0].set_ylabel('Execution Time (ms)')
            
            # Queue time distribution
            queue_times = [task.get_queue_time() * 1000 for task in scheduler.completed_tasks]
            
            axs[1, 1].hist(queue_times, bins=20)
            axs[1, 1].set_title('Queue Time Distribution')
            axs[1, 1].set_xlabel('Queue Time (ms)')
            axs[1, 1].set_ylabel('Number of Tasks')
            
            plt.tight_layout()
            
            # Save to file if specified
            if output_file:
                plt.savefig(output_file)
                print(f"Visualization saved to {output_file}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available, skipping visualization")
    
    # Save results to file if specified
    if output_file and output_file.endswith('.json'):
        results = {
            "strategy": scheduler_strategy,
            "workers": num_workers,
            "tasks": num_tasks,
            "iterations": iteration,
            "stats": scheduler.stats,
            "workload_stats": workload_stats
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {output_file}")
    
    # Return statistics
    return {
        "tasks_completed": scheduler.stats["tasks_completed"],
        "tasks_failed": scheduler.stats["tasks_failed"],
        "avg_queue_time_ms": scheduler.stats["avg_queue_time_ms"],
        "avg_execution_time_ms": scheduler.stats["avg_execution_time_ms"]
    }


def run_strategy_comparison(
    strategies: List[str] = ["adaptive", "resource_aware", "performance_aware", "round_robin"],
    num_workers: int = 5,
    num_tasks: int = 100,
    output_file: Optional[str] = None
):
    """
    Run a comparison of different scheduling strategies.
    
    Args:
        strategies: List of strategies to compare
        num_workers: Number of simulated workers
        num_tasks: Number of tasks to schedule
        output_file: File to save the results
    """
    results = {}
    
    for strategy in strategies:
        print(f"\n===== Running {strategy} strategy =====")
        strategy_result = run_simulation(
            num_workers=num_workers,
            num_tasks=num_tasks,
            scheduler_strategy=strategy,
            visualization=False
        )
        results[strategy] = strategy_result
    
    # Print comparison
    print("\n===== Strategy Comparison =====")
    print(f"{'Strategy':<20} {'Completed':<10} {'Failed':<10} {'Avg Queue (ms)':<15} {'Avg Exec (ms)':<15}")
    print("-" * 70)
    for strategy, result in results.items():
        print(f"{strategy:<20} {result['tasks_completed']:<10} {result['tasks_failed']:<10} {result['avg_queue_time_ms']:<15.2f} {result['avg_execution_time_ms']:<15.2f}")
    
    # Visualize comparison
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        strategies = list(results.keys())
        completed = [results[s]["tasks_completed"] for s in strategies]
        queue_times = [results[s]["avg_queue_time_ms"] for s in strategies]
        execution_times = [results[s]["avg_execution_time_ms"] for s in strategies]
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Tasks completed
        axs[0].bar(strategies, completed)
        axs[0].set_title('Tasks Completed by Strategy')
        axs[0].set_xlabel('Strategy')
        axs[0].set_ylabel('Number of Tasks')
        
        # Queue times
        axs[1].bar(strategies, queue_times)
        axs[1].set_title('Average Queue Time by Strategy')
        axs[1].set_xlabel('Strategy')
        axs[1].set_ylabel('Queue Time (ms)')
        
        # Execution times
        axs[2].bar(strategies, execution_times)
        axs[2].set_title('Average Execution Time by Strategy')
        axs[2].set_xlabel('Strategy')
        axs[2].set_ylabel('Execution Time (ms)')
        
        plt.tight_layout()
        
        # Save to file if specified
        if output_file:
            plt.savefig(output_file)
            print(f"Comparison visualization saved to {output_file}")
        else:
            plt.show()
    
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    
    # Save results to file if specified
    if output_file and output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Comparison results saved to {output_file}")
    
    return results


def test_actual_hardware_detection():
    """Test the actual hardware detection on the current system."""
    print("Testing actual hardware detection on current system...")
    
    # Create detector
    detector = EnhancedHardwareDetector()
    
    # Detect hardware
    profiles = detector.detect_hardware()
    
    # Print hardware information
    print(f"Detected {len(profiles)} hardware profiles")
    for i, profile in enumerate(profiles):
        print(f"\nProfile {i+1}:")
        print(f"- Hardware Class: {profile.hardware_class.value}")
        print(f"- Architecture: {profile.architecture.value}")
        print(f"- Vendor: {profile.vendor.value}")
        print(f"- Model: {profile.model_name}")
        print(f"- Memory: {profile.memory.total_bytes / (1024 * 1024 * 1024):.2f} GB")
        print(f"- Backends: {[backend.value for backend in profile.supported_backends]}")
        print(f"- Precisions: {[precision.value for precision in profile.supported_precisions]}")
        print(f"- Features: {[feature.value for feature in profile.features]}")
    
    # Get optimal hardware for workloads
    workloads = ["nlp", "vision", "audio"]
    for workload in workloads:
        optimal = detector.find_optimal_hardware_for_workload(workload)
        print(f"\nOptimal hardware for {workload}:")
        if optimal:
            print(f"- Hardware Class: {optimal['hardware_class']}")
            print(f"- Model: {optimal['model_name']}")
            print(f"- Effectiveness: {optimal['effectiveness_score']:.2f}")
        else:
            print("- No suitable hardware found")
    
    # Get comprehensive hardware info
    info = get_enhanced_hardware_info()
    
    # Print platform info
    print("\nPlatform Information:")
    for key, value in info["platform_info"].items():
        print(f"- {key}: {value}")
    
    # Print browser info
    print("\nBrowser Information:")
    for browser, details in info["browser_info"].items():
        if details.get("available"):
            print(f"- {browser}: WebGPU={details.get('webgpu')}, WebNN={details.get('webnn')}, Version={details.get('version')}")
    
    return info


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Heterogeneous Scheduler Test")
    parser.add_argument("--workers", type=int, default=5, help="Number of simulated workers")
    parser.add_argument("--tasks", type=int, default=100, help="Number of tasks to schedule")
    parser.add_argument("--strategy", type=str, default="adaptive", 
                        choices=["adaptive", "resource_aware", "performance_aware", "round_robin"],
                        help="Scheduling strategy")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--output", type=str, help="Output file for results (.png for visualization, .json for data)")
    parser.add_argument("--detect-hardware", action="store_true", help="Test actual hardware detection")
    
    args = parser.parse_args()
    
    if args.detect_hardware:
        test_actual_hardware_detection()
    elif args.compare:
        run_strategy_comparison(
            num_workers=args.workers,
            num_tasks=args.tasks,
            output_file=args.output
        )
    else:
        run_simulation(
            num_workers=args.workers,
            num_tasks=args.tasks,
            scheduler_strategy=args.strategy,
            visualization=not args.no_viz,
            output_file=args.output
        )


if __name__ == "__main__":
    main()