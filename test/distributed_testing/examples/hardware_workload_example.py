#!/usr/bin/env python3
"""
Example script demonstrating the Enhanced Hardware-Aware Workload Management system.

This example shows how to:
1. Initialize the hardware taxonomy and workload manager
2. Register hardware profiles
3. Create and register workloads with different profiles
4. Execute workloads on optimal hardware
5. Use the Multi-Device Orchestrator for complex tasks
6. Demonstrate advanced fault tolerance capabilities
7. Utilize performance tracking and resource monitoring
8. Implement thermal state tracking and simulation
9. Create and monitor workload dependency graphs
"""

import os
import sys
import time
import logging
import argparse
import uuid
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
import json
from datetime import datetime, timedelta
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try both import paths for flexibility
try:
    # Import hardware workload management components
    from .hardware_workload_management import (
        WorkloadType, WorkloadProfileMetric, WorkloadProfile, WorkloadExecutionPlan,
        HardwareWorkloadManager, MultiDeviceOrchestrator, SubtaskDefinition, SubtaskStatus,
        WorkloadExecutionGraph, create_workload_profile
    )
except ImportError:
    # Try alternative import path
    from data.duckdb.distributed_testing.hardware_workload_management import (
        WorkloadType, WorkloadProfileMetric, WorkloadProfile, WorkloadExecutionPlan,
        HardwareWorkloadManager, MultiDeviceOrchestrator, SubtaskDefinition, SubtaskStatus,
        WorkloadExecutionGraph, create_workload_profile
    )

# Import hardware taxonomy components
from data.duckdb.distributed_testing.hardware_taxonomy import (
    HardwareClass, HardwareArchitecture, HardwareVendor,
    SoftwareBackend, PrecisionType, AcceleratorFeature,
    HardwareCapabilityProfile, HardwareTaxonomy,
    create_cpu_profile, create_gpu_profile, create_npu_profile, create_browser_profile
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_workload_example")


def setup_hardware_taxonomy() -> HardwareTaxonomy:
    """Set up a hardware taxonomy with example hardware profiles."""
    taxonomy = HardwareTaxonomy()
    
    # Create some CPU profiles
    cpu1 = create_cpu_profile(
        model_name="Intel Xeon E5-2680",
        vendor=HardwareVendor.INTEL,
        cores=24,
        memory_gb=128.0,
        clock_speed_mhz=2400,
        has_avx=True,
        has_avx2=True
    )
    
    cpu2 = create_cpu_profile(
        model_name="AMD EPYC 7742",
        vendor=HardwareVendor.AMD,
        cores=64,
        memory_gb=256.0,
        clock_speed_mhz=2250,
        has_avx=True,
        has_avx2=True,
        has_avx512=True
    )
    
    # Create some GPU profiles
    gpu1 = create_gpu_profile(
        model_name="NVIDIA A100",
        vendor=HardwareVendor.NVIDIA,
        compute_units=108,
        memory_gb=40.0,
        clock_speed_mhz=1410,
        has_tensor_cores=True,
        compute_capability="8.0",
        memory_bandwidth_gbps=1555.0,
        tdp_w=400.0
    )
    
    gpu2 = create_gpu_profile(
        model_name="AMD Radeon Pro VII",
        vendor=HardwareVendor.AMD,
        compute_units=60,
        memory_gb=16.0,
        clock_speed_mhz=1800,
        memory_bandwidth_gbps=1024.0,
        tdp_w=250.0
    )
    
    # Create some NPU profiles
    npu1 = create_npu_profile(
        model_name="Qualcomm Snapdragon NPU",
        vendor=HardwareVendor.QUALCOMM,
        compute_units=16,
        memory_gb=8.0,
        clock_speed_mhz=1000,
        has_quantization=True,
        tdp_w=15.0
    )
    
    # Create some browser profiles
    browser1 = create_browser_profile(
        browser_name="chrome",
        supports_webgpu=True,
        supports_webnn=False,
        gpu_profile=gpu1  # Using A100 as base GPU (scaled down)
    )
    
    browser2 = create_browser_profile(
        browser_name="firefox",
        supports_webgpu=True,
        supports_webnn=False,
        gpu_profile=gpu2  # Using Radeon as base GPU (scaled down)
    )
    
    browser3 = create_browser_profile(
        browser_name="edge",
        supports_webgpu=True,
        supports_webnn=True,
        gpu_profile=gpu1  # Using A100 as base GPU (scaled down)
    )
    
    # Register profiles in taxonomy
    for profile in [cpu1, cpu2, gpu1, gpu2, npu1, browser1, browser2, browser3]:
        taxonomy.register_hardware_profile(profile)
    
    # Register worker hardware
    worker1_profiles = [cpu1, gpu1]
    worker2_profiles = [cpu2, gpu2]
    worker3_profiles = [cpu1, npu1]
    worker4_profiles = [browser1]
    worker5_profiles = [browser2]
    worker6_profiles = [browser3]
    
    taxonomy.register_worker_hardware("worker1", worker1_profiles)
    taxonomy.register_worker_hardware("worker2", worker2_profiles)
    taxonomy.register_worker_hardware("worker3", worker3_profiles)
    taxonomy.register_worker_hardware("worker4", worker4_profiles)
    taxonomy.register_worker_hardware("worker5", worker5_profiles)
    taxonomy.register_worker_hardware("worker6", worker6_profiles)
    
    # Update specialization map based on registered hardware
    taxonomy.update_specialization_map()
    
    return taxonomy


def create_example_workloads() -> List[Dict[str, Any]]:
    """Create example workload configurations."""
    workloads = [
        {
            "name": "NLP model inference (BERT)",
            "workload_type": "nlp",
            "model_id": "bert-base-uncased",
            "min_memory_gb": 2.0,
            "min_compute_units": 4,
            "metrics": {
                "compute_intensity": 0.7,
                "memory_intensity": 0.5,
                "latency_sensitivity": 0.3,
                "throughput_sensitivity": 0.8
            },
            "priority": 2,  # High priority
            "preferred_hardware_class": "GPU",
            "backend_requirements": ["PYTORCH", "CUDA"],
            "precision_requirements": ["FP16"],
            "feature_requirements": ["TENSOR_CORES"],
            "is_shardable": False,
            "allocation_strategy": "single",
            "estimated_duration_seconds": 30
        },
        {
            "name": "Vision model training (ResNet)",
            "workload_type": "vision",
            "model_id": "resnet50",
            "min_memory_gb": 8.0,
            "min_compute_units": 16,
            "metrics": {
                "compute_intensity": 0.9,
                "memory_intensity": 0.7,
                "latency_sensitivity": 0.2,
                "throughput_sensitivity": 0.9
            },
            "priority": 3,  # Medium priority
            "preferred_hardware_class": "GPU",
            "backend_requirements": ["PYTORCH", "CUDA"],
            "precision_requirements": ["FP32", "FP16"],
            "is_shardable": True,
            "min_shards": 2,
            "max_shards": 4,
            "allocation_strategy": "sharded",
            "estimated_duration_seconds": 300
        },
        {
            "name": "Audio processing (Whisper)",
            "workload_type": "audio",
            "model_id": "openai/whisper-small",
            "min_memory_gb": 4.0,
            "min_compute_units": 8,
            "metrics": {
                "compute_intensity": 0.6,
                "memory_intensity": 0.5,
                "latency_sensitivity": 0.7,
                "throughput_sensitivity": 0.4
            },
            "priority": 2,  # High priority
            "preferred_hardware_class": None,  # No preference
            "backend_requirements": ["PYTORCH"],
            "precision_requirements": ["FP32"],
            "is_shardable": False,
            "allocation_strategy": "single",
            "estimated_duration_seconds": 60
        },
        {
            "name": "Text inference (LLM)",
            "workload_type": "nlp",
            "model_id": "meta-llama/Llama-2-7b",
            "min_memory_gb": 16.0,
            "min_compute_units": 32,
            "metrics": {
                "compute_intensity": 0.8,
                "memory_intensity": 0.9,
                "latency_sensitivity": 0.4,
                "throughput_sensitivity": 0.6
            },
            "priority": 1,  # Highest priority
            "preferred_hardware_class": "GPU",
            "backend_requirements": ["PYTORCH", "CUDA"],
            "precision_requirements": ["FP16", "INT8"],
            "is_shardable": True,
            "min_shards": 2,
            "max_shards": 4,
            "allocation_strategy": "sharded",
            "estimated_duration_seconds": 120
        },
        {
            "name": "Browser-based inference (Vision)",
            "workload_type": "vision",
            "model_id": "vit-base-patch16-224",
            "min_memory_gb": 1.0,
            "min_compute_units": 2,
            "metrics": {
                "compute_intensity": 0.6,
                "memory_intensity": 0.4,
                "latency_sensitivity": 0.8,
                "throughput_sensitivity": 0.3
            },
            "priority": 3,  # Medium priority
            "preferred_hardware_class": "HYBRID",  # Browser-based
            "backend_requirements": ["WEBGPU"],
            "precision_requirements": ["FP16"],
            "is_shardable": False,
            "allocation_strategy": "single",
            "estimated_duration_seconds": 10
        }
    ]
    
    return workloads


def create_multi_device_workload_config() -> Dict[str, Any]:
    """Create a configuration for a multi-device workload."""
    return {
        "name": "Multi-Device LLM Inference",
        "aggregation_method": "concat",
        "subtasks": {
            "tokenize": {
                "workload_type": "nlp",
                "model_id": "tokenizer",
                "min_memory_gb": 1.0,
                "min_compute_units": 2,
                "metrics": {
                    "compute_intensity": 0.3,
                    "memory_intensity": 0.3
                },
                "priority": 1,
                "dependencies": [],
                "preferred_hardware_class": "CPU"
            },
            "encode": {
                "workload_type": "nlp",
                "model_id": "encoder",
                "min_memory_gb": 4.0,
                "min_compute_units": 8,
                "metrics": {
                    "compute_intensity": 0.7,
                    "memory_intensity": 0.5
                },
                "priority": 2,
                "dependencies": ["tokenize"],
                "preferred_hardware_class": "GPU"
            },
            "generate": {
                "workload_type": "nlp",
                "model_id": "decoder",
                "min_memory_gb": 8.0,
                "min_compute_units": 16,
                "metrics": {
                    "compute_intensity": 0.8,
                    "memory_intensity": 0.7
                },
                "priority": 1,
                "dependencies": ["encode"],
                "preferred_hardware_class": "GPU"
            },
            "postprocess": {
                "workload_type": "nlp",
                "model_id": "postprocessor",
                "min_memory_gb": 1.0,
                "min_compute_units": 2,
                "metrics": {
                    "compute_intensity": 0.2,
                    "memory_intensity": 0.2
                },
                "priority": 3,
                "dependencies": ["generate"],
                "preferred_hardware_class": "CPU"
            }
        }
    }


def workload_callback(workload_id: str, plan: WorkloadExecutionPlan) -> None:
    """Callback function for workload events."""
    logger.info(f"Workload {workload_id} event: {plan.execution_status}")
    
    if plan.execution_status == "completed":
        duration = plan.get_actual_duration()
        if duration:
            logger.info(f"Workload {workload_id} completed in {duration:.2f}s "
                      f"(estimated: {plan.estimated_execution_time:.2f}s)")


def simulate_workload_execution(workload_manager: HardwareWorkloadManager) -> None:
    """
    Simulate workload execution by updating execution status after delays.
    
    This is a mock function to simulate workload execution in this example.
    In a real system, the actual execution would be done by worker nodes.
    """
    for workload_id, plan in workload_manager.execution_plans.items():
        if plan.execution_status == "planned":
            # Mark as executing
            workload_manager.update_execution_status(workload_id, "executing")
            
            # Simulate execution time (in a real system, this would be the actual execution)
            # For simulation, we'll make it shorter to speed up the example
            simulated_duration = plan.estimated_execution_time / 10.0  # 10x faster for simulation
            logger.info(f"Simulating execution of workload {workload_id} for {simulated_duration:.2f}s")
            time.sleep(simulated_duration)
            
            # Mark as completed (in real system, this would come from worker reporting)
            workload_manager.update_execution_status(workload_id, "completed")


def print_execution_summary(workload_manager: HardwareWorkloadManager, 
                          workload_names: Dict[str, str]) -> None:
    """Print a summary of workload executions."""
    print("\n===== Workload Execution Summary =====")
    print(f"{'Workload':<30} {'Status':<12} {'Duration':<10} {'Hardware':<20} {'Efficiency':<10}")
    print("-" * 85)
    
    for workload_id, plan in workload_manager.execution_plans.items():
        name = workload_names.get(workload_id, workload_id)
        status = plan.execution_status
        
        duration = "N/A"
        if plan.started_at and plan.completed_at:
            actual_duration = (plan.completed_at - plan.started_at).total_seconds()
            duration = f"{actual_duration:.2f}s"
        
        hardware = []
        for hw_id, _ in plan.hardware_assignments:
            if "_" in hw_id:  # Extract just the worker part
                hw_id = hw_id.split("_")[0]
            hardware.append(hw_id)
        
        hardware_str = ", ".join(hardware)
        if len(hardware_str) > 20:
            hardware_str = hardware_str[:17] + "..."
        
        efficiency = f"{plan.estimated_efficiency:.2f}"
        
        print(f"{name[:30]:<30} {status:<12} {duration:<10} {hardware_str:<20} {efficiency:<10}")
    
    print("=" * 85)


def run_multi_device_example(workload_manager: HardwareWorkloadManager) -> None:
    """
    Run an example demonstrating the Enhanced Multi-Device Orchestrator with advanced features.
    
    This includes:
    - Advanced fault tolerance capabilities
    - Resource monitoring and performance tracking
    - Critical path analysis for complex workloads
    - Recovery strategies for different failure scenarios
    - Thermal state tracking and management
    """
    # Initialize the orchestrator with advanced features
    orchestrator = MultiDeviceOrchestrator(workload_manager)
    
    # Configure fault tolerance options
    fault_tolerance_config = {
        "enabled": True,
        "default_recovery_strategy": "retry",
        "max_retries": 3,
        "retry_delay_seconds": 1,
        "critical_subtasks": ["encode", "generate"],
        "checkpointing": {
            "enabled": True,
            "interval_seconds": 5
        },
        "alternative_implementations": {
            "generate": "generate_fallback"
        }
    }
    
    # Set advanced options for the orchestrator
    orchestrator.configure({
        "fault_tolerance": fault_tolerance_config,
        "performance_tracking": {
            "enabled": True,
            "history_size": 50
        },
        "resource_monitoring": {
            "enabled": True,
            "high_utilization_threshold": 0.85,
            "sampling_interval_seconds": 1
        },
        "thermal_tracking": {
            "enabled": True,
            "critical_temperature": 85.0,  # Celsius
            "throttling_temperature": 75.0  # Celsius
        },
        "critical_path_analysis": {
            "enabled": True,
            "path_priority_boost": 2
        }
    })
    
    # Create a multi-device workload config
    workload_config = create_multi_device_workload_config()
    workload_id = "multi_device_example"
    
    # Register the workload with the orchestrator
    orchestrator.register_workload(workload_id, workload_config)
    logger.info(f"Registered multi-device workload: {workload_config['name']}")
    
    # Create execution graph for visualization
    execution_graph = orchestrator.get_execution_graph(workload_id)
    logger.info(f"Created execution graph with {len(execution_graph.subtasks)} subtasks")
    
    # Analyze the critical path for prioritization
    critical_path = orchestrator.analyze_critical_path(workload_id)
    logger.info(f"Critical path subtasks: {critical_path}")
    
    # Get ready subtasks (those with no dependencies)
    ready_subtasks = orchestrator.get_ready_subtasks(workload_id)
    logger.info(f"Initial ready subtasks: {[st['subtask_id'] for st in ready_subtasks]}")
    
    # Process each ready subtask
    for subtask in ready_subtasks:
        subtask_id = subtask["subtask_id"]
        config = subtask["config"]
        
        # Add fault tolerance options for critical subtasks
        if subtask_id in fault_tolerance_config.get("critical_subtasks", []):
            config["fault_tolerance"] = {
                "recovery_strategy": "retry_then_reassign",
                "max_retries": 5,
                "is_critical": True
            }
        
        # Create workload profile for the subtask
        workload_profile = create_workload_profile(
            workload_type=config["workload_type"],
            model_id=config["model_id"],
            min_memory_gb=config["min_memory_gb"],
            min_compute_units=config["min_compute_units"],
            metrics=config.get("metrics", {}),
            priority=config.get("priority", 3),
            preferred_hardware_class=config.get("preferred_hardware_class"),
            workload_id=f"{workload_id}_{subtask_id}"
        )
        
        # Register with workload manager
        full_subtask_id = workload_manager.register_workload(workload_profile)
        logger.info(f"Registered subtask {subtask_id} with workload manager as {full_subtask_id}")
    
    # Simulate execution of subtasks 
    simulate_workload_execution(workload_manager)
    
    # Simulate a hardware failure for one of the subtasks (for demonstration)
    if ready_subtasks and len(ready_subtasks) > 0:
        # Pick the first subtask to simulate failure
        failure_subtask = ready_subtasks[0]["subtask_id"]
        full_subtask_id = f"{workload_id}_{failure_subtask}"
        
        logger.info(f"Simulating hardware failure for subtask {failure_subtask}")
        
        # Simulate failure detection and recovery
        error_details = {
            "error_type": "hardware_failure",
            "message": "Simulated GPU memory error",
            "timestamp": datetime.now().isoformat()
        }
        
        # Trigger fault tolerance system
        orchestrator.handle_subtask_failure(workload_id, failure_subtask, error_details)
        
        # After recovery, the subtask should be retried
        logger.info(f"Recovery initiated for failed subtask {failure_subtask}")
        
        # Simulate successful retry
        time.sleep(1)  # Brief delay for retry
        result = {"status": "success", "execution_time": 0.5, "retry_count": 1}
        orchestrator.record_subtask_result(workload_id, failure_subtask, result)
        logger.info(f"Recovery successful for subtask {failure_subtask} after retry")
    
    # Record results of completed subtasks
    for workload_id_full, plan in workload_manager.execution_plans.items():
        if plan.execution_status == "completed" and "_" in workload_id_full:
            # This is a subtask (format: "multi_device_example_subtask_name")
            parts = workload_id_full.split("_")
            if len(parts) >= 3:
                main_workload_id = parts[0]
                subtask_id = parts[-1]  # Last part is the subtask name
                
                # Skip the failed/retried subtask as we've already handled it
                if subtask_id == failure_subtask:
                    continue
                
                # Record mock result
                result = {"status": "success", "execution_time": plan.get_actual_duration()}
                orchestrator.record_subtask_result(main_workload_id, subtask_id, result)
                logger.info(f"Recorded result for subtask {subtask_id} of workload {main_workload_id}")
    
    # Get next ready subtasks
    ready_subtasks = orchestrator.get_ready_subtasks(workload_id)
    logger.info(f"Next ready subtasks: {[st['subtask_id'] for st in ready_subtasks]}")
    
    # Process next batch of subtasks
    for subtask in ready_subtasks:
        subtask_id = subtask["subtask_id"]
        config = subtask["config"]
        
        # Add resource monitoring options for intensive subtasks
        if subtask_id == "generate":
            # This is a computationally intensive subtask
            config["resource_monitoring"] = {
                "monitor_interval_seconds": 0.5,
                "high_memory_threshold_gb": 7.5,
                "high_cpu_threshold": 0.9
            }
        
        # Create workload profile for the subtask
        workload_profile = create_workload_profile(
            workload_type=config["workload_type"],
            model_id=config["model_id"],
            min_memory_gb=config["min_memory_gb"],
            min_compute_units=config["min_compute_units"],
            metrics=config.get("metrics", {}),
            priority=config.get("priority", 3),
            preferred_hardware_class=config.get("preferred_hardware_class"),
            workload_id=f"{workload_id}_{subtask_id}"
        )
        
        # Register with workload manager
        full_subtask_id = workload_manager.register_workload(workload_profile)
        logger.info(f"Registered subtask {subtask_id} with workload manager as {full_subtask_id}")
    
    # Simulate execution again
    simulate_workload_execution(workload_manager)
    
    # Simulate thermal throttling for generate subtask
    if any(st["subtask_id"] == "generate" for st in ready_subtasks):
        logger.info("Simulating thermal throttling for 'generate' subtask")
        
        # Simulate thermal state update
        thermal_state = {
            "temperature": 78.0,  # Above throttling temperature
            "power_draw_watts": 120,
            "throttling_active": True,
            "fan_speed_percent": 85,
            "thermal_headroom": -3.0  # Negative means over threshold
        }
        
        # Update thermal state
        orchestrator.update_thermal_state("worker1_GPU", thermal_state)
        
        # Get performance impact
        performance_impact = orchestrator.get_thermal_performance_impact("worker1_GPU")
        logger.info(f"Thermal throttling performance impact: {performance_impact:.2f}x slowdown")
        
        # Orchestrator should automatically adjust execution plans for thermal conditions
        thermal_adaptations = orchestrator.get_thermal_adaptations(workload_id)
        logger.info(f"Applied thermal adaptations: {thermal_adaptations}")
    
    # Record results for the second batch
    for workload_id_full, plan in workload_manager.execution_plans.items():
        if plan.execution_status == "completed" and "_" in workload_id_full:
            parts = workload_id_full.split("_")
            if len(parts) >= 3:
                main_workload_id = parts[0]
                subtask_id = parts[-1]  # Last part is the subtask name
                
                # Skip already recorded results (using orchestrator's results tracking)
                if orchestrator.has_subtask_result(main_workload_id, subtask_id):
                    continue
                
                # Record mock result
                result = {"status": "success", "execution_time": plan.get_actual_duration()}
                orchestrator.record_subtask_result(main_workload_id, subtask_id, result)
                logger.info(f"Recorded result for subtask {subtask_id} of workload {main_workload_id}")
    
    # Continue this process until all subtasks are completed
    max_iterations = 10  # Safety limit
    iteration = 0
    
    while not orchestrator.is_workload_completed(workload_id) and iteration < max_iterations:
        iteration += 1
        ready_subtasks = orchestrator.get_ready_subtasks(workload_id)
        if not ready_subtasks:
            break  # No more ready subtasks but workload not completed - possible deadlock
        
        logger.info(f"Next batch of ready subtasks (iteration {iteration}): {[st['subtask_id'] for st in ready_subtasks]}")
        
        # Process subtasks
        for subtask in ready_subtasks:
            subtask_id = subtask["subtask_id"]
            config = subtask["config"]
            
            # Construct rich workload profile
            workload_profile = create_workload_profile(
                workload_type=config["workload_type"],
                model_id=config["model_id"],
                min_memory_gb=config["min_memory_gb"],
                min_compute_units=config["min_compute_units"],
                metrics=config.get("metrics", {}),
                priority=config.get("priority", 3),
                preferred_hardware_class=config.get("preferred_hardware_class"),
                workload_id=f"{workload_id}_{subtask_id}"
            )
            
            full_subtask_id = workload_manager.register_workload(workload_profile)
            logger.info(f"Registered subtask {subtask_id} with workload manager as {full_subtask_id}")
            
            # For the last subtask, demonstrate checkpoint/resume functionality
            if subtask_id == "postprocess":
                logger.info("Demonstrating checkpoint/resume functionality for 'postprocess' subtask")
                
                # Create a checkpoint
                checkpoint_id = orchestrator.create_checkpoint(workload_id)
                logger.info(f"Created checkpoint {checkpoint_id} for workload {workload_id}")
                
                # Simulate a system interruption
                logger.info("Simulating system interruption...")
                time.sleep(1)
                
                # Resume from checkpoint
                orchestrator.resume_from_checkpoint(checkpoint_id)
                logger.info(f"Resumed workload {workload_id} from checkpoint {checkpoint_id}")
        
        # Simulate execution
        simulate_workload_execution(workload_manager)
        
        # Record results with advanced handling
        for workload_id_full, plan in workload_manager.execution_plans.items():
            if plan.execution_status == "completed" and "_" in workload_id_full:
                parts = workload_id_full.split("_")
                if len(parts) >= 3:
                    main_workload_id = parts[0]
                    subtask_id = parts[-1]  # Last part is the subtask name
                    
                    # Check if this result has already been recorded
                    if orchestrator.has_subtask_result(main_workload_id, subtask_id):
                        continue
                    
                    # Get detailed performance metrics
                    performance_metrics = {
                        "execution_time": plan.get_actual_duration() or 0.5,
                        "cpu_utilization": random.uniform(0.5, 0.95),
                        "memory_utilization": random.uniform(0.4, 0.85),
                        "efficiency_score": random.uniform(0.7, 0.98)
                    }
                    
                    # Record result with performance metrics
                    result = {"status": "success", **performance_metrics}
                    orchestrator.record_subtask_result(main_workload_id, subtask_id, result)
                    logger.info(f"Recorded result for subtask {subtask_id} with performance metrics")
    
    # Get performance statistics for the workload
    performance_stats = orchestrator.get_performance_statistics(workload_id)
    logger.info(f"Performance statistics: {performance_stats}")
    
    # Check if workload is completed
    if orchestrator.is_workload_completed(workload_id):
        # Get aggregated results
        results = orchestrator.get_aggregated_results(workload_id)
        logger.info(f"Multi-device workload completed with results: {results}")
        
        # Get execution timeline
        timeline = orchestrator.get_execution_timeline(workload_id)
        logger.info(f"Execution timeline has {len(timeline)} events")
        
        # Get resource utilization report
        resource_report = orchestrator.get_resource_utilization_report(workload_id)
        logger.info(f"Resource utilization report generated")
        
        # Get bottleneck analysis
        bottlenecks = orchestrator.analyze_bottlenecks(workload_id)
        logger.info(f"Identified {len(bottlenecks)} bottlenecks in the workload execution")
        
        # Print success with performance summary
        logger.info("Multi-device workload completed successfully with enhanced monitoring and fault tolerance")
    else:
        logger.warning(f"Multi-device workload not completed, remaining subtasks: "
                     f"{len(workload_config['subtasks']) - len(orchestrator.get_completed_subtasks(workload_id))}")
        
        # Get failure report
        failures = orchestrator.get_failure_report(workload_id)
        logger.warning(f"Workload execution failures: {failures}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Enhanced Hardware-Aware Workload Management Example")
    parser.add_argument("--multi-device", action="store_true", help="Run multi-device orchestration example")
    parser.add_argument("--fault-tolerance", action="store_true", help="Enable fault tolerance features")
    parser.add_argument("--thermal-tracking", action="store_true", help="Enable thermal state tracking")
    parser.add_argument("--resource-monitoring", action="store_true", help="Enable resource monitoring")
    parser.add_argument("--performance-tracking", action="store_true", help="Enable performance tracking")
    parser.add_argument("--all-features", action="store_true", help="Enable all advanced features")
    parser.add_argument("--db-path", help="Path to database for storing results")
    parser.add_argument("--visualize", action="store_true", help="Visualize execution graphs and timelines")
    parser.add_argument("--simulate-failures", action="store_true", help="Simulate hardware failures")
    args = parser.parse_args()
    
    # Set up hardware taxonomy
    logger.info("Setting up hardware taxonomy...")
    taxonomy = setup_hardware_taxonomy()
    
    # Initialize workload manager with advanced options
    logger.info("Initializing hardware workload manager...")
    workload_manager = HardwareWorkloadManager(taxonomy, db_path=args.db_path)
    
    # Configure manager with advanced features
    advanced_config = {}
    
    if args.fault_tolerance or args.all_features:
        advanced_config["fault_tolerance"] = {
            "enabled": True,
            "default_recovery_strategy": "retry",
            "max_retries": 3,
            "retry_delay_seconds": 1
        }
        logger.info("Fault tolerance features enabled")
    
    if args.thermal_tracking or args.all_features:
        advanced_config["thermal_tracking"] = {
            "enabled": True,
            "critical_temperature": 85.0,
            "throttling_temperature": 75.0,
            "monitor_interval_seconds": 1.0
        }
        logger.info("Thermal state tracking enabled")
    
    if args.resource_monitoring or args.all_features:
        advanced_config["resource_monitoring"] = {
            "enabled": True,
            "high_utilization_threshold": 0.85,
            "sampling_interval_seconds": 1
        }
        logger.info("Resource monitoring enabled")
    
    if args.performance_tracking or args.all_features:
        advanced_config["performance_tracking"] = {
            "enabled": True,
            "history_size": 50,
            "track_subtask_duration": True,
            "track_resource_utilization": True
        }
        logger.info("Performance tracking enabled")
    
    if args.visualize or args.all_features:
        advanced_config["visualization"] = {
            "enabled": True,
            "generate_execution_graphs": True,
            "generate_timelines": True,
            "generate_thermal_heatmaps": True,
            "generate_resource_utilization_charts": True
        }
        logger.info("Visualization features enabled")
    
    # Configure workload manager with advanced features if any are enabled
    if advanced_config:
        workload_manager.configure(advanced_config)
    
    # Start the workload manager
    workload_manager.start()
    
    # Register event callbacks
    workload_manager.register_event_callback("workload_completed", workload_callback)
    workload_manager.register_event_callback("workload_failed", workload_callback)
    
    if args.fault_tolerance or args.all_features:
        # Register additional fault tolerance callbacks
        workload_manager.register_event_callback("workload_retry", 
                                              lambda wid, data: logger.info(f"Retrying workload {wid}"))
        workload_manager.register_event_callback("workload_reassigned", 
                                              lambda wid, data: logger.info(f"Reassigned workload {wid}"))
        workload_manager.register_event_callback("recovery_strategy_applied", 
                                              lambda wid, data: logger.info(f"Applied recovery strategy for {wid}"))
    
    if args.resource_monitoring or args.all_features:
        # Register resource monitoring callbacks
        workload_manager.register_event_callback("high_resource_utilization", 
                                              lambda device_id, data: logger.warning(f"High utilization on {device_id}"))
        workload_manager.register_event_callback("resource_constraint_detected", 
                                              lambda device_id, data: logger.warning(f"Resource constraint on {device_id}"))
    
    if args.thermal_tracking or args.all_features:
        # Register thermal state callbacks
        workload_manager.register_event_callback("thermal_throttling", 
                                              lambda device_id, data: logger.warning(f"Thermal throttling on {device_id}"))
        workload_manager.register_event_callback("thermal_critical", 
                                              lambda device_id, data: logger.critical(f"Thermal critical on {device_id}"))
    
    # Create workload profiles
    logger.info("Creating workload profiles...")
    workload_configs = create_example_workloads()
    
    # Track names for pretty printing
    workload_names = {}
    
    # Add failure simulation for some workloads if enabled
    if args.simulate_failures:
        logger.info("Adding failure simulation to some workloads...")
        for i, config in enumerate(workload_configs):
            # Simulate failures for 30% of workloads
            if i % 3 == 0:
                config["simulate_failure"] = {
                    "type": "hardware_failure" if i % 2 == 0 else "software_error",
                    "probability": 0.7,
                    "retry_probability": 0.8  # 80% chance of successful retry
                }
                logger.info(f"Added failure simulation to '{config['name']}'")
    
    # Register workloads
    logger.info("Registering workloads...")
    for config in workload_configs:
        # Prepare advanced options based on enabled features
        advanced_options = {}
        
        if args.fault_tolerance or args.all_features:
            advanced_options["fault_tolerance"] = {
                "enabled": True,
                "recovery_strategy": "retry_then_reassign" if config.get("priority", 3) <= 2 else "retry",
                "max_retries": 3,
                "is_critical": config.get("priority", 3) <= 2
            }
        
        if args.resource_monitoring or args.all_features:
            advanced_options["resource_monitoring"] = {
                "enabled": True,
                "high_memory_threshold_gb": config.get("min_memory_gb", 1.0) * 0.9,
                "high_cpu_threshold": 0.9
            }
        
        if args.thermal_tracking or args.all_features and config.get("preferred_hardware_class") == "GPU":
            advanced_options["thermal_tracking"] = {
                "enabled": True,
                "critical_temperature": 85.0,
                "throttling_temperature": 75.0
            }
        
        # Create the workload profile with all configured options
        workload_profile = create_workload_profile(
            workload_type=config["workload_type"],
            model_id=config["model_id"],
            min_memory_gb=config["min_memory_gb"],
            min_compute_units=config["min_compute_units"],
            metrics=config["metrics"],
            priority=config["priority"],
            preferred_hardware_class=config["preferred_hardware_class"],
            backend_requirements=config.get("backend_requirements"),
            precision_requirements=config.get("precision_requirements"),
            feature_requirements=config.get("feature_requirements"),
            is_shardable=config.get("is_shardable", False),
            min_shards=config.get("min_shards", 1),
            max_shards=config.get("max_shards", 1),
            allocation_strategy=config.get("allocation_strategy", "single"),
            estimated_duration_seconds=config.get("estimated_duration_seconds", 60)
        )
        
        # Add advanced options if any
        if advanced_options:
            for key, value in advanced_options.items():
                if hasattr(workload_profile, key):
                    setattr(workload_profile, key, value)
                else:
                    # Add to custom properties if attribute doesn't exist
                    workload_profile.custom_properties[key] = value
        
        # Register the workload
        workload_id = workload_manager.register_workload(workload_profile)
        workload_names[workload_id] = config["name"]
        logger.info(f"Registered workload {workload_id}: {config['name']}")
    
    # Simulate failures if requested
    if args.simulate_failures:
        logger.info("Simulating hardware failures during execution...")
        # Workload manager will handle this automatically based on the failure simulation configs
    
    # Simulate execution
    logger.info("Simulating workload execution...")
    simulate_workload_execution(workload_manager)
    
    # Print summary
    print_execution_summary(workload_manager, workload_names)
    
    # Get and print statistics if performance tracking is enabled
    if args.performance_tracking or args.all_features:
        try:
            stats = workload_manager.get_performance_statistics()
            print("\n===== Performance Statistics =====")
            print(f"Average execution time: {stats.get('avg_execution_time', 'N/A')}s")
            print(f"Execution time variance: {stats.get('execution_time_variance', 'N/A')}")
            print(f"Resource utilization: {stats.get('avg_resource_utilization', 'N/A')}%")
            print(f"Workload efficiency: {stats.get('avg_efficiency', 'N/A')}")
            print("=" * 85)
        except Exception as e:
            logger.warning(f"Could not retrieve performance statistics: {str(e)}")
    
    # Run multi-device example if requested
    if args.multi_device:
        logger.info("\nRunning enhanced multi-device orchestration example...")
        run_multi_device_example(workload_manager)
    
    # Generate visualizations if requested
    if args.visualize or args.all_features:
        try:
            logger.info("Generating visualizations...")
            vis_result = workload_manager.generate_visualizations()
            logger.info(f"Generated {vis_result.get('num_visualizations', 0)} visualizations")
            
            # Print paths to visualization files
            if 'visualization_paths' in vis_result:
                print("\n===== Visualization Files =====")
                for viz_type, path in vis_result['visualization_paths'].items():
                    print(f"{viz_type}: {path}")
                print("=" * 85)
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {str(e)}")
    
    # Stop workload manager
    logger.info("Stopping workload manager...")
    workload_manager.stop()
    
    # Print final status
    if args.all_features:
        logger.info("Enhanced Hardware-Aware Workload Management example completed successfully with all features enabled")
    else:
        logger.info("Example completed successfully")


if __name__ == "__main__":
    main()