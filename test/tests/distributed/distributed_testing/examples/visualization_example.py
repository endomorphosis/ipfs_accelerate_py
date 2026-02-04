#!/usr/bin/env python3
"""
Hardware-Aware Scheduling Visualization Example

This script demonstrates how to use the visualization capabilities of the
Hardware-Aware Workload Management system and its integration with the Load Balancer.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
import random
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components
from test.tests.distributed.distributed_testing.hardware_workload_management import (
    HardwareWorkloadManager, WorkloadProfile, WorkloadType, WorkloadProfileMetric,
    HardwareTaxonomy, create_workload_profile
)
from test.tests.distributed.distributed_testing.hardware_aware_scheduler import HardwareAwareScheduler
from test.tests.distributed.distributed_testing.examples.hardware_aware_visualization import (
    HardwareSchedulingVisualizer, create_visualizer
)
from test.tests.distributed.distributed_testing.examples.load_balancer_integration import (
    create_hardware_aware_load_balancer, shutdown_integration
)

# Import for simulating examples
from test.tests.distributed.distributed_testing.examples.load_balancer_integration_example import (
    create_sample_worker_capabilities,
    create_sample_test_requirements,
    simulate_worker_load
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("visualization_example")


def create_visualization_directory(name: str = None) -> str:
    """
    Create a directory for visualizations.
    
    Args:
        name: Optional name for the directory (defaults to timestamp)
        
    Returns:
        Path to the created directory
    """
    if name is None:
        # Use timestamp as directory name
        name = f"hardware_scheduler_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directory in current working directory
    output_dir = os.path.join(os.getcwd(), name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def run_efficiency_visualization_example(output_dir: str) -> None:
    """
    Run an example of hardware efficiency visualization.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create visualizer
    visualizer = create_visualizer(output_dir=output_dir)
    
    # Create taxonomy and workload manager
    taxonomy = HardwareTaxonomy()
    workload_manager = HardwareWorkloadManager(taxonomy)
    
    # Create hardware profiles for the example
    hardware_profiles = []
    
    # CPU profile
    cpu_profile = taxonomy.create_cpu_profile(
        model_name="Intel Core i7-10700K",
        vendor=HardwareVendor.INTEL,
        cores=8,
        memory_gb=32.0,
        clock_speed_mhz=3800,
        has_avx=True,
        has_avx2=True
    )
    hardware_profiles.append(cpu_profile)
    
    # GPU profile
    gpu_profile = taxonomy.create_gpu_profile(
        model_name="NVIDIA RTX 3080",
        vendor=HardwareVendor.NVIDIA,
        compute_units=68,
        memory_gb=12.0,
        clock_speed_mhz=1440,
        has_tensor_cores=True,
        memory_bandwidth_gbps=760.0
    )
    hardware_profiles.append(gpu_profile)
    
    # NPU profile
    npu_profile = taxonomy.create_npu_profile(
        model_name="Qualcomm Hexagon 780",
        vendor=HardwareVendor.QUALCOMM,
        compute_units=8,
        memory_gb=4.0,
        clock_speed_mhz=1000,
        has_quantization=True
    )
    hardware_profiles.append(npu_profile)
    
    # Create example workload profiles
    workload_types = [
        WorkloadType.VISION,
        WorkloadType.NLP,
        WorkloadType.AUDIO
    ]
    
    for wl_type in workload_types:
        # Create workload profile
        workload_id = f"{wl_type.value.lower()}_workload"
        workload_profile = create_workload_profile(
            workload_type=wl_type.value,
            model_id=f"example-{wl_type.value.lower()}-model",
            min_memory_gb=4.0,
            min_compute_units=2,
            priority=3,
            workload_id=workload_id
        )
        
        # Calculate efficiency scores for different hardware profiles
        efficiency_scores = {}
        for hw_profile in hardware_profiles:
            # Calculate efficiency score (normally done by the workload manager)
            efficiency = workload_profile.get_efficiency_score(hw_profile)
            
            # Create hardware ID (normally created by the scheduler)
            hw_id = f"worker1_{hw_profile.model_name}"
            
            efficiency_scores[hw_id] = efficiency
        
        # Visualize hardware efficiency
        visualizer.visualize_hardware_efficiency(
            hardware_profiles=hardware_profiles,
            workload_profile=workload_profile,
            efficiency_scores=efficiency_scores,
            filename=f"efficiency_{wl_type.value.lower()}_workload"
        )
        
        logger.info(f"Created efficiency visualization for {wl_type.value} workload")
    
    workload_manager.stop()


def run_workload_distribution_example(output_dir: str) -> None:
    """
    Run an example of workload distribution visualization.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create visualizer
    visualizer = create_visualizer(output_dir=output_dir)
    
    # Create worker assignments
    worker_types = {
        "worker1": "generic",
        "worker2": "gpu",
        "worker3": "tpu",
        "worker4": "browser",
        "worker5": "mobile"
    }
    
    # Simulate workload distribution
    worker_assignments = {
        "worker1": [f"test_{i}" for i in range(3)],
        "worker2": [f"test_{i}" for i in range(3, 12)],
        "worker3": [f"test_{i}" for i in range(12, 16)],
        "worker4": [f"test_{i}" for i in range(16, 19)],
        "worker5": [f"test_{i}" for i in range(19, 20)]
    }
    
    # Visualize workload distribution
    visualizer.visualize_workload_distribution(
        worker_assignments=worker_assignments,
        worker_types=worker_types,
        filename="workload_distribution_example"
    )
    
    logger.info("Created workload distribution visualization")


def run_thermal_states_example(output_dir: str) -> None:
    """
    Run an example of thermal states visualization.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create visualizer
    visualizer = create_visualizer(output_dir=output_dir)
    
    # Create thermal states
    thermal_states = {
        "worker1": {"temperature": 0.2, "warming_state": True, "cooling_state": False},
        "worker2": {"temperature": 0.8, "warming_state": False, "cooling_state": True},
        "worker3": {"temperature": 0.5, "warming_state": False, "cooling_state": False},
        "worker4": {"temperature": 0.3, "warming_state": True, "cooling_state": False},
        "worker5": {"temperature": 0.7, "warming_state": False, "cooling_state": False}
    }
    
    # Visualize thermal states
    visualizer.visualize_thermal_states(
        thermal_states=thermal_states,
        filename="thermal_states_example"
    )
    
    logger.info("Created thermal states visualization")


def run_resource_utilization_example(output_dir: str) -> None:
    """
    Run an example of resource utilization visualization.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create visualizer
    visualizer = create_visualizer(output_dir=output_dir)
    
    # Create worker loads
    worker_loads = {}
    
    for i, worker_id in enumerate(["worker1", "worker2", "worker3", "worker4", "worker5"]):
        # Simulate different load levels
        load_factor = (i + 1) / 5.0
        
        # Create worker load
        worker_loads[worker_id] = {
            "cpu_utilization": 20.0 + (load_factor * 60.0),
            "memory_utilization": 30.0 + (load_factor * 50.0),
            "gpu_utilization": 10.0 + (load_factor * 80.0) if "worker2" in worker_id else 0.0,
            "io_utilization": 5.0 + (load_factor * 30.0),
            "network_utilization": 10.0 + (load_factor * 40.0)
        }
    
    # Visualize resource utilization
    visualizer.visualize_resource_utilization(
        worker_loads=worker_loads,
        filename="resource_utilization_example"
    )
    
    logger.info("Created resource utilization visualization")


def run_execution_times_example(output_dir: str) -> None:
    """
    Run an example of execution times visualization.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create visualizer
    visualizer = create_visualizer(output_dir=output_dir)
    
    # Create execution data
    execution_data = {}
    
    workload_types = ["VISION", "NLP", "AUDIO", "MULTIMODAL", "TRAINING", "INFERENCE"]
    
    for i in range(20):
        workload_id = f"workload_{i}"
        workload_type = random.choice(workload_types)
        
        # Estimated time
        estimated_time = 30.0 + (random.random() * 90.0)
        
        # Actual time with some variation
        variation = 0.7 + (random.random() * 0.6)
        actual_time = estimated_time * variation
        
        # Add outliers occasionally
        if random.random() < 0.1:
            actual_time = estimated_time * (1.5 + random.random())
        elif random.random() < 0.1:
            actual_time = estimated_time * (0.5 - (random.random() * 0.3))
        
        execution_data[workload_id] = {
            "estimated_time": estimated_time,
            "actual_time": actual_time,
            "workload_type": workload_type
        }
    
    # Visualize execution times
    visualizer.visualize_execution_times(
        execution_data=execution_data,
        filename="execution_times_example"
    )
    
    logger.info("Created execution times visualization")


def run_history_tracking_example(output_dir: str) -> None:
    """
    Run an example of history tracking and visualization.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create visualizer
    visualizer = create_visualizer(output_dir=output_dir)
    
    # Create simulated history data
    start_time = datetime.now() - timedelta(hours=2)
    
    # Worker IDs
    worker_ids = ["worker1", "worker2", "worker3", "worker4", "worker5"]
    
    # Record assignments
    for i in range(50):
        # Simulate assignment
        timestamp = start_time + timedelta(minutes=i*2)
        workload_id = f"workload_{i}"
        worker_id = random.choice(worker_ids)
        efficiency_score = 0.5 + (random.random() * 0.5)
        workload_type = random.choice(["VISION", "NLP", "AUDIO", "MULTIMODAL"])
        
        # Record assignment
        visualizer.record_assignment(
            workload_id=workload_id,
            worker_id=worker_id,
            efficiency_score=efficiency_score,
            workload_type=workload_type,
            timestamp=timestamp
        )
        
        # Record thermal state updates
        for worker_id in worker_ids:
            # Simulate thermal state
            temperature = max(0.0, min(1.0, random.gauss(0.5, 0.2)))
            warming_state = temperature < 0.3 and random.random() < 0.3
            cooling_state = temperature > 0.7 and random.random() < 0.3
            
            # Record thermal state
            visualizer.record_thermal_state(
                worker_id=worker_id,
                temperature=temperature,
                warming_state=warming_state,
                cooling_state=cooling_state,
                timestamp=timestamp
            )
            
            # Record resource utilization
            cpu_util = max(0.0, min(100.0, random.gauss(50.0, 20.0)))
            memory_util = max(0.0, min(100.0, random.gauss(60.0, 15.0)))
            gpu_util = max(0.0, min(100.0, random.gauss(70.0, 25.0))) if worker_id == "worker2" else 0.0
            
            # Record resource utilization
            visualizer.record_resource_utilization(
                worker_id=worker_id,
                utilization={
                    "cpu_utilization": cpu_util,
                    "memory_utilization": memory_util,
                    "gpu_utilization": gpu_util,
                    "io_utilization": max(0.0, min(100.0, random.gauss(30.0, 10.0))),
                    "network_utilization": max(0.0, min(100.0, random.gauss(25.0, 8.0)))
                },
                timestamp=timestamp
            )
        
        # Record execution times occasionally
        if i > 5 and random.random() < 0.7:
            completed_workload_id = f"workload_{i-5}"
            estimated_time = 30.0 + (random.random() * 90.0)
            variation = 0.7 + (random.random() * 0.6)
            actual_time = estimated_time * variation
            
            visualizer.record_execution_time(
                workload_id=completed_workload_id,
                estimated_time=estimated_time,
                actual_time=actual_time,
                workload_type=random.choice(["VISION", "NLP", "AUDIO", "MULTIMODAL"]),
                worker_id=random.choice(worker_ids),
                timestamp=timestamp
            )
    
    # Visualize history data
    visualizer.visualize_history(
        history_data=visualizer.history,
        filename_prefix="history_example"
    )
    
    # Save history to file
    history_file = visualizer.save_history(filename="scheduling_history_example.json")
    
    # Generate HTML report
    report_file = visualizer.generate_summary_report(
        filename="scheduling_summary_example.html",
        include_visualizations=True
    )
    
    logger.info(f"Created history visualizations, saved history to {history_file}, and generated report at {report_file}")


def run_integrated_example(output_dir: str) -> None:
    """
    Run an integrated example of the hardware-aware scheduler with visualization.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create visualizer
    visualizer = create_visualizer(output_dir=output_dir)
    
    # Create hardware-aware load balancer
    load_balancer, workload_manager, scheduler = create_hardware_aware_load_balancer()
    
    # Start load balancer
    load_balancer.start()
    
    # Register workers
    worker_types = {
        "worker1": "generic",
        "worker2": "gpu",
        "worker3": "tpu",
        "worker4": "browser",
        "worker5": "mobile"
    }
    
    for worker_id, worker_type in worker_types.items():
        capabilities = create_sample_worker_capabilities(worker_id, worker_type)
        load_balancer.register_worker(worker_id, capabilities)
        logger.info(f"Registered worker {worker_id} of type {worker_type}")
    
    # Submit tests
    test_types = [
        "vision_classification", 
        "nlp_text_classification", 
        "audio_speech_recognition",
        "vision_object_detection",
        "nlp_large_language_model",
        "nlp_text_embedding",
        "audio_speech_synthesis",
        "vision_segmentation"
    ]
    
    model_ids = {
        "vision": ["vit-base", "resnet50", "yolov5"],
        "nlp": ["bert-base", "t5-large", "gpt2", "llama-7b"],
        "audio": ["whisper-small", "wav2vec2", "hubert"]
    }
    
    # Record of worker assignments
    worker_assignments = {worker_id: [] for worker_id in worker_types}
    
    # Submit 20 tests
    for i in range(20):
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
        test_id = f"test_{i+1}"
        
        # Create test requirements
        requirements = create_sample_test_requirements(test_id, test_type, model_id)
        
        # Submit test
        load_balancer.submit_test(requirements)
        
        logger.info(f"Submitted test {test_id} of type {test_type} with model {model_id}")
        
        # Small delay to simulate realistic submission pattern
        time.sleep(0.1)
    
    # Wait a bit for scheduling to complete
    logger.info("Waiting for scheduling to complete...")
    time.sleep(2)
    
    # Track assignments and update worker loads
    for worker_id in worker_types:
        assignments = load_balancer.get_worker_assignments(worker_id)
        assigned_test_ids = [a.test_id for a in assignments]
        worker_assignments[worker_id] = assigned_test_ids
        
        # Update worker load
        load = simulate_worker_load(worker_id, assigned_test_ids)
        load_balancer.update_worker_load(worker_id, load)
        
        # Record thermal state
        temperature = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
        visualizer.record_thermal_state(
            worker_id=worker_id,
            temperature=temperature,
            warming_state=load.warming_state,
            cooling_state=load.cooling_state
        )
        
        # Record resource utilization
        visualizer.record_resource_utilization(
            worker_id=worker_id,
            utilization={
                "cpu_utilization": load.cpu_utilization,
                "memory_utilization": load.memory_utilization,
                "gpu_utilization": load.gpu_utilization,
                "io_utilization": load.io_utilization,
                "network_utilization": load.network_utilization
            }
        )
    
    # Visualize workload distribution
    visualizer.visualize_workload_distribution(
        worker_assignments=worker_assignments,
        worker_types=worker_types,
        filename="integrated_workload_distribution"
    )
    
    # Simulate execution completion
    execution_data = {}
    
    for worker_id, test_ids in worker_assignments.items():
        for test_id in test_ids:
            # Get assignment
            assignment = load_balancer.get_assignment(test_id)
            
            if assignment:
                # Mark as running
                load_balancer.update_assignment_status(test_id, "running")
                
                # Wait a moment to simulate execution
                time.sleep(0.1)
                
                # Simulate execution result
                success = random.random() > 0.1  # 90% success rate
                status = "completed" if success else "failed"
                
                # Add a bit of randomness to execution time
                estimated_time = assignment.test_requirements.expected_duration
                actual_time = estimated_time * (0.8 + (random.random() * 0.4))
                
                # Record execution time
                execution_data[test_id] = {
                    "estimated_time": estimated_time,
                    "actual_time": actual_time,
                    "workload_type": assignment.test_requirements.test_type.upper() if hasattr(assignment.test_requirements, 'test_type') else "UNKNOWN"
                }
                
                # Record in visualizer
                visualizer.record_execution_time(
                    workload_id=test_id,
                    estimated_time=estimated_time,
                    actual_time=actual_time,
                    workload_type=assignment.test_requirements.test_type,
                    worker_id=worker_id
                )
                
                # Mark as completed
                result = {
                    "output": f"Test result for {test_id}",
                    "success": success,
                    "execution_time": actual_time
                }
                load_balancer.update_assignment_status(test_id, status, result)
                
                logger.info(f"Completed test {test_id} with status {status} and execution time {actual_time:.2f}s (estimated: {estimated_time:.2f}s)")
    
    # Visualize execution times
    visualizer.visualize_execution_times(
        execution_data=execution_data,
        filename="integrated_execution_times"
    )
    
    # Generate HTML report
    report_file = visualizer.generate_summary_report(
        filename="integrated_summary.html",
        include_visualizations=True
    )
    
    logger.info(f"Generated summary report at {report_file}")
    
    # Clean up
    shutdown_integration(load_balancer, workload_manager)


def run_visualization_examples() -> None:
    """Run the visualization examples."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hardware-aware scheduler visualization examples")
    parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--example", type=str, choices=["efficiency", "distribution", "thermal", "resource", "execution", "history", "integrated", "all"], 
                      default="all", help="Example to run (default: all)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir or create_visualization_directory()
    logger.info(f"Saving visualizations to {output_dir}")
    
    # Run the requested example(s)
    if args.example in ["efficiency", "all"]:
        run_efficiency_visualization_example(output_dir)
    
    if args.example in ["distribution", "all"]:
        run_workload_distribution_example(output_dir)
    
    if args.example in ["thermal", "all"]:
        run_thermal_states_example(output_dir)
    
    if args.example in ["resource", "all"]:
        run_resource_utilization_example(output_dir)
    
    if args.example in ["execution", "all"]:
        run_execution_times_example(output_dir)
    
    if args.example in ["history", "all"]:
        run_history_tracking_example(output_dir)
    
    if args.example in ["integrated", "all"]:
        run_integrated_example(output_dir)
    
    logger.info(f"All visualization examples completed. Results are in {output_dir}")
    print(f"\nAll visualization examples completed. Results are in {output_dir}\n")
    print(f"To view the HTML reports, open the following file in a web browser:")
    print(f"  - {os.path.join(output_dir, 'integrated_summary.html')}")
    print(f"  - {os.path.join(output_dir, 'scheduling_summary_example.html')}")


if __name__ == "__main__":
    run_visualization_examples()