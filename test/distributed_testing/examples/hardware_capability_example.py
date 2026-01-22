#!/usr/bin/env python3
"""
Hardware Capability Example for Distributed Testing Framework

This example demonstrates how to use the enhanced hardware capability detection system
to detect and utilize hardware capabilities in the distributed testing framework.

The example covers:
1. Detecting hardware capabilities on worker nodes
2. Storing capabilities in a DuckDB database
3. Finding workers with specific hardware types
4. Matching workloads to compatible hardware
5. Hardware-aware task scheduling
6. WebGPU/WebNN detection and utilization
7. Interactive visualization of hardware capabilities

Usage:
    python hardware_capability_example.py [--option]
"""

import os
import sys
import json
import logging
import time
import argparse
import uuid
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Import hardware capability detector
    from hardware_capability_detector import (
        HardwareCapabilityDetector, 
        HardwareType, 
        HardwareVendor,
        PrecisionType
    )
except ImportError:
    # Try alternative paths
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    try:
        from distributed_testing.hardware_capability_detector import (
            HardwareCapabilityDetector, 
            HardwareType, 
            HardwareVendor,
            PrecisionType
        )
    except ImportError:
        from test.distributed_testing.hardware_capability_detector import (
            HardwareCapabilityDetector, 
            HardwareType, 
            HardwareVendor,
            PrecisionType
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_capability_example")

# Define hardware-aware task scheduler
class HardwareAwareTaskScheduler:
    """
    Task scheduler that assigns tasks to workers based on hardware capabilities.
    
    This class uses the hardware capability detector to find compatible workers
    for tasks based on their hardware requirements.
    """
    
    def __init__(self, detector: HardwareCapabilityDetector):
        """
        Initialize the task scheduler.
        
        Args:
            detector: Hardware capability detector
        """
        self.detector = detector
        self.workers = {}  # worker_id -> worker_info
        self.tasks = {}    # task_id -> task_info
        self.assignments = {}  # task_id -> worker_id
    
    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        """
        Register a worker with the scheduler.
        
        Args:
            worker_id: Worker ID
            capabilities: Worker hardware capabilities
        """
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "capabilities": capabilities,
            "last_seen": datetime.now(),
            "status": "idle",
            "current_task": None,
            "completed_tasks": []
        }
        logger.info(f"Registered worker {worker_id} with scheduler")
    
    def create_task(self, 
                   task_type: str, 
                   hardware_requirements: Dict[str, Any], 
                   priority: int = 3,
                   min_memory_gb: Optional[float] = None) -> str:
        """
        Create a new task with hardware requirements.
        
        Args:
            task_type: Type of task (e.g., "benchmark", "test")
            hardware_requirements: Hardware requirements
            priority: Task priority (1=highest, 5=lowest)
            min_memory_gb: Minimum memory requirement in GB
            
        Returns:
            Task ID
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create task
        self.tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "hardware_requirements": hardware_requirements,
            "min_memory_gb": min_memory_gb,
            "priority": priority,
            "status": "pending",
            "created": datetime.now(),
            "assigned_worker": None,
            "execution_start": None,
            "execution_end": None,
            "result": None
        }
        
        logger.info(f"Created task {task_id} of type {task_type} with priority {priority}")
        return task_id
    
    def find_compatible_worker(self, task_id: str) -> Optional[str]:
        """
        Find a compatible worker for the given task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Compatible worker ID or None if no compatible worker found
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return None
        
        task = self.tasks[task_id]
        
        # Get hardware requirements
        hardware_requirements = task["hardware_requirements"]
        min_memory_gb = task["min_memory_gb"]
        
        # Find compatible workers
        compatible_workers = self.detector.find_compatible_workers(
            hardware_requirements, min_memory_gb
        )
        
        # Filter out busy workers
        available_workers = [
            worker_id for worker_id in compatible_workers
            if worker_id in self.workers and self.workers[worker_id]["status"] == "idle"
        ]
        
        if not available_workers:
            logger.warning(f"No available workers for task {task_id}")
            return None
        
        # Sort by priority or other criteria
        # For now, just return the first available worker
        return available_workers[0]
    
    def assign_task(self, task_id: str, worker_id: str) -> bool:
        """
        Assign a task to a worker.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            
        Returns:
            True if assignment was successful, False otherwise
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        if worker_id not in self.workers:
            logger.error(f"Worker {worker_id} not found")
            return False
        
        # Check if worker is available
        if self.workers[worker_id]["status"] != "idle":
            logger.error(f"Worker {worker_id} is not idle (status: {self.workers[worker_id]['status']})")
            return False
        
        # Assign task to worker
        self.tasks[task_id]["status"] = "assigned"
        self.tasks[task_id]["assigned_worker"] = worker_id
        
        # Update worker status
        self.workers[worker_id]["status"] = "busy"
        self.workers[worker_id]["current_task"] = task_id
        
        # Record assignment
        self.assignments[task_id] = worker_id
        
        logger.info(f"Assigned task {task_id} to worker {worker_id}")
        return True
    
    def schedule_pending_tasks(self) -> int:
        """
        Schedule pending tasks to compatible workers.
        
        Returns:
            Number of tasks scheduled
        """
        # Get pending tasks
        pending_tasks = {
            task_id: task for task_id, task in self.tasks.items()
            if task["status"] == "pending"
        }
        
        # Sort by priority
        sorted_tasks = sorted(
            pending_tasks.items(),
            key=lambda x: x[1]["priority"]
        )
        
        # Schedule tasks
        scheduled_count = 0
        for task_id, task in sorted_tasks:
            # Find compatible worker
            worker_id = self.find_compatible_worker(task_id)
            if worker_id:
                # Assign task to worker
                if self.assign_task(task_id, worker_id):
                    scheduled_count += 1
        
        logger.info(f"Scheduled {scheduled_count} tasks")
        return scheduled_count
    
    def simulate_task_execution(self) -> None:
        """
        Simulate task execution for demonstration purposes.
        """
        # Find assigned tasks
        assigned_tasks = {
            task_id: task for task_id, task in self.tasks.items()
            if task["status"] == "assigned"
        }
        
        # Simulate execution
        for task_id, task in assigned_tasks.items():
            worker_id = task["assigned_worker"]
            
            # Update status
            self.tasks[task_id]["status"] = "executing"
            self.tasks[task_id]["execution_start"] = datetime.now()
            
            logger.info(f"Simulating execution of task {task_id} on worker {worker_id}")
            
            # Simulate work (with varied execution times based on task type)
            if task["type"] == "benchmark":
                execution_time = random.uniform(1.5, 3.5)
            elif task["type"] == "test":
                execution_time = random.uniform(0.5, 1.5)
            else:
                execution_time = random.uniform(0.2, 1.0)
            
            # Adjust execution time based on hardware affinity
            hw_type = task["hardware_requirements"].get("hardware_type")
            if hw_type == HardwareType.GPU or hw_type == "gpu":
                execution_time *= 0.6  # GPU tasks are faster
            elif hw_type == HardwareType.NPU or hw_type == "npu":
                execution_time *= 0.5  # NPU tasks are even faster
            
            # Simulate execution
            time.sleep(execution_time)
            
            # Update status
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["execution_end"] = datetime.now()
            self.tasks[task_id]["result"] = {
                "execution_time": execution_time,
                "success": True,
                "metrics": {
                    "latency_ms": execution_time * 1000,
                    "throughput": 1.0 / execution_time,
                    "memory_usage_mb": random.uniform(100, 500)
                }
            }
            
            # Update worker status
            self.workers[worker_id]["status"] = "idle"
            self.workers[worker_id]["current_task"] = None
            self.workers[worker_id]["completed_tasks"].append(task_id)
            
            logger.info(f"Completed task {task_id} on worker {worker_id} in {execution_time:.2f}s")
    
    def print_scheduler_status(self) -> None:
        """Print the current status of the scheduler."""
        print("\n===== Scheduler Status =====")
        print(f"Workers: {len(self.workers)}")
        print(f"Tasks: {len(self.tasks)}")
        print(f"Assignments: {len(self.assignments)}")
        
        # Worker status
        print("\n----- Worker Status -----")
        for worker_id, worker in self.workers.items():
            status = worker["status"]
            current_task = worker["current_task"] or "None"
            completed_tasks = len(worker["completed_tasks"])
            print(f"Worker {worker_id}: {status}, Current Task: {current_task}, Completed Tasks: {completed_tasks}")
        
        # Task status
        print("\n----- Task Status -----")
        task_status_counts = {}
        for task in self.tasks.values():
            status = task["status"]
            if status not in task_status_counts:
                task_status_counts[status] = 0
            task_status_counts[status] += 1
        
        for status, count in task_status_counts.items():
            print(f"{status}: {count}")
        
        # Recent completed tasks
        print("\n----- Recent Completed Tasks -----")
        completed_tasks = [task for task in self.tasks.values() if task["status"] == "completed"]
        completed_tasks.sort(key=lambda x: x["execution_end"] or datetime.min, reverse=True)
        
        for task in completed_tasks[:5]:  # Show only the 5 most recent
            task_id = task["task_id"]
            worker_id = task["assigned_worker"]
            execution_time = task["result"]["execution_time"] if task["result"] else 0
            print(f"Task {task_id} completed on {worker_id} in {execution_time:.2f}s")


def run_hardware_capability_example(options: argparse.Namespace) -> None:
    """
    Run the hardware capability example with the given options.
    
    Args:
        options: Command line options
    """
    # Initialize hardware capability detector
    detector = HardwareCapabilityDetector(
        worker_id=options.worker_id,
        db_path=options.db_path,
        enable_browser_detection=options.enable_browser_detection
    )
    
    # Detect hardware capabilities
    print("Detecting hardware capabilities...")
    capabilities = detector.detect_all_capabilities_with_browsers() if options.enable_browser_detection else detector.detect_all_capabilities()
    
    # Display hardware capabilities
    print(f"\nWorker ID: {capabilities.worker_id}")
    print(f"Hostname: {capabilities.hostname}")
    print(f"OS: {capabilities.os_type} {capabilities.os_version}")
    print(f"CPU Count: {capabilities.cpu_count}")
    print(f"Total Memory: {capabilities.total_memory_gb:.2f} GB")
    print(f"Detected {len(capabilities.hardware_capabilities)} hardware capabilities")
    
    # Store capabilities in database if requested
    if options.db_path and not options.detect_only:
        print("\nStoring hardware capabilities in database...")
        detector.store_capabilities(capabilities)
        print(f"Capabilities stored in {options.db_path}")
    
    # Run task scheduling simulation if requested
    if options.task_scheduling:
        print("\nRunning task scheduling simulation...")
        run_task_scheduling_simulation(detector, capabilities)
    
    # Run worker compatibility example if requested
    if options.worker_compatibility:
        print("\nRunning worker compatibility example...")
        run_worker_compatibility_example(detector)
    
    # Output to JSON file if requested
    if options.output_json:
        save_capabilities_to_json(capabilities, options.output_json)


def run_task_scheduling_simulation(detector: HardwareCapabilityDetector, capabilities) -> None:
    """
    Run a task scheduling simulation using the hardware capability detector.
    
    Args:
        detector: Hardware capability detector
        capabilities: Current worker's hardware capabilities
    """
    # Create hardware-aware task scheduler
    scheduler = HardwareAwareTaskScheduler(detector)
    
    # Register this worker
    worker_id = capabilities.worker_id
    scheduler.register_worker(worker_id, capabilities)
    
    # Create additional simulated workers with different hardware
    create_simulated_workers(scheduler)
    
    # Create tasks with different hardware requirements
    create_sample_tasks(scheduler)
    
    # Schedule pending tasks
    print("\nScheduling pending tasks...")
    num_scheduled = scheduler.schedule_pending_tasks()
    print(f"Scheduled {num_scheduled} tasks")
    
    # Simulate task execution
    print("\nSimulating task execution...")
    for i in range(3):  # Run 3 rounds of simulation
        print(f"\nSimulation round {i+1}:")
        scheduler.simulate_task_execution()
        scheduler.schedule_pending_tasks()
    
    # Print final status
    scheduler.print_scheduler_status()


def create_simulated_workers(scheduler: HardwareAwareTaskScheduler) -> None:
    """
    Create simulated workers with different hardware configurations.
    
    Args:
        scheduler: Hardware-aware task scheduler
    """
    # Worker with NVIDIA GPU
    worker_gpu = {
        "worker_id": f"worker_gpu_{uuid.uuid4().hex[:6]}",
        "hostname": "gpu-worker-01",
        "os_type": "Linux",
        "os_version": "Ubuntu 22.04",
        "cpu_count": 16,
        "total_memory_gb": 64.0,
        "hardware_capabilities": [
            {
                "hardware_type": HardwareType.CPU,
                "vendor": HardwareVendor.INTEL,
                "model": "Intel Xeon E5-2680",
                "cores": 16,
                "memory_gb": 64.0
            },
            {
                "hardware_type": HardwareType.GPU,
                "vendor": HardwareVendor.NVIDIA,
                "model": "NVIDIA A100",
                "memory_gb": 40.0,
                "supported_precisions": [
                    PrecisionType.FP32,
                    PrecisionType.FP16,
                    PrecisionType.INT8
                ]
            }
        ]
    }
    
    # Worker with TPU
    worker_tpu = {
        "worker_id": f"worker_tpu_{uuid.uuid4().hex[:6]}",
        "hostname": "tpu-worker-01",
        "os_type": "Linux",
        "os_version": "Debian 11",
        "cpu_count": 32,
        "total_memory_gb": 128.0,
        "hardware_capabilities": [
            {
                "hardware_type": HardwareType.CPU,
                "vendor": HardwareVendor.AMD,
                "model": "AMD EPYC 7742",
                "cores": 32,
                "memory_gb": 128.0
            },
            {
                "hardware_type": HardwareType.TPU,
                "vendor": HardwareVendor.GOOGLE,
                "model": "Google TPU v4",
                "memory_gb": 32.0,
                "supported_precisions": [
                    PrecisionType.FP32,
                    PrecisionType.BF16,
                    PrecisionType.INT8
                ]
            }
        ]
    }
    
    # Worker with NPU
    worker_npu = {
        "worker_id": f"worker_npu_{uuid.uuid4().hex[:6]}",
        "hostname": "npu-worker-01",
        "os_type": "Linux",
        "os_version": "Android 14",
        "cpu_count": 8,
        "total_memory_gb": 16.0,
        "hardware_capabilities": [
            {
                "hardware_type": HardwareType.CPU,
                "vendor": HardwareVendor.QUALCOMM,
                "model": "Qualcomm Snapdragon",
                "cores": 8,
                "memory_gb": 16.0
            },
            {
                "hardware_type": HardwareType.NPU,
                "vendor": HardwareVendor.QUALCOMM,
                "model": "Qualcomm AI Engine",
                "memory_gb": 8.0,
                "supported_precisions": [
                    PrecisionType.FP32,
                    PrecisionType.FP16,
                    PrecisionType.INT8,
                    PrecisionType.INT4
                ]
            }
        ]
    }
    
    # Worker with WebGPU/WebNN
    worker_web = {
        "worker_id": f"worker_web_{uuid.uuid4().hex[:6]}",
        "hostname": "web-worker-01",
        "os_type": "Linux",
        "os_version": "Ubuntu 22.04",
        "cpu_count": 4,
        "total_memory_gb": 8.0,
        "hardware_capabilities": [
            {
                "hardware_type": HardwareType.CPU,
                "vendor": HardwareVendor.INTEL,
                "model": "Intel Core i5",
                "cores": 4,
                "memory_gb": 8.0
            },
            {
                "hardware_type": HardwareType.WEBGPU,
                "vendor": HardwareVendor.NVIDIA,
                "model": "Chrome WebGPU",
                "memory_gb": 2.0,
                "supported_precisions": [
                    PrecisionType.FP32,
                    PrecisionType.FP16
                ]
            },
            {
                "hardware_type": HardwareType.WEBNN,
                "vendor": HardwareVendor.INTEL,
                "model": "Edge WebNN",
                "memory_gb": 1.0,
                "supported_precisions": [
                    PrecisionType.FP32,
                    PrecisionType.FP16
                ]
            }
        ]
    }
    
    # Register workers
    for worker in [worker_gpu, worker_tpu, worker_npu, worker_web]:
        scheduler.register_worker(worker["worker_id"], worker)


def create_sample_tasks(scheduler: HardwareAwareTaskScheduler) -> None:
    """
    Create sample tasks with different hardware requirements.
    
    Args:
        scheduler: Hardware-aware task scheduler
    """
    # GPU compute task
    scheduler.create_task(
        task_type="benchmark",
        hardware_requirements={
            "hardware_type": HardwareType.GPU,
            "vendor": HardwareVendor.NVIDIA
        },
        priority=1,
        min_memory_gb=16.0
    )
    
    # NPU inference task
    scheduler.create_task(
        task_type="benchmark",
        hardware_requirements={
            "hardware_type": HardwareType.NPU,
            "vendor": HardwareVendor.QUALCOMM
        },
        priority=2,
        min_memory_gb=4.0
    )
    
    # WebGPU visualization task
    scheduler.create_task(
        task_type="test",
        hardware_requirements={
            "hardware_type": HardwareType.WEBGPU
        },
        priority=3,
        min_memory_gb=1.0
    )
    
    # CPU-only task
    scheduler.create_task(
        task_type="test",
        hardware_requirements={
            "hardware_type": HardwareType.CPU
        },
        priority=4,
        min_memory_gb=4.0
    )
    
    # TPU computation task
    scheduler.create_task(
        task_type="benchmark",
        hardware_requirements={
            "hardware_type": HardwareType.TPU
        },
        priority=2,
        min_memory_gb=16.0
    )
    
    # WebNN inference task
    scheduler.create_task(
        task_type="inference",
        hardware_requirements={
            "hardware_type": HardwareType.WEBNN
        },
        priority=3,
        min_memory_gb=0.5
    )
    
    # Generic task (no specific hardware)
    scheduler.create_task(
        task_type="utility",
        hardware_requirements={},
        priority=5
    )
    
    # Create a batch of similar tasks
    for i in range(5):
        scheduler.create_task(
            task_type="benchmark",
            hardware_requirements={
                "hardware_type": HardwareType.GPU
            },
            priority=3,
            min_memory_gb=4.0
        )


def run_worker_compatibility_example(detector: HardwareCapabilityDetector) -> None:
    """
    Run an example showing how to find compatible workers for different workloads.
    
    Args:
        detector: Hardware capability detector
    """
    # Create simulated workers in the database if it doesn't exist
    if detector.db_connection:
        create_simulated_workers_in_db(detector)
    
    # Define different workload types
    workloads = [
        {
            "name": "BERT Inference",
            "requirements": {
                "hardware_type": HardwareType.GPU
            },
            "min_memory_gb": 4.0,
            "preferred_hardware_types": [
                HardwareType.GPU, 
                HardwareType.TPU,
                HardwareType.NPU,
                HardwareType.CPU
            ]
        },
        {
            "name": "Vision Model Training",
            "requirements": {
                "hardware_type": HardwareType.GPU,
                "vendor": HardwareVendor.NVIDIA
            },
            "min_memory_gb": 16.0,
            "preferred_hardware_types": [
                HardwareType.GPU,
                HardwareType.TPU
            ]
        },
        {
            "name": "WebGPU Visualization",
            "requirements": {
                "hardware_type": HardwareType.WEBGPU
            },
            "min_memory_gb": 1.0,
            "preferred_hardware_types": [
                HardwareType.WEBGPU,
                HardwareType.GPU
            ]
        },
        {
            "name": "Mobile NPU Inference",
            "requirements": {
                "hardware_type": HardwareType.NPU
            },
            "min_memory_gb": 2.0,
            "preferred_hardware_types": [
                HardwareType.NPU,
                HardwareType.TPU,
                HardwareType.GPU
            ]
        }
    ]
    
    # Find compatible workers for each workload
    print("\n===== Worker Compatibility for Workloads =====")
    for workload in workloads:
        name = workload["name"]
        requirements = workload["requirements"]
        min_memory_gb = workload["min_memory_gb"]
        preferred_hardware_types = workload["preferred_hardware_types"]
        
        compatible_workers = detector.find_compatible_workers(
            requirements, min_memory_gb, preferred_hardware_types
        )
        
        print(f"\nWorkload: {name}")
        print(f"Requirements: {requirements}, Min Memory: {min_memory_gb} GB")
        print(f"Compatible Workers ({len(compatible_workers)}):")
        
        for worker_id in compatible_workers:
            # Get worker capabilities
            worker_capabilities = detector.get_worker_capabilities(worker_id)
            
            if worker_capabilities:
                hostname = worker_capabilities.hostname
                hardware_str = ", ".join([
                    f"{hw.hardware_type.name} ({hw.vendor.name})" 
                    for hw in worker_capabilities.hardware_capabilities
                ])
                
                print(f"  - {worker_id} ({hostname}): {hardware_str}")
            else:
                print(f"  - {worker_id} (capabilities not available)")


def create_simulated_workers_in_db(detector: HardwareCapabilityDetector) -> None:
    """
    Create simulated workers in the database.
    
    Args:
        detector: Hardware capability detector
    """
    from dataclasses import dataclass, field
    
    # Check if workers already exist
    try:
        worker_count = detector.db_connection.execute(
            "SELECT COUNT(*) FROM worker_hardware"
        ).fetchone()[0]
        
        if worker_count > 1:
            logger.info(f"Database already contains {worker_count} workers")
            return
    except Exception as e:
        logger.error(f"Error checking worker count: {str(e)}")
    
    # Create sample workers with hardware profiles
    try:
        from distributed_testing.enhanced_hardware_capability import (
            WorkerHardwareCapabilities,
            HardwareCapability
        )
        
        # Worker with NVIDIA GPU
        worker_gpu = WorkerHardwareCapabilities(
            worker_id=f"worker_gpu_{uuid.uuid4().hex[:6]}",
            hostname="gpu-worker-01",
            os_type="Linux",
            os_version="Ubuntu 22.04",
            cpu_count=16,
            total_memory_gb=64.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.INTEL,
                    model="Intel Xeon E5-2680",
                    cores=16,
                    memory_gb=64.0
                ),
                HardwareCapability(
                    hardware_type=HardwareType.GPU,
                    vendor=HardwareVendor.NVIDIA,
                    model="NVIDIA A100",
                    memory_gb=40.0,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8
                    ]
                )
            ],
            last_updated=time.time()
        )
        
        # Worker with TPU
        worker_tpu = WorkerHardwareCapabilities(
            worker_id=f"worker_tpu_{uuid.uuid4().hex[:6]}",
            hostname="tpu-worker-01",
            os_type="Linux",
            os_version="Debian 11",
            cpu_count=32,
            total_memory_gb=128.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.AMD,
                    model="AMD EPYC 7742",
                    cores=32,
                    memory_gb=128.0
                ),
                HardwareCapability(
                    hardware_type=HardwareType.TPU,
                    vendor=HardwareVendor.GOOGLE,
                    model="Google TPU v4",
                    memory_gb=32.0,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.BF16,
                        PrecisionType.INT8
                    ]
                )
            ],
            last_updated=time.time()
        )
        
        # Worker with NPU
        worker_npu = WorkerHardwareCapabilities(
            worker_id=f"worker_npu_{uuid.uuid4().hex[:6]}",
            hostname="npu-worker-01",
            os_type="Linux",
            os_version="Android 14",
            cpu_count=8,
            total_memory_gb=16.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.QUALCOMM,
                    model="Qualcomm Snapdragon",
                    cores=8,
                    memory_gb=16.0
                ),
                HardwareCapability(
                    hardware_type=HardwareType.NPU,
                    vendor=HardwareVendor.QUALCOMM,
                    model="Qualcomm AI Engine",
                    memory_gb=8.0,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8,
                        PrecisionType.INT4
                    ]
                )
            ],
            last_updated=time.time()
        )
        
        # Worker with WebGPU/WebNN
        worker_web = WorkerHardwareCapabilities(
            worker_id=f"worker_web_{uuid.uuid4().hex[:6]}",
            hostname="web-worker-01",
            os_type="Linux",
            os_version="Ubuntu 22.04",
            cpu_count=4,
            total_memory_gb=8.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.INTEL,
                    model="Intel Core i5",
                    cores=4,
                    memory_gb=8.0
                ),
                HardwareCapability(
                    hardware_type=HardwareType.WEBGPU,
                    vendor=HardwareVendor.NVIDIA,
                    model="Chrome WebGPU",
                    memory_gb=2.0,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16
                    ]
                ),
                HardwareCapability(
                    hardware_type=HardwareType.WEBNN,
                    vendor=HardwareVendor.INTEL,
                    model="Edge WebNN",
                    memory_gb=1.0,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16
                    ]
                )
            ],
            last_updated=time.time()
        )
        
        # Store worker capabilities in database
        for worker in [worker_gpu, worker_tpu, worker_npu, worker_web]:
            detector.store_capabilities(worker)
            logger.info(f"Stored simulated worker {worker.worker_id} in database")
    
    except Exception as e:
        logger.error(f"Error creating simulated workers: {str(e)}")


def save_capabilities_to_json(capabilities, output_file: str) -> None:
    """
    Save capabilities to a JSON file.
    
    Args:
        capabilities: Hardware capabilities
        output_file: Path to output JSON file
    """
    try:
        # Convert capabilities to dictionary for JSON serialization
        capabilities_dict = {
            "worker_id": capabilities.worker_id,
            "hostname": capabilities.hostname,
            "os_type": capabilities.os_type,
            "os_version": capabilities.os_version,
            "cpu_count": capabilities.cpu_count,
            "total_memory_gb": capabilities.total_memory_gb,
            "hardware_capabilities": [],
            "last_updated": datetime.now().isoformat()
        }
        
        # Convert hardware capabilities
        for hw in capabilities.hardware_capabilities:
            hw_type = hw.hardware_type.value if hasattr(hw.hardware_type, 'value') else hw.hardware_type
            vendor = hw.vendor.value if hasattr(hw.vendor, 'value') else hw.vendor
            
            # Convert precisions
            precisions = []
            for p in hw.supported_precisions:
                if hasattr(p, 'value'):
                    precisions.append(p.value)
                else:
                    precisions.append(p)
            
            # Convert scores
            scores = {}
            for k, v in hw.scores.items():
                if hasattr(v, 'value'):
                    scores[k] = v.value
                else:
                    scores[k] = v
            
            # Create hardware capability dict
            hw_dict = {
                "hardware_type": hw_type,
                "vendor": vendor,
                "model": hw.model,
                "version": hw.version,
                "driver_version": hw.driver_version,
                "compute_units": hw.compute_units,
                "cores": hw.cores,
                "memory_gb": hw.memory_gb,
                "supported_precisions": precisions,
                "capabilities": hw.capabilities,
                "scores": scores
            }
            
            capabilities_dict["hardware_capabilities"].append(hw_dict)
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(capabilities_dict, f, indent=2)
        
        print(f"\nCapabilities written to {output_file}")
        
    except Exception as e:
        print(f"\nError writing to JSON file: {str(e)}")


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Hardware Capability Example for Distributed Testing Framework")
    parser.add_argument("--worker-id", help="Worker ID (default: auto-generated)")
    parser.add_argument("--db-path", default="hardware_capabilities.duckdb", help="Path to DuckDB database for storing results")
    parser.add_argument("--enable-browser-detection", action="store_true", help="Enable browser-based WebGPU/WebNN detection")
    parser.add_argument("--detect-only", action="store_true", help="Only detect capabilities, don't store in database")
    parser.add_argument("--output-json", help="Path to output JSON file for capabilities")
    parser.add_argument("--task-scheduling", action="store_true", help="Run task scheduling simulation")
    parser.add_argument("--worker-compatibility", action="store_true", help="Run worker compatibility example")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    
    options = parser.parse_args()
    
    # If --all is specified, enable all examples
    if options.all:
        options.task_scheduling = True
        options.worker_compatibility = True
    
    # Run the example
    run_hardware_capability_example(options)


if __name__ == "__main__":
    main()