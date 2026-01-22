#!/usr/bin/env python3
"""
Demo Script: Running Coordinator with Hardware Monitoring Integration

This script demonstrates the integration of the hardware utilization monitor with 
the coordinator for resource-aware task scheduling. It simulates a distributed
testing environment with multiple workers and tasks, showing how hardware metrics 
influence task scheduling.

Features demonstrated:
1. Coordinator setup with hardware monitoring integration
2. Worker registration with capability detection
3. Task creation and hardware-aware scheduling
4. Real-time resource utilization monitoring
5. Resource-aware task assignment
6. Task execution with hardware utilization tracking
7. HTML report generation with utilization metrics

Usage:
    python run_coordinator_with_hardware_monitoring.py --workers 3 --tasks 10 --duration 60
"""

import os
import sys
import time
import json
import random
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_hardware_demo")

# Import coordinator components
from coordinator import Coordinator
from task_scheduler import TaskScheduler
from worker_registry import WorkerRegistry

# Import hardware monitoring components
from hardware_utilization_monitor import (
    HardwareUtilizationMonitor, 
    MonitoringLevel,
    ResourceUtilization,
    TaskResourceUsage,
    HardwareAlert
)

from hardware_capability_detector import (
    HardwareCapabilityDetector,
    HardwareType,
    HardwareVendor,
    PrecisionType,
    CapabilityScore,
    HardwareCapability,
    WorkerHardwareCapabilities
)

from coordinator_hardware_monitoring_integration import (
    CoordinatorHardwareMonitoringIntegration
)

async def simulate_worker(worker_id: str, coordinator: Coordinator, hardware_capabilities: Dict[str, Any]):
    """
    Simulate a worker node.
    
    Args:
        worker_id: Worker ID
        coordinator: Coordinator instance
        hardware_capabilities: Hardware capabilities of the worker
    """
    logger.info(f"Starting worker {worker_id}")
    
    # Register with coordinator
    registration_data = {
        "worker_id": worker_id,
        "hostname": f"worker-{worker_id}",
        "ip_address": f"192.168.1.{random.randint(10, 200)}",
        "capabilities": hardware_capabilities,
        "status": "active"
    }
    
    await coordinator.register_worker(worker_id, registration_data)
    
    # Simulate worker connection
    coordinator.worker_connections[worker_id] = {
        "connected": True,
        "last_seen": datetime.now().isoformat()
    }
    
    logger.info(f"Worker {worker_id} registered with coordinator")

async def simulate_task_execution(task_id: str, worker_id: str, coordinator: Coordinator, 
                                 integration: CoordinatorHardwareMonitoringIntegration,
                                 execution_time: int = 10,
                                 cpu_intensive: bool = False,
                                 memory_intensive: bool = False):
    """
    Simulate task execution.
    
    Args:
        task_id: Task ID
        worker_id: Worker ID
        coordinator: Coordinator instance
        integration: Hardware monitoring integration instance
        execution_time: Task execution time in seconds
        cpu_intensive: Whether the task is CPU intensive
        memory_intensive: Whether the task is memory intensive
    """
    logger.info(f"Executing task {task_id} on worker {worker_id}")
    
    # Start task monitoring
    integration.start_task_monitoring(task_id, worker_id)
    
    # Update task status
    coordinator.tasks[task_id]["status"] = "running"
    
    # Simulate execution
    start_time = time.time()
    
    try:
        # Execution loop
        for i in range(execution_time):
            # Simulate CPU load for CPU-intensive tasks
            if cpu_intensive:
                # Actually do some computation to generate load
                _ = [i*i for i in range(500000)]
            
            # Simulate memory usage for memory-intensive tasks
            if memory_intensive:
                # Allocate memory to generate load
                big_list = [0] * (1000000 if memory_intensive else 10000)
                
            # Sleep briefly to simulate other work
            await asyncio.sleep(1)
        
        # Task completed successfully
        execution_time_seconds = time.time() - start_time
        
        # Update task result
        task_result = {
            "task_id": task_id,
            "status": "completed",
            "worker_id": worker_id,
            "execution_time_seconds": execution_time_seconds,
            "result": {"success": True, "message": "Task completed successfully"}
        }
        
        # Stop task monitoring
        integration.stop_task_monitoring(task_id, worker_id, success=True)
        
        # Update task status
        coordinator.tasks[task_id]["status"] = "completed"
        coordinator.tasks[task_id]["result"] = task_result
        
        # Move task from running to completed
        coordinator.running_tasks.pop(task_id, None)
        coordinator.completed_tasks.append(task_id)
        
        # Update worker performance
        if hasattr(coordinator, 'task_scheduler'):
            coordinator.task_scheduler.update_worker_performance(worker_id, task_result)
        
        logger.info(f"Task {task_id} completed successfully on worker {worker_id}")
        
    except Exception as e:
        # Task failed
        execution_time_seconds = time.time() - start_time
        
        # Update task result
        task_result = {
            "task_id": task_id,
            "status": "failed",
            "worker_id": worker_id,
            "execution_time_seconds": execution_time_seconds,
            "error": str(e)
        }
        
        # Stop task monitoring
        integration.stop_task_monitoring(task_id, worker_id, success=False, error=str(e))
        
        # Update task status
        coordinator.tasks[task_id]["status"] = "failed"
        coordinator.tasks[task_id]["result"] = task_result
        
        # Move task from running to failed
        coordinator.running_tasks.pop(task_id, None)
        coordinator.failed_tasks.append(task_id)
        
        # Update worker performance
        if hasattr(coordinator, 'task_scheduler'):
            coordinator.task_scheduler.update_worker_performance(worker_id, task_result)
        
        logger.error(f"Task {task_id} failed on worker {worker_id}: {str(e)}")

async def create_test_task(coordinator: Coordinator, task_id: str, task_type: str, 
                        priority: int = 1, model: Optional[str] = None,
                        requires_gpu: bool = False, min_memory_gb: float = 0):
    """
    Create a test task.
    
    Args:
        coordinator: Coordinator instance
        task_id: Task ID
        task_type: Task type (benchmark, test, custom)
        priority: Task priority
        model: Model name (for benchmark tasks)
        requires_gpu: Whether the task requires GPU
        min_memory_gb: Minimum memory required in GB
    """
    # Create task config
    config = {}
    if model:
        config["model"] = model
    
    # Create task requirements
    requirements = {}
    if requires_gpu:
        requirements["hardware"] = ["gpu"]
    if min_memory_gb > 0:
        requirements["min_memory_gb"] = min_memory_gb
    
    # Create task
    task = {
        "task_id": task_id,
        "type": task_type,
        "priority": priority,
        "config": config,
        "requirements": requirements,
        "created": datetime.now().isoformat(),
        "status": "pending"
    }
    
    # Add task to coordinator
    coordinator.tasks[task_id] = task
    coordinator.pending_tasks.append(task_id)
    
    logger.info(f"Created task {task_id} (type: {task_type}, priority: {priority}, requires_gpu: {requires_gpu})")
    
    return task_id

async def simulate_coordinator(args):
    """
    Simulate a coordinator with hardware monitoring integration.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting coordinator with hardware monitoring integration")
    
    # Create coordinator components
    worker_registry = WorkerRegistry()
    
    # Create task scheduler
    task_scheduler = TaskScheduler(
        coordinator=None,  # Will be set later
        prioritize_hardware_match=True,
        load_balance=True,
        consider_worker_performance=True,
        max_tasks_per_worker=args.max_tasks_per_worker,
        enable_task_affinity=True,
        enable_worker_specialization=True,
        enable_predictive_scheduling=True
    )
    
    # Create coordinator
    coordinator = Coordinator(
        worker_registry=worker_registry,
        task_scheduler=task_scheduler
    )
    
    # Set coordinator reference in task scheduler
    task_scheduler.coordinator = coordinator
    
    # Create hardware capability detector
    hardware_detector = HardwareCapabilityDetector(
        db_path=args.db_path
    )
    
    # Create hardware monitoring integration
    integration = CoordinatorHardwareMonitoringIntegration(
        coordinator=coordinator,
        db_path=args.db_path,
        monitoring_level=MonitoringLevel.DETAILED,
        enable_resource_aware_scheduling=True,
        hardware_detector=hardware_detector,
        utilization_threshold=80.0,
        update_interval_seconds=1.0
    )
    
    # Initialize integration
    integration.initialize()
    
    logger.info("Coordinator initialized with hardware monitoring integration")
    
    # Initialize workers
    worker_tasks = []
    
    # Create workers with different capabilities
    for i in range(args.num_workers):
        worker_id = f"worker{i+1}"
        
        # Assign different hardware capabilities to workers
        if i % 3 == 0:
            # CPU-only worker
            capabilities = {
                "hardware": ["cpu"],
                "cpu": {
                    "cores": 8,
                    "architecture": "x86_64"
                },
                "memory": {
                    "total_gb": 16.0,
                    "available_gb": 14.5
                }
            }
        elif i % 3 == 1:
            # CPU + GPU worker
            capabilities = {
                "hardware": ["cpu", "gpu"],
                "cpu": {
                    "cores": 16,
                    "architecture": "x86_64"
                },
                "gpu": {
                    "name": "NVIDIA RTX 3080",
                    "cuda_compute": 8.6,
                    "memory_gb": 10.0
                },
                "memory": {
                    "total_gb": 32.0,
                    "available_gb": 28.5
                }
            }
        else:
            # High-end worker
            capabilities = {
                "hardware": ["cpu", "gpu"],
                "cpu": {
                    "cores": 32,
                    "architecture": "x86_64"
                },
                "gpu": {
                    "name": "NVIDIA RTX 4090",
                    "cuda_compute": 8.9,
                    "memory_gb": 24.0
                },
                "memory": {
                    "total_gb": 64.0,
                    "available_gb": 58.5
                }
            }
        
        worker_task = asyncio.create_task(
            simulate_worker(worker_id, coordinator, capabilities)
        )
        worker_tasks.append(worker_task)
    
    # Wait for all workers to register
    await asyncio.gather(*worker_tasks)
    
    # Start task creation and scheduling
    logger.info("Creating and scheduling tasks")
    
    # Create tasks with different requirements
    task_ids = []
    for i in range(args.num_tasks):
        # Generate random task properties
        task_type = random.choice(["benchmark", "test", "custom"])
        priority = random.randint(1, 5)
        
        # For benchmark tasks, include model name
        model = None
        if task_type == "benchmark":
            model = random.choice([
                "bert-base-uncased", "t5-small", "vit-base", 
                "whisper-tiny", "clip-vit", "llama-7b"
            ])
        
        # Some tasks require GPU
        requires_gpu = random.random() < 0.6  # 60% of tasks require GPU
        
        # Some tasks require more memory
        min_memory_gb = random.choice([0, 0, 0, 4, 8, 16])  # Most tasks don't specify memory
        
        # Create task
        task_id = f"task{i+1}"
        await create_test_task(
            coordinator=coordinator,
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            model=model,
            requires_gpu=requires_gpu,
            min_memory_gb=min_memory_gb
        )
        
        task_ids.append(task_id)
    
    # Schedule initial batch of tasks
    scheduled_count = await coordinator.task_scheduler.schedule_pending_tasks()
    logger.info(f"Scheduled {scheduled_count} tasks initially")
    
    # Execute tasks and schedule more
    task_executions = []
    start_time = time.time()
    
    # Main simulation loop
    try:
        while time.time() - start_time < args.duration and (coordinator.pending_tasks or coordinator.running_tasks):
            # Schedule more tasks if available
            if coordinator.pending_tasks:
                scheduled_count = await coordinator.task_scheduler.schedule_pending_tasks()
                if scheduled_count > 0:
                    logger.info(f"Scheduled {scheduled_count} more tasks")
            
            # Execute running tasks
            current_running = list(coordinator.running_tasks.items())  # Make a copy to avoid modification during iteration
            for task_id, worker_id in current_running:
                # Skip already executing tasks
                if task_id in [t[0] for t in task_executions]:
                    continue
                
                # Determine task properties
                task = coordinator.tasks[task_id]
                task_type = task.get("type", "unknown")
                
                # Determine execution characteristics
                execution_time = random.randint(5, 15)  # Random execution time between 5-15 seconds
                cpu_intensive = random.random() < 0.5   # 50% chance of CPU-intensive task
                memory_intensive = random.random() < 0.3 # 30% chance of memory-intensive task
                
                # Start task execution
                task_execution = asyncio.create_task(
                    simulate_task_execution(
                        task_id, worker_id, coordinator, integration,
                        execution_time=execution_time,
                        cpu_intensive=cpu_intensive,
                        memory_intensive=memory_intensive
                    )
                )
                task_executions.append((task_id, task_execution))
            
            # Wait a bit before checking again
            await asyncio.sleep(1)
            
            # Log current status periodically
            if int(time.time() - start_time) % 10 == 0:
                logger.info(f"Status: {len(coordinator.pending_tasks)} pending, {len(coordinator.running_tasks)} running, {len(coordinator.completed_tasks)} completed, {len(coordinator.failed_tasks)} failed")
                
                # Log worker utilization
                for worker_id, utilization in integration.get_worker_utilization().items():
                    cpu_percent = utilization.get("cpu_percent", 0.0)
                    memory_percent = utilization.get("memory_percent", 0.0)
                    
                    # Get max GPU utilization if available
                    gpu_percent = 0.0
                    gpu_utilization = utilization.get("gpu_utilization", [])
                    if gpu_utilization:
                        gpu_percent = max([gpu.get("load", 0.0) for gpu in gpu_utilization])
                    
                    logger.info(f"Worker {worker_id}: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, GPU {gpu_percent:.1f}%")
    
    finally:
        # Wait for any remaining task executions to complete
        for task_id, task_execution in task_executions:
            if not task_execution.done():
                try:
                    await task_execution
                except Exception as e:
                    logger.error(f"Task {task_id} execution failed: {str(e)}")
        
        # Generate HTML report
        if args.report:
            integration.generate_html_report(args.report)
            logger.info(f"Generated HTML report at {args.report}")
        
        # Shutdown integration
        integration.shutdown()
        
        # Print final statistics
        logger.info("Simulation completed!")
        logger.info(f"Final status: {len(coordinator.completed_tasks)} completed, {len(coordinator.failed_tasks)} failed")
        
        # Print task completion breakdown by worker
        worker_tasks = {}
        for task_id in coordinator.completed_tasks:
            task = coordinator.tasks[task_id]
            if "result" in task and "worker_id" in task["result"]:
                worker_id = task["result"]["worker_id"]
                if worker_id not in worker_tasks:
                    worker_tasks[worker_id] = 0
                worker_tasks[worker_id] += 1
        
        logger.info("Tasks completed by worker:")
        for worker_id, count in worker_tasks.items():
            logger.info(f"Worker {worker_id}: {count} tasks")
        
        # Export results to JSON if requested
        if args.export_json:
            results = {
                "timestamp": datetime.now().isoformat(),
                "duration": args.duration,
                "num_workers": args.num_workers,
                "num_tasks": args.num_tasks,
                "completed_tasks": len(coordinator.completed_tasks),
                "failed_tasks": len(coordinator.failed_tasks),
                "worker_tasks": worker_tasks,
                "task_details": {
                    task_id: {
                        "type": task.get("type"),
                        "status": task.get("status"),
                        "worker_id": task.get("result", {}).get("worker_id") if "result" in task else None,
                        "execution_time": task.get("result", {}).get("execution_time_seconds") if "result" in task else None
                    }
                    for task_id, task in coordinator.tasks.items()
                }
            }
            
            with open(args.export_json, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Exported results to {args.export_json}")

def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Coordinator with Hardware Monitoring Demo")
    parser.add_argument("--db-path", default="./hardware_metrics.duckdb", help="Path to DuckDB database")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of workers to simulate")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to create")
    parser.add_argument("--duration", type=int, default=60, help="Simulation duration in seconds")
    parser.add_argument("--max-tasks-per-worker", type=int, default=2, help="Maximum tasks per worker")
    parser.add_argument("--report", default="hardware_utilization_report.html", help="Path for HTML report")
    parser.add_argument("--export-json", help="Export results to JSON file")
    
    args = parser.parse_args()
    
    # Print demo information
    print("=" * 80)
    print("Coordinator with Hardware Monitoring Demo")
    print("=" * 80)
    print(f"Simulating {args.num_workers} workers and {args.num_tasks} tasks for {args.duration} seconds")
    print(f"Max tasks per worker: {args.max_tasks_per_worker}")
    print(f"Database path: {args.db_path}")
    print(f"HTML report: {args.report}")
    print(f"JSON export: {args.export_json or 'Not enabled'}")
    print("=" * 80)
    
    # Run simulation
    asyncio.run(simulate_coordinator(args))

if __name__ == "__main__":
    main()