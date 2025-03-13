#!/usr/bin/env python3
"""
Custom Scheduler Example for Distributed Testing Framework

This example demonstrates how to use and extend the custom scheduler
system in the distributed testing framework.
"""

import asyncio
import argparse
import logging
import os
import sys
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import distributed testing framework components
from distributed_testing.coordinator import Coordinator
from distributed_testing.worker import Worker
from distributed_testing.task_scheduler import TaskScheduler
from distributed_testing.plugins.scheduler.scheduler_coordinator import SchedulerCoordinator
from distributed_testing.plugins.scheduler.scheduler_plugin_interface import SchedulingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("custom_scheduler_example")

async def setup_coordinator():
    """Setup coordinator with custom scheduler support."""
    # Create coordinator with default task scheduler
    coordinator = Coordinator(
        host="localhost",
        port=8080,
        enable_advanced_scheduler=True
    )
    
    # Initialize coordinator
    await coordinator.initialize()
    
    # Create scheduler coordinator
    scheduler_coordinator = SchedulerCoordinator(
        coordinator=coordinator,
        plugin_dirs=[
            "distributed_testing/plugins/scheduler",
            "plugins/scheduler",
            "scheduler"
        ]
    )
    
    # Initialize scheduler coordinator
    await scheduler_coordinator.initialize()
    
    return coordinator, scheduler_coordinator

async def setup_workers(coordinator, num_workers=4):
    """Setup test workers with different capabilities."""
    workers = []
    
    # Create workers with different capabilities
    for i in range(num_workers):
        # Define worker capabilities
        if i == 0:
            # High-end GPU worker
            capabilities = {
                "hardware": ["CPU", "CUDA"],
                "gpu": {
                    "name": "RTX 4090",
                    "cuda_compute": 8.9,
                    "memory_gb": 24
                },
                "memory": {
                    "total_gb": 64
                },
                "cpu": {
                    "cores": 16
                },
                "hardware_type": "high_end_gpu"
            }
        elif i == 1:
            # Mid-range GPU worker
            capabilities = {
                "hardware": ["CPU", "CUDA"],
                "gpu": {
                    "name": "RTX 3070",
                    "cuda_compute": 8.6,
                    "memory_gb": 8
                },
                "memory": {
                    "total_gb": 32
                },
                "cpu": {
                    "cores": 8
                },
                "hardware_type": "mid_range_gpu"
            }
        elif i == 2:
            # CPU-focused worker
            capabilities = {
                "hardware": ["CPU"],
                "memory": {
                    "total_gb": 128
                },
                "cpu": {
                    "cores": 32
                },
                "hardware_type": "high_end_cpu"
            }
        else:
            # Basic worker
            capabilities = {
                "hardware": ["CPU"],
                "memory": {
                    "total_gb": 16
                },
                "cpu": {
                    "cores": 4
                },
                "hardware_type": "basic"
            }
        
        # Create worker
        worker = Worker(
            coordinator_host="localhost",
            coordinator_port=8080,
            capabilities=capabilities,
            worker_id=f"worker-{i}"
        )
        
        # Register with coordinator (note: in real usage, workers run as separate processes)
        await worker.register_with_coordinator()
        
        workers.append(worker)
    
    return workers

async def create_test_tasks(coordinator, num_tasks=20):
    """Create test tasks with various properties for scheduling."""
    tasks = []
    
    # Define users and projects for fair share testing
    users = ["user1", "user2", "user3", "user4"]
    projects = ["project1", "project2", "project3"]
    
    # Define models and types
    models = [
        "bert-base-uncased", "roberta-base", "t5-small", "t5-base",
        "llama-7b", "vit-base", "clip-vit", "whisper-tiny", "whisper-base"
    ]
    
    # Create tasks with different properties
    for i in range(num_tasks):
        # Randomly select user, project, and model
        user_id = random.choice(users)
        project_id = random.choice(projects)
        model = random.choice(models)
        
        # Determine task type based on model
        if "bert" in model or "roberta" in model:
            task_type = "text_embedding"
        elif "t5" in model or "llama" in model:
            task_type = "text_generation"
        elif "vit" in model or "clip" in model:
            task_type = "vision"
        elif "whisper" in model:
            task_type = "audio"
        else:
            task_type = "benchmark"
        
        # Determine hardware requirements
        hardware_requirements = {}
        
        if "llama" in model and "7b" in model:
            # LLM requires GPU
            hardware_requirements = {
                "hardware": ["CUDA"],
                "min_memory_gb": 8,
                "min_cuda_compute": 8.0
            }
        elif task_type == "vision" or task_type == "audio":
            # Vision and audio benefit from GPU
            hardware_requirements = {
                "hardware": ["CUDA"],
                "min_memory_gb": 4
            }
        
        # Generate random priority (1-10)
        priority = random.randint(1, 10)
        
        # Create task data
        task_data = {
            "task_id": f"task-{i}",
            "type": task_type,
            "model_type": model,
            "user_id": user_id,
            "project_id": project_id,
            "priority": priority,
            "config": {
                "model": model,
                "batch_size": random.choice([1, 2, 4, 8]),
                "precision": random.choice(["fp32", "fp16", "int8"])
            },
            "requirements": hardware_requirements
        }
        
        # Add task to coordinator
        await coordinator.add_task(task_data)
        
        tasks.append(task_data)
    
    return tasks

async def run_scheduling_cycle(coordinator, scheduler_coordinator, strategy=None):
    """Run a full scheduling cycle with specified strategy."""
    # Set strategy if provided
    if strategy:
        await scheduler_coordinator.set_strategy(strategy)
    
    # Get active plugin
    active_plugin = scheduler_coordinator.get_active_plugin_name()
    active_strategy = None
    
    if active_plugin:
        plugin_info = scheduler_coordinator.get_plugin_info(active_plugin)
        if plugin_info:
            active_strategy = plugin_info["active_strategy"]
    
    logger.info(f"Running scheduling cycle with plugin '{active_plugin}' and strategy '{active_strategy}'")
    
    # Schedule pending tasks
    scheduled_count = await coordinator.task_scheduler.schedule_pending_tasks()
    
    logger.info(f"Scheduled {scheduled_count} tasks")
    
    # Process tasks (normally this would be done by workers)
    await process_scheduled_tasks(coordinator)
    
    # Get metrics
    metrics = scheduler_coordinator.get_metrics()
    
    return scheduled_count, metrics

async def process_scheduled_tasks(coordinator):
    """Simulate processing of scheduled tasks."""
    # Get running tasks
    running_tasks = list(coordinator.running_tasks.items())
    
    # Process each running task
    for task_id, worker_id in running_tasks:
        # Get task data
        task_data = coordinator.tasks.get(task_id, {})
        
        # Simulate task execution time
        execution_time = random.uniform(0.5, 5.0)
        
        # Simulate task completion (90% success rate)
        if random.random() < 0.9:
            # Mark task as completed
            await coordinator.mark_task_completed(
                task_id=task_id,
                worker_id=worker_id,
                result={"status": "completed", "execution_time": execution_time}
            )
        else:
            # Mark task as failed
            await coordinator.mark_task_failed(
                task_id=task_id,
                worker_id=worker_id,
                error={"status": "failed", "execution_time": execution_time}
            )

async def main():
    """Run the custom scheduler example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Custom Scheduler Example")
    parser.add_argument("--scheduler", type=str, default="FairnessScheduler",
                       help="Scheduler plugin to use (default: FairnessScheduler)")
    parser.add_argument("--strategy", type=str, default="fair_share",
                       help="Scheduling strategy to use (default: fair_share)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of workers to create (default: 4)")
    parser.add_argument("--num-tasks", type=int, default=20,
                       help="Number of tasks to create (default: 20)")
    parser.add_argument("--cycles", type=int, default=5,
                       help="Number of scheduling cycles to run (default: 5)")
    args = parser.parse_args()
    
    # Setup coordinator and scheduler coordinator
    logger.info("Setting up coordinator and scheduler coordinator...")
    coordinator, scheduler_coordinator = await setup_coordinator()
    
    # Setup workers
    logger.info(f"Setting up {args.num_workers} workers...")
    workers = await setup_workers(coordinator, args.num_workers)
    
    # Create test tasks
    logger.info(f"Creating {args.num_tasks} test tasks...")
    tasks = await create_test_tasks(coordinator, args.num_tasks)
    
    # Display available scheduler plugins
    available_plugins = scheduler_coordinator.get_available_plugins()
    logger.info(f"Available scheduler plugins: {', '.join(available_plugins)}")
    
    # Activate the specified scheduler plugin
    logger.info(f"Activating scheduler plugin '{args.scheduler}'...")
    if args.scheduler not in available_plugins:
        logger.error(f"Scheduler plugin '{args.scheduler}' not found.")
        return
    
    # Activate the scheduler
    success = await scheduler_coordinator.activate_scheduler(args.scheduler)
    if not success:
        logger.error(f"Failed to activate scheduler plugin '{args.scheduler}'")
        return
    
    # Set the strategy
    logger.info(f"Setting scheduling strategy to '{args.strategy}'...")
    success = await scheduler_coordinator.set_strategy(args.strategy)
    if not success:
        logger.warning(f"Failed to set scheduling strategy '{args.strategy}', using default")
    
    # Run scheduling cycles
    for cycle in range(args.cycles):
        logger.info(f"Running scheduling cycle {cycle+1}/{args.cycles}...")
        
        # Run scheduling cycle
        scheduled_count, metrics = await run_scheduling_cycle(coordinator, scheduler_coordinator)
        
        # Display metrics
        if "fairness" in metrics:
            logger.info(f"Fairness metrics:")
            for key, value in metrics["fairness"].items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: {len(value)} entries")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Display active tasks and workers
        active_tasks = len([t for t in coordinator.tasks.values() if t.get("status") == "running"])
        completed_tasks = len(coordinator.completed_tasks)
        failed_tasks = len(coordinator.failed_tasks)
        
        logger.info(f"Tasks: {active_tasks} active, {completed_tasks} completed, {failed_tasks} failed")
        
        # Wait before next cycle
        await asyncio.sleep(1)
    
    # Display final statistics
    logger.info("Final statistics:")
    logger.info(f"Total tasks: {len(coordinator.tasks)}")
    logger.info(f"Completed tasks: {len(coordinator.completed_tasks)}")
    logger.info(f"Failed tasks: {len(coordinator.failed_tasks)}")
    logger.info(f"Pending tasks: {len(coordinator.pending_tasks)}")
    
    # Shutdown
    logger.info("Shutting down...")
    for worker in workers:
        await worker.disconnect()
    
    # Restore original scheduler
    await scheduler_coordinator.restore_original_scheduler()

if __name__ == "__main__":
    asyncio.run(main())