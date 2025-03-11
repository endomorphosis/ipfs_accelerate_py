#!/usr/bin/env python3
"""
Distributed Testing Framework - Test Script for Intelligent Task Scheduler

This script demonstrates the capabilities of the intelligent task scheduler
by creating workers with different hardware capabilities and submitting various tasks.

Usage:
    python run_test_intelligent_scheduler.py --host localhost --port 8080 --num-workers 3 --run-time 60
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import necessary modules or show helpful error
try:
    import aiohttp
    import websockets
    import duckdb
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    logger.error("Please install required packages: pip install aiohttp websockets duckdb")
    sys.exit(1)

async def run_coordinator(args):
    """Run the coordinator server."""
    # Import here to avoid circular imports
    from coordinator import DistributedTestingCoordinator
    
    # Create database path if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.db_path)), exist_ok=True)
    
    # Create coordinator
    coordinator = DistributedTestingCoordinator(
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        security_config=args.security_config,
        enable_health_monitor=True,
        enable_auto_recovery=True,
        enable_advanced_scheduler=True,
        enable_load_balancer=True
    )
    
    # Configure task scheduler to enable intelligent scheduling
    if coordinator.task_scheduler:
        coordinator.task_scheduler.max_tasks_per_worker = 2  # Allow 2 tasks per worker for demonstration
        coordinator.task_scheduler.enable_task_affinity = True
        coordinator.task_scheduler.enable_worker_specialization = True
        coordinator.task_scheduler.enable_predictive_scheduling = True
        logger.info("Intelligent task scheduler configured")
    
    # Generate API key for worker authentication
    api_key = coordinator.security_manager.generate_api_key("worker-node", ["worker"])
    logger.info("Generated API key for worker authentication")
    
    # Save security config if needed
    if args.security_config:
        coordinator.security_manager.save_config(args.security_config)
        logger.info(f"Saved security configuration to {args.security_config}")
    
    # Start coordinator
    await coordinator.start()
    
    return api_key

async def run_worker(args, worker_id, api_key, capabilities):
    """Run a worker node with specific capabilities."""
    # Import here to avoid circular imports
    from worker import DistributedTestingWorker
    
    # Create worker client
    worker = DistributedTestingWorker(
        coordinator_url=f"http://{args.host}:{args.port}",
        api_key=api_key,
        worker_id=worker_id,
        capabilities=capabilities
    )
    
    # Start worker
    await worker.start()
    
    return worker

async def create_benchmark_task(args, model, batch_sizes, hardware, min_memory_gb, priority=1):
    """Create a benchmark task."""
    async with aiohttp.ClientSession() as session:
        # Prepare task data
        task_data = {
            "type": "benchmark",
            "priority": priority,
            "config": {
                "model": model,
                "batch_sizes": batch_sizes,
                "precision": "fp16",
                "iterations": 10
            },
            "requirements": {
                "hardware": hardware,
                "min_memory_gb": min_memory_gb
            }
        }
        
        # Add API URL
        api_url = f"http://{args.host}:{args.port}/api/tasks"
        
        # Send request
        async with session.post(api_url, json=task_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                logger.info(f"Task created: {result['task_id']} (model: {model}, hardware: {hardware})")
                return result
            else:
                error_text = await resp.text()
                logger.error(f"Error creating task: {resp.status} - {error_text}")
                return None

async def create_test_task(args, test_file, hardware, min_memory_gb, priority=1):
    """Create a test task."""
    async with aiohttp.ClientSession() as session:
        # Prepare task data
        task_data = {
            "type": "test",
            "priority": priority,
            "config": {
                "test_file": test_file,
                "test_args": ["--verbose"]
            },
            "requirements": {
                "hardware": hardware,
                "min_memory_gb": min_memory_gb
            }
        }
        
        # Add API URL
        api_url = f"http://{args.host}:{args.port}/api/tasks"
        
        # Send request
        async with session.post(api_url, json=task_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                logger.info(f"Task created: {result['task_id']} (file: {test_file}, hardware: {hardware})")
                return result
            else:
                error_text = await resp.text()
                logger.error(f"Error creating task: {resp.status} - {error_text}")
                return None

async def get_status(args):
    """Get coordinator status."""
    async with aiohttp.ClientSession() as session:
        # Add API URL
        api_url = f"http://{args.host}:{args.port}/status"
        
        # Send request
        async with session.get(api_url) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                return None

async def test_intelligent_scheduler(args):
    """Run a test of the intelligent task scheduler."""
    # Start coordinator
    api_key = await run_coordinator(args)
    
    # Define worker capabilities
    worker_configs = [
        {
            "id": "worker-cuda-1",
            "capabilities": {
                "hardware": ["cpu", "cuda"],
                "memory": {"total_gb": 16},
                "gpu": {"name": "NVIDIA RTX 3080", "count": 1, "memory_gb": 10, "cuda_compute": 8.6},
                "cpu": {"model": "Intel Core i9", "cores": 8, "threads": 16}
            }
        },
        {
            "id": "worker-cuda-2",
            "capabilities": {
                "hardware": ["cpu", "cuda"],
                "memory": {"total_gb": 32},
                "gpu": {"name": "NVIDIA A100", "count": 1, "memory_gb": 40, "cuda_compute": 8.0},
                "cpu": {"model": "AMD EPYC", "cores": 16, "threads": 32}
            }
        },
        {
            "id": "worker-cpu",
            "capabilities": {
                "hardware": ["cpu"],
                "memory": {"total_gb": 64},
                "cpu": {"model": "Intel Xeon", "cores": 32, "threads": 64}
            }
        },
        {
            "id": "worker-rocm",
            "capabilities": {
                "hardware": ["cpu", "rocm"],
                "memory": {"total_gb": 16},
                "gpu": {"name": "AMD MI100", "count": 1, "memory_gb": 32},
                "cpu": {"model": "AMD Ryzen", "cores": 12, "threads": 24}
            }
        }
    ]
    
    # Start workers
    workers = []
    for config in worker_configs[:args.num_workers]:  # Limit to requested number of workers
        worker = await run_worker(args, config["id"], api_key, config["capabilities"])
        workers.append(worker)
        logger.info(f"Started worker {config['id']} with capabilities: {config['capabilities']['hardware']}")
    
    # Wait for workers to register
    await asyncio.sleep(2)
    
    # Create tasks to demonstrate intelligent scheduling
    tasks = []
    
    # Bert family tasks (should be scheduled together due to affinity)
    tasks.append(await create_benchmark_task(args, "bert-base-uncased", [1, 2, 4], ["cuda"], 6, priority=2))
    tasks.append(await create_benchmark_task(args, "bert-large-uncased", [1, 2], ["cuda"], 10, priority=1))
    tasks.append(await create_benchmark_task(args, "distilbert-base-uncased", [1, 2, 4, 8], ["cuda"], 4, priority=3))
    
    # T5 family tasks
    tasks.append(await create_benchmark_task(args, "t5-small", [1, 2, 4], ["cuda"], 4, priority=2))
    tasks.append(await create_benchmark_task(args, "t5-base", [1, 2], ["cuda"], 8, priority=1))
    
    # Vision model tasks
    tasks.append(await create_benchmark_task(args, "vit-base", [1, 2, 4], ["cuda"], 6, priority=2))
    tasks.append(await create_benchmark_task(args, "clip-vit", [1, 2], ["cuda"], 8, priority=3))
    
    # CPU-only tasks
    tasks.append(await create_benchmark_task(args, "bert-tiny", [1, 2, 4, 8, 16], ["cpu"], 2, priority=2))
    tasks.append(await create_test_task(args, "/path/to/test_cpu.py", ["cpu"], 1, priority=1))
    
    # ROCM tasks if we have a ROCM worker
    if any("rocm" in w.capabilities.get("hardware", []) for w in workers):
        tasks.append(await create_benchmark_task(args, "bert-base-uncased", [1, 2, 4], ["rocm"], 6, priority=2))
        tasks.append(await create_benchmark_task(args, "resnet50", [1, 4, 8, 16], ["rocm"], 4, priority=1))
    
    # Run for specified time
    start_time = datetime.now()
    run_time = args.run_time
    
    logger.info(f"Running test for {run_time} seconds...")
    
    while (datetime.now() - start_time).total_seconds() < run_time:
        # Get status
        status = await get_status(args)
        if status:
            # Log status
            logger.info(f"Workers: {status['workers']['total']} ({status['workers']['active']} active, {status['workers']['busy']} busy)")
            logger.info(f"Tasks: {sum(status['tasks'].values())} (pending: {status['tasks']['pending']}, running: {status['tasks']['running']}, completed: {status['tasks']['completed']})")
        
        # Wait before checking again
        await asyncio.sleep(5)
        
        # Add a new task every 10 seconds to see dynamic scheduling
        if (datetime.now() - start_time).total_seconds() % 10 < 5:
            # Choose a random model and hardware
            models = ["bert-base-uncased", "t5-small", "vit-base", "bert-tiny", "clip-vit"]
            hardwares = [["cuda"], ["cpu"], ["rocm"] if any("rocm" in w.capabilities.get("hardware", []) for w in workers) else ["cuda"]]
            
            model = random.choice(models)
            hardware = random.choice(hardwares)
            memory = random.choice([2, 4, 6, 8])
            
            await create_benchmark_task(args, model, [1, 2], hardware, memory, priority=random.randint(1, 3))
    
    # Get final status to see statistics from the intelligent scheduler
    status = await get_status(args)
    if status:
        logger.info(f"Final status: {json.dumps(status, indent=2)}")
    
    # Shut down workers
    for worker in workers:
        await worker.shutdown()
    
    logger.info("Test completed")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the intelligent task scheduler")
    parser.add_argument("--host", default="localhost", help="Host for the coordinator")
    parser.add_argument("--port", type=int, default=8080, help="Port for the coordinator")
    parser.add_argument("--db-path", default="./test_intelligent_scheduler.duckdb", help="Path to database")
    parser.add_argument("--security-config", default="./test_security_config.json", help="Path to security config")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of workers to start")
    parser.add_argument("--run-time", type=int, default=60, help="How long to run the test in seconds")
    
    args = parser.parse_args()
    
    try:
        await test_intelligent_scheduler(args)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())