#!/usr/bin/env python3
"""
Test runner for the advanced adaptive load balancer of the distributed testing framework.

This script starts a coordinator and a few worker nodes with different capabilities,
creates various tasks, and demonstrates the advanced load balancing features including:

1. Dynamic threshold adjustment based on system-wide load
2. Cost-benefit analysis for migrations
3. Predictive load balancing
4. Resource efficiency considerations
5. Hardware-specific balancing strategies

Usage:
    python run_test_adaptive_load_balancer.py [--options]
"""

import anyio
import argparse
import logging
import signal
import sys
import time
import json
import random
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
coordinator_process = None
worker_processes = []
security_config = {}
coordinator_url = None
api_key = None

async def run_coordinator(db_path='./test_adaptive_load_balancer.duckdb', port=8082):
    """Run the coordinator process with adaptive load balancer enabled."""
    import subprocess
    
    # Delete existing database if it exists to start fresh
    db_file = Path(db_path)
    if db_file.exists():
        os.remove(db_file)
        logger.info(f"Removed existing database: {db_path}")
    
    # Start coordinator with all features enabled
    cmd = [
        'python', 'coordinator.py',
        '--db-path', db_path,
        '--port', str(port),
        '--security-config', './test_adaptive_load_balancer_security.json',
        '--generate-admin-key',
        '--generate-worker-key'
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a bit for coordinator to start
    await anyio.sleep(2)
    
    # Load security config to get API keys
    global security_config, coordinator_url, api_key
    
    with open('./test_adaptive_load_balancer_security.json', 'r') as f:
        security_config = json.load(f)
    
    # Get worker API key
    for key, data in security_config.get('api_keys', {}).items():
        if 'worker' in data.get('roles', []):
            api_key = key
            break
    
    coordinator_url = f"http://localhost:{port}"
    
    logger.info(f"Coordinator started with adaptive load balancer at {coordinator_url}")
    logger.info(f"Worker API key: {api_key}")
    
    return process

async def run_worker(worker_id, capabilities, port=8082, delay=0):
    """Run a worker node process with specified capabilities."""
    import subprocess
    
    # Wait for specified delay
    if delay > 0:
        await anyio.sleep(delay)
    
    # Create capabilities JSON
    capabilities_json = json.dumps(capabilities)
    
    # Start worker process
    cmd = [
        'python', 'worker.py',
        '--coordinator', f"http://localhost:{port}",
        '--api-key', api_key,
        '--worker-id', worker_id,
        '--capabilities', capabilities_json
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    logger.info(f"Worker {worker_id} started with capabilities: {capabilities}")
    
    return process

async def create_task(task_type, config, requirements, priority=1, port=8082):
    """Create a task in the coordinator."""
    import aiohttp
    
    task_data = {
        "type": task_type,
        "priority": priority,
        "config": config,
        "requirements": requirements
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{port}/api/tasks",
            json=task_data,
            headers={"Authorization": f"Bearer {api_key}"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Created task: {data.get('task_id')} ({task_type})")
                return data.get('task_id')
            else:
                logger.error(f"Failed to create task: {await response.text()}")
                return None

async def create_test_tasks(port=8082):
    """Create a diverse set of test tasks to demonstrate load balancing."""
    # Create benchmark tasks for different model types with varying requirements
    
    # CPU-only benchmark task
    await create_task(
        "benchmark",
        {
            "model": "bert-tiny",
            "batch_sizes": [1, 2, 4],
            "precision": "fp32",
            "iterations": 50
        },
        {
            "hardware": ["cpu"],
            "min_memory_gb": 2,
        },
        priority=2
    )
    
    # CUDA benchmark task
    await create_task(
        "benchmark",
        {
            "model": "bert-base-uncased",
            "batch_sizes": [1, 2, 4, 8, 16],
            "precision": "fp16",
            "iterations": 100
        },
        {
            "hardware": ["cuda"],
            "min_memory_gb": 4,
            "min_cuda_compute": 7.0
        },
        priority=1
    )
    
    # Large CUDA benchmark task
    await create_task(
        "benchmark",
        {
            "model": "t5-base",
            "batch_sizes": [1, 2, 4, 8],
            "precision": "fp16",
            "iterations": 100
        },
        {
            "hardware": ["cuda"],
            "min_memory_gb": 8,
            "min_cuda_compute": 7.0
        },
        priority=1
    )
    
    # ROCm benchmark task
    await create_task(
        "benchmark",
        {
            "model": "vit-base",
            "batch_sizes": [1, 4, 16],
            "precision": "fp16",
            "iterations": 100
        },
        {
            "hardware": ["rocm"],
            "min_memory_gb": 4
        },
        priority=3
    )
    
    # Multi-hardware benchmark task
    await create_task(
        "benchmark",
        {
            "model": "whisper-tiny",
            "batch_sizes": [1, 2],
            "precision": "fp32",
            "iterations": 50
        },
        {
            "hardware": ["cpu", "cuda", "rocm"],
            "min_memory_gb": 2
        },
        priority=2
    )
    
    # Power-efficient task
    await create_task(
        "benchmark",
        {
            "model": "mobilevit-small",
            "batch_sizes": [1, 4, 8],
            "precision": "int8",
            "iterations": 50
        },
        {
            "hardware": ["cpu", "openvino", "qnn"],
            "min_memory_gb": 1,
            "power_efficient": True
        },
        priority=2
    )
    
    # WebNN specific benchmark task
    await create_task(
        "benchmark",
        {
            "model": "albert-base-v2",
            "batch_sizes": [1, 2],
            "precision": "fp32",
            "iterations": 30
        },
        {
            "hardware": ["webnn"],
            "min_memory_gb": 1
        },
        priority=3
    )
    
    # WebGPU specific benchmark task
    await create_task(
        "benchmark",
        {
            "model": "clip-vit-base-patch32",
            "batch_sizes": [1, 2],
            "precision": "fp16",
            "iterations": 30
        },
        {
            "hardware": ["webgpu"],
            "min_memory_gb": 2
        },
        priority=3
    )
    
    # Create test tasks
    await create_task(
        "test",
        {
            "test_file": "test_bert_performance.py",
            "test_args": ["--verbose", "--no-cache"]
        },
        {
            "hardware": ["cpu"],
            "min_memory_gb": 2
        },
        priority=2
    )
    
    # Create test task for CUDA
    await create_task(
        "test",
        {
            "test_file": "test_cuda_performance.py",
            "test_args": ["--verbose", "--device", "cuda"]
        },
        {
            "hardware": ["cuda"],
            "min_memory_gb": 4
        },
        priority=2
    )
    
    # CPU-intensive test task
    await create_task(
        "test",
        {
            "test_file": "test_cpu_intensive.py",
            "test_args": ["--stress-test", "--duration", "300"]
        },
        {
            "hardware": ["cpu"],
            "min_cores": 4,
            "min_memory_gb": 4
        },
        priority=3
    )
    
    # Memory-intensive test task
    await create_task(
        "test",
        {
            "test_file": "test_memory_intensive.py",
            "test_args": ["--memory-test", "--usage", "6gb"]
        },
        {
            "hardware": ["cpu"],
            "min_memory_gb": 8
        },
        priority=2
    )
    
    logger.info("Created all test tasks")

async def monitor_system(port=8082, interval=5, duration=600):
    """
    Monitor the system status and log key metrics, focusing on load balancing.
    
    Args:
        port: Coordinator port
        interval: Monitoring interval in seconds
        duration: Total monitoring duration in seconds
    """
    import aiohttp
    
    start_time = time.time()
    end_time = start_time + duration
    
    # Create session for API calls
    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            try:
                # Get system status
                async with session.get(
                    f"http://localhost:{port}/status",
                    headers={"Authorization": f"Bearer {api_key}"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract key metrics
                        worker_count = len(data.get("workers", {}))
                        active_workers = sum(1 for w in data.get("workers", {}).values() if w.get("status") == "active")
                        
                        task_counts = data.get("task_counts", {})
                        pending_tasks = task_counts.get("pending", 0)
                        running_tasks = task_counts.get("running", 0)
                        completed_tasks = task_counts.get("completed", 0)
                        failed_tasks = task_counts.get("failed", 0)
                        
                        # Get load balancer metrics
                        load_balancer_stats = data.get("load_balancer", {})
                        system_utilization = load_balancer_stats.get("system_utilization", {})
                        avg_util = system_utilization.get("average", 0)
                        min_util = system_utilization.get("min", 0)
                        max_util = system_utilization.get("max", 0)
                        imbalance_score = system_utilization.get("imbalance_score", 0)
                        
                        # Migration metrics
                        migrations = load_balancer_stats.get("migrations", {})
                        active_migrations = migrations.get("active", 0)
                        migrations_last_hour = migrations.get("last_hour", 0)
                        
                        # Current thresholds
                        config = load_balancer_stats.get("config", {})
                        high_threshold = config.get("utilization_threshold_high", 0.85)
                        low_threshold = config.get("utilization_threshold_low", 0.2)
                        
                        # Log summary
                        logger.info(
                            f"Status: {active_workers}/{worker_count} workers | "
                            f"Tasks: {pending_tasks} pending, {running_tasks} running, "
                            f"{completed_tasks} completed, {failed_tasks} failed | "
                            f"Load: {avg_util:.2%} avg ({min_util:.2%}-{max_util:.2%}) | "
                            f"Imbalance: {imbalance_score:.2%} | "
                            f"Thresholds: {low_threshold:.2f}-{high_threshold:.2f} | "
                            f"Migrations: {active_migrations} active, {migrations_last_hour} in last hour"
                        )
                    else:
                        logger.error(f"Failed to get status: {await response.text()}")
            
            except Exception as e:
                logger.error(f"Error monitoring system: {str(e)}")
            
            # Wait for next monitoring interval
            await anyio.sleep(interval)

async def create_worker_with_random_load(worker_id, port=8082, base_capabilities=None):
    """Create a worker with specified capabilities and simulate load"""
    import aiohttp
    
    # Set up base capabilities if not provided
    if base_capabilities is None:
        base_capabilities = {
            "hardware": ["cpu"],
            "memory": {"total_gb": 8},
            "max_tasks": 4
        }
    
    # Add worker with specified capabilities
    worker_process = await run_worker(worker_id, base_capabilities, port)
    worker_processes.append(worker_process)
    
    # Wait for worker to register
    await anyio.sleep(2)
    
    # Start task to periodically update worker load
    # TODO: Replace with task group - asyncio.create_task(simulate_worker_load(worker_id, port))
    
    return worker_id

async def simulate_worker_load(worker_id, port=8082, update_interval=10):
    """Simulate varying load on a worker by updating its hardware metrics."""
    import aiohttp
    import random
    import math
    
    # Define load pattern
    pattern_options = ["increasing", "decreasing", "stable", "volatile", "cyclic"]
    pattern = random.choice(pattern_options)
    
    # Base metrics
    cpu_base = random.uniform(0.2, 0.5)
    memory_base = random.uniform(0.3, 0.6)
    gpu_base = random.uniform(0.1, 0.4) if random.random() > 0.5 else 0
    
    # For cyclic pattern
    cycle_period = random.randint(6, 12)  # in update intervals
    cycle_phase = random.uniform(0, 2 * math.pi)
    
    # Create session for API calls
    async with aiohttp.ClientSession() as session:
        step = 0
        while True:
            try:
                # Calculate metrics based on pattern
                if pattern == "increasing":
                    # Gradually increasing load
                    factor = min(1.0, 0.6 + step * 0.03)
                    variation = random.uniform(-0.05, 0.05)
                elif pattern == "decreasing":
                    # Gradually decreasing load
                    factor = max(0.2, 0.8 - step * 0.03)
                    variation = random.uniform(-0.05, 0.05)
                elif pattern == "stable":
                    # Relatively stable load
                    factor = 1.0
                    variation = random.uniform(-0.1, 0.1)
                elif pattern == "volatile":
                    # Highly variable load
                    factor = 1.0
                    variation = random.uniform(-0.3, 0.3)
                elif pattern == "cyclic":
                    # Cyclic load pattern (sinusoidal)
                    factor = 1.0
                    cycle_position = (step / cycle_period) * 2 * math.pi + cycle_phase
                    variation = 0.3 * math.sin(cycle_position)
                
                # Calculate final metrics
                cpu_percent = max(0, min(100, (cpu_base + variation) * 100 * factor))
                memory_percent = max(0, min(100, (memory_base + variation * 0.7) * 100 * factor))
                
                if gpu_base > 0:
                    gpu_utilization = max(0, min(100, (gpu_base + variation) * 100 * factor))
                    gpu_memory = max(0, min(100, (gpu_base + variation * 0.8) * 100 * factor))
                    gpu_metrics = [{"utilization_percent": gpu_utilization, "memory_utilization_percent": gpu_memory}]
                else:
                    gpu_metrics = []
                
                # Prepare hardware metrics
                hardware_metrics = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent
                }
                
                if gpu_metrics:
                    hardware_metrics["gpu"] = gpu_metrics
                
                # Update worker metrics
                async with session.post(
                    f"http://localhost:{port}/api/workers/{worker_id}/metrics",
                    json={"hardware_metrics": hardware_metrics},
                    headers={"Authorization": f"Bearer {api_key}"}
                ) as response:
                    if response.status == 200:
                        pass  # Success, no need to log
                    else:
                        logger.error(f"Failed to update worker metrics: {await response.text()}")
                
                # Increment step counter
                step += 1
                
                # Wait for next update
                await anyio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error simulating worker load: {str(e)}")
                await anyio.sleep(update_interval)

async def add_dynamic_workers(port=8082, delay_between=20, total_workers=10):
    """Add workers dynamically over time to demonstrate system adaptation."""
    # Define various worker capabilities
    worker_templates = [
        {
            "name": "cpu-worker-{id}",
            "capabilities": {
                "hardware": ["cpu"],
                "memory": {"total_gb": 16},
                "cpu": {"cores": 8},
                "max_tasks": 4
            }
        },
        {
            "name": "cuda-worker-{id}",
            "capabilities": {
                "hardware": ["cpu", "cuda"],
                "memory": {"total_gb": 32},
                "cpu": {"cores": 16},
                "gpu": {"cuda_compute": 8.0, "memory_gb": 16},
                "max_tasks": 4
            }
        },
        {
            "name": "rocm-worker-{id}",
            "capabilities": {
                "hardware": ["cpu", "rocm"],
                "memory": {"total_gb": 32},
                "cpu": {"cores": 16},
                "gpu": {"memory_gb": 12},
                "max_tasks": 4
            }
        },
        {
            "name": "openvino-worker-{id}",
            "capabilities": {
                "hardware": ["cpu", "openvino"],
                "memory": {"total_gb": 16},
                "cpu": {"cores": 8},
                "max_tasks": 4
            }
        },
        {
            "name": "efficient-worker-{id}",
            "capabilities": {
                "hardware": ["cpu", "openvino", "qnn"],
                "memory": {"total_gb": 8},
                "cpu": {"cores": 8},
                "max_tasks": 4,
                "energy_efficiency": 0.9
            }
        },
        {
            "name": "web-worker-{id}",
            "capabilities": {
                "hardware": ["cpu", "webnn", "webgpu"],
                "memory": {"total_gb": 8},
                "cpu": {"cores": 4},
                "max_tasks": 2
            }
        }
    ]
    
    # Add initial batch of workers
    initial_count = min(4, total_workers)
    for i in range(initial_count):
        template = random.choice(worker_templates)
        worker_id = template["name"].format(id=i+1)
        await create_worker_with_random_load(worker_id, port, template["capabilities"])
    
    # Add remaining workers with delay
    remaining = total_workers - initial_count
    for i in range(remaining):
        # Wait between adding workers
        await anyio.sleep(delay_between)
        
        # Add a new worker
        template = random.choice(worker_templates)
        worker_id = template["name"].format(id=initial_count+i+1)
        await create_worker_with_random_load(worker_id, port, template["capabilities"])
        
        # Also submit some new tasks occasionally
        if i % 2 == 0:
            await create_test_tasks(port)

async def run_dynamic_environment(port=8082, duration=600):
    """
    Run a dynamic test environment with changing worker availability and load.
    
    This demonstrates how the advanced load balancer adapts to changing conditions.
    """
    # Start monitoring
    monitor_task = # TODO: Replace with task group - asyncio.create_task(monitor_system(port, interval=5, duration=duration))
    
    # Add dynamic workers
    workers_task = # TODO: Replace with task group - asyncio.create_task(add_dynamic_workers(port, delay_between=30, total_workers=8))
    
    # Create initial batch of tasks
    await create_test_tasks(port)
    
    # Wait for test duration
    await anyio.sleep(duration)
    
    # Cancel ongoing tasks
    monitor_task.cancel()
    workers_task.cancel()
    
    try:
        await monitor_task
    except anyio.get_cancelled_exc_class():
        pass
    
    try:
        await workers_task
    except anyio.get_cancelled_exc_class():
        pass

async def cleanup_processes():
    """Clean up all processes."""
    global coordinator_process, worker_processes
    
    # Terminate worker processes
    for process in worker_processes:
        if process:
            process.terminate()
            try:
                process.wait(timeout=2)
            except:
                process.kill()
    
    # Terminate coordinator process
    if coordinator_process:
        coordinator_process.terminate()
        try:
            coordinator_process.wait(timeout=2)
        except:
            coordinator_process.kill()
    
    logger.info("All processes terminated")

async def main(args):
    """Main entry point for the test runner."""
    global coordinator_process
    
    try:
        # Start coordinator
        coordinator_process = await run_coordinator(db_path=args.db_path, port=args.port)
        
        # Run the dynamic test environment
        await run_dynamic_environment(port=args.port, duration=args.run_time)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
    finally:
        # Clean up
        await cleanup_processes()
        logger.info("Test complete")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the advanced adaptive load balancer.")
    parser.add_argument("--port", type=int, default=8082, help="Port for the coordinator server")
    parser.add_argument("--db-path", type=str, default="./test_adaptive_load_balancer.duckdb", help="Path to the DuckDB database")
    parser.add_argument("--run-time", type=int, default=600, help="How long to run the test in seconds")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: None)
    
    # Run the test
    try:
        anyio.run(main(args))
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0)