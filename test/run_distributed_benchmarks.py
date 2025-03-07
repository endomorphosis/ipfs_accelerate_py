#!/usr/bin/env python3
"""
Distributed Benchmarking Script

This script integrates the distributed testing framework with the benchmarking system
to enable parallel execution of benchmarks across multiple machines.

Usage:
    python run_distributed_benchmarks.py --coordinator localhost:8080 --workers 10 
                                        --models bert,t5,whisper 
                                        --hardware cuda,cpu,openvino
                                        --db-path ./benchmark_db.duckdb
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("distributed_benchmarks.log")
    ]
)
logger = logging.getLogger(__name__)

# Try importing necessary benchmarking modules
try:
    sys.path.append(str(Path(__file__).resolve().parent))
    from distributed_testing import coordinator as dtf_coordinator
    logger.info("Successfully imported distributed testing coordinator module")
except ImportError as e:
    logger.error(f"Failed to import distributed testing module: {e}")
    logger.error("Make sure the distributed testing framework is properly installed")
    sys.exit(1)

# Constants for benchmark configuration
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_PRECISION = "fp16"
DEFAULT_ITERATIONS = 50
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_SEQUENCE_LENGTH = 128

class DistributedBenchmarkManager:
    """Manages distributed benchmark execution using the distributed testing framework."""
    
    def __init__(
        self,
        coordinator_host: str = "localhost",
        coordinator_port: int = 8080,
        db_path: str = "./benchmark_db.duckdb",
        worker_count: int = 5,
        models: List[str] = None,
        hardware_platforms: List[str] = None,
        batch_sizes: List[int] = None,
        precision: str = DEFAULT_PRECISION,
        iterations: int = DEFAULT_ITERATIONS,
        warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        profile_memory: bool = True,
        profile_power: bool = True,
    ):
        """
        Initialize the distributed benchmark manager.
        
        Args:
            coordinator_host: Host for the coordinator server
            coordinator_port: Port for the coordinator server
            db_path: Path to the DuckDB database
            worker_count: Number of worker nodes to use
            models: List of models to benchmark
            hardware_platforms: List of hardware platforms to test
            batch_sizes: List of batch sizes to test
            precision: Precision format (fp32, fp16, int8, int4)
            iterations: Number of iterations for each benchmark
            warmup_iterations: Number of warmup iterations
            sequence_length: Sequence length for text models
            profile_memory: Whether to profile memory usage
            profile_power: Whether to profile power consumption
        """
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.db_path = db_path
        self.worker_count = worker_count
        self.models = models or ["bert-base-uncased", "t5-small", "whisper-tiny"]
        self.hardware_platforms = hardware_platforms or ["cpu", "cuda", "openvino"]
        self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
        self.precision = precision
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.sequence_length = sequence_length
        self.profile_memory = profile_memory
        self.profile_power = profile_power
        
        # Coordinator instance
        self.coordinator = None
        
        # Run ID
        self.run_id = str(uuid.uuid4())
        
        # Tasks
        self.benchmark_tasks = []
        
        logger.info(f"Initialized distributed benchmark manager with run ID {self.run_id}")
    
    async def start_coordinator(self):
        """Start the coordinator server."""
        logger.info(f"Starting coordinator on {self.coordinator_host}:{self.coordinator_port}")
        
        # Initialize coordinator
        self.coordinator = dtf_coordinator.DistributedTestingCoordinator(
            db_path=self.db_path,
            host=self.coordinator_host,
            port=self.coordinator_port
        )
        
        # Start coordinator in background
        await self.coordinator.start()
    
    def generate_benchmark_tasks(self):
        """Generate benchmark tasks for all model-hardware-batch size combinations."""
        tasks = []
        priority = 1
        
        for model in self.models:
            for hardware in self.hardware_platforms:
                # Set task requirements based on hardware
                if hardware == "cuda":
                    requirements = {
                        "hardware": ["cuda"],
                        "min_memory_gb": 4
                    }
                elif hardware == "rocm":
                    requirements = {
                        "hardware": ["rocm"],
                        "min_memory_gb": 4
                    }
                elif hardware == "openvino":
                    requirements = {
                        "hardware": ["openvino"],
                        "min_memory_gb": 2
                    }
                elif hardware == "mps":
                    requirements = {
                        "hardware": ["mps"],
                        "min_memory_gb": 2
                    }
                elif hardware == "webgpu":
                    requirements = {
                        "hardware": ["webgpu"],
                    }
                elif hardware == "webnn":
                    requirements = {
                        "hardware": ["webnn"],
                    }
                elif hardware == "qualcomm":
                    requirements = {
                        "hardware": ["qualcomm"],
                        "min_memory_gb": 2
                    }
                else:  # Default to CPU
                    requirements = {
                        "hardware": ["cpu"],
                        "min_memory_gb": 2
                    }
                
                # Create benchmark task
                task = {
                    "type": "benchmark",
                    "priority": priority,
                    "config": {
                        "model": model,
                        "hardware": hardware,
                        "batch_sizes": self.batch_sizes,
                        "precision": self.precision,
                        "iterations": self.iterations,
                        "warmup_iterations": self.warmup_iterations,
                        "sequence_length": self.sequence_length,
                        "profile_memory": self.profile_memory,
                        "profile_power": self.profile_power,
                        "run_id": self.run_id
                    },
                    "requirements": requirements
                }
                
                tasks.append(task)
                
                # Increment priority for next task
                priority += 1
        
        self.benchmark_tasks = tasks
        logger.info(f"Generated {len(tasks)} benchmark tasks")
        
        return tasks
    
    async def submit_tasks(self):
        """Submit benchmark tasks to the coordinator."""
        if not self.coordinator:
            logger.error("Coordinator not started, cannot submit tasks")
            return
        
        if not self.benchmark_tasks:
            self.generate_benchmark_tasks()
        
        logger.info(f"Submitting {len(self.benchmark_tasks)} tasks to coordinator")
        
        # Prepare for HTTP requests
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            for i, task in enumerate(self.benchmark_tasks):
                # Submit task to coordinator
                try:
                    async with session.post(
                        f"http://{self.coordinator_host}:{self.coordinator_port}/api/tasks",
                        json=task
                    ) as resp:
                        result = await resp.json()
                        logger.info(f"Task {i+1}/{len(self.benchmark_tasks)} submission result: {result}")
                        
                        # Small delay between submissions to avoid overwhelming coordinator
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error submitting task {i+1}: {str(e)}")
    
    async def monitor_progress(self, timeout_minutes=60):
        """Monitor benchmark progress until completion or timeout."""
        if not self.coordinator:
            logger.error("Coordinator not started, cannot monitor progress")
            return
        
        import aiohttp
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        logger.info(f"Monitoring benchmark progress for up to {timeout_minutes} minutes")
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout_seconds:
                try:
                    # Get status from coordinator
                    async with session.get(
                        f"http://{self.coordinator_host}:{self.coordinator_port}/status"
                    ) as resp:
                        status = await resp.json()
                        
                        # Extract task stats
                        tasks = status.get("tasks", {})
                        total = tasks.get("total", 0)
                        pending = tasks.get("pending", 0)
                        running = tasks.get("running", 0)
                        completed = tasks.get("completed", 0)
                        failed = tasks.get("failed", 0)
                        
                        # Calculate progress
                        if total > 0:
                            progress = (completed + failed) / total * 100
                        else:
                            progress = 0
                        
                        logger.info(f"Progress: {progress:.1f}% ({completed}/{total} completed, {failed} failed, {running} running, {pending} pending)")
                        
                        # Check if all tasks are done
                        if pending == 0 and running == 0 and (completed + failed) == total and total > 0:
                            logger.info("All benchmark tasks completed!")
                            return True
                        
                except Exception as e:
                    logger.error(f"Error getting status: {str(e)}")
                
                # Wait before checking again
                await asyncio.sleep(10)
        
        logger.warning(f"Benchmark monitoring timed out after {timeout_minutes} minutes")
        return False
    
    async def create_benchmark_report(self):
        """Create benchmark report from the results."""
        if not self.coordinator:
            logger.error("Coordinator not started, cannot create report")
            return
        
        logger.info("Creating benchmark report from results")
        
        # Here, implement code to query the database and create a benchmark report
        # For now, just log a placeholder message
        logger.info(f"Benchmark report would be created from run_id {self.run_id}")
        logger.info("Check the database for detailed results")
        
        # You can extend this to create a real report using the database query tools
    
    async def run_benchmarks(self, timeout_minutes=60):
        """Run the complete benchmark process."""
        try:
            # Start coordinator
            await self.start_coordinator()
            
            # Generate and submit tasks
            self.generate_benchmark_tasks()
            await self.submit_tasks()
            
            # Monitor progress
            completed = await self.monitor_progress(timeout_minutes)
            
            # Create report
            if completed:
                await self.create_benchmark_report()
            
            logger.info("Benchmark process completed")
            
        except Exception as e:
            logger.error(f"Error running benchmarks: {str(e)}")


async def main():
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Run distributed benchmarks")
    
    # Coordinator options
    parser.add_argument("--coordinator", default="localhost:8080", 
                        help="Coordinator host:port")
    
    # Database options
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                        help="Path to DuckDB database")
    
    # Worker options
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of worker nodes to use")
    
    # Benchmark options
    parser.add_argument("--models", default="bert-base-uncased,t5-small,whisper-tiny",
                        help="Comma-separated list of models to benchmark")
    
    parser.add_argument("--hardware", default="cpu,cuda,openvino",
                        help="Comma-separated list of hardware platforms to test")
    
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32",
                        help="Comma-separated list of batch sizes to test")
    
    parser.add_argument("--precision", default=DEFAULT_PRECISION,
                        choices=["fp32", "fp16", "int8", "int4"],
                        help="Precision format for benchmarks")
    
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="Number of iterations for each benchmark")
    
    parser.add_argument("--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS,
                        help="Number of warmup iterations before timing")
    
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help="Sequence length for text models")
    
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in minutes for the benchmark process")
    
    parser.add_argument("--no-profile-memory", action="store_true",
                        help="Disable memory profiling")
    
    parser.add_argument("--no-profile-power", action="store_true",
                        help="Disable power consumption profiling")
    
    args = parser.parse_args()
    
    # Parse coordinator host and port
    if ":" in args.coordinator:
        host, port = args.coordinator.split(":")
        port = int(port)
    else:
        host = args.coordinator
        port = 8080
    
    # Parse lists
    models = args.models.split(",") if args.models else None
    hardware = args.hardware.split(",") if args.hardware else None
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")] if args.batch_sizes else None
    
    # Create benchmark manager
    benchmark_manager = DistributedBenchmarkManager(
        coordinator_host=host,
        coordinator_port=port,
        db_path=args.db_path,
        worker_count=args.workers,
        models=models,
        hardware_platforms=hardware,
        batch_sizes=batch_sizes,
        precision=args.precision,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        sequence_length=args.sequence_length,
        profile_memory=not args.no_profile_memory,
        profile_power=not args.no_profile_power,
    )
    
    # Run benchmarks
    await benchmark_manager.run_benchmarks(args.timeout)

if __name__ == "__main__":
    asyncio.run(main())