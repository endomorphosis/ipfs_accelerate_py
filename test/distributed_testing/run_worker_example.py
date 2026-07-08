#!/usr/bin/env python3
"""
IPFS Accelerate Distributed Testing Framework - Worker Example

This script demonstrates how to run a worker node for the distributed testing framework.
It provides a complete working example with configuration options and demonstrates
best practices for worker deployment.

Usage:
    python run_worker_example.py --coordinator http://coordinator:8080 --api-key YOUR_API_KEY

Examples:
    # Run with default settings (connects to localhost)
    python run_worker_example.py

    # Connect to a specific coordinator with authentication
    python run_worker_example.py --coordinator http://coordinator.example.com:8080 --api-key abcdef1234567890

    # Run with hardware-specific tags and custom worker name
    python run_worker_example.py --coordinator http://coordinator.example.com:8080 --api-key YOUR_API_KEY --tags gpu,cuda,transformers --name gpu-worker-01

    # Connect to a coordinator with a token file and database for result caching
    python run_worker_example.py --coordinator http://coordinator.example.com:8080 --token-file ./worker_token.txt --db-path ./worker_results.duckdb
"""

import argparse
import anyio
import json
import logging
import os
import platform
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

# Add parent directory to path to make imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the worker implementation
from .worker import DistributedTestingWorker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Optional - set up more verbose logging for debugging
# logging.getLogger('worker').setLevel(logging.DEBUG)


def parse_tags(tags_str: str) -> Dict[str, Any]:
    """
    Parse tags string into capability dictionary.
    
    Args:
        tags_str: Comma-separated tags (e.g., "gpu,cuda,transformers")
        
    Returns:
        Dictionary with tags mapped to capability structure
    """
    tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
    
    # Create structured capabilities
    capabilities = {}
    
    # Hardware-related tags
    hardware_tags = ['cpu', 'gpu', 'cuda', 'rocm', 'mps', 'openvino', 'vulkan', 
                    'webgpu', 'webnn', 'dsp', 'npu', 'qualcomm']
    
    for tag in tags:
        if tag in hardware_tags:
            if 'hardware' not in capabilities:
                capabilities['hardware'] = []
            capabilities['hardware'].append(tag)
        else:
            # Software or model-related tags
            if 'software' not in capabilities:
                capabilities['software'] = {}
            capabilities['software'][tag] = True
    
    return capabilities


async def run_worker(args):
    """
    Run worker node with specified arguments.
    
    Args:
        args: Command line arguments
    """
    # Create log directory if needed
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, f"worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        # Add file handler with the specified log directory
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    # Parse capability tags if provided
    additional_capabilities = {}
    if args.tags:
        additional_capabilities = parse_tags(args.tags)
    
    # Add profile-based capabilities
    if args.profile:
        if 'software' not in additional_capabilities:
            additional_capabilities['software'] = {}
        additional_capabilities['software']['profile'] = args.profile
    
    # Add performance tuning capabilities
    if args.optimize_for:
        if 'software' not in additional_capabilities:
            additional_capabilities['software'] = {}
        additional_capabilities['software']['optimize_for'] = args.optimize_for
    
    if args.mixed_precision:
        if 'software' not in additional_capabilities:
            additional_capabilities['software'] = {}
        additional_capabilities['software']['mixed_precision'] = True
    
    # Parse coordinator registry for high availability
    coordinator_registry = []
    if args.coordinator_registry:
        coordinator_registry = [url.strip() for url in args.coordinator_registry.split(',') if url.strip()]
    
    # Always add the primary coordinator to the registry
    if args.coordinator not in coordinator_registry:
        coordinator_registry.insert(0, args.coordinator)
    
    # Load token from file if specified
    token = args.token
    if args.token_file and not token:
        try:
            with open(args.token_file, 'r') as f:
                token = f.read().strip()
            logger.info(f"Loaded token from {args.token_file}")
        except Exception as e:
            logger.error(f"Failed to read token from file: {str(e)}")
    
    # Create worker with specified configuration
    hostname = args.name or platform.node()
    worker = DistributedTestingWorker(
        coordinator_url=args.coordinator,
        hostname=hostname,
        db_path=args.db_path,
        worker_id=args.worker_id,
        api_key=args.api_key,
        token=token,
    )
    
    # Configure worker based on additional arguments
    worker.heartbeat_interval = args.heartbeat_interval
    
    # Add health check configuration
    worker.health_check_interval = args.health_check_interval
    worker.health_limits = {
        'cpu_percent': args.max_cpu_percent,
        'memory_percent': args.max_memory_percent,
        'gpu_percent': args.max_gpu_percent
    }
    
    # Add task execution configuration
    worker.max_concurrent_tasks = args.max_concurrent_tasks
    if args.task_timeout:
        worker.task_timeout = args.task_timeout
    
    # Add high availability configuration
    if args.high_availability:
        worker.high_availability_mode = True
        worker.coordinator_registry = coordinator_registry
        worker.reconnect_interval = args.reconnect_interval
        worker.reconnect_attempts = args.reconnect_attempts
        logger.info(f"High availability mode enabled with {len(coordinator_registry)} coordinators")
    
    # Add performance tuning configuration
    worker.memory_buffer = args.memory_buffer
    worker.enable_swap = args.enable_swap
    worker.cache_models = args.cache_models
    if args.cuda_streams:
        worker.cuda_streams = args.cuda_streams
    
    # Add recovery options
    if args.recover:
        worker.recovery_mode = True
        logger.info("Recovery mode enabled")
    
    if args.reset_state:
        worker.reset_state = True
        logger.info("State reset enabled")
    
    # Store the additional capabilities for inclusion during hardware detection
    worker.additional_capabilities = additional_capabilities
    
    # Log worker configuration
    logger.info(f"Starting worker with ID {worker.worker_id}")
    logger.info(f"Connecting to coordinator at {args.coordinator}")
    logger.info(f"Using hostname: {hostname}")
    logger.info(f"Execution profile: {args.profile}")
    logger.info(f"Max concurrent tasks: {args.max_concurrent_tasks}")
    
    if args.db_path:
        logger.info(f"Database path: {args.db_path}")
    
    if additional_capabilities:
        logger.info(f"Additional capabilities: {json.dumps(additional_capabilities, indent=2)}")
    
    # Start the worker
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.exception(f"Error running worker: {str(e)}")
        return 1
    
    return 0


def main():
    """Main entry point for the worker example."""
    parser = argparse.ArgumentParser(description="IPFS Accelerate Distributed Testing Framework - Worker Example")
    
    # Coordinator connection options
    parser.add_argument("--coordinator", default="http://localhost:8080", 
                      help="URL of the coordinator server (default: http://localhost:8080)")
    
    # Authentication options
    parser.add_argument("--api-key", help="API key for authentication with coordinator")
    parser.add_argument("--token", help="JWT token for authentication (alternative to API key)")
    parser.add_argument("--token-file", help="Path to file containing JWT token")
    
    # Worker identity options
    parser.add_argument("--name", help="Friendly name for the worker (default: system hostname)")
    parser.add_argument("--worker-id", help="Worker ID (default: generated UUID)")
    
    # Storage options
    parser.add_argument("--db-path", help="Path to DuckDB database for caching results (optional)")
    parser.add_argument("--log-dir", help="Directory for log files (default: ./logs)")
    
    # Capability options
    parser.add_argument("--tags", help="Comma-separated list of capability tags (e.g., 'gpu,cuda,transformers')")
    
    # High availability options
    parser.add_argument("--high-availability", action="store_true", 
                      help="Enable high availability mode for better fault tolerance")
    parser.add_argument("--reconnect-interval", type=int, default=5, 
                      help="Initial interval between reconnection attempts in seconds (default: 5)")
    parser.add_argument("--coordinator-registry", 
                      help="Comma-separated list of coordinator URLs for fallback")
    
    # Health monitoring options
    parser.add_argument("--health-check-interval", type=int, default=30,
                      help="Interval between health checks in seconds (default: 30)")
    parser.add_argument("--max-cpu-percent", type=float, default=95.0,
                      help="Maximum CPU usage percentage before throttling (default: 95.0)")
    parser.add_argument("--max-memory-percent", type=float, default=95.0,
                      help="Maximum memory usage percentage before throttling (default: 95.0)")
    parser.add_argument("--max-gpu-percent", type=float, default=95.0,
                      help="Maximum GPU memory usage percentage before throttling (default: 95.0)")
    
    # Task execution options
    parser.add_argument("--profile", choices=["benchmark", "test", "light"], default="test",
                      help="Predefined execution profile (default: test)")
    parser.add_argument("--max-concurrent-tasks", type=int,
                      help="Maximum number of tasks to execute concurrently (profile-dependent)")
    parser.add_argument("--task-timeout", type=int,
                      help="Default timeout for task execution in seconds (profile-dependent)")
    
    # Performance options
    parser.add_argument("--memory-buffer", type=float, default=0.1,
                      help="Percentage of memory to reserve as buffer (default: 0.1)")
    parser.add_argument("--enable-swap", action="store_true",
                      help="Allow use of disk space for memory overflow")
    parser.add_argument("--cache-models", action="store_true",
                      help="Keep frequently used models in memory")
    parser.add_argument("--cuda-streams", type=int,
                      help="Number of CUDA streams for GPU tasks")
    parser.add_argument("--optimize-for", choices=["throughput", "latency"], default="throughput",
                      help="Optimization target for task execution (default: throughput)")
    parser.add_argument("--mixed-precision", action="store_true",
                      help="Enable automatic mixed precision where supported")
    
    # Recovery options
    parser.add_argument("--recover", action="store_true",
                      help="Trigger recovery mode on startup")
    parser.add_argument("--reset-state", action="store_true",
                      help="Reset worker state on startup")
    
    # Basic options
    parser.add_argument("--heartbeat-interval", type=int, default=10, 
                      help="Heartbeat interval in seconds (default: 10)")
    parser.add_argument("--reconnect-attempts", type=int, default=10, 
                      help="Maximum number of reconnection attempts (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Apply profile-dependent defaults
    if args.profile == "benchmark" and not args.max_concurrent_tasks:
        args.max_concurrent_tasks = 2
    elif args.profile == "test" and not args.max_concurrent_tasks:
        args.max_concurrent_tasks = 4
    elif args.profile == "light" and not args.max_concurrent_tasks:
        args.max_concurrent_tasks = 8
    
    # Run worker
    return anyio.run(run_worker(args))


if __name__ == "__main__":
    sys.exit(main())