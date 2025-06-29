#!/usr/bin/env python3
"""
Distributed Testing Framework - Task Creator

This script helps create and submit tasks to the distributed testing framework.
It provides a command-line interface for creating benchmark, test, and custom tasks.

Usage:
    python create_task.py --type benchmark --model bert-base-uncased --hardware cuda --batch-sizes 1,2,4,8,16 --priority 1
    python create_task.py --type test --test-file /path/to/test.py --hardware cpu --priority 2
    python create_task.py --type custom --name custom-task --args arg1,arg2 --priority 3
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_task(
    task_type: str,
    coordinator_url: str,
    api_key: str,
    priority: int = 1,
    config: Dict[str, Any] = None,
    requirements: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a task in the distributed testing framework.
    
    Args:
        task_type: Type of task (benchmark, test, custom)
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        priority: Task priority (higher number = higher priority)
        config: Task configuration
        requirements: Task requirements
        
    Returns:
        Server response as a dictionary
    """
    # Prepare task data
    task_data = {
        "type": task_type,
        "priority": priority,
        "config": config or {},
        "requirements": requirements or {},
    }
    
    # Add API URL if not present
    if not coordinator_url.endswith("/api/tasks"):
        if not coordinator_url.endswith("/"):
            coordinator_url += "/"
        coordinator_url += "api/tasks"
    
    # Send request to coordinator
    headers = {"X-API-Key": api_key} if api_key else {}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(coordinator_url, json=task_data, headers=headers) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result
            else:
                error_text = await resp.text()
                raise ValueError(f"Error creating task: {resp.status} - {error_text}")

async def create_benchmark_task(
    coordinator_url: str,
    api_key: str,
    model: str,
    batch_sizes: List[int],
    precision: str = "fp32",
    iterations: int = 10,
    hardware: List[str] = None,
    min_memory_gb: int = None,
    min_cuda_compute: float = None,
    priority: int = 1
) -> Dict[str, Any]:
    """
    Create a benchmark task.
    
    Args:
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        model: Model name to benchmark
        batch_sizes: List of batch sizes to test
        precision: Precision to use (fp32, fp16, int8, etc.)
        iterations: Number of iterations per batch size
        hardware: Required hardware types
        min_memory_gb: Minimum memory required (GB)
        min_cuda_compute: Minimum CUDA compute capability
        priority: Task priority (higher number = higher priority)
        
    Returns:
        Server response as a dictionary
    """
    # Prepare task configuration
    config = {
        "model": model,
        "batch_sizes": batch_sizes,
        "precision": precision,
        "iterations": iterations,
    }
    
    # Prepare task requirements
    requirements = {}
    if hardware:
        requirements["hardware"] = hardware
    if min_memory_gb is not None:
        requirements["min_memory_gb"] = min_memory_gb
    if min_cuda_compute is not None:
        requirements["min_cuda_compute"] = min_cuda_compute
    
    # Create task
    return await create_task(
        task_type="benchmark",
        coordinator_url=coordinator_url,
        api_key=api_key,
        priority=priority,
        config=config,
        requirements=requirements
    )

async def create_test_task(
    coordinator_url: str,
    api_key: str,
    test_file: str,
    test_args: List[str] = None,
    hardware: List[str] = None,
    min_memory_gb: int = None,
    priority: int = 1
) -> Dict[str, Any]:
    """
    Create a test task.
    
    Args:
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        test_file: Path to test file
        test_args: Arguments to pass to test
        hardware: Required hardware types
        min_memory_gb: Minimum memory required (GB)
        priority: Task priority (higher number = higher priority)
        
    Returns:
        Server response as a dictionary
    """
    # Prepare task configuration
    config = {
        "test_file": test_file,
        "test_args": test_args or [],
    }
    
    # Prepare task requirements
    requirements = {}
    if hardware:
        requirements["hardware"] = hardware
    if min_memory_gb is not None:
        requirements["min_memory_gb"] = min_memory_gb
    
    # Create task
    return await create_task(
        task_type="test",
        coordinator_url=coordinator_url,
        api_key=api_key,
        priority=priority,
        config=config,
        requirements=requirements
    )

async def create_custom_task(
    coordinator_url: str,
    api_key: str,
    name: str,
    args: List[str] = None,
    hardware: List[str] = None,
    min_memory_gb: int = None,
    priority: int = 1
) -> Dict[str, Any]:
    """
    Create a custom task.
    
    Args:
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        name: Name of custom task
        args: Arguments for custom task
        hardware: Required hardware types
        min_memory_gb: Minimum memory required (GB)
        priority: Task priority (higher number = higher priority)
        
    Returns:
        Server response as a dictionary
    """
    # Prepare task configuration
    config = {
        "name": name,
        "args": args or [],
    }
    
    # Prepare task requirements
    requirements = {}
    if hardware:
        requirements["hardware"] = hardware
    if min_memory_gb is not None:
        requirements["min_memory_gb"] = min_memory_gb
    
    # Create task
    return await create_task(
        task_type="custom",
        coordinator_url=coordinator_url,
        api_key=api_key,
        priority=priority,
        config=config,
        requirements=requirements
    )

async def load_api_key(security_config_path: str) -> Optional[str]:
    """
    Load API key from security configuration.
    
    Args:
        security_config_path: Path to security configuration file
        
    Returns:
        API key or None if not found
    """
    if not os.path.exists(security_config_path):
        logger.warning(f"Security configuration file not found: {security_config_path}")
        return None
    
    try:
        with open(security_config_path, 'r') as f:
            config = json.load(f)
            
        # Look for an admin API key
        for key, info in config.get("api_keys", {}).items():
            if "admin" in info.get("roles", []):
                logger.info(f"Found admin API key in security configuration")
                return key
            
        # If no admin key, look for any key with worker role
        for key, info in config.get("api_keys", {}).items():
            if "worker" in info.get("roles", []):
                logger.info(f"Found worker API key in security configuration")
                return key
                
        logger.warning("No suitable API key found in security configuration")
        return None
        
    except Exception as e:
        logger.error(f"Error loading security configuration: {str(e)}")
        return None

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create tasks for the distributed testing framework")
    parser.add_argument("--coordinator", default="http://localhost:8080",
                       help="URL of coordinator server")
    parser.add_argument("--api-key",
                       help="API key for authentication")
    parser.add_argument("--security-config", default="./security_config.json",
                       help="Path to security configuration file")
    parser.add_argument("--type", choices=["benchmark", "test", "custom"], required=True,
                       help="Type of task to create")
    parser.add_argument("--priority", type=int, default=1,
                       help="Task priority (higher number = higher priority)")
    
    # Arguments for benchmark tasks
    parser.add_argument("--model",
                       help="Model name for benchmark task")
    parser.add_argument("--batch-sizes", type=lambda s: [int(x) for x in s.split(',')],
                       help="Comma-separated list of batch sizes for benchmark task")
    parser.add_argument("--precision", default="fp32",
                       help="Precision for benchmark task (fp32, fp16, int8, etc.)")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations for benchmark task")
    
    # Arguments for test tasks
    parser.add_argument("--test-file",
                       help="Path to test file for test task")
    parser.add_argument("--test-args", type=lambda s: s.split(','),
                       help="Comma-separated list of arguments for test task")
    
    # Arguments for custom tasks
    parser.add_argument("--name",
                       help="Name for custom task")
    parser.add_argument("--args", type=lambda s: s.split(','),
                       help="Comma-separated list of arguments for custom task")
    
    # Common task requirements
    parser.add_argument("--hardware", type=lambda s: s.split(','),
                       help="Comma-separated list of required hardware types")
    parser.add_argument("--min-memory-gb", type=int,
                       help="Minimum memory required (GB)")
    parser.add_argument("--min-cuda-compute", type=float,
                       help="Minimum CUDA compute capability")
    
    args = parser.parse_args()
    
    # If API key not provided, try to load from security config
    api_key = args.api_key
    if not api_key:
        api_key = await load_api_key(args.security_config)
        if not api_key:
            logger.warning("No API key provided or found in security config. Authentication may fail.")
    
    try:
        # Create task based on type
        if args.type == "benchmark":
            if not args.model or not args.batch_sizes:
                logger.error("Model and batch sizes are required for benchmark tasks")
                sys.exit(1)
                
            result = await create_benchmark_task(
                coordinator_url=args.coordinator,
                api_key=api_key,
                model=args.model,
                batch_sizes=args.batch_sizes,
                precision=args.precision,
                iterations=args.iterations,
                hardware=args.hardware,
                min_memory_gb=args.min_memory_gb,
                min_cuda_compute=args.min_cuda_compute,
                priority=args.priority
            )
            
        elif args.type == "test":
            if not args.test_file:
                logger.error("Test file is required for test tasks")
                sys.exit(1)
                
            result = await create_test_task(
                coordinator_url=args.coordinator,
                api_key=api_key,
                test_file=args.test_file,
                test_args=args.test_args,
                hardware=args.hardware,
                min_memory_gb=args.min_memory_gb,
                priority=args.priority
            )
            
        elif args.type == "custom":
            if not args.name:
                logger.error("Name is required for custom tasks")
                sys.exit(1)
                
            result = await create_custom_task(
                coordinator_url=args.coordinator,
                api_key=api_key,
                name=args.name,
                args=args.args,
                hardware=args.hardware,
                min_memory_gb=args.min_memory_gb,
                priority=args.priority
            )
        
        # Print result
        logger.info(f"Task created successfully: {json.dumps(result, indent=2)}")
        
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())