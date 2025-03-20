#!/usr/bin/env python3
"""
IPFS Accelerate Distributed Testing Framework - Task Submission Tool

This script provides functionality for submitting tasks to the coordinator 
for distributed execution. It supports single tasks, batch submissions,
and periodic task generation for continuous testing scenarios.

Usage:
    python submit_tasks.py --coordinator http://coordinator:8080 --task-file tasks.json
    python submit_tasks.py --coordinator http://coordinator:8080 --generate benchmark --model bert-base-uncased
    python submit_tasks.py --coordinator http://coordinator:8080 --periodic 60 --task-file tasks.json

Examples:
    # Submit tasks from a JSON file
    python submit_tasks.py --coordinator http://localhost:8080 --task-file task_examples.json

    # Generate and submit a benchmark task for a specific model
    python submit_tasks.py --coordinator http://localhost:8080 --generate benchmark \
        --model bert-base-uncased --batch-sizes 1,2,4,8,16 --precision fp16,fp32 --priority 10

    # Submit a test task directly
    python submit_tasks.py --coordinator http://localhost:8080 --submit-test \
        --test-file test_bert.py --test-args "--batch-size 4" --priority 5

    # Submit tasks periodically (every 5 minutes)
    python submit_tasks.py --coordinator http://localhost:8080 --task-file tasks.json --periodic 300
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

import aiohttp
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"task_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# List of available models for task generation
AVAILABLE_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "vit-base-patch16-224",
    "vit-large-patch16-224",
    "t5-small",
    "t5-base",
    "clip-vit-base-patch32",
    "distilbert-base-uncased",
    "gpt2",
    "gpt2-medium",
    "roberta-base",
    "roberta-large"
]

class TaskSubmitter:
    """Task submission client for the distributed testing framework."""
    
    def __init__(
        self,
        coordinator_url: str,
        api_key: Optional[str] = None,
        token: Optional[str] = None
    ):
        """
        Initialize the task submitter.
        
        Args:
            coordinator_url: URL of the coordinator server
            api_key: Optional API key for authentication
            token: Optional JWT token for authentication
        """
        self.coordinator_url = coordinator_url
        self.api_key = api_key
        self.token = token
        self.session = None
        
        # Validate coordinator URL
        if not coordinator_url.startswith(("http://", "https://")):
            self.coordinator_url = f"http://{coordinator_url}"
        
        # Remove trailing slash if present
        self.coordinator_url = self.coordinator_url.rstrip("/")
        
        logger.info(f"Task submitter initialized with coordinator URL: {self.coordinator_url}")
    
    async def connect(self):
        """Connect to the coordinator server."""
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Test connection
        try:
            # Create authentication headers
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            elif self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            # Check coordinator status
            async with self.session.get(f"{self.coordinator_url}/status", headers=headers) as response:
                if response.status == 200:
                    status_data = await response.json()
                    logger.info(f"Connected to coordinator. Status: {status_data.get('status', 'unknown')}")
                    
                    # Log useful info from status if available
                    if "workers" in status_data:
                        logger.info(f"Active workers: {status_data['workers'].get('active', 0)}")
                    if "tasks" in status_data:
                        logger.info(f"Pending tasks: {status_data['tasks'].get('pending', 0)}")
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to connect to coordinator: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to coordinator: {str(e)}")
            return False
    
    async def close(self):
        """Close the connection to the coordinator."""
        if self.session:
            await self.session.close()
            logger.info("Closed connection to coordinator")
    
    async def submit_task(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Submit a single task to the coordinator.
        
        Args:
            task: Task data
            
        Returns:
            Task ID if submission was successful, None otherwise
        """
        if not self.session:
            logger.error("Not connected to coordinator")
            return None
        
        # Create authentication headers
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        try:
            # Submit task
            async with self.session.post(
                f"{self.coordinator_url}/tasks", 
                json=task, 
                headers=headers
            ) as response:
                if response.status == 201:  # Created
                    result = await response.json()
                    task_id = result.get("task_id")
                    logger.info(f"Task submitted successfully, ID: {task_id}")
                    return task_id
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to submit task: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            return None
    
    async def submit_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Submit multiple tasks to the coordinator.
        
        Args:
            tasks: List of task data
            
        Returns:
            List of task IDs for successfully submitted tasks
        """
        if not self.session:
            logger.error("Not connected to coordinator")
            return []
        
        # Create authentication headers
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        try:
            # Submit tasks
            async with self.session.post(
                f"{self.coordinator_url}/tasks/batch", 
                json={"tasks": tasks}, 
                headers=headers
            ) as response:
                if response.status == 201:  # Created
                    result = await response.json()
                    task_ids = result.get("task_ids", [])
                    logger.info(f"Batch of {len(tasks)} tasks submitted successfully")
                    return task_ids
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to submit batch tasks: {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error submitting batch tasks: {str(e)}")
            return []
    
    async def generate_benchmark_task(
        self,
        model: str,
        batch_sizes: List[int] = None,
        precision: List[str] = None,
        iterations: int = 10,
        priority: int = 5,
        hardware_requirements: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a benchmark task for a specific model.
        
        Args:
            model: Model name
            batch_sizes: List of batch sizes to benchmark (default: [1, 2, 4, 8, 16])
            precision: List of precision formats to benchmark (default: ["fp32"])
            iterations: Number of iterations per batch size (default: 10)
            priority: Task priority (default: 5)
            hardware_requirements: Optional hardware requirements
            
        Returns:
            Generated task data
        """
        # Set defaults
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        if precision is None:
            precision = ["fp32"]
        if hardware_requirements is None:
            hardware_requirements = {"gpu": True, "memory_gb": 8}
        
        # Generate task
        task = {
            "type": "benchmark",
            "name": f"{model} Benchmark ({', '.join(precision)})",
            "priority": priority,
            "config": {
                "model": model,
                "batch_sizes": batch_sizes,
                "precision": precision,
                "iterations": iterations,
                "hardware_requirements": hardware_requirements
            }
        }
        
        return task
    
    async def generate_test_task(
        self,
        test_file: str,
        test_args: List[str] = None,
        priority: int = 5,
        hardware_requirements: Dict[str, Any] = None,
        software_requirements: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Generate a test task.
        
        Args:
            test_file: Test file path
            test_args: List of test arguments
            priority: Task priority (default: 5)
            hardware_requirements: Optional hardware requirements
            software_requirements: Optional software requirements
            
        Returns:
            Generated task data
        """
        # Set defaults
        if test_args is None:
            test_args = []
        if hardware_requirements is None:
            hardware_requirements = {}
        if software_requirements is None:
            software_requirements = {}
        
        # Extract test name from file path
        test_name = os.path.basename(test_file)
        if test_name.endswith(".py"):
            test_name = test_name[:-3]
        
        # Generate task
        task = {
            "type": "test",
            "name": f"{test_name} Test",
            "priority": priority,
            "config": {
                "test_file": test_file,
                "test_args": test_args,
                "hardware_requirements": hardware_requirements,
                "software_requirements": software_requirements
            }
        }
        
        return task
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status data or None if not found
        """
        if not self.session:
            logger.error("Not connected to coordinator")
            return None
        
        # Create authentication headers
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        try:
            # Get task status
            async with self.session.get(
                f"{self.coordinator_url}/tasks/{task_id}", 
                headers=headers
            ) as response:
                if response.status == 200:
                    status_data = await response.json()
                    return status_data
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get task status: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            return None
    
    async def wait_for_task_completion(
        self, 
        task_id: str, 
        poll_interval: int = 5,
        timeout: int = 600
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID
            poll_interval: Polling interval in seconds (default: 5)
            timeout: Timeout in seconds (default: 600)
            
        Returns:
            Final task status data or None if timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = await self.get_task_status(task_id)
            
            if status_data:
                status = status_data.get("status")
                logger.info(f"Task {task_id} status: {status}")
                
                if status in ["completed", "failed", "cancelled"]:
                    return status_data
            
            await asyncio.sleep(poll_interval)
        
        logger.error(f"Timeout waiting for task {task_id} completion")
        return None


async def load_tasks_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load tasks from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of task data
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Check if the JSON has a "tasks" key
        if "tasks" in data:
            tasks = data["tasks"]
        else:
            # If not, assume the whole JSON is an array of tasks
            tasks = data
        
        logger.info(f"Loaded {len(tasks)} tasks from {file_path}")
        return tasks
    except Exception as e:
        logger.error(f"Error loading tasks from {file_path}: {str(e)}")
        return []


async def submit_tasks_from_file(submitter: TaskSubmitter, file_path: str) -> List[str]:
    """
    Submit tasks from a JSON file.
    
    Args:
        submitter: TaskSubmitter instance
        file_path: Path to the JSON file
        
    Returns:
        List of submitted task IDs
    """
    # Load tasks
    tasks = await load_tasks_from_file(file_path)
    if not tasks:
        return []
    
    # Submit tasks
    task_ids = await submitter.submit_tasks(tasks)
    
    return task_ids


async def periodic_task_submission(
    submitter: TaskSubmitter,
    file_path: str,
    interval: int,
    max_iterations: Optional[int] = None
):
    """
    Submit tasks periodically.
    
    Args:
        submitter: TaskSubmitter instance
        file_path: Path to the JSON file
        interval: Interval in seconds between submissions
        max_iterations: Maximum number of iterations (default: None, run indefinitely)
    """
    iteration = 0
    
    try:
        while max_iterations is None or iteration < max_iterations:
            logger.info(f"Periodic task submission - iteration {iteration + 1}")
            
            # Submit tasks
            task_ids = await submit_tasks_from_file(submitter, file_path)
            
            if task_ids:
                logger.info(f"Submitted {len(task_ids)} tasks")
            else:
                logger.warning("No tasks were submitted")
            
            # Increment iteration counter
            iteration += 1
            
            # Sleep until next submission
            if max_iterations is None or iteration < max_iterations:
                logger.info(f"Waiting {interval} seconds until next submission")
                await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Periodic task submission cancelled")
    except Exception as e:
        logger.error(f"Error in periodic task submission: {str(e)}")


async def run_task_submitter(args):
    """
    Run the task submitter with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Create task submitter
    submitter = TaskSubmitter(
        coordinator_url=args.coordinator,
        api_key=args.api_key,
        token=args.token
    )
    
    try:
        # Connect to coordinator
        connected = await submitter.connect()
        if not connected:
            logger.error("Failed to connect to coordinator")
            return 1
        
        # Handle different submission modes
        if args.task_file and args.periodic:
            # Periodic submission
            logger.info(f"Starting periodic task submission every {args.periodic} seconds")
            
            try:
                await periodic_task_submission(
                    submitter,
                    args.task_file,
                    args.periodic,
                    args.max_iterations
                )
            except KeyboardInterrupt:
                logger.info("Periodic task submission stopped by user")
            
        elif args.task_file:
            # One-time submission from file
            task_ids = await submit_tasks_from_file(submitter, args.task_file)
            
            if not task_ids:
                logger.error("No tasks were submitted")
                return 1
            
            logger.info(f"Submitted {len(task_ids)} tasks")
            
            # Wait for task completion if requested
            if args.wait:
                logger.info(f"Waiting for tasks to complete (timeout: {args.timeout} seconds)")
                
                completed_count = 0
                failed_count = 0
                cancelled_count = 0
                
                for task_id in task_ids:
                    status_data = await submitter.wait_for_task_completion(
                        task_id,
                        poll_interval=args.poll_interval,
                        timeout=args.timeout
                    )
                    
                    if status_data:
                        status = status_data.get("status")
                        if status == "completed":
                            completed_count += 1
                        elif status == "failed":
                            failed_count += 1
                        elif status == "cancelled":
                            cancelled_count += 1
                
                logger.info(f"Task completion: {completed_count} completed, {failed_count} failed, {cancelled_count} cancelled")
            
        elif args.generate == "benchmark":
            # Generate and submit benchmark task
            batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")] if args.batch_sizes else [1, 2, 4, 8, 16]
            precision = args.precision.split(",") if args.precision else ["fp32"]
            
            # Generate task
            task = await submitter.generate_benchmark_task(
                model=args.model,
                batch_sizes=batch_sizes,
                precision=precision,
                iterations=args.iterations,
                priority=args.priority
            )
            
            # Submit task
            task_id = await submitter.submit_task(task)
            
            if not task_id:
                logger.error("Failed to submit benchmark task")
                return 1
            
            logger.info(f"Submitted benchmark task for model {args.model}, ID: {task_id}")
            
            # Wait for task completion if requested
            if args.wait:
                logger.info(f"Waiting for task completion (timeout: {args.timeout} seconds)")
                
                status_data = await submitter.wait_for_task_completion(
                    task_id,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout
                )
                
                if status_data:
                    status = status_data.get("status")
                    logger.info(f"Task completed with status: {status}")
                    
                    if status == "completed" and args.print_result:
                        result = status_data.get("result", {})
                        print(json.dumps(result, indent=2))
            
        elif args.submit_test:
            # Generate and submit test task
            test_args = args.test_args.split() if args.test_args else []
            
            # Generate task
            task = await submitter.generate_test_task(
                test_file=args.test_file,
                test_args=test_args,
                priority=args.priority
            )
            
            # Submit task
            task_id = await submitter.submit_task(task)
            
            if not task_id:
                logger.error("Failed to submit test task")
                return 1
            
            logger.info(f"Submitted test task for file {args.test_file}, ID: {task_id}")
            
            # Wait for task completion if requested
            if args.wait:
                logger.info(f"Waiting for task completion (timeout: {args.timeout} seconds)")
                
                status_data = await submitter.wait_for_task_completion(
                    task_id,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout
                )
                
                if status_data:
                    status = status_data.get("status")
                    logger.info(f"Task completed with status: {status}")
                    
                    if status == "completed" and args.print_result:
                        result = status_data.get("result", {})
                        print(json.dumps(result, indent=2))
        
        else:
            logger.error("No action specified (--task-file, --generate, or --submit-test)")
            return 1
        
    finally:
        # Clean up
        await submitter.close()
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IPFS Accelerate Distributed Testing Framework - Task Submission Tool")
    
    # Coordinator connection
    parser.add_argument("--coordinator", default="http://localhost:8080",
                      help="URL of the coordinator server (default: http://localhost:8080)")
    
    # Authentication
    parser.add_argument("--api-key", help="API key for authentication with coordinator")
    parser.add_argument("--token", help="JWT token for authentication (alternative to API key)")
    
    # Task file submission
    parser.add_argument("--task-file", help="Path to the JSON file containing task definitions")
    
    # Generate task
    parser.add_argument("--generate", choices=["benchmark", "test"],
                      help="Generate and submit a specific type of task")
    
    # Benchmark task options
    parser.add_argument("--model", help="Model name for benchmark task")
    parser.add_argument("--batch-sizes", help="Comma-separated list of batch sizes (e.g., '1,2,4,8,16')")
    parser.add_argument("--precision", help="Comma-separated list of precision formats (e.g., 'fp16,fp32')")
    parser.add_argument("--iterations", type=int, default=10,
                      help="Number of iterations per batch size (default: 10)")
    
    # Test task options
    parser.add_argument("--submit-test", action="store_true", help="Submit a test task")
    parser.add_argument("--test-file", help="Test file path for test task")
    parser.add_argument("--test-args", help="Arguments for test task (space-separated)")
    
    # Common options
    parser.add_argument("--priority", type=int, default=5,
                      help="Task priority (default: 5)")
    
    # Periodic submission
    parser.add_argument("--periodic", type=int, help="Submit tasks periodically (interval in seconds)")
    parser.add_argument("--max-iterations", type=int, help="Maximum number of iterations for periodic submission")
    
    # Waiting and result options
    parser.add_argument("--wait", action="store_true", help="Wait for task completion")
    parser.add_argument("--timeout", type=int, default=600,
                      help="Timeout in seconds when waiting for task completion (default: 600)")
    parser.add_argument("--poll-interval", type=int, default=5,
                      help="Polling interval in seconds when waiting for task completion (default: 5)")
    parser.add_argument("--print-result", action="store_true", help="Print task result (requires --wait)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.generate == "benchmark" and not args.model:
        parser.error("--model is required when using --generate benchmark")
    
    if args.submit_test and not args.test_file:
        parser.error("--test-file is required when using --submit-test")
    
    # Run the submitter
    return asyncio.run(run_task_submitter(args))


if __name__ == "__main__":
    sys.exit(main())