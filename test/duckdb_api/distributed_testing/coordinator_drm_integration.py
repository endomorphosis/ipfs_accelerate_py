#!/usr/bin/env python3
"""
Distributed Testing Framework - Dynamic Resource Management Demo

This script demonstrates the Dynamic Resource Management (DRM) system 
by simulating a distributed testing environment with adaptive scaling.

The demo creates a coordinator with DRM enabled and simulates workers 
with different resource profiles and task workloads to show the scaling
behavior of the system.
"""

import os
import sys
import json
import time
import random
import logging
import threading
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("drm_demo")

# Add parent directory to path to import modules correctly
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager, ScalingDecision
from duckdb_api.distributed_testing.resource_performance_predictor import ResourcePerformancePredictor
from duckdb_api.distributed_testing.cloud_provider_manager import CloudProviderManager


class MockWorker:
    """Mock worker for DRM demo."""
    
    def __init__(self, worker_id, resources):
        """
        Initialize mock worker.
        
        Args:
            worker_id: Worker ID
            resources: Worker resources
        """
        self.worker_id = worker_id
        self.resources = resources
        self.tasks = []
        self.resources_over_time = []
        self.running = True
    
    def get_resources(self):
        """Get current resource state."""
        return self.resources
    
    def execute_task(self, task):
        """
        Simulate task execution with resource consumption.
        
        Args:
            task: Task to execute
        
        Returns:
            dict: Task result
        """
        logger.info(f"Worker {self.worker_id} executing task {task['task_id']}")
        
        # Simulate resource usage
        cpu_cores_used = task.get("requirements", {}).get("cpu_cores", 1)
        memory_mb_used = task.get("requirements", {}).get("memory_mb", 1024)
        gpu_memory_mb_used = task.get("requirements", {}).get("gpu_memory_mb", 0)
        
        # Update available resources
        self.resources["cpu"]["available_cores"] -= cpu_cores_used
        self.resources["memory"]["available_mb"] -= memory_mb_used
        
        if gpu_memory_mb_used > 0 and "gpu" in self.resources:
            self.resources["gpu"]["available_memory_mb"] -= gpu_memory_mb_used
        
        # Add task to active tasks
        self.tasks.append({
            "task_id": task["task_id"],
            "start_time": time.time(),
            "cpu_cores": cpu_cores_used,
            "memory_mb": memory_mb_used,
            "gpu_memory_mb": gpu_memory_mb_used
        })
        
        # Record resource state
        self.resources_over_time.append({
            "timestamp": time.time(),
            "resources": self.resources.copy()
        })
        
        # Simulate execution time
        execution_time = random.uniform(
            task.get("requirements", {}).get("execution_time_min", 5),
            task.get("requirements", {}).get("execution_time_max", 15)
        )
        
        # Sleep for execution time
        time.sleep(execution_time)
        
        # Create result
        result = {
            "task_id": task["task_id"],
            "status": "completed",
            "execution_time_seconds": execution_time,
            "execution_metrics": {
                "execution_time_seconds": execution_time,
                "peak_cpu_cores": cpu_cores_used,
                "peak_memory_mb": memory_mb_used,
                "peak_gpu_memory_mb": gpu_memory_mb_used
            }
        }
        
        # Release resources
        self.resources["cpu"]["available_cores"] += cpu_cores_used
        self.resources["memory"]["available_mb"] += memory_mb_used
        
        if gpu_memory_mb_used > 0 and "gpu" in self.resources:
            self.resources["gpu"]["available_memory_mb"] += gpu_memory_mb_used
        
        # Remove task from active tasks
        self.tasks = [t for t in self.tasks if t["task_id"] != task["task_id"]]
        
        # Record resource state after task completion
        self.resources_over_time.append({
            "timestamp": time.time(),
            "resources": self.resources.copy()
        })
        
        logger.info(f"Worker {self.worker_id} completed task {task['task_id']}")
        return result


class MockCloudProvider:
    """Mock cloud provider for DRM demo."""
    
    def __init__(self, name):
        """
        Initialize mock cloud provider.
        
        Args:
            name: Provider name
        """
        self.name = name
        self.workers = {}
        self.worker_counter = 0
    
    def create_worker(self, config):
        """
        Create a worker instance.
        
        Args:
            config: Worker configuration
        
        Returns:
            dict: Worker information
        """
        worker_id = f"{self.name}-worker-{self.worker_counter}"
        self.worker_counter += 1
        
        # Create resources based on worker type
        worker_type = config.get("worker_type", "default")
        
        if worker_type == "gpu":
            resources = {
                "cpu": {"cores": 16, "physical_cores": 8, "available_cores": 16},
                "memory": {"total_mb": 32768, "available_mb": 32768},
                "gpu": {"devices": 2, "available_devices": 2, "total_memory_mb": 16384, "available_memory_mb": 16384}
            }
        elif worker_type == "memory":
            resources = {
                "cpu": {"cores": 8, "physical_cores": 4, "available_cores": 8},
                "memory": {"total_mb": 65536, "available_mb": 65536},
                "gpu": {"devices": 0, "available_devices": 0, "total_memory_mb": 0, "available_memory_mb": 0}
            }
        elif worker_type == "cpu":
            resources = {
                "cpu": {"cores": 32, "physical_cores": 16, "available_cores": 32},
                "memory": {"total_mb": 16384, "available_mb": 16384},
                "gpu": {"devices": 0, "available_devices": 0, "total_memory_mb": 0, "available_memory_mb": 0}
            }
        else:  # default
            resources = {
                "cpu": {"cores": 4, "physical_cores": 2, "available_cores": 4},
                "memory": {"total_mb": 8192, "available_mb": 8192},
                "gpu": {"devices": 0, "available_devices": 0, "total_memory_mb": 0, "available_memory_mb": 0}
            }
        
        # Create mock worker
        mock_worker = MockWorker(worker_id, resources)
        
        # Store worker
        self.workers[worker_id] = {
            "worker": mock_worker,
            "status": "running",
            "worker_type": worker_type,
            "creation_time": time.time()
        }
        
        logger.info(f"Created worker {worker_id} of type {worker_type}")
        
        return {
            "worker_id": worker_id,
            "status": "running",
            "provider": self.name,
            "worker_type": worker_type
        }
    
    def terminate_worker(self, worker_id):
        """
        Terminate a worker instance.
        
        Args:
            worker_id: Worker ID
        
        Returns:
            bool: Success status
        """
        if worker_id in self.workers:
            self.workers[worker_id]["status"] = "terminated"
            self.workers[worker_id]["worker"].running = False
            logger.info(f"Terminated worker {worker_id}")
            return True
        
        logger.warning(f"Worker {worker_id} not found")
        return False
    
    def get_worker_status(self, worker_id):
        """
        Get worker status.
        
        Args:
            worker_id: Worker ID
        
        Returns:
            dict: Worker status
        """
        if worker_id in self.workers:
            return {
                "worker_id": worker_id,
                "status": self.workers[worker_id]["status"],
                "provider": self.name,
                "worker_type": self.workers[worker_id]["worker_type"]
            }
        
        return None
    
    def get_available_resources(self):
        """
        Get available resources.
        
        Returns:
            dict: Available resources
        """
        return {
            "provider": self.name,
            "max_workers": 10,
            "active_workers": len([w for w in self.workers.values() if w["status"] == "running"])
        }


class MockCoordinator:
    """Mock coordinator for DRM demo."""
    
    def __init__(self):
        """Initialize mock coordinator."""
        self.running = True
        self.worker_api_key = "test_api_key"
        self.host = "localhost"
        self.port = 8080
        
        # Initialize DRM components
        self.dynamic_resource_manager = DynamicResourceManager()
        self.resource_performance_predictor = ResourcePerformancePredictor()
        self.cloud_provider_manager = CloudProviderManager()
        
        # Add mock cloud provider
        self.mock_cloud_provider = MockCloudProvider("test-provider")
        self.cloud_provider_manager.add_provider("test-provider", self.mock_cloud_provider)
        
        # Initialize task queue and active tasks
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = []
        
        # Initialize workers
        self.workers = {}
    
    def add_task(self, task):
        """
        Add task to queue.
        
        Args:
            task: Task to add
        """
        self.task_queue.append(task)
        logger.info(f"Added task {task['task_id']} to queue")
    
    def get_next_task(self, worker_id):
        """
        Get next task for worker.
        
        Args:
            worker_id: Worker ID
        
        Returns:
            dict: Next task or None
        """
        if not self.task_queue:
            return None
        
        # Get worker resources
        worker_resources = None
        if worker_id in self.dynamic_resource_manager.worker_resources:
            worker_resources = self.dynamic_resource_manager.worker_resources[worker_id]
        
        # Find suitable task
        for i, task in enumerate(self.task_queue):
            # Check if worker has enough resources
            if worker_resources:
                resources_check = self.dynamic_resource_manager.check_resource_availability(
                    worker_id=worker_id,
                    resource_requirements=task.get("requirements", {})
                )
                
                if not resources_check["available"]:
                    continue
                
                # Reserve resources
                reservation_id = self.dynamic_resource_manager.reserve_resources(
                    worker_id=worker_id,
                    task_id=task["task_id"],
                    resource_requirements=task.get("requirements", {})
                )
                
                if reservation_id:
                    task["resource_reservation_id"] = reservation_id
            
            # Remove task from queue
            self.task_queue.pop(i)
            
            # Add to active tasks
            self.active_tasks[task["task_id"]] = {
                "task": task,
                "worker_id": worker_id,
                "start_time": time.time()
            }
            
            logger.info(f"Assigned task {task['task_id']} to worker {worker_id}")
            return task
        
        return None
    
    def complete_task(self, task_id, result):
        """
        Complete task.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]["task"]
            worker_id = self.active_tasks[task_id]["worker_id"]
            
            # Release resources
            if "resource_reservation_id" in task:
                self.dynamic_resource_manager.release_resources(task["resource_reservation_id"])
            
            # Record resource usage
            if hasattr(self, "resource_performance_predictor"):
                execution_metrics = result.get("execution_metrics", {})
                
                resource_usage = {
                    "model_type": task.get("config", {}).get("model_type", "unknown"),
                    "model_name": task.get("config", {}).get("model", "unknown"),
                    "batch_size": task.get("config", {}).get("batch_size", 1),
                    "success": True,
                    "execution_time_ms": execution_metrics.get("execution_time_seconds", 0) * 1000,
                    "cpu_cores_used": execution_metrics.get("peak_cpu_cores", 0),
                    "memory_mb_used": execution_metrics.get("peak_memory_mb", 0),
                    "gpu_memory_mb_used": execution_metrics.get("peak_gpu_memory_mb", 0)
                }
                
                self.resource_performance_predictor.record_task_execution(task_id, resource_usage)
            
            # Move to completed tasks
            self.completed_tasks.append({
                "task": task,
                "worker_id": worker_id,
                "result": result,
                "completion_time": time.time()
            })
            
            # Remove from active tasks
            del self.active_tasks[task_id]
            
            logger.info(f"Completed task {task_id}")
    
    def register_worker(self, worker_id, resources):
        """
        Register worker.
        
        Args:
            worker_id: Worker ID
            resources: Worker resources
        """
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "resources": resources,
            "registration_time": time.time(),
            "last_heartbeat": time.time()
        }
        
        # Register with DRM
        self.dynamic_resource_manager.register_worker(worker_id, resources)
        
        logger.info(f"Registered worker {worker_id}")
    
    def update_worker_resources(self, worker_id, resources):
        """
        Update worker resources.
        
        Args:
            worker_id: Worker ID
            resources: Worker resources
        """
        if worker_id in self.workers:
            self.workers[worker_id]["resources"] = resources
            self.workers[worker_id]["last_heartbeat"] = time.time()
            
            # Update DRM
            self.dynamic_resource_manager.update_worker_resources(worker_id, resources)
            
            logger.debug(f"Updated resources for worker {worker_id}")
    
    def evaluate_scaling(self):
        """
        Evaluate scaling and take action.
        
        Returns:
            ScalingDecision: Scaling decision
        """
        # Get scaling decision
        scaling_decision = self.dynamic_resource_manager.evaluate_scaling()
        
        # Take action based on decision
        if scaling_decision.action == "scale_up":
            logger.info(
                f"Scaling up: {scaling_decision.count} {scaling_decision.worker_type} "
                f"workers due to {scaling_decision.reason}"
            )
            
            # Scale up by creating new workers
            for i in range(scaling_decision.count):
                worker_result = self.cloud_provider_manager.create_worker(
                    provider="test-provider",
                    resources=scaling_decision.resource_requirements,
                    worker_type=scaling_decision.worker_type,
                    coordinator_url=f"ws://{self.host}:{self.port}",
                    api_key=self.worker_api_key
                )
                
                if worker_result:
                    # Get worker object from provider
                    worker_id = worker_result["worker_id"]
                    worker_obj = self.mock_cloud_provider.workers[worker_id]["worker"]
                    
                    # Register worker
                    self.register_worker(worker_id, worker_obj.get_resources())
                    
                    # Start worker thread
                    threading.Thread(target=self._run_worker, args=(worker_id, worker_obj)).start()
        
        elif scaling_decision.action == "scale_down":
            logger.info(
                f"Scaling down: removing {len(scaling_decision.worker_ids)} "
                f"workers due to {scaling_decision.reason}"
            )
            
            # Scale down by terminating workers
            for worker_id in scaling_decision.worker_ids:
                # Terminate worker
                self.cloud_provider_manager.terminate_worker("test-provider", worker_id)
                
                # Deregister worker
                if worker_id in self.workers:
                    self.workers.pop(worker_id)
                
                # Deregister from DRM
                self.dynamic_resource_manager.deregister_worker(worker_id)
                
                logger.info(f"Terminated worker {worker_id}")
        
        return scaling_decision
    
    def _run_worker(self, worker_id, worker):
        """
        Run worker thread.
        
        Args:
            worker_id: Worker ID
            worker: Worker object
        """
        logger.info(f"Started worker thread for {worker_id}")
        
        while self.running and worker.running:
            try:
                # Get next task
                task = self.get_next_task(worker_id)
                
                if task:
                    # Execute task
                    result = worker.execute_task(task)
                    
                    # Complete task
                    self.complete_task(task["task_id"], result)
                else:
                    # No task, sleep for a bit
                    time.sleep(1)
                
                # Update resources (simulate heartbeat)
                self.update_worker_resources(worker_id, worker.get_resources())
            
            except Exception as e:
                logger.error(f"Error in worker thread {worker_id}: {e}")
        
        logger.info(f"Worker thread for {worker_id} terminated")


def create_sample_tasks(num_tasks=20):
    """
    Create sample tasks for testing.
    
    Args:
        num_tasks: Number of tasks to create
    
    Returns:
        list: Sample tasks
    """
    tasks = []
    
    task_types = [
        {
            "type": "text-embedding",
            "model": "bert-base-uncased",
            "batch_size": 32,
            "requirements": {
                "cpu_cores": 2,
                "memory_mb": 4096,
                "execution_time_min": 3,
                "execution_time_max": 8
            }
        },
        {
            "type": "text-generation",
            "model": "llama-7b",
            "batch_size": 4,
            "requirements": {
                "cpu_cores": 4,
                "memory_mb": 16384,
                "gpu_memory_mb": 8192,
                "execution_time_min": 10,
                "execution_time_max": 20
            }
        },
        {
            "type": "vision",
            "model": "vit-base-patch16-224",
            "batch_size": 16,
            "requirements": {
                "cpu_cores": 2,
                "memory_mb": 4096,
                "gpu_memory_mb": 2048,
                "execution_time_min": 5,
                "execution_time_max": 12
            }
        },
        {
            "type": "multimodal",
            "model": "clip-vit-base-patch32",
            "batch_size": 8,
            "requirements": {
                "cpu_cores": 4,
                "memory_mb": 8192,
                "gpu_memory_mb": 4096,
                "execution_time_min": 8,
                "execution_time_max": 15
            }
        }
    ]
    
    for i in range(num_tasks):
        # Select random task type
        task_type = random.choice(task_types)
        
        # Create task
        task = {
            "task_id": f"task-{i}",
            "type": "benchmark",
            "config": {
                "model_type": task_type["type"],
                "model": task_type["model"],
                "batch_size": task_type["batch_size"]
            },
            "requirements": task_type["requirements"].copy()
        }
        
        tasks.append(task)
    
    return tasks


def run_demo(args):
    """
    Run the DRM demo.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting DRM demo")
    
    # Create mock coordinator
    coordinator = MockCoordinator()
    
    # Create initial worker
    worker_result = coordinator.cloud_provider_manager.create_worker(
        provider="test-provider",
        worker_type="default"
    )
    
    if worker_result:
        # Get worker object from provider
        worker_id = worker_result["worker_id"]
        worker_obj = coordinator.mock_cloud_provider.workers[worker_id]["worker"]
        
        # Register worker
        coordinator.register_worker(worker_id, worker_obj.get_resources())
        
        # Start worker thread
        threading.Thread(target=coordinator._run_worker, args=(worker_id, worker_obj)).start()
    
    # Create sample tasks
    tasks = create_sample_tasks(args.num_tasks)
    
    # Start scheduling thread
    def scheduling_thread():
        """Thread for scheduling tasks over time."""
        remaining_tasks = tasks.copy()
        
        while coordinator.running and remaining_tasks:
            # Add some tasks
            num_to_add = min(3, len(remaining_tasks))
            for _ in range(num_to_add):
                if remaining_tasks:
                    task = remaining_tasks.pop(0)
                    coordinator.add_task(task)
            
            # Sleep for a bit
            time.sleep(random.uniform(2, 5))
    
    # Start scaling evaluation thread
    def scaling_thread():
        """Thread for periodic scaling evaluation."""
        while coordinator.running:
            # Evaluate scaling
            coordinator.evaluate_scaling()
            
            # Sleep for evaluation interval
            time.sleep(10)
    
    # Start monitoring thread
    def monitoring_thread():
        """Thread for monitoring system state."""
        while coordinator.running:
            # Get system state
            num_workers = len(coordinator.workers)
            num_active_tasks = len(coordinator.active_tasks)
            num_queued_tasks = len(coordinator.task_queue)
            num_completed_tasks = len(coordinator.completed_tasks)
            
            # Calculate utilization
            utilization = coordinator.dynamic_resource_manager.get_worker_utilization()
            overall_util = utilization["overall"]["overall"]
            
            logger.info(
                f"State: {num_workers} workers, {num_active_tasks} active tasks, "
                f"{num_queued_tasks} queued tasks, {num_completed_tasks} completed, "
                f"utilization: {overall_util:.1%}"
            )
            
            # Check if demo is complete
            if num_completed_tasks >= args.num_tasks and num_active_tasks == 0 and num_queued_tasks == 0:
                logger.info("All tasks completed, stopping demo")
                coordinator.running = False
                break
            
            # Sleep for monitoring interval
            time.sleep(5)
    
    # Start threads
    threads = [
        threading.Thread(target=scheduling_thread),
        threading.Thread(target=scaling_thread),
        threading.Thread(target=monitoring_thread)
    ]
    
    for thread in threads:
        thread.daemon = True
        thread.start()
    
    try:
        # Wait for completion or timeout
        start_time = time.time()
        while coordinator.running:
            if time.time() - start_time > args.timeout:
                logger.info(f"Demo timeout after {args.timeout} seconds")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted")
    
    finally:
        # Cleanup
        coordinator.running = False
        
        # Wait for threads to terminate
        for thread in threads:
            thread.join(timeout=2)
        
        # Print summary
        logger.info(f"Demo completed with {len(coordinator.completed_tasks)} tasks")
        
        # Print scaling events
        if args.verbose:
            logger.info("Resource Manager Statistics:")
            worker_stats = coordinator.dynamic_resource_manager.get_worker_statistics()
            logger.info(f"Final system state: {json.dumps(worker_stats, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Resource Management Demo")
    parser.add_argument("--num-tasks", type=int, default=20, help="Number of tasks to create")
    parser.add_argument("--timeout", type=int, default=300, help="Demo timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run demo
    run_demo(args)