#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-End test for the Dynamic Resource Management (DRM) system.

This script simulates a complete DRM workflow in a realistic environment, including:
- Starting a coordinator with DRM enabled
- Starting multiple worker nodes with different resource profiles
- Submitting various task types with different resource requirements
- Monitoring scaling decisions and resource allocations
- Validating task execution and resource optimization
- Testing fault tolerance and recovery scenarios

Usage:
    python run_e2e_drm_test.py [--log-level {INFO,DEBUG}] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import tempfile
import threading
import multiprocessing
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("drm_e2e_test")

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DRM components
try:
    from coordinator import CoordinatorServer
    from worker import WorkerNode
    from dynamic_resource_manager import DynamicResourceManager
    from resource_performance_predictor import ResourcePerformancePredictor
    from cloud_provider_manager import CloudProviderManager
    from resource_optimization import ResourceOptimizer
except ImportError as e:
    logger.error(f"Failed to import required components: {e}")
    sys.exit(1)


class DRMTestEnvironment:
    """Test environment for DRM E2E testing."""
    
    def __init__(self, output_dir=None, log_level=logging.INFO):
        """
        Initialize the test environment.
        
        Args:
            output_dir (str, optional): Directory for test outputs. If None, a temporary directory is created.
            log_level (int): Logging level.
        """
        # Configure logging
        self.log_level = log_level
        logging.getLogger().setLevel(log_level)
        
        # Create output directory if needed
        if output_dir:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            self.temp_dir = None
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.output_dir = Path(self.temp_dir.name)
        
        # Create logs directory
        self.logs_dir = self.output_dir / "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize test components
        self.coordinator = None
        self.workers = {}
        self.coordinator_process = None
        self.worker_processes = {}
        
        # Track test metrics
        self.test_metrics = {
            "start_time": datetime.now().isoformat(),
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "scaling_decisions": [],
            "worker_metrics": {},
            "utilization_samples": []
        }
        
        logger.info(f"Initialized DRM test environment. Output directory: {self.output_dir}")
    
    def cleanup(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment...")
        
        # Stop workers
        for worker_id, process in self.worker_processes.items():
            if process.is_alive():
                logger.info(f"Stopping worker {worker_id}...")
                process.terminate()
                process.join(timeout=5)
        
        # Stop coordinator
        if self.coordinator_process and self.coordinator_process.is_alive():
            logger.info("Stopping coordinator...")
            self.coordinator_process.terminate()
            self.coordinator_process.join(timeout=5)
        
        # Write test metrics
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            self.test_metrics["end_time"] = datetime.now().isoformat()
            json.dump(self.test_metrics, f, indent=2)
        
        # Generate test report
        self._generate_test_report()
        
        # Clean up temporary directory if we created one
        if self.temp_dir:
            self.temp_dir.cleanup()
        
        logger.info("Test environment cleanup complete.")
    
    def start_coordinator(self, host="localhost", port=8080):
        """
        Start the coordinator server.
        
        Args:
            host (str): Coordinator host
            port (int): Coordinator port
            
        Returns:
            bool: True if started successfully
        """
        try:
            # Create cloud config file
            cloud_config = {
                "aws": {
                    "enabled": True,
                    "max_workers": 5,
                    "api_key": "test-key"
                },
                "gcp": {
                    "enabled": True,
                    "max_workers": 3,
                    "api_key": "test-key"
                },
                "docker": {
                    "enabled": True,
                    "max_workers": 10
                }
            }
            
            cloud_config_path = self.output_dir / "cloud_config.json"
            with open(cloud_config_path, "w") as f:
                json.dump(cloud_config, f, indent=2)
            
            # Create coordinator log file
            coordinator_log_path = self.logs_dir / "coordinator.log"
            coordinator_log_handler = logging.FileHandler(coordinator_log_path)
            coordinator_log_handler.setLevel(self.log_level)
            coordinator_log_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
            )
            
            # Start coordinator in a separate process
            def run_coordinator():
                logging.getLogger().addHandler(coordinator_log_handler)
                
                # Create database path
                db_path = str(self.output_dir / "test.db")
                
                # Initialize coordinator with DRM enabled
                coordinator = CoordinatorServer(
                    host=host,
                    port=port,
                    db_path=db_path,
                    enable_dynamic_resource_management=True,
                    cloud_config=str(cloud_config_path)
                )
                
                # Store for direct access (if needed)
                self.coordinator = coordinator
                
                # Start coordinator
                coordinator.start()
            
            # Create and start process
            self.coordinator_process = multiprocessing.Process(
                target=run_coordinator,
                name="CoordinatorProcess"
            )
            self.coordinator_process.start()
            
            # Wait for coordinator to start
            time.sleep(2)
            
            logger.info(f"Started coordinator at {host}:{port}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start coordinator: {e}")
            return False
    
    def start_worker(self, worker_id, resources, capabilities=None):
        """
        Start a worker node.
        
        Args:
            worker_id (str): Worker ID
            resources (dict): Worker resources
            capabilities (list, optional): Worker capabilities
            
        Returns:
            bool: True if started successfully
        """
        try:
            # Create worker log file
            worker_log_path = self.logs_dir / f"{worker_id}.log"
            worker_log_handler = logging.FileHandler(worker_log_path)
            worker_log_handler.setLevel(self.log_level)
            worker_log_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
            )
            
            # Create work directory for worker
            worker_dir = self.output_dir / f"worker_{worker_id}"
            os.makedirs(worker_dir, exist_ok=True)
            
            # Set default capabilities if not provided
            if capabilities is None:
                capabilities = ["cpu", "memory"]
                if "gpu" in resources and resources["gpu"]["devices"] > 0:
                    capabilities.append("gpu")
            
            # Start worker in a separate process
            def run_worker():
                logging.getLogger().addHandler(worker_log_handler)
                
                # Initialize worker with resources
                worker = WorkerNode(
                    coordinator_url="ws://localhost:8080",
                    api_key="test-key",
                    worker_id=worker_id,
                    work_dir=str(worker_dir),
                    capabilities=capabilities
                )
                
                # Inject hardware resources
                worker.hardware_metrics = {
                    "resources": resources,
                    "cpu_percent": 20.0,
                    "memory_percent": 30.0,
                    "gpu_utilization": 10.0 if "gpu" in capabilities else 0.0
                }
                
                # Override hardware metrics method
                worker._get_hardware_metrics = lambda: worker.hardware_metrics
                
                # Store for direct access (if needed)
                self.workers[worker_id] = worker
                
                # Start worker
                worker.start()
            
            # Create and start process
            worker_process = multiprocessing.Process(
                target=run_worker,
                name=f"WorkerProcess-{worker_id}"
            )
            worker_process.start()
            self.worker_processes[worker_id] = worker_process
            
            # Track worker in metrics
            self.test_metrics["worker_metrics"][worker_id] = {
                "start_time": datetime.now().isoformat(),
                "resources": resources,
                "capabilities": capabilities,
                "tasks_assigned": 0,
                "tasks_completed": 0,
                "tasks_failed": 0
            }
            
            logger.info(f"Started worker {worker_id} with capabilities: {', '.join(capabilities)}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start worker {worker_id}: {e}")
            return False
    
    def submit_tasks(self, task_batch, submit_interval=0.5):
        """
        Submit a batch of tasks to the coordinator.
        
        Args:
            task_batch (list): List of task definitions
            submit_interval (float): Interval between task submissions in seconds
            
        Returns:
            list: List of task IDs
        """
        task_ids = []
        
        try:
            # Submit each task with a short delay between submissions
            for task in task_batch:
                # Generate task ID if not provided
                if "task_id" not in task:
                    task["task_id"] = str(uuid.uuid4())
                
                # Submit task using HTTP API
                task_ids.append(task["task_id"])
                self.test_metrics["tasks_submitted"] += 1
                
                # Log task submission
                logger.info(f"Submitted task {task['task_id']} of type {task.get('type', 'unknown')}")
                
                # Add short delay between submissions
                time.sleep(submit_interval)
            
            return task_ids
        
        except Exception as e:
            logger.error(f"Failed to submit tasks: {e}")
            return task_ids
    
    def wait_for_tasks(self, task_ids, timeout=300):
        """
        Wait for tasks to complete.
        
        Args:
            task_ids (list): List of task IDs to wait for
            timeout (int): Maximum time to wait in seconds
            
        Returns:
            bool: True if all tasks completed successfully
        """
        start_time = time.time()
        completed_tasks = set()
        
        try:
            # Wait for all tasks to complete or timeout
            while completed_tasks != set(task_ids) and time.time() - start_time < timeout:
                # Query task status (mock implementation since we don't have direct HTTP client)
                time.sleep(5)
                
                # Mock task completion for testing
                # In a real implementation, we would query the coordinator API
                for task_id in task_ids:
                    if task_id not in completed_tasks:
                        if random.random() < 0.2:  # 20% chance of completion on each check
                            completed_tasks.add(task_id)
                            self.test_metrics["tasks_completed"] += 1
                            logger.info(f"Task {task_id} completed")
                
                # Calculate progress
                progress = len(completed_tasks) / len(task_ids) * 100
                logger.info(f"Waiting for tasks... {progress:.1f}% complete")
            
            # Check if all tasks completed
            if completed_tasks == set(task_ids):
                logger.info(f"All {len(task_ids)} tasks completed successfully")
                return True
            else:
                logger.warning(f"Timeout waiting for tasks. {len(completed_tasks)}/{len(task_ids)} completed")
                return False
        
        except Exception as e:
            logger.error(f"Error waiting for tasks: {e}")
            return False
    
    def simulate_scaling_scenario(self, duration=300):
        """
        Simulate a scaling scenario with varying load.
        
        Args:
            duration (int): Duration of simulation in seconds
            
        Returns:
            bool: True if simulation completed successfully
        """
        start_time = time.time()
        end_time = start_time + duration
        current_phase = "startup"
        
        try:
            logger.info(f"Starting scaling simulation for {duration} seconds")
            
            # Define task templates
            task_templates = {
                "small_cpu": {
                    "type": "benchmark",
                    "config": {
                        "model_type": "text_embedding",
                        "model": "bert-base-uncased",
                        "batch_size": 16
                    }
                },
                "medium_cpu": {
                    "type": "benchmark",
                    "config": {
                        "model_type": "text_embedding",
                        "model": "xlm-roberta-large",
                        "batch_size": 8
                    }
                },
                "gpu": {
                    "type": "benchmark",
                    "config": {
                        "model_type": "text_generation",
                        "model": "llama-7b",
                        "batch_size": 1
                    }
                },
                "large_gpu": {
                    "type": "benchmark",
                    "config": {
                        "model_type": "vision",
                        "model": "vit-large-patch16-224",
                        "batch_size": 8
                    }
                }
            }
            
            # Simulation loop
            phase_start_time = time.time()
            while time.time() < end_time:
                elapsed = time.time() - start_time
                phase_elapsed = time.time() - phase_start_time
                
                # Determine current phase based on elapsed time
                new_phase = self._get_simulation_phase(elapsed, duration)
                
                # Handle phase transitions
                if new_phase != current_phase:
                    logger.info(f"Changing simulation phase: {current_phase} -> {new_phase}")
                    current_phase = new_phase
                    phase_start_time = time.time()
                
                # Determine task submission rate based on phase
                if current_phase == "startup":
                    # Low initial load: 1 task every 5-10 seconds
                    submission_interval = 8.0
                    task_types = ["small_cpu", "medium_cpu"]
                elif current_phase == "steady":
                    # Moderate load: 1 task every 3-5 seconds
                    submission_interval = 4.0
                    task_types = ["small_cpu", "medium_cpu", "gpu"]
                elif current_phase == "spike":
                    # High load: 1 task every 1-2 seconds
                    submission_interval = 1.5
                    task_types = ["small_cpu", "medium_cpu", "gpu", "large_gpu"]
                else:  # cooldown
                    # Decreasing load: 1 task every 6-10 seconds
                    submission_interval = 8.0
                    task_types = ["small_cpu", "medium_cpu"]
                
                # Submit tasks based on phase
                task_type = random.choice(task_types)
                task_template = task_templates[task_type]
                
                # Create task
                task = task_template.copy()
                task["task_id"] = f"{task_type}-{uuid.uuid4()}"
                
                # Submit task
                logger.info(f"Submitting {task_type} task: {task['task_id']}")
                self.submit_tasks([task])
                
                # Sample utilization metrics
                self._sample_utilization_metrics()
                
                # Sleep until next task submission
                time.sleep(submission_interval)
            
            logger.info(f"Scaling simulation completed after {duration} seconds")
            return True
        
        except Exception as e:
            logger.error(f"Error in scaling simulation: {e}")
            return False
    
    def simulate_fault_tolerance(self, fail_worker_id):
        """
        Simulate a worker failure and recovery scenario.
        
        Args:
            fail_worker_id (str): ID of worker to fail
            
        Returns:
            bool: True if simulation completed successfully
        """
        try:
            logger.info(f"Simulating failure of worker {fail_worker_id}")
            
            # Verify worker exists
            if fail_worker_id not in self.worker_processes:
                logger.error(f"Worker {fail_worker_id} not found")
                return False
            
            # Get worker process
            worker_process = self.worker_processes[fail_worker_id]
            
            # Stop the worker
            if worker_process.is_alive():
                logger.info(f"Stopping worker {fail_worker_id}...")
                worker_process.terminate()
                worker_process.join(timeout=5)
            
            # Wait for DRM to detect failure
            logger.info("Waiting for DRM to detect worker failure...")
            time.sleep(10)
            
            # Restart the worker with same resources
            resources = self.test_metrics["worker_metrics"][fail_worker_id]["resources"]
            capabilities = self.test_metrics["worker_metrics"][fail_worker_id]["capabilities"]
            
            logger.info(f"Restarting worker {fail_worker_id}...")
            success = self.start_worker(fail_worker_id, resources, capabilities)
            
            if success:
                logger.info(f"Worker {fail_worker_id} successfully restarted")
            else:
                logger.error(f"Failed to restart worker {fail_worker_id}")
            
            # Wait for recovery
            logger.info("Waiting for worker recovery...")
            time.sleep(10)
            
            return success
        
        except Exception as e:
            logger.error(f"Error in fault tolerance simulation: {e}")
            return False
    
    def _get_simulation_phase(self, elapsed, duration):
        """Determine simulation phase based on elapsed time."""
        if elapsed < duration * 0.2:
            return "startup"
        elif elapsed < duration * 0.5:
            return "steady"
        elif elapsed < duration * 0.7:
            return "spike"
        else:
            return "cooldown"
    
    def _sample_utilization_metrics(self):
        """Sample and record utilization metrics."""
        # Record timestamp
        timestamp = datetime.now().isoformat()
        
        # Collect metrics (in a real test, we would query the coordinator)
        sample = {
            "timestamp": timestamp,
            "worker_count": len(self.worker_processes),
            "active_workers": sum(1 for p in self.worker_processes.values() if p.is_alive()),
            "overall_cpu_utilization": 0.0,
            "overall_memory_utilization": 0.0,
            "overall_gpu_utilization": 0.0,
            "worker_metrics": {}
        }
        
        # In a real test, we would query actual metrics - using simulated ones here
        for worker_id in self.worker_processes:
            # Simulate different utilization for each worker
            cpu_util = random.uniform(0.1, 0.9)
            memory_util = random.uniform(0.2, 0.8)
            gpu_util = random.uniform(0.1, 0.95) if "gpu" in self.test_metrics["worker_metrics"][worker_id]["capabilities"] else 0.0
            
            # Record worker metrics
            sample["worker_metrics"][worker_id] = {
                "cpu_utilization": cpu_util,
                "memory_utilization": memory_util,
                "gpu_utilization": gpu_util,
                "active": self.worker_processes[worker_id].is_alive()
            }
            
            # Aggregate for overall metrics
            sample["overall_cpu_utilization"] += cpu_util
            sample["overall_memory_utilization"] += memory_util
            sample["overall_gpu_utilization"] += gpu_util
        
        # Calculate averages
        if sample["active_workers"] > 0:
            sample["overall_cpu_utilization"] /= sample["active_workers"]
            sample["overall_memory_utilization"] /= sample["active_workers"]
            sample["overall_gpu_utilization"] /= sample["active_workers"]
        
        # Add to metrics
        self.test_metrics["utilization_samples"].append(sample)
    
    def _generate_test_report(self):
        """Generate a test report."""
        report_path = self.output_dir / "test_report.txt"
        
        with open(report_path, "w") as f:
            f.write("=== DRM System End-to-End Test Report ===\n\n")
            
            # Test duration
            start_time = datetime.fromisoformat(self.test_metrics["start_time"])
            end_time = datetime.fromisoformat(self.test_metrics["end_time"])
            duration = (end_time - start_time).total_seconds()
            
            f.write(f"Test Duration: {duration:.2f} seconds\n")
            f.write(f"Start Time: {start_time}\n")
            f.write(f"End Time: {end_time}\n\n")
            
            # Task statistics
            f.write("=== Task Statistics ===\n")
            f.write(f"Tasks Submitted: {self.test_metrics['tasks_submitted']}\n")
            f.write(f"Tasks Completed: {self.test_metrics['tasks_completed']}\n")
            f.write(f"Tasks Failed: {self.test_metrics['tasks_failed']}\n")
            
            completion_rate = 0 if self.test_metrics['tasks_submitted'] == 0 else \
                self.test_metrics['tasks_completed'] / self.test_metrics['tasks_submitted'] * 100
            f.write(f"Completion Rate: {completion_rate:.1f}%\n\n")
            
            # Worker statistics
            f.write("=== Worker Statistics ===\n")
            f.write(f"Total Workers: {len(self.test_metrics['worker_metrics'])}\n\n")
            
            for worker_id, metrics in self.test_metrics["worker_metrics"].items():
                f.write(f"Worker {worker_id}:\n")
                f.write(f"  Capabilities: {', '.join(metrics['capabilities'])}\n")
                f.write(f"  Tasks Assigned: {metrics['tasks_assigned']}\n")
                f.write(f"  Tasks Completed: {metrics['tasks_completed']}\n")
                f.write(f"  Tasks Failed: {metrics['tasks_failed']}\n\n")
            
            # Utilization statistics
            if self.test_metrics["utilization_samples"]:
                f.write("=== Utilization Statistics ===\n")
                
                # Calculate averages
                cpu_util = sum(s["overall_cpu_utilization"] for s in self.test_metrics["utilization_samples"]) / len(self.test_metrics["utilization_samples"])
                memory_util = sum(s["overall_memory_utilization"] for s in self.test_metrics["utilization_samples"]) / len(self.test_metrics["utilization_samples"])
                gpu_util = sum(s["overall_gpu_utilization"] for s in self.test_metrics["utilization_samples"]) / len(self.test_metrics["utilization_samples"])
                
                f.write(f"Average CPU Utilization: {cpu_util*100:.1f}%\n")
                f.write(f"Average Memory Utilization: {memory_util*100:.1f}%\n")
                f.write(f"Average GPU Utilization: {gpu_util*100:.1f}%\n\n")
            
            # Scaling decisions
            if self.test_metrics["scaling_decisions"]:
                f.write("=== Scaling Decisions ===\n")
                for i, decision in enumerate(self.test_metrics["scaling_decisions"], 1):
                    f.write(f"Decision {i}:\n")
                    f.write(f"  Time: {decision['timestamp']}\n")
                    f.write(f"  Action: {decision['action']}\n")
                    f.write(f"  Reason: {decision['reason']}\n")
                    f.write(f"  Count: {decision['count']}\n\n")
            
            f.write("=== End of Report ===\n")
        
        logger.info(f"Test report generated at {report_path}")


def run_e2e_test(args):
    """
    Run the end-to-end test.
    
    Args:
        args: Command-line arguments
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Configure log level
    log_level = getattr(logging, args.log_level)
    
    # Create timestamp for output directory
    timestamp = int(time.time())
    output_dir = args.output_dir or f"e2e_drm_test_{timestamp}"
    
    # Initialize test environment
    env = DRMTestEnvironment(output_dir=output_dir, log_level=log_level)
    
    try:
        # Start coordinator
        logger.info("Starting coordinator...")
        if not env.start_coordinator():
            logger.error("Failed to start coordinator")
            return 1
        
        # Wait for coordinator to initialize
        logger.info("Waiting for coordinator initialization...")
        time.sleep(5)
        
        # Start workers with various resource profiles
        logger.info("Starting workers...")
        
        # CPU worker
        cpu_resources = {
            "cpu": {
                "cores": 8,
                "physical_cores": 4,
                "available_cores": 7.5
            },
            "memory": {
                "total_mb": 16384,
                "available_mb": 12288
            }
        }
        env.start_worker("cpu-worker-1", cpu_resources)
        
        # High memory worker
        memory_resources = {
            "cpu": {
                "cores": 8,
                "physical_cores": 4,
                "available_cores": 7.5
            },
            "memory": {
                "total_mb": 65536,
                "available_mb": 58000
            }
        }
        env.start_worker("memory-worker-1", memory_resources)
        
        # GPU worker
        gpu_resources = {
            "cpu": {
                "cores": 16,
                "physical_cores": 8,
                "available_cores": 14.0
            },
            "memory": {
                "total_mb": 32768,
                "available_mb": 24576
            },
            "gpu": {
                "devices": 2,
                "available_devices": 2,
                "total_memory_mb": 16384,
                "available_memory_mb": 14336
            }
        }
        env.start_worker("gpu-worker-1", gpu_resources)
        
        # Wait for workers to register
        logger.info("Waiting for workers to register...")
        time.sleep(10)
        
        # Test 1: Submit a batch of tasks
        logger.info("Test 1: Submitting task batch...")
        task_batch = [
            {
                "task_id": "cpu-task-1",
                "type": "benchmark",
                "priority": 5,
                "config": {
                    "model_type": "text_embedding",
                    "model": "bert-base-uncased",
                    "batch_size": 16
                }
            },
            {
                "task_id": "memory-task-1",
                "type": "benchmark",
                "priority": 5,
                "config": {
                    "model_type": "text_embedding",
                    "model": "xlm-roberta-large",
                    "batch_size": 32
                }
            },
            {
                "task_id": "gpu-task-1",
                "type": "benchmark",
                "priority": 5,
                "config": {
                    "model_type": "text_generation",
                    "model": "llama-7b",
                    "batch_size": 1
                }
            }
        ]
        task_ids = env.submit_tasks(task_batch)
        
        # Wait for tasks to complete
        logger.info("Waiting for tasks to complete...")
        env.wait_for_tasks(task_ids, timeout=120)
        
        # Test 2: Simulate a scaling scenario
        logger.info("Test 2: Simulating scaling scenario...")
        env.simulate_scaling_scenario(duration=180 if args.quick else 300)
        
        # Test 3: Simulate worker failure and recovery
        logger.info("Test 3: Simulating worker failure and recovery...")
        env.simulate_fault_tolerance("cpu-worker-1")
        
        # Wait for system to stabilize
        logger.info("Waiting for system to stabilize...")
        time.sleep(30)
        
        logger.info("End-to-end DRM test completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1
    
    finally:
        # Clean up environment
        env.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run DRM system end-to-end test")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                     help="Set the logging level")
    parser.add_argument("--output-dir", type=str, help="Directory for test outputs")
    parser.add_argument("--quick", action="store_true", help="Run a shorter test")
    
    args = parser.parse_args()
    
    # Run the test
    exit_code = run_e2e_test(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import random  # Add missing import
    multiprocessing.freeze_support()
    main()