#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating how to use the Multi-Device Orchestrator with Dynamic Resource Management.

This example shows how to:
1. Initialize the Dynamic Resource Manager and register workers
2. Set up the Multi-Device Orchestrator
3. Execute complex tasks across heterogeneous hardware
4. Monitor resource utilization and task status
5. Collect and process results from distributed task execution
"""

import os
import sys
import time
import json
import logging
import threading
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure correct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import orchestrator components
from multi_device_orchestrator import (
    MultiDeviceOrchestrator,
    TaskStatus,
    SubtaskStatus,
    SplitStrategy
)
from dynamic_resource_manager import DynamicResourceManager
from resource_performance_predictor import ResourcePerformancePredictor


def run_bert_benchmark_across_devices(task_manager, worker_manager, resource_manager):
    """
    Run a BERT benchmark task distributed across multiple devices.
    
    This example demonstrates running a BERT model benchmark with:
    1. Model parallel execution (embedding on CPU, encoder on GPU, decoder on GPU)
    2. Resource allocation based on component requirements
    3. Monitoring of resource utilization during execution
    4. Result merging from distributed execution
    
    Args:
        task_manager: Task manager instance
        worker_manager: Worker manager instance
        resource_manager: Dynamic resource manager instance
    
    Returns:
        dict: Merged benchmark results
    """
    # Create orchestrator
    orchestrator = MultiDeviceOrchestrator(
        task_manager=task_manager,
        worker_manager=worker_manager,
        resource_manager=resource_manager
    )
    
    # Define BERT benchmark task with model parallelism
    bert_task = {
        "task_id": "bert_benchmark_task",
        "type": "benchmark",
        "config": {
            "model": "bert-base-uncased",
            "batch_size": 8,
            "sequence_length": 128,
            "iterations": 100,
            "precision": "fp16",
            "metrics": ["latency", "throughput", "memory_usage"]
        },
        "model_components": [
            "embedding",   # CPU component: token embeddings + position embeddings
            "encoder",     # GPU component: transformer encoder blocks
            "pooler"       # CPU component: final pooling layer
        ],
        "component_resources": {
            "embedding": {
                "cpu_cores": 2,
                "memory_mb": 4096,
                "gpu_memory_mb": 0  # No GPU required
            },
            "encoder": {
                "cpu_cores": 2,
                "memory_mb": 8192,
                "gpu_memory_mb": 4096  # Requires GPU
            },
            "pooler": {
                "cpu_cores": 1,
                "memory_mb": 2048,
                "gpu_memory_mb": 0  # No GPU required
            }
        }
    }
    
    # Orchestrate BERT task with model parallel strategy
    logger.info("Orchestrating BERT benchmark with model parallelism...")
    task_id = orchestrator.orchestrate_task(
        task_data=bert_task,
        strategy=SplitStrategy.MODEL_PARALLEL
    )
    
    # Monitor resource utilization during execution
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(resource_manager, task_id, orchestrator),
        daemon=True
    )
    monitor_thread.start()
    
    # Wait for task completion
    logger.info(f"Waiting for task {task_id} to complete...")
    start_time = time.time()
    max_wait = 60.0  # Maximum wait time in seconds
    
    task_completed = False
    while time.time() - start_time < max_wait and not task_completed:
        status = orchestrator.get_task_status(task_id)
        logger.info(f"Task status: {status['status']}, Completion: {status.get('completion_percentage', 0)}%")
        
        if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task_completed = True
            break
        
        time.sleep(2.0)
    
    # Stop monitoring
    monitor_thread.join(timeout=1.0)
    
    # Check final status
    status = orchestrator.get_task_status(task_id)
    if status["status"] == TaskStatus.COMPLETED:
        logger.info(f"Task {task_id} completed successfully!")
        result = orchestrator.get_task_result(task_id)
        
        # Print merged results
        logger.info("BERT Benchmark Results:")
        logger.info(f"Number of components: {result.get('num_components', 0)}")
        
        # Extract metrics from components
        for component, component_result in result.get("component_results", {}).items():
            logger.info(f"Component {component}:")
            if "metrics" in component_result:
                for metric, value in component_result["metrics"].items():
                    logger.info(f"  {metric}: {value}")
        
        return result
    else:
        logger.error(f"Task {task_id} failed with status: {status['status']}")
        if "error" in status:
            logger.error(f"Error: {status['error']}")
        return None


def run_dataset_processing_across_devices(task_manager, worker_manager, resource_manager, num_items=100):
    """
    Run a dataset processing task distributed across multiple devices.
    
    This example demonstrates:
    1. Data parallel execution (split input data across workers)
    2. Resource allocation based on performance requirements
    3. Resource-aware task scheduling
    4. Result aggregation from parallel execution
    
    Args:
        task_manager: Task manager instance
        worker_manager: Worker manager instance
        resource_manager: Dynamic resource manager instance
        num_items: Number of dataset items to process
    
    Returns:
        dict: Merged processing results
    """
    # Create orchestrator
    orchestrator = MultiDeviceOrchestrator(
        task_manager=task_manager,
        worker_manager=worker_manager,
        resource_manager=resource_manager
    )
    
    # Define dataset processing task with data parallelism
    dataset_task = {
        "task_id": "dataset_processing_task",
        "type": "data_processing",
        "config": {
            "dataset_name": "example_dataset",
            "processing_steps": ["tokenize", "normalize", "embed", "store"],
            "execution_time_ms": 1000  # Simulate processing time
        },
        "input_data": [f"data_item_{i}" for i in range(num_items)],
        "resource_requirements": {
            "cpu_cores": 2,
            "memory_mb": 4096,
            "gpu_memory_mb": 0  # No GPU required for basic processing
        },
        "num_partitions": 4  # Split into 4 partitions for parallel processing
    }
    
    # Orchestrate dataset task with data parallel strategy
    logger.info("Orchestrating dataset processing with data parallelism...")
    task_id = orchestrator.orchestrate_task(
        task_data=dataset_task,
        strategy=SplitStrategy.DATA_PARALLEL
    )
    
    # Monitor resource utilization during execution
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(resource_manager, task_id, orchestrator),
        daemon=True
    )
    monitor_thread.start()
    
    # Wait for task completion
    logger.info(f"Waiting for task {task_id} to complete...")
    start_time = time.time()
    max_wait = 60.0  # Maximum wait time in seconds
    
    task_completed = False
    while time.time() - start_time < max_wait and not task_completed:
        status = orchestrator.get_task_status(task_id)
        logger.info(f"Task status: {status['status']}, Completion: {status.get('completion_percentage', 0)}%")
        
        if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task_completed = True
            break
        
        time.sleep(2.0)
    
    # Stop monitoring
    monitor_thread.join(timeout=1.0)
    
    # Check final status
    status = orchestrator.get_task_status(task_id)
    if status["status"] == TaskStatus.COMPLETED:
        logger.info(f"Task {task_id} completed successfully!")
        result = orchestrator.get_task_result(task_id)
        
        # Print merged results
        logger.info("Dataset Processing Results:")
        logger.info(f"Number of partitions: {result.get('num_partitions', 0)}")
        logger.info(f"Total processed items: {len(result.get('results', []))}")
        
        # Extract metrics
        if "metrics" in result:
            for metric, value in result["metrics"].items():
                logger.info(f"  {metric}: {value}")
        
        return result
    else:
        logger.error(f"Task {task_id} failed with status: {status['status']}")
        if "error" in status:
            logger.error(f"Error: {status['error']}")
        return None


def run_model_ensemble_across_devices(task_manager, worker_manager, resource_manager):
    """
    Run a model ensemble across multiple devices.
    
    This example demonstrates:
    1. Ensemble execution (run multiple model variants in parallel)
    2. Mixed resource allocation across heterogeneous hardware
    3. Result aggregation with voting/averaging
    
    Args:
        task_manager: Task manager instance
        worker_manager: Worker manager instance
        resource_manager: Dynamic resource manager instance
    
    Returns:
        dict: Merged ensemble results
    """
    # Create orchestrator
    orchestrator = MultiDeviceOrchestrator(
        task_manager=task_manager,
        worker_manager=worker_manager,
        resource_manager=resource_manager
    )
    
    # Define ensemble task with multiple model variants
    ensemble_task = {
        "task_id": "model_ensemble_task",
        "type": "inference",
        "config": {
            "input_text": "The quick brown fox jumps over the lazy dog",
            "task_type": "text_classification",
            "execution_time_ms": 1500  # Simulate inference time
        },
        "ensemble_configs": [
            {
                "variant": "bert_base_cpu",
                "model_name": "bert-base-uncased",
                "precision": "fp32",
                "hardware": "cpu",
                "resource_requirements": {
                    "cpu_cores": 4,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 0
                }
            },
            {
                "variant": "bert_large_gpu",
                "model_name": "bert-large-uncased",
                "precision": "fp16",
                "hardware": "gpu",
                "resource_requirements": {
                    "cpu_cores": 2,
                    "memory_mb": 8192,
                    "gpu_memory_mb": 8192
                }
            },
            {
                "variant": "roberta_base_cpu",
                "model_name": "roberta-base",
                "precision": "fp32",
                "hardware": "cpu",
                "resource_requirements": {
                    "cpu_cores": 4,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 0
                }
            }
        ]
    }
    
    # Orchestrate ensemble task
    logger.info("Orchestrating model ensemble across devices...")
    task_id = orchestrator.orchestrate_task(
        task_data=ensemble_task,
        strategy=SplitStrategy.ENSEMBLE
    )
    
    # Monitor resource utilization during execution
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(resource_manager, task_id, orchestrator),
        daemon=True
    )
    monitor_thread.start()
    
    # Wait for task completion
    logger.info(f"Waiting for task {task_id} to complete...")
    start_time = time.time()
    max_wait = 60.0  # Maximum wait time in seconds
    
    task_completed = False
    while time.time() - start_time < max_wait and not task_completed:
        status = orchestrator.get_task_status(task_id)
        logger.info(f"Task status: {status['status']}, Completion: {status.get('completion_percentage', 0)}%")
        
        if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task_completed = True
            break
        
        time.sleep(2.0)
    
    # Stop monitoring
    monitor_thread.join(timeout=1.0)
    
    # Check final status
    status = orchestrator.get_task_status(task_id)
    if status["status"] == TaskStatus.COMPLETED:
        logger.info(f"Task {task_id} completed successfully!")
        result = orchestrator.get_task_result(task_id)
        
        # Print merged results
        logger.info("Model Ensemble Results:")
        logger.info(f"Number of ensemble variants: {result.get('num_ensembles', 0)}")
        
        # Print ensemble predictions
        if "ensemble_predictions" in result:
            logger.info("Ensemble Predictions:")
            for prediction, value in result["ensemble_predictions"].items():
                logger.info(f"  {prediction}: {value}")
        
        # Print ensemble metrics
        if "ensemble_metrics" in result:
            logger.info("Ensemble Metrics:")
            for metric, stats in result["ensemble_metrics"].items():
                logger.info(f"  {metric}:")
                for stat_name, stat_value in stats.items():
                    logger.info(f"    {stat_name}: {stat_value}")
        
        return result
    else:
        logger.error(f"Task {task_id} failed with status: {status['status']}")
        if "error" in status:
            logger.error(f"Error: {status['error']}")
        return None


def monitor_resources(resource_manager, task_id, orchestrator):
    """
    Monitor resource utilization during task execution.
    
    Args:
        resource_manager: Dynamic resource manager instance
        task_id: Task ID to monitor
        orchestrator: Orchestrator instance for task status
    """
    logger.info("Starting resource utilization monitoring...")
    
    # Monitor until task completes
    while True:
        # Get task status
        status = orchestrator.get_task_status(task_id)
        if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            break
        
        # Get resource utilization
        utilization = resource_manager.get_worker_utilization()
        
        # Log overall utilization
        overall = utilization["overall"]
        logger.info(f"Overall utilization: CPU={overall['cpu']:.1%}, Memory={overall['memory']:.1%}, GPU={overall['gpu']:.1%}")
        
        # Log worker-specific utilization
        for worker_id, metrics in utilization["workers"].items():
            worker_util = metrics["utilization"]
            logger.info(f"Worker {worker_id}: CPU={worker_util['cpu']:.1%}, Memory={worker_util['memory']:.1%}, "
                       f"GPU={worker_util.get('gpu', 0):.1%}, Tasks={metrics['tasks']}")
        
        # Sleep before next check
        time.sleep(5.0)
    
    logger.info("Resource monitoring completed.")


class SimulatedTaskManager:
    """Simulated task manager for demonstration purposes."""
    
    def __init__(self):
        """Initialize the simulated task manager."""
        self.tasks = {}
        self.task_counter = 0
        self.task_finished_callback = None
    
    def add_task(self, task_data, priority=5):
        """Add a task to the manager."""
        task_id = task_data.get("task_id", f"task_{self.task_counter}")
        self.task_counter += 1
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "data": task_data,
            "priority": priority,
            "status": "pending",
            "subtask_id": task_data.get("subtask_id"),
            "parent_task_id": task_data.get("parent_task_id"),
            "callback": task_data.get("callback")
        }
        
        # Simulate task execution in background thread
        threading.Thread(
            target=self._execute_task,
            args=(task_id,),
            daemon=True
        ).start()
        
        return task_id
    
    def cancel_task(self, task_id):
        """Cancel a task."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "cancelled"
            return True
        return False
    
    def get_task_status(self, task_id):
        """Get the status of a task."""
        if task_id in self.tasks:
            return self.tasks[task_id]["status"]
        return "not_found"
    
    def set_task_finished_callback(self, callback):
        """Set a callback function for task completion."""
        self.task_finished_callback = callback
    
    def _execute_task(self, task_id):
        """Simulate task execution."""
        if task_id not in self.tasks:
            return
        
        # Simulate task execution time
        task_data = self.tasks[task_id]["data"]
        execution_time = task_data.get("execution_time_ms", 1000) / 1000.0
        
        # For ensemble tasks, vary execution time by variant
        if "ensemble_variant" in task_data:
            variant = task_data["ensemble_variant"]
            if "gpu" in variant:
                execution_time *= 0.7  # GPU is faster
            elif "tpu" in variant:
                execution_time *= 0.6  # TPU is even faster
        
        # Sleep to simulate execution
        time.sleep(execution_time)
        
        # Check if task was cancelled
        if self.tasks[task_id]["status"] == "cancelled":
            return
        
        # Update task status
        self.tasks[task_id]["status"] = "completed"
        
        # Generate result data based on task type
        result = {
            "status": "completed",
            "task_id": task_id,
            "execution_time_ms": execution_time * 1000,
        }
        
        # Add task-specific results
        task_type = task_data.get("type", "")
        
        if "benchmark" in task_type:
            # Add benchmark metrics
            result["metrics"] = {
                "latency_ms": execution_time * 1000,
                "throughput": 100 / execution_time,
                "memory_mb": task_data.get("config", {}).get("memory_mb", 1024)
            }
        
        elif "data_processing" in task_type:
            # Add data processing results
            result["results"] = []
            if "input_data" in task_data:
                result["results"] = [
                    {"input": item, "output": f"Processed {item}"}
                    for item in task_data["input_data"]
                ]
            result["metrics"] = {
                "processing_time_ms": execution_time * 1000,
                "items_per_second": len(task_data.get("input_data", [])) / execution_time
            }
        
        elif "inference" in task_type:
            # Add inference results
            model_variant = task_data.get("ensemble_variant", "")
            text = task_data.get("config", {}).get("input_text", "")
            
            # Simulate different model outputs
            if "bert" in model_variant.lower():
                result["predictions"] = {
                    "positive": 0.8,
                    "negative": 0.15,
                    "neutral": 0.05
                }
            elif "roberta" in model_variant.lower():
                result["predictions"] = {
                    "positive": 0.75,
                    "negative": 0.2,
                    "neutral": 0.05
                }
            else:
                result["predictions"] = {
                    "positive": 0.7,
                    "negative": 0.2,
                    "neutral": 0.1
                }
            
            result["metrics"] = {
                "inference_time_ms": execution_time * 1000,
                "tokens_per_second": len(text.split()) / execution_time
            }
        
        # If this is a model component, add component-specific outputs
        if "model_component" in task_data:
            component = task_data["model_component"]
            
            if component == "embedding":
                result["output"] = "Embedding tensor of shape [batch_size, seq_len, hidden_size]"
                result["metrics"]["embedding_time_ms"] = execution_time * 800
            
            elif component == "encoder":
                result["output"] = "Encoder tensor of shape [batch_size, seq_len, hidden_size]"
                result["metrics"]["encoder_time_ms"] = execution_time * 900
                result["metrics"]["attention_time_ms"] = execution_time * 600
            
            elif component == "pooler":
                result["output"] = "Pooler tensor of shape [batch_size, hidden_size]"
                result["metrics"]["pooler_time_ms"] = execution_time * 100
        
        # If this is a subtask, call the callback
        callback = self.tasks[task_id].get("callback")
        if callback and callback.get("type") == "subtask_result":
            subtask_id = callback.get("subtask_id")
            if subtask_id and self.task_finished_callback:
                self.task_finished_callback(subtask_id, result, True)


class SimulatedWorkerManager:
    """Simulated worker manager for demonstration purposes."""
    
    def __init__(self):
        """Initialize the simulated worker manager."""
        self.workers = {}
    
    def add_worker(self, worker_id, capabilities):
        """Add a worker with specific capabilities."""
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "capabilities": capabilities,
            "status": "active",
            "last_seen": datetime.now()
        }
    
    def get_worker_status(self, worker_id):
        """Get the status of a worker."""
        if worker_id in self.workers:
            return self.workers[worker_id]["status"]
        return "not_found"
    
    def get_worker_capabilities(self, worker_id):
        """Get the capabilities of a worker."""
        if worker_id in self.workers:
            return self.workers[worker_id]["capabilities"]
        return {}


def setup_simulated_environment():
    """
    Set up a simulated environment with task manager, worker manager, and resource manager.
    
    This creates:
    1. A simulated task manager to handle task execution
    2. A simulated worker manager with various worker types
    3. A dynamic resource manager with registered workers
    
    Returns:
        tuple: (task_manager, worker_manager, resource_manager)
    """
    # Create simulated managers
    task_manager = SimulatedTaskManager()
    worker_manager = SimulatedWorkerManager()
    
    # Create resource manager
    resource_manager = DynamicResourceManager(
        target_utilization=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3
    )
    
    # Add simulated workers with different capabilities
    worker_manager.add_worker("cpu_server_1", {
        "hardware_type": "cpu",
        "cpu_cores": 16,
        "memory_gb": 64,
        "gpu": None
    })
    
    worker_manager.add_worker("cpu_server_2", {
        "hardware_type": "cpu",
        "cpu_cores": 32,
        "memory_gb": 128,
        "gpu": None
    })
    
    worker_manager.add_worker("gpu_server_1", {
        "hardware_type": "gpu",
        "cpu_cores": 16,
        "memory_gb": 64,
        "gpu": "nvidia-v100",
        "gpu_count": 2,
        "gpu_memory_gb": 16
    })
    
    worker_manager.add_worker("gpu_server_2", {
        "hardware_type": "gpu",
        "cpu_cores": 24,
        "memory_gb": 96,
        "gpu": "nvidia-a100",
        "gpu_count": 4,
        "gpu_memory_gb": 40
    })
    
    worker_manager.add_worker("tpu_server", {
        "hardware_type": "tpu",
        "cpu_cores": 16,
        "memory_gb": 64,
        "tpu_cores": 8,
        "tpu_memory_gb": 128
    })
    
    # Register workers with resource manager
    for worker_id, worker_info in worker_manager.workers.items():
        capabilities = worker_info["capabilities"]
        
        # Convert to resource manager format
        resources = {
            "cpu": {
                "cores": capabilities.get("cpu_cores", 1),
                "available_cores": capabilities.get("cpu_cores", 1)
            },
            "memory": {
                "total_mb": capabilities.get("memory_gb", 1) * 1024,
                "available_mb": capabilities.get("memory_gb", 1) * 1024
            }
        }
        
        # Add GPU if present
        if capabilities.get("gpu"):
            resources["gpu"] = {
                "devices": capabilities.get("gpu_count", 1),
                "memory_mb": capabilities.get("gpu_memory_gb", 1) * 1024,
                "available_memory_mb": capabilities.get("gpu_memory_gb", 1) * 1024
            }
        
        # Add TPU if present
        if capabilities.get("tpu_cores"):
            resources["tpu"] = {
                "devices": capabilities.get("tpu_cores", 1),
                "memory_mb": capabilities.get("tpu_memory_gb", 1) * 1024,
                "available_memory_mb": capabilities.get("tpu_memory_gb", 1) * 1024
            }
        
        resource_manager.register_worker(worker_id, resources)
    
    # Set up callback from task manager to orchestrator
    def setup_callback_for_orchestrator(orchestrator):
        task_manager.set_task_finished_callback(orchestrator.process_subtask_result)
    
    # Return simulated environment
    return task_manager, worker_manager, resource_manager, setup_callback_for_orchestrator


def main(example_type="all"):
    """
    Run the multi-device orchestrator example.
    
    Args:
        example_type: Type of example to run (bert, dataset, ensemble, or all)
    """
    logger.info("Setting up simulated environment...")
    task_manager, worker_manager, resource_manager, setup_callback = setup_simulated_environment()
    
    try:
        # Create orchestrator to set up callback
        orchestrator = MultiDeviceOrchestrator(
            task_manager=task_manager,
            worker_manager=worker_manager,
            resource_manager=resource_manager
        )
        setup_callback(orchestrator)
        
        # Run the requested example(s)
        if example_type in ["bert", "all"]:
            logger.info("\n\n=== Running BERT Model Parallel Example ===\n")
            run_bert_benchmark_across_devices(task_manager, worker_manager, resource_manager)
        
        if example_type in ["dataset", "all"]:
            logger.info("\n\n=== Running Dataset Processing Data Parallel Example ===\n")
            run_dataset_processing_across_devices(task_manager, worker_manager, resource_manager, num_items=50)
        
        if example_type in ["ensemble", "all"]:
            logger.info("\n\n=== Running Model Ensemble Example ===\n")
            run_model_ensemble_across_devices(task_manager, worker_manager, resource_manager)
        
        # Final resource utilization report
        logger.info("\n\n=== Final Resource Utilization ===\n")
        utilization = resource_manager.get_worker_utilization()
        overall = utilization["overall"]
        logger.info(f"Overall utilization: CPU={overall['cpu']:.1%}, Memory={overall['memory']:.1%}, GPU={overall['gpu']:.1%}")
        
        for worker_id, metrics in utilization["workers"].items():
            worker_util = metrics["utilization"]
            logger.info(f"Worker {worker_id}: CPU={worker_util['cpu']:.1%}, Memory={worker_util['memory']:.1%}, "
                       f"GPU={worker_util.get('gpu', 0):.1%}, Tasks={metrics['tasks']}")
        
    except Exception as e:
        logger.exception(f"Error in multi-device orchestrator example: {e}")
    
    finally:
        # Clean up resources
        resource_manager.cleanup()
        logger.info("Example completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Device Orchestrator Example")
    parser.add_argument("--example", choices=["bert", "dataset", "ensemble", "all"], default="all",
                       help="Type of example to run")
    args = parser.parse_args()
    
    main(example_type=args.example)