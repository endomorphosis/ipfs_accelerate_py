#!/usr/bin/env python3
"""
Coordinator-Orchestrator Integration Example

This module demonstrates how to use the Coordinator-Orchestrator Integration
for complex distributed task orchestration scenarios across multiple worker nodes.

Example usage:
$ python coordinator_orchestrator_example.py --host 127.0.0.1 --port 8080 --db-path ./benchmark_db.duckdb
"""

import os
import sys
import json
import time
import uuid
import asyncio
import argparse
import logging
import threading
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure parent directory is in the path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules
from duckdb_api.distributed_testing.coordinator import CoordinatorServer
from duckdb_api.distributed_testing.multi_device_orchestrator import SplitStrategy
from duckdb_api.distributed_testing.coordinator_orchestrator_integration import (
    integrate_orchestrator_with_coordinator
)


async def run_coordinator_with_orchestration(host, port, db_path, api_key=None):
    """
    Run a coordinator server with integrated orchestration capabilities.
    
    Args:
        host: Host address to bind
        port: Port to bind
        db_path: Path to the DuckDB database
        api_key: Optional API key for authentication
    """
    # Create and initialize the coordinator server
    coordinator = CoordinatorServer(host=host, port=port, db_path=db_path, api_key=api_key)
    
    # Start the coordinator server
    await coordinator.start()
    
    try:
        # Integrate the orchestrator with the coordinator
        integration = integrate_orchestrator_with_coordinator(coordinator)
        logger.info("Coordinator-Orchestrator integration complete")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Stop the integration and coordinator
        if hasattr(coordinator, 'orchestrator_integration'):
            coordinator.orchestrator_integration.stop()
        await coordinator.stop()


def example_data_parallel_task():
    """
    Example of a data parallel orchestrated task.
    This task processes a dataset in parallel across multiple worker nodes.
    
    Returns:
        dict: Task definition
    """
    # Create a large dataset to process in parallel
    dataset = [{"id": i, "value": f"data_{i}"} for i in range(100)]
    
    # Prepare the task
    task = {
        "task_data": {
            "type": "data_processing",
            "name": "example_data_parallel",
            "description": "Process a dataset in parallel across multiple workers",
            "input_data": dataset,
            "config": {
                "processing_type": "basic_transform",
                "output_format": "json"
            }
        },
        "strategy": "data_parallel",
        "priority": 5
    }
    
    return task


def example_model_parallel_task():
    """
    Example of a model parallel orchestrated task.
    This task distributes a BERT model across multiple worker nodes.
    
    Returns:
        dict: Task definition
    """
    # Define model components for parallel execution
    model_task = {
        "task_data": {
            "type": "model_inference",
            "name": "bert_model_parallel",
            "description": "Run BERT model with model parallelism",
            "model_components": [
                "embedding", 
                "encoder_1_to_4", 
                "encoder_5_to_8", 
                "encoder_9_to_12", 
                "pooler"
            ],
            "input_text": "Example text for BERT model inference",
            "config": {
                "model_type": "bert",
                "model_name": "bert-base-uncased",
                "precision": "fp16",
                "batch_size": 4
            }
        },
        "strategy": "model_parallel",
        "priority": 7
    }
    
    return model_task


def example_ensemble_task():
    """
    Example of an ensemble orchestrated task.
    This task runs multiple model variants in parallel and combines their results.
    
    Returns:
        dict: Task definition
    """
    # Define an ensemble of model configurations
    ensemble_task = {
        "task_data": {
            "type": "model_ensemble",
            "name": "vision_ensemble",
            "description": "Run an ensemble of vision models",
            "input_data": ["path/to/image1.jpg", "path/to/image2.jpg"],
            "ensemble_configs": [
                {"model_name": "vit-base-patch16-224", "variant": "vit_base"},
                {"model_name": "vit-large-patch16-224", "variant": "vit_large"},
                {"model_name": "deit-base-patch16-224", "variant": "deit_base"},
                {"model_name": "beit-base-patch16-224", "variant": "beit_base"}
            ],
            "config": {
                "ensemble_method": "averaging",
                "threshold": 0.5
            }
        },
        "strategy": "ensemble",
        "priority": 6
    }
    
    return ensemble_task


def example_function_parallel_task():
    """
    Example of a function parallel orchestrated task.
    This task distributes different benchmark functions across worker nodes.
    
    Returns:
        dict: Task definition
    """
    # Define functions for parallel execution
    function_task = {
        "task_data": {
            "type": "benchmark",
            "name": "comprehensive_benchmark",
            "description": "Run comprehensive benchmarks across multiple workers",
            "model_name": "bert-base-uncased",
            "functions": [
                "latency_benchmark", 
                "throughput_benchmark", 
                "memory_usage_benchmark", 
                "power_consumption_benchmark",
                "precision_evaluation"
            ],
            "config": {
                "batch_sizes": [1, 2, 4, 8, 16],
                "iterations": 100,
                "warmup_iterations": 10
            }
        },
        "strategy": "function_parallel",
        "priority": 5
    }
    
    return function_task


def example_pipeline_parallel_task():
    """
    Example of a pipeline parallel orchestrated task.
    This task processes data in stages across multiple worker nodes.
    
    Returns:
        dict: Task definition
    """
    # Define pipeline stages
    pipeline_task = {
        "task_data": {
            "type": "data_pipeline",
            "name": "audio_processing_pipeline",
            "description": "Process audio files through a multi-stage pipeline",
            "input_data": ["path/to/audio1.wav", "path/to/audio2.wav"],
            "pipeline_stages": [
                "audio_loading", 
                "feature_extraction", 
                "model_inference", 
                "post_processing", 
                "result_formatting"
            ],
            "config": {
                "model_name": "whisper-small",
                "feature_type": "mel_spectrogram",
                "sample_rate": 16000
            }
        },
        "strategy": "pipeline_parallel",
        "priority": 8
    }
    
    return pipeline_task


async def run_orchestration_examples(host, port, api_key=None):
    """
    Run examples of different orchestration strategies.
    
    Args:
        host: Host address of the coordinator server
        port: Port of the coordinator server
        api_key: Optional API key for authentication
    """
    # Create API base URL
    base_url = f"http://{host}:{port}"
    
    # Function to call API
    async def call_api(endpoint, data):
        """Call the coordinator API."""
        import aiohttp
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
            
        url = f"{base_url}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                return await response.json()
    
    # Initialize examples
    examples = [
        ("Data Parallel", example_data_parallel_task()),
        ("Model Parallel", example_model_parallel_task()),
        ("Ensemble", example_ensemble_task()),
        ("Function Parallel", example_function_parallel_task()),
        ("Pipeline Parallel", example_pipeline_parallel_task())
    ]
    
    # Run each example
    task_ids = {}
    for name, task in examples:
        logger.info(f"Submitting {name} orchestration task")
        
        # Submit the task
        result = await call_api("/api/orchestrate", task)
        
        if result.get("success"):
            task_id = result["task_id"]
            task_ids[name] = task_id
            logger.info(f"  Submitted {name} task with ID: {task_id}")
        else:
            logger.error(f"  Failed to submit {name} task: {result.get('error')}")
    
    # Monitor task status
    if task_ids:
        logger.info("Monitoring task status...")
        for _ in range(10):  # Poll for status 10 times
            for name, task_id in task_ids.items():
                result = await call_api("/api/orchestrated_task", {"task_id": task_id})
                
                if result.get("success"):
                    task_status = result["task_status"]
                    completion = task_status.get("completion_percentage", 0)
                    status = task_status.get("status", "unknown")
                    
                    logger.info(f"  {name} - Status: {status}, Completion: {completion}%")
                
            await asyncio.sleep(5)  # Wait before polling again
        
        # Show final task listing
        logger.info("Final task listing:")
        result = await call_api("/api/orchestrated_tasks", {})
        
        if result.get("success"):
            tasks = result.get("tasks", [])
            for task in tasks:
                logger.info(f"  Task {task['task_id']} - Status: {task['status']}, "
                          f"Completion: {task['completion_percentage']}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Coordinator-Orchestrator Integration Example")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--examples", action="store_true", help="Run example orchestration tasks")
    
    args = parser.parse_args()
    
    if args.examples:
        # Run examples against an existing coordinator
        asyncio.run(run_orchestration_examples(args.host, args.port, args.api_key))
    else:
        # Run coordinator with orchestration
        asyncio.run(run_coordinator_with_orchestration(
            args.host, args.port, args.db_path, args.api_key
        ))


if __name__ == "__main__":
    main()