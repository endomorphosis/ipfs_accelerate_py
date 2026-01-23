#!/usr/bin/env python3
"""
Integrated System Example

This script demonstrates the complete integration of all major components of the 
Distributed Testing Framework with a simple runnable example that showcases:

1. Coordinator setup with all components
2. Multi-Device Orchestrator with task distribution
3. Fault Tolerance System with error handling
4. Comprehensive Monitoring Dashboard with visualization
5. Mock worker creation and task submission

This example is self-contained and includes error injection to demonstrate
fault tolerance capabilities.
"""

import os
import sys
import time
import json
import anyio
import logging
import threading
import subprocess
import random
import webbrowser
from pathlib import Path
from datetime import datetime

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = str(Path(current_dir).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("integrated_example")

def launch_mock_workers(coordinator_url, api_key, count=3):
    """Launch mock worker processes for testing."""
    worker_processes = []
    
    # Create temporary directory for worker files
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="example_workers_")
    
    # Different worker capabilities to demonstrate heterogeneous environment
    worker_capabilities = [
        # CPU worker
        {
            "hardware_types": ["cpu"],
            "memory_gb": 8,
            "cpu_cores": 4
        },
        # CPU + CUDA worker
        {
            "hardware_types": ["cpu", "cuda"],
            "cuda_compute": 8.0,
            "memory_gb": 16
        },
        # CPU + WebGPU worker
        {
            "hardware_types": ["cpu", "webgpu"],
            "browsers": ["chrome", "firefox"],
            "memory_gb": 8
        }
    ]
    
    # Launch worker processes
    for i in range(count):
        worker_id = f"example_worker_{i}"
        worker_dir = os.path.join(temp_dir, f"worker_{i}")
        os.makedirs(worker_dir, exist_ok=True)
        
        # Select capability set (cycling through the available options)
        capability_index = i % len(worker_capabilities)
        capabilities_json = json.dumps(worker_capabilities[capability_index])
        
        worker_cmd = [
            sys.executable,
            os.path.join(parent_dir, "duckdb_api/distributed_testing/worker.py"),
            "--coordinator", coordinator_url,
            "--api-key", api_key,
            "--worker-id", worker_id,
            "--work-dir", worker_dir,
            "--reconnect-interval", "2",
            "--heartbeat-interval", "3",
            "--capabilities", capabilities_json
        ]
        
        logger.info(f"Starting worker {worker_id} with capabilities: {worker_capabilities[capability_index]}")
        process = subprocess.Popen(worker_cmd)
        worker_processes.append({
            "process": process,
            "worker_id": worker_id,
            "work_dir": worker_dir,
            "capabilities": worker_capabilities[capability_index]
        })
    
    return worker_processes, temp_dir

def submit_example_tasks(coordinator, count=5, include_faults=True):
    """Submit example tasks including normal and faulty ones."""
    submitted_tasks = []
    
    # Example tasks showcasing different types and configurations
    example_tasks = [
        # Standard CPU benchmark task
        {
            "type": "benchmark",
            "config": {
                "model": "bert-base-uncased",
                "batch_sizes": [1, 2, 4],
                "iterations": 2
            },
            "requirements": {"hardware": ["cpu"]},
            "priority": 1,
            "description": "BERT benchmark (CPU)"
        },
        # GPU benchmark task
        {
            "type": "benchmark",
            "config": {
                "model": "t5-small",
                "batch_sizes": [1, 2],
                "precision": "fp16",
                "iterations": 2
            },
            "requirements": {"hardware": ["cuda"]},
            "priority": 2,
            "description": "T5 benchmark (GPU)"
        },
        # WebGPU task
        {
            "type": "benchmark",
            "config": {
                "model": "vit-base-patch16-224",
                "batch_sizes": [1],
                "browsers": ["chrome"],
                "iterations": 1
            },
            "requirements": {"hardware": ["webgpu"]},
            "priority": 3,
            "description": "ViT benchmark (WebGPU)"
        },
        # Multi-device orchestration task (model parallel)
        {
            "type": "multi_device_orchestration",
            "config": {
                "model": "llama-7b",
                "strategy": "model_parallel",
                "num_workers": 2
            },
            "requirements": {"hardware": ["cuda"]},
            "priority": 4,
            "description": "Llama-7B model parallel execution"
        },
        # Multi-device orchestration task (data parallel)
        {
            "type": "multi_device_orchestration",
            "config": {
                "model": "clip-vit-base",
                "strategy": "data_parallel",
                "batch_size": 16
            },
            "requirements": {"hardware": ["cpu", "cuda", "webgpu"]},
            "priority": 2,
            "description": "CLIP data parallel execution"
        }
    ]
    
    # Faulty task examples to demonstrate fault tolerance
    faulty_tasks = [
        # Missing required parameter
        {
            "type": "benchmark",
            "config": {
                # Missing model name
                "batch_sizes": [1, 2]
            },
            "requirements": {"hardware": ["cpu"]},
            "priority": 1,
            "description": "Faulty task - missing model parameter"
        },
        # Requesting non-existent hardware
        {
            "type": "benchmark",
            "config": {
                "model": "bert-base-uncased",
                "batch_sizes": [1]
            },
            "requirements": {"hardware": ["quantum_gpu"]},
            "priority": 2,
            "description": "Faulty task - non-existent hardware"
        },
        # Invalid command
        {
            "type": "command",
            "config": {
                "command": ["command_that_does_not_exist"]
            },
            "requirements": {"hardware": ["cpu"]},
            "priority": 3,
            "description": "Faulty task - invalid command"
        }
    ]
    
    # Submit tasks
    for i in range(count):
        # Determine if this should be a faulty task
        is_faulty = include_faults and random.random() < 0.2  # 20% chance of fault
        
        if is_faulty:
            # Select a random faulty task
            task_info = random.choice(faulty_tasks)
            logger.info(f"Submitting faulty task: {task_info['description']}")
        else:
            # Select a random normal task
            task_info = random.choice(example_tasks)
            logger.info(f"Submitting task: {task_info['description']}")
        
        try:
            task_id = coordinator.add_task(
                task_info["type"],
                task_info["config"],
                task_info["requirements"],
                task_info["priority"]
            )
            
            submitted_tasks.append({
                "task_id": task_id,
                "description": task_info["description"],
                "faulty": is_faulty
            })
            
            # Short sleep to avoid overwhelming the system
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
    
    return submitted_tasks

async def run_integrated_example():
    """Run the integrated system example."""
    logger.info("Starting Integrated System Example")
    
    try:
        # Import required components
        from duckdb_api.distributed_testing.coordinator import CoordinatorServer
        from duckdb_api.distributed_testing.multi_device_orchestrator import MultiDeviceOrchestrator
        from duckdb_api.distributed_testing.fault_tolerance_system import FaultToleranceSystem
        from duckdb_api.distributed_testing.comprehensive_monitoring_dashboard import ComprehensiveMonitoringDashboard
        
        # Set up paths and configuration
        db_dir = os.path.join(current_dir, "data")
        os.makedirs(db_dir, exist_ok=True)
        
        db_path = os.path.join(db_dir, "integrated_example.duckdb")
        metrics_db_path = os.path.join(db_dir, "integrated_example_metrics.duckdb")
        security_config_path = os.path.join(db_dir, "security_config.json")
        
        # Generate security configuration
        import secrets
        security_config = {
            "token_secret": secrets.token_hex(32),
            "api_keys": {
                "admin": secrets.token_hex(16),
                "worker": secrets.token_hex(16),
                "user": secrets.token_hex(16)
            }
        }
        
        with open(security_config_path, 'w') as f:
            json.dump(security_config, f, indent=2)
        
        logger.info(f"Generated security configuration at {security_config_path}")
        
        # Create and start coordinator
        coordinator = CoordinatorServer(
            host="localhost",
            port=8080,
            db_path=db_path,
            heartbeat_timeout=60,
            visualization_path=db_dir,
            performance_analyzer=True,
            token_secret=security_config["token_secret"]
        )
        
        # Create and attach multi-device orchestrator
        orchestrator = MultiDeviceOrchestrator(
            coordinator=coordinator,
            default_strategy="auto",
            enable_distributed=True,
            max_workers=4
        )
        
        coordinator.multi_device_orchestrator = orchestrator
        logger.info("Multi-Device Orchestrator initialized")
        
        # Create and attach fault tolerance system
        fault_tolerance = FaultToleranceSystem(
            coordinator=coordinator,
            task_manager=getattr(coordinator, 'task_manager', None),
            worker_manager=getattr(coordinator, 'worker_manager', None),
            max_retries=3,
            circuit_break_threshold=5,
            circuit_break_timeout=300,
            error_window_size=100,
            error_rate_threshold=0.5
        )
        
        coordinator.fault_tolerance_system = fault_tolerance
        logger.info("Fault Tolerance System initialized")
        
        # Start coordinator server
        coordinator_task = # TODO: Replace with task group - asyncio.create_task(coordinator.start())
        logger.info(f"Coordinator started on ws://localhost:8080")
        logger.info(f"Admin API Key: {security_config['api_keys']['admin']}")
        logger.info(f"Worker API Key: {security_config['api_keys']['worker']}")
        
        # Wait for coordinator to initialize
        await anyio.sleep(2)
        
        # Create and start comprehensive dashboard
        def start_dashboard():
            try:
                dashboard = ComprehensiveMonitoringDashboard(
                    coordinator=coordinator,
                    port=8888,
                    coordinator_url="ws://localhost:8080",
                    db_path=metrics_db_path
                )
                
                dashboard.start()
                return dashboard
            except Exception as e:
                logger.error(f"Error starting dashboard: {e}")
                return None
        
        # Start dashboard in a separate thread
        dashboard_thread = threading.Thread(target=start_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        logger.info("Comprehensive Monitoring Dashboard started at http://localhost:8888")
        
        # Optional: Open browser to dashboard
        webbrowser.open("http://localhost:8888")
        
        # Start mock workers
        worker_processes, temp_dir = launch_mock_workers(
            coordinator_url="ws://localhost:8080",
            api_key=security_config["api_keys"]["worker"],
            count=3
        )
        
        logger.info(f"Started {len(worker_processes)} mock workers")
        
        # Wait for workers to register
        await anyio.sleep(5)
        
        # Submit example tasks
        submitted_tasks = submit_example_tasks(
            coordinator=coordinator,
            count=10,
            include_faults=True
        )
        
        logger.info(f"Submitted {len(submitted_tasks)} tasks")
        
        # Wait for tasks to process (60 seconds)
        for i in range(6):
            logger.info(f"Running... ({i+1}/6) - Press Ctrl+C to stop early")
            await anyio.sleep(10)
        
        # Clean up
        logger.info("Example completed, shutting down...")
        
        # Stop coordinator (this will stop orchestrator and fault tolerance system)
        await coordinator.stop()
        
        # Terminate worker processes
        for worker in worker_processes:
            worker["process"].terminate()
            worker["process"].wait()
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        logger.info("Example shutdown complete")
        
    except Exception as e:
        logger.error(f"Error in integrated example: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    try:
        anyio.run(run_integrated_example())
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")

if __name__ == "__main__":
    main()