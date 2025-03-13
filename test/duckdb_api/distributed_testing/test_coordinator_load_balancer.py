#!/usr/bin/env python3
"""
Test the Coordinator with Load Balancer Integration

This script demonstrates how to test the Coordinator with the
LoadBalancerIntegration enabled.
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_coordinator_lb")

def simulate_worker(coordinator, worker_id, capabilities):
    """
    Simulate a worker that registers with the coordinator and
    processes tasks from the load balancer.
    
    Args:
        coordinator: The coordinator server
        worker_id: ID for this worker
        capabilities: Worker capabilities
    """
    # Register worker
    token_info = coordinator.register_worker(worker_id, capabilities)
    token = token_info.get("token")
    
    if not token:
        logger.error(f"Failed to register worker {worker_id}")
        return
    
    logger.info(f"Worker {worker_id} registered with coordinator")
    
    # Process tasks in a loop
    task_count = 0
    try:
        while True:
            # Get next task from coordinator (uses load balancer)
            task = coordinator.get_task(worker_id, token)
            
            if not task:
                # No task available, wait a bit
                time.sleep(1)
                continue
                
            task_id = task.get("task_id")
            logger.info(f"Worker {worker_id} received task {task_id}")
            
            # Simulate task processing
            process_time = task.get("config", {}).get("expected_duration", 5)
            time.sleep(min(process_time, 2))  # Cap at 2 seconds for simulation
            
            # Complete task
            result = {
                "success": True,
                "output": f"Task {task_id} completed by worker {worker_id}",
                "metrics": {
                    "processing_time": process_time,
                    "memory_used": task.get("config", {}).get("expected_memory", 1.0),
                    "processed_at": datetime.now().isoformat()
                }
            }
            
            coordinator.complete_task(worker_id, task_id, result, token)
            logger.info(f"Worker {worker_id} completed task {task_id}")
            
            task_count += 1
            if task_count >= 5:  # Process up to 5 tasks for demo
                logger.info(f"Worker {worker_id} reached task limit, exiting")
                break
                
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")
    finally:
        # Unregister worker
        coordinator.unregister_worker(worker_id, token)
        logger.info(f"Worker {worker_id} unregistered")

def create_test_tasks(coordinator, count=20):
    """
    Create test tasks of different types and requirements.
    
    Args:
        coordinator: The coordinator server
        count: Number of tasks to create
    
    Returns:
        List of created task IDs
    """
    task_ids = []
    
    # Define model families and types
    model_families = ["vision", "text", "audio", "multimodal"]
    test_types = ["performance", "compatibility", "integration"]
    
    for i in range(count):
        # Create a task with different requirements
        model_family = model_families[i % len(model_families)]
        test_type = test_types[i % len(test_types)]
        
        # Create different model IDs based on family
        model_id = None
        if model_family == "vision":
            model_id = f"vit-base-{i}"
        elif model_family == "text":
            model_id = f"bert-base-{i}"
        elif model_family == "audio":
            model_id = f"whisper-base-{i}"
        elif model_family == "multimodal":
            model_id = f"clip-base-{i}"
        
        # Create task config with model and hardware requirements
        config = {
            "model": {
                "model_id": model_id,
                "model_family": model_family
            },
            "test_type": test_type,
            "expected_duration": 5.0 + (i % 5),  # 5-10 seconds
            "expected_memory": 1.0 + (i % 4),    # 1-5 GB
            "priority": 1 + (i % 3),             # 1-3 (1 is highest)
            "hardware_requirements": {
                "minimum_memory": 0.5 + (i % 2),
                "preferred_backend": "cuda" if i % 3 == 0 else "cpu"
            }
        }
        
        # Create the task
        task_id = coordinator.add_task(
            task_type=test_type,
            config=config,
            priority=config["priority"]
        )
        
        if task_id:
            task_ids.append(task_id)
            logger.info(f"Created task {task_id} for model {model_id}")
        
    return task_ids

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Coordinator with Load Balancer")
    parser.add_argument("--task-count", type=int, default=20, help="Number of tasks to create")
    parser.add_argument("--worker-count", type=int, default=4, help="Number of workers to simulate")
    parser.add_argument("--disable-load-balancer", action="store_true", help="Disable load balancer integration")
    parser.add_argument("--scheduler", type=str, default="performance_based", 
                       choices=["performance_based", "round_robin", "weighted_round_robin", "priority_based", "affinity_based", "composite"],
                       help="Load balancer scheduler type")
    
    args = parser.parse_args()
    
    # Apply the patches to integrate the load balancer
    try:
        # Import coordinator patch (applies patches automatically)
        from duckdb_api.distributed_testing.coordinator_patch import apply_patches, remove_patches
        logger.info("Applied coordinator load balancer integration patches")
    except ImportError:
        logger.error("Failed to import coordinator_patch module. Make sure it exists in the distributed_testing directory.")
        sys.exit(1)
    
    # Import coordinator
    try:
        from duckdb_api.distributed_testing.coordinator import CoordinatorServer
    except ImportError:
        logger.error("Failed to import CoordinatorServer. Make sure it exists in the distributed_testing directory.")
        sys.exit(1)
    
    # Create in-memory database for testing
    db_path = ":memory:"
    
    # Load balancer configuration
    load_balancer_config = {
        "db_path": db_path,
        "monitoring_interval": 2,
        "rebalance_interval": 10,
        "default_scheduler": {
            "type": args.scheduler
        },
        "model_family_schedulers": {
            "vision": {"type": "performance_based"},
            "text": {"type": "weighted_round_robin"},
            "audio": {"type": "affinity_based"},
            "multimodal": {
                "type": "composite",
                "algorithms": [
                    {"type": "performance_based", "weight": 0.6},
                    {"type": "affinity_based", "weight": 0.4}
                ]
            }
        }
    }
    
    # Create coordinator
    coordinator = CoordinatorServer(
        host="localhost",
        port=0,  # Use any available port
        db_path=db_path,
        heartbeat_timeout=30,
        enable_load_balancer=not args.disable_load_balancer,
        load_balancer_config=load_balancer_config
    )
    
    # Get internal API for testing
    api = coordinator.get_internal_api()
    
    try:
        # Start coordinator
        coordinator_thread = threading.Thread(target=coordinator.start)
        coordinator_thread.daemon = True
        coordinator_thread.start()
        
        # Wait for coordinator to start
        time.sleep(2)
        
        logger.info(f"Coordinator started with load balancer {'enabled' if not args.disable_load_balancer else 'disabled'}")
        
        # Create test tasks
        task_ids = create_test_tasks(api, args.task_count)
        
        # Wait a moment for tasks to be registered
        time.sleep(1)
        
        # Create workers with different capabilities
        workers = []
        for i in range(args.worker_count):
            worker_id = f"worker-{i}"
            
            # Create different worker types
            if i % 4 == 0:
                # GPU worker good for vision models
                capabilities = {
                    "hostname": f"gpu-host-{i}",
                    "hardware": {
                        "gpu": {
                            "available": True,
                            "count": 1,
                            "cuda_available": True
                        },
                        "cpu": {
                            "cores": 8,
                            "threads": 16
                        },
                        "memory": {
                            "available_gb": 16.0
                        }
                    },
                    "tags": {
                        "vision_optimized": True
                    }
                }
            elif i % 4 == 1:
                # Audio-optimized worker
                capabilities = {
                    "hostname": f"audio-host-{i}",
                    "hardware": {
                        "gpu": {
                            "available": True,
                            "count": 1,
                            "cuda_available": True
                        },
                        "cpu": {
                            "cores": 8,
                            "threads": 16
                        },
                        "memory": {
                            "available_gb": 16.0
                        }
                    },
                    "tags": {
                        "audio_optimized": True
                    }
                }
            elif i % 4 == 2:
                # Text-optimized worker
                capabilities = {
                    "hostname": f"text-host-{i}",
                    "hardware": {
                        "cpu": {
                            "cores": 16,
                            "threads": 32
                        },
                        "memory": {
                            "available_gb": 32.0
                        }
                    },
                    "tags": {
                        "text_optimized": True
                    }
                }
            else:
                # General purpose worker
                capabilities = {
                    "hostname": f"general-host-{i}",
                    "hardware": {
                        "cpu": {
                            "cores": 4,
                            "threads": 8
                        },
                        "memory": {
                            "available_gb": 8.0
                        }
                    }
                }
            
            # Start worker thread
            worker_thread = threading.Thread(
                target=simulate_worker,
                args=(api, worker_id, capabilities)
            )
            worker_thread.daemon = True
            worker_thread.start()
            workers.append((worker_id, worker_thread))
            
            # Stagger worker start
            time.sleep(0.5)
        
        # Wait for all workers to finish
        for worker_id, thread in workers:
            thread.join(timeout=30)
            
        # Get task completion statistics
        task_stats = {}
        for task_id in task_ids:
            task = api.get_task(task_id)
            if task:
                worker_id = task.get("worker_id", "unassigned")
                status = task.get("status", "unknown")
                if status not in task_stats:
                    task_stats[status] = 0
                task_stats[status] += 1
                
                if status == "completed":
                    logger.info(f"Task {task_id} completed by worker {worker_id}")
        
        # Print statistics
        logger.info("Task completion statistics:")
        for status, count in task_stats.items():
            logger.info(f"  {status}: {count}")
            
        # Print load balancer statistics if enabled
        if not args.disable_load_balancer and hasattr(coordinator, 'load_balancer'):
            logger.info("Load balancer statistics:")
            lb = coordinator.load_balancer
            if hasattr(lb, 'bridge') and lb.bridge:
                # Get stats from bridge
                bridge = lb.bridge
                logger.info(f"  Workers registered: {len(bridge.coordinator_to_lb_worker_map)}")
                logger.info(f"  Tests submitted: {len(bridge.coordinator_to_lb_test_map)}")

    except Exception as e:
        logger.error(f"Test error: {e}")
    finally:
        # Stop coordinator
        coordinator.stop()
        
        # Remove patches
        remove_patches()
        
        logger.info("Test completed")

if __name__ == "__main__":
    main()