#!/usr/bin/env python3
"""
Integration test for Dynamic Resource Management Visualization.

This script creates a simulated DynamicResourceManager with test data
and uses it to demonstrate all visualization capabilities of the
DRMVisualization module.

Usage:
    python run_drm_visualization_integration_test.py [--interactive] [--dashboard]

Options:
    --interactive    Use interactive Plotly visualizations
    --dashboard      Start the web dashboard server
"""

import os
import sys
import time
import argparse
import logging
import json
from datetime import datetime, timedelta
import random

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules
from dynamic_resource_manager import DynamicResourceManager, ScalingDecision
from dynamic_resource_management_visualization import DRMVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("drm_viz_integration_test")


def create_test_drm():
    """Create a test DynamicResourceManager with simulated data."""
    drm = DynamicResourceManager(
        target_utilization=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3
    )
    
    # Register test workers with varying capabilities
    drm.register_worker("worker-cpu-1", {
        "cpu": {"cores": 8, "available_cores": 6},
        "memory": {"total_mb": 16384, "available_mb": 12288}
    })
    
    drm.register_worker("worker-cpu-2", {
        "cpu": {"cores": 16, "available_cores": 12},
        "memory": {"total_mb": 32768, "available_mb": 24576}
    })
    
    drm.register_worker("worker-gpu-1", {
        "cpu": {"cores": 8, "available_cores": 6},
        "memory": {"total_mb": 16384, "available_mb": 12288},
        "gpu": {"devices": 1, "memory_mb": 8192, "available_memory_mb": 6144}
    })
    
    drm.register_worker("worker-gpu-2", {
        "cpu": {"cores": 16, "available_cores": 12},
        "memory": {"total_mb": 32768, "available_mb": 24576},
        "gpu": {"devices": 2, "memory_mb": 16384, "available_memory_mb": 12288}
    })
    
    # Create a simulated last scaling decision
    drm.last_scaling_decision = ScalingDecision(
        action="scale_up",
        reason="High overall utilization (85%) exceeds scale-up threshold (80%)",
        count=2,
        utilization=0.85,
        worker_type="gpu",
        resource_requirements={
            "cpu_cores": 8,
            "memory_mb": 16384,
            "gpu_memory_mb": 8192
        }
    )
    
    return drm


def simulate_workload(drm, duration_seconds=30, task_interval=2):
    """
    Simulate a dynamic workload by creating and releasing resource reservations.
    
    Args:
        drm: DynamicResourceManager instance
        duration_seconds: How long to run the simulation
        task_interval: Seconds between task creation
    """
    logger.info(f"Simulating workload for {duration_seconds} seconds...")
    
    # Task templates with varying resource requirements
    task_templates = [
        {"cpu_cores": 2, "memory_mb": 4096},
        {"cpu_cores": 4, "memory_mb": 8192, "gpu_memory_mb": 2048},
        {"cpu_cores": 1, "memory_mb": 2048},
        {"cpu_cores": 8, "memory_mb": 16384, "gpu_memory_mb": 4096},
    ]
    
    active_tasks = {}
    start_time = time.time()
    task_id_counter = 1
    
    while time.time() - start_time < duration_seconds:
        # Randomly create or release tasks
        if random.random() < 0.7 and len(active_tasks) < 20:  # 70% chance to create task if < 20 active
            # Create a new task
            task_id = f"task-{task_id_counter}"
            task_id_counter += 1
            
            # Choose random task template
            template = random.choice(task_templates)
            
            # Reserve resources
            reservation_id = drm.reserve_resources(task_id, template)
            if reservation_id:
                active_tasks[task_id] = {
                    "reservation_id": reservation_id,
                    "created_at": time.time(),
                    "duration": random.uniform(5, 15)  # Random duration between 5-15 seconds
                }
                logger.info(f"Created {task_id} with reservation {reservation_id}")
        
        # Check for tasks to release
        current_time = time.time()
        for task_id in list(active_tasks.keys()):
            task = active_tasks[task_id]
            if current_time - task["created_at"] > task["duration"]:
                # Release resources
                if drm.release_resources(task["reservation_id"]):
                    logger.info(f"Released {task_id} with reservation {task['reservation_id']}")
                    del active_tasks[task_id]
        
        # Sleep before next iteration
        time.sleep(task_interval)
    
    # Cleanup - release any remaining tasks
    for task_id, task in active_tasks.items():
        if drm.release_resources(task["reservation_id"]):
            logger.info(f"Cleanup: Released {task_id} with reservation {task['reservation_id']}")
    
    logger.info("Workload simulation completed")


def create_simulated_cloud_data(visualization):
    """Add simulated cloud provider data to the visualization."""
    # Create simulated cloud usage history
    now = datetime.now()
    cloud_usage_history = {}
    
    # AWS provider data
    aws_workers = []
    aws_costs = []
    for i in range(10):
        timestamp = now - timedelta(minutes=(10-i)*5)
        # Simulate increasing worker count
        count = 3 + min(i, 5)
        workers = [f"aws-worker-{j}" for j in range(1, count+1)]
        
        aws_workers.append({
            "timestamp": timestamp,
            "count": count,
            "workers": workers
        })
        
        # Simulate increasing cost
        aws_costs.append({
            "timestamp": timestamp,
            "cost": 0.10 * count
        })
    
    # GCP provider data
    gcp_workers = []
    gcp_costs = []
    for i in range(10):
        timestamp = now - timedelta(minutes=(10-i)*5)
        # Simulate stable worker count
        count = 2
        workers = [f"gcp-worker-{j}" for j in range(1, count+1)]
        
        gcp_workers.append({
            "timestamp": timestamp,
            "count": count,
            "workers": workers
        })
        
        # Simulate stable cost
        gcp_costs.append({
            "timestamp": timestamp,
            "cost": 0.15 * count
        })
    
    # Add to visualization
    cloud_usage_history["aws"] = {
        "workers": aws_workers,
        "cost": aws_costs
    }
    
    cloud_usage_history["gcp"] = {
        "workers": gcp_workers,
        "cost": gcp_costs
    }
    
    visualization.cloud_usage_history = cloud_usage_history
    logger.info("Added simulated cloud provider data")


def simulate_scaling_history(visualization, count=10):
    """Add simulated scaling history to the visualization."""
    now = datetime.now()
    scaling_history = []
    
    # Alternate between scale up and scale down events
    for i in range(count):
        timestamp = now - timedelta(minutes=(count-i)*15)
        
        if i % 3 == 0:  # Scale up
            decision = {
                "action": "scale_up",
                "reason": f"High utilization ({random.uniform(0.8, 0.95):.2f}) exceeds threshold",
                "count": random.randint(1, 3),
                "worker_type": random.choice(["cpu", "gpu"]),
                "utilization": random.uniform(0.8, 0.95)
            }
        elif i % 3 == 1:  # Scale down
            decision = {
                "action": "scale_down",
                "reason": f"Low utilization ({random.uniform(0.1, 0.3):.2f}) below threshold",
                "count": random.randint(1, 2),
                "worker_ids": [f"worker-{random.randint(1, 10)}" for _ in range(random.randint(1, 2))],
                "utilization": random.uniform(0.1, 0.3)
            }
        else:  # Maintain
            decision = {
                "action": "maintain",
                "reason": f"Utilization ({random.uniform(0.4, 0.7):.2f}) within target range",
                "utilization": random.uniform(0.4, 0.7)
            }
        
        scaling_history.append({
            "timestamp": timestamp,
            "decision": decision
        })
    
    visualization.scaling_history = scaling_history
    logger.info("Added simulated scaling history")


def main():
    """Run the integration test."""
    parser = argparse.ArgumentParser(description="DRM Visualization Integration Test")
    parser.add_argument("--interactive", action="store_true", 
                        help="Use interactive Plotly visualizations")
    parser.add_argument("--dashboard", action="store_true", 
                        help="Start the web dashboard server")
    parser.add_argument("--output-dir", default="./visualization_output",
                        help="Output directory for visualizations")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of workload simulation in seconds")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting DRM visualization integration test")
    
    # Create test DRM
    drm = create_test_drm()
    logger.info(f"Created test DRM with {len(drm.worker_resources)} workers")
    
    # Create visualization
    visualization = DRMVisualization(
        dynamic_resource_manager=drm,
        output_dir=args.output_dir,
        interactive=args.interactive,
        dashboard_port=8889,
        update_interval=2  # Fast updates for testing
    )
    logger.info(f"Created DRM visualization with output dir: {args.output_dir}")
    
    # Add simulated cloud data
    create_simulated_cloud_data(visualization)
    
    # Add simulated scaling history
    simulate_scaling_history(visualization)
    
    # Run workload simulation
    simulate_workload(drm, duration_seconds=args.duration)
    
    # Start dashboard if requested
    if args.dashboard:
        url = visualization.start_dashboard_server()
        logger.info(f"Dashboard server started at {url}")
        logger.info("Press Ctrl+C to stop")
        try:
            # Keep running while the dashboard is active
            while visualization.dashboard_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        finally:
            visualization.stop_dashboard_server()
    
    # Generate all visualizations
    logger.info("Generating visualizations...")
    
    heatmap_path = visualization.create_resource_utilization_heatmap()
    logger.info(f"Generated resource utilization heatmap: {heatmap_path}")
    
    scaling_path = visualization.create_scaling_history_visualization()
    logger.info(f"Generated scaling history visualization: {scaling_path}")
    
    allocation_path = visualization.create_resource_allocation_visualization()
    logger.info(f"Generated resource allocation visualization: {allocation_path}")
    
    efficiency_path = visualization.create_resource_efficiency_visualization()
    logger.info(f"Generated resource efficiency visualization: {efficiency_path}")
    
    cloud_path = visualization.create_cloud_resource_visualization()
    logger.info(f"Generated cloud resource visualization: {cloud_path}")
    
    # Create dashboard only in interactive mode
    dashboard_path = None
    if args.interactive:
        try:
            dashboard_path = visualization.create_resource_dashboard()
            logger.info(f"Generated comprehensive dashboard: {dashboard_path}")
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
    else:
        logger.info("Skipping dashboard creation in non-interactive mode")
    
    # Clean up
    visualization.cleanup()
    drm.cleanup()
    
    logger.info("Integration test completed successfully")
    logger.info(f"Output files are in: {args.output_dir}")
    
    # Print instructions for viewing the dashboard
    if dashboard_path:
        logger.info(f"To view the dashboard, open: {dashboard_path}")
    else:
        logger.info("To generate an interactive dashboard, run with --interactive flag")


if __name__ == "__main__":
    main()