#!/usr/bin/env python3
"""
Dynamic Resource Management Visualization Example

This script demonstrates the complete DRM visualization system with simulated data.
It showcases how to:
1. Create a DynamicResourceManager instance
2. Register multiple worker nodes with various resource configurations
3. Simulate task allocation and resource usage patterns
4. Create visualizations of resource utilization, scaling, allocation, and efficiency
5. Start the integrated dashboard to display these visualizations
6. Demonstrate auto-refresh and real-time updates

Usage:
    python run_drm_visualization_example.py [--host HOST] [--port PORT] [--duration MINUTES]
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("drm_visualization_example")

# Make sure the package is in the path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Mock ScalingDecision class to simulate scaling decisions
class ScalingDecision:
    """Mock scaling decision for demonstration purposes."""
    
    def __init__(self, action, reason, count=0, worker_ids=None):
        self.action = action  # scale_up, scale_down, maintain
        self.reason = reason
        self.count = count
        self.worker_ids = worker_ids or []


# Mock DynamicResourceManager class to simulate resource management
class MockDynamicResourceManager:
    """
    Mock implementation of DynamicResourceManager for demonstration purposes.
    
    This class simulates a DynamicResourceManager with worker registration,
    resource tracking, utilization calculation, and scaling decisions.
    """
    
    def __init__(self):
        # Configuration
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.target_utilization = 0.6
        
        # Workers and resources
        self.workers = {}
        self.reservations = {}
        self.tasks = {}
        
        # Scaling history
        self.scaling_history = []
        self.last_scaling_decision = None
        
        # Simulated metrics
        self.utilization_trend = "increasing"  # increasing, decreasing, stable
        self.utilization_variability = 0.1
        
        # Cloud resources (if using cloud infrastructure)
        self.cloud_resources = {
            "aws": {
                "instances": random.randint(5, 15),
                "cost_per_hour": 0.5
            },
            "gcp": {
                "instances": random.randint(3, 10),
                "cost_per_hour": 0.6
            },
            "azure": {
                "instances": random.randint(2, 8),
                "cost_per_hour": 0.55
            }
        }
        
        logger.info("MockDynamicResourceManager initialized")
        
    def register_worker(self, worker_id, resources):
        """
        Register a worker with the resource manager.
        
        Args:
            worker_id: Unique ID for the worker
            resources: Dict with worker resources (cpu, memory, gpu, etc.)
        
        Returns:
            bool: True if registration successful
        """
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already registered")
            
        # Add worker
        self.workers[worker_id] = {
            "resources": resources,
            "registered_at": datetime.now(),
            "last_heartbeat": datetime.now(),
            "status": "active",
            "utilization": {
                "cpu": random.uniform(0.1, 0.5),
                "memory": random.uniform(0.1, 0.4),
                "gpu": random.uniform(0.1, 0.3) if resources.get("gpu", 0) > 0 else 0,
                "disk": random.uniform(0.05, 0.3)
            },
            "tasks": []
        }
        
        logger.info(f"Worker {worker_id} registered with {resources}")
        return True
        
    def get_worker_resources(self):
        """
        Get the resources for all registered workers.
        
        Returns:
            Dict: Worker resources by worker ID
        """
        return {
            worker_id: worker_data["resources"]
            for worker_id, worker_data in self.workers.items()
        }
        
    def get_worker_statistics(self):
        """
        Get statistics about workers and their resources.
        
        Returns:
            Dict: Worker statistics
        """
        total_workers = len(self.workers)
        active_tasks = sum(len(worker_data["tasks"]) for worker_data in self.workers.values())
        resource_reservations = len(self.reservations)
        
        # Calculate overall utilization
        if total_workers > 0:
            cpu_util = sum(worker["utilization"]["cpu"] for worker in self.workers.values()) / total_workers
            memory_util = sum(worker["utilization"]["memory"] for worker in self.workers.values()) / total_workers
            
            # GPU utilization (only for workers with GPUs)
            gpu_workers = [w for w in self.workers.values() if w["resources"].get("gpu", 0) > 0]
            gpu_util = sum(worker["utilization"]["gpu"] for worker in gpu_workers) / len(gpu_workers) if gpu_workers else 0
            
            disk_util = sum(worker["utilization"]["disk"] for worker in self.workers.values()) / total_workers
            
            overall_util = (cpu_util * 0.4 + memory_util * 0.3 + gpu_util * 0.2 + disk_util * 0.1)
        else:
            cpu_util = memory_util = gpu_util = disk_util = overall_util = 0
        
        return {
            "total_workers": total_workers,
            "active_workers": sum(1 for w in self.workers.values() if w["status"] == "active"),
            "active_tasks": active_tasks,
            "resource_reservations": resource_reservations,
            "overall_utilization": {
                "cpu": cpu_util,
                "memory": memory_util,
                "gpu": gpu_util,
                "disk": disk_util,
                "overall": overall_util
            },
            "workers": {
                worker_id: {
                    "utilization": worker_data["utilization"],
                    "tasks": len(worker_data["tasks"]),
                    "resources": worker_data["resources"],
                    "status": worker_data["status"]
                }
                for worker_id, worker_data in self.workers.items()
            }
        }
    
    def get_resource_utilization(self):
        """
        Get the current resource utilization across all workers.
        
        Returns:
            Dict: Utilization by worker ID
        """
        return {
            worker_id: worker_data["utilization"]
            for worker_id, worker_data in self.workers.items()
        }
    
    def get_scaling_history(self):
        """
        Get the history of scaling decisions.
        
        Returns:
            List: Scaling history
        """
        return self.scaling_history
    
    def get_cloud_resources(self):
        """
        Get available cloud resources.
        
        Returns:
            Dict: Cloud resources by provider
        """
        return self.cloud_resources
    
    def get_resource_efficiency(self):
        """
        Get resource allocation efficiency for workers.
        
        Returns:
            Dict: Efficiency by worker ID
        """
        return {
            worker_id: random.uniform(0.5, 0.95)
            for worker_id in self.workers.keys()
        }
    
    def update_worker_utilization(self):
        """
        Update worker utilization based on trends and randomness.
        """
        for worker_id, worker_data in self.workers.items():
            # Determine trend direction
            if self.utilization_trend == "increasing":
                direction = 1
            elif self.utilization_trend == "decreasing":
                direction = -1
            else:  # stable
                direction = 0
                
            # Add randomness
            cpu_change = direction * random.uniform(0, 0.05) + random.uniform(-self.utilization_variability, self.utilization_variability)
            memory_change = direction * random.uniform(0, 0.04) + random.uniform(-self.utilization_variability, self.utilization_variability)
            gpu_change = direction * random.uniform(0, 0.06) + random.uniform(-self.utilization_variability, self.utilization_variability)
            disk_change = direction * random.uniform(0, 0.02) + random.uniform(-self.utilization_variability, self.utilization_variability)
            
            # Update utilization
            worker_data["utilization"]["cpu"] = max(0.1, min(0.95, worker_data["utilization"]["cpu"] + cpu_change))
            worker_data["utilization"]["memory"] = max(0.1, min(0.95, worker_data["utilization"]["memory"] + memory_change))
            
            # Only update GPU if the worker has GPU
            if worker_data["resources"].get("gpu", 0) > 0:
                worker_data["utilization"]["gpu"] = max(0.1, min(0.95, worker_data["utilization"]["gpu"] + gpu_change))
            
            worker_data["utilization"]["disk"] = max(0.05, min(0.9, worker_data["utilization"]["disk"] + disk_change))
    
    def evaluate_scaling(self):
        """
        Evaluate whether scaling is needed based on resource utilization.
        
        Returns:
            ScalingDecision: Decision about scaling
        """
        stats = self.get_worker_statistics()
        overall_util = stats["overall_utilization"]["overall"]
        
        # Check if scaling is needed
        if overall_util > self.scale_up_threshold:
            # Need to scale up
            count = random.randint(1, 3)
            decision = ScalingDecision(
                action="scale_up",
                reason=f"High utilization ({overall_util:.2f} > {self.scale_up_threshold})",
                count=count
            )
            
            # Register new workers
            for i in range(count):
                new_worker_id = f"worker-{len(self.workers) + i + 1}"
                resources = {
                    "cpu": random.choice([2, 4, 8]),
                    "memory": random.choice([4096, 8192, 16384]),
                    "gpu": random.choice([0, 1, 2]),
                    "disk": random.choice([50, 100, 200])
                }
                self.register_worker(new_worker_id, resources)
                
        elif overall_util < self.scale_down_threshold and len(self.workers) > 1:
            # Need to scale down
            count = min(random.randint(1, 2), len(self.workers) - 1)
            worker_ids = random.sample(list(self.workers.keys()), count)
            
            decision = ScalingDecision(
                action="scale_down",
                reason=f"Low utilization ({overall_util:.2f} < {self.scale_down_threshold})",
                count=count,
                worker_ids=worker_ids
            )
            
            # Remove workers
            for worker_id in worker_ids:
                del self.workers[worker_id]
                
        else:
            # No scaling needed
            decision = ScalingDecision(
                action="maintain",
                reason=f"Utilization within thresholds ({self.scale_down_threshold} < {overall_util:.2f} < {self.scale_up_threshold})"
            )
            
        # Record decision
        self.scaling_history.append({
            "timestamp": datetime.now(),
            "decision": decision
        })
        
        self.last_scaling_decision = decision
        return decision
    
    def update_utilization_trend(self):
        """
        Update the utilization trend based on random changes over time.
        """
        # Randomly change trend
        if random.random() < 0.2:
            self.utilization_trend = random.choice(["increasing", "decreasing", "stable"])
            
            if self.utilization_trend == "increasing":
                logger.info("Utilization trend is now increasing")
            elif self.utilization_trend == "decreasing":
                logger.info("Utilization trend is now decreasing")
            else:
                logger.info("Utilization trend is now stable")
                
        # Randomly change variability
        if random.random() < 0.1:
            self.utilization_variability = random.uniform(0.05, 0.2)
            logger.info(f"Utilization variability is now {self.utilization_variability:.2f}")
            

def main():
    """Run the DRM visualization example."""
    parser = argparse.ArgumentParser(description='DRM Visualization Example')
    parser.add_argument('--host', type=str, default='localhost', help='Dashboard host')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--duration', type=int, default=10, help='Duration in minutes')
    parser.add_argument('--workers', type=int, default=5, help='Initial number of workers')
    parser.add_argument('--update-interval', type=int, default=10, help='Seconds between updates')
    
    args = parser.parse_args()
    
    try:
        # Import required modules
        from dynamic_resource_management_visualization import DRMVisualization
        from dashboard.drm_visualization_integration import DRMVisualizationIntegration
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        logger.error("Make sure the distributed_testing package is in your PYTHONPATH.")
        return 1
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mock DRM
    drm = MockDynamicResourceManager()
    
    # Register initial workers
    logger.info(f"Registering {args.workers} workers")
    for i in range(args.workers):
        worker_id = f"worker-{i+1}"
        resources = {
            "cpu": random.choice([2, 4, 8, 16]),
            "memory": random.choice([4096, 8192, 16384, 32768]),
            "gpu": random.choice([0, 1, 2, 4]),
            "disk": random.choice([50, 100, 200, 500])
        }
        drm.register_worker(worker_id, resources)
    
    # Create visualization instance
    visualization = DRMVisualization(
        dynamic_resource_manager=drm,
        output_dir=output_dir,
        interactive=True,
        update_interval=args.update_interval
    )
    
    # Create dashboard integration
    integration = DRMVisualizationIntegration(
        output_dir=output_dir,
        update_interval=args.update_interval,
        resource_manager=drm
    )
    
    # Start dashboard server
    dashboard_url = visualization.start_dashboard_server(port=args.port)
    logger.info(f"Dashboard server started at {dashboard_url}")
    
    # Print instructions
    print(f"\n{'='*80}")
    print(f"DRM Visualization Example Running")
    print(f"{'='*80}")
    print(f"Dashboard URL: {dashboard_url}")
    print(f"Duration: {args.duration} minutes")
    print(f"Initial workers: {args.workers}")
    print(f"Update interval: {args.update_interval} seconds")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    try:
        # Simulation loop
        start_time = time.time()
        end_time = start_time + (args.duration * 60)
        update_count = 0
        
        while time.time() < end_time:
            current_time = time.time()
            elapsed_minutes = (current_time - start_time) / 60
            remaining_minutes = args.duration - elapsed_minutes
            
            logger.info(f"Update {update_count+1} (Elapsed: {elapsed_minutes:.1f} min, Remaining: {remaining_minutes:.1f} min)")
            
            # Update worker utilization
            drm.update_worker_utilization()
            
            # Every ~1 minute, evaluate scaling
            if update_count % (60 // args.update_interval) == 0:
                logger.info("Evaluating scaling decisions")
                decision = drm.evaluate_scaling()
                logger.info(f"Scaling decision: {decision.action} - {decision.reason}")
                
                # Update trend
                drm.update_utilization_trend()
            
            # Update visualizations
            logger.info("Updating visualizations")
            visualization_paths = integration.update_visualizations(force=True)
            logger.info(f"Updated {len(visualization_paths)} visualizations")
            
            # Wait for next update
            update_count += 1
            next_update_time = start_time + (update_count * args.update_interval)
            sleep_time = max(0, next_update_time - time.time())
            
            if sleep_time > 0:
                logger.info(f"Waiting {sleep_time:.1f} seconds until next update")
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        # Cleanup
        visualization.cleanup()
        integration.cleanup()
        
        # Create a final comprehensive dashboard
        logger.info("Creating final dashboard")
        dashboard_path = visualization.create_resource_dashboard()
        logger.info(f"Final dashboard created at {dashboard_path}")
        
        print(f"\n{'='*80}")
        print(f"DRM Visualization Example Completed")
        print(f"{'='*80}")
        print(f"Duration: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"Updates: {update_count}")
        print(f"Final dashboard: {dashboard_path}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")
        
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error running DRM visualization example: {e}", exc_info=True)
        sys.exit(1)