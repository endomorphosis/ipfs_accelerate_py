#!/usr/bin/env python3
"""
Run the Monitoring Dashboard with Dynamic Resource Management Visualization Integration.

This script demonstrates how to run the monitoring dashboard with the DRM visualization
integration enabled, allowing real-time monitoring of resource allocation, utilization, 
and scaling decisions.

Usage:
    python run_dashboard_with_drm_visualization.py [--host HOST] [--port PORT]
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("drm_dashboard_demo")

# Make sure the package is in the path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

async def main():
    """Run the monitoring dashboard with DRM visualization integration."""
    parser = argparse.ArgumentParser(description='Run Monitoring Dashboard with DRM Visualization')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind server to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--no-mock', action='store_true', help='Disable mock data generation')
    args = parser.parse_args()
    
    # Import needed modules
    try:
        from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
        from duckdb_api.distributed_testing.dashboard.drm_visualization_integration import DRMVisualizationIntegration
        from duckdb_api.distributed_testing.dynamic_resource_management_visualization import DRMVisualization
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        logger.error("Make sure the duckdb_api.distributed_testing package is installed and in your PYTHONPATH.")
        return 1
    
    # Create a mock resource manager if requested
    if not args.no_mock:
        logger.info("Creating mock dynamic resource manager for demonstration")
        from unittest.mock import MagicMock
        import random
        import time
        
        # Create a mock resource manager
        mock_resource_manager = MagicMock()
        
        # Define worker resources
        worker_resources = {
            f"worker-{i}": {
                "cpu": random.choice([2, 4, 8, 16]),
                "memory": random.choice([4096, 8192, 16384, 32768]),
                "gpu": random.choice([0, 1, 2, 4]),
                "disk": random.choice([50, 100, 200, 500])
            } for i in range(1, 6)
        }
        
        # Define cloud resources
        cloud_resources = {
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
        
        # Create scaling history
        scaling_history = []
        current_time = time.time()
        for i in range(10):
            event_time = current_time - (i * 3600)
            event_type = random.choice(["scale_up", "scale_down"])
            scaling_history.append({
                "timestamp": event_time,
                "event": event_type,
                "workers_added" if event_type == "scale_up" else "workers_removed": random.randint(1, 3),
                "reason": random.choice([
                    "High CPU utilization",
                    "Memory pressure",
                    "Queue backlog",
                    "Idle workers",
                    "Scheduled scaling",
                    "Manual intervention"
                ])
            })
        
        # Set up mock methods
        mock_resource_manager.get_worker_resources.return_value = worker_resources
        mock_resource_manager.get_resource_utilization.return_value = {
            worker_id: {
                "cpu": random.uniform(0.3, 0.9),
                "memory": random.uniform(0.2, 0.8),
                "gpu": random.uniform(0.1, 0.9) if resources.get("gpu", 0) > 0 else 0,
                "disk": random.uniform(0.1, 0.7)
            }
            for worker_id, resources in worker_resources.items()
        }
        mock_resource_manager.get_scaling_history.return_value = scaling_history
        mock_resource_manager.get_cloud_resources.return_value = cloud_resources
        mock_resource_manager.get_resource_efficiency.return_value = {
            worker_id: random.uniform(0.5, 0.95)
            for worker_id in worker_resources.keys()
        }
        
        resource_manager = mock_resource_manager
        logger.info("Mock resource manager created successfully")
    else:
        # Try to import the real resource manager
        try:
            from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager
            resource_manager = DynamicResourceManager()
            logger.info("Using real DynamicResourceManager instance")
        except (ImportError, Exception) as e:
            logger.error(f"Error initializing real DynamicResourceManager: {e}")
            logger.warning("Falling back to running without resource manager")
            resource_manager = None
    
    # Create dashboard directory
    dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboards")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Initialize the dashboard
    logger.info(f"Initializing dashboard (host={args.host}, port={args.port})")
    dashboard = MonitoringDashboard(
        host=args.host,
        port=args.port,
        dashboard_dir=dashboard_dir,
        enable_visualization_integration=True
    )
    
    # Store the resource manager in the dashboard
    dashboard.resource_manager = resource_manager
    
    # Start the dashboard
    logger.info(f"Starting dashboard at http://{args.host}:{args.port}")
    logger.info("Navigate to http://{args.host}:{args.port}/drm-dashboard to view the DRM visualizations")
    await dashboard.start()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        logging.error(f"Error running dashboard: {e}", exc_info=True)
        sys.exit(1)