#!/usr/bin/env python3
"""
Run Coordinator with Load Balancer Integration

This script demonstrates how to run the Coordinator with the
LoadBalancerIntegration enabled.
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_with_load_balancer")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Coordinator with Load Balancer Integration")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--heartbeat-timeout", type=int, default=60, help="Heartbeat timeout in seconds")
    parser.add_argument("--disable-load-balancer", action="store_true", help="Disable load balancer integration")
    parser.add_argument("--scheduler", type=str, default="performance_based", 
                       choices=["performance_based", "round_robin", "weighted_round_robin", "priority_based", "affinity_based", "composite"],
                       help="Load balancer scheduler type")
    parser.add_argument("--monitoring-interval", type=int, default=15, help="Monitoring interval in seconds")
    parser.add_argument("--rebalance-interval", type=int, default=90, help="Rebalance interval in seconds")
    parser.add_argument("--visualization-path", type=str, help="Path for performance visualizations")
    
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
    
    # Load balancer configuration
    load_balancer_config = {
        "db_path": args.db_path,
        "monitoring_interval": args.monitoring_interval,
        "rebalance_interval": args.rebalance_interval,
        "default_scheduler": {
            "type": args.scheduler
        },
        "test_type_schedulers": {
            "performance": {"type": "performance_based"},
            "compatibility": {"type": "affinity_based"},
            "integration": {
                "type": "composite",
                "algorithms": [
                    {"type": "performance_based", "weight": 0.7},
                    {"type": "priority_based", "weight": 0.3}
                ]
            }
        }
    }
    
    # Configure test type to scheduler mapping based on model family
    load_balancer_config["model_family_schedulers"] = {
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
    
    # Create and start coordinator
    try:
        coordinator = CoordinatorServer(
            host=args.host,
            port=args.port,
            db_path=args.db_path,
            heartbeat_timeout=args.heartbeat_timeout,
            visualization_path=args.visualization_path,
            performance_analyzer=True,
            enable_load_balancer=not args.disable_load_balancer,
            load_balancer_config=load_balancer_config
        )
        
        # Start in a separate thread
        coordinator_thread = threading.Thread(target=coordinator.start)
        coordinator_thread.daemon = True
        coordinator_thread.start()
        
        logger.info(f"Coordinator started on {args.host}:{args.port} with load balancer {'enabled' if not args.disable_load_balancer else 'disabled'}")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down coordinator...")
            coordinator.stop()
    except Exception as e:
        logger.error(f"Error running coordinator: {e}")
        # Remove patches if error occurs
        remove_patches()

if __name__ == "__main__":
    main()