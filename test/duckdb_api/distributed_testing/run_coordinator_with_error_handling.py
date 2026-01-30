#!/usr/bin/env python3
"""
Run Coordinator with Enhanced Error Handling

This script launches the coordinator server with enhanced error handling capabilities integrated.

Usage:
    python run_coordinator_with_error_handling.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.duckdb.distributed_testing.coordinator import Coordinator
from data.duckdb.distributed_testing.coordinator_error_integration import integrate_error_handler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_launcher")

def main():
    """Main function to run the coordinator with enhanced error handling."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run coordinator with enhanced error handling")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to the DuckDB database")
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Logging level")
    parser.add_argument("--load-balancer", action="store_true", help="Enable load balancer integration")
    parser.add_argument("--dashboard", action="store_true", help="Enable monitoring dashboard")
    parser.add_argument("--result-aggregator", action="store_true", help="Enable result aggregator integration")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create coordinator instance
    logger.info("Creating coordinator instance...")
    coordinator = Coordinator(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        api_key=args.api_key
    )
    
    # Integrate enhanced error handling
    logger.info("Integrating enhanced error handling...")
    integrate_error_handler(coordinator)
    logger.info("Enhanced error handling integrated successfully")
    
    # Enable additional integrations if requested
    if args.load_balancer:
        logger.info("Enabling load balancer integration...")
        try:
            from data.duckdb.distributed_testing.coordinator_load_balancer_integration import integrate_load_balancer
            integrate_load_balancer(coordinator)
            logger.info("Load balancer integration enabled")
        except ImportError:
            logger.warning("Load balancer integration not available. Skipping.")
    
    if args.result_aggregator:
        logger.info("Enabling result aggregator integration...")
        try:
            from data.duckdb.distributed_testing.result_aggregator_integration import integrate_result_aggregator
            integrate_result_aggregator(coordinator)
            logger.info("Result aggregator integration enabled")
        except ImportError:
            logger.warning("Result aggregator integration not available. Skipping.")
    
    if args.dashboard:
        logger.info("Enabling monitoring dashboard...")
        try:
            from data.duckdb.distributed_testing.dashboard.monitoring_dashboard_integration import integrate_dashboard
            integrate_dashboard(coordinator)
            logger.info("Monitoring dashboard enabled")
        except ImportError:
            logger.warning("Monitoring dashboard integration not available. Skipping.")
    
    # Run the coordinator server
    logger.info(f"Starting coordinator server on {args.host}:{args.port}...")
    coordinator.run()

if __name__ == "__main__":
    main()