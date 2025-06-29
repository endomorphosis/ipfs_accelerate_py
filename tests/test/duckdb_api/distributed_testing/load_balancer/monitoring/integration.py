#!/usr/bin/env python3
"""
Load Balancer Monitoring Integration

This module integrates the monitoring dashboard with the load balancer
and coordinator components of the distributed testing framework.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import monitoring components
from duckdb_api.distributed_testing.load_balancer.monitoring.metrics_collector import MetricsCollector
from duckdb_api.distributed_testing.load_balancer.monitoring.dashboard_server import DashboardServer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("monitoring_integration")

class MonitoringIntegration:
    """
    Integrates the monitoring dashboard with the load balancer and coordinator.
    
    This class provides:
    - A metrics collector that gathers data from the load balancer and coordinator
    - A dashboard server that visualizes the collected metrics
    - Integration with the existing components
    """
    
    def __init__(self,
                 load_balancer=None,
                 coordinator=None,
                 db_path: str = "metrics.duckdb",
                 dashboard_host: str = "localhost",
                 dashboard_port: int = 5000,
                 collection_interval: float = 1.0,
                 static_folder: Optional[str] = None):
        """
        Initialize the monitoring integration.
        
        Args:
            load_balancer: Load balancer instance
            coordinator: Coordinator instance
            db_path: Path to the metrics database
            dashboard_host: Host to bind the dashboard server to
            dashboard_port: Port to bind the dashboard server to
            collection_interval: Interval in seconds between metric collections
            static_folder: Folder containing static dashboard files
        """
        # Store references to components
        self.load_balancer = load_balancer
        self.coordinator = coordinator
        
        # Create metrics collector
        self.metrics_collector = MetricsCollector(
            db_path=db_path,
            collection_interval=collection_interval
        )
        
        # Set metrics sources
        self.metrics_collector.set_sources(
            load_balancer=load_balancer,
            coordinator=coordinator
        )
        
        # Create dashboard server
        self.dashboard_server = DashboardServer(
            metrics_collector=self.metrics_collector,
            host=dashboard_host,
            port=dashboard_port,
            static_folder=static_folder
        )
        
        # Integration state
        self.running = False
    
    def start(self):
        """Start the monitoring integration."""
        if self.running:
            logger.warning("Monitoring integration already running")
            return
        
        # Start metrics collector
        self.metrics_collector.start()
        
        # Start dashboard server
        self.dashboard_server.start()
        
        self.running = True
        logger.info("Monitoring integration started")
    
    def stop(self):
        """Stop the monitoring integration."""
        if not self.running:
            logger.warning("Monitoring integration not running")
            return
        
        # Stop dashboard server
        self.dashboard_server.stop()
        
        # Stop metrics collector
        self.metrics_collector.stop()
        
        self.running = False
        logger.info("Monitoring integration stopped")
    
    def set_sources(self, load_balancer=None, coordinator=None):
        """
        Update the metrics sources.
        
        Args:
            load_balancer: Load balancer instance
            coordinator: Coordinator instance
        """
        # Update references
        if load_balancer is not None:
            self.load_balancer = load_balancer
        
        if coordinator is not None:
            self.coordinator = coordinator
        
        # Update metrics collector sources
        self.metrics_collector.set_sources(
            load_balancer=self.load_balancer,
            coordinator=self.coordinator
        )
        
        logger.info("Monitoring sources updated")


def integrate_monitoring(load_balancer=None, coordinator=None, 
                        dashboard_port: int = 5000, 
                        start: bool = True) -> MonitoringIntegration:
    """
    Integrate monitoring dashboard with load balancer and coordinator.
    
    Args:
        load_balancer: Load balancer instance
        coordinator: Coordinator instance
        dashboard_port: Port for the dashboard server
        start: Whether to start the monitoring integration immediately
        
    Returns:
        MonitoringIntegration instance
    """
    # Create integration
    integration = MonitoringIntegration(
        load_balancer=load_balancer,
        coordinator=coordinator,
        dashboard_port=dashboard_port
    )
    
    # Start if requested
    if start:
        integration.start()
    
    return integration


def integrate_with_coordinator_server(coordinator_server, 
                                     dashboard_port: int = 5000, 
                                     start: bool = True) -> MonitoringIntegration:
    """
    Integrate monitoring dashboard with a coordinator server.
    
    This utility function extracts the load balancer and coordinator
    from a coordinator server instance and sets up monitoring.
    
    Args:
        coordinator_server: CoordinatorServer instance
        dashboard_port: Port for the dashboard server
        start: Whether to start the monitoring integration immediately
        
    Returns:
        MonitoringIntegration instance
    """
    # Extract components from coordinator server
    coordinator = coordinator_server
    load_balancer = getattr(coordinator_server, 'load_balancer', None)
    
    # Create and return integration
    return integrate_monitoring(
        load_balancer=load_balancer,
        coordinator=coordinator,
        dashboard_port=dashboard_port,
        start=start
    )


def main():
    """Main entry point for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Balancer Monitoring Integration")
    parser.add_argument("--dashboard-host", type=str, default="localhost", help="Host to bind the dashboard server to")
    parser.add_argument("--dashboard-port", type=int, default=5000, help="Port to bind the dashboard server to")
    parser.add_argument("--db-path", type=str, default="metrics.duckdb", help="Path to metrics database")
    parser.add_argument("--collection-interval", type=float, default=1.0, help="Metrics collection interval in seconds")
    parser.add_argument("--static-folder", type=str, help="Folder containing static dashboard files")
    
    args = parser.parse_args()
    
    # Create integration without load balancer or coordinator for standalone operation
    integration = MonitoringIntegration(
        db_path=args.db_path,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        collection_interval=args.collection_interval,
        static_folder=args.static_folder
    )
    
    # Start integration
    integration.start()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop integration
        integration.stop()
        logger.info("Monitoring integration stopped")


if __name__ == "__main__":
    main()