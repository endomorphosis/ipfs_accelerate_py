#!/usr/bin/env python3
"""
Bridge Service for Continuous Synchronization

This module provides a service for continuous synchronization between
the Benchmark API and the Predictive Performance API, allowing benchmark
results to be automatically recorded in the predictive performance database.
"""

import os
import sys
import json
import logging
import argparse
import time
import signal
import atexit
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bridge_service")

# Import bridge components
try:
    from test.integration.benchmark_predictive_performance_bridge import BenchmarkPredictivePerformanceBridge
    from test.integration.bridge_config import load_config, get_bridge_config, save_config, update_bridge_config
except ImportError as e:
    logger.error(f"Failed to import bridge components: {e}")
    sys.exit(1)

class BridgeService:
    """
    Service for continuous synchronization between components.
    
    This class provides functionality to:
    1. Run synchronization on a schedule
    2. Monitor the status of bridges
    3. Generate reports on synchronization status
    4. Handle graceful shutdown
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        log_file: Optional[str] = None
    ):
        """
        Initialize the bridge service.
        
        Args:
            config_path: Optional path to the configuration file
            log_file: Optional path to the log file
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.running = False
        self.bridges = {}
        
        # Configure logging
        self._configure_logging(log_file)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.shutdown)
    
    def _configure_logging(self, log_file: Optional[str] = None):
        """
        Configure logging for the service.
        
        Args:
            log_file: Optional path to the log file
        """
        log_config = self.config.get("logging", {})
        
        # Set log level
        level_name = log_config.get("level", "INFO")
        level = getattr(logging, level_name, logging.INFO)
        logging.getLogger().setLevel(level)
        
        # Set log file
        file_path = log_file or log_config.get("file")
        if file_path:
            try:
                file_handler = logging.FileHandler(file_path)
                file_handler.setFormatter(logging.Formatter(log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")))
                logging.getLogger().addHandler(file_handler)
                logger.info(f"Logging to {file_path}")
            except Exception as e:
                logger.error(f"Failed to configure file logging: {e}")
    
    def _signal_handler(self, signum, frame):
        """
        Handle signals to ensure clean shutdown.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def initialize_bridges(self):
        """
        Initialize bridge components based on configuration.
        
        Returns:
            True if at least one bridge was initialized, False otherwise
        """
        # Initialize benchmark-predictive performance bridge if enabled
        bp_config = get_bridge_config(self.config, "benchmark_predictive_performance")
        if bp_config.get("enabled", True):
            try:
                self.bridges["benchmark_predictive_performance"] = BenchmarkPredictivePerformanceBridge(
                    benchmark_db_path=bp_config.get("benchmark_db_path", "benchmark_db.duckdb"),
                    predictive_api_url=bp_config.get("predictive_api_url", "http://localhost:8080"),
                    api_key=bp_config.get("api_key")
                )
                logger.info("Initialized Benchmark-Predictive Performance bridge")
            except Exception as e:
                logger.error(f"Failed to initialize Benchmark-Predictive Performance bridge: {e}")
        
        # TODO: Initialize other bridges here as they are developed
        
        return len(self.bridges) > 0
    
    def check_bridge_connections(self) -> Dict[str, Dict[str, bool]]:
        """
        Check connections for all bridges.
        
        Returns:
            Dictionary with connection status for each bridge
        """
        status = {}
        
        for name, bridge in self.bridges.items():
            if hasattr(bridge, "check_connections"):
                try:
                    status[name] = bridge.check_connections()
                except Exception as e:
                    logger.error(f"Error checking connections for {name}: {e}")
                    status[name] = {"error": str(e)}
        
        return status
    
    def run_sync_cycle(self) -> Dict[str, Any]:
        """
        Run a synchronization cycle for all bridges.
        
        Returns:
            Dictionary with synchronization results
        """
        results = {}
        
        # Synchronize benchmark-predictive performance bridge
        if "benchmark_predictive_performance" in self.bridges:
            bridge = self.bridges["benchmark_predictive_performance"]
            bp_config = get_bridge_config(self.config, "benchmark_predictive_performance")
            
            try:
                # First sync high priority models
                high_priority_models = bp_config.get("high_priority_models", [])
                high_priority_results = {}
                
                for model in high_priority_models:
                    logger.info(f"Synchronizing high priority model: {model}")
                    result = bridge.sync_by_model(
                        model_name=model,
                        days=bp_config.get("sync_days_lookback", 30),
                        limit=bp_config.get("sync_limit", 100)
                    )
                    high_priority_results[model] = result
                
                # Then sync recent results
                logger.info("Synchronizing recent benchmark results")
                recent_results = bridge.sync_recent_results(
                    limit=bp_config.get("sync_limit", 100)
                )
                
                # Generate report if configured
                report = None
                if bp_config.get("report_output_dir"):
                    try:
                        report = bridge.generate_report()
                        
                        # Save report to file
                        report_dir = bp_config.get("report_output_dir")
                        os.makedirs(report_dir, exist_ok=True)
                        report_path = os.path.join(
                            report_dir,
                            f"benchmark_predictive_sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        )
                        
                        with open(report_path, "w") as f:
                            json.dump(report, f, indent=2)
                        
                        logger.info(f"Saved synchronization report to {report_path}")
                    except Exception as e:
                        logger.error(f"Error generating or saving report: {e}")
                
                # Compile results
                results["benchmark_predictive_performance"] = {
                    "high_priority_models": high_priority_results,
                    "recent_results": recent_results,
                    "report": report
                }
                
                logger.info(f"Completed benchmark-predictive performance synchronization cycle")
                
            except Exception as e:
                logger.error(f"Error in benchmark-predictive performance sync cycle: {e}")
                results["benchmark_predictive_performance"] = {"error": str(e)}
        
        # TODO: Add sync cycle for other bridges here as they are developed
        
        return results
    
    def start(self):
        """
        Start the bridge service.
        
        This will start the continuous synchronization based on the configured interval.
        """
        if self.running:
            logger.warning("Bridge service is already running")
            return
        
        logger.info("Starting bridge service")
        
        # Initialize bridges
        if not self.initialize_bridges():
            logger.error("Failed to initialize any bridges, service will not start")
            return
        
        # Check connections
        connection_status = self.check_bridge_connections()
        logger.info(f"Bridge connection status: {connection_status}")
        
        # Start service loop
        self.running = True
        
        try:
            # Run initial sync cycle
            logger.info("Running initial synchronization cycle")
            self.run_sync_cycle()
            
            # Continuous sync loop
            while self.running:
                # Get sync interval from configuration
                bp_config = get_bridge_config(self.config, "benchmark_predictive_performance")
                interval_minutes = bp_config.get("sync_interval_minutes", 60)
                auto_sync = bp_config.get("auto_sync_enabled", False)
                
                if auto_sync:
                    # Wait for next sync cycle
                    logger.info(f"Waiting {interval_minutes} minutes until next sync cycle")
                    
                    # Sleep in small intervals to allow for clean shutdown
                    sleep_seconds = interval_minutes * 60
                    while sleep_seconds > 0 and self.running:
                        time.sleep(min(sleep_seconds, 5))
                        sleep_seconds -= 5
                    
                    if self.running:
                        logger.info("Starting scheduled synchronization cycle")
                        self.run_sync_cycle()
                else:
                    logger.info("Auto-sync is disabled. Service will remain running but won't auto-sync")
                    
                    # Just sleep and wait for commands or shutdown
                    while self.running:
                        time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in bridge service: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """
        Shutdown the bridge service.
        
        This will close all bridges and stop the service.
        """
        if not self.running:
            return
        
        logger.info("Shutting down bridge service")
        
        # Close bridges
        for name, bridge in self.bridges.items():
            if hasattr(bridge, "close"):
                try:
                    bridge.close()
                    logger.info(f"Closed {name} bridge")
                except Exception as e:
                    logger.error(f"Error closing {name} bridge: {e}")
        
        self.bridges = {}
        self.running = False
        logger.info("Bridge service shutdown complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bridge Service for Continuous Synchronization")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    parser.add_argument("--run-once", action="store_true", help="Run one synchronization cycle and exit")
    parser.add_argument("--report-only", action="store_true", help="Generate synchronization report and exit")
    parser.add_argument("--output", type=str, help="Path to write report JSON (for --report-only)")
    
    args = parser.parse_args()
    
    # Create service
    service = BridgeService(
        config_path=args.config,
        log_file=args.log_file
    )
    
    # Initialize bridges
    if not service.initialize_bridges():
        logger.error("Failed to initialize any bridges, exiting")
        return 1
    
    # Check connections
    connection_status = service.check_bridge_connections()
    all_connected = all(
        all(status.values()) 
        for bridge_name, status in connection_status.items()
        if isinstance(status, dict)
    )
    
    if not all_connected:
        logger.error(f"Not all bridge connections are available: {connection_status}")
        return 1
    
    # Run once or report only mode
    if args.run_once:
        logger.info("Running single synchronization cycle")
        results = service.run_sync_cycle()
        
        # Print summary
        for bridge_name, result in results.items():
            if "error" in result:
                print(f"{bridge_name}: Error - {result['error']}")
            elif "recent_results" in result:
                recent = result["recent_results"]
                print(f"{bridge_name} - Recent: {recent.get('message')}")
                
                if "high_priority_models" in result:
                    for model, model_result in result["high_priority_models"].items():
                        print(f"{bridge_name} - {model}: {model_result.get('message')}")
        
        return 0
    
    elif args.report_only:
        logger.info("Generating synchronization report")
        
        # Get benchmark-predictive performance bridge
        if "benchmark_predictive_performance" in service.bridges:
            bridge = service.bridges["benchmark_predictive_performance"]
            
            try:
                report = bridge.generate_report()
                
                # Print summary
                print(f"Report generated at {report['generated_at']}")
                print(f"Total results: {report['total_results']}")
                print(f"Synced results: {report['synced_results']} ({report['sync_percentage']}%)")
                
                # Write report to file if requested
                if args.output:
                    with open(args.output, "w") as f:
                        json.dump(report, f, indent=2)
                    print(f"Report written to {args.output}")
                
                return 0
                
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                return 1
        else:
            logger.error("Benchmark-Predictive Performance bridge not available")
            return 1
    
    # Start service in continuous mode
    else:
        logger.info("Starting bridge service in continuous mode")
        service.start()
        return 0

if __name__ == "__main__":
    sys.exit(main())