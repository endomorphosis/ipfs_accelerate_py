#!/usr/bin/env python3
"""
Run the End-to-End Test for the Advanced Fault Tolerance System

This script makes it easy to run the end-to-end test for the Advanced Fault Tolerance System,
which tests circuit breakers, recovery strategies, and dashboard visualization.

Example usage:
    # Run with default settings
    python run_fault_tolerance_e2e_test.py

    # Run with custom workers and tasks
    python run_fault_tolerance_e2e_test.py --workers 5 --tasks 30

    # Run with more failures
    python run_fault_tolerance_e2e_test.py --failures 10 --worker-failures 2
"""

import os
import sys
import anyio
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import test module
from duckdb_api.distributed_testing.tests.test_end_to_end_fault_tolerance import (
    FaultToleranceTestHarness, main as test_main
)

# Import browser automation bridge
from ipfs_accelerate_selenium_bridge import (
    BrowserAutomationBridge, create_browser_circuit_breaker, CircuitState, 
    get_browser_circuit_breaker_metrics, get_global_health_percentage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("run_fault_tolerance_e2e_test")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run End-to-End Test for Advanced Fault Tolerance System")
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Hostname for the coordinator server"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the coordinator server"
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8080,
        help="Port for the dashboard server"
    )
    
    parser.add_argument(
        "--output-dir",
        default=f"./e2e_fault_tolerance_test_{int(time.time())}",
        help="Directory for test outputs"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of worker clients"
    )
    
    parser.add_argument(
        "--tasks",
        type=int,
        default=20,
        help="Number of tasks to submit"
    )
    
    parser.add_argument(
        "--failures",
        type=int,
        default=5,
        help="Number of task failures to introduce"
    )
    
    parser.add_argument(
        "--worker-failures",
        type=int,
        default=1,
        help="Number of worker failures to introduce"
    )
    
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Generate dashboard from existing metrics only"
    )
    
    parser.add_argument(
        "--use-real-browsers",
        action="store_true",
        help="Enable real browser testing with Selenium (requires properly configured virtual environment)"
    )
    
    parser.add_argument(
        "--browser",
        choices=["chrome", "firefox", "edge", "all"],
        default="all",
        help="Specific browser to test with (default: all available browsers)"
    )
    
    parser.add_argument(
        "--platform",
        choices=["webgpu", "webnn", "all"],
        default="all",
        help="Web platform to test (default: all available platforms)"
    )
    
    args = parser.parse_args()
    
    # If dashboard-only mode, handle separately
    if args.dashboard_only:
        return generate_dashboard_only(args)
    
    try:
        # Check for Selenium if using real browsers
        if args.use_real_browsers:
            try:
                import selenium
                from selenium import webdriver
                logger.info(f"Using Selenium version {selenium.__version__} for real browser testing")
                
                # Check for BrowserAutomationBridge
                try:
                    from ipfs_accelerate_selenium_bridge import BrowserAutomationBridge
                    logger.info("Using BrowserAutomationBridge for real browser testing")
                except ImportError:
                    logger.error("BrowserAutomationBridge not found but --use-real-browsers specified.")
                    logger.error("Please make sure ipfs_accelerate_selenium_bridge.py is properly installed")
                    return 1
                    
            except ImportError:
                logger.error("Selenium not found but --use-real-browsers specified.")
                logger.error("Please activate the virtual environment in the parent directory:")
                logger.error("  source ../venv/bin/activate")
                logger.error("Then install Selenium:") 
                logger.error("  pip install selenium webdriver-manager")
                return 1
        
        # Get the event loop
        loop = # TODO: Remove event loop management - asyncio.get_event_loop()
        
        # Create test harness
        harness = FaultToleranceTestHarness(
            coordinator_host=args.host,
            coordinator_port=args.port,
            dashboard_port=args.dashboard_port,
            output_dir=args.output_dir,
            num_workers=args.workers,
            task_count=args.tasks,
            use_real_browsers=args.use_real_browsers
        )
        
        # Pass browser and platform preferences to the test harness
        if args.use_real_browsers:
            if hasattr(harness, 'browser_preferences'):
                harness.browser_preferences = {
                    'browser': args.browser,
                    'platform': args.platform
                }
        
        # Run test
        success = loop.run_until_complete(harness.run_test())
        
        # Generate report
        loop.run_until_complete(harness.generate_report())
        
        # Print result
        if success:
            logger.info("✅ Fault tolerance end-to-end test completed successfully!")
            logger.info(f"Report available at: {args.output_dir}/fault_tolerance_test_report.md")
            logger.info(f"Dashboard available at: {args.output_dir}/dashboards/circuit_breakers/circuit_breaker_dashboard.html")
            return 0
        else:
            logger.error("❌ Fault tolerance end-to-end test failed!")
            return 1
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Error running test: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

def generate_dashboard_only(args):
    """Generate dashboard from existing metrics."""
    from duckdb_api.distributed_testing.dashboard.circuit_breaker_visualization import (
        CircuitBreakerVisualization
    )
    
    try:
        # Check if metrics file exists
        metrics_file = os.path.join(args.output_dir, "fault_tolerance_metrics.json")
        if not os.path.exists(metrics_file):
            logger.error(f"Metrics file not found: {metrics_file}")
            return 1
        
        # Load metrics
        import json
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        # Create visualization
        visualization = CircuitBreakerVisualization(
            output_dir=os.path.join(args.output_dir, "dashboards/circuit_breakers")
        )
        
        # Generate dashboard
        dashboard_html = visualization.generate_dashboard(metrics.get("circuit_breakers", {}))
        
        logger.info("✅ Dashboard generated successfully!")
        logger.info(f"Dashboard available at: {args.output_dir}/dashboards/circuit_breakers/circuit_breaker_dashboard.html")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())