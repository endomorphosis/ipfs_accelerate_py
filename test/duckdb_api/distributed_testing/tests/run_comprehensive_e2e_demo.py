#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing Demo

This script provides a comprehensive demonstration of the end-to-end testing framework
with all features enabled, including real-time monitoring, visualization, and dashboard
integration.

Usage:
    python -m duckdb_api.distributed_testing.tests.run_comprehensive_e2e_demo [options]

Options:
    --dashboard-only           Start only the dashboard, not the tests
    --monitoring-only          Start only the real-time monitoring
    --visualization-only       Start only the visualization
    --port PORT                Dashboard port (default: 8082)
    --quick                    Run quick test with fewer workers and shorter duration
    --include-failures         Include fault tolerance tests
    --open-browser             Open dashboard in browser
    --debug                    Enable debug logging
"""

import argparse
import anyio
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Add parent directory to path to ensure imports work properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

def start_monitoring_dashboard(port=8082, debug=False):
    """Start the monitoring dashboard with all features enabled."""
    cmd = [
        sys.executable,
        "-m",
        "duckdb_api.distributed_testing.run_monitoring_dashboard",
        "--port", str(port),
        "--enable-e2e-test-integration",
        "--enable-result-aggregator-integration",
        "--theme", "dark",
        "--browser"
    ]
    
    if debug:
        cmd.append("--debug")
    
    print(f"Starting monitoring dashboard on port {port}...")
    dashboard_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not debug else None,
        stderr=subprocess.PIPE if not debug else None
    )
    
    # Give the dashboard time to start
    time.sleep(5)
    print("Monitoring dashboard started")
    
    return dashboard_process

def start_realtime_monitoring(dashboard_url, test_id=None, debug=False):
    """Start real-time monitoring for tests."""
    cmd = [
        sys.executable,
        "-m",
        "duckdb_api.distributed_testing.tests.realtime_monitoring",
        "--dashboard-url", dashboard_url
    ]
    
    if test_id:
        cmd.extend(["--test-id", test_id])
    
    if debug:
        cmd.append("--debug")
    
    print(f"Starting real-time monitoring...")
    monitor_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not debug else None,
        stderr=subprocess.PIPE if not debug else None
    )
    
    print("Real-time monitoring started")
    return monitor_process

def run_e2e_tests(quick=False, include_failures=False, debug=False):
    """Run end-to-end tests."""
    cmd = [
        sys.executable,
        "-m",
        "duckdb_api.distributed_testing.tests.run_e2e_tests"
    ]
    
    if quick:
        cmd.append("--quick")
    
    if include_failures:
        cmd.append("--fault-tolerance")
    
    if debug:
        cmd.append("--debug")
    
    print("Running end-to-end tests...")
    test_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not debug else None,
        stderr=subprocess.PIPE if not debug else None
    )
    
    test_exit_code = test_process.wait()
    print(f"End-to-end tests completed with exit code: {test_exit_code}")
    
    return test_exit_code == 0

def generate_visualizations(open_browser=False, debug=False):
    """Generate visualizations for test results."""
    cmd = [
        sys.executable,
        "-m",
        "duckdb_api.distributed_testing.tests.e2e_visualization",
        "--generate-standalone",
        "--theme", "dark"
    ]
    
    if open_browser:
        cmd.append("--open-browser")
    
    if debug:
        cmd.append("--debug")
    
    print("Generating visualizations...")
    viz_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not debug else None,
        stderr=subprocess.PIPE if not debug else None
    )
    
    viz_exit_code = viz_process.wait()
    print(f"Visualization generation completed with exit code: {viz_exit_code}")
    
    return viz_exit_code == 0

def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="Comprehensive End-to-End Testing Demo")
    
    # Configuration options
    parser.add_argument("--dashboard-only", action="store_true",
                       help="Start only the dashboard, not the tests")
    parser.add_argument("--monitoring-only", action="store_true",
                       help="Start only the real-time monitoring")
    parser.add_argument("--visualization-only", action="store_true",
                       help="Start only the visualization")
    parser.add_argument("--port", type=int, default=8082,
                       help="Dashboard port (default: 8082)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with fewer workers and shorter duration")
    parser.add_argument("--include-failures", action="store_true",
                       help="Include fault tolerance tests")
    parser.add_argument("--open-browser", action="store_true",
                       help="Open dashboard in browser")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    dashboard_process = None
    monitor_process = None
    
    try:
        # Create the dashboard URL
        dashboard_url = f"http://localhost:{args.port}"
        
        # Start only specific components if requested
        if args.dashboard_only:
            dashboard_process = start_monitoring_dashboard(args.port, args.debug)
            
            if args.open_browser:
                webbrowser.open(dashboard_url)
            
            print(f"Dashboard running at {dashboard_url}")
            print("Press Ctrl+C to stop")
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
            
        elif args.monitoring_only:
            print("Starting real-time monitoring only...")
            monitor_process = start_realtime_monitoring(dashboard_url, debug=args.debug)
            
            if args.open_browser:
                webbrowser.open(f"{dashboard_url}/e2e-test-monitoring")
            
            print("Real-time monitoring started. Press Ctrl+C to stop")
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
        
        elif args.visualization_only:
            print("Generating visualizations only...")
            generate_visualizations(args.open_browser, args.debug)
        
        else:
            # Run complete demo with all components
            print("\n=== Starting Comprehensive End-to-End Testing Demo ===\n")
            
            # Step 1: Start monitoring dashboard
            print("\n=== Step 1: Starting Monitoring Dashboard ===\n")
            dashboard_process = start_monitoring_dashboard(args.port, args.debug)
            
            # Step 2: Start real-time monitoring
            print("\n=== Step 2: Starting Real-Time Monitoring ===\n")
            monitor_process = start_realtime_monitoring(dashboard_url, debug=args.debug)
            
            # Open browser if requested
            if args.open_browser:
                print("\nOpening dashboard in browser...")
                webbrowser.open(f"{dashboard_url}/e2e-test-monitoring")
                # Give time for browser to open
                time.sleep(2)
            
            # Step 3: Run end-to-end tests
            print("\n=== Step 3: Running End-to-End Tests ===\n")
            test_success = run_e2e_tests(args.quick, args.include_failures, args.debug)
            
            # Step 4: Generate visualizations
            print("\n=== Step 4: Generating Visualizations ===\n")
            viz_success = generate_visualizations(False, args.debug)
            
            # Step 5: Display summary
            print("\n=== Demo Summary ===\n")
            print(f"Dashboard URL: {dashboard_url}")
            print(f"Real-Time Monitoring: {dashboard_url}/e2e-test-monitoring")
            print(f"Test Results: {dashboard_url}/e2e-test-results")
            print(f"Test Success: {'Yes' if test_success else 'No'}")
            print(f"Visualization Success: {'Yes' if viz_success else 'No'}")
            print("\nDashboard is still running. Press Ctrl+C to stop.")
            
            # Keep dashboard running until interrupted
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        # Clean up processes
        if monitor_process:
            try:
                monitor_process.terminate()
                monitor_process.wait(timeout=5)
            except:
                pass
        
        if dashboard_process:
            try:
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)
            except:
                pass
        
        print("Demo stopped")

if __name__ == "__main__":
    main()