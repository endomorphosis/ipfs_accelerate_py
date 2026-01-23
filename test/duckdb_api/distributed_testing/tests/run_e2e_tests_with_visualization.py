#!/usr/bin/env python3
"""
Run End-to-End Tests with Dashboard Integration

This script runs the end-to-end tests, generates visualizations, and integrates
with the monitoring dashboard.

Usage:
    python -m duckdb_api.distributed_testing.tests.run_e2e_tests_with_visualization [options]

Options:
    --quick                           Run quick test only
    --comprehensive                   Run comprehensive tests
    --fault-tolerance                 Include fault tolerance tests
    --dashboard-url URL               URL of monitoring dashboard (default: http://localhost:8082)
    --report-dir DIR                  Directory for test reports (default: ./e2e_test_reports)
    --visualization-dir DIR           Directory for visualizations (default: ./e2e_visualizations)
    --theme THEME                     Visualization theme [light,dark] (default: dark)
    --generate-standalone             Generate standalone HTML visualizations
    --skip-dashboard-integration      Skip integration with monitoring dashboard
    --open-browser                    Open visualization in browser when done
    --debug                           Enable debug logging
"""

import argparse
import anyio
import json
import logging
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

# Add parent directory to path to ensure imports work properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

async def run_test(args):
    """Run the end-to-end test with the given arguments."""
    # Construct the command for running tests
    test_cmd = [
        "python", "-m", "duckdb_api.distributed_testing.tests.run_e2e_tests"
    ]
    
    # Add options based on args
    if args.quick:
        test_cmd.append("--quick")
    if args.comprehensive:
        test_cmd.append("--comprehensive")
    if args.fault_tolerance:
        test_cmd.append("--fault-tolerance")
    if args.generate_report:
        test_cmd.append("--generate-report")
    if args.debug:
        test_cmd.append("--debug")
    
    # Run the test process
    print(f"Running end-to-end tests with command: {' '.join(test_cmd)}")
    test_process = subprocess.Popen(
        test_cmd,
        stdout=subprocess.PIPE if not args.debug else None,
        stderr=subprocess.PIPE if not args.debug else None
    )
    
    # Wait for test to complete
    test_exit_code = test_process.wait()
    
    if test_exit_code != 0:
        print(f"Warning: Test process exited with code {test_exit_code}")
    
    return test_exit_code == 0

async def generate_visualizations(args):
    """Generate visualizations for the test results."""
    # Construct the command for visualization
    viz_cmd = [
        "python", "-m", "duckdb_api.distributed_testing.tests.e2e_visualization",
        "--report-dir", args.report_dir,
        "--output-dir", args.visualization_dir,
        "--theme", args.theme,
        "--visualization-types", "all"
    ]
    
    # Add options based on args
    if args.generate_standalone:
        viz_cmd.append("--generate-standalone")
    if args.open_browser:
        viz_cmd.append("--open-browser")
    if args.debug:
        viz_cmd.append("--debug")
    
    # Run the visualization process
    print(f"Generating visualizations with command: {' '.join(viz_cmd)}")
    viz_process = subprocess.Popen(
        viz_cmd,
        stdout=subprocess.PIPE if not args.debug else None,
        stderr=subprocess.PIPE if not args.debug else None
    )
    
    # Wait for visualization to complete
    viz_exit_code = viz_process.wait()
    
    if viz_exit_code != 0:
        print(f"Warning: Visualization process exited with code {viz_exit_code}")
    
    return viz_exit_code == 0

async def integrate_with_dashboard(args):
    """Integrate visualizations with the monitoring dashboard."""
    if args.skip_dashboard_integration:
        print("Skipping dashboard integration (--skip-dashboard-integration specified)")
        return True
    
    # Find the latest test report
    report_files = list(Path(args.report_dir).glob("*_results.json"))
    if not report_files:
        print(f"No test reports found in {args.report_dir}")
        return False
    
    # Sort by modification time, most recent first
    report_files.sort(key=os.path.getmtime, reverse=True)
    latest_report = report_files[0]
    
    # Extract test ID from filename
    test_id = latest_report.name.replace("_results.json", "")
    
    # Load report data
    with open(latest_report, 'r') as f:
        report_data = json.load(f)
    
    # Find visualization files
    summary_file = Path(args.visualization_dir) / f"{test_id}_summary.html"
    component_file = Path(args.visualization_dir) / f"{test_id}_component_status.html"
    timing_file = Path(args.visualization_dir) / f"{test_id}_timing.html"
    failures_file = Path(args.visualization_dir) / f"{test_id}_failures.html"
    
    # Check if all visualization files exist
    if not all(f.exists() for f in [summary_file, component_file, timing_file, failures_file]):
        print("One or more visualization files are missing")
        return False
    
    # Load visualization content
    with open(summary_file, 'r') as f:
        summary_content = f.read()
    
    with open(component_file, 'r') as f:
        component_content = f.read()
    
    with open(timing_file, 'r') as f:
        timing_content = f.read()
    
    with open(failures_file, 'r') as f:
        failures_content = f.read()
    
    # Import required modules
    try:
        import aiohttp
    except ImportError:
        print("Error: aiohttp module is required for dashboard integration")
        return False
    
    # Prepare data to send to dashboard
    data = {
        'test_id': test_id,
        'visualizations': {
            'summary': summary_content,
            'component': component_content,
            'timing': timing_content,
            'failures': failures_content
        }
    }
    
    # Send to dashboard API
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{args.dashboard_url}/api/e2e-test-results",
                json=data
            ) as response:
                if response.status == 200:
                    print(f"Successfully integrated visualizations with dashboard at {args.dashboard_url}")
                    
                    # Open dashboard in browser if requested
                    if args.open_browser:
                        dashboard_page = f"{args.dashboard_url}/e2e-test-results/{test_id}"
                        print(f"Opening dashboard page: {dashboard_page}")
                        webbrowser.open(dashboard_page)
                    
                    return True
                else:
                    print(f"Failed to integrate with dashboard: HTTP {response.status} - {await response.text()}")
                    return False
    except Exception as e:
        print(f"Error integrating with dashboard: {e}")
        return False

async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run End-to-End Tests with Dashboard Integration")
    
    # Test options
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--fault-tolerance", action="store_true", help="Include fault tolerance tests")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML report of test results")
    
    # Visualization options
    parser.add_argument("--report-dir", default="./e2e_test_reports", help="Directory for test reports")
    parser.add_argument("--visualization-dir", default="./e2e_visualizations", help="Directory for visualizations")
    parser.add_argument("--theme", choices=["light", "dark"], default="dark", help="Visualization theme")
    parser.add_argument("--generate-standalone", action="store_true", help="Generate standalone HTML visualizations")
    
    # Dashboard options
    parser.add_argument("--dashboard-url", default="http://localhost:8082", help="URL of monitoring dashboard")
    parser.add_argument("--skip-dashboard-integration", action="store_true", help="Skip integration with monitoring dashboard")
    parser.add_argument("--open-browser", action="store_true", help="Open visualization in browser when done")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Create directories if they don't exist
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)
    
    # Step 1: Run the test
    print("\n=== Step 1: Running End-to-End Tests ===\n")
    test_success = await run_test(args)
    
    if not test_success:
        print("\nWarning: Tests did not complete successfully. Continuing with visualization...")
    
    # Step 2: Generate visualizations
    print("\n=== Step 2: Generating Visualizations ===\n")
    viz_success = await generate_visualizations(args)
    
    if not viz_success:
        print("Error: Failed to generate visualizations")
        return 1
    
    # Step 3: Integrate with dashboard
    print("\n=== Step 3: Integrating with Dashboard ===\n")
    integration_success = await integrate_with_dashboard(args)
    
    # Print summary
    print("\n=== Summary ===\n")
    print(f"Test Execution: {'Success' if test_success else 'Failed'}")
    print(f"Visualization: {'Success' if viz_success else 'Failed'}")
    print(f"Dashboard Integration: {'Success' if integration_success else 'Failed'}")
    
    # Return exit code
    return 0 if test_success and viz_success else 1

if __name__ == "__main__":
    sys.exit(anyio.run(main()))