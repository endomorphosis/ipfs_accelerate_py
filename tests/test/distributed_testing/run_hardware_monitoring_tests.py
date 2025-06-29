#!/usr/bin/env python3
"""
Test Runner for Hardware Monitoring System

This script runs the hardware monitoring test suite, which includes:
1. Tests for the hardware utilization monitor
2. Tests for the coordinator hardware monitoring integration
3. End-to-end tests for the entire hardware monitoring system

Usage:
    python run_hardware_monitoring_tests.py [options]

Options:
    --verbose: Display verbose test output
    --run-long-tests: Run long-running tests (including end-to-end demo)
    --db-path: Path to test database (default: temporary file)
    --html-report: Generate HTML test report at specified path
"""

import os
import sys
import tempfile
import unittest
import argparse
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_monitoring_tests")


def generate_html_report(result, file_path, test_suite):
    """
    Generate a simple HTML report of test results.
    
    Args:
        result: TestResult object
        file_path: Path to output file
        test_suite: TestSuite object
    """
    try:
        # Create simple HTML report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hardware Monitoring Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .summary {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
        .success {{
            color: #2ecc71;
        }}
        .failure {{
            color: #e74c3c;
        }}
        .error {{
            color: #e74c3c;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h1>Hardware Monitoring Test Report</h1>
    <div class="summary">
        <h2>Test Summary</h2>
        <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Tests Run: {result.testsRun}</p>
        <p>Failures: <span class="{'success' if len(result.failures) == 0 else 'failure'}">{len(result.failures)}</span></p>
        <p>Errors: <span class="{'success' if len(result.errors) == 0 else 'error'}">{len(result.errors)}</span></p>
        <p>Skipped: {len(getattr(result, 'skipped', []))}</p>
        <p>Status: <span class="{'success' if result.wasSuccessful() else 'failure'}">{('Passed' if result.wasSuccessful() else 'Failed')}</span></p>
    </div>
    """
    
        # Add basic test information without trying to inspect the test objects
        html += """
    <div>
        <h2>Tests</h2>
        <table>
            <tr>
                <th>Test Class</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>TestHardwareUtilizationMonitor</td>
                <td>Tests for the hardware utilization monitor component</td>
            </tr>
            <tr>
                <td>TestCoordinatorHardwareMonitoringIntegration</td>
                <td>Tests for the coordinator hardware monitoring integration component</td>
            </tr>
        </table>
    </div>
    """
    
        # Write to file
        with open(file_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated HTML report at {file_path}")
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        # Fall back to a very basic report
        with open(file_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head><title>Test Report</title></head>
<body>
<h1>Hardware Monitoring Test Report</h1>
<p>Tests Run: {result.testsRun}</p>
<p>Status: {'Passed' if result.wasSuccessful() else 'Failed'}</p>
</body>
</html>
            """)
        logger.info(f"Generated simplified HTML report at {file_path}")

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests(args):
    """
    Run the hardware monitoring test suite.
    
    Args:
        args: Command-line arguments
    """
    # Set environment variables for tests
    if args.run_long_tests:
        os.environ['RUN_LONG_TESTS'] = '1'
    
    if args.db_path:
        os.environ['TEST_DB_PATH'] = args.db_path
    
    # Discover and load tests
    logger.info("Discovering tests...")
    test_loader = unittest.TestLoader()
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    
    # Only load our hardware utilization monitor tests
    test_suite = test_loader.discover(test_dir, pattern="test_hardware_utilization_monitor.py")
    
    # Run tests
    logger.info("Running tests...")
    
    # Run tests with text runner
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(test_suite)
    
    # Generate HTML report if requested
    if args.html_report:
        generate_html_report(result, args.html_report, test_suite)
    
    # Log results
    logger.info(f"Tests completed: {result.testsRun} run, {len(result.errors)} errors, {len(result.failures)} failures")
    
    # Return exit code based on result
    return 0 if result.wasSuccessful() else 1


def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Hardware Monitoring Test Runner")
    parser.add_argument("--verbose", action="store_true", help="Display verbose test output")
    parser.add_argument("--run-long-tests", action="store_true", help="Run long-running tests")
    parser.add_argument("--db-path", help="Path to test database (default: temporary file)")
    parser.add_argument("--html-report", help="Generate HTML test report at specified path")
    
    args = parser.parse_args()
    
    # Print test runner information
    print("=" * 80)
    print("Hardware Monitoring Test Runner")
    print("=" * 80)
    print(f"Verbose: {args.verbose}")
    print(f"Run long tests: {args.run_long_tests}")
    print(f"Database path: {args.db_path or 'Using temporary file'}")
    print(f"HTML report: {args.html_report or 'Not enabled'}")
    print("=" * 80)
    
    try:
        # Run tests
        exit_code = run_tests(args)
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()