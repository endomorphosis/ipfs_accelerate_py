#!/usr/bin/env python3
"""
Run End-to-End Testing Suite for Distributed Testing Framework

This script runs comprehensive end-to-end tests for the Distributed Testing Framework
with various configurations to validate the complete system.

Usage:
    python run_e2e_tests.py [options]

Options:
    --quick               Run a quick test with minimal duration
    --comprehensive       Run comprehensive tests with all configurations
    --fault-tolerance     Include tests for fault tolerance
    --generate-report     Generate HTML report of test results
    --skip-long-tests     Skip longer duration tests
    --help                Show this help message
"""

import argparse
import anyio
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# Test configurations
QUICK_TEST = {
    "name": "Quick Test",
    "workers": 3,
    "duration": 30,
    "hardware": "cpu,gpu",
    "failures": False
}

BASIC_TESTS = [
    {
        "name": "Basic CPU Test",
        "workers": 3,
        "duration": 60,
        "hardware": "cpu",
        "failures": False
    },
    {
        "name": "Basic GPU Test",
        "workers": 3,
        "duration": 60,
        "hardware": "gpu",
        "failures": False
    },
    {
        "name": "WebGPU/WebNN Test",
        "workers": 3,
        "duration": 60,
        "hardware": "webgpu,webnn",
        "failures": False
    }
]

COMPREHENSIVE_TESTS = [
    {
        "name": "Comprehensive All Hardware",
        "workers": 5,
        "duration": 120,
        "hardware": "all",
        "failures": False
    },
    {
        "name": "High Load Test",
        "workers": 10,
        "duration": 180,
        "hardware": "all",
        "failures": False
    }
]

FAULT_TOLERANCE_TESTS = [
    {
        "name": "Fault Tolerance Basic",
        "workers": 5,
        "duration": 90,
        "hardware": "all",
        "failures": True
    },
    {
        "name": "Fault Tolerance High Load",
        "workers": 10,
        "duration": 180,
        "hardware": "all",
        "failures": True
    }
]

async def run_test(config, base_port=8080, debug=False):
    """Run a single end-to-end test with the provided configuration."""
    print(f"\n{'='*80}")
    print(f"Running Test: {config['name']}")
    print(f"Configuration: {config['workers']} workers, {config['duration']}s duration")
    print(f"Hardware: {config['hardware']}")
    print(f"Include Failures: {config['failures']}")
    print(f"{'-'*80}")
    
    # Calculate ports to avoid conflicts when running multiple tests
    dashboard_port = base_port
    coordinator_port = base_port + 1
    result_aggregator_port = base_port + 2
    
    # Build command
    cmd = [
        "python", "-m", "duckdb_api.distributed_testing.tests.test_end_to_end_framework",
        "--workers", str(config["workers"]),
        "--test-duration", str(config["duration"]),
        "--hardware-profiles", config["hardware"],
        "--dashboard-port", str(dashboard_port),
        "--coordinator-port", str(coordinator_port),
        "--result-aggregator-port", str(result_aggregator_port),
        "--report-dir", "./e2e_test_reports"
    ]
    
    if config["failures"]:
        cmd.append("--include-failures")
    
    if debug:
        cmd.append("--debug")
    
    # Run the test
    start_time = datetime.datetime.now()
    print(f"Starting test at {start_time.strftime('%H:%M:%S')}")
    
    process = subprocess.Popen(cmd)
    exit_code = process.wait()
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"Test completed at {end_time.strftime('%H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Exit code: {exit_code}")
    
    result = {
        "name": config["name"],
        "success": exit_code == 0,
        "exit_code": exit_code,
        "duration": duration,
        "config": config
    }
    
    print(f"Test result: {'SUCCESS' if result['success'] else 'FAILURE'}")
    print(f"{'='*80}\n")
    
    return result

async def run_test_suite(test_configs, base_port=8080, debug=False):
    """Run a suite of tests and return the results."""
    results = []
    
    for i, config in enumerate(test_configs):
        # Use different ports for each test to avoid conflicts
        port = base_port + (i * 10)
        result = await run_test(config, base_port=port, debug=debug)
        results.append(result)
    
    return results

def generate_report(results, output_file="e2e_test_report.html"):
    """Generate an HTML report from test results."""
    now = datetime.datetime.now()
    
    # Count successes and failures
    total = len(results)
    succeeded = sum(1 for r in results if r["success"])
    failed = total - succeeded
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Distributed Testing Framework E2E Test Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .summary {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            .summary-counts {{
                display: flex;
                gap: 20px;
                margin-top: 10px;
            }}
            .count-box {{
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }}
            .total {{
                background-color: #e9ecef;
            }}
            .succeeded {{
                background-color: #d4edda;
                color: #155724;
            }}
            .failed {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .success {{
                color: #155724;
                font-weight: bold;
            }}
            .failure {{
                color: #721c24;
                font-weight: bold;
            }}
            .test-details {{
                margin-top: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 10px;
            }}
            .test-meta {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
            }}
            .test-meta-item {{
                background-color: #e9ecef;
                padding: 5px 10px;
                border-radius: 3px;
                margin-right: 10px;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Distributed Testing Framework E2E Test Report</h1>
            <p>Generated at: {now.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="summary-counts">
                    <div class="count-box total">Total: {total}</div>
                    <div class="count-box succeeded">Succeeded: {succeeded}</div>
                    <div class="count-box failed">Failed: {failed}</div>
                </div>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Result</th>
                        <th>Duration (s)</th>
                        <th>Workers</th>
                        <th>Hardware</th>
                        <th>Failures Included</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add table rows
    for result in results:
        status_class = "success" if result["success"] else "failure"
        status_text = "SUCCESS" if result["success"] else "FAILURE"
        
        html += f"""
                    <tr>
                        <td>{result["name"]}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{result["duration"]:.1f}</td>
                        <td>{result["config"]["workers"]}</td>
                        <td>{result["config"]["hardware"]}</td>
                        <td>{"Yes" if result["config"]["failures"] else "No"}</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
            
            <h2>Detailed Results</h2>
    """
    
    # Add detailed sections
    for result in results:
        status_class = "success" if result["success"] else "failure"
        status_text = "SUCCESS" if result["success"] else "FAILURE"
        
        html += f"""
            <div class="test-details">
                <h3>{result["name"]}</h3>
                <p class="{status_class}">Result: {status_text}</p>
                
                <div class="test-meta">
                    <div class="test-meta-item">Duration: {result["duration"]:.1f} seconds</div>
                    <div class="test-meta-item">Exit Code: {result["exit_code"]}</div>
                    <div class="test-meta-item">Workers: {result["config"]["workers"]}</div>
                    <div class="test-meta-item">Test Duration: {result["config"]["duration"]} seconds</div>
                    <div class="test-meta-item">Hardware: {result["config"]["hardware"]}</div>
                    <div class="test-meta-item">Failures Included: {"Yes" if result["config"]["failures"] else "No"}</div>
                </div>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, "w") as f:
        f.write(html)
    
    print(f"Generated HTML report: {output_file}")
    return output_file

async def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run E2E Tests for Distributed Testing Framework")
    parser.add_argument("--quick", action="store_true", help="Run a quick test with minimal duration")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive tests with all configurations")
    parser.add_argument("--fault-tolerance", action="store_true", help="Include tests for fault tolerance")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML report of test results")
    parser.add_argument("--skip-long-tests", action="store_true", help="Skip longer duration tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    all_results = []
    
    # Determine which tests to run
    if args.quick:
        print("Running quick test only...")
        results = await run_test_suite([QUICK_TEST], debug=args.debug)
        all_results.extend(results)
    else:
        # Run basic tests
        print("Running basic tests...")
        results = await run_test_suite(BASIC_TESTS, debug=args.debug)
        all_results.extend(results)
        
        # Run comprehensive tests if requested
        if args.comprehensive and not args.skip_long_tests:
            print("Running comprehensive tests...")
            results = await run_test_suite(COMPREHENSIVE_TESTS, base_port=8100, debug=args.debug)
            all_results.extend(results)
        
        # Run fault tolerance tests if requested
        if args.fault_tolerance:
            print("Running fault tolerance tests...")
            if args.skip_long_tests:
                # Only run the first, shorter test
                results = await run_test_suite([FAULT_TOLERANCE_TESTS[0]], base_port=8200, debug=args.debug)
            else:
                results = await run_test_suite(FAULT_TOLERANCE_TESTS, base_port=8200, debug=args.debug)
            all_results.extend(results)
    
    # Print summary
    total = len(all_results)
    succeeded = sum(1 for r in all_results if r["success"])
    failed = total - succeeded
    
    print("\n" + "="*80)
    print("Test Suite Summary")
    print("="*80)
    print(f"Total Tests: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print("="*80)
    
    # Generate HTML report if requested
    if args.generate_report:
        report_file = generate_report(all_results)
        print(f"HTML report generated: {report_file}")
    
    # Return exit code based on test results
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = anyio.run(main())
    sys.exit(exit_code)