#!/usr/bin/env python3
"""
End-to-End Integration Test Runner for the Distributed Testing Framework.

This script runs the end-to-end integration tests for the Distributed Testing Framework,
setting up all the necessary components, running the tests, and collecting the results.

Usage:
    python run_e2e_integration_test.py [options]

Options:
    --output-dir DIR       Directory to save test results (default: test_output/e2e)
    --temp-dir DIR         Temporary directory for test files (default: /tmp/dt_e2e_test)
    --test-filter PATTERN  Pattern to filter tests (e.g., TestE2E*)
    --timeout SECONDS      Test timeout in seconds (default: 1800)
    --verbose              Enable verbose output
    --no-cleanup           Don't clean up temporary files after tests
    --ci                   Run in CI mode (adjust settings for CI environment)
"""

import os
import sys
import time
import json
import shutil
import logging
import argparse
import unittest
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('e2e_test_runner')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run E2E integration tests for the Distributed Testing Framework.')
    parser.add_argument('--output-dir', type=str, default='test_output/e2e',
                        help='Directory to save test results (default: test_output/e2e)')
    parser.add_argument('--temp-dir', type=str, default='/tmp/dt_e2e_test',
                        help='Temporary directory for test files (default: /tmp/dt_e2e_test)')
    parser.add_argument('--test-filter', type=str, default='Test*',
                        help='Pattern to filter tests (e.g., TestE2E*)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Test timeout in seconds (default: 1800)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Don\'t clean up temporary files after tests')
    parser.add_argument('--ci', action='store_true',
                        help='Run in CI mode (adjust settings for CI environment)')
    return parser.parse_args()

def setup_test_environment(temp_dir, verbose=False):
    """Set up the test environment."""
    logger.info(f"Setting up test environment in {temp_dir}")
    
    # Create temp directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'conf'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'data'), exist_ok=True)
    
    # Create test configurations
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    if os.path.exists(config_dir):
        for config_file in os.listdir(config_dir):
            src_path = os.path.join(config_dir, config_file)
            dst_path = os.path.join(temp_dir, 'conf', config_file)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                if verbose:
                    logger.info(f"Copied config file: {config_file}")
    
    # Create E2E test specific config
    e2e_test_config = {
        "coordinator": {
            "host": "localhost",
            "port": 8080,
            "use_ssl": False,
            "api_key": "test_e2e_key",
            "test_mode": True
        },
        "dynamic_resource_manager": {
            "enabled": True,
            "min_workers": 2,
            "max_workers": 10,
            "scaling_strategy": "queue_length",
            "scale_up_threshold": 5,
            "scale_down_threshold": 2,
            "cooldown_period": 30,
            "test_mode": True
        },
        "performance_trend_analyzer": {
            "enabled": True,
            "metrics_collection_interval": 10,
            "analysis_interval": 30,
            "anomaly_detection_sensitivity": 2.0,
            "test_mode": True
        },
        "test": {
            "timeout": 1800,
            "worker_startup_time": 5,
            "task_processing_time": 2,
            "record_resource_metrics": True,
            "record_component_interactions": True
        }
    }
    
    with open(os.path.join(temp_dir, 'conf', 'e2e_test_config.json'), 'w') as f:
        json.dump(e2e_test_config, f, indent=2)
    
    return temp_dir

def discover_tests(test_filter):
    """Discover tests matching the filter pattern."""
    logger.info(f"Discovering tests matching pattern: {test_filter}")
    
    # Get the path to the tests directory
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Discover tests
    loader = unittest.TestLoader()
    discovered_tests = loader.discover(tests_dir, pattern=f"{test_filter}.py")
    
    return discovered_tests

def run_tests(test_suite, output_dir, temp_dir, verbose=False, timeout=1800):
    """Run the test suite and collect results."""
    logger.info(f"Running tests with timeout={timeout}s")
    
    # Set environment variables for tests
    os.environ['DT_TEST_TEMP_DIR'] = temp_dir
    os.environ['DT_TEST_TIMEOUT'] = str(timeout)
    os.environ['DT_TEST_VERBOSE'] = str(int(verbose))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create result collector
    result_file = os.path.join(output_dir, 'results.json')
    
    class JSONTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
        
        def addSuccess(self, test):
            super().addSuccess(test)
            self._record_test_result(test, 'pass')
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self._record_test_result(test, 'fail', error=str(err))
        
        def addError(self, test, err):
            super().addError(test, err)
            self._record_test_result(test, 'error', error=str(err))
        
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self._record_test_result(test, 'skip', reason=reason)
        
        def _record_test_result(self, test, status, error=None, reason=None):
            result = {
                'test_name': test.id(),
                'test_type': 'e2e',
                'status': status,
                'execution_time': getattr(test, 'execution_time', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add error information if available
            if error:
                result['error'] = str(error)
            
            # Add skip reason if available
            if reason:
                result['skip_reason'] = reason
            
            # Add test-specific data if available
            if hasattr(test, 'metrics'):
                result['metrics'] = test.metrics
            
            if hasattr(test, 'resource_metrics'):
                result['resource_metrics'] = test.resource_metrics
            
            if hasattr(test, 'component_interactions'):
                result['component_interactions'] = test.component_interactions
            
            if hasattr(test, 'scaling_events'):
                result['scaling_events'] = test.scaling_events
            
            self.test_results.append(result)
    
    # Create a custom test runner
    class CustomTestRunner(unittest.TextTestRunner):
        def run(self, test):
            result = JSONTestResult(self.stream, self.descriptions, self.verbosity)
            test.run(result)
            return result
    
    # Run tests
    start_time = time.time()
    runner = CustomTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Calculate overall metrics
    total_time = end_time - start_time
    success_count = len(result.successes)
    failure_count = len(result.failures)
    error_count = len(result.errors)
    skip_count = len(result.skips)
    total_count = success_count + failure_count + error_count + skip_count
    
    # Save results to JSON file
    logger.info(f"Saving test results to {result_file}")
    with open(result_file, 'w') as f:
        json.dump(result.test_results, f, indent=2)
    
    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'duration': total_time,
        'total_tests': total_count,
        'passed': success_count,
        'failed': failure_count,
        'errors': error_count,
        'skipped': skip_count,
        'pass_rate': success_count / total_count if total_count > 0 else 0
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate HTML report
    html_summary = f"""<!DOCTYPE html>
<html>
<head>
    <title>E2E Integration Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .skip {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>E2E Integration Test Results</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Duration: {total_time:.2f} seconds</p>
        <p>Total Tests: {total_count}</p>
        <p class="pass">Passed: {success_count} ({(success_count / total_count * 100):.1f}%)</p>
        <p class="fail">Failed: {failure_count}</p>
        <p class="fail">Errors: {error_count}</p>
        <p class="skip">Skipped: {skip_count}</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Execution Time</th>
            <th>Details</th>
        </tr>
"""
    
    for test_result in result.test_results:
        status_class = {
            'pass': 'pass',
            'fail': 'fail',
            'error': 'fail',
            'skip': 'skip'
        }.get(test_result['status'], '')
        
        details = ""
        if 'error' in test_result:
            details = test_result['error']
        elif 'skip_reason' in test_result:
            details = test_result['skip_reason']
        
        html_summary += f"""
        <tr>
            <td>{test_result['test_name']}</td>
            <td class="{status_class}">{test_result['status'].upper()}</td>
            <td>{test_result['execution_time']:.2f}s</td>
            <td>{details}</td>
        </tr>"""
    
    html_summary += """
    </table>
</body>
</html>
"""
    
    html_report_file = os.path.join(output_dir, 'report.html')
    with open(html_report_file, 'w') as f:
        f.write(html_summary)
    
    # Return success/failure
    logger.info(f"Tests completed: {success_count} passed, {failure_count} failed, {error_count} errors, {skip_count} skipped")
    return failure_count == 0 and error_count == 0

def cleanup_test_environment(temp_dir):
    """Clean up the test environment."""
    logger.info(f"Cleaning up test environment in {temp_dir}")
    
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Error cleaning up test environment: {e}")

def main():
    """Main execution function."""
    args = parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    temp_dir = args.temp_dir
    
    try:
        # Set up test environment
        temp_dir = setup_test_environment(temp_dir, args.verbose)
        
        # Discover tests
        test_suite = discover_tests(args.test_filter)
        
        # Run tests
        success = run_tests(test_suite, output_dir, temp_dir, args.verbose, args.timeout)
        
        # Clean up if requested
        if not args.no_cleanup:
            cleanup_test_environment(temp_dir)
        
        # Generate visualizations if not in CI mode
        if not args.ci:
            try:
                # Check if visualization script exists
                vis_script = os.path.join(script_dir, 'visualize_test_results.py')
                if os.path.exists(vis_script):
                    logger.info("Generating visualizations...")
                    vis_output_dir = os.path.join(output_dir, 'visualizations')
                    os.makedirs(vis_output_dir, exist_ok=True)
                    
                    subprocess.run([
                        sys.executable, 
                        vis_script, 
                        '--input-dir', output_dir,
                        '--output-dir', vis_output_dir
                    ], check=True)
                    
                    logger.info(f"Visualizations saved to {vis_output_dir}")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
        
        # Return appropriate exit code
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        return 1
    
    finally:
        # Ensure we clean up even if there was an error
        if not args.no_cleanup and os.path.exists(temp_dir):
            cleanup_test_environment(temp_dir)

if __name__ == "__main__":
    sys.exit(main())