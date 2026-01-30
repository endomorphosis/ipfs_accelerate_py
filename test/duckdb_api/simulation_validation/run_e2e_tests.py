#!/usr/bin/env python3
"""
Runner script for the end-to-end tests of the Simulation Accuracy and Validation Framework.

This script provides a comprehensive testing framework for the Simulation Validation Framework,
running tests across all components and workflows. It is designed to work both in development
environments and CI/CD pipelines, with various options for test selection, reporting, and
visualization generation.

Features:
- Selectively run specific test types (database, connector, visualization, etc.)
- Run comprehensive end-to-end tests with full component integration
- Generate test reports in various formats (console, JSON, HTML, JUnit XML)
- Generate example visualizations for documentation purposes
- Run performance tests and generate detailed performance reports
- Support for CI/CD integration with GitHub Actions
- Generate test data for demonstration and validation
- Skip long-running tests for faster feedback during development
- Support for parallel test execution to speed up large test suites
- Integration with the monitoring dashboard for visual feedback
- Code coverage reporting for quality assurance

Usage examples:
- Run all tests:
  python run_e2e_tests.py
  
- Run specific test types:
  python run_e2e_tests.py --run-db --run-connector
  
- Run comprehensive tests with HTML report:
  python run_e2e_tests.py --run-comprehensive --html-report --output-dir ./reports
  
- Generate test data and visualizations without running tests:
  python run_e2e_tests.py --generate-test-data --generate-examples
  
- Run in CI mode with JUnit XML reporting:
  python run_e2e_tests.py --ci-mode --junit-xml --coverage

- Generate performance report:
  python run_e2e_tests.py --run-comprehensive --performance-report
"""

import os
import sys
import unittest
import time
import json
import datetime
import argparse
import platform
import socket
import getpass
import subprocess
import shutil
import multiprocessing
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import test modules
from data.duckdb.simulation_validation.test_db_integration import TestSimulationValidationDBIntegration
from data.duckdb.simulation_validation.test_visualization_db_connector import TestValidationVisualizerDBConnector
from data.duckdb.simulation_validation.test_e2e_visualization_db_integration import TestE2EVisualizationDBIntegration
from data.duckdb.simulation_validation.test.test_comprehensive_e2e import TestComprehensiveEndToEnd

# Set up argument parser
parser = argparse.ArgumentParser(description="Run end-to-end tests for the Simulation Validation Framework")
parser.add_argument("--output-dir", type=str, help="Directory to store test reports", default=None)
parser.add_argument("--html-report", action="store_true", help="Generate HTML report of test results")
parser.add_argument("--run-db", action="store_true", help="Run database integration tests only")
parser.add_argument("--run-connector", action="store_true", help="Run visualization connector tests only")
parser.add_argument("--run-e2e", action="store_true", help="Run standard end-to-end tests only")
parser.add_argument("--run-comprehensive", action="store_true", help="Run comprehensive end-to-end tests only")
parser.add_argument("--run-validation", action="store_true", help="Run validation component tests")
parser.add_argument("--run-calibration", action="store_true", help="Run calibration component tests")
parser.add_argument("--run-drift", action="store_true", help="Run drift detection component tests")
parser.add_argument("--run-dashboard", action="store_true", help="Run dashboard integration tests")
parser.add_argument("--run-visualization", action="store_true", help="Run visualization component tests")
parser.add_argument("--generate-examples", action="store_true", help="Generate example visualizations")
parser.add_argument("--skip-long-tests", action="store_true", help="Skip long-running tests")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
parser.add_argument("--performance-report", action="store_true", help="Generate a performance report")
parser.add_argument("--generate-test-data", action="store_true", help="Generate test data only, without running tests")
parser.add_argument("--ci-mode", action="store_true", help="Run in CI mode with GitHub Actions compatible output")
parser.add_argument("--parallel", action="store_true", help="Run tests in parallel for faster execution")
parser.add_argument("--junit-xml", action="store_true", help="Generate JUnit XML report for CI integration")
parser.add_argument("--coverage", action="store_true", help="Generate code coverage report")
parser.add_argument("--dashboard-integration", action="store_true", help="Test integration with the monitoring dashboard")
parser.add_argument("--system-info", action="store_true", help="Include system information in reports")
args = parser.parse_args()

# Create output directory if specified
if args.output_dir:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
else:
    # Use a temporary directory if not specified
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir)

# Setup additional directories
data_dir = output_dir / "test_data"
report_dir = output_dir / "reports"
coverage_dir = output_dir / "coverage"
visualization_dir = output_dir / "visualizations"

# Create additional directories
data_dir.mkdir(exist_ok=True)
report_dir.mkdir(exist_ok=True)
coverage_dir.mkdir(exist_ok=True)
visualization_dir.mkdir(exist_ok=True)

# Setup HTML test report if requested
if args.html_report:
    try:
        from unittest import HTMLTestRunner
    except ImportError:
        print("HTMLTestRunner not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "html-testRunner"])
        from unittest import HTMLTestRunner

# Setup JUnit XML reporter if requested
if args.junit_xml:
    try:
        import xmlrunner
    except ImportError:
        print("xmlrunner not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unittest-xml-reporting"])
        import xmlrunner

# Setup coverage if requested
if args.coverage:
    try:
        import coverage
    except ImportError:
        print("coverage not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
        import coverage
    
    # Start coverage tracking
    cov = coverage.Coverage(source=["duckdb_api.simulation_validation"])
    cov.start()

# Initialize system information if requested
if args.system_info:
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.datetime.now().isoformat()
    }
else:
    system_info = None


class TestSummary:
    """Class to track test execution results and times."""
    
    def __init__(self):
        """Initialize test summary."""
        self.start_time = time.time()
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_errors = 0
        self.test_results = []
        
    def add_result(self, test_name, result, execution_time):
        """Add a test result to the summary."""
        status = "PASS" if result else "FAIL"
        self.test_results.append({
            "test_name": test_name,
            "status": status,
            "execution_time": execution_time
        })
        self.tests_run += 1
        if result:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
    
    def add_error(self, test_name, error_message):
        """Add a test error to the summary."""
        self.test_results.append({
            "test_name": test_name,
            "status": "ERROR",
            "error_message": error_message
        })
        self.tests_run += 1
        self.tests_errors += 1
    
    def get_total_time(self):
        """Get the total execution time."""
        return time.time() - self.start_time
    
    def generate_summary(self):
        """Generate a text summary of test results."""
        summary = []
        summary.append("=" * 80)
        summary.append("TEST EXECUTION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Tests Run:    {self.tests_run}")
        summary.append(f"Tests Passed: {self.tests_passed}")
        summary.append(f"Tests Failed: {self.tests_failed}")
        summary.append(f"Test Errors:  {self.tests_errors}")
        summary.append(f"Total Time:   {self.get_total_time():.2f} seconds")
        summary.append("-" * 80)
        summary.append("DETAILED RESULTS:")
        summary.append("-" * 80)
        
        for i, result in enumerate(self.test_results, 1):
            status = result["status"]
            status_color = {
                "PASS": "\033[92m",  # Green
                "FAIL": "\033[91m",  # Red
                "ERROR": "\033[93m"  # Yellow
            }.get(status, "")
            reset_color = "\033[0m"
            
            if "execution_time" in result:
                summary.append(f"{i}. {status_color}{status}{reset_color} - {result['test_name']} ({result['execution_time']:.2f}s)")
            else:
                summary.append(f"{i}. {status_color}{status}{reset_color} - {result['test_name']}")
                if "error_message" in result:
                    summary.append(f"   Error: {result['error_message']}")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def generate_json_report(self, output_path):
        """Generate a JSON report of test results."""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_errors": self.tests_errors,
            "total_time": self.get_total_time(),
            "results": self.test_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def get_pass_percentage(self):
        """Get the percentage of tests that passed."""
        if self.tests_run == 0:
            return 0
        return (self.tests_passed / self.tests_run) * 100


def run_test_suite(test_suite, summary):
    """Run a test suite and update the summary."""
    for test in test_suite:
        # Extract test name
        test_name = test.id().split('.')[-1]
        full_test_name = test.id()
        
        if args.verbose:
            print(f"Running test: {full_test_name}")
        
        # Run the test
        start_time = time.time()
        result = unittest.TestResult()
        test.run(result)
        execution_time = time.time() - start_time
        
        # Check for errors or failures
        if result.errors:
            error_message = result.errors[0][1]
            summary.add_error(full_test_name, error_message)
            if args.verbose:
                print(f"  Error: {error_message}")
        elif result.failures:
            failure_message = result.failures[0][1]
            summary.add_result(full_test_name, False, execution_time)
            if args.verbose:
                print(f"  Failed: {failure_message}")
        else:
            summary.add_result(full_test_name, True, execution_time)
            if args.verbose:
                print(f"  Passed in {execution_time:.2f}s")


def run_tests():
    """Run all test suites."""
    # Create test summary
    summary = TestSummary()
    
    # Handle special case: generate test data only
    if args.generate_test_data:
        print("\nGenerating test data without running tests...")
        return generate_test_data()
    
    # Import additional test modules as needed
    if args.run_validation or args.run_calibration or args.run_drift or args.run_dashboard or args.run_visualization:
        try:
            # Import additional test modules
            from data.duckdb.simulation_validation.test_validation import TestValidation
            from data.duckdb.simulation_validation.test_validator import TestValidator 
            from data.duckdb.simulation_validation.test_calibration import TestCalibration
            from data.duckdb.simulation_validation.test_drift_detection import TestDriftDetection
            from data.duckdb.simulation_validation.test_visualization import TestVisualization
            from data.duckdb.simulation_validation.test_dashboard_integration import TestDashboardIntegration
        except ImportError as e:
            print(f"Warning: Cannot import all component test modules: {e}")
            print("Some test modules might not be available. Only importing what's available.")
    
    # Determine which test suites to run
    run_all = not any([
        args.run_db, args.run_connector, args.run_e2e, args.run_comprehensive,
        args.run_validation, args.run_calibration, args.run_drift,
        args.run_dashboard, args.run_visualization
    ])
    
    # Collect all suites to run
    all_suites = []
    
    # Run database integration tests
    if run_all or args.run_db:
        print("\nRunning database integration tests...")
        db_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSimulationValidationDBIntegration)
        if args.skip_long_tests:
            # Filter out long-running tests
            filtered_tests = []
            for test in db_suite:
                if not any(long_test in test.id() for long_test in ["test_large_dataset", "test_stress_"]):
                    filtered_tests.append(test)
            db_suite = unittest.TestSuite(filtered_tests)
        
        if args.parallel:
            all_suites.append(db_suite)
        else:
            run_test_suite(db_suite, summary)
    
    # Run visualization connector tests
    if run_all or args.run_connector:
        print("\nRunning visualization connector tests...")
        connector_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestValidationVisualizerDBConnector)
        
        if args.parallel:
            all_suites.append(connector_suite)
        else:
            run_test_suite(connector_suite, summary)
    
    # Run standard end-to-end tests
    if run_all or args.run_e2e:
        print("\nRunning standard end-to-end integration tests...")
        e2e_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestE2EVisualizationDBIntegration)
        
        if args.parallel:
            all_suites.append(e2e_suite)
        else:
            run_test_suite(e2e_suite, summary)
    
    # Run comprehensive end-to-end tests
    if run_all or args.run_comprehensive:
        print("\nRunning comprehensive end-to-end tests...")
        comprehensive_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestComprehensiveEndToEnd)
        if args.skip_long_tests:
            # Filter out long-running tests (performance tests)
            filtered_tests = []
            for test in comprehensive_suite:
                if not any(long_test in test.id() for long_test in ["test_23_performance_with_large_dataset", "test_24_generate_performance_report"]):
                    filtered_tests.append(test)
            comprehensive_suite = unittest.TestSuite(filtered_tests)
        
        if args.parallel:
            all_suites.append(comprehensive_suite)
        else:
            run_test_suite(comprehensive_suite, summary)
    
    # Run validation component tests
    if args.run_validation:
        try:
            print("\nRunning validation component tests...")
            validation_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestValidation)
            validation_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestValidator))
            
            if args.parallel:
                all_suites.append(validation_suite)
            else:
                run_test_suite(validation_suite, summary)
        except (NameError, ImportError) as e:
            print(f"Error running validation tests: {e}")
    
    # Run calibration component tests
    if args.run_calibration:
        try:
            print("\nRunning calibration component tests...")
            calibration_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCalibration)
            
            if args.parallel:
                all_suites.append(calibration_suite)
            else:
                run_test_suite(calibration_suite, summary)
        except (NameError, ImportError) as e:
            print(f"Error running calibration tests: {e}")
    
    # Run drift detection component tests
    if args.run_drift:
        try:
            print("\nRunning drift detection component tests...")
            drift_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDriftDetection)
            
            if args.parallel:
                all_suites.append(drift_suite)
            else:
                run_test_suite(drift_suite, summary)
        except (NameError, ImportError) as e:
            print(f"Error running drift detection tests: {e}")
    
    # Run visualization component tests
    if args.run_visualization:
        try:
            print("\nRunning visualization component tests...")
            visualization_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVisualization)
            
            if args.parallel:
                all_suites.append(visualization_suite)
            else:
                run_test_suite(visualization_suite, summary)
        except (NameError, ImportError) as e:
            print(f"Error running visualization tests: {e}")
    
    # Run dashboard integration tests
    if args.run_dashboard or args.dashboard_integration:
        try:
            print("\nRunning dashboard integration tests...")
            dashboard_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDashboardIntegration)
            
            if args.parallel:
                all_suites.append(dashboard_suite)
            else:
                run_test_suite(dashboard_suite, summary)
        except (NameError, ImportError) as e:
            print(f"Error running dashboard integration tests: {e}")
    
    # If running in parallel, run all collected suites
    if args.parallel and all_suites:
        print(f"\nRunning {len(all_suites)} test suites in parallel...")
        
        combined_suite = unittest.TestSuite(all_suites)
        
        if args.ci_mode:
            print("Running in CI mode with GitHub Actions compatible output...")
            # Simplified CI output format
            for test in combined_suite:
                test_id = test.id()
                print(f"::group::{test_id}")
                result = unittest.TestResult()
                start_time = time.time()
                test.run(result)
                execution_time = time.time() - start_time
                
                if result.errors:
                    print(f"::error::{result.errors[0][1]}")
                    summary.add_error(test_id, result.errors[0][1])
                elif result.failures:
                    print(f"::error::{result.failures[0][1]}")
                    summary.add_result(test_id, False, execution_time)
                else:
                    print(f"Test passed in {execution_time:.2f}s")
                    summary.add_result(test_id, True, execution_time)
                print("::endgroup::")
        else:
            # Use multiprocessing for parallel execution
            pool = multiprocessing.Pool(processes=min(len(all_suites), os.cpu_count() or 1))
            
            def run_suite(suite):
                suite_result = unittest.TestResult()
                start_time = time.time()
                suite.run(suite_result)
                execution_time = time.time() - start_time
                return (suite, suite_result, execution_time)
            
            results = pool.map(run_suite, all_suites)
            pool.close()
            pool.join()
            
            # Process results
            for suite, result, execution_time in results:
                for test in suite:
                    test_id = test.id()
                    if result.errors:
                        for test_case, error in result.errors:
                            if test_case.id() == test_id:
                                summary.add_error(test_id, error)
                                if args.verbose:
                                    print(f"Error in {test_id}: {error}")
                    elif result.failures:
                        for test_case, failure in result.failures:
                            if test_case.id() == test_id:
                                summary.add_result(test_id, False, execution_time)
                                if args.verbose:
                                    print(f"Failure in {test_id}: {failure}")
                    else:
                        summary.add_result(test_id, True, execution_time)
                        if args.verbose:
                            print(f"Passed: {test_id} ({execution_time:.2f}s)")
    
    # Generate example visualizations if requested
    if args.generate_examples:
        print("\nGenerating example visualizations...")
        generate_example_visualizations(output_dir)
    
    # Print summary report
    print("\n" + summary.generate_summary())
    
    # Generate JSON report
    json_report_path = report_dir / "test_report.json"
    summary.generate_json_report(json_report_path)
    print(f"\nJSON report saved to: {json_report_path}")
    
    # Generate HTML report if requested
    if args.html_report:
        html_report_path = report_dir / "test_report.html"
        generate_html_report(html_report_path)
        print(f"HTML report saved to: {html_report_path}")
    
    # Generate JUnit XML report if requested
    if args.junit_xml:
        try:
            xml_report_path = report_dir / "junit-results.xml"
            with open(xml_report_path, 'wb') as f:
                import xmlrunner
                test_suite = unittest.TestSuite()
                for suite in all_suites:
                    test_suite.addTests(suite)
                xmlrunner.XMLTestRunner(output=f).run(test_suite)
            print(f"JUnit XML report saved to: {xml_report_path}")
        except Exception as e:
            print(f"Error generating JUnit XML report: {e}")
    
    # Generate performance report if requested
    if args.performance_report:
        try:
            # Import the performance reporter
            from data.duckdb.simulation_validation.test.test_comprehensive_e2e import TestComprehensiveEndToEnd
            
            # Create an instance of the test class
            test_instance = TestComprehensiveEndToEnd("test_24_generate_performance_report")
            
            # Set up the test class
            test_instance.setUpClass()
            
            # Run the performance report test
            test_instance.test_24_generate_performance_report()
            
            # Generate the full test report
            test_instance.test_25_generate_full_test_report()
            
            # Clean up
            test_instance.tearDownClass()
            
            # Copy reports to the report directory
            for report_file in os.listdir(test_instance.reports_dir):
                src = os.path.join(test_instance.reports_dir, report_file)
                dst = os.path.join(report_dir, report_file)
                shutil.copy2(src, dst)
            
            print(f"\nPerformance reports generated in: {report_dir}")
        except Exception as e:
            print(f"Error generating performance report: {e}")
    
    # Finalize coverage report if applicable
    if args.coverage:
        try:
            print("\nGenerating code coverage report...")
            cov.stop()
            cov.save()
            
            # Generate HTML report
            cov.html_report(directory=str(coverage_dir))
            print(f"HTML coverage report saved to: {coverage_dir}")
            
            # Generate XML report for CI integration
            cov.xml_report(outfile=str(coverage_dir / "coverage.xml"))
            print(f"XML coverage report saved to: {coverage_dir / 'coverage.xml'}")
            
            # Print coverage summary
            total = cov.report()
            print(f"Total coverage: {total:.2f}%")
        except Exception as e:
            print(f"Error generating coverage report: {e}")
    
    # Return success if all tests passed
    return summary.tests_failed == 0 and summary.tests_errors == 0


def generate_html_report(output_path):
    """Generate an HTML report of test results."""
    try:
        with open(output_path, 'wb') as f:
            # Add system info if requested
            description = 'Results of database integration, visualization connector, and end-to-end tests'
            if args.system_info and system_info:
                description += f"\n\nSystem Information:\n"
                description += f"Platform: {system_info['platform']}\n"
                description += f"Python: {system_info['python_version']}\n"
                description += f"Hostname: {system_info['hostname']}\n"
                description += f"CPU Count: {system_info['cpu_count']}\n"
                description += f"Test Date: {system_info['timestamp']}\n"
            
            runner = HTMLTestRunner.HTMLTestRunner(
                stream=f,
                title='Simulation Validation Framework Test Report',
                description=description
            )
            
            # Create test suites
            suites = []
            
            # Include all available test types based on arguments
            if args.run_db or not any([args.run_connector, args.run_e2e, args.run_comprehensive, 
                                     args.run_validation, args.run_calibration, args.run_drift, 
                                     args.run_dashboard, args.run_visualization]):
                suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestSimulationValidationDBIntegration))
            
            if args.run_connector or not any([args.run_db, args.run_e2e, args.run_comprehensive, 
                                           args.run_validation, args.run_calibration, args.run_drift, 
                                           args.run_dashboard, args.run_visualization]):
                suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestValidationVisualizerDBConnector))
            
            if args.run_e2e or not any([args.run_db, args.run_connector, args.run_comprehensive, 
                                      args.run_validation, args.run_calibration, args.run_drift, 
                                      args.run_dashboard, args.run_visualization]):
                suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestE2EVisualizationDBIntegration))
            
            if args.run_comprehensive or not any([args.run_db, args.run_connector, args.run_e2e, 
                                               args.run_validation, args.run_calibration, args.run_drift, 
                                               args.run_dashboard, args.run_visualization]):
                comprehensive_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestComprehensiveEndToEnd)
                if args.skip_long_tests:
                    # Filter out long-running tests
                    filtered_tests = []
                    for test in comprehensive_suite:
                        if not any(long_test in test.id() for long_test in ["test_23_performance_with_large_dataset", "test_24_generate_performance_report"]):
                            filtered_tests.append(test)
                    comprehensive_suite = unittest.TestSuite(filtered_tests)
                suites.append(comprehensive_suite)
            
            # Add component-specific tests if requested
            if args.run_validation:
                try:
                    from data.duckdb.simulation_validation.test_validation import TestValidation
                    from data.duckdb.simulation_validation.test_validator import TestValidator
                    suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestValidation))
                    suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestValidator))
                except (ImportError, NameError):
                    pass
            
            if args.run_calibration:
                try:
                    from data.duckdb.simulation_validation.test_calibration import TestCalibration
                    suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestCalibration))
                except (ImportError, NameError):
                    pass
            
            if args.run_drift:
                try:
                    from data.duckdb.simulation_validation.test_drift_detection import TestDriftDetection
                    suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestDriftDetection))
                except (ImportError, NameError):
                    pass
            
            if args.run_visualization:
                try:
                    from data.duckdb.simulation_validation.test_visualization import TestVisualization
                    suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestVisualization))
                except (ImportError, NameError):
                    pass
            
            if args.run_dashboard or args.dashboard_integration:
                try:
                    from data.duckdb.simulation_validation.test_dashboard_integration import TestDashboardIntegration
                    suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestDashboardIntegration))
                except (ImportError, NameError):
                    pass
            
            # Run the tests with the HTML runner
            runner.run(unittest.TestSuite(suites))
            
            # Create index file for all reports
            create_report_index(report_dir)
            
    except Exception as e:
        print(f"Error generating HTML report: {e}")


def create_report_index(report_dir):
    """Create an index.html file for all reports."""
    index_path = report_dir / "index.html"
    
    # Get list of available reports
    reports = []
    for file in report_dir.glob("*.html"):
        if file.name != "index.html":
            reports.append({
                "name": file.stem.replace("_", " ").title(),
                "path": file.name
            })
    
    for file in report_dir.glob("*.json"):
        reports.append({
            "name": file.stem.replace("_", " ").title() + " (JSON)",
            "path": file.name
        })
    
    for file in report_dir.glob("*.xml"):
        reports.append({
            "name": file.stem.replace("_", " ").title() + " (XML)",
            "path": file.name
        })
    
    # Create the index file
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Simulation Validation Framework - Test Reports</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        ul {{
            list-style-type: none;
            padding: 0;
        }}
        li {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
            font-weight: bold;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.8em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Simulation Validation Framework - Test Reports</h1>
    
    <p>
        This page contains links to all generated test reports for the Simulation Validation Framework.
    </p>
    
    <h2>Available Reports</h2>
    <ul>
"""
    
    # Add links to each report
    for report in reports:
        html_content += f"""        <li>
            <a href="{report['path']}" target="_blank">{report['name']}</a>
        </li>
"""
    
    # Add timestamp and links to other directories
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content += f"""    </ul>
    
    <h2>Additional Resources</h2>
    <ul>
        <li><a href="../test_data/index.json" target="_blank">Test Data Index</a></li>
        <li><a href="../visualizations/index.html" target="_blank">Visualization Examples</a></li>
        <li><a href="../coverage/index.html" target="_blank">Coverage Report</a></li>
    </ul>
    
    <div class="timestamp">Generated on: {timestamp}</div>
</body>
</html>
"""
    
    # Write the index file
    with open(index_path, 'w') as f:
        f.write(html_content)


def generate_example_visualizations(output_dir):
    """Generate example visualizations for documentation and demos."""
    try:
        from data.duckdb.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector
        from data.duckdb.simulation_validation.test.test_data_generator import TestDataGenerator
        from data.duckdb.simulation_validation.db_integration import SimulationValidationDBIntegration
        
        print("  Setting up test data generator and database...")
        # Create a temporary database with sample data
        example_dir = output_dir / "examples"
        example_dir.mkdir(exist_ok=True)
        
        # Create test database for examples
        db_path = output_dir / "example_db.duckdb"
        db_integration = SimulationValidationDBIntegration(db_path=str(db_path))
        db_integration.initialize_database()
        
        # Generate test data
        generator = TestDataGenerator(seed=42)
        dataset = generator.generate_complete_dataset(
            num_models=3,
            num_hardware_types=3,
            days_per_series=30,
            include_calibrations=True,
            include_drifts=True
        )
        
        # Store data in the database
        print("  Populating database with test data...")
        for hw_result in dataset["hardware_results"][:50]:
            db_integration.store_hardware_result(hw_result)
            
        for sim_result in dataset["simulation_results"][:50]:
            db_integration.store_simulation_result(sim_result)
            
        for val_result in dataset["validation_results"][:50]:
            db_integration.store_validation_result(val_result)
            
        for cal_record in dataset["calibration_records"]:
            db_integration.store_calibration_record(cal_record)
            
        for drift_result in dataset["drift_detection_results"]:
            db_integration.store_drift_detection_result(drift_result)
        
        # Create connector with the database
        connector = ValidationVisualizerDBConnector(db_integration=db_integration)
        
        # Get a sample of models and hardware from the dataset
        models = set(hw.model_id for hw in dataset["hardware_results"][:20])
        hardware_types = set(hw.hardware_id for hw in dataset["hardware_results"][:20])
        
        # Generate examples for documentation
        print("  Generating example visualizations...")
        examples = [
            {
                "name": "mape_comparison",
                "function": connector.create_mape_comparison_chart_from_db,
                "args": {
                    "hardware_ids": list(hardware_types)[:2],
                    "model_ids": list(models)[:1],
                    "metric_name": "throughput_items_per_second",
                    "output_path": str(example_dir / "mape_comparison.html")
                }
            },
            {
                "name": "hardware_heatmap",
                "function": connector.create_hardware_comparison_heatmap_from_db,
                "args": {
                    "metric_name": "average_latency_ms",
                    "model_ids": list(models)[:2],
                    "output_path": str(example_dir / "hardware_heatmap.html")
                }
            },
            {
                "name": "time_series",
                "function": connector.create_time_series_chart_from_db,
                "args": {
                    "metric_name": "throughput_items_per_second",
                    "hardware_id": list(hardware_types)[0],
                    "model_id": list(models)[0],
                    "output_path": str(example_dir / "time_series.html")
                }
            },
            {
                "name": "drift_visualization",
                "function": connector.create_drift_visualization_from_db,
                "args": {
                    "hardware_type": list(hardware_types)[0],
                    "model_type": list(models)[0],
                    "output_path": str(example_dir / "drift_visualization.html")
                }
            },
            {
                "name": "calibration_improvement",
                "function": connector.create_calibration_improvement_chart_from_db,
                "args": {
                    "hardware_type": list(hardware_types)[0],
                    "model_type": list(models)[0],
                    "output_path": str(example_dir / "calibration_improvement.html")
                }
            },
            {
                "name": "simulation_vs_hardware",
                "function": connector.create_simulation_vs_hardware_chart_from_db,
                "args": {
                    "metric_name": "throughput_items_per_second",
                    "hardware_id": list(hardware_types)[0],
                    "model_id": list(models)[0],
                    "interactive": True,
                    "output_path": str(example_dir / "simulation_vs_hardware.html")
                }
            },
            {
                "name": "comprehensive_dashboard",
                "function": connector.create_comprehensive_dashboard_from_db,
                "args": {
                    "hardware_id": list(hardware_types)[0],
                    "model_id": list(models)[0],
                    "output_path": str(example_dir / "comprehensive_dashboard.html")
                }
            }
        ]
        
        # Create each example
        for example in examples:
            try:
                print(f"  Generating {example['name']} example...")
                result = example["function"](**example["args"])
                print(f"    Saved to: {example['args']['output_path']}")
            except Exception as e:
                print(f"    Error generating {example['name']}: {e}")
        
        # Create an index.html file that links to all examples
        create_example_index(example_dir, examples)
        
        # Close the database connection
        db_integration.close()
        
        print(f"Examples generated in: {example_dir}")
        
    except Exception as e:
        print(f"Error generating examples: {e}")


def generate_test_data():
    """Generate test data for evaluation and testing purposes."""
    try:
        from data.duckdb.simulation_validation.test.test_data_generator import TestDataGenerator
        
        print("Generating comprehensive test dataset...")
        # Create test data generator
        generator = TestDataGenerator(seed=42)  # Use fixed seed for reproducibility
        
        # Generate different dataset variants
        datasets = {
            "baseline": generator.generate_complete_dataset(
                num_models=3,
                num_hardware_types=3,
                days_per_series=30,
                include_calibrations=False,
                include_drifts=False
            ),
            "with_calibration": generator.generate_complete_dataset(
                num_models=2,
                num_hardware_types=2,
                days_per_series=20,
                include_calibrations=True,
                include_drifts=False
            ),
            "with_drift": generator.generate_complete_dataset(
                num_models=2,
                num_hardware_types=2,
                days_per_series=20,
                include_calibrations=False,
                include_drifts=True
            ),
            "comprehensive": generator.generate_complete_dataset(
                num_models=4,
                num_hardware_types=6,
                days_per_series=60,
                include_calibrations=True,
                include_drifts=True
            )
        }
        
        # Save all datasets to files
        for name, dataset in datasets.items():
            file_path = data_dir / f"{name}_dataset.json"
            generator.save_dataset_to_json(dataset, str(file_path))
            print(f"  Saved {name} dataset to: {file_path}")
        
        # Generate specific scenarios
        # Calibration scenario
        hw_results, sim_results, val_results, cal_record = generator.generate_calibration_scenario(
            model_id="bert-base-uncased",
            hardware_id="gpu_rtx3080",
            num_days_before=15,
            num_days_after=15
        )
        
        calibration_scenario = {
            "hardware_results": hw_results,
            "simulation_results": sim_results,
            "validation_results": val_results,
            "calibration_records": [cal_record]
        }
        
        calibration_path = data_dir / "calibration_scenario.json"
        generator.save_dataset_to_json(calibration_scenario, str(calibration_path))
        print(f"  Saved calibration scenario to: {calibration_path}")
        
        # Drift scenario
        hw_results, sim_results, val_results, drift_record = generator.generate_drift_scenario(
            model_id="vit-base-patch16-224",
            hardware_id="cpu_intel_xeon",
            num_days_before=15,
            num_days_after=15,
            drift_magnitude=0.3,
            drift_direction="positive"
        )
        
        drift_scenario = {
            "hardware_results": hw_results,
            "simulation_results": sim_results,
            "validation_results": val_results,
            "drift_detection_results": [drift_record]
        }
        
        drift_path = data_dir / "drift_scenario.json"
        generator.save_dataset_to_json(drift_scenario, str(drift_path))
        print(f"  Saved drift scenario to: {drift_path}")
        
        # Create index file
        create_dataset_index(data_dir, datasets, calibration_scenario, drift_scenario)
        
        print(f"Test data generation complete. Files saved to: {data_dir}")
        return True
    except Exception as e:
        print(f"Error generating test data: {e}")
        return False


def create_dataset_index(data_dir, datasets, calibration_scenario, drift_scenario):
    """Create an index file for the generated datasets."""
    index_path = data_dir / "index.json"
    
    index = {
        "timestamp": datetime.datetime.now().isoformat(),
        "datasets": {},
        "scenarios": {
            "calibration_scenario": str(data_dir / "calibration_scenario.json"),
            "drift_scenario": str(data_dir / "drift_scenario.json")
        }
    }
    
    # Add dataset information
    for name, dataset in datasets.items():
        index["datasets"][name] = {
            "path": str(data_dir / f"{name}_dataset.json"),
            "counts": {
                "hardware_results": len(dataset["hardware_results"]),
                "simulation_results": len(dataset["simulation_results"]),
                "validation_results": len(dataset["validation_results"]),
                "calibration_records": len(dataset["calibration_records"]),
                "drift_detection_results": len(dataset["drift_detection_results"])
            }
        }
    
    # Save index file
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def create_example_index(example_dir, examples):
    """Create an index.html file that links to all examples."""
    index_path = example_dir / "index.html"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Simulation Validation Framework - Visualization Examples</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        ul {{
            list-style-type: none;
            padding: 0;
        }}
        li {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
            font-weight: bold;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .description {{
            color: #666;
            margin-top: 5px;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.8em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Simulation Validation Framework - Visualization Examples</h1>
    
    <p>
        This page contains examples of various visualizations generated by the 
        Simulation Validation Framework's visualization system. These examples 
        demonstrate the integration between the database and visualization components.
    </p>
    
    <h2>Available Examples</h2>
    <ul>
"""
    
    # Add links to each example
    for example in examples:
        name = example["name"]
        filename = os.path.basename(example["args"]["output_path"])
        
        # Create a human-readable title
        title = " ".join(word.capitalize() for word in name.split("_"))
        
        # Add description based on the example type
        descriptions = {
            "mape_comparison": "Comparison of Mean Absolute Percentage Error (MAPE) across different hardware types and models.",
            "hardware_heatmap": "Heatmap visualization showing performance metrics across different hardware types.",
            "time_series": "Time series chart showing how metrics change over time for a specific hardware and model.",
            "drift_visualization": "Visualization of drift detection results, showing whether simulation accuracy has changed over time.",
            "calibration_improvement": "Chart showing the improvement in simulation accuracy after calibration.",
            "simulation_vs_hardware": "Scatter plot comparing simulation predictions with actual hardware measurements.",
            "comprehensive_dashboard": "Complete dashboard with multiple visualizations for a comprehensive view of simulation accuracy."
        }
        
        description = descriptions.get(name, "")
        
        html_content += f"""        <li>
            <a href="{filename}" target="_blank">{title}</a>
            <div class="description">{description}</div>
        </li>
"""
    
    # Add timestamp and closing tags
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content += f"""    </ul>
    
    <div class="timestamp">Generated on: {timestamp}</div>
</body>
</html>
"""
    
    # Write the index file
    with open(index_path, 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    # Display banner
    print("\n" + "=" * 80)
    print(" SIMULATION VALIDATION FRAMEWORK - END-TO-END TEST RUNNER ")
    print("=" * 80)
    
    # Display system information if requested
    if args.system_info and system_info:
        print("\nSystem Information:")
        print(f"Platform: {system_info['platform']}")
        print(f"Python: {system_info['python_version']}")
        print(f"Hostname: {system_info['hostname']}")
        print(f"CPU Count: {system_info['cpu_count']}")
        print(f"User: {system_info['user']}")
        print(f"Timestamp: {system_info['timestamp']}")
        print("-" * 80)
    
    # Get and display the current directory
    print(f"\nCurrent directory: {os.getcwd()}")
    print(f"Output directory: {output_dir}")
    
    # Display summary of what will be run
    print("\nTest Configuration:")
    if args.generate_test_data:
        print("- Only generating test data (no tests will be run)")
    elif args.generate_examples:
        print("- Generating example visualizations")
    
    if not args.generate_test_data:
        test_flags = []
        if args.run_db:
            test_flags.append("Database Integration")
        if args.run_connector:
            test_flags.append("Visualization Connector")
        if args.run_e2e:
            test_flags.append("Standard End-to-End")
        if args.run_comprehensive:
            test_flags.append("Comprehensive End-to-End")
        if args.run_validation:
            test_flags.append("Validation Component")
        if args.run_calibration:
            test_flags.append("Calibration Component")
        if args.run_drift:
            test_flags.append("Drift Detection Component")
        if args.run_dashboard or args.dashboard_integration:
            test_flags.append("Dashboard Integration")
        if args.run_visualization:
            test_flags.append("Visualization Component")
        
        if not test_flags:
            print("- Running all test suites")
        else:
            print(f"- Running specific test suites: {', '.join(test_flags)}")
        
        if args.skip_long_tests:
            print("- Skipping long-running tests")
        if args.parallel:
            print("- Running tests in parallel")
        if args.ci_mode:
            print("- Running in CI mode with GitHub Actions compatible output")
        if args.html_report:
            print("- Generating HTML test report")
        if args.junit_xml:
            print("- Generating JUnit XML report")
        if args.coverage:
            print("- Generating code coverage report")
        if args.performance_report:
            print("- Generating performance report")
    
    print("-" * 80)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Run the tests
        success = run_tests()
        
        # Display completion message
        execution_time = time.time() - start_time
        print(f"\nExecution completed in {execution_time:.2f} seconds")
        print(f"Status: {'SUCCESS' if success else 'FAILURE'}")
        
        if success:
            print(f"\nOutput files are available in:")
            print(f"- Reports: {report_dir}")
            print(f"- Test Data: {data_dir}")
            print(f"- Visualizations: {visualization_dir}")
            if args.coverage:
                print(f"- Coverage: {coverage_dir}")
        
        # Exit with appropriate status code
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)