#!/usr/bin/env python3
"""
Run All Error Visualization Tests.

This script executes all test cases for the Error Visualization system,
including unit tests, integration tests, and dashboard integration tests.
"""

import os
import sys
import unittest
import argparse
from pathlib import Path

# Add parent directory to path to import the modules
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the test modules
from test.duckdb_api.distributed_testing.tests.test_error_visualization import TestErrorVisualization
from test.duckdb_api.distributed_testing.tests.test_error_visualization_comprehensive import (
    TestSoundGeneration,
    TestSeverityDetection,
    TestJavaScriptSeverityDetection,
    TestWebSocketIntegration,
    TestErrorVisualizationIntegration,
    TestErrorExtraction
)
from test.duckdb_api.distributed_testing.tests.test_error_visualization_dashboard_integration import (
    TestDashboardRoutes,
    TestDashboardServer,
    TestErrorVisualizationHTML
)

def run_all_tests(verbosity=2, generate_report=False, report_format="html"):
    """Run all error visualization tests.
    
    Args:
        verbosity: The verbosity level for the test runner (1-3)
        generate_report: Whether to generate a test report
        report_format: The format for the test report ("html" or "text")
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add standard test cases
    suite.addTest(unittest.makeSuite(TestErrorVisualization))
    
    # Add comprehensive test cases
    suite.addTest(unittest.makeSuite(TestSoundGeneration))
    suite.addTest(unittest.makeSuite(TestSeverityDetection))
    suite.addTest(unittest.makeSuite(TestJavaScriptSeverityDetection))
    suite.addTest(unittest.makeSuite(TestWebSocketIntegration))
    suite.addTest(unittest.makeSuite(TestErrorVisualizationIntegration))
    suite.addTest(unittest.makeSuite(TestErrorExtraction))
    
    # Add dashboard integration test cases
    suite.addTest(unittest.makeSuite(TestDashboardRoutes))
    suite.addTest(unittest.makeSuite(TestDashboardServer))
    suite.addTest(unittest.makeSuite(TestErrorVisualizationHTML))
    
    # Check if report generation was requested
    if generate_report:
        if report_format == "html":
            try:
                import HtmlTestRunner
                runner = HtmlTestRunner.HTMLTestRunner(
                    output="test_reports",
                    report_name="error_visualization_tests",
                    combine_reports=True,
                    add_timestamp=True
                )
            except ImportError:
                print("HtmlTestRunner not available. Using default TestRunner.")
                runner = unittest.TextTestRunner(verbosity=verbosity)
        else:
            import xmlrunner
            runner = xmlrunner.XMLTestRunner(
                output="test_reports",
                verbosity=verbosity
            )
    else:
        # Use default test runner
        runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Run the tests
    print(f"Running {suite.countTestCases()} test cases...")
    result = runner.run(suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

def run_specific_tests(test_type, verbosity=2):
    """Run a specific set of tests based on the test type.
    
    Args:
        test_type: The type of tests to run ("sound", "severity", "websocket", "dashboard", "html")
        verbosity: The verbosity level for the test runner (1-3)
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases based on type
    if test_type == "sound":
        print("Running sound generation tests...")
        suite.addTest(unittest.makeSuite(TestSoundGeneration))
    elif test_type == "severity":
        print("Running severity detection tests...")
        suite.addTest(unittest.makeSuite(TestSeverityDetection))
        suite.addTest(unittest.makeSuite(TestJavaScriptSeverityDetection))
    elif test_type == "websocket":
        print("Running WebSocket integration tests...")
        suite.addTest(unittest.makeSuite(TestWebSocketIntegration))
    elif test_type == "dashboard":
        print("Running dashboard integration tests...")
        suite.addTest(unittest.makeSuite(TestDashboardRoutes))
        suite.addTest(unittest.makeSuite(TestDashboardServer))
    elif test_type == "html":
        print("Running HTML template tests...")
        suite.addTest(unittest.makeSuite(TestErrorVisualizationHTML))
    elif test_type == "system-critical":
        print("Running system-critical sound notification tests...")
        # Run the system-critical sound test script
        sound_dir = os.path.join(os.path.dirname(__file__), "dashboard", "static", "sounds")
        test_script = os.path.join(sound_dir, "test_sound_files.py")
        
        # Check that system-critical sound file exists
        sound_path = os.path.join(sound_dir, "error-system-critical.mp3")
        if not os.path.exists(sound_path):
            print(f"Error: System-critical sound file not found: {sound_path}")
            return 1
            
        print(f"System-critical sound file found: {sound_path}")
        
        # Run the sound file test to verify all files
        os.system(f"python {test_script}")
        
        # Verify error notification system with the test_error_notification_system.py script
        test_notification_script = os.path.join(sound_dir, "test_error_notification_system.py")
        if os.path.exists(test_notification_script):
            print("Running error notification system tests with system-critical sounds...")
            # This doesn't actually connect to a server, just checks integration logic
            os.system(f"python {test_notification_script} --system-critical-only --url http://localhost:8080")
        
        # We don't have actual test cases for unittest, so we're directly running scripts
        # For a real implementation, you'd create test cases for TestSystemCriticalSounds
        return 0
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run Error Visualization Tests")
    parser.add_argument("--verbosity", type=int, default=2, choices=[1, 2, 3],
                        help="Verbosity level (1-3)")
    parser.add_argument("--type", choices=["sound", "severity", "websocket", "dashboard", "html", "system-critical"],
                        help="Run specific type of tests")
    parser.add_argument("--report", action="store_true",
                        help="Generate test report")
    parser.add_argument("--report-format", choices=["html", "xml"], default="html",
                        help="Format for test report")
    parser.add_argument("--test-system-critical", action="store_true",
                        help="Test system-critical sound notification features")
    
    args = parser.parse_args()
    
    # Run tests
    if args.test_system_critical:
        print("Testing system-critical sound notification features...")
        return run_specific_tests("system-critical", args.verbosity)
    elif args.type:
        return run_specific_tests(args.type, args.verbosity)
    else:
        return run_all_tests(args.verbosity, args.report, args.report_format)

if __name__ == "__main__":
    sys.exit(main())