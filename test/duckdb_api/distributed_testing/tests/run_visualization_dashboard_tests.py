#!/usr/bin/env python3
"""
Run tests for the Visualization Dashboard Integration with Monitoring Dashboard.

This script runs comprehensive tests for the integration between the Advanced
Visualization System and the Monitoring Dashboard.
"""

import os
import sys
import argparse
import unittest
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_visualization_dashboard_tests")

# Add parent directory to path for module imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import test modules
from test_visualization_dashboard_integration import (
    TestVisualizationDashboardIntegration,
    TestMonitoringDashboardWithVisualization
)


def run_tests(test_pattern=None, verbose=False):
    """Run the visualization dashboard integration tests.
    
    Args:
        test_pattern: Optional pattern to filter tests
        verbose: Whether to show verbose output
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all tests from both test classes
    if test_pattern:
        # Filter tests by pattern
        loader = unittest.TestLoader()
        
        # Get tests from TestVisualizationDashboardIntegration
        vis_tests = loader.loadTestsFromTestCase(TestVisualizationDashboardIntegration)
        filtered_vis_tests = [t for t in vis_tests if test_pattern in t.id()]
        if filtered_vis_tests:
            suite.addTests(filtered_vis_tests)
        
        # Get tests from TestMonitoringDashboardWithVisualization
        monitor_tests = loader.loadTestsFromTestCase(TestMonitoringDashboardWithVisualization)
        filtered_monitor_tests = [t for t in monitor_tests if test_pattern in t.id()]
        if filtered_monitor_tests:
            suite.addTests(filtered_monitor_tests)
    else:
        # Add all tests
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVisualizationDashboardIntegration))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMonitoringDashboardWithVisualization))
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    # Run tests
    result = runner.run(suite)
    
    # Return True if all tests pass
    return result.wasSuccessful()


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run Visualization Dashboard Integration Tests")
    parser.add_argument("--test", help="Run a specific test pattern")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--create-dummy-data", action="store_true", 
                       help="Create dummy data for tests")
    args = parser.parse_args()
    
    # Create dummy data if requested
    if args.create_dummy_data:
        dummy_dir = os.path.join(os.getcwd(), "test_dashboard_data")
        os.makedirs(dummy_dir, exist_ok=True)
        logger.info(f"Created dummy data directory: {dummy_dir}")
    
    # Run tests
    success = run_tests(args.test, args.verbose)
    
    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())