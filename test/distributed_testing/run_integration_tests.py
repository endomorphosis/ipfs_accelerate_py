#!/usr/bin/env python3
"""
Runner script for integration tests of Dynamic Resource Manager,
Performance Trend Analyzer, and End-to-End system tests with the Coordinator.

This script runs all integration tests to verify the proper interaction
between the new components and the existing distributed testing framework.
"""

import os
import sys
import argparse
import unittest
import anyio
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the test classes
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
from test_dynamic_resource_manager import TestDynamicResourceManagerIntegration
from test_performance_trend_analyzer import TestPerformanceTrendAnalyzerIntegration
from test_e2e_integrated_system import TestE2EIntegratedSystem


def create_test_suite(component=None):
    """
    Create a test suite with the specified component tests.
    
    Args:
        component: The component to test (drm, pta, e2e, or None for all)
        
    Returns:
        A unittest TestSuite
    """
    suite = unittest.TestSuite()
    
    # Add Dynamic Resource Manager tests
    if component is None or component == "drm":
        suite.addTest(TestDynamicResourceManagerIntegration('test_resource_manager_provisions_initial_workers'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_resource_manager_scales_up_with_high_queue'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_resource_manager_scales_down_with_low_queue'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_worker_registration_with_coordinator'))
        suite.addTest(TestDynamicResourceManagerIntegration('test_anomaly_detection_and_recovery'))
    
    # Add Performance Trend Analyzer tests
    if component is None or component == "pta":
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_connects_to_coordinator'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_collects_metrics'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_detects_anomalies'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_identifies_trends'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_analyzer_generates_visualizations'))
        suite.addTest(TestPerformanceTrendAnalyzerIntegration('test_database_integration'))
    
    # Add End-to-End Integration tests
    if component is None or component == "e2e":
        suite.addTest(TestE2EIntegratedSystem('test_e2e_integrated_system'))
    
    return suite


def run_tests(component=None, verbose=False):
    """
    Run the integration tests.
    
    Args:
        component: The component to test (drm, pta, or None for all)
        verbose: Whether to show verbose output
    """
    try:
        # No explicit event loop management needed
        
        # Create test suite
        suite = create_test_suite(component)
        
        # Create test runner
        verbosity = 2 if verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        
        # Run the tests
        result = runner.run(suite)
        
        # Return exit code based on test results
        return 0 if result.wasSuccessful() else 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1
    finally:
        # No explicit loop cleanup required
        pass


def main():
    """Main entry point for the integration test runner."""
    parser = argparse.ArgumentParser(description="Run integration tests for the Distributed Testing Framework")
    parser.add_argument("--component", choices=["drm", "pta", "e2e"], 
                       help="Specific component to test (drm=Dynamic Resource Manager, pta=Performance Trend Analyzer, e2e=End-to-End Integration)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    return run_tests(component=args.component, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())