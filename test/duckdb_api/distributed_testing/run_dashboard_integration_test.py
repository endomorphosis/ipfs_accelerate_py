#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run integration tests for the monitoring dashboard.
"""

import os
import sys
import unittest
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the tests."""
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Import the tests
    from tests.test_dashboard_integration import TestDashboardIntegration
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDashboardIntegration)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())