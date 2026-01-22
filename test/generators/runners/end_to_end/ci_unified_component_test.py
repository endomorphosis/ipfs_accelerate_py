#!/usr/bin/env python3
"""
CI/CD Integration Test for Unified Component Tester

This script provides a simplified test for the unified component tester
that can be run in CI/CD environments. It tests the core functionality
of the unified component tester with a small set of model and hardware
combinations to ensure the system is working correctly.

Usage:
    python ci_unified_component_test.py
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import the unified component tester
from unified_component_tester import (
    UnifiedComponentTester,
    run_unified_test,
    MODEL_FAMILIES,
    SUPPORTED_HARDWARE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for testing
TEST_MODEL_NAME = "bert-base-uncased"
TEST_HARDWARE = "cpu"
TEST_DB_PATH = os.path.join(script_dir, "test_template_db.duckdb")


class CIUnifiedComponentTest(unittest.TestCase):
    """CI/CD test case for the UnifiedComponentTester."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after the test."""
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_component_generation(self, mock_run):
        """Test component generation functionality."""
        # Mock subprocess.run
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Success",
            stderr=""
        )
        
        # Create tester
        tester = UnifiedComponentTester(
            model_name=TEST_MODEL_NAME,
            hardware=TEST_HARDWARE,
            db_path=TEST_DB_PATH,
            template_db_path=TEST_DB_PATH,
            quick_test=True,
            keep_temp=True
        )
        
        # Generate components
        skill_file, test_file, benchmark_file = tester.generate_components(self.temp_dir)
        
        # Check that files were created
        self.assertTrue(os.path.exists(skill_file))
        self.assertTrue(os.path.exists(test_file))
        self.assertTrue(os.path.exists(benchmark_file))
        
        # Check file content contains model and hardware information
        with open(skill_file, 'r') as f:
            skill_content = f.read()
            self.assertIn(TEST_MODEL_NAME, skill_content)
            self.assertIn(TEST_HARDWARE, skill_content)
    
    @patch('subprocess.run')
    def test_simplified_execution(self, mock_run):
        """Test a simplified execution workflow."""
        # Mock subprocess.run for both test and benchmark
        def mock_side_effect(*args, **kwargs):
            # If this is the benchmark call, create a benchmark result file
            if any("benchmark" in arg for arg in args[0]):
                # Extract the output file path from args
                for i, arg in enumerate(args[0]):
                    if arg == "--output":
                        output_file = args[0][i+1]
                        
                        # Create the benchmark result file
                        mock_benchmark_results = {
                            "model_name": TEST_MODEL_NAME,
                            "hardware": TEST_HARDWARE,
                            "results_by_batch": {
                                "1": {
                                    "average_latency_ms": 10.5,
                                    "std_latency_ms": 1.2,
                                    "min_latency_ms": 9.8,
                                    "max_latency_ms": 12.3,
                                    "average_throughput_items_per_second": 95.2
                                }
                            }
                        }
                        
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        import json
                        with open(output_file, 'w') as f:
                            json.dump(mock_benchmark_results, f)
                        break
            
            # Return success for all subprocess calls
            return MagicMock(
                returncode=0,
                stdout="Ran 4 tests in 2.345s\n\nOK (successes=4)",
                stderr=""
            )
        
        mock_run.side_effect = mock_side_effect
        
        # Run the simplified test
        result = run_unified_test(
            model_name=TEST_MODEL_NAME,
            hardware=TEST_HARDWARE,
            db_path=TEST_DB_PATH,
            template_db_path=TEST_DB_PATH,
            update_expected=True,
            generate_docs=True,
            quick_test=True
        )
        
        # Check the result
        self.assertTrue(result.get("success", False))
        self.assertIn("test_results", result)
        self.assertIn("benchmark_results", result)
        
        # Verify test results
        test_results = result.get("test_results", {})
        self.assertTrue(test_results.get("success", False))
        
        # Verify benchmark results
        benchmark_results = result.get("benchmark_results", {})
        self.assertTrue(benchmark_results.get("success", False))
        self.assertIn("benchmark_results", benchmark_results)
        self.assertIn("results_by_batch", benchmark_results.get("benchmark_results", {}))
    
    def test_model_family_detection(self):
        """Test model family detection for CI."""
        # Test the detection for multiple model families
        test_cases = [
            ("bert-base-uncased", "text-embedding"),
            ("gpt2", "text-generation"),
            ("google/vit-base-patch16-224", "vision"),
            ("openai/whisper-tiny", "audio"),
            ("openai/clip-vit-base-patch32", "multimodal")
        ]
        
        for model_name, expected_family in test_cases:
            tester = UnifiedComponentTester(
                model_name=model_name,
                hardware=TEST_HARDWARE,
                quick_test=True
            )
            
            family = tester._determine_model_family()
            self.assertEqual(family, expected_family, f"Failed to detect {expected_family} for {model_name}")


def main():
    """Run the CI tests for unified component tester."""
    logger.info("Starting CI tests for unified component tester")
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CIUnifiedComponentTest))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Determine exit code based on test results
    exit_code = 0 if result.wasSuccessful() else 1
    
    # Print summary
    logger.info("CI test summary:")
    logger.info(f"Ran {result.testsRun} tests")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()