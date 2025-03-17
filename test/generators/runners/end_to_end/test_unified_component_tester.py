#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Component Tester

This script implements a comprehensive test suite for the unified_component_tester.py
to validate its functionality across different model families and hardware platforms.
It tests all the key features of the unified component tester:

1. Component generation for different model families
2. Test execution with result validation
3. Benchmark running and result collection
4. Documentation generation with enhanced templates
5. Result storage and comparison
6. Parallel execution capabilities
7. CLI functionality

Usage:
    python test_unified_component_tester.py 
    python test_unified_component_tester.py --comprehensive
    python test_unified_component_tester.py --model-family vision
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
import argparse
import logging
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import the unified component tester
from unified_component_tester import (
    UnifiedComponentTester, 
    run_unified_test,
    run_batch_tests,
    MODEL_FAMILIES,
    SUPPORTED_HARDWARE,
    PRIORITY_HARDWARE
)

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constants
TEST_DB_PATH = os.path.join(script_dir, "test_template_db.duckdb")
TEST_MODEL_NAME = "bert-base-uncased"
TEST_HARDWARE = "cpu"


class TestUnifiedComponentTester(unittest.TestCase):
    """Test case for the UnifiedComponentTester class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test instance
        self.tester = UnifiedComponentTester(
            model_name=TEST_MODEL_NAME,
            hardware=TEST_HARDWARE,
            db_path=TEST_DB_PATH,
            template_db_path=TEST_DB_PATH,
            update_expected=False,
            generate_docs=True,
            quick_test=True,
            keep_temp=True,
            verbose=True,
            tolerance=0.01
        )
        
    def tearDown(self):
        """Clean up after the test."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that the UnifiedComponentTester initializes correctly."""
        self.assertEqual(self.tester.model_name, TEST_MODEL_NAME)
        self.assertEqual(self.tester.hardware, TEST_HARDWARE)
        self.assertEqual(self.tester.db_path, TEST_DB_PATH)
        self.assertEqual(self.tester.template_db_path, TEST_DB_PATH)
        self.assertEqual(self.tester.model_family, "text-embedding")
        self.assertIsNotNone(self.tester.model_validator)
        self.assertIsNotNone(self.tester.result_comparer)
    
    def test_determine_model_family(self):
        """Test model family determination logic."""
        # Test BERT model family detection
        self.assertEqual(self.tester._determine_model_family(), "text-embedding")
        
        # Test other model families
        tester = UnifiedComponentTester(
            model_name="gpt2",
            hardware=TEST_HARDWARE,
            quick_test=True
        )
        self.assertEqual(tester._determine_model_family(), "text-generation")
        
        tester = UnifiedComponentTester(
            model_name="google/vit-base-patch16-224",
            hardware=TEST_HARDWARE,
            quick_test=True
        )
        self.assertEqual(tester._determine_model_family(), "vision")
        
        tester = UnifiedComponentTester(
            model_name="openai/whisper-tiny",
            hardware=TEST_HARDWARE,
            quick_test=True
        )
        self.assertEqual(tester._determine_model_family(), "audio")
        
        tester = UnifiedComponentTester(
            model_name="openai/clip-vit-base-patch32",
            hardware=TEST_HARDWARE,
            quick_test=True
        )
        self.assertEqual(tester._determine_model_family(), "multimodal")
    
    def test_generate_components(self):
        """Test component generation."""
        try:
            skill_file, test_file, benchmark_file = self.tester.generate_components(self.temp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(skill_file))
            self.assertTrue(os.path.exists(test_file))
            self.assertTrue(os.path.exists(benchmark_file))
            
            # Check file content
            with open(skill_file, 'r') as f:
                skill_content = f.read()
                self.assertIn(TEST_MODEL_NAME, skill_content)
                self.assertIn(TEST_HARDWARE, skill_content)
                
            with open(test_file, 'r') as f:
                test_content = f.read()
                self.assertIn(TEST_MODEL_NAME, test_content)
                self.assertIn(TEST_HARDWARE, test_content)
                
            with open(benchmark_file, 'r') as f:
                benchmark_content = f.read()
                self.assertIn(TEST_MODEL_NAME, benchmark_content)
                self.assertIn(TEST_HARDWARE, benchmark_content)
                
        except Exception as e:
            self.fail(f"Component generation failed with error: {e}")
    
    @patch('subprocess.run')
    def test_run_test(self, mock_run):
        """Test the run_test method with mocked subprocess."""
        # Mock successful test execution
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Ran 4 tests in 2.345s\n\nOK (successes=4)",
            stderr=""
        )
        
        # Generate components
        skill_file, test_file, benchmark_file = self.tester.generate_components(self.temp_dir)
        
        # Run the test
        test_result = self.tester.run_test(self.temp_dir)
        
        # Verify the test was called correctly
        mock_run.assert_called_once()
        mock_run_args = mock_run.call_args[0][0]
        self.assertEqual(mock_run_args[1], test_file)
        
        # Verify test results
        self.assertTrue(test_result.get("success", False))
        self.assertIn("execution_time", test_result)
        self.assertEqual(test_result.get("test_count", 0), 4)
        self.assertEqual(test_result.get("returncode", 1), 0)
    
    @patch('subprocess.run')
    def test_run_benchmark(self, mock_run):
        """Test the run_benchmark method with mocked subprocess."""
        # Create a mock benchmark result file
        benchmark_result_file = os.path.join(self.temp_dir, f"benchmark_{TEST_MODEL_NAME.replace('/', '_')}_{TEST_HARDWARE}_results.json")
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
        
        with open(benchmark_result_file, 'w') as f:
            json.dump(mock_benchmark_results, f)
        
        # Mock successful benchmark execution
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Benchmark results saved to benchmark_results.json",
            stderr=""
        )
        
        # Generate components
        skill_file, test_file, benchmark_file = self.tester.generate_components(self.temp_dir)
        
        # Run the benchmark
        benchmark_result = self.tester.run_benchmark(self.temp_dir)
        
        # Verify the benchmark was called correctly
        mock_run.assert_called_once()
        mock_run_args = mock_run.call_args[0][0]
        self.assertEqual(mock_run_args[1], benchmark_file)
        
        # Verify benchmark results
        self.assertTrue(benchmark_result.get("success", False))
        self.assertIn("execution_time", benchmark_result)
        self.assertIn("benchmark_results", benchmark_result)
        
        batch_results = benchmark_result.get("benchmark_results", {}).get("results_by_batch", {})
        self.assertIn("1", batch_results)
        self.assertIn("average_latency_ms", batch_results.get("1", {}))
    
    def test_generate_documentation(self):
        """Test the documentation generation."""
        # Generate components
        skill_file, test_file, benchmark_file = self.tester.generate_components(self.temp_dir)
        
        # Generate documentation
        doc_result = self.tester.generate_documentation(
            self.temp_dir,
            test_results={"success": True, "test_count": 4},
            benchmark_results={"results_by_batch": {"1": {"average_latency_ms": 10.5, "average_throughput_items_per_second": 95.2}}}
        )
        
        # Verify documentation result
        self.assertTrue(doc_result.get("success", False))
        self.assertIn("documentation_path", doc_result)
        
        # Check that documentation file exists
        doc_path = doc_result.get("documentation_path")
        self.assertTrue(os.path.exists(doc_path))
        
        # Check documentation content
        with open(doc_path, 'r') as f:
            doc_content = f.read()
            self.assertIn(TEST_MODEL_NAME, doc_content)
            self.assertIn(TEST_HARDWARE, doc_content)
            self.assertIn("# Overview", doc_content)
            self.assertIn("# Model Architecture", doc_content)
    
    @patch('subprocess.run')
    def test_run(self, mock_run):
        """Test the complete run method with mocked subprocess."""
        # Mock successful test execution
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Ran 4 tests in 2.345s\n\nOK (successes=4)",
            stderr=""
        )
        
        # Create a mock benchmark result file
        def mock_side_effect(*args, **kwargs):
            # Extract the file path from args
            for arg in args[0]:
                if arg.startswith('--output'):
                    output_index = args[0].index(arg) + 1
                    output_file = args[0][output_index]
                    
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
                    with open(output_file, 'w') as f:
                        json.dump(mock_benchmark_results, f)
                    break
            
            return MagicMock(returncode=0, stdout="Success", stderr="")
        
        mock_run.side_effect = mock_side_effect
        
        # Run the complete test
        result = self.tester.run()
        
        # Verify results
        self.assertTrue(result.get("success", False))
        self.assertIn("test_results", result)
        self.assertIn("benchmark_results", result)
        self.assertIn("doc_results", result)
        self.assertIn("storage_results", result)
        self.assertIn("comparison_results", result)


class TestBatchFunctionality(unittest.TestCase):
    """Test case for batch functionality and CLI."""
    
    @patch('unified_component_tester.run_unified_test')
    def test_run_batch_tests(self, mock_run_unified_test):
        """Test the run_batch_tests function."""
        # Mock the run_unified_test function
        mock_run_unified_test.return_value = {
            "success": True,
            "execution_time": 5.0,
            "test_results": {"success": True},
            "benchmark_results": {"success": True}
        }
        
        # Run batch tests
        results = run_batch_tests(
            models=[TEST_MODEL_NAME],
            hardware_platforms=[TEST_HARDWARE],
            quick_test=True,
            max_workers=1
        )
        
        # Verify batch results
        self.assertEqual(results["total_tests"], 1)
        self.assertEqual(results["success_count"], 1)
        self.assertEqual(results["failure_count"], 0)
        self.assertEqual(results["success_rate"], 1.0)
        self.assertIn("execution_time", results)
        self.assertIn("results", results)
        
        # Verify unified test was called correctly
        mock_run_unified_test.assert_called_once_with(
            model_name=TEST_MODEL_NAME,
            hardware=TEST_HARDWARE,
            db_path=None,
            template_db_path=None,
            update_expected=False,
            generate_docs=False,
            quick_test=True,
            keep_temp=False,
            verbose=False,
            tolerance=0.01,
            output_dir=None
        )
    
    @patch('unified_component_tester.run_batch_tests')
    def test_main_function(self, mock_run_batch_tests):
        """Test the main function with command line arguments."""
        # Mock the run_batch_tests function
        mock_run_batch_tests.return_value = {
            "total_tests": 1,
            "success_count": 1,
            "failure_count": 0,
            "success_rate": 1.0,
            "execution_time": 5.0,
            "results": [{"success": True}]
        }
        
        # Test with different command line arguments
        test_cases = [
            # Test with specific model and hardware
            ["--model", TEST_MODEL_NAME, "--hardware", TEST_HARDWARE, "--quick-test"],
            
            # Test with model family and priority hardware
            ["--model-family", "text-embedding", "--priority-hardware", "--quick-test"],
            
            # Test with all options
            [
                "--model", TEST_MODEL_NAME, 
                "--hardware", TEST_HARDWARE, 
                "--db-path", TEST_DB_PATH,
                "--template-db-path", TEST_DB_PATH,
                "--update-expected", 
                "--generate-docs",
                "--quick-test",
                "--keep-temp",
                "--verbose",
                "--tolerance", "0.02",
                "--max-workers", "2",
                "--output-dir", self.temp_dir
            ]
        ]
        
        for args in test_cases:
            # Create a temporary directory for each test case
            self.temp_dir = tempfile.mkdtemp()
            
            try:
                # Mock sys.argv
                with patch('sys.argv', ['unified_component_tester.py'] + args):
                    # Run main
                    from unified_component_tester import main
                    main()
                
                # Verify batch tests was called
                mock_run_batch_tests.assert_called()
            finally:
                shutil.rmtree(self.temp_dir)


class TestModelFamilyCoverage(unittest.TestCase):
    """Test coverage for different model families."""
    
    @patch('subprocess.run')
    def test_text_embedding_model(self, mock_run):
        """Test with a text embedding model."""
        self._test_model_family("bert-base-uncased", "text-embedding", mock_run)
    
    @patch('subprocess.run')
    def test_text_generation_model(self, mock_run):
        """Test with a text generation model."""
        self._test_model_family("gpt2", "text-generation", mock_run)
    
    @patch('subprocess.run')
    def test_vision_model(self, mock_run):
        """Test with a vision model."""
        self._test_model_family("google/vit-base-patch16-224", "vision", mock_run)
    
    @patch('subprocess.run')
    def test_audio_model(self, mock_run):
        """Test with an audio model."""
        self._test_model_family("openai/whisper-tiny", "audio", mock_run)
    
    @patch('subprocess.run')
    def test_multimodal_model(self, mock_run):
        """Test with a multimodal model."""
        self._test_model_family("openai/clip-vit-base-patch32", "multimodal", mock_run)
    
    def _test_model_family(self, model_name, expected_family, mock_run):
        """Helper method to test a specific model family."""
        # Setup mock for subprocess.run
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Success",
            stderr=""
        )
        
        # Create the tester
        tester = UnifiedComponentTester(
            model_name=model_name,
            hardware=TEST_HARDWARE,
            db_path=TEST_DB_PATH,
            template_db_path=TEST_DB_PATH,
            quick_test=True,
            keep_temp=True
        )
        
        # Check model family determination
        self.assertEqual(tester._determine_model_family(), expected_family)
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Generate components
            skill_file, test_file, benchmark_file = tester.generate_components(temp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(skill_file))
            self.assertTrue(os.path.exists(test_file))
            self.assertTrue(os.path.exists(benchmark_file))
            
            # Check that model family specific content is included
            with open(skill_file, 'r') as f:
                skill_content = f.read()
                self.assertIn(model_name, skill_content)
                self.assertIn(expected_family, skill_content)
                
        finally:
            shutil.rmtree(temp_dir)


class TestHardwarePlatformCoverage(unittest.TestCase):
    """Test coverage for different hardware platforms."""
    
    @patch('subprocess.run')
    def test_cpu_platform(self, mock_run):
        """Test with CPU hardware platform."""
        self._test_hardware_platform("cpu", mock_run)
    
    @patch('subprocess.run')
    def test_cuda_platform(self, mock_run):
        """Test with CUDA hardware platform."""
        self._test_hardware_platform("cuda", mock_run)
    
    @patch('subprocess.run')
    def test_webgpu_platform(self, mock_run):
        """Test with WebGPU hardware platform."""
        self._test_hardware_platform("webgpu", mock_run)
    
    def _test_hardware_platform(self, hardware, mock_run):
        """Helper method to test a specific hardware platform."""
        # Setup mock for subprocess.run
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Success",
            stderr=""
        )
        
        # Create the tester
        tester = UnifiedComponentTester(
            model_name=TEST_MODEL_NAME,
            hardware=hardware,
            db_path=TEST_DB_PATH,
            template_db_path=TEST_DB_PATH,
            quick_test=True,
            keep_temp=True
        )
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Generate components
            skill_file, test_file, benchmark_file = tester.generate_components(temp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(skill_file))
            self.assertTrue(os.path.exists(test_file))
            self.assertTrue(os.path.exists(benchmark_file))
            
            # Check that hardware specific content is included
            with open(skill_file, 'r') as f:
                skill_content = f.read()
                self.assertIn(hardware, skill_content)
                
        finally:
            shutil.rmtree(temp_dir)


def run_selected_tests(model_family: Optional[str] = None, 
                      hardware: Optional[str] = None, 
                      comprehensive: bool = False) -> None:
    """
    Run a subset of tests based on the specified model family and hardware.
    
    Args:
        model_family: Optional model family to test (e.g., "text-embedding", "vision")
        hardware: Optional hardware platform to test (e.g., "cpu", "cuda")
        comprehensive: Whether to run comprehensive tests (all combinations)
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add basic tests
    suite.addTest(unittest.makeSuite(TestUnifiedComponentTester))
    suite.addTest(unittest.makeSuite(TestBatchFunctionality))
    
    # Add model family tests
    if model_family:
        if model_family == "text-embedding":
            suite.addTest(TestModelFamilyCoverage("test_text_embedding_model"))
        elif model_family == "text-generation":
            suite.addTest(TestModelFamilyCoverage("test_text_generation_model"))
        elif model_family == "vision":
            suite.addTest(TestModelFamilyCoverage("test_vision_model"))
        elif model_family == "audio":
            suite.addTest(TestModelFamilyCoverage("test_audio_model"))
        elif model_family == "multimodal":
            suite.addTest(TestModelFamilyCoverage("test_multimodal_model"))
        else:
            logger.warning(f"Unknown model family: {model_family}")
    elif comprehensive:
        suite.addTest(unittest.makeSuite(TestModelFamilyCoverage))
    
    # Add hardware platform tests
    if hardware:
        if hardware == "cpu":
            suite.addTest(TestHardwarePlatformCoverage("test_cpu_platform"))
        elif hardware == "cuda":
            suite.addTest(TestHardwarePlatformCoverage("test_cuda_platform"))
        elif hardware == "webgpu":
            suite.addTest(TestHardwarePlatformCoverage("test_webgpu_platform"))
        else:
            logger.warning(f"Unknown hardware platform: {hardware}")
    elif comprehensive:
        suite.addTest(unittest.makeSuite(TestHardwarePlatformCoverage))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nRan {result.testsRun} tests")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Unified Component Tester")
    
    # Test selection options
    parser.add_argument("--model-family", choices=MODEL_FAMILIES.keys(), 
                        help="Test specific model family")
    parser.add_argument("--hardware", choices=SUPPORTED_HARDWARE,
                        help="Test specific hardware platform")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive tests (all combinations)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.model_family or args.hardware or args.comprehensive:
        run_selected_tests(
            model_family=args.model_family,
            hardware=args.hardware,
            comprehensive=args.comprehensive
        )
    else:
        unittest.main()