#!/usr/bin/env python3
"""
Integration Test for Generator and Benchmark Components

This module provides integration tests for verifying the end-to-end workflow
from model code generation to benchmarking. It ensures that models generated
by the refactored_generator_suite can be successfully benchmarked by the
refactored_benchmark_suite.

Example usage:
    python -m refactored_test_suite.integration.test_generator_benchmark_integration
"""

import os
import sys
import unittest
import json
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directories to path to allow imports
current_dir = Path(__file__).resolve().parent
test_dir = current_dir.parent.parent
sys.path.append(str(test_dir))

# Import required modules
try:
    from refactored_generator_suite.generate_simple_model import generate_model
    GENERATOR_AVAILABLE = True
except ImportError:
    logger.warning("Generator suite not available, some tests will be skipped")
    GENERATOR_AVAILABLE = False

try:
    from refactored_benchmark_suite.run_skillset_benchmark import run_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    logger.warning("Benchmark suite not available, some tests will be skipped")
    BENCHMARK_AVAILABLE = False

class GeneratorBenchmarkIntegrationTest(unittest.TestCase):
    """Integration tests for the generator and benchmark components."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        cls.test_models = ["bert-base-uncased", "whisper-tiny", "vit-base-patch16-224"]
        cls.temp_dir = Path(test_dir) / "temp_integration_test"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.generated_files = []

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if cls.generated_files:
            logger.info(f"Cleaning up {len(cls.generated_files)} generated files")
            for file_path in cls.generated_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up file {file_path}: {e}")

    @unittest.skipIf(not (GENERATOR_AVAILABLE and BENCHMARK_AVAILABLE),
                    "Generator or benchmark suite not available")
    def test_generator_to_benchmark_pipeline(self):
        """Test the end-to-end workflow from model generation to benchmarking."""
        for model_name in self.test_models:
            with self.subTest(model=model_name):
                # Step 1: Generate model implementation
                logger.info(f"Generating model implementation for {model_name}")
                try:
                    output_path = generate_model(
                        model_name, 
                        output_dir=str(self.temp_dir),
                        force=True
                    )
                    self.generated_files.append(output_path)
                    
                    # Verify model file was generated
                    self.assertTrue(os.path.exists(output_path))
                    logger.info(f"Successfully generated model at {output_path}")
                    
                    # Step 2: Run benchmark on the generated model
                    logger.info(f"Running benchmark for {model_name}")
                    benchmark_results = run_benchmark(
                        output_path,
                        hardware="cpu",
                        batch_sizes=[1],
                        runs=1,  # Use minimal runs for testing
                        benchmark_type="inference"
                    )
                    
                    # Verify benchmark results
                    self.assertIsNotNone(benchmark_results)
                    self.assertEqual(benchmark_results["status"], "success")
                    self.assertIn("latency_ms", benchmark_results)
                    
                    # Save benchmark results
                    results_path = self.temp_dir / f"{model_name}_benchmark_results.json"
                    with open(results_path, 'w') as f:
                        json.dump(benchmark_results, f, indent=2)
                    self.generated_files.append(str(results_path))
                    
                    logger.info(f"Integration test passed for {model_name}")
                    
                except Exception as e:
                    self.fail(f"Integration test failed for {model_name}: {e}")

    @unittest.skipIf(not (GENERATOR_AVAILABLE and BENCHMARK_AVAILABLE),
                    "Generator or benchmark suite not available")
    def test_generator_to_benchmark_with_options(self):
        """Test generation and benchmarking with custom options."""
        model_name = "bert-base-uncased"
        
        try:
            # Generate model with custom hardware
            logger.info(f"Generating model with custom hardware options")
            output_path = generate_model(
                model_name,
                output_dir=str(self.temp_dir),
                hardware=["cpu", "cuda"],  # Specify multiple hardware backends
                force=True
            )
            self.generated_files.append(output_path)
            
            # Run benchmark with throughput test
            logger.info(f"Running throughput benchmark")
            benchmark_results = run_benchmark(
                output_path,
                hardware="cpu",
                batch_sizes=[1, 2],
                runs=1,
                benchmark_type="throughput",
                concurrent_workers=2
            )
            
            # Verify throughput results
            self.assertIsNotNone(benchmark_results)
            self.assertEqual(benchmark_results["status"], "success")
            self.assertIn("throughput", benchmark_results)
            
            # Save results
            results_path = self.temp_dir / f"{model_name}_throughput_results.json"
            with open(results_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            self.generated_files.append(str(results_path))
            
            logger.info(f"Advanced options integration test passed")
            
        except Exception as e:
            self.fail(f"Advanced options integration test failed: {e}")

    @unittest.skipIf(not (GENERATOR_AVAILABLE and BENCHMARK_AVAILABLE),
                    "Generator or benchmark suite not available")
    def test_error_handling(self):
        """Test error handling in the integration pipeline."""
        # Test with invalid model name
        with self.assertRaises(Exception):
            generate_model("non-existent-model-123456789", output_dir=str(self.temp_dir))
        
        # Test with invalid hardware
        if GENERATOR_AVAILABLE:
            # Generate valid model first
            model_name = "bert-base-uncased"
            output_path = generate_model(model_name, output_dir=str(self.temp_dir), force=True)
            self.generated_files.append(output_path)
            
            # Then test invalid hardware
            if BENCHMARK_AVAILABLE:
                with self.assertRaises(Exception):
                    run_benchmark(output_path, hardware="invalid_hardware_type")

class ApiIntegrationTest(unittest.TestCase):
    """Integration tests for the API components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up API test environment."""
        # Check if API modules are available
        try:
            from refactored_test_suite.api.test_client import ApiTestClient
            cls.API_AVAILABLE = True
        except ImportError:
            logger.warning("API client not available, API tests will be skipped")
            cls.API_AVAILABLE = False
    
    @unittest.skipIf(not GENERATOR_AVAILABLE or not hasattr(ApiIntegrationTest, 'API_AVAILABLE') or not ApiIntegrationTest.API_AVAILABLE,
                    "Generator or API client not available")
    def test_api_integration(self):
        """Test integration with API components."""
        # This is a placeholder for future API integration tests
        # Will be implemented as API components are refactored
        logger.info("API integration test placeholder - to be implemented")
        self.skipTest("API integration tests not yet implemented")

def run_tests():
    """Run the integration tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()