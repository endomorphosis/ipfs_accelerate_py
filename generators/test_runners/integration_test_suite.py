#\!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for IPFS Accelerate Python

This test suite verifies that all components of the system work together
properly across different hardware platforms, model types, and APIs.
"""

import os
import sys
import json
import time
import datetime
import argparse
import unittest
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__),
                                         f"integration_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("integration_test")

# Try to import necessary modules
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not available")

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not available")

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers library not available")

# Define test categories
CATEGORIES = {
    "hardware_detection": "Tests that check hardware availability",
    "api_functionality": "Tests for API functionality across endpoints",
    "cross_platform": "Tests for cross-platform compatibility",
    "web_integration": "Tests for web platform integration",
    "database": "Tests for database integration and functionality",
    "p2p": "Tests for P2P optimizations",
    "performance": "Performance benchmarks across configurations"
}

@dataclass
class TestResult:
    """Class for storing test results."""
    category: str
    name: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    hardware_type: Optional[str] = None
    model_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert the test result to a dictionary."""
        return {
            "category": self.category,
            "name": self.name,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "hardware_type": self.hardware_type,
            "model_name": self.model_name,
            "details": self.details
        }

class IntegrationTestSuite:
    """Main test suite for integration testing."""
    
    def __init__(self, categories=None, hardware=None, ci_mode=False, output=None):
        """Initialize the test suite.
        
        Args:
            categories (list): List of test categories to run
            hardware (list): List of hardware types to test on
            ci_mode (bool): Whether to run in CI mode
            output (str): Path to output file for results
        """
        self.categories = categories or list(CATEGORIES.keys())
        self.hardware = hardware or ["cpu"]
        self.ci_mode = ci_mode
        self.output = output
        self.results = []
        self.start_time = time.time()
        
        logger.info(f"Initializing Integration Test Suite")
        logger.info(f"Categories: {self.categories}")
        logger.info(f"Hardware: {self.hardware}")
        logger.info(f"CI Mode: {self.ci_mode}")
    
    def run_tests(self):
        """Run all tests in the specified categories."""
        for category in self.categories:
            if category not in CATEGORIES:
                logger.warning(f"Unknown category: {category}")
                continue
            
            logger.info(f"Running tests for category: {category}")
            
            # Call the appropriate test method based on category
            method_name = f"test_{category}"
            if hasattr(self, method_name):
                try:
                    getattr(self, method_name)()
                except Exception as e:
                    logger.error(f"Error running tests for category {category}: {e}")
                    traceback.print_exc()
            else:
                logger.warning(f"No test method found for category: {category}")
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def add_result(self, category, name, success, error_message=None, 
                  execution_time=0.0, hardware_type=None, model_name=None, 
                  details=None):
        """Add a test result to the results list."""
        result = TestResult(
            category=category,
            name=name,
            success=success,
            error_message=error_message,
            execution_time=execution_time,
            hardware_type=hardware_type,
            model_name=model_name,
            details=details or {}
        )
        self.results.append(result)
        
        # Log the result
        if success:
            logger.info(f"✅ {category} - {name}: Passed in {execution_time:.2f}s")
        else:
            logger.error(f"❌ {category} - {name}: Failed in {execution_time:.2f}s - {error_message}")
        
        return result
    
    def save_results(self):
        """Save the test results to a file."""
        if not self.output:
            self.output = f"integration_test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to dictionaries
        results_dict = {
            "timestamp": datetime.datetime.now().isoformat(),
            "execution_time": time.time() - self.start_time,
            "categories": self.categories,
            "hardware": self.hardware,
            "ci_mode": self.ci_mode,
            "results": [result.to_dict() for result in self.results]
        }
        
        # Save to file
        with open(self.output, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {self.output}")
    
    def print_summary(self):
        """Print a summary of the test results."""
        total = len(self.results)
        passed = sum(1 for result in self.results if result.success)
        failed = total - passed
        
        logger.info("=" * 50)
        logger.info(f"Integration Test Suite Summary")
        logger.info("=" * 50)
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {(passed / total * 100) if total > 0 else 0:.2f}%")
        logger.info(f"Total execution time: {time.time() - self.start_time:.2f}s")
        logger.info("=" * 50)
        
        # Print category-specific results
        for category in self.categories:
            category_results = [r for r in self.results if r.category == category]
            category_total = len(category_results)
            category_passed = sum(1 for r in category_results if r.success)
            
            if category_total > 0:
                logger.info(f"{category}: {category_passed}/{category_total} passed " +
                           f"({(category_passed / category_total * 100):.2f}%)")
    
    def test_hardware_detection(self):
        """Test hardware detection capabilities."""
        for hardware_type in self.hardware:
            start_time = time.time()
            try:
                # Test CPU availability always available
                if hardware_type == "cpu":
                    self.add_result(
                        category="hardware_detection",
                        name=f"Detect {hardware_type}",
                        success=True,
                        execution_time=time.time() - start_time,
                        hardware_type=hardware_type,
                        details={"available": True}
                    )
                    continue
                
                # For other hardware types, check availability
                available = False
                details = {"available": False}
                
                # PyTorch CUDA detection
                if hardware_type == "cuda" and HAS_TORCH:
                    available = torch.cuda.is_available()
                    if available:
                        details = {
                            "available": True,
                            "device_count": torch.cuda.device_count(),
                            "device_name": torch.cuda.get_device_name(0),
                            "memory_allocated": torch.cuda.memory_allocated(0),
                            "memory_reserved": torch.cuda.memory_reserved(0)
                        }
                
                # PyTorch MPS (Metal Performance Shaders) detection for Apple Silicon
                elif hardware_type == "mps" and HAS_TORCH:
                    if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
                        available = torch.mps.is_available()
                        if available:
                            details = {
                                "available": True,
                                "platform": "Apple Silicon"
                            }
                
                # Add more hardware detection as needed
                
                self.add_result(
                    category="hardware_detection",
                    name=f"Detect {hardware_type}",
                    success=available,
                    error_message=None if available else f"{hardware_type} not available",
                    execution_time=time.time() - start_time,
                    hardware_type=hardware_type,
                    details=details
                )
            
            except Exception as e:
                self.add_result(
                    category="hardware_detection",
                    name=f"Detect {hardware_type}",
                    success=False,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                    hardware_type=hardware_type,
                    details={"exception": str(e)}
                )
    
    def test_cross_platform(self):
        """Test cross-platform compatibility."""
        # For now, just test importing key modules
        modules_to_test = [
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "duckdb"
        ]
        
        for module_name in modules_to_test:
            start_time = time.time()
            try:
                # Try to import the module
                module = importlib.import_module(module_name)
                
                # Get module version if available
                version = getattr(module, "__version__", "unknown")
                
                self.add_result(
                    category="cross_platform",
                    name=f"Import {module_name}",
                    success=True,
                    execution_time=time.time() - start_time,
                    details={"version": version}
                )
            except ImportError as e:
                self.add_result(
                    category="cross_platform",
                    name=f"Import {module_name}",
                    success=False,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                    details={"error": str(e)}
                )
    
    def test_database(self):
        """Test database functionality."""
        if not HAS_DUCKDB:
            logger.warning("DuckDB not available, skipping database tests")
            return
        
        # Test creating a connection
        start_time = time.time()
        try:
            # Create an in-memory database
            conn = duckdb.connect(":memory:")
            
            # Create a simple table
            conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
            conn.execute("INSERT INTO test VALUES (1, 'test')")
            
            # Query the table
            result = conn.execute("SELECT * FROM test").fetchall()
            
            # Check result
            success = len(result) == 1 and result[0][0] == 1 and result[0][1] == 'test'
            
            self.add_result(
                category="database",
                name="DuckDB basic functionality",
                success=success,
                execution_time=time.time() - start_time,
                details={"result": str(result)}
            )
        except Exception as e:
            self.add_result(
                category="database",
                name="DuckDB basic functionality",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                details={"error": str(e)}
            )

def main():
    """Run the integration test suite from the command line."""
    parser = argparse.ArgumentParser(description="Run the integration test suite")
    parser.add_argument("--categories", "-c", nargs="+", choices=list(CATEGORIES.keys()),
                        help="Categories of tests to run")
    parser.add_argument("--hardware", "-h", nargs="+", default=["cpu"],
                        help="Hardware types to test on")
    parser.add_argument("--ci-mode", action="store_true",
                        help="Run in CI mode")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Run the tests
    suite = IntegrationTestSuite(
        categories=args.categories,
        hardware=args.hardware,
        ci_mode=args.ci_mode,
        output=args.output
    )
    
    suite.run_tests()

if __name__ == "__main__":
    main()
