#!/usr/bin/env python3
"""
Test script for enhanced_ci_cd_reports.py

This script tests the functionality of the enhanced CI/CD reports, particularly
the SimulationValidator and CrossHardwareComparison classes.

Usage:
    python test_enhanced_reports.py
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the modules to test
from enhanced_ci_cd_reports import (
    SimulationValidator, 
    CrossHardwareComparison,
    SIMULATION_TOLERANCE
)

class TestSimulationValidator(unittest.TestCase):
    """Test case for the SimulationValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = SimulationValidator(tolerance=SIMULATION_TOLERANCE)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample test results
        self.sample_results = {
            "bert-base-uncased": {
                "cpu": {
                    "status": "success",
                    "metadata": {},
                    "metrics": {
                        "throughput": 100.0,
                        "latency": 10.0
                    }
                },
                "cuda": {
                    "status": "success",
                    "metadata": {},
                    "metrics": {
                        "throughput": 350.0,  # 3.5x CPU throughput (expected ratio)
                        "latency": 2.85  # ~3.5x faster than CPU (expected ratio)
                    }
                },
                "webgpu": {
                    "status": "success",
                    "metadata": {"simulation": True},  # Explicitly marked as simulation
                    "metrics": {
                        "throughput": 200.0,  # 2.0x CPU throughput (expected ratio)
                        "latency": 5.0  # 2.0x faster than CPU (expected ratio)
                    }
                },
                "openvino": {
                    "status": "success",
                    "metadata": {},
                    "metrics": {
                        "throughput": 320.0,  # Too high compared to expected 1.5x CPU
                        "latency": 3.0  # Too fast compared to expected ~1.5x CPU
                    }
                }
            }
        }
        
        # Write sample results to disk
        for model, hw_results in self.sample_results.items():
            for hw, result in hw_results.items():
                result_dir = os.path.join(self.temp_dir, model, hw, "20250311_120000")
                os.makedirs(result_dir, exist_ok=True)
                
                result_file = os.path.join(result_dir, "result.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Create result_path entry
                self.sample_results[model][hw]["result_path"] = result_dir
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_is_simulation(self):
        """Test the is_simulation method."""
        # True for result with simulation flag
        self.assertTrue(self.validator.is_simulation(self.sample_results["bert-base-uncased"]["webgpu"]))
        
        # False for result without simulation flag
        self.assertFalse(self.validator.is_simulation(self.sample_results["bert-base-uncased"]["cpu"]))
    
    def test_validate_performance(self):
        """Test the validate_performance method."""
        # Test with valid simulation (cuda vs cpu)
        result = self.validator.validate_performance(
            self.sample_results["bert-base-uncased"]["cuda"],
            self.sample_results["bert-base-uncased"]["cpu"]
        )
        self.assertTrue(result["valid"])
        
        # Test with valid simulation (webgpu vs cpu)
        result = self.validator.validate_performance(
            self.sample_results["bert-base-uncased"]["webgpu"],
            self.sample_results["bert-base-uncased"]["cpu"]
        )
        self.assertTrue(result["valid"])
        
        # Test with invalid simulation (openvino vs cpu)
        # The ratio is too high compared to expected
        result = self.validator.validate_performance(
            self.sample_results["bert-base-uncased"]["openvino"],
            self.sample_results["bert-base-uncased"]["cpu"]
        )
        self.assertFalse(result["valid"])
    
    def test_validate_results(self):
        """Test the validate_results method."""
        result = self.validator.validate_results(self.sample_results)
        self.assertIn("validations", result)
        self.assertIn("simulations", result)
        self.assertIn("bert-base-uncased", result["validations"])
        
        # Check if webgpu is detected as simulation
        self.assertIn("webgpu", result["simulations"]["bert-base-uncased"])
        
        # Openvino should have failed validation
        self.assertFalse(
            result["validations"].get("bert-base-uncased", {}).get("openvino", {}).get("cpu", {}).get("valid", True)
        )

class TestCrossHardwareComparison(unittest.TestCase):
    """Test case for the CrossHardwareComparison class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.comparator = CrossHardwareComparison(output_dir=tempfile.mkdtemp())
        
        # Create sample test results
        self.sample_results = {
            "bert-base-uncased": {
                "cpu": {
                    "status": "success",
                    "metrics": {
                        "throughput": 100.0,
                        "latency": 10.0,
                        "memory": 1000.0
                    }
                },
                "cuda": {
                    "status": "success",
                    "metrics": {
                        "throughput": 350.0,
                        "latency": 2.85,
                        "memory": 2000.0
                    }
                },
                "webgpu": {
                    "status": "success",
                    "metrics": {
                        "throughput": 200.0,
                        "latency": 5.0,
                        "memory": 1500.0
                    }
                }
            },
            "t5-small": {
                "cpu": {
                    "status": "success",
                    "metrics": {
                        "throughput": 50.0,
                        "latency": 20.0,
                        "memory": 2000.0
                    }
                },
                "cuda": {
                    "status": "success",
                    "metrics": {
                        "throughput": 175.0,
                        "latency": 5.7,
                        "memory": 3000.0
                    }
                }
            }
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.comparator.output_dir)
    
    def test_determine_model_family(self):
        """Test the _determine_model_family method."""
        self.assertEqual("text-embedding", self.comparator._determine_model_family("bert-base-uncased"))
        self.assertEqual("text-generation", self.comparator._determine_model_family("t5-small"))
        self.assertEqual("vision", self.comparator._determine_model_family("vit-base"))
        self.assertEqual("audio", self.comparator._determine_model_family("whisper-tiny"))
        self.assertEqual("multimodal", self.comparator._determine_model_family("clip-vit"))
        self.assertEqual("unknown", self.comparator._determine_model_family("unknown-model"))
    
    def test_extract_performance_metrics(self):
        """Test the _extract_performance_metrics method."""
        metrics = self.comparator._extract_performance_metrics(self.sample_results["bert-base-uncased"]["cuda"])
        self.assertIn("throughput", metrics)
        self.assertIn("latency", metrics)
        self.assertIn("memory", metrics)
        self.assertEqual(350.0, metrics["throughput"])
        self.assertEqual(2.85, metrics["latency"])
    
    def test_compare_metrics(self):
        """Test the _compare_metrics method."""
        comparison = self.comparator._compare_metrics(
            self.sample_results["bert-base-uncased"]["cuda"]["metrics"],
            self.sample_results["bert-base-uncased"]["cpu"]["metrics"]
        )
        
        self.assertIn("throughput_ratio", comparison)
        self.assertIn("latency_ratio", comparison)
        self.assertAlmostEqual(3.5, comparison["throughput_ratio"], places=1)
        self.assertAlmostEqual(3.5, comparison["latency_ratio"], places=1)
    
    def test_find_best_hardware_for_family(self):
        """Test the _find_best_hardware_for_family method."""
        # Create a simple family comparison data for testing
        family_data = {
            "cuda": {
                "cpu": {"throughput_ratio": 3.5, "latency_ratio": 3.5, "overall_score": 3.5},
                "webgpu": {"throughput_ratio": 1.75, "latency_ratio": 1.75, "overall_score": 1.75}
            },
            "webgpu": {
                "cpu": {"throughput_ratio": 2.0, "latency_ratio": 2.0, "overall_score": 2.0}
            },
            "cpu": {}
        }
        
        best_hw = self.comparator._find_best_hardware_for_family(family_data)
        
        self.assertIn("overall", best_hw)
        self.assertIn("throughput", best_hw)
        self.assertIn("latency", best_hw)
        self.assertEqual("cuda", best_hw["overall"])
        self.assertEqual("cuda", best_hw["throughput"])
        self.assertEqual("cuda", best_hw["latency"])
    
    def test_generate_comparison(self):
        """Test the generate_comparison method."""
        result = self.comparator.generate_comparison(self.sample_results)
        
        self.assertIn("model_metrics", result)
        self.assertIn("hardware_comparison", result)
        self.assertIn("family_comparison", result)
        self.assertIn("optimal_hardware", result)
        self.assertIn("family_optimal_hardware", result)
        
        # Check if bert-base-uncased metrics were processed
        self.assertIn("bert-base-uncased", result["model_metrics"])
        
        # Check if cuda is identified as the best hardware for bert
        self.assertEqual("cuda", result["optimal_hardware"]["bert-base-uncased"]["best_throughput"])

if __name__ == "__main__":
    unittest.main()