#!/usr/bin/env python3
"""
Template Validation Module for End-to-End Testing Framework

This module provides tools to validate model templates and compare test results
against expected values.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import project utilities (assuming they exist)
try:
    from simple_utils import setup_logging
except ImportError:
    # Define a simple setup_logging function if the import fails
    def setup_logging(logger, level=logging.INFO):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

# Set up logging
logger = logging.getLogger(__name__)
setup_logging(logger)

class ModelValidator:
    """Validates model templates and generated code for correctness."""
    
    def __init__(self, model_name: str, hardware: str, 
                 template_path: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the model validator.
        
        Args:
            model_name: Name of the model being validated
            hardware: Hardware platform the model is running on
            template_path: Path to the template directory (optional)
            verbose: Whether to output verbose logs
        """
        self.model_name = model_name
        self.hardware = hardware
        self.template_path = template_path
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
    def validate_skill(self, skill_path: str) -> Dict[str, Any]:
        """
        Validate a generated skill file.
        
        Args:
            skill_path: Path to the generated skill file
            
        Returns:
            Dictionary with validation results
        """
        logger.debug(f"Validating skill: {skill_path}")
        
        if not os.path.exists(skill_path):
            return {"valid": False, "error": f"Skill file not found: {skill_path}"}
        
        # Analyze the skill file
        # This is a placeholder for actual validation logic
        try:
            # Simple check: read the file and make sure it's valid Python code
            with open(skill_path, 'r') as f:
                skill_content = f.read()
                
            # Check for required elements in the skill
            validation_results = {
                "valid": True,
                "warnings": [],
                "missing_elements": []
            }
            
            # Check for class definition
            expected_class_name = f"{self.model_name.replace('-', '_').title()}Skill"
            if expected_class_name not in skill_content:
                validation_results["missing_elements"].append(f"Missing expected class: {expected_class_name}")
                validation_results["valid"] = False
            
            # Check for required methods
            required_methods = ["setup", "run"]
            for method in required_methods:
                if f"def {method}" not in skill_content:
                    validation_results["missing_elements"].append(f"Missing required method: {method}")
                    validation_results["valid"] = False
            
            # Check for hardware-specific code
            if self.hardware not in skill_content:
                validation_results["warnings"].append(f"No hardware-specific code found for {self.hardware}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating skill {skill_path}: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def validate_test(self, test_path: str) -> Dict[str, Any]:
        """
        Validate a generated test file.
        
        Args:
            test_path: Path to the generated test file
            
        Returns:
            Dictionary with validation results
        """
        logger.debug(f"Validating test: {test_path}")
        
        if not os.path.exists(test_path):
            return {"valid": False, "error": f"Test file not found: {test_path}"}
        
        # Analyze the test file
        # This is a placeholder for actual validation logic
        try:
            # Simple check: read the file and make sure it's valid Python code
            with open(test_path, 'r') as f:
                test_content = f.read()
                
            # Check for required elements in the test
            validation_results = {
                "valid": True,
                "warnings": [],
                "missing_elements": []
            }
            
            # Check for unittest imports
            if "import unittest" not in test_content:
                validation_results["missing_elements"].append("Missing unittest import")
                validation_results["valid"] = False
            
            # Check for test class definition
            expected_class_name = f"Test{self.model_name.replace('-', '_').title()}"
            if expected_class_name not in test_content:
                validation_results["missing_elements"].append(f"Missing expected test class: {expected_class_name}")
                validation_results["valid"] = False
            
            # Check for essential test methods
            if "test_" not in test_content:
                validation_results["missing_elements"].append("No test methods found (should start with 'test_')")
                validation_results["valid"] = False
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating test {test_path}: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def validate_benchmark(self, benchmark_path: str) -> Dict[str, Any]:
        """
        Validate a generated benchmark file.
        
        Args:
            benchmark_path: Path to the generated benchmark file
            
        Returns:
            Dictionary with validation results
        """
        logger.debug(f"Validating benchmark: {benchmark_path}")
        
        if not os.path.exists(benchmark_path):
            return {"valid": False, "error": f"Benchmark file not found: {benchmark_path}"}
        
        # Analyze the benchmark file
        # This is a placeholder for actual validation logic
        try:
            # Simple check: read the file and make sure it's valid Python code
            with open(benchmark_path, 'r') as f:
                benchmark_content = f.read()
                
            # Check for required elements in the benchmark
            validation_results = {
                "valid": True,
                "warnings": [],
                "missing_elements": []
            }
            
            # Check for required imports
            required_imports = ["time", "json"]
            for imp in required_imports:
                if f"import {imp}" not in benchmark_content:
                    validation_results["missing_elements"].append(f"Missing required import: {imp}")
                    validation_results["valid"] = False
            
            # Check for benchmark function
            if "def benchmark" not in benchmark_content:
                validation_results["missing_elements"].append("Missing benchmark function")
                validation_results["valid"] = False
            
            # Check for performance metrics
            if "latency" not in benchmark_content and "throughput" not in benchmark_content:
                validation_results["warnings"].append("No performance metrics found in benchmark")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating benchmark {benchmark_path}: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def validate_all(self, skill_path: str, test_path: str, benchmark_path: str) -> Dict[str, Any]:
        """
        Validate all components together.
        
        Args:
            skill_path: Path to the generated skill file
            test_path: Path to the generated test file
            benchmark_path: Path to the generated benchmark file
            
        Returns:
            Dictionary with combined validation results
        """
        skill_validation = self.validate_skill(skill_path)
        test_validation = self.validate_test(test_path)
        benchmark_validation = self.validate_benchmark(benchmark_path)
        
        # Combine results
        all_valid = (
            skill_validation.get("valid", False) and 
            test_validation.get("valid", False) and 
            benchmark_validation.get("valid", False)
        )
        
        return {
            "valid": all_valid,
            "skill": skill_validation,
            "test": test_validation,
            "benchmark": benchmark_validation
        }


class ResultComparer:
    """
    Enhanced result comparison tool with support for tensor data, specialized numeric comparison,
    and configurable tolerance levels for different data types.
    """
    
    def __init__(self, 
                 tolerance: float = 0.1, 
                 tensor_rtol: float = 1e-5, 
                 tensor_atol: float = 1e-5,
                 tensor_comparison_mode: str = 'auto'):
        """
        Initialize the result comparer with enhanced capabilities.
        
        Args:
            tolerance: General tolerance for numeric comparisons (as a percentage)
            tensor_rtol: Relative tolerance for tensor comparison
            tensor_atol: Absolute tolerance for tensor comparison
            tensor_comparison_mode: Mode for tensor comparison ('auto', 'exact', 'statistical')
        """
        self.tolerance = tolerance
        self.tensor_rtol = tensor_rtol
        self.tensor_atol = tensor_atol
        self.tensor_comparison_mode = tensor_comparison_mode
        
        # Initialize numpy if available for better tensor comparison
        self.has_numpy = True
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.has_numpy = False
    
    def compare(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare expected and actual results.
        
        Args:
            expected: Expected result dictionary
            actual: Actual result dictionary
            
        Returns:
            Dictionary with comparison results
        """
        if not expected or not actual:
            return {"matches": False, "reason": "Missing expected or actual results"}
        
        differences = {}
        
        # Compare all fields in expected results
        for key in expected:
            if key not in actual:
                differences[key] = {"type": "missing_field", "expected": expected[key], "actual": None}
                continue
                
            if isinstance(expected[key], dict) and isinstance(actual[key], dict):
                # Recursively compare nested dictionaries
                nested_comparison = self.compare(expected[key], actual[key])
                if not nested_comparison.get("matches", False) and "differences" in nested_comparison:
                    differences[key] = nested_comparison["differences"]
            
            elif isinstance(expected[key], (int, float)) and isinstance(actual[key], (int, float)):
                # Compare numeric values with tolerance
                expected_val = float(expected[key])
                actual_val = float(actual[key])
                
                # Skip comparison if expected value is zero (to avoid division by zero)
                if expected_val == 0:
                    if actual_val != 0:
                        differences[key] = {
                            "type": "value_mismatch",
                            "expected": expected_val,
                            "actual": actual_val
                        }
                else:
                    # Calculate relative difference
                    rel_diff = abs(expected_val - actual_val) / abs(expected_val)
                    
                    if rel_diff > self.tolerance:
                        differences[key] = {
                            "type": "value_mismatch",
                            "expected": expected_val,
                            "actual": actual_val,
                            "relative_difference": rel_diff,
                            "tolerance": self.tolerance
                        }
            
            elif isinstance(expected[key], list) and isinstance(actual[key], list):
                # Compare lists
                if len(expected[key]) != len(actual[key]):
                    differences[key] = {
                        "type": "list_length_mismatch",
                        "expected_length": len(expected[key]),
                        "actual_length": len(actual[key])
                    }
                else:
                    # Try to compare as numpy arrays if possible
                    try:
                        expected_arr = np.array(expected[key], dtype=float)
                        actual_arr = np.array(actual[key], dtype=float)
                        
                        # Compare arrays with tolerance
                        if not np.allclose(expected_arr, actual_arr, rtol=self.tolerance, atol=self.tolerance*np.abs(expected_arr).mean()):
                            # Find the locations of differences
                            diff_indices = np.where(~np.isclose(expected_arr, actual_arr, rtol=self.tolerance, atol=self.tolerance*np.abs(expected_arr).mean()))[0]
                            diff_examples = {int(i): {"expected": float(expected_arr[i]), "actual": float(actual_arr[i])} 
                                            for i in diff_indices[:5]}  # Show first 5 differences
                            
                            differences[key] = {
                                "type": "array_mismatch",
                                "total_elements": len(expected_arr),
                                "different_elements": len(diff_indices),
                                "examples": diff_examples
                            }
                    except (ValueError, TypeError):
                        # If not numeric arrays, compare element by element
                        list_diffs = []
                        for i, (exp_item, act_item) in enumerate(zip(expected[key], actual[key])):
                            if exp_item != act_item:
                                list_diffs.append((i, exp_item, act_item))
                        
                        if list_diffs:
                            differences[key] = {
                                "type": "list_content_mismatch",
                                "different_items": len(list_diffs),
                                "examples": {idx: {"expected": exp, "actual": act} 
                                            for idx, exp, act in list_diffs[:5]}  # Show first 5 differences
                            }
            
            elif expected[key] != actual[key]:
                # Simple equality comparison for other types
                differences[key] = {
                    "type": "value_mismatch",
                    "expected": expected[key],
                    "actual": actual[key]
                }
        
        # Check for additional fields in actual results
        for key in actual:
            if key not in expected:
                differences[f"unexpected_{key}"] = {
                    "type": "unexpected_field",
                    "expected": None,
                    "actual": actual[key]
                }
        
        return {
            "matches": len(differences) == 0,
            "differences": differences
        }
    
    def compare_with_file(self, expected_path: str, actual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare actual results with expected results from a file.
        
        Args:
            expected_path: Path to the expected results file (JSON)
            actual: Actual result dictionary
            
        Returns:
            Dictionary with comparison results
        """
        if not os.path.exists(expected_path):
            return {"matches": False, "reason": f"Expected results file not found: {expected_path}"}
        
        try:
            with open(expected_path, 'r') as f:
                expected = json.load(f)
            
            return self.compare(expected, actual)
            
        except Exception as e:
            logger.error(f"Error comparing results with file {expected_path}: {str(e)}")
            return {"matches": False, "reason": f"Comparison error: {str(e)}"}
    
    def statistical_tensor_compare(self, expected_data: List[float], actual_data: List[float]) -> Dict[str, Any]:
        """
        Perform statistical comparison of tensor-like data.
        
        Args:
            expected_data: Expected tensor data as flat list
            actual_data: Actual tensor data as flat list
            
        Returns:
            Dictionary with statistical comparison results
        """
        if not self.has_numpy:
            return {
                "matches": False, 
                "reason": "statistical comparison requires numpy"
            }
            
        try:
            # Convert to numpy arrays
            exp_arr = self.np.array(expected_data, dtype=float)
            act_arr = self.np.array(actual_data, dtype=float)
            
            # Basic statistics
            exp_mean = float(self.np.mean(exp_arr))
            act_mean = float(self.np.mean(act_arr))
            exp_std = float(self.np.std(exp_arr))
            act_std = float(self.np.std(act_arr))
            
            # Calculate relative differences in statistics
            mean_rel_diff = abs(exp_mean - act_mean) / (abs(exp_mean) if abs(exp_mean) > 1e-10 else 1.0)
            std_rel_diff = abs(exp_std - act_std) / (abs(exp_std) if abs(exp_std) > 1e-10 else 1.0)
            
            # Check if distributions are similar enough
            stats_match = mean_rel_diff <= self.tolerance and std_rel_diff <= self.tolerance
            
            # Additional metrics for large tensors
            if len(exp_arr) > 100:
                # Percentiles
                exp_p50 = float(self.np.percentile(exp_arr, 50))
                act_p50 = float(self.np.percentile(act_arr, 50))
                exp_p95 = float(self.np.percentile(exp_arr, 95))
                act_p95 = float(self.np.percentile(act_arr, 95))
                
                # Check percentile differences
                p50_rel_diff = abs(exp_p50 - act_p50) / (abs(exp_p50) if abs(exp_p50) > 1e-10 else 1.0)
                p95_rel_diff = abs(exp_p95 - act_p95) / (abs(exp_p95) if abs(exp_p95) > 1e-10 else 1.0)
                
                # Update match status with percentile comparison
                stats_match = stats_match and p50_rel_diff <= self.tolerance and p95_rel_diff <= self.tolerance
                
                return {
                    "matches": stats_match,
                    "statistics": {
                        "mean": {"expected": exp_mean, "actual": act_mean, "rel_diff": mean_rel_diff},
                        "std": {"expected": exp_std, "actual": act_std, "rel_diff": std_rel_diff},
                        "p50": {"expected": exp_p50, "actual": act_p50, "rel_diff": p50_rel_diff},
                        "p95": {"expected": exp_p95, "actual": act_p95, "rel_diff": p95_rel_diff}
                    }
                }
            else:
                return {
                    "matches": stats_match,
                    "statistics": {
                        "mean": {"expected": exp_mean, "actual": act_mean, "rel_diff": mean_rel_diff},
                        "std": {"expected": exp_std, "actual": act_std, "rel_diff": std_rel_diff}
                    }
                }
        except Exception as e:
            return {
                "matches": False,
                "reason": f"statistical comparison error: {str(e)}"
            }
            
    def deep_compare_tensors(self, expected: Dict[str, Any], actual: Dict[str, Any], 
                            rtol: float = None, atol: float = None) -> Dict[str, Any]:
        """
        Special comparison for tensor outputs with more advanced tolerance settings.
        
        Args:
            expected: Expected result dictionary with tensor data
            actual: Actual result dictionary with tensor data
            rtol: Relative tolerance for tensor comparison (overrides default)
            atol: Absolute tolerance for tensor comparison (overrides default)
            
        Returns:
            Dictionary with comparison results
        """
        # Use instance defaults if not specified
        rtol = rtol if rtol is not None else self.tensor_rtol
        atol = atol if atol is not None else self.tensor_atol
        differences = {}
        
        # Find all tensor keys in expected and actual
        expected_tensor_keys = []
        actual_tensor_keys = []
        
        def find_tensor_keys(obj, path="", keys_list=None):
            if keys_list is None:
                keys_list = []
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    if isinstance(v, dict) and any(tensor_key in v for tensor_key in ["data", "shape", "dtype"]):
                        keys_list.append(new_path)
                    else:
                        find_tensor_keys(v, new_path, keys_list)
            
            return keys_list
        
        expected_tensor_keys = find_tensor_keys(expected)
        actual_tensor_keys = find_tensor_keys(actual)
        
        # Compare tensor sets
        missing_tensors = set(expected_tensor_keys) - set(actual_tensor_keys)
        unexpected_tensors = set(actual_tensor_keys) - set(expected_tensor_keys)
        common_tensors = set(expected_tensor_keys).intersection(set(actual_tensor_keys))
        
        if missing_tensors:
            differences["missing_tensors"] = list(missing_tensors)
        
        if unexpected_tensors:
            differences["unexpected_tensors"] = list(unexpected_tensors)
        
        # Compare common tensors
        for tensor_key in common_tensors:
            # Navigate to the tensor in each dictionary
            path_parts = tensor_key.split(".")
            exp_tensor = expected
            act_tensor = actual
            
            try:
                for part in path_parts:
                    exp_tensor = exp_tensor[part]
                    act_tensor = act_tensor[part]
                
                # Now compare the tensors
                # This assumes the tensors are represented as dictionaries with data, shape, and dtype
                exp_shape = exp_tensor.get("shape")
                act_shape = act_tensor.get("shape")
                
                if exp_shape != act_shape:
                    if tensor_key not in differences:
                        differences[tensor_key] = {}
                    differences[tensor_key]["shape_mismatch"] = {
                        "expected": exp_shape,
                        "actual": act_shape
                    }
                    continue
                
                # Compare data
                exp_data = exp_tensor.get("data")
                act_data = act_tensor.get("data")
                
                if isinstance(exp_data, list) and isinstance(act_data, list):
                    try:
                        exp_arr = np.array(exp_data)
                        act_arr = np.array(act_data)
                        
                        if not np.allclose(exp_arr, act_arr, rtol=rtol, atol=atol):
                            max_diff = np.max(np.abs(exp_arr - act_arr))
                            mean_diff = np.mean(np.abs(exp_arr - act_arr))
                            
                            if tensor_key not in differences:
                                differences[tensor_key] = {}
                            differences[tensor_key]["data_mismatch"] = {
                                "max_difference": float(max_diff),
                                "mean_difference": float(mean_diff),
                                "rtol": rtol,
                                "atol": atol
                            }
                            
                            # Add examples of differences
                            diff_indices = np.where(~np.isclose(exp_arr, act_arr, rtol=rtol, atol=atol))
                            if diff_indices[0].size > 0:
                                examples = {}
                                for i in range(min(5, diff_indices[0].size)):
                                    idx = tuple(d[i] for d in diff_indices)
                                    examples[str(idx)] = {
                                        "expected": float(exp_arr[idx]),
                                        "actual": float(act_arr[idx]),
                                        "difference": float(act_arr[idx] - exp_arr[idx])
                                    }
                                differences[tensor_key]["difference_examples"] = examples
                    
                    except (ValueError, TypeError) as e:
                        if tensor_key not in differences:
                            differences[tensor_key] = {}
                        differences[tensor_key]["comparison_error"] = str(e)
            
            except (KeyError, TypeError) as e:
                if tensor_key not in differences:
                    differences[tensor_key] = {}
                differences[tensor_key]["navigation_error"] = str(e)
        
        return {
            "matches": len(differences) == 0,
            "differences": differences
        }


if __name__ == "__main__":
    # Simple test for the ResultComparer
    expected = {
        "output": {"value": 10.0},
        "metrics": {
            "latency_ms": 15.5,
            "throughput": 80.0,
            "memory_mb": 512
        }
    }
    
    actual = {
        "output": {"value": 10.2},  # Within 10% tolerance
        "metrics": {
            "latency_ms": 18.0,  # More than 10% different
            "throughput": 82.0,
            "memory_mb": 520
        }
    }
    
    comparer = ResultComparer(tolerance=0.1)
    result = comparer.compare(expected, actual)
    
    print(json.dumps(result, indent=2))