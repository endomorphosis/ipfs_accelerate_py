#!/usr/bin/env python3
"""
Test for EnhancedStatisticalValidator.

This module tests the functionality of the EnhancedStatisticalValidator class,
which provides advanced statistical methods for validating simulation accuracy.
"""

import os
import sys
import logging
import unittest
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directories to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

from duckdb_api.simulation_validation.statistical.enhanced_statistical_validator import (
    EnhancedStatisticalValidator,
    get_enhanced_statistical_validator_instance
)

class TestEnhancedStatisticalValidator(unittest.TestCase):
    """Test cases for EnhancedStatisticalValidator."""
    
    def setUp(self):
        """Set up test data and resources."""
        # Create validator instance
        self.validator = EnhancedStatisticalValidator()
        
        # Create sample simulation and hardware results
        # Single result for basic validation
        self.sim_result = SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="rtx3080",
            metrics={
                "throughput_items_per_second": 95.0,
                "average_latency_ms": 105.0,
                "memory_peak_mb": 5250.0,
                "power_consumption_w": 210.0,
                "initialization_time_ms": 420.0,
                "warmup_time_ms": 210.0
            },
            batch_size=32,
            precision="fp16",
            timestamp=datetime.datetime.now().isoformat(),
            simulation_version="v1.0",
            additional_metadata={"configuration": "default"}
        )
        
        self.hw_result = HardwareResult(
            model_id="bert-base-uncased",
            hardware_id="rtx3080",
            metrics={
                "throughput_items_per_second": 100.0,
                "average_latency_ms": 100.0,
                "memory_peak_mb": 5000.0,
                "power_consumption_w": 200.0,
                "initialization_time_ms": 400.0,
                "warmup_time_ms": 200.0
            },
            batch_size=32,
            precision="fp16",
            timestamp=datetime.datetime.now().isoformat(),
            hardware_details={"gpu": "RTX 3080"},
            test_environment={"driver_version": "456.71"},
            additional_metadata={"measurement_method": "avg_of_5_runs"}
        )
        
        # Multiple results for batch validation and distribution comparison
        self.sim_results = [self.sim_result]
        self.hw_results = [self.hw_result]
        
        # Add more results for batch testing
        for i in range(5):
            # Create variations of the base results with some randomness
            sim_metrics = {
                "throughput_items_per_second": 95.0 + np.random.normal(0, 2),
                "average_latency_ms": 105.0 + np.random.normal(0, 2),
                "memory_peak_mb": 5250.0 + np.random.normal(0, 50),
                "power_consumption_w": 210.0 + np.random.normal(0, 5),
                "initialization_time_ms": 420.0 + np.random.normal(0, 10),
                "warmup_time_ms": 210.0 + np.random.normal(0, 5)
            }
            
            hw_metrics = {
                "throughput_items_per_second": 100.0 + np.random.normal(0, 2),
                "average_latency_ms": 100.0 + np.random.normal(0, 2),
                "memory_peak_mb": 5000.0 + np.random.normal(0, 50),
                "power_consumption_w": 200.0 + np.random.normal(0, 5),
                "initialization_time_ms": 400.0 + np.random.normal(0, 10),
                "warmup_time_ms": 200.0 + np.random.normal(0, 5)
            }
            
            sim_result = SimulationResult(
                model_id="bert-base-uncased",
                hardware_id="rtx3080",
                metrics=sim_metrics,
                batch_size=32,
                precision="fp16",
                timestamp=datetime.datetime.now().isoformat(),
                simulation_version="v1.0",
                additional_metadata={"configuration": "default"}
            )
            
            hw_result = HardwareResult(
                model_id="bert-base-uncased",
                hardware_id="rtx3080",
                metrics=hw_metrics,
                batch_size=32,
                precision="fp16",
                timestamp=datetime.datetime.now().isoformat(),
                hardware_details={"gpu": "RTX 3080"},
                test_environment={"driver_version": "456.71"},
                additional_metadata={"measurement_method": "avg_of_5_runs"}
            )
            
            self.sim_results.append(sim_result)
            self.hw_results.append(hw_result)
    
    def test_initialization(self):
        """Test validator initialization."""
        # Test default initialization
        validator = EnhancedStatisticalValidator()
        self.assertIsNotNone(validator)
        self.assertIsNotNone(validator.config)
        
        # Test initialization with custom config
        custom_config = {
            "metrics_to_validate": ["throughput_items_per_second", "average_latency_ms"],
            "error_metrics": ["mape", "rmse"],
            "confidence_intervals": {"enabled": False}
        }
        validator = EnhancedStatisticalValidator(custom_config)
        self.assertEqual(validator.config["metrics_to_validate"], custom_config["metrics_to_validate"])
        self.assertEqual(validator.config["error_metrics"], custom_config["error_metrics"])
        self.assertEqual(validator.config["confidence_intervals"]["enabled"], False)
    
    def test_basic_validation(self):
        """Test basic validation functionality."""
        # Validate single result
        validation_result = self.validator.validate(self.sim_result, self.hw_result)
        
        # Check basic validation functionality
        self.assertIsNotNone(validation_result)
        self.assertEqual(validation_result.simulation_result, self.sim_result)
        self.assertEqual(validation_result.hardware_result, self.hw_result)
        self.assertIsNotNone(validation_result.metrics_comparison)
        
        # Check metrics_comparison structure
        self.assertIn("throughput_items_per_second", validation_result.metrics_comparison)
        self.assertIn("mape", validation_result.metrics_comparison["throughput_items_per_second"])
        
        # Check MAPE calculation
        expected_mape = abs(95.0 - 100.0) / 100.0 * 100.0  # 5%
        self.assertAlmostEqual(
            validation_result.metrics_comparison["throughput_items_per_second"]["mape"],
            expected_mape,
            places=1
        )
    
    def test_enhanced_metrics(self):
        """Test enhanced metrics calculation."""
        # Validate single result
        validation_result = self.validator.validate(self.sim_result, self.hw_result)
        
        # Check additional metrics were calculated
        self.assertIsNotNone(validation_result.additional_metrics)
        self.assertIn("enhanced_metrics", validation_result.additional_metrics)
        
        # Check enhanced metrics structure for throughput
        enhanced_metrics = validation_result.additional_metrics["enhanced_metrics"]
        self.assertIn("throughput_items_per_second", enhanced_metrics)
        
        # Check specific enhanced metrics
        throughput_metrics = enhanced_metrics["throughput_items_per_second"]
        self.assertIn("mae", throughput_metrics)
        self.assertIn("mse", throughput_metrics)
        self.assertIn("bias", throughput_metrics)
        
        # Check MAE calculation
        expected_mae = abs(95.0 - 100.0)  # 5.0
        self.assertAlmostEqual(throughput_metrics["mae"], expected_mae, places=1)
        
        # Check MSE calculation
        expected_mse = (95.0 - 100.0) ** 2  # 25.0
        self.assertAlmostEqual(throughput_metrics["mse"], expected_mse, places=1)
        
        # Check bias calculation
        expected_bias = 95.0 - 100.0  # -5.0
        self.assertAlmostEqual(throughput_metrics["bias"], expected_bias, places=1)
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        # Validate single result
        validation_result = self.validator.validate(self.sim_result, self.hw_result)
        
        # Check confidence intervals were calculated
        self.assertIsNotNone(validation_result.additional_metrics)
        self.assertIn("confidence_intervals", validation_result.additional_metrics)
        
        # Check confidence interval structure
        ci_data = validation_result.additional_metrics["confidence_intervals"]
        self.assertIn("throughput_items_per_second", ci_data)
        self.assertIn("mape", ci_data["throughput_items_per_second"])
        
        # Check confidence interval properties
        ci_mape = ci_data["throughput_items_per_second"]["mape"]
        self.assertIn("value", ci_mape)
        self.assertIn("lower_bound", ci_mape)
        self.assertIn("upper_bound", ci_mape)
        self.assertIn("confidence_level", ci_mape)
        
        # Check confidence level
        self.assertAlmostEqual(ci_mape["confidence_level"], 0.95, places=2)
        
        # Check bounds are reasonable
        self.assertLess(ci_mape["lower_bound"], ci_mape["value"])
        self.assertGreater(ci_mape["upper_bound"], ci_mape["value"])
    
    def test_bland_altman_analysis(self):
        """Test Bland-Altman analysis."""
        # Validate single result
        validation_result = self.validator.validate(self.sim_result, self.hw_result)
        
        # Check Bland-Altman analysis was performed
        self.assertIsNotNone(validation_result.additional_metrics)
        self.assertIn("bland_altman", validation_result.additional_metrics)
        
        # Check Bland-Altman structure
        ba_data = validation_result.additional_metrics["bland_altman"]
        self.assertIn("throughput_items_per_second", ba_data)
        
        # Check Bland-Altman metrics
        ba_throughput = ba_data["throughput_items_per_second"]
        self.assertIn("bias", ba_throughput)
        self.assertIn("lower_loa", ba_throughput)
        self.assertIn("upper_loa", ba_throughput)
        
        # Check bias calculation
        expected_bias = 95.0 - 100.0  # -5.0
        self.assertAlmostEqual(ba_throughput["bias"], expected_bias, places=1)
        
        # Check limits of agreement are reasonable
        self.assertLess(ba_throughput["lower_loa"], ba_throughput["bias"])
        self.assertGreater(ba_throughput["upper_loa"], ba_throughput["bias"])
    
    def test_batch_validation(self):
        """Test batch validation functionality."""
        # Validate batch of results
        validation_results = self.validator.validate_batch(self.sim_results, self.hw_results)
        
        # Check batch validation results
        self.assertIsNotNone(validation_results)
        self.assertGreaterEqual(len(validation_results), len(self.sim_results))
        
        # Check individual validations
        for validation_result in validation_results:
            self.assertIsNotNone(validation_result.metrics_comparison)
            self.assertIn("throughput_items_per_second", validation_result.metrics_comparison)
    
    def test_power_analysis(self):
        """Test power analysis for batch validation."""
        # Validate batch of results
        validation_results = self.validator.validate_batch(self.sim_results, self.hw_results)
        
        # Check power analysis was performed
        self.assertGreater(len(validation_results), 0)
        self.assertIsNotNone(validation_results[0].additional_metrics)
        self.assertIn("power_analysis", validation_results[0].additional_metrics)
        
        # Check power analysis structure
        power_analysis = validation_results[0].additional_metrics["power_analysis"]
        self.assertIn("0.2", power_analysis)  # Small effect size
        self.assertIn("0.5", power_analysis)  # Medium effect size
        self.assertIn("0.8", power_analysis)  # Large effect size
        
        # Check power analysis metrics
        small_effect = power_analysis["0.2"]
        self.assertIn("power", small_effect)
        self.assertIn("effect_size", small_effect)
        self.assertIn("sample_size", small_effect)
        self.assertIn("is_sufficient", small_effect)
        
        # Check sample size
        self.assertEqual(small_effect["sample_size"], len(self.sim_results))
    
    def test_summarize_validation(self):
        """Test validation summary generation."""
        # Validate batch of results
        validation_results = self.validator.validate_batch(self.sim_results, self.hw_results)
        
        # Generate summary
        summary = self.validator.summarize_validation(validation_results)
        
        # Check summary structure
        self.assertIsNotNone(summary)
        self.assertIn("metrics", summary)
        self.assertIn("models", summary)
        self.assertIn("hardware", summary)
        self.assertIn("overall", summary)
        self.assertIn("enhanced", summary)
        
        # Check enhanced metrics summary
        enhanced_summary = summary["enhanced"]
        self.assertIn("enhanced_metrics", enhanced_summary)
        self.assertIn("confidence_intervals", enhanced_summary)
        self.assertIn("bland_altman", enhanced_summary)
        self.assertIn("power_analysis", enhanced_summary)
    
    def test_factory_function(self):
        """Test the factory function for creating validator instances."""
        # Use factory function to create instance
        validator = get_enhanced_statistical_validator_instance()
        
        # Check instance type
        self.assertIsInstance(validator, EnhancedStatisticalValidator)
        
        # Test with custom config file (non-existent)
        validator = get_enhanced_statistical_validator_instance("nonexistent_config.json")
        self.assertIsInstance(validator, EnhancedStatisticalValidator)


if __name__ == "__main__":
    unittest.main()