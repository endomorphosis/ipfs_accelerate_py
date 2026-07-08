#!/usr/bin/env python3
"""
Comprehensive test suite for the Simulation Accuracy and Validation Framework.

This script provides an end-to-end test of the Simulation Accuracy and Validation Framework,
including validation, calibration, drift detection, and visualization components.
"""

import os
import sys
import argparse
import logging
import datetime
import json
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_validation_suite")

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from data.duckdb.simulation_validation.simulation_validation_framework import get_framework_instance
from data.duckdb.simulation_validation.core.base import (
    SimulationResult, 
    HardwareResult, 
    ValidationResult
)

# Optional imports for testing specific components
try:
    from data.duckdb.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
    comparison_pipeline_available = True
except ImportError:
    comparison_pipeline_available = False
    logger.warning("ComparisonPipeline not available, some tests will be skipped")

try:
    from data.duckdb.simulation_validation.statistical.statistical_validator import StatisticalValidator
    statistical_validator_available = True
except ImportError:
    statistical_validator_available = False
    logger.warning("StatisticalValidator not available, some tests will be skipped")

try:
    from data.duckdb.simulation_validation.calibration.basic_calibrator import BasicSimulationCalibrator
    from data.duckdb.simulation_validation.calibration.advanced_calibrator import AdvancedSimulationCalibrator
    calibrator_available = True
except ImportError:
    calibrator_available = False
    logger.warning("Calibrators not available, calibration tests will be skipped")

try:
    from data.duckdb.simulation_validation.drift_detection.basic_detector import BasicDriftDetector
    from data.duckdb.simulation_validation.drift_detection.advanced_detector import AdvancedDriftDetector
    drift_detector_available = True
except ImportError:
    drift_detector_available = False
    logger.warning("DriftDetectors not available, drift detection tests will be skipped")

try:
    from data.duckdb.simulation_validation.visualization.validation_visualizer import ValidationVisualizer
    visualizer_available = True
except ImportError:
    visualizer_available = False
    logger.warning("ValidationVisualizer not available, visualization tests will be skipped")

try:
    from data.duckdb.simulation_validation.methodology import ValidationMethodology
    methodology_available = True
except ImportError:
    methodology_available = False
    logger.warning("ValidationMethodology not available, methodology tests will be skipped")


class TestUtils:
    """Utility functions for testing the framework."""

    @staticmethod
    def generate_sample_data(num_samples=3, add_bias=False, bias_multiplier=1.0):
        """
        Generate sample simulation and hardware results for testing.
        
        Args:
            num_samples: Number of samples to generate for each hardware/model combination
            add_bias: Whether to add systematic bias to simulation results
            bias_multiplier: Multiplier for bias amount (higher means more bias)
            
        Returns:
            Tuple of (simulation_results, hardware_results)
        """
        # Hardware types and models
        hardware_ids = ["cpu_intel_xeon", "gpu_rtx3080", "webgpu_chrome"]
        model_ids = ["bert-base-uncased", "vit-base-patch16-224", "llama-7b"]
        batch_sizes = [1, 4, 16]
        precisions = ["fp32", "fp16"]
        
        # Generate timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Lists to store results
        simulation_results = []
        hardware_results = []
        
        # Generate results for each combination
        for hardware_id in hardware_ids:
            for model_id in model_ids:
                for batch_size in batch_sizes:
                    for precision in precisions:
                        # Define base metrics for this combination
                        if "gpu" in hardware_id:
                            throughput_base = 200 if model_id == "bert-base-uncased" else 150
                            latency_base = 20 if model_id == "bert-base-uncased" else 30
                            memory_base = 1000 if model_id == "bert-base-uncased" else 800
                            power_base = 150
                        elif "webgpu" in hardware_id:
                            throughput_base = 100 if model_id == "bert-base-uncased" else 80
                            latency_base = 40 if model_id == "bert-base-uncased" else 50
                            memory_base = 800 if model_id == "bert-base-uncased" else 600
                            power_base = 100
                        else:  # CPU
                            throughput_base = 50 if model_id == "bert-base-uncased" else 40
                            latency_base = 80 if model_id == "bert-base-uncased" else 100
                            memory_base = 500 if model_id == "bert-base-uncased" else 400
                            power_base = 80
                        
                        # Apply batch size scaling
                        throughput_scale = batch_size ** 0.8
                        latency_scale = batch_size ** 0.2
                        memory_scale = batch_size ** 0.5
                        
                        # Apply precision scaling
                        if precision == "fp16":
                            throughput_scale *= 1.3
                            memory_scale *= 0.6
                        
                        # Generate multiple samples
                        for i in range(num_samples):
                            # Hardware sample with random variation
                            hw_throughput = throughput_base * throughput_scale * (1 + np.random.normal(0, 0.05))
                            hw_latency = latency_base * latency_scale * (1 + np.random.normal(0, 0.05))
                            hw_memory = memory_base * memory_scale * (1 + np.random.normal(0, 0.03))
                            hw_power = power_base * (1 + np.random.normal(0, 0.05))
                            
                            # Create hardware result
                            hw_result = HardwareResult(
                                model_id=model_id,
                                hardware_id=hardware_id,
                                metrics={
                                    "throughput_items_per_second": hw_throughput,
                                    "average_latency_ms": hw_latency,
                                    "memory_peak_mb": hw_memory,
                                    "power_consumption_w": hw_power
                                },
                                batch_size=batch_size,
                                precision=precision,
                                timestamp=timestamp,
                                hardware_details={
                                    "device_name": hardware_id.replace("_", " ").title(),
                                    "cores": 8 if "cpu" in hardware_id else 0,
                                    "compute_units": 0 if "cpu" in hardware_id else 68,
                                    "driver_version": "450.80.02" if "gpu" in hardware_id else "N/A",
                                    "browser_version": "91.0.4472.124" if "webgpu" in hardware_id else "N/A"
                                },
                                test_environment={
                                    "os": "Linux",
                                    "python_version": "3.8.10",
                                    "temperature_c": 45 + np.random.normal(0, 5),
                                    "background_load": np.random.uniform(0, 15)
                                }
                            )
                            hardware_results.append(hw_result)
                            
                            # Define bias factors if needed
                            if add_bias:
                                # Apply different biases by hardware type
                                if "gpu" in hardware_id:
                                    throughput_bias = 1.2 * bias_multiplier
                                    latency_bias = 0.8 * bias_multiplier
                                    memory_bias = 0.85 * bias_multiplier
                                    power_bias = 0.9 * bias_multiplier
                                elif "webgpu" in hardware_id:
                                    throughput_bias = 1.3 * bias_multiplier
                                    latency_bias = 0.7 * bias_multiplier
                                    memory_bias = 0.8 * bias_multiplier
                                    power_bias = 0.85 * bias_multiplier
                                else:  # CPU
                                    throughput_bias = 1.1 * bias_multiplier
                                    latency_bias = 0.9 * bias_multiplier
                                    memory_bias = 0.95 * bias_multiplier
                                    power_bias = 0.92 * bias_multiplier
                            else:
                                # Slightly biased by default
                                throughput_bias = 1.05
                                latency_bias = 0.95
                                memory_bias = 0.97
                                power_bias = 0.98
                            
                            # Simulation sample with bias
                            sim_throughput = hw_throughput * throughput_bias * (1 + np.random.normal(0, 0.1))
                            sim_latency = hw_latency * latency_bias * (1 + np.random.normal(0, 0.1))
                            sim_memory = hw_memory * memory_bias * (1 + np.random.normal(0, 0.08))
                            sim_power = hw_power * power_bias * (1 + np.random.normal(0, 0.1))
                            
                            # Create simulation result
                            sim_result = SimulationResult(
                                model_id=model_id,
                                hardware_id=hardware_id,
                                metrics={
                                    "throughput_items_per_second": sim_throughput,
                                    "average_latency_ms": sim_latency,
                                    "memory_peak_mb": sim_memory,
                                    "power_consumption_w": sim_power
                                },
                                batch_size=batch_size,
                                precision=precision,
                                timestamp=timestamp,
                                simulation_version="v1.2.3",
                                additional_metadata={
                                    "simulation_engine": "ipfs_accelerate_sim",
                                    "simulation_parameters": {
                                        "throughput_scale": throughput_bias,
                                        "latency_scale": latency_bias,
                                        "memory_scale": memory_bias,
                                        "power_scale": power_bias
                                    }
                                }
                            )
                            simulation_results.append(sim_result)
        
        return simulation_results, hardware_results

    @staticmethod
    def generate_time_series_data(num_days=30, day_interval=1, add_drift=False, drift_start_day=15):
        """
        Generate time series validation results for drift detection tests.
        
        Args:
            num_days: Number of days to generate data for
            day_interval: Interval between data points in days
            add_drift: Whether to add drift after a certain point
            drift_start_day: Day on which to start adding drift
            
        Returns:
            List of validation results with timestamps
        """
        validation_results = []
        
        # Hardware/model combinations for time series
        combinations = [
            ("gpu_rtx3080", "bert-base-uncased"),
            ("gpu_rtx3080", "vit-base-patch16-224"),
            ("cpu_intel_xeon", "bert-base-uncased")
        ]
        
        base_timestamp = datetime.datetime.now() - datetime.timedelta(days=num_days)
        
        for day in range(0, num_days, day_interval):
            timestamp = base_timestamp + datetime.timedelta(days=day)
            
            for hw_id, model_id in combinations:
                # Base metrics with small random variation
                if "gpu" in hw_id:
                    base_throughput = 200 if model_id == "bert-base-uncased" else 150
                    base_latency = 20 if model_id == "bert-base-uncased" else 30
                    base_memory = 1000 if model_id == "bert-base-uncased" else 800
                    base_power = 150
                else:  # CPU
                    base_throughput = 50 if model_id == "bert-base-uncased" else 40
                    base_latency = 80 if model_id == "bert-base-uncased" else 100
                    base_memory = 500 if model_id == "bert-base-uncased" else 400
                    base_power = 80
                
                # Add random variation
                random_factor = 1.0 + np.random.normal(0, 0.05)
                
                # Add drift if enabled and past the drift start day
                if add_drift and day >= drift_start_day:
                    drift_days = day - drift_start_day
                    drift_factor = 1.0 + (drift_days / 100)  # Gradual drift 1% per day after start
                else:
                    drift_factor = 1.0
                
                # Hardware metrics
                hw_throughput = base_throughput * random_factor
                hw_latency = base_latency * random_factor
                hw_memory = base_memory * random_factor
                hw_power = base_power * random_factor
                
                hw_result = HardwareResult(
                    model_id=model_id,
                    hardware_id=hw_id,
                    metrics={
                        "throughput_items_per_second": hw_throughput,
                        "average_latency_ms": hw_latency,
                        "memory_peak_mb": hw_memory,
                        "power_consumption_w": hw_power
                    },
                    batch_size=16,
                    precision="fp16",
                    timestamp=timestamp.isoformat(),
                    hardware_details={"type": hw_id.split("_")[0]},
                    test_environment={"temperature_c": 45 + np.random.normal(0, 5)}
                )
                
                # Simulation metrics with bias and drift
                if "gpu" in hw_id:
                    throughput_bias = 1.1 * drift_factor
                    latency_bias = 0.9 / drift_factor if drift_factor > 0 else 0.9
                    memory_bias = 0.95 * drift_factor
                    power_bias = 0.92 * drift_factor
                else:  # CPU
                    throughput_bias = 1.05 * drift_factor
                    latency_bias = 0.95 / drift_factor if drift_factor > 0 else 0.95
                    memory_bias = 0.97 * drift_factor
                    power_bias = 0.98 * drift_factor
                
                sim_result = SimulationResult(
                    model_id=model_id,
                    hardware_id=hw_id,
                    metrics={
                        "throughput_items_per_second": hw_throughput * throughput_bias * random_factor,
                        "average_latency_ms": hw_latency * latency_bias * random_factor,
                        "memory_peak_mb": hw_memory * memory_bias * random_factor,
                        "power_consumption_w": hw_power * power_bias * random_factor
                    },
                    batch_size=16,
                    precision="fp16",
                    timestamp=timestamp.isoformat(),
                    simulation_version="v1.2.3"
                )
                
                # Create validation result
                metrics_comparison = {}
                for metric, sim_value in sim_result.metrics.items():
                    hw_value = hw_result.metrics[metric]
                    abs_error = abs(sim_value - hw_value)
                    rel_error = abs_error / hw_value if hw_value != 0 else 0
                    mape = rel_error * 100.0
                    
                    metrics_comparison[metric] = {
                        "simulation_value": sim_value,
                        "hardware_value": hw_value,
                        "absolute_error": abs_error,
                        "relative_error": rel_error,
                        "mape": mape
                    }
                
                validation_result = ValidationResult(
                    simulation_result=sim_result,
                    hardware_result=hw_result,
                    metrics_comparison=metrics_comparison,
                    validation_timestamp=timestamp.isoformat(),
                    validation_version="v1.0"
                )
                
                validation_results.append(validation_result)
        
        return validation_results


class TestFrameworkBase(unittest.TestCase):
    """Base test case for simulation framework tests."""
    
    def setUp(self):
        """Set up test case."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize framework instance
        self.framework = get_framework_instance()
        
        # Generate sample data
        self.simulation_results, self.hardware_results = TestUtils.generate_sample_data()
        
        # Generate time series data
        self.time_series_results = TestUtils.generate_time_series_data()
        
        # Generate data with bias for calibration tests
        self.biased_sim_results, self.biased_hw_results = TestUtils.generate_sample_data(
            add_bias=True, bias_multiplier=1.5
        )
        
        # Generate time series data with drift for drift detection tests
        self.drift_time_series = TestUtils.generate_time_series_data(
            add_drift=True, drift_start_day=15
        )
    
    def tearDown(self):
        """Clean up after test case."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestValidation(TestFrameworkBase):
    """Test cases for validation components."""
    
    def test_data_generation(self):
        """Test that sample data generation works correctly."""
        self.assertTrue(len(self.simulation_results) > 0, "Failed to generate simulation results")
        self.assertTrue(len(self.hardware_results) > 0, "Failed to generate hardware results")
        self.assertEqual(len(self.simulation_results), len(self.hardware_results), 
                         "Mismatch in number of simulation and hardware results")
    
    def test_comparison_pipeline(self):
        """Test the comparison pipeline functionality."""
        if not comparison_pipeline_available:
            self.skipTest("ComparisonPipeline not available")
        
        pipeline = ComparisonPipeline()
        aligned_pairs = pipeline.align_data(self.simulation_results, self.hardware_results)
        
        self.assertTrue(len(aligned_pairs) > 0, "Failed to align simulation and hardware results")
        
        validation_results = pipeline.compare_results(aligned_pairs)
        
        self.assertTrue(len(validation_results) > 0, "Failed to generate validation results")
        
        # Check that validation results have the expected structure
        for val_result in validation_results[:5]:  # Check the first few results
            self.assertIsInstance(val_result, ValidationResult, "Result is not a ValidationResult")
            self.assertIsInstance(val_result.simulation_result, SimulationResult, 
                              "Simulation result is not a SimulationResult")
            self.assertIsInstance(val_result.hardware_result, HardwareResult, 
                              "Hardware result is not a HardwareResult")
            self.assertTrue(len(val_result.metrics_comparison) > 0, 
                            "Metrics comparison is empty")
            
            # Check that commonly expected metrics are present
            metric_found = False
            for metric in ["throughput_items_per_second", "average_latency_ms", 
                          "memory_peak_mb", "power_consumption_w"]:
                if metric in val_result.metrics_comparison:
                    metric_found = True
                    comparison = val_result.metrics_comparison[metric]
                    self.assertIn("mape", comparison, f"MAPE not found in {metric} comparison")
                    self.assertIn("absolute_error", comparison, 
                                 f"Absolute error not found in {metric} comparison")
            
            self.assertTrue(metric_found, "No expected metrics found in comparison")
    
    def test_statistical_validator(self):
        """Test the statistical validator functionality."""
        if not comparison_pipeline_available or not statistical_validator_available:
            self.skipTest("ComparisonPipeline or StatisticalValidator not available")
        
        pipeline = ComparisonPipeline()
        validator = StatisticalValidator()
        
        aligned_pairs = pipeline.align_data(self.simulation_results, self.hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Generate summary
        summary = validator.summarize_validation(validation_results)
        
        self.assertIsInstance(summary, dict, "Summary is not a dictionary")
        self.assertIn("overall", summary, "Summary doesn't include overall section")
        
        # Check overall statistics include expected metrics
        if "mape" in summary["overall"]:
            self.assertIn("mean", summary["overall"]["mape"], "MAPE mean not in summary")
            self.assertIsInstance(summary["overall"]["mape"]["mean"], (float, int, np.number), 
                              "MAPE mean is not a number")
        
        # Test confidence score calculation
        if len(validation_results) > 0:
            val_result = validation_results[0]
            hw_id = val_result.hardware_result.hardware_id
            model_id = val_result.hardware_result.model_id
            
            # Filter validation results for this hardware/model
            filtered_results = [r for r in validation_results
                                if r.hardware_result.hardware_id == hw_id 
                                and r.hardware_result.model_id == model_id]
            
            if len(filtered_results) >= 3:  # Need enough results for confidence
                confidence = validator.calculate_confidence_score(
                    filtered_results, hw_id, model_id
                )
                
                self.assertIsInstance(confidence, dict, "Confidence score is not a dictionary")
                self.assertIn("overall_confidence", confidence, 
                          "Overall confidence not in confidence score")
                self.assertIsInstance(confidence["overall_confidence"], (float, int, np.number), 
                                  "Overall confidence is not a number")


class TestCalibration(TestFrameworkBase):
    """Test cases for calibration components."""
    
    def test_basic_calibrator(self):
        """Test the basic calibrator functionality."""
        if not comparison_pipeline_available or not calibrator_available:
            self.skipTest("ComparisonPipeline or BasicSimulationCalibrator not available")
        
        pipeline = ComparisonPipeline()
        calibrator = BasicSimulationCalibrator()
        
        # Generate validation results with biased simulation
        aligned_pairs = pipeline.align_data(self.biased_sim_results, self.biased_hw_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Define initial simulation parameters
        simulation_parameters = {
            "correction_factors": {}
        }
        
        # Calibrate the simulation
        updated_parameters = calibrator.calibrate(validation_results, simulation_parameters)
        
        self.assertIsInstance(updated_parameters, dict, "Updated parameters is not a dictionary")
        self.assertIn("correction_factors", updated_parameters, 
                     "Correction factors not in updated parameters")
        
        # Check that at least one correction factor was generated
        correction_factors = updated_parameters["correction_factors"]
        self.assertTrue(len(correction_factors) > 0, "No correction factors generated")
        
        # Get a specific hardware/model combo for detailed testing
        hw_id = None
        model_id = None
        
        for hardware_id in correction_factors:
            if hardware_id in correction_factors:
                hw_id = hardware_id
                model_ids = correction_factors[hardware_id]
                if model_ids:
                    model_id = next(iter(model_ids))
                    break
        
        if hw_id and model_id:
            factors = correction_factors[hw_id][model_id]
            self.assertTrue(len(factors) > 0, f"No factors for {hw_id}/{model_id}")
            
            # Check that typical metrics have correction factors
            for metric in ["throughput_items_per_second", "average_latency_ms", 
                          "memory_peak_mb", "power_consumption_w"]:
                if metric in factors:
                    factor = factors[metric]
                    # Factor could be a number or a list/tuple for regression
                    self.assertTrue(
                        isinstance(factor, (int, float, np.number, list, tuple)) or
                        (isinstance(factor, dict) and "base_factor" in factor),
                        f"Invalid factor type for {metric}: {type(factor)}"
                    )
    
    def test_apply_calibration(self):
        """Test applying calibration to simulation results."""
        if not comparison_pipeline_available or not calibrator_available:
            self.skipTest("ComparisonPipeline or BasicSimulationCalibrator not available")
        
        pipeline = ComparisonPipeline()
        calibrator = BasicSimulationCalibrator()
        
        # Generate validation results with biased simulation
        aligned_pairs = pipeline.align_data(self.biased_sim_results, self.biased_hw_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Define initial simulation parameters
        simulation_parameters = {
            "correction_factors": {}
        }
        
        # Calibrate the simulation
        updated_parameters = calibrator.calibrate(validation_results, simulation_parameters)
        
        # Apply calibration to a simulation result
        if len(self.biased_sim_results) > 0:
            sim_result = self.biased_sim_results[0]
            
            calibrated_result = calibrator.apply_calibration(sim_result, updated_parameters)
            
            self.assertIsInstance(calibrated_result, SimulationResult, 
                              "Calibrated result is not a SimulationResult")
            
            # Check that metrics have been adjusted
            for metric in sim_result.metrics:
                if metric in calibrated_result.metrics:
                    # The values should be different due to calibration
                    self.assertNotEqual(
                        sim_result.metrics[metric], 
                        calibrated_result.metrics[metric],
                        f"Calibration did not change {metric} value"
                    )
    
    def test_advanced_calibrator(self):
        """Test the advanced calibrator functionality."""
        if not comparison_pipeline_available or not calibrator_available:
            self.skipTest("ComparisonPipeline or AdvancedSimulationCalibrator not available")
            
        try:
            # Check if the advanced calibrator dependencies are available
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            sklearn_available = True
        except ImportError:
            sklearn_available = False
            self.skipTest("scikit-learn not available for advanced calibrator")
        
        pipeline = ComparisonPipeline()
        calibrator = AdvancedSimulationCalibrator()
        
        # Generate validation results with biased simulation
        aligned_pairs = pipeline.align_data(self.biased_sim_results, self.biased_hw_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Define initial simulation parameters
        simulation_parameters = {
            "correction_factors": {}
        }
        
        # Calibrate the simulation
        updated_parameters = calibrator.calibrate(validation_results, simulation_parameters)
        
        self.assertIsInstance(updated_parameters, dict, "Updated parameters is not a dictionary")
        self.assertIn("correction_factors", updated_parameters, 
                     "Correction factors not in updated parameters")
        
        # Find and test an "ensemble" calibration if available
        found_ensemble = False
        for hw_id, hw_factors in updated_parameters["correction_factors"].items():
            for model_id, model_factors in hw_factors.items():
                for metric, factor in model_factors.items():
                    if isinstance(factor, dict) and "ensemble_factors" in factor:
                        found_ensemble = True
                        ensemble_factors = factor["ensemble_factors"]
                        self.assertTrue(len(ensemble_factors) > 0, "Empty ensemble factors")
                        break
                
                if found_ensemble:
                    break
            
            if found_ensemble:
                break
        
        # Evaluate calibration
        if len(validation_results) >= 2:
            # Split validation results for before/after comparison
            midpoint = len(validation_results) // 2
            before_calibration = validation_results[:midpoint]
            after_calibration = validation_results[midpoint:]
            
            evaluation = calibrator.evaluate_calibration(before_calibration, after_calibration)
            
            self.assertIsInstance(evaluation, dict, "Evaluation is not a dictionary")
            self.assertIn("overall", evaluation, "Overall section missing from evaluation")
            
            if "mape" in evaluation["overall"]:
                self.assertIn("before", evaluation["overall"]["mape"], 
                             "Before MAPE missing from evaluation")
                self.assertIn("after", evaluation["overall"]["mape"], 
                             "After MAPE missing from evaluation")


class TestDriftDetection(TestFrameworkBase):
    """Test cases for drift detection components."""
    
    def test_basic_drift_detector(self):
        """Test the basic drift detector functionality."""
        if not drift_detector_available:
            self.skipTest("BasicDriftDetector not available")
        
        detector = BasicDriftDetector()
        
        # Split time series for before/after comparison
        midpoint = len(self.drift_time_series) // 2
        historical_results = self.drift_time_series[:midpoint]
        new_results = self.drift_time_series[midpoint:]
        
        # Detect drift
        drift_results = detector.detect_drift(historical_results, new_results)
        
        self.assertIsInstance(drift_results, dict, "Drift results is not a dictionary")
        self.assertIn("is_significant", drift_results, "is_significant flag missing from drift results")
        self.assertIn("drift_metrics", drift_results, "drift_metrics missing from drift results")
        
        # Check drift metrics structure
        drift_metrics = drift_results["drift_metrics"]
        self.assertIsInstance(drift_metrics, dict, "Drift metrics is not a dictionary")
        
        # At least one metric should be present
        self.assertTrue(len(drift_metrics) > 0, "No metrics in drift results")
        
        # Check structure of a metric's drift info
        for metric, info in drift_metrics.items():
            self.assertIn("historical_mape", info, 
                         f"historical_mape missing for {metric}")
            self.assertIn("new_mape", info, 
                         f"new_mape missing for {metric}")
            self.assertIn("absolute_change", info, 
                         f"absolute_change missing for {metric}")
            self.assertIn("is_significant", info, 
                         f"is_significant flag missing for {metric}")
    
    def test_advanced_drift_detector(self):
        """Test the advanced drift detector functionality."""
        if not drift_detector_available:
            self.skipTest("AdvancedDriftDetector not available")
            
        try:
            # Check if the advanced detector dependencies are available
            import scipy.stats as stats
            from sklearn.decomposition import PCA
            sklearn_available = True
        except ImportError:
            sklearn_available = False
            self.skipTest("scipy or scikit-learn not available for advanced drift detector")
        
        detector = AdvancedDriftDetector()
        
        # Split time series for before/after comparison
        midpoint = len(self.drift_time_series) // 2
        historical_results = self.drift_time_series[:midpoint]
        new_results = self.drift_time_series[midpoint:]
        
        # Detect drift
        drift_results = detector.detect_drift(historical_results, new_results)
        
        self.assertIsInstance(drift_results, dict, "Drift results is not a dictionary")
        self.assertIn("is_significant", drift_results, "is_significant flag missing from drift results")
        self.assertIn("analysis_types", drift_results, "analysis_types missing from drift results")
        
        # Check that at least basic analysis is included
        self.assertIn("basic", drift_results["analysis_types"], 
                     "basic analysis missing from analysis_types")
        
        # Check for advanced analysis types
        advanced_analyses = [
            ("multi_dimensional", "multi_dimensional_analysis"),
            ("distribution", "distribution_analysis"),
            ("correlation", "correlation_analysis"),
            ("time_series", "time_series_analysis"),
            ("anomaly", "anomaly_detection")
        ]
        
        for analysis_type, result_key in advanced_analyses:
            if analysis_type in drift_results["analysis_types"]:
                self.assertIn(result_key, drift_results, 
                             f"{result_key} missing despite {analysis_type} being in analysis_types")
                
                analysis_result = drift_results[result_key]
                self.assertIsInstance(analysis_result, dict, 
                                     f"{result_key} is not a dictionary")
                self.assertIn("is_significant", analysis_result, 
                             f"is_significant flag missing from {result_key}")
    
    def test_drift_thresholds(self):
        """Test setting drift detection thresholds."""
        if not drift_detector_available:
            self.skipTest("BasicDriftDetector not available")
        
        detector = BasicDriftDetector()
        
        # Default thresholds
        default_thresholds = detector.drift_thresholds.copy()
        
        # Set new thresholds
        new_thresholds = {
            "throughput_items_per_second": 2.0,
            "average_latency_ms": 2.0,
            "memory_peak_mb": 2.0,
            "power_consumption_w": 2.0,
            "overall": 1.5
        }
        
        detector.set_drift_thresholds(new_thresholds)
        
        # Check that thresholds were updated
        for metric, threshold in new_thresholds.items():
            self.assertEqual(detector.drift_thresholds[metric], threshold, 
                            f"Threshold for {metric} not updated correctly")
        
        # Test drift detection with new thresholds
        midpoint = len(self.drift_time_series) // 2
        historical_results = self.drift_time_series[:midpoint]
        new_results = self.drift_time_series[midpoint:]
        
        drift_results = detector.detect_drift(historical_results, new_results)
        
        # Lower thresholds should make drift more likely to be significant
        self.assertIn("is_significant", drift_results, "is_significant flag missing from drift results")
        
        # Test drift status retrieval
        drift_status = detector.get_drift_status()
        self.assertIsInstance(drift_status, dict, "Drift status is not a dictionary")
        self.assertIn("is_drifting", drift_status, "is_drifting flag missing from drift status")


class TestVisualization(TestFrameworkBase):
    """Test cases for visualization components."""
    
    def test_mape_comparison_chart(self):
        """Test creating a MAPE comparison chart."""
        if not visualizer_available or not comparison_pipeline_available:
            self.skipTest("ValidationVisualizer or ComparisonPipeline not available")
        
        pipeline = ComparisonPipeline()
        visualizer = ValidationVisualizer()
        
        aligned_pairs = pipeline.align_data(self.simulation_results, self.hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Create output path
        output_path = os.path.join(self.temp_dir, "mape_comparison.html")
        
        # Create chart
        result = visualizer.create_mape_comparison_chart(
            validation_results,
            metric_name="all",
            output_path=output_path,
            interactive=False
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path), "Output file not created")
        
        # Check file content
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertTrue(len(content) > 0, "Output file is empty")
    
    def test_hardware_comparison_heatmap(self):
        """Test creating a hardware comparison heatmap."""
        if not visualizer_available or not comparison_pipeline_available:
            self.skipTest("ValidationVisualizer or ComparisonPipeline not available")
        
        pipeline = ComparisonPipeline()
        visualizer = ValidationVisualizer()
        
        aligned_pairs = pipeline.align_data(self.simulation_results, self.hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Create output path
        output_path = os.path.join(self.temp_dir, "hardware_heatmap.html")
        
        # Create heatmap
        result = visualizer.create_hardware_comparison_heatmap(
            validation_results,
            metric_name="throughput_items_per_second",
            output_path=output_path,
            interactive=False
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path), "Output file not created")
    
    def test_time_series_chart(self):
        """Test creating a time series chart."""
        if not visualizer_available:
            self.skipTest("ValidationVisualizer not available")
        
        visualizer = ValidationVisualizer()
        
        # Create output path
        output_path = os.path.join(self.temp_dir, "time_series.html")
        
        # Find a hardware/model combo with time series data
        hw_id = "gpu_rtx3080"
        model_id = "bert-base-uncased"
        
        # Create chart
        result = visualizer.create_time_series_chart(
            self.time_series_results,
            metric_name="throughput_items_per_second",
            hardware_id=hw_id,
            model_id=model_id,
            show_trend=True,
            output_path=output_path,
            interactive=False
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path), "Output file not created")
    
    def test_drift_detection_visualization(self):
        """Test creating a drift detection visualization."""
        if not visualizer_available or not drift_detector_available:
            self.skipTest("ValidationVisualizer or DriftDetector not available")
        
        visualizer = ValidationVisualizer()
        detector = BasicDriftDetector()
        
        # Split time series
        midpoint = len(self.drift_time_series) // 2
        historical_results = self.drift_time_series[:midpoint]
        new_results = self.drift_time_series[midpoint:]
        
        # Detect drift
        drift_results = detector.detect_drift(historical_results, new_results)
        
        # Create output path
        output_path = os.path.join(self.temp_dir, "drift_detection.html")
        
        # Create visualization
        result = visualizer.create_drift_detection_visualization(
            drift_results,
            output_path=output_path,
            interactive=False
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path), "Output file not created")
    
    def test_comprehensive_dashboard(self):
        """Test creating a comprehensive dashboard."""
        if not visualizer_available or not comparison_pipeline_available:
            self.skipTest("ValidationVisualizer or ComparisonPipeline not available")
            
        try:
            import plotly
            plotly_available = True
        except ImportError:
            plotly_available = False
            self.skipTest("Plotly not available for dashboard creation")
        
        pipeline = ComparisonPipeline()
        visualizer = ValidationVisualizer()
        
        aligned_pairs = pipeline.align_data(self.simulation_results, self.hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Create output path
        output_path = os.path.join(self.temp_dir, "dashboard.html")
        
        # Create dashboard
        result = visualizer.create_comprehensive_dashboard(
            validation_results,
            output_path=output_path
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path), "Output file not created")


class TestFrameworkIntegration(TestFrameworkBase):
    """Test cases for framework integration."""
    
    def test_end_to_end_validation(self):
        """Test end-to-end validation workflow through the framework."""
        # Test data validation
        validation_results = self.framework.validate(
            self.simulation_results, 
            self.hardware_results, 
            protocol="standard"
        )
        
        self.assertIsInstance(validation_results, list, "Validation results is not a list")
        self.assertTrue(len(validation_results) > 0, "Validation results list is empty")
        
        # Generate a report
        report_path = os.path.join(self.temp_dir, "validation_report.md")
        report = self.framework.generate_report(
            validation_results,
            format="markdown",
            output_path=report_path
        )
        
        self.assertTrue(os.path.exists(report_path), "Report file not created")
        
        # Get validation summary
        summary = self.framework.summarize_validation(validation_results)
        self.assertIsInstance(summary, dict, "Summary is not a dictionary")
        self.assertIn("overall", summary, "Summary doesn't include overall section")
    
    def test_calibration_workflow(self):
        """Test calibration workflow through the framework."""
        # Skip if calibrator not available
        if not hasattr(self.framework, "calibrator") or self.framework.calibrator is None:
            self.skipTest("Calibrator not available in framework")
        
        # Generate validation results with biased data
        validation_results = self.framework.validate(
            self.biased_sim_results, 
            self.biased_hw_results,
            protocol="standard"
        )
        
        # Define initial simulation parameters
        simulation_parameters = {
            "correction_factors": {}
        }
        
        # Calibrate the simulation
        updated_parameters = self.framework.calibrate(
            validation_results,
            simulation_parameters
        )
        
        self.assertIsInstance(updated_parameters, dict, "Updated parameters is not a dictionary")
        self.assertIn("correction_factors", updated_parameters, 
                     "Correction factors not in updated parameters")
    
    def test_drift_detection_workflow(self):
        """Test drift detection workflow through the framework."""
        # Skip if drift detector not available
        if not hasattr(self.framework, "drift_detector") or self.framework.drift_detector is None:
            self.skipTest("Drift detector not available in framework")
        
        # Split time series
        midpoint = len(self.drift_time_series) // 2
        historical_results = self.drift_time_series[:midpoint]
        new_results = self.drift_time_series[midpoint:]
        
        # Detect drift
        drift_results = self.framework.check_drift(
            historical_results,
            new_results
        )
        
        self.assertIsInstance(drift_results, dict, "Drift results is not a dictionary")
        self.assertIn("is_significant", drift_results, "is_significant flag missing from drift results")
    
    def test_visualization_workflow(self):
        """Test visualization workflow through the framework."""
        # Skip if visualizer not available
        if not hasattr(self.framework, "visualizer") or self.framework.visualizer is None:
            self.skipTest("Visualizer not available in framework")
        
        # Generate validation results
        validation_results = self.framework.validate(
            self.simulation_results, 
            self.hardware_results,
            protocol="standard"
        )
        
        # Create MAPE comparison chart
        mape_path = os.path.join(self.temp_dir, "framework_mape.html")
        self.framework.visualize_mape_comparison(
            validation_results,
            metric_name="all",
            output_path=mape_path
        )
        
        self.assertTrue(os.path.exists(mape_path), "MAPE chart file not created")
        
        # Create hardware comparison heatmap
        heatmap_path = os.path.join(self.temp_dir, "framework_heatmap.html")
        self.framework.visualize_hardware_comparison_heatmap(
            validation_results,
            metric_name="throughput_items_per_second",
            output_path=heatmap_path
        )
        
        self.assertTrue(os.path.exists(heatmap_path), "Heatmap file not created")


def main():
    """Run the test suite."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Simulation Validation Framework test suite")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Output directory for test results")
    args = parser.parse_args()
    
    # Set output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Redirect unittest output to a file
        output_file = output_dir / "test_results.txt"
        with open(output_file, 'w') as f:
            runner = unittest.TextTestRunner(stream=f, verbosity=2)
            unittest.main(argv=['first-arg-is-ignored'], exit=False, testRunner=runner)
            
        logger.info(f"Test results saved to: {output_file}")
    else:
        # Run tests normally
        unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    main()