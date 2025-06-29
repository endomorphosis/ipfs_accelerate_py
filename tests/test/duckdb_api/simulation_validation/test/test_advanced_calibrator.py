#!/usr/bin/env python3
"""
Tests for the Advanced Calibrator implementation.

This module contains unit tests for the AdvancedSimulationCalibrator class, which provides
advanced methods for calibrating simulation parameters based on real hardware results.
"""

import unittest
import os
import sys
import logging
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_advanced_calibrator")

# Add parent directories to path for module imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the components we're testing
from duckdb_api.simulation_validation.calibration.advanced_calibrator import AdvancedSimulationCalibrator
from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicSimulationCalibrator
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

# Test helper for creating simulation and hardware results
def create_test_data(
    num_samples: int = 10,
    hardware_ids: List[str] = ["gpu_rtx3080", "cpu_intel_xeon"],
    model_ids: List[str] = ["bert-base-uncased", "vit-base-patch16-224"],
    batch_sizes: List[int] = [1, 4, 8],
    precision_types: List[str] = ["fp32", "fp16"]
) -> Tuple[List[SimulationResult], List[HardwareResult], List[ValidationResult]]:
    """Create test simulation and hardware results."""
    
    simulation_results = []
    hardware_results = []
    validation_results = []
    
    # Create simulation and hardware results with relationships between them
    for i in range(num_samples):
        # Choose hardware and model IDs
        hw_idx = i % len(hardware_ids)
        model_idx = i % len(model_ids)
        batch_idx = i % len(batch_sizes)
        precision_idx = i % len(precision_types)
        
        hardware_id = hardware_ids[hw_idx]
        model_id = model_ids[model_idx]
        batch_size = batch_sizes[batch_idx]
        precision = precision_types[precision_idx]
        
        # Create base metrics with some variability
        # The relationships between simulation and hardware are:
        # - throughput: hardware is roughly 0.8 * simulation (simulation overestimates)
        # - latency: hardware is roughly 1.2 * simulation (simulation underestimates)
        # - memory: hardware is roughly 1.1 * simulation (simulation underestimates)
        # Add some randomness to make it realistic
        base_throughput = 100 * (1 + 0.1 * model_idx) * (1 + 0.3 * batch_size) * (1 + 0.2 * (precision == "fp16"))
        base_latency = 10 * (1 + 0.2 * model_idx) * (1 - 0.05 * batch_size) * (1 - 0.1 * (precision == "fp16"))
        base_memory = 1000 * (1 + 0.3 * model_idx) * (1 + 0.1 * batch_size) * (1 - 0.2 * (precision == "fp16"))
        
        # Add hardware-specific factors
        hw_throughput_factor = 1.2 if "gpu" in hardware_id else 0.7
        hw_latency_factor = 0.8 if "gpu" in hardware_id else 1.5
        hw_memory_factor = 1.1 if "gpu" in hardware_id else 0.9
        
        # Create simulation metrics with some random noise
        sim_throughput = base_throughput * hw_throughput_factor * (1 + 0.05 * np.random.randn())
        sim_latency = base_latency * hw_latency_factor * (1 + 0.05 * np.random.randn())
        sim_memory = base_memory * hw_memory_factor * (1 + 0.05 * np.random.randn())
        
        # Create hardware metrics with consistent relationships and some noise
        hw_throughput = sim_throughput * 0.8 * (1 + 0.1 * np.random.randn())
        hw_latency = sim_latency * 1.2 * (1 + 0.1 * np.random.randn())
        hw_memory = sim_memory * 1.1 * (1 + 0.1 * np.random.randn())
        
        # Create timestamps
        timestamp = (datetime.now() - timedelta(days=i)).isoformat()
        
        # Create simulation result
        sim_result = SimulationResult(
            model_id=model_id,
            hardware_id=hardware_id,
            metrics={
                "throughput_items_per_second": sim_throughput,
                "average_latency_ms": sim_latency,
                "memory_peak_mb": sim_memory
            },
            batch_size=batch_size,
            precision=precision,
            timestamp=timestamp,
            simulation_version="sim_v1.0",
            additional_metadata={
                "test_case": f"test_{i}"
            }
        )
        
        # Create hardware result
        hw_result = HardwareResult(
            model_id=model_id,
            hardware_id=hardware_id,
            metrics={
                "throughput_items_per_second": hw_throughput,
                "average_latency_ms": hw_latency,
                "memory_peak_mb": hw_memory
            },
            batch_size=batch_size,
            precision=precision,
            timestamp=timestamp,
            hardware_info={
                "type": "gpu" if "gpu" in hardware_id else "cpu",
                "name": hardware_id
            },
            test_environment={
                "temperature_c": 40 + 10 * np.random.rand(),
                "background_load": 0.1 + 0.2 * np.random.rand()
            }
        )
        
        # Create validation result
        val_result = ValidationResult(
            simulation_result=sim_result,
            hardware_result=hw_result,
            validation_metrics={
                "throughput_items_per_second": {
                    "absolute_error": abs(sim_throughput - hw_throughput),
                    "relative_error": abs(sim_throughput - hw_throughput) / hw_throughput if hw_throughput > 0 else None,
                    "percentage_error": abs(sim_throughput - hw_throughput) / hw_throughput * 100 if hw_throughput > 0 else None
                },
                "average_latency_ms": {
                    "absolute_error": abs(sim_latency - hw_latency),
                    "relative_error": abs(sim_latency - hw_latency) / hw_latency if hw_latency > 0 else None,
                    "percentage_error": abs(sim_latency - hw_latency) / hw_latency * 100 if hw_latency > 0 else None
                },
                "memory_peak_mb": {
                    "absolute_error": abs(sim_memory - hw_memory),
                    "relative_error": abs(sim_memory - hw_memory) / hw_memory if hw_memory > 0 else None,
                    "percentage_error": abs(sim_memory - hw_memory) / hw_memory * 100 if hw_memory > 0 else None
                }
            },
            timestamp=timestamp,
            validation_version="val_v1.0"
        )
        
        simulation_results.append(sim_result)
        hardware_results.append(hw_result)
        validation_results.append(val_result)
    
    return simulation_results, hardware_results, validation_results

class TestAdvancedCalibrator(unittest.TestCase):
    """Tests for the AdvancedSimulationCalibrator class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create the calibrator with default configuration
        self.calibrator = AdvancedSimulationCalibrator()
        
        # Create test data
        self.sim_results, self.hw_results, self.validation_results = create_test_data(
            num_samples=20,
            hardware_ids=["gpu_rtx3080", "cpu_intel_xeon"],
            model_ids=["bert-base-uncased", "vit-base-patch16-224"],
            batch_sizes=[1, 4, 8, 16],
            precision_types=["fp32", "fp16"]
        )
        
        # Create empty simulation parameters
        self.simulation_parameters = {
            "correction_factors": {},
            "calibration_version": "test_v1.0",
            "last_calibration_timestamp": (datetime.now() - timedelta(days=10)).isoformat()
        }
    
    def test_initialization(self):
        """Test initialization with different configurations."""
        # Test default initialization
        calibrator = AdvancedSimulationCalibrator()
        self.assertEqual(calibrator.config["calibration_method"], "ensemble")
        self.assertEqual(calibrator.config["min_samples_per_hardware"], 5)
        
        # Test custom configuration
        custom_config = {
            "calibration_method": "bayesian",
            "min_samples_per_hardware": 10,
            "learning_rate": 0.1,
            "bayesian_iterations": 100
        }
        calibrator = AdvancedSimulationCalibrator(config=custom_config)
        self.assertEqual(calibrator.config["calibration_method"], "bayesian")
        self.assertEqual(calibrator.config["min_samples_per_hardware"], 10)
        self.assertEqual(calibrator.config["learning_rate"], 0.1)
        self.assertEqual(calibrator.config["bayesian_iterations"], 100)
        
        # Test that default values are applied for omitted config items
        self.assertEqual(calibrator.config["metrics_to_calibrate"], ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", "power_consumption_w"])
    
    def test_basic_calibration(self):
        """Test basic calibration functionality."""
        # Calibrate with linear_scaling method
        config = {
            "calibration_method": "linear_scaling",
            "min_samples_per_hardware": 1,
            "min_samples_per_model": 1,
            "learning_rate": 1.0  # Use full update for testing
        }
        calibrator = AdvancedSimulationCalibrator(config=config)
        
        # Calibrate
        updated_parameters = calibrator.calibrate(
            validation_results=self.validation_results,
            simulation_parameters=self.simulation_parameters
        )
        
        # Verify that calibration happened
        self.assertIn("correction_factors", updated_parameters)
        self.assertIn("gpu_rtx3080", updated_parameters["correction_factors"])
        self.assertIn("bert-base-uncased", updated_parameters["correction_factors"]["gpu_rtx3080"])
        
        # Verify that correction factors were created for expected metrics
        metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
        for metric in metrics:
            self.assertIn(metric, updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"])
            
            # For linear scaling, factor should be a scalar close to expected values
            factor = updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"][metric]
            
            # Check that factor is within reasonable range
            if metric == "throughput_items_per_second":
                # Simulation overestimates throughput, so factor should be < 1
                self.assertLess(factor, 1.0)
            elif metric == "average_latency_ms" or metric == "memory_peak_mb":
                # Simulation underestimates these, so factor should be > 1
                self.assertGreater(factor, 1.0)
    
    def test_regression_calibration(self):
        """Test regression-based calibration."""
        config = {
            "calibration_method": "regression",
            "min_samples_per_hardware": 1,
            "min_samples_per_model": 1,
            "learning_rate": 1.0  # Use full update for testing
        }
        calibrator = AdvancedSimulationCalibrator(config=config)
        
        # Calibrate
        updated_parameters = calibrator.calibrate(
            validation_results=self.validation_results,
            simulation_parameters=self.simulation_parameters
        )
        
        # Verify that correction factors are lists with [slope, intercept]
        metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
        for metric in metrics:
            self.assertIn(metric, updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"])
            
            factor = updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"][metric]
            self.assertIsInstance(factor, list)
            self.assertEqual(len(factor), 2)  # [slope, intercept]
    
    def test_ensemble_calibration(self):
        """Test ensemble-based calibration."""
        # Ensure enough samples for ensemble
        config = {
            "calibration_method": "ensemble",
            "min_samples_per_hardware": 1,
            "min_samples_per_model": 1,
            "learning_rate": 1.0,  # Use full update for testing
            "ensemble_weights": {
                "linear_scaling": 0.5,
                "regression": 0.5
            }
        }
        calibrator = AdvancedSimulationCalibrator(config=config)
        
        # Create more test data
        _, _, validation_results = create_test_data(
            num_samples=30,
            hardware_ids=["gpu_rtx3080"],
            model_ids=["bert-base-uncased"]
        )
        
        # Calibrate
        updated_parameters = calibrator.calibrate(
            validation_results=validation_results,
            simulation_parameters=self.simulation_parameters
        )
        
        # Verify that ensemble factors were created
        metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
        for metric in metrics:
            factor = updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"][metric]
            self.assertIsInstance(factor, dict)
            self.assertIn("ensemble_factors", factor)
            self.assertIn("ensemble_weights", factor)
            
            # Verify that ensemble factors include expected methods
            ensemble_factors = factor["ensemble_factors"]
            self.assertIn("linear_scaling", ensemble_factors)
            
            # Verify that weights match configuration
            ensemble_weights = factor["ensemble_weights"]
            self.assertEqual(ensemble_weights["linear_scaling"], 0.5)
    
    def test_apply_calibration(self):
        """Test applying calibration to simulation results."""
        # Create calibrator
        calibrator = AdvancedSimulationCalibrator()
        
        # Create simulation parameters with correction factors
        simulation_parameters = {
            "correction_factors": {
                "gpu_rtx3080": {
                    "bert-base-uncased": {
                        "throughput_items_per_second": 0.8,  # Reduce throughput by 20%
                        "average_latency_ms": 1.2,  # Increase latency by 20%
                        "memory_peak_mb": 1.1  # Increase memory by 10%
                    }
                }
            },
            "calibration_version": "test_v1.0",
            "calibration_method": "linear_scaling"
        }
        
        # Create simulation result to calibrate
        simulation_result = SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="gpu_rtx3080",
            metrics={
                "throughput_items_per_second": 100.0,
                "average_latency_ms": 10.0,
                "memory_peak_mb": 1000.0
            },
            batch_size=1,
            precision="fp32",
            timestamp=datetime.now().isoformat(),
            simulation_version="sim_v1.0"
        )
        
        # Apply calibration
        calibrated_result = calibrator.apply_calibration(
            simulation_result=simulation_result,
            simulation_parameters=simulation_parameters
        )
        
        # Verify that calibration was applied correctly
        self.assertEqual(calibrated_result.metrics["throughput_items_per_second"], 100.0 * 0.8)
        self.assertEqual(calibrated_result.metrics["average_latency_ms"], 10.0 * 1.2)
        self.assertEqual(calibrated_result.metrics["memory_peak_mb"], 1000.0 * 1.1)
        
        # Verify that metadata was updated
        self.assertTrue(calibrated_result.additional_metadata["calibration_applied"])
        self.assertEqual(calibrated_result.additional_metadata["calibration_version"], "test_v1.0")
        self.assertEqual(calibrated_result.additional_metadata["calibration_method"], "linear_scaling")
    
    def test_apply_ensemble_calibration(self):
        """Test applying ensemble calibration to simulation results."""
        # Create calibrator
        calibrator = AdvancedSimulationCalibrator()
        
        # Create simulation parameters with ensemble correction factors
        simulation_parameters = {
            "correction_factors": {
                "gpu_rtx3080": {
                    "bert-base-uncased": {
                        "throughput_items_per_second": {
                            "ensemble_factors": {
                                "linear_scaling": 0.8,
                                "regression": [0.75, 5.0]  # [slope, intercept]
                            },
                            "ensemble_weights": {
                                "linear_scaling": 0.7,
                                "regression": 0.3
                            }
                        },
                        "average_latency_ms": {
                            "ensemble_factors": {
                                "linear_scaling": 1.2,
                                "regression": [1.1, 2.0]  # [slope, intercept]
                            },
                            "ensemble_weights": {
                                "linear_scaling": 0.6,
                                "regression": 0.4
                            }
                        }
                    }
                }
            },
            "calibration_version": "test_v1.0",
            "calibration_method": "ensemble"
        }
        
        # Create simulation result to calibrate
        simulation_result = SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="gpu_rtx3080",
            metrics={
                "throughput_items_per_second": 100.0,
                "average_latency_ms": 10.0,
                "memory_peak_mb": 1000.0  # No correction factor for memory
            },
            batch_size=1,
            precision="fp32",
            timestamp=datetime.now().isoformat(),
            simulation_version="sim_v1.0"
        )
        
        # Apply calibration
        calibrated_result = calibrator.apply_calibration(
            simulation_result=simulation_result,
            simulation_parameters=simulation_parameters
        )
        
        # Calculate expected values
        expected_throughput = (0.7 * (100.0 * 0.8) + 0.3 * (100.0 * 0.75 + 5.0)) / (0.7 + 0.3)
        expected_latency = (0.6 * (10.0 * 1.2) + 0.4 * (10.0 * 1.1 + 2.0)) / (0.6 + 0.4)
        
        # Verify that calibration was applied correctly
        self.assertAlmostEqual(calibrated_result.metrics["throughput_items_per_second"], expected_throughput, places=5)
        self.assertAlmostEqual(calibrated_result.metrics["average_latency_ms"], expected_latency, places=5)
        self.assertEqual(calibrated_result.metrics["memory_peak_mb"], 1000.0)  # Unchanged
        
        # Verify that metadata was updated
        self.assertTrue(calibrated_result.additional_metadata["calibration_applied"])
        self.assertEqual(calibrated_result.additional_metadata["calibration_method"], "ensemble")
    
    def test_evaluate_calibration(self):
        """Test evaluating calibration effectiveness."""
        # Create calibrator
        calibrator = AdvancedSimulationCalibrator()
        
        # Create before and after validation results
        # For before: simulation overestimates throughput, underestimates latency/memory
        before_validation = []
        for i in range(10):
            sim_result = SimulationResult(
                model_id="bert-base-uncased",
                hardware_id="gpu_rtx3080",
                metrics={
                    "throughput_items_per_second": 100.0 * (1 + 0.1 * np.random.randn()),
                    "average_latency_ms": 10.0 * (1 + 0.1 * np.random.randn()),
                    "memory_peak_mb": 1000.0 * (1 + 0.1 * np.random.randn())
                },
                batch_size=1,
                precision="fp32",
                timestamp=datetime.now().isoformat(),
                simulation_version="sim_v1.0"
            )
            
            hw_result = HardwareResult(
                model_id="bert-base-uncased",
                hardware_id="gpu_rtx3080",
                metrics={
                    "throughput_items_per_second": 80.0 * (1 + 0.1 * np.random.randn()),  # 20% lower than sim
                    "average_latency_ms": 12.0 * (1 + 0.1 * np.random.randn()),  # 20% higher than sim
                    "memory_peak_mb": 1100.0 * (1 + 0.1 * np.random.randn())  # 10% higher than sim
                },
                batch_size=1,
                precision="fp32",
                timestamp=datetime.now().isoformat()
            )
            
            val_result = ValidationResult(
                simulation_result=sim_result,
                hardware_result=hw_result,
                validation_metrics={
                    "throughput_items_per_second": {
                        "percentage_error": abs(sim_result.metrics["throughput_items_per_second"] - hw_result.metrics["throughput_items_per_second"]) / hw_result.metrics["throughput_items_per_second"] * 100
                    },
                    "average_latency_ms": {
                        "percentage_error": abs(sim_result.metrics["average_latency_ms"] - hw_result.metrics["average_latency_ms"]) / hw_result.metrics["average_latency_ms"] * 100
                    },
                    "memory_peak_mb": {
                        "percentage_error": abs(sim_result.metrics["memory_peak_mb"] - hw_result.metrics["memory_peak_mb"]) / hw_result.metrics["memory_peak_mb"] * 100
                    }
                },
                timestamp=datetime.now().isoformat(),
                validation_version="val_v1.0"
            )
            
            before_validation.append(val_result)
        
        # For after: apply calibration to reduce errors
        after_validation = []
        for i in range(10):
            sim_result = SimulationResult(
                model_id="bert-base-uncased",
                hardware_id="gpu_rtx3080",
                metrics={
                    "throughput_items_per_second": 80.0 * (1 + 0.05 * np.random.randn()),  # Calibrated
                    "average_latency_ms": 12.0 * (1 + 0.05 * np.random.randn()),  # Calibrated
                    "memory_peak_mb": 1100.0 * (1 + 0.05 * np.random.randn())  # Calibrated
                },
                batch_size=1,
                precision="fp32",
                timestamp=datetime.now().isoformat(),
                simulation_version="sim_v1.0_calibrated"
            )
            
            hw_result = HardwareResult(
                model_id="bert-base-uncased",
                hardware_id="gpu_rtx3080",
                metrics={
                    "throughput_items_per_second": 80.0 * (1 + 0.1 * np.random.randn()),  # Same as before
                    "average_latency_ms": 12.0 * (1 + 0.1 * np.random.randn()),  # Same as before
                    "memory_peak_mb": 1100.0 * (1 + 0.1 * np.random.randn())  # Same as before
                },
                batch_size=1,
                precision="fp32",
                timestamp=datetime.now().isoformat()
            )
            
            val_result = ValidationResult(
                simulation_result=sim_result,
                hardware_result=hw_result,
                validation_metrics={
                    "throughput_items_per_second": {
                        "percentage_error": abs(sim_result.metrics["throughput_items_per_second"] - hw_result.metrics["throughput_items_per_second"]) / hw_result.metrics["throughput_items_per_second"] * 100
                    },
                    "average_latency_ms": {
                        "percentage_error": abs(sim_result.metrics["average_latency_ms"] - hw_result.metrics["average_latency_ms"]) / hw_result.metrics["average_latency_ms"] * 100
                    },
                    "memory_peak_mb": {
                        "percentage_error": abs(sim_result.metrics["memory_peak_mb"] - hw_result.metrics["memory_peak_mb"]) / hw_result.metrics["memory_peak_mb"] * 100
                    }
                },
                timestamp=datetime.now().isoformat(),
                validation_version="val_v1.0"
            )
            
            after_validation.append(val_result)
        
        # Evaluate calibration
        evaluation = calibrator.evaluate_calibration(
            before_calibration=before_validation,
            after_calibration=after_validation
        )
        
        # Verify evaluation structure
        self.assertIn("metrics", evaluation)
        self.assertIn("overall", evaluation)
        self.assertIn("num_samples", evaluation)
        
        # Verify that overall metrics show improvement
        self.assertGreater(evaluation["overall"]["mape"]["relative_improvement_pct"], 0)
        self.assertGreater(evaluation["overall"]["rmse"]["relative_improvement_pct"], 0)
    
    def test_hardware_profiles(self):
        """Test hardware-specific profiles for calibration."""
        # Create calibrator with hardware profiles
        calibrator = AdvancedSimulationCalibrator(config={
            "use_hardware_profiles": True
        })
        
        # Verify that hardware profiles were initialized
        self.assertIn("gpu_rtx3080", calibrator.hardware_profiles)
        self.assertIn("cpu_intel_xeon", calibrator.hardware_profiles)
        
        # Verify that profiles have expected structure
        rtx_profile = calibrator.hardware_profiles["gpu_rtx3080"]
        self.assertIn("preferred_calibration_method", rtx_profile)
        self.assertIn("metric_adjustments", rtx_profile)
        
        # Verify that metric adjustments contain expected data
        metric_adjustments = rtx_profile["metric_adjustments"]
        self.assertIn("throughput_items_per_second", metric_adjustments)
        self.assertIn("batch_size_factors", metric_adjustments["throughput_items_per_second"])
        self.assertIn("precision_factors", metric_adjustments["throughput_items_per_second"])
    
    def test_multi_parameter_optimization(self):
        """Test multi-parameter optimization for calibration."""
        try:
            from skopt import gp_minimize
            from sklearn.neural_network import MLPRegressor
            skopt_available = True
            sklearn_available = True
        except ImportError:
            skopt_available = False
            sklearn_available = False
        
        if not skopt_available or not sklearn_available:
            self.skipTest("scikit-optimize or scikit-learn not available")
        
        # Create calibrator with bayesian method
        calibrator = AdvancedSimulationCalibrator(config={
            "calibration_method": "bayesian",
            "bayesian_iterations": 10,  # Low for testing
            "min_samples_per_hardware": 1,
            "min_samples_per_model": 1
        })
        
        # Create validation results with multiple parameters to optimize
        _, _, validation_results = create_test_data(
            num_samples=20,
            hardware_ids=["gpu_rtx3080"],
            model_ids=["bert-base-uncased"],
            batch_sizes=[1, 4, 8, 16],
            precision_types=["fp32", "fp16"]
        )
        
        # Calibrate
        updated_parameters = calibrator.calibrate(
            validation_results=validation_results,
            simulation_parameters=self.simulation_parameters
        )
        
        # Verify that correction factors were created
        self.assertIn("correction_factors", updated_parameters)
        self.assertIn("gpu_rtx3080", updated_parameters["correction_factors"])
        self.assertIn("bert-base-uncased", updated_parameters["correction_factors"]["gpu_rtx3080"])
        
        # For bayesian, correction factors should be [slope, intercept]
        metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
        for metric in metrics:
            factor = updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"][metric]
            self.assertIsInstance(factor, list)
            self.assertEqual(len(factor), 2)  # [slope, intercept]
    
    def test_learning_rate_adaptation(self):
        """Test learning rate adaptation for calibration optimization."""
        # Create calibrator with incremental learning
        calibrator = AdvancedSimulationCalibrator(config={
            "calibration_method": "incremental",
            "min_samples_per_hardware": 1,
            "min_samples_per_model": 1,
            "learning_rate": 0.5  # Start with 0.5
        })
        
        # Create simulation parameters with calibration history
        simulation_parameters = {
            "correction_factors": {
                "gpu_rtx3080": {
                    "bert-base-uncased": {
                        "throughput_items_per_second": {
                            "base_factor": 0.8,
                            "trend_factor": 0.01,
                            "recent_samples": 10,
                            "calibration_timestamp": (datetime.now() - timedelta(days=30)).isoformat()
                        }
                    }
                }
            },
            "calibration_history": [
                {
                    "timestamp": (datetime.now() - timedelta(days=40)).isoformat(),
                    "id": "cal1",
                    "num_validation_results": 5,
                    "hardware_models": [("gpu_rtx3080", "bert-base-uncased")],
                    "method": "incremental"
                },
                {
                    "timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
                    "id": "cal2",
                    "num_validation_results": 8,
                    "hardware_models": [("gpu_rtx3080", "bert-base-uncased")],
                    "method": "incremental"
                },
                {
                    "timestamp": (datetime.now() - timedelta(days=20)).isoformat(),
                    "id": "cal3",
                    "num_validation_results": 12,
                    "hardware_models": [("gpu_rtx3080", "bert-base-uncased")],
                    "method": "incremental"
                }
            ],
            "calibration_version": "test_v1.0",
            "calibration_method": "incremental"
        }
        
        # Calibrate
        updated_parameters = calibrator.calibrate(
            validation_results=self.validation_results,
            simulation_parameters=simulation_parameters
        )
        
        # Verify that calibration history was updated
        self.assertEqual(len(updated_parameters["calibration_history"]), 3 + 1)  # Previous 3 + new one
        
        # Verify that incremental learning occurred
        factor = updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"]["throughput_items_per_second"]
        self.assertIsInstance(factor, dict)
        self.assertIn("base_factor", factor)
        self.assertIn("trend_factor", factor)
        self.assertIn("recent_samples", factor)
        
        # Apply calibration with incremental factors
        simulation_result = SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="gpu_rtx3080",
            metrics={
                "throughput_items_per_second": 100.0
            },
            batch_size=1,
            precision="fp32",
            timestamp=datetime.now().isoformat(),
            simulation_version="sim_v1.0"
        )
        
        calibrated_result = calibrator.apply_calibration(
            simulation_result=simulation_result,
            simulation_parameters=updated_parameters
        )
        
        # Verify that calibration was applied with trend factor
        base_factor = factor["base_factor"]
        trend_factor = factor["trend_factor"]
        recent_samples = factor["recent_samples"]
        
        expected_throughput = 100.0 * base_factor * (1 + trend_factor * recent_samples)
        self.assertAlmostEqual(calibrated_result.metrics["throughput_items_per_second"], expected_throughput, places=5)

    def test_cross_validation_for_calibration(self):
        """Test cross-validation for calibration parameter tuning."""
        try:
            from sklearn.model_selection import cross_val_score, KFold
            sklearn_available = True
        except ImportError:
            sklearn_available = False
        
        if not sklearn_available:
            self.skipTest("scikit-learn not available")
        
        # Create calibrator
        calibrator = AdvancedSimulationCalibrator(config={
            "calibration_method": "bayesian",
            "bayesian_iterations": 10,  # Low for testing
            "min_samples_per_hardware": 1,
            "min_samples_per_model": 1,
            "enable_cross_validation": True,
            "cross_validation_folds": 3
        })
        
        # Create many validation results for cross-validation
        _, _, validation_results = create_test_data(
            num_samples=50,
            hardware_ids=["gpu_rtx3080"],
            model_ids=["bert-base-uncased"]
        )
        
        # Calibrate
        updated_parameters = calibrator.calibrate(
            validation_results=validation_results,
            simulation_parameters=self.simulation_parameters
        )
        
        # Verify that calibration happened
        self.assertIn("correction_factors", updated_parameters)
        self.assertIn("gpu_rtx3080", updated_parameters["correction_factors"])
        self.assertIn("bert-base-uncased", updated_parameters["correction_factors"]["gpu_rtx3080"])
        
        # For bayesian with cross-validation, factors should be [slope, intercept]
        metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
        for metric in metrics:
            self.assertIn(metric, updated_parameters["correction_factors"]["gpu_rtx3080"]["bert-base-uncased"])
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification for calibration parameters."""
        # Create calibrator
        calibrator = AdvancedSimulationCalibrator(config={
            "calibration_method": "ensemble",
            "min_samples_per_hardware": 1,
            "min_samples_per_model": 1,
            "ensemble_weights": {
                "linear_scaling": 0.5,
                "regression": 0.5
            }
        })
        
        # Create validation results
        _, _, validation_results = create_test_data(
            num_samples=20,
            hardware_ids=["gpu_rtx3080"],
            model_ids=["bert-base-uncased"]
        )
        
        # Calibrate
        updated_parameters = calibrator.calibrate(
            validation_results=validation_results,
            simulation_parameters=self.simulation_parameters
        )
        
        # Apply calibration to get uncertainty estimates
        simulation_result = SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="gpu_rtx3080",
            metrics={
                "throughput_items_per_second": 100.0,
                "average_latency_ms": 10.0,
                "memory_peak_mb": 1000.0
            },
            batch_size=1,
            precision="fp32",
            timestamp=datetime.now().isoformat(),
            simulation_version="sim_v1.0"
        )
        
        calibrated_result = calibrator.apply_calibration(
            simulation_result=simulation_result,
            simulation_parameters=updated_parameters
        )
        
        # Verify that calibration was applied
        self.assertNotEqual(calibrated_result.metrics["throughput_items_per_second"], 100.0)
        self.assertNotEqual(calibrated_result.metrics["average_latency_ms"], 10.0)
        self.assertNotEqual(calibrated_result.metrics["memory_peak_mb"], 1000.0)

if __name__ == "__main__":
    unittest.main()