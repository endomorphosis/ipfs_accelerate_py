#!/usr/bin/env python3
"""
Tests for the Automatic Parameter Discovery implementation.

This module contains unit tests for the AutomaticParameterDiscovery class, which provides
utilities for automatically discovering which simulation parameters need calibration
and analyzing their sensitivity to different conditions.
"""

import os
import sys
import unittest
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_parameter_discovery")

# Add parent directories to path for module imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the components we're testing
from duckdb_api.simulation_validation.calibration.parameter_discovery import AutomaticParameterDiscovery
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
                "test_case": f"test_{i}",
                "gpu_clock_mhz": 1200 + i * 50,  # Varying parameter
                "memory_clock_mhz": 6000 + i * 100,  # Varying parameter
                "cuda_cores": 4000 + i * 200,  # Varying parameter
                "power_limit_w": 250 - i * 5  # Varying parameter
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
                "name": hardware_id,
                "memory_gb": 16 + (4 if "gpu" in hardware_id else 0),
                "cores": 8 if "cpu" in hardware_id else 0,
                "cuda_cores": 4000 + i * 200 if "gpu" in hardware_id else 0
            },
            test_environment={
                "temperature_c": 40 + 10 * np.random.rand(),
                "background_load": 0.1 + 0.2 * np.random.rand(),
                "os_version": "Linux 5.10",
                "driver_version": "470.82.01" if "gpu" in hardware_id else "N/A"
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

class TestParameterDiscovery(unittest.TestCase):
    """Tests for the AutomaticParameterDiscovery class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create the parameter discovery tool with default configuration
        self.parameter_discovery = AutomaticParameterDiscovery()
        
        # Create test data
        self.sim_results, self.hw_results, self.validation_results = create_test_data(
            num_samples=20,
            hardware_ids=["gpu_rtx3080", "cpu_intel_xeon"],
            model_ids=["bert-base-uncased", "vit-base-patch16-224"],
            batch_sizes=[1, 4, 8, 16],
            precision_types=["fp32", "fp16"]
        )
    
    def test_initialization(self):
        """Test initialization with different configurations."""
        # Test default initialization
        discovery = AutomaticParameterDiscovery()
        self.assertEqual(discovery.config["importance_calculation_method"], "permutation")
        self.assertEqual(discovery.config["min_samples_for_analysis"], 5)
        
        # Test custom configuration
        custom_config = {
            "importance_calculation_method": "correlation",
            "min_samples_for_analysis": 10,
            "sensitivity_threshold": 0.1,
            "batch_size_analysis": False
        }
        discovery = AutomaticParameterDiscovery(config=custom_config)
        self.assertEqual(discovery.config["importance_calculation_method"], "correlation")
        self.assertEqual(discovery.config["min_samples_for_analysis"], 10)
        self.assertEqual(discovery.config["sensitivity_threshold"], 0.1)
        self.assertFalse(discovery.config["batch_size_analysis"])
        
        # Test that default values are applied for omitted config items
        self.assertEqual(discovery.config["cross_parameter_analysis"], True)
    
    def test_discover_parameters(self):
        """Test parameter discovery functionality."""
        # Run parameter discovery
        parameter_recommendations = self.parameter_discovery.discover_parameters(self.validation_results)
        
        # Verify that recommendations were generated
        self.assertIn("parameters_by_metric", parameter_recommendations)
        self.assertIn("overall_priority_list", parameter_recommendations)
        self.assertIn("sensitivity_insights", parameter_recommendations)
        self.assertIn("optimization_recommendations", parameter_recommendations)
        
        # At least one metric should have parameters
        metrics_with_params = [m for m in parameter_recommendations["parameters_by_metric"] if parameter_recommendations["parameters_by_metric"][m]]
        self.assertGreater(len(metrics_with_params), 0)
        
        # Verify that at least one parameter was found
        self.assertGreater(len(parameter_recommendations["overall_priority_list"]), 0)
        
        # Verify that the parameters include expected fields
        if parameter_recommendations["overall_priority_list"]:
            param = parameter_recommendations["overall_priority_list"][0]
            self.assertIn("parameter", param)
            self.assertIn("importance", param)
    
    def test_analyze_parameter_sensitivity(self):
        """Test parameter sensitivity analysis."""
        # Choose a parameter to analyze
        parameter_name = "metadata_gpu_clock_mhz"
        
        # Run parameter sensitivity analysis
        sensitivity_results = self.parameter_discovery.analyze_parameter_sensitivity(
            self.validation_results, parameter_name
        )
        
        # Verify that sensitivity results were generated for at least one metric
        self.assertGreater(len(sensitivity_results), 0)
        
        # Verify that the sensitivity results include expected fields
        for metric, metric_sensitivity in sensitivity_results.items():
            self.assertIn("correlation", metric_sensitivity)
            self.assertIn("bin_statistics", metric_sensitivity)
            
            # Check that bin statistics were generated
            bin_stats = metric_sensitivity["bin_statistics"]
            self.assertGreater(len(bin_stats), 0)
            
            # Check that each bin includes expected statistics
            for bin_stat in bin_stats:
                self.assertIn("parameter_bin", bin_stat)
                self.assertIn("mean", bin_stat)
                self.assertIn("std", bin_stat)
                self.assertIn("count", bin_stat)
    
    def test_get_parameter_importance(self):
        """Test retrieving parameter importance scores."""
        # Run parameter discovery first
        self.parameter_discovery.discover_parameters(self.validation_results)
        
        # Get parameter importance
        importance_scores = self.parameter_discovery.get_parameter_importance()
        
        # Verify that importance scores were generated for at least one metric
        self.assertGreater(len(importance_scores), 0)
        
        # Verify that the importance scores include at least one parameter
        for metric, scores in importance_scores.items():
            self.assertGreater(len(scores), 0)
            
            # Check that scores sum to approximately 1.0
            score_sum = sum(scores.values())
            self.assertAlmostEqual(score_sum, 1.0, delta=0.01)
    
    def test_generate_insight_report(self):
        """Test generating a comprehensive insight report."""
        # Run parameter discovery first
        self.parameter_discovery.discover_parameters(self.validation_results)
        
        # Generate insight report
        insight_report = self.parameter_discovery.generate_insight_report()
        
        # Verify that the report includes expected sections
        self.assertIn("parameter_importance", insight_report)
        self.assertIn("parameter_sensitivity", insight_report)
        self.assertIn("discovered_parameters", insight_report)
        self.assertIn("recommendations", insight_report)
        self.assertIn("insights", insight_report)
        self.assertIn("timestamp", insight_report)
        
        # Verify that insights section includes expected fields
        insights = insight_report["insights"]
        self.assertIn("key_findings", insights)
        self.assertIn("optimization_opportunities", insights)
        self.assertIn("parameter_relationships", insights)
        
        # Check that key findings were generated
        self.assertGreater(len(insights["key_findings"]), 0)
        
        # Check that optimization opportunities were generated
        self.assertGreater(len(insights["optimization_opportunities"]), 0)

if __name__ == "__main__":
    unittest.main()