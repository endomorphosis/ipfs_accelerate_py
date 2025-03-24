#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for hardware-aware metrics visualization.

This script tests the visualization functions for power efficiency and
bandwidth metrics in the refactored benchmark suite.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkResults
from visualizers.plots import plot_power_efficiency, plot_bandwidth_utilization

@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "matplotlib is not available")
class TestHardwareAwareVisualization(unittest.TestCase):
    """Tests for hardware-aware metrics visualization."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a benchmark config
        self.config = BenchmarkConfig(
            model_id="bert-base-uncased",
            hardware=["cpu", "cuda"],
            batch_sizes=[1, 2, 4],
            metrics=["latency", "throughput", "power", "bandwidth"],
            output_dir=self.temp_dir.name
        )
        
        # Create benchmark results
        self.results = []
        for hw in ["cpu", "cuda"]:
            for bs in [1, 2, 4]:
                # Create a result with power and bandwidth metrics
                result = BenchmarkResult(
                    model_id="bert-base-uncased",
                    hardware=hw,
                    batch_size=bs,
                    sequence_length=16,
                    metrics={
                        "latency_ms": 10.0 if hw == "cuda" else 50.0,
                        "throughput_items_per_sec": bs * 10.0 if hw == "cuda" else bs * 2.0,
                        "power_avg_watts": 250.0 if hw == "cuda" else 100.0,
                        "power_max_watts": 300.0 if hw == "cuda" else 120.0,
                        "energy_joules": 30.0 if hw == "cuda" else 50.0,
                        "gflops_per_watt": 40.0 if hw == "cuda" else 10.0,
                        "throughput_per_watt": bs * 0.04 if hw == "cuda" else bs * 0.02,
                        "avg_bandwidth_gbps": 500.0 if hw == "cuda" else 50.0,
                        "peak_theoretical_bandwidth_gbps": 900.0 if hw == "cuda" else 100.0,
                        "bandwidth_utilization_percent": 55.5 if hw == "cuda" else 50.0,
                        "arithmetic_intensity_flops_per_byte": 5.0,
                        "compute_bound": False,
                        "roofline_data": {
                            "peak_compute_flops": 10e12 if hw == "cuda" else 2e12,
                            "peak_memory_bandwidth_bytes_per_sec": 900e9 if hw == "cuda" else 100e9,
                            "ridge_point_flops_per_byte": 11.11 if hw == "cuda" else 20.0,
                            "arithmetic_intensity_flops_per_byte": 5.0,
                            "is_compute_bound": False
                        }
                    }
                )
                self.results.append(result)
        
        # Create BenchmarkResults object
        self.benchmark_results = BenchmarkResults(self.results, self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_power_efficiency_visualization(self):
        """Test power efficiency visualization."""
        # Generate output path
        output_path = os.path.join(self.temp_dir.name, "power_efficiency_test.png")
        
        # Call visualization function
        result_path = plot_power_efficiency(self.benchmark_results, output_path)
        
        # Check that file was created
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_bandwidth_utilization_visualization(self):
        """Test bandwidth utilization visualization."""
        # Generate output path
        output_path = os.path.join(self.temp_dir.name, "bandwidth_utilization_test.png")
        
        # Call visualization function
        result_path = plot_bandwidth_utilization(self.benchmark_results, output_path)
        
        # Check that file was created
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_power_efficiency_visualization_integration(self):
        """Test power efficiency visualization through BenchmarkResults."""
        # Call visualization method
        result_path = self.benchmark_results.plot_power_efficiency()
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
    
    def test_bandwidth_utilization_visualization_integration(self):
        """Test bandwidth utilization visualization through BenchmarkResults."""
        # Call visualization method
        result_path = self.benchmark_results.plot_bandwidth_utilization()
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
    
    def test_visualization_without_data(self):
        """Test visualization with no relevant data."""
        # Create results without power or bandwidth metrics
        results = []
        for hw in ["cpu", "cuda"]:
            for bs in [1, 2]:
                result = BenchmarkResult(
                    model_id="bert-base-uncased",
                    hardware=hw,
                    batch_size=bs,
                    sequence_length=16,
                    metrics={
                        "latency_ms": 10.0,
                        "throughput_items_per_sec": bs * 10.0,
                    }
                )
                results.append(result)
        
        # Create BenchmarkResults object
        benchmark_results = BenchmarkResults(results, self.config)
        
        # Test that visualization returns None when no relevant data exists
        self.assertIsNone(benchmark_results.plot_power_efficiency())
        self.assertIsNone(benchmark_results.plot_bandwidth_utilization())
    
    def test_visualization_detail_level(self):
        """Test visualization with different detail levels."""
        # For power_efficiency, test with and without gflops_per_watt
        power_results = []
        for hw in ["cpu", "cuda"]:
            # Create a minimal result with only power_avg_watts
            minimal_result = BenchmarkResult(
                model_id="bert-base-uncased",
                hardware=hw,
                batch_size=1,
                sequence_length=16,
                metrics={
                    "power_avg_watts": 100.0,
                }
            )
            power_results.append(minimal_result)
            
            # Create a detailed result with all power metrics
            detailed_result = BenchmarkResult(
                model_id="bert-base-uncased",
                hardware=hw,
                batch_size=2,
                sequence_length=16,
                metrics={
                    "power_avg_watts": 100.0,
                    "gflops_per_watt": 10.0,
                    "throughput_per_watt": 0.1,
                }
            )
            power_results.append(detailed_result)
        
        # Create BenchmarkResults object
        power_benchmark_results = BenchmarkResults(power_results, self.config)
        
        # Test that visualization works with different detail levels
        power_plot_path = power_benchmark_results.plot_power_efficiency()
        self.assertTrue(os.path.exists(power_plot_path))

if __name__ == '__main__':
    unittest.main()
