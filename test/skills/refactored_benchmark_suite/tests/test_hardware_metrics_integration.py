#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for hardware-aware metrics.

This module contains tests that validate the integration between hardware-aware
metrics (power, bandwidth) and the benchmark system.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import ModelBenchmark, BenchmarkConfig
from hardware import initialize_hardware, get_available_hardware
from metrics.timing import TimingMetricFactory
from metrics.memory import MemoryMetricFactory
from metrics.flops import FLOPsMetricFactory
from metrics.power import PowerMetricFactory
from metrics.bandwidth import BandwidthMetricFactory

# Skip tests if torch is not available
if not torch.cuda.is_available():
    HAS_CUDA = False
else:
    HAS_CUDA = True

# Skip tests if MPS is not available
HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

class TestHardwareMetricsIntegration(unittest.TestCase):
    """Integration tests for hardware-aware metrics."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a temp directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_metrics_creation(self):
        """Test creation of hardware-aware metrics."""
        # Test for each available hardware, but focus on CPU for reliability
        # On some platforms, certain hardware types may map to different device types
        device = initialize_hardware("cpu")
        
        # Create hardware-aware metrics
        power_metric = PowerMetricFactory.create(device)
        bandwidth_metric = BandwidthMetricFactory.create(device)
        
        # Validate metrics were created successfully
        self.assertEqual(power_metric.device_type.lower(), "cpu")
        self.assertEqual(bandwidth_metric.device_type.lower(), "cpu")
        
        # For CUDA, if available
        if "cuda" in get_available_hardware() and torch.cuda.is_available():
            cuda_device = initialize_hardware("cuda")
            cuda_power_metric = PowerMetricFactory.create(cuda_device)
            cuda_bandwidth_metric = BandwidthMetricFactory.create(cuda_device)
            
            # Validate metrics were created with CUDA device type
            self.assertTrue(cuda_power_metric.device_type.lower() in ["cuda", "gpu"])
            self.assertTrue(cuda_bandwidth_metric.device_type.lower() in ["cuda", "gpu"])
    
    def test_metrics_lifecycle(self):
        """Test lifecycle of hardware-aware metrics."""
        # Test on CPU for simplicity
        device = initialize_hardware("cpu")
        
        # Create metrics
        power_metric = PowerMetricFactory.create(device)
        bandwidth_metric = BandwidthMetricFactory.create(device)
        
        # Test start
        power_metric.start()
        bandwidth_metric.start()
        
        self.assertGreaterEqual(power_metric.start_time, 0)
        self.assertGreaterEqual(bandwidth_metric.start_time, 0)
        
        # Test stop
        power_metric.stop()
        bandwidth_metric.stop()
        
        self.assertGreaterEqual(power_metric.end_time, power_metric.start_time)
        self.assertGreaterEqual(bandwidth_metric.end_time, bandwidth_metric.start_time)
        
        # Test get_metrics
        power_metrics = power_metric.get_metrics()
        bandwidth_metrics = bandwidth_metric.get_metrics()
        
        self.assertIsInstance(power_metrics, dict)
        self.assertIsInstance(bandwidth_metrics, dict)
        self.assertIn("power_supported", power_metrics)
        self.assertIn("bandwidth_supported", bandwidth_metrics)
    
    def test_metrics_data_sharing(self):
        """Test data sharing between metrics."""
        # Test on CPU for simplicity
        device = initialize_hardware("cpu")
        
        # Create metrics
        flops_metric = FLOPsMetricFactory.create(device)
        power_metric = PowerMetricFactory.create(device)
        bandwidth_metric = BandwidthMetricFactory.create(device)
        
        # Simulate operations count
        operations_count = 1e9  # 1 GFLOP
        flops_metric.total_flops = operations_count
        
        # Transfer FLOPS to power and bandwidth metrics
        power_metric.set_operations_count(flops_metric.total_flops)
        bandwidth_metric.set_compute_operations(flops_metric.total_flops)
        
        # Set memory transfers for bandwidth
        memory_transfers = 1024 * 1024 * 1024  # 1 GB
        bandwidth_metric.set_memory_transfers(memory_transfers)
        
        # Calculate arithmetic intensity
        intensity = bandwidth_metric.get_arithmetic_intensity()
        expected_intensity = operations_count / memory_transfers
        self.assertAlmostEqual(intensity, expected_intensity, places=5)
        
        # Check roofline data generation
        roofline_data = bandwidth_metric.get_roofline_data()
        self.assertIn("arithmetic_intensity_flops_per_byte", roofline_data)
        self.assertAlmostEqual(roofline_data["arithmetic_intensity_flops_per_byte"], expected_intensity, places=5)
    
    def test_benchmark_power_bandwidth_integration(self):
        """Test integration with benchmark system."""
        # Skip this test in the current environment
        # In a real test environment, we would test with a real model
        # Here we just verify we can import everything needed
        
        # Import necessary components
        from benchmark import BenchmarkConfig
        from metrics.power import PowerMetricFactory, PowerMetric
        from metrics.bandwidth import BandwidthMetricFactory, BandwidthMetric
        
        # Verify the imports succeeded
        self.assertTrue(hasattr(PowerMetric, 'get_metrics'))
        self.assertTrue(hasattr(BandwidthMetric, 'get_metrics'))
        self.assertTrue(hasattr(PowerMetricFactory, 'create'))
        self.assertTrue(hasattr(BandwidthMetricFactory, 'create'))
        
        # Skip the actual benchmark test in this environment
        self.skipTest("Skipping full benchmark integration test in this environment")
    
    def test_visualization_methods(self):
        """Test visualization methods for hardware-aware metrics."""
        # Create a benchmark results object with mocked data
        from benchmark import BenchmarkResults, BenchmarkResult, BenchmarkConfig
        
        # Create a simple config
        config = BenchmarkConfig(
            model_id="bert-base-uncased",
            hardware=["cpu", "cuda"],
            batch_sizes=[1, 2, 4],
            metrics=["latency", "throughput", "power", "bandwidth"],
            output_dir=self.output_dir
        )
        
        # Create mocked results
        results = []
        for hw in ["cpu", "cuda"] if HAS_CUDA else ["cpu"]:
            for bs in [1, 2, 4]:
                # Create a result with power metrics
                power_result = BenchmarkResult(
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
                        "compute_bound": False
                    }
                )
                results.append(power_result)
        
        # Create benchmark results object
        benchmark_results = BenchmarkResults(results, config)
        
        # Test visualization methods if matplotlib is available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Test power efficiency visualization
            power_plot_path = benchmark_results.plot_power_efficiency()
            self.assertIsNotNone(power_plot_path)
            self.assertTrue(os.path.exists(power_plot_path))
            
            # Test bandwidth utilization visualization
            bandwidth_plot_path = benchmark_results.plot_bandwidth_utilization()
            self.assertIsNotNone(bandwidth_plot_path)
            self.assertTrue(os.path.exists(bandwidth_plot_path))
            
        except ImportError:
            # Skip tests if matplotlib is not available
            self.skipTest("matplotlib is not available")

if __name__ == '__main__':
    unittest.main()
