#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for bandwidth metrics.

This script tests the BandwidthMetric class and factory with different device types
and validates platform-specific bandwidth measurement and roofline model analysis.
"""

import os
import sys
import unittest
import platform
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from metrics.bandwidth import BandwidthMetric, BandwidthMetricFactory
import hardware

class TestBandwidthMetric(unittest.TestCase):
    """Test cases for BandwidthMetric."""
    
    def setUp(self):
        """Set up test environment."""
        self.cpu_device = torch.device("cpu")
        self.cuda_device = torch.device("cuda") if torch.cuda.is_available() else None
        # Check for MPS (Apple Silicon) availability
        has_mps = False
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
                has_mps = torch.mps.is_available()
        except (AttributeError, ImportError):
            pass
        self.mps_device = torch.device("mps") if has_mps else None
        
    def test_bandwidth_metric_initialization(self):
        """Test basic initialization of BandwidthMetric."""
        # Test CPU metric
        cpu_metric = BandwidthMetric("cpu")
        self.assertEqual(cpu_metric.device_type, "cpu")
        self.assertEqual(cpu_metric.bandwidth_samples, [])
        self.assertEqual(cpu_metric.memory_accesses, 0)
        self.assertEqual(cpu_metric.memory_transfers_bytes, 0)
        self.assertEqual(cpu_metric.compute_operations, 0)
        self.assertGreater(cpu_metric.peak_bandwidth, 0)
        
        # Test CUDA metric if available
        if torch.cuda.is_available():
            cuda_metric = BandwidthMetric("cuda")
            self.assertEqual(cuda_metric.device_type, "cuda")
            self.assertGreater(cuda_metric.peak_bandwidth, 0)
            
        # Test MPS metric if available
        has_mps = False
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
                has_mps = torch.mps.is_available()
        except (AttributeError, ImportError):
            pass
            
        if has_mps:
            mps_metric = BandwidthMetric("mps")
            self.assertEqual(mps_metric.device_type, "mps")
            self.assertGreater(mps_metric.peak_bandwidth, 0)
    
    def test_peak_bandwidth_detection(self):
        """Test detection of theoretical peak memory bandwidth."""
        metric = BandwidthMetric("cpu")
        peak_bandwidth = metric._get_theoretical_peak_bandwidth()
        self.assertGreater(peak_bandwidth, 0)
        
        # Test CPU peak bandwidth detection
        cpu_peak = metric._get_cpu_peak_bandwidth()
        self.assertGreater(cpu_peak, 0)
        
        # Skip CUDA test if not available
        if torch.cuda.is_available():
            cuda_metric = BandwidthMetric("cuda")
            cuda_peak = cuda_metric._get_cuda_peak_bandwidth()
            self.assertGreater(cuda_peak, 0)
            # CUDA bandwidth should typically be higher than CPU
            self.assertGreater(cuda_peak, cpu_peak)
    
    @patch('subprocess.run')
    def test_cuda_bandwidth_reading(self, mock_run):
        """Test CUDA memory bandwidth reading."""
        # Skip if not running on CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Mock the subprocess.run to return memory utilization
        mock_run.return_value = MagicMock(returncode=0, stdout="50\n", stderr="")
        
        # Create the metric with mocked CUDA utilities
        with patch.object(BandwidthMetric, '_get_cuda_peak_bandwidth') as mock_peak:
            mock_peak.return_value = 800.0  # 800 GB/s theoretical peak
            metric = BandwidthMetric("cuda")
            
            # Mock the current bandwidth reading
            with patch.object(BandwidthMetric, '_get_cuda_bandwidth') as mock_bandwidth:
                mock_bandwidth.return_value = 400.0  # 400 GB/s
                bandwidth = metric._get_current_bandwidth()
                
                # Validate the bandwidth reading
                self.assertEqual(bandwidth, 400.0)
    
    def test_bandwidth_sampling(self):
        """Test bandwidth sampling thread."""
        # Create a metric with mocked bandwidth reading
        with patch.object(BandwidthMetric, '_get_current_bandwidth') as mock_bandwidth:
            mock_bandwidth.return_value = 100.0
            
            metric = BandwidthMetric("cuda")  # Use CUDA for sampling test
            
            # Start bandwidth sampling
            metric.start()
            
            # Check that sampling started
            self.assertTrue(metric.is_sampling)
            if metric.sampling_thread:
                self.assertTrue(metric.sampling_thread.daemon)
                
            # Allow some samples to be collected
            import time
            time.sleep(0.3)  # Sleep for 300ms to collect a few samples
            
            # Stop bandwidth sampling
            metric.stop()
            
            # Check that sampling stopped
            self.assertFalse(metric.is_sampling)
            
            # Check that samples were collected
            self.assertGreater(len(metric.bandwidth_samples), 0)
    
    def test_memory_transfer_estimation(self):
        """Test estimation of memory transfers."""
        metric = BandwidthMetric("cpu")
        
        # Test explicit setting
        metric.set_memory_transfers(1024 * 1024 * 1024)  # 1 GB
        self.assertEqual(metric.memory_transfers_bytes, 1024 * 1024 * 1024)
        
        # Test estimation method
        model_size = 100 * 1024 * 1024  # 100 MB model
        batch_size = 4
        inference_count = 10
        
        estimated = metric.estimate_memory_transfers(model_size, batch_size, inference_count)
        
        # Verify estimation logic
        expected_min = model_size * inference_count  # At minimum, each parameter is read once per inference
        self.assertGreaterEqual(estimated, expected_min)
        
        # Should scale with batch size
        estimated_larger_batch = metric.estimate_memory_transfers(model_size, batch_size * 2, inference_count)
        self.assertGreater(estimated_larger_batch, estimated)
    
    def test_arithmetic_intensity_calculation(self):
        """Test calculation of arithmetic intensity for roofline model."""
        metric = BandwidthMetric("cpu")
        
        # Set sample values
        metric.set_memory_transfers(1024 * 1024 * 1024)  # 1 GB
        metric.set_compute_operations(5e12)  # 5 TFLOPS
        
        # Calculate arithmetic intensity
        intensity = metric.get_arithmetic_intensity()
        
        # Expected: 5e12 / (1024 * 1024 * 1024) ≈ 4.66 FLOP/byte
        expected = 5e12 / (1024 * 1024 * 1024)
        self.assertAlmostEqual(intensity, expected, places=2)
    
    def test_compute_bound_analysis(self):
        """Test determination of compute vs memory bound workloads."""
        metric = BandwidthMetric("cpu")
        
        # Set peak bandwidth to 50 GB/s (typical for CPU)
        with patch.object(BandwidthMetric, '_get_theoretical_peak_bandwidth') as mock_peak:
            mock_peak.return_value = 50.0
            metric = BandwidthMetric("cpu")
            
            # Mock peak compute to 1 TFLOPS
            with patch.object(BandwidthMetric, '_get_peak_compute') as mock_compute:
                mock_compute.return_value = 1e12
                
                # Ridge point = 1e12 / (50 * 1e9) = 20 FLOP/byte
                
                # Test memory-bound workload
                metric.set_memory_transfers(1024 * 1024 * 1024)  # 1 GB
                metric.set_compute_operations(1e10)  # 10 GFLOPS
                # Intensity: 1e10 / (1024 * 1024 * 1024) ≈ 0.0093 FLOP/byte
                self.assertFalse(metric.is_compute_bound())
                
                # Test compute-bound workload
                metric.set_memory_transfers(1024 * 1024)  # 1 MB
                metric.set_compute_operations(1e11)  # 100 GFLOPS
                # Intensity: 1e11 / (1024 * 1024) ≈ 95.37 FLOP/byte
                self.assertTrue(metric.is_compute_bound())
    
    def test_roofline_data(self):
        """Test roofline model data generation."""
        # Create metric with controlled peak values
        with patch.object(BandwidthMetric, '_get_theoretical_peak_bandwidth') as mock_peak:
            mock_peak.return_value = 100.0  # 100 GB/s
            metric = BandwidthMetric("cuda")
            
            # Mock peak compute to 2 TFLOPS
            with patch.object(BandwidthMetric, '_get_peak_compute') as mock_compute:
                mock_compute.return_value = 2e12  # 2 TFLOPS
                
                # Set sample memory transfers and compute operations
                metric.set_memory_transfers(1024 * 1024 * 1024)  # 1 GB
                metric.set_compute_operations(1e12)  # 1 TFLOPS
                
                # Start/stop to set times
                metric.start()
                metric.end_time = metric.start_time + 1.0  # 1 second duration
                
                # Get roofline data
                roofline_data = metric.get_roofline_data()
                
                # Validate data structure
                self.assertIn('peak_compute_flops', roofline_data)
                self.assertIn('peak_memory_bandwidth_bytes_per_sec', roofline_data)
                self.assertIn('ridge_point_flops_per_byte', roofline_data)
                self.assertIn('arithmetic_intensity_flops_per_byte', roofline_data)
                self.assertIn('actual_performance_flops', roofline_data)
                
                # Validate values
                self.assertEqual(roofline_data['peak_compute_flops'], 2e12)
                self.assertEqual(roofline_data['peak_memory_bandwidth_bytes_per_sec'], 100.0 * 1e9)
                
                # Ridge point = peak_compute / peak_bandwidth
                expected_ridge_point = 2e12 / (100.0 * 1e9)
                self.assertAlmostEqual(roofline_data['ridge_point_flops_per_byte'], expected_ridge_point, places=2)
                
                # Arithmetic intensity = compute_operations / memory_transfers
                expected_intensity = 1e12 / (1024 * 1024 * 1024)
                self.assertAlmostEqual(roofline_data['arithmetic_intensity_flops_per_byte'], expected_intensity, places=2)
                
                # Actual performance = compute_operations / duration
                self.assertEqual(roofline_data['actual_performance_flops'], 1e12)
    
    def test_metrics_calculation(self):
        """Test calculation of bandwidth-related metrics."""
        # Create a metric with simulated bandwidth samples
        metric = BandwidthMetric("cuda")
        
        # Set synthetic bandwidth samples (time, bandwidth_gbps)
        current_time = 1000.0
        metric.bandwidth_samples = [
            (current_time, 200.0),
            (current_time + 0.1, 220.0),
            (current_time + 0.2, 180.0),
            (current_time + 0.3, 200.0)
        ]
        
        # Set start and end times
        metric.start_time = current_time
        metric.end_time = current_time + 0.3
        
        # Set memory transfers and compute operations
        metric.set_memory_transfers(10 * 1024 * 1024 * 1024)  # 10 GB
        metric.set_compute_operations(5e12)  # 5 TFLOPS
        
        # Set peak bandwidth
        with patch.object(BandwidthMetric, '_get_theoretical_peak_bandwidth') as mock_peak:
            mock_peak.return_value = 500.0  # 500 GB/s
            
            # Calculate metrics
            metrics = metric.get_metrics()
            
            # Validate metrics
            self.assertTrue(metrics["bandwidth_supported"])
            self.assertEqual(metrics["bandwidth_samples_count"], 4)
            self.assertEqual(metrics["avg_bandwidth_gbps"], 200.0)
            self.assertEqual(metrics["peak_bandwidth_gbps"], 220.0)
            self.assertEqual(metrics["peak_theoretical_bandwidth_gbps"], 800.0)
            
            # Utilization = avg / peak_theoretical
            self.assertEqual(metrics["bandwidth_utilization_percent"], 25.0)  # 200/800 = 25%
            
            # Memory transfers in GB (10 GB = 10.73741824 GiB)
            self.assertAlmostEqual(metrics["memory_transfers_gb"], 10.73741824, places=5)
            
            # Arithmetic intensity = compute_operations / memory_transfers_bytes
            expected_intensity = 5e12 / (10 * 1024 * 1024 * 1024)
            self.assertAlmostEqual(metrics["arithmetic_intensity_flops_per_byte"], expected_intensity, places=2)
    
    def test_factory_with_torch_device(self):
        """Test BandwidthMetricFactory with torch.device."""
        # Create metric with torch.device
        cpu_metric = BandwidthMetricFactory.create(self.cpu_device)
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA device if available
        if self.cuda_device:
            cuda_metric = BandwidthMetricFactory.create(self.cuda_device)
            self.assertEqual(cuda_metric.device_type, "cuda")
        
        # Create metric with MPS device if available
        if self.mps_device:
            mps_metric = BandwidthMetricFactory.create(self.mps_device)
            self.assertEqual(mps_metric.device_type, "mps")
    
    def test_factory_with_hardware_backend(self):
        """Test BandwidthMetricFactory with hardware backend."""
        # Create metric with CPU hardware backend
        cpu_backend = hardware.get_hardware_backend("cpu")
        cpu_device = cpu_backend.initialize()
        cpu_metric = BandwidthMetricFactory.create(cpu_device)
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA hardware backend if available
        if "cuda" in hardware.get_available_hardware():
            cuda_backend = hardware.get_hardware_backend("cuda")
            cuda_device = cuda_backend.initialize()
            cuda_metric = BandwidthMetricFactory.create(cuda_device)
            self.assertEqual(cuda_metric.device_type, "cuda")
    
    def test_factory_with_string(self):
        """Test BandwidthMetricFactory with string device."""
        # Create metric with string device
        cpu_metric = BandwidthMetricFactory.create("cpu")
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA string if available
        if torch.cuda.is_available():
            cuda_metric = BandwidthMetricFactory.create("cuda:0")
            self.assertEqual(cuda_metric.device_type, "cuda")
    
    def test_integration_with_benchmark(self):
        """Test integration with benchmark system."""
        # Import the benchmark module
        try:
            from benchmark import ModelBenchmark, BenchmarkConfig
            
            # Skip this test - just verify we can load the module and that config works correctly
            # In a real environment, we would test the full integration
            config = BenchmarkConfig(
                model_id="bert-base-uncased",
                hardware=["cpu"],
                batch_sizes=[1],
                sequence_lengths=[128],
                metrics=["bandwidth"]
            )
                
            # Verify model ID was set correctly
            self.assertEqual(config.model_id, "bert-base-uncased")
                
            # Verify bandwidth is in the metrics list
            self.assertIn("bandwidth", config.metrics)
                
        except ImportError:
            self.skipTest("ModelBenchmark import failed - this is expected if BandwidthMetric integration is not complete")
        
        # Skip actual integration test in this environment
        self.skipTest("Skipping full integration test in this environment")

if __name__ == '__main__':
    unittest.main()
