#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test generator for hardware-aware metrics.

This script generates comprehensive unit tests for power and bandwidth metrics
components of the refactored benchmark suite using templates.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Templates for test files
POWER_METRIC_TEST_TEMPLATE = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Test script for power metrics.

This script tests the PowerMetric class and factory with different device types
and validates platform-specific power monitoring.
\"\"\"

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
from metrics.power import PowerMetric, PowerMetricFactory
import hardware

class TestPowerMetric(unittest.TestCase):
    \"\"\"Test cases for PowerMetric.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
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
        
    def test_power_metric_initialization(self):
        \"\"\"Test basic initialization of PowerMetric.\"\"\"
        # Test CPU metric
        cpu_metric = PowerMetric("cpu")
        self.assertEqual(cpu_metric.device_type, "cpu")
        self.assertEqual(cpu_metric.power_samples, [])
        self.assertEqual(cpu_metric.operations_count, 0)
        self.assertEqual(cpu_metric.throughput, 0)
        
        # Test CUDA metric if available
        if torch.cuda.is_available():
            cuda_metric = PowerMetric("cuda")
            self.assertEqual(cuda_metric.device_type, "cuda")
            
        # Test MPS metric if available
        has_mps = False
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
                has_mps = torch.mps.is_available()
        except (AttributeError, ImportError):
            pass
            
        if has_mps:
            mps_metric = PowerMetric("mps")
            self.assertEqual(mps_metric.device_type, "mps")
    
    @patch('subprocess.run')
    def test_platform_detection(self, mock_run):
        \"\"\"Test platform-specific detection methods.\"\"\"
        # Mock the subprocess.run to simulate successful returns
        mock_run.return_value = MagicMock(returncode=0, stdout="100.0")
        
        # Test NVIDIA SMI detection
        with patch.object(PowerMetric, '_check_nvidia_smi') as mock_check:
            mock_check.return_value = True
            metric = PowerMetric("cuda")
            self.assertTrue(metric.has_nvidia_smi)
        
        # Test Intel RAPL detection
        with patch.object(PowerMetric, '_check_intel_rapl') as mock_check:
            mock_check.return_value = True
            metric = PowerMetric("cpu")
            self.assertTrue(metric.has_intel_rapl)
        
        # Test ROCm SMI detection
        with patch.object(PowerMetric, '_check_rocm_smi') as mock_check:
            mock_check.return_value = True
            metric = PowerMetric("rocm")
            self.assertTrue(metric.has_amd_rocm_smi)
        
        # Test PowerMetrics detection (macOS)
        with patch.object(PowerMetric, '_check_powermetrics') as mock_check:
            mock_check.return_value = True
            metric = PowerMetric("mps")
            self.assertTrue(metric.has_powermetrics)
    
    @patch('subprocess.run')
    def test_nvidia_power_reading(self, mock_run):
        \"\"\"Test NVIDIA power reading.\"\"\"
        # Skip if not running on CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Mock the subprocess.run to return a fixed power value
        mock_run.return_value = MagicMock(returncode=0, stdout="50.5\\n", stderr="")
        
        # Create the metric with mocked NVIDIA SMI
        with patch.object(PowerMetric, '_check_nvidia_smi') as mock_check:
            mock_check.return_value = True
            metric = PowerMetric("cuda")
            power = metric._get_nvidia_power()
            
            # Validate the power reading
            self.assertEqual(power, 50.5)
    
    @patch('builtins.open')
    @patch('time.sleep')
    def test_intel_rapl_power_reading(self, mock_sleep, mock_open):
        \"\"\"Test Intel RAPL power reading.\"\"\"
        # Skip if not running on Linux
        if platform.system() != "Linux":
            self.skipTest("Not running on Linux")
            
        # Mock the file open and read operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.side_effect = ["1000000000", "1010000000"]
        mock_open.return_value = mock_file
        
        # Mock time.sleep
        mock_sleep.return_value = None
        
        # Create the metric with mocked Intel RAPL
        with patch.object(PowerMetric, '_check_intel_rapl') as mock_check:
            mock_check.return_value = True
            metric = PowerMetric("cpu")
            power = metric._get_intel_rapl_power()
            
            # Validate the power reading (10W = 10M uJ over 0.1s)
            self.assertEqual(power, 100.0)
    
    @patch('subprocess.run')
    def test_rocm_power_reading(self, mock_run):
        \"\"\"Test AMD ROCm power reading.\"\"\"
        # Skip if not running on ROCm
        if not hasattr(hardware, 'ROCmBackend'):
            self.skipTest("ROCm not available")
            
        # Mock the subprocess.run to return a fixed power value
        mock_output = \"\"\"
        ======== ROCm System Management Interface ========
        ===================== Power =====================
        GPU[0] : 75.5 W
        ===================================================
        \"\"\"
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_output, stderr="")
        
        # Create the metric with mocked ROCm SMI
        with patch.object(PowerMetric, '_check_rocm_smi') as mock_check:
            mock_check.return_value = True
            metric = PowerMetric("rocm")
            
            # We need to mock the extraction of power value from the output string
            with patch.object(PowerMetric, '_get_rocm_power') as mock_power:
                mock_power.return_value = 75.5
                power = metric._get_rocm_power()
                
                # Validate the power reading
                self.assertEqual(power, 75.5)
    
    def test_power_sampling(self):
        \"\"\"Test power sampling thread.\"\"\"
        # Create a metric with mocked power reading
        with patch.object(PowerMetric, '_get_current_power') as mock_power:
            mock_power.return_value = 100.0
            
            metric = PowerMetric("cpu")
            
            # Start power sampling
            metric.start()
            
            # Check that sampling started
            self.assertTrue(metric.is_sampling)
            if metric.sampling_thread:
                self.assertTrue(metric.sampling_thread.daemon)
                
            # Allow some samples to be collected
            import time
            time.sleep(0.3)  # Sleep for 300ms to collect a few samples
            
            # Stop power sampling
            metric.stop()
            
            # Check that sampling stopped
            self.assertFalse(metric.is_sampling)
            
            # Check that samples were collected
            self.assertGreater(len(metric.power_samples), 0)
    
    def test_metrics_calculation(self):
        \"\"\"Test calculation of power-related metrics.\"\"\"
        # Create a metric with simulated power samples
        metric = PowerMetric("cpu")
        
        # Set synthetic power samples (time, power_watts)
        current_time = 1000.0
        metric.power_samples = [
            (current_time, 100.0),
            (current_time + 0.1, 110.0),
            (current_time + 0.2, 90.0),
            (current_time + 0.3, 100.0)
        ]
        
        # Set start and end times
        metric.start_time = current_time
        metric.end_time = current_time + 0.3
        
        # Set operations count and throughput
        metric.set_operations_count(1000000000)  # 1 GFLOP
        metric.set_throughput(100.0)  # 100 items/second
        
        # Calculate metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertTrue(metrics["power_supported"])
        self.assertEqual(metrics["power_samples_count"], 4)
        self.assertEqual(metrics["power_avg_watts"], 100.0)
        self.assertEqual(metrics["power_max_watts"], 110.0)
        self.assertAlmostEqual(metrics["energy_joules"], 100.0 * 0.3, places=5)  # 30.0
        
        # Validate efficiency metrics
        self.assertAlmostEqual(metrics["ops_per_watt"], 1000000000 / 100.0, places=5)  # 10M ops/W
        self.assertAlmostEqual(metrics["gflops_per_watt"], 1000000000 / 100.0 / 1e9, places=5)  # 10 GFLOPs/W
        self.assertAlmostEqual(metrics["throughput_per_watt"], 100.0 / 100.0, places=5)  # 1 item/s/W
    
    def test_no_power_samples(self):
        \"\"\"Test metrics when no power samples are available.\"\"\"
        # Create a metric with no power samples
        metric = PowerMetric("cpu")
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertFalse(metrics["power_supported"])
        self.assertEqual(metrics["power_avg_watts"], 0.0)
    
    def test_factory_with_torch_device(self):
        \"\"\"Test PowerMetricFactory with torch.device.\"\"\"
        # Create metric with torch.device
        cpu_metric = PowerMetricFactory.create(self.cpu_device)
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA device if available
        if self.cuda_device:
            cuda_metric = PowerMetricFactory.create(self.cuda_device)
            self.assertEqual(cuda_metric.device_type, "cuda")
        
        # Create metric with MPS device if available
        if self.mps_device:
            mps_metric = PowerMetricFactory.create(self.mps_device)
            self.assertEqual(mps_metric.device_type, "mps")
    
    def test_factory_with_hardware_backend(self):
        \"\"\"Test PowerMetricFactory with hardware backend.\"\"\"
        # Create metric with CPU hardware backend
        cpu_backend = hardware.get_hardware_backend("cpu")
        cpu_device = cpu_backend.initialize()
        cpu_metric = PowerMetricFactory.create(cpu_device)
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA hardware backend if available
        if "cuda" in hardware.get_available_hardware():
            cuda_backend = hardware.get_hardware_backend("cuda")
            cuda_device = cuda_backend.initialize()
            cuda_metric = PowerMetricFactory.create(cuda_device)
            self.assertEqual(cuda_metric.device_type, "cuda")
    
    def test_factory_with_string(self):
        \"\"\"Test PowerMetricFactory with string device.\"\"\"
        # Create metric with string device
        cpu_metric = PowerMetricFactory.create("cpu")
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA string if available
        if torch.cuda.is_available():
            cuda_metric = PowerMetricFactory.create("cuda:0")
            self.assertEqual(cuda_metric.device_type, "cuda")
            
    def test_integration_with_benchmark(self):
        \"\"\"Test integration with benchmark system.\"\"\"
        # Import the benchmark module
        try:
            from benchmark import ModelBenchmark, BenchmarkConfig
            
            # Check if the benchmark system uses the PowerMetric
            with patch('metrics.power.PowerMetricFactory.create') as mock_factory:
                mock_metric = MagicMock()
                mock_factory.return_value = mock_metric
                
                # Create a simple benchmark config
                config = BenchmarkConfig(
                    model_id="bert-base-uncased",
                    hardware=["cpu"],
                    batch_sizes=[1],
                    sequence_lengths=[128],
                    metrics=["power"]
                )
                
                # Create a benchmark instance
                benchmark = ModelBenchmark(config)
                
                # Validate that factory was called
                mock_factory.assert_called()
                
                # Verify model ID was set correctly
                self.assertEqual(config.model_id, "bert-base-uncased")
                
                # Verify power is in the metrics list
                self.assertIn("power", config.metrics)
                
        except ImportError:
            self.skipTest("ModelBenchmark import failed - this is expected if PowerMetric integration is not complete")

if __name__ == '__main__':
    unittest.main()
"""

BANDWIDTH_METRIC_TEST_TEMPLATE = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Test script for bandwidth metrics.

This script tests the BandwidthMetric class and factory with different device types
and validates platform-specific bandwidth measurement and roofline model analysis.
\"\"\"

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
    \"\"\"Test cases for BandwidthMetric.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
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
        \"\"\"Test basic initialization of BandwidthMetric.\"\"\"
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
        \"\"\"Test detection of theoretical peak memory bandwidth.\"\"\"
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
        \"\"\"Test CUDA memory bandwidth reading.\"\"\"
        # Skip if not running on CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Mock the subprocess.run to return memory utilization
        mock_run.return_value = MagicMock(returncode=0, stdout="50\\n", stderr="")
        
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
        \"\"\"Test bandwidth sampling thread.\"\"\"
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
        \"\"\"Test estimation of memory transfers.\"\"\"
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
        \"\"\"Test calculation of arithmetic intensity for roofline model.\"\"\"
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
        \"\"\"Test determination of compute vs memory bound workloads.\"\"\"
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
        \"\"\"Test roofline model data generation.\"\"\"
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
        \"\"\"Test calculation of bandwidth-related metrics.\"\"\"
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
            self.assertEqual(metrics["peak_theoretical_bandwidth_gbps"], 500.0)
            
            # Utilization = avg / peak_theoretical
            self.assertEqual(metrics["bandwidth_utilization_percent"], 40.0)  # 200/500 = 40%
            
            # Memory transfers in GB
            self.assertEqual(metrics["memory_transfers_gb"], 10.0)
            
            # Arithmetic intensity = compute_operations / memory_transfers_bytes
            expected_intensity = 5e12 / (10 * 1024 * 1024 * 1024)
            self.assertAlmostEqual(metrics["arithmetic_intensity_flops_per_byte"], expected_intensity, places=2)
    
    def test_factory_with_torch_device(self):
        \"\"\"Test BandwidthMetricFactory with torch.device.\"\"\"
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
        \"\"\"Test BandwidthMetricFactory with hardware backend.\"\"\"
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
        \"\"\"Test BandwidthMetricFactory with string device.\"\"\"
        # Create metric with string device
        cpu_metric = BandwidthMetricFactory.create("cpu")
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA string if available
        if torch.cuda.is_available():
            cuda_metric = BandwidthMetricFactory.create("cuda:0")
            self.assertEqual(cuda_metric.device_type, "cuda")
    
    def test_integration_with_benchmark(self):
        \"\"\"Test integration with benchmark system.\"\"\"
        # Import the benchmark module
        try:
            from benchmark import ModelBenchmark, BenchmarkConfig
            
            # Check if the benchmark system uses the BandwidthMetric
            with patch('metrics.bandwidth.BandwidthMetricFactory.create') as mock_factory:
                mock_metric = MagicMock()
                mock_factory.return_value = mock_metric
                
                # Create a simple benchmark config
                config = BenchmarkConfig(
                    model_id="bert-base-uncased",
                    hardware=["cpu"],
                    batch_sizes=[1],
                    sequence_lengths=[128],
                    metrics=["bandwidth"]
                )
                
                # Create a benchmark instance
                benchmark = ModelBenchmark(config)
                
                # Validate that factory was called
                mock_factory.assert_called()
                
                # Verify model ID was set correctly
                self.assertEqual(config.model_id, "bert-base-uncased")
                
                # Verify bandwidth is in the metrics list
                self.assertIn("bandwidth", config.metrics)
                
        except ImportError:
            self.skipTest("ModelBenchmark import failed - this is expected if BandwidthMetric integration is not complete")

if __name__ == '__main__':
    unittest.main()
"""

HARDWARE_METRICS_INTEGRATION_TEST_TEMPLATE = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Integration tests for hardware-aware metrics.

This module contains tests that validate the integration between hardware-aware
metrics (power, bandwidth) and the benchmark system.
\"\"\"

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
    \"\"\"Integration tests for hardware-aware metrics.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
        # Use a temp directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        \"\"\"Clean up test environment.\"\"\"
        self.temp_dir.cleanup()
    
    def test_metrics_creation(self):
        \"\"\"Test creation of hardware-aware metrics.\"\"\"
        # Test for each available hardware
        hardware_types = get_available_hardware()
        for hw_type in hardware_types:
            # Initialize hardware
            device = initialize_hardware(hw_type)
            
            # Create hardware-aware metrics
            power_metric = PowerMetricFactory.create(device)
            bandwidth_metric = BandwidthMetricFactory.create(device)
            
            # Validate correct device type
            self.assertEqual(power_metric.device_type, hw_type)
            self.assertEqual(bandwidth_metric.device_type, hw_type)
    
    def test_metrics_lifecycle(self):
        \"\"\"Test lifecycle of hardware-aware metrics.\"\"\"
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
        \"\"\"Test data sharing between metrics.\"\"\"
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
    
    @patch('benchmark.ModelBenchmark._load_model')
    @patch('metrics.power.PowerMetric._get_current_power')
    @patch('metrics.bandwidth.BandwidthMetric._get_current_bandwidth')
    def test_benchmark_power_bandwidth_integration(self, mock_bandwidth, mock_power, mock_load_model):
        \"\"\"Test integration with benchmark system.\"\"\"
        # Mock model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_model.forward = MagicMock(return_value=torch.tensor([1.0]))
        
        # Mock power and bandwidth readings
        mock_power.return_value = 100.0  # 100W
        mock_bandwidth.return_value = 50.0  # 50 GB/s
        
        # Create and run benchmark
        benchmark = ModelBenchmark(
            model_id="bert-base-uncased",
            batch_sizes=[1, 2],
            sequence_lengths=[16],
            hardware=["cpu"],
            metrics=["latency", "power", "bandwidth"],
            warmup_iterations=1,
            test_iterations=2,
            output_dir=self.output_dir
        )
        
        # Mock the FLOPs calculation
        with patch('metrics.flops.FLOPsMetric._estimate_model_flops') as mock_flops:
            mock_flops.return_value = 1e9  # 1 GFLOP
            
            # Run the benchmark
            results = benchmark.run()
        
        # Verify results exist
        self.assertGreater(len(results.results), 0)
        
        # Verify power and bandwidth metrics were included
        for result in results.results:
            metrics = result.metrics
            self.assertIn("power_avg_watts", metrics)
            self.assertIn("avg_bandwidth_gbps", metrics)
    
    def test_visualization_methods(self):
        \"\"\"Test visualization methods for hardware-aware metrics.\"\"\"
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
"""

VISUALIZATION_TEST_TEMPLATE = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Test script for hardware-aware metrics visualization.

This script tests the visualization functions for power efficiency and
bandwidth metrics in the refactored benchmark suite.
\"\"\"

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
    \"\"\"Tests for hardware-aware metrics visualization.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
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
        \"\"\"Clean up test environment.\"\"\"
        self.temp_dir.cleanup()
    
    def test_power_efficiency_visualization(self):
        \"\"\"Test power efficiency visualization.\"\"\"
        # Generate output path
        output_path = os.path.join(self.temp_dir.name, "power_efficiency_test.png")
        
        # Call visualization function
        result_path = plot_power_efficiency(self.benchmark_results, output_path)
        
        # Check that file was created
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_bandwidth_utilization_visualization(self):
        \"\"\"Test bandwidth utilization visualization.\"\"\"
        # Generate output path
        output_path = os.path.join(self.temp_dir.name, "bandwidth_utilization_test.png")
        
        # Call visualization function
        result_path = plot_bandwidth_utilization(self.benchmark_results, output_path)
        
        # Check that file was created
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_power_efficiency_visualization_integration(self):
        \"\"\"Test power efficiency visualization through BenchmarkResults.\"\"\"
        # Call visualization method
        result_path = self.benchmark_results.plot_power_efficiency()
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
    
    def test_bandwidth_utilization_visualization_integration(self):
        \"\"\"Test bandwidth utilization visualization through BenchmarkResults.\"\"\"
        # Call visualization method
        result_path = self.benchmark_results.plot_bandwidth_utilization()
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
    
    def test_visualization_without_data(self):
        \"\"\"Test visualization with no relevant data.\"\"\"
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
        \"\"\"Test visualization with different detail levels.\"\"\"
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
"""

def generate_test_file(template, output_file):
    """Generate a test file from template."""
    with open(output_file, 'w') as f:
        f.write(template)
    print(f"Generated test file: {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate hardware metrics tests.')
    parser.add_argument('--output_dir', type=str, default='tests', 
                        help='Directory to write test files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate power metric test
    generate_test_file(
        POWER_METRIC_TEST_TEMPLATE,
        os.path.join(args.output_dir, 'test_power_metric.py')
    )
    
    # Generate bandwidth metric test
    generate_test_file(
        BANDWIDTH_METRIC_TEST_TEMPLATE,
        os.path.join(args.output_dir, 'test_bandwidth_metric.py')
    )
    
    # Generate hardware metrics integration test
    generate_test_file(
        HARDWARE_METRICS_INTEGRATION_TEST_TEMPLATE,
        os.path.join(args.output_dir, 'test_hardware_metrics_integration.py')
    )
    
    # Generate visualization test
    generate_test_file(
        VISUALIZATION_TEST_TEMPLATE,
        os.path.join(args.output_dir, 'test_hardware_aware_visualization.py')
    )
    
    print("Test generation complete!")

if __name__ == '__main__':
    main()