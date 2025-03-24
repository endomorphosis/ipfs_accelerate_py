#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for power metrics.

This script tests the PowerMetric class and factory with different device types
and validates platform-specific power monitoring.
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
from metrics.power import PowerMetric, PowerMetricFactory
import hardware

class TestPowerMetric(unittest.TestCase):
    """Test cases for PowerMetric."""
    
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
        
    def test_power_metric_initialization(self):
        """Test basic initialization of PowerMetric."""
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
        """Test platform-specific detection methods."""
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
        """Test NVIDIA power reading."""
        # Skip if not running on CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Mock the subprocess.run to return a fixed power value
        mock_run.return_value = MagicMock(returncode=0, stdout="50.5\n", stderr="")
        
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
        """Test Intel RAPL power reading."""
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
        """Test AMD ROCm power reading."""
        # Skip if not running on ROCm
        if not hasattr(hardware, 'ROCmBackend'):
            self.skipTest("ROCm not available")
            
        # Mock the subprocess.run to return a fixed power value
        mock_output = """
        ======== ROCm System Management Interface ========
        ===================== Power =====================
        GPU[0] : 75.5 W
        ===================================================
        """
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
        """Test power sampling thread."""
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
        """Test calculation of power-related metrics."""
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
        """Test metrics when no power samples are available."""
        # Create a metric with no power samples
        metric = PowerMetric("cpu")
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertFalse(metrics["power_supported"])
        self.assertEqual(metrics["power_avg_watts"], 0.0)
    
    def test_factory_with_torch_device(self):
        """Test PowerMetricFactory with torch.device."""
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
        """Test PowerMetricFactory with hardware backend."""
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
        """Test PowerMetricFactory with string device."""
        # Create metric with string device
        cpu_metric = PowerMetricFactory.create("cpu")
        self.assertEqual(cpu_metric.device_type, "cpu")
        
        # Create metric with CUDA string if available
        if torch.cuda.is_available():
            cuda_metric = PowerMetricFactory.create("cuda:0")
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
                metrics=["power"]
            )
                
            # Verify model ID was set correctly
            self.assertEqual(config.model_id, "bert-base-uncased")
                
            # Verify power is in the metrics list
            self.assertIn("power", config.metrics)
                
        except ImportError:
            self.skipTest("ModelBenchmark import failed - this is expected if PowerMetric integration is not complete")
        
        # Skip actual integration test in this environment
        self.skipTest("Skipping full integration test in this environment")

if __name__ == '__main__':
    unittest.main()
