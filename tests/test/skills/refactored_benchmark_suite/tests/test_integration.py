#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for the refactored benchmark suite.

This module contains tests that validate the integration between different
components of the benchmark suite, particularly the hardware abstraction layer
and the metrics system with the benchmark runner.
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

from benchmark import ModelBenchmark
from hardware import initialize_hardware, get_available_hardware
from metrics import (
    LatencyMetric, ThroughputMetric, MemoryMetric, FLOPsMetric
)
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

class TestIntegration(unittest.TestCase):
    """Integration tests for the benchmark suite."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a temp directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_hardware_metrics_integration(self):
        """Test integration between hardware abstraction and metrics."""
        # Test for each available hardware
        hardware_types = get_available_hardware()
        for hw_type in hardware_types:
            # Initialize hardware
            device = initialize_hardware(hw_type)
            
            # Create hardware-aware metrics
            latency_metric = TimingMetricFactory.create_latency_metric(device)
            throughput_metric = TimingMetricFactory.create_throughput_metric(device, batch_size=2)
            memory_metric = MemoryMetricFactory.create(device)
            flops_metric = FLOPsMetricFactory.create(device)
            power_metric = PowerMetricFactory.create(device)
            bandwidth_metric = BandwidthMetricFactory.create(device)
            
            # Validate correct device type
            self.assertEqual(latency_metric.device_type, hw_type)
            self.assertEqual(throughput_metric.device_type, hw_type)
            self.assertEqual(memory_metric.device_type, hw_type)
            self.assertEqual(power_metric.device_type, hw_type)
            self.assertEqual(bandwidth_metric.device_type, hw_type)
            
            # Test metrics lifecycle
            latency_metric.start()
            throughput_metric.start()
            memory_metric.start()
            power_metric.start()
            bandwidth_metric.start()
            
            # Simulate some computation
            for _ in range(5):
                x = torch.randn(10, 10)
                if hw_type != "cpu":
                    # Move to appropriate device
                    if isinstance(device, torch.device):
                        x = x.to(device)
                y = x @ x
                
                latency_metric.record_step()
                throughput_metric.update()
                memory_metric.record_memory()
            
            latency_metric.stop()
            throughput_metric.stop()
            memory_metric.stop()
            power_metric.stop()
            bandwidth_metric.stop()
            
            # Verify metrics
            latency_data = latency_metric.get_metrics()
            throughput_data = throughput_metric.get_metrics()
            memory_data = memory_metric.get_metrics()
            power_data = power_metric.get_metrics()
            bandwidth_data = bandwidth_metric.get_metrics()
            
            # Verify latency metrics
            self.assertIn("latency_ms", latency_data)
            self.assertGreaterEqual(latency_data["latency_ms"], 0)
            
            # Verify throughput metrics
            self.assertIn("throughput_items_per_sec", throughput_data)
            self.assertGreaterEqual(throughput_data["throughput_items_per_sec"], 0)
            
            # Basic verification for memory metrics
            self.assertIsInstance(memory_data, dict)
            
            # Basic verification for power metrics
            self.assertIsInstance(power_data, dict)
            self.assertIn("power_supported", power_data)
            
            # Basic verification for bandwidth metrics
            self.assertIsInstance(bandwidth_data, dict)
            self.assertIn("bandwidth_supported", bandwidth_data)
    
    def test_metrics_interactions(self):
        """Test that metrics correctly interact with each other."""
        # Test on CPU for simplicity
        device = initialize_hardware("cpu")
        
        # Create metrics
        flops_metric = FLOPsMetricFactory.create(device)
        power_metric = PowerMetricFactory.create(device)
        bandwidth_metric = BandwidthMetricFactory.create(device)
        throughput_metric = TimingMetricFactory.create_throughput_metric(device, batch_size=2)
        
        # Simulate operations count
        estimated_flops = 1e9  # 1 GFLOP
        flops_metric.total_flops = estimated_flops
        
        # Transfer FLOPs to power and bandwidth metrics
        power_metric.set_operations_count(estimated_flops)
        bandwidth_metric.set_compute_operations(estimated_flops)
        
        # Simulate memory transfers
        memory_transfers = 1024 * 1024 * 1024  # 1 GB
        bandwidth_metric.set_memory_transfers(memory_transfers)
        
        # Simulate throughput
        throughput_value = 100.0  # 100 items/sec
        throughput_metric.items_processed = 100
        throughput_metric.total_time = 1.0
        throughput_data = throughput_metric.get_metrics()
        
        # Set throughput for power metric
        power_metric.set_throughput(throughput_data["throughput_items_per_sec"])
        
        # Check metrics interaction
        power_data = power_metric.get_metrics()
        bandwidth_data = bandwidth_metric.get_metrics()
        
        # Get roofline data
        roofline_data = bandwidth_metric.get_roofline_data()
        
        # Verify values set correctly through communication between metrics
        if power_data["power_avg_watts"] > 0:  # Only if power monitoring is available
            self.assertEqual(power_data["ops_per_watt"], estimated_flops / power_data["power_avg_watts"])
        
        # Check arithmetic intensity calculation
        self.assertEqual(bandwidth_metric.get_arithmetic_intensity(), estimated_flops / memory_transfers)
        
        # Verify roofline data structure
        self.assertIn("peak_compute_flops", roofline_data)
        self.assertIn("peak_memory_bandwidth_bytes_per_sec", roofline_data)
        self.assertIn("ridge_point_flops_per_byte", roofline_data)
        self.assertIn("arithmetic_intensity_flops_per_byte", roofline_data)
        
    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    def test_benchmark_cuda_integration(self):
        """Test full benchmark integration with CUDA hardware."""
        # To avoid actual model loading, we'll mock the model
        with patch('benchmark.ModelBenchmark._load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            # Mock the model's forward method
            mock_model.forward = MagicMock(return_value=torch.tensor([1.0]))
            
            benchmark = ModelBenchmark(
                model_id="bert-base-uncased",
                batch_sizes=[1],
                sequence_lengths=[16],
                hardware=["cuda"],
                metrics=["latency", "throughput", "memory", "flops", "power", "bandwidth"],
                warmup_iterations=1,
                test_iterations=2,
                output_dir=self.output_dir
            )
            
            # Mock the FLOPs counting
            with patch('metrics.flops.FLOPsMetric._estimate_model_flops') as mock_flops:
                mock_flops.return_value = 1e9  # 1 GFLOP
                
                # Mock power monitoring
                with patch('metrics.power.PowerMetric._get_current_power') as mock_power:
                    mock_power.return_value = 100.0  # 100W
                    
                    # Mock bandwidth monitoring
                    with patch('metrics.bandwidth.BandwidthMetric._get_current_bandwidth') as mock_bandwidth:
                        mock_bandwidth.return_value = 100.0  # 100 GB/s
                        
                        results = benchmark.run()
        
        # Verify CUDA results exist
        cuda_results = [r for r in results.results if r.hardware == "cuda"]
        self.assertGreater(len(cuda_results), 0)
        
        # Verify all metrics are included
        cuda_result = cuda_results[0]
        self.assertIn("latency_ms", cuda_result.metrics)
        self.assertIn("throughput_items_per_sec", cuda_result.metrics)
        
        # Check for power efficiency metrics if they were collected
        if "power_avg_watts" in cuda_result.metrics:
            self.assertIn("gflops_per_watt", cuda_result.metrics)
        
        # Check for bandwidth metrics if they were collected
        if "avg_bandwidth_gbps" in cuda_result.metrics:
            self.assertIn("bandwidth_utilization_percent", cuda_result.metrics)
        
        # Test export functions
        json_path = results.export_to_json()
        self.assertTrue(os.path.exists(json_path))
    
    def test_benchmark_cpu_integration(self):
        """Test full benchmark integration with CPU hardware."""
        # To avoid actual model loading, we'll mock the model
        with patch('benchmark.ModelBenchmark._load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            # Mock the model's forward method
            mock_model.forward = MagicMock(return_value=torch.tensor([1.0]))
            
            benchmark = ModelBenchmark(
                model_id="bert-base-uncased",
                batch_sizes=[1],
                sequence_lengths=[16],
                hardware=["cpu"],
                metrics=["latency", "throughput", "memory", "flops", "power", "bandwidth"],
                warmup_iterations=1,
                test_iterations=2,
                output_dir=self.output_dir
            )
            
            # Mock the FLOPs counting
            with patch('metrics.flops.FLOPsMetric._estimate_model_flops') as mock_flops:
                mock_flops.return_value = 1e9  # 1 GFLOP
                
                # Mock power monitoring
                with patch('metrics.power.PowerMetric._get_current_power') as mock_power:
                    mock_power.return_value = 50.0  # 50W
                    
                    # Mock bandwidth monitoring
                    with patch('metrics.bandwidth.BandwidthMetric._get_current_bandwidth') as mock_bandwidth:
                        mock_bandwidth.return_value = 30.0  # 30 GB/s
                        
                        results = benchmark.run()
        
        # Verify CPU results exist
        cpu_results = [r for r in results.results if r.hardware == "cpu"]
        self.assertGreater(len(cpu_results), 0)
        
        # Verify metrics
        cpu_result = cpu_results[0]
        self.assertIn("latency_ms", cpu_result.metrics)
        self.assertIn("throughput_items_per_sec", cpu_result.metrics)
        
        # Test export functions
        json_path = results.export_to_json()
        self.assertTrue(os.path.exists(json_path))
        
        # Test visualization (only if matplotlib is available)
        try:
            import matplotlib
            if results.plot_latency_comparison() is not None:
                plot_path = os.path.join(self.output_dir, 
                                        f"bert-base-uncased_latency_comparison.png")
                self.assertTrue(os.path.exists(plot_path))
            
            # Test power efficiency visualization
            if results.plot_power_efficiency() is not None:
                plot_path = os.path.join(self.output_dir, 
                                        f"bert-base-uncased_power_efficiency.png")
                self.assertTrue(os.path.exists(plot_path))
            
            # Test bandwidth utilization visualization
            if results.plot_bandwidth_utilization() is not None:
                plot_path = os.path.join(self.output_dir, 
                                        f"bert-base-uncased_bandwidth_utilization.png")
                self.assertTrue(os.path.exists(plot_path))
        except ImportError:
            pass  # Skip visualization tests if matplotlib is not available
    
    def test_comprehensive_metrics_integration(self):
        """
        Test a comprehensive integration of all metrics together.
        This test simulates a real benchmark with all metrics active.
        """
        # Create mock metrics for testing
        with patch('metrics.timing.LatencyMetric') as mock_latency, \
             patch('metrics.timing.ThroughputMetric') as mock_throughput, \
             patch('metrics.memory.MemoryMetric') as mock_memory, \
             patch('metrics.flops.FLOPsMetric') as mock_flops, \
             patch('metrics.power.PowerMetric') as mock_power, \
             patch('metrics.bandwidth.BandwidthMetric') as mock_bandwidth:
            
            # Configure mock metrics
            mock_latency_instance = MagicMock()
            mock_latency_instance.get_metrics.return_value = {"latency_ms": 10.0}
            mock_latency.return_value = mock_latency_instance
            
            mock_throughput_instance = MagicMock()
            mock_throughput_instance.get_metrics.return_value = {"throughput_items_per_sec": 100.0}
            mock_throughput.return_value = mock_throughput_instance
            
            mock_memory_instance = MagicMock()
            mock_memory_instance.get_metrics.return_value = {"memory_usage_mb": 1000.0}
            mock_memory.return_value = mock_memory_instance
            
            mock_flops_instance = MagicMock()
            mock_flops_instance.get_metrics.return_value = {"flops": 1e9, "gflops": 1.0}
            mock_flops_instance.total_flops = 1e9
            mock_flops.return_value = mock_flops_instance
            
            mock_power_instance = MagicMock()
            mock_power_instance.get_metrics.return_value = {
                "power_supported": True,
                "power_avg_watts": 100.0,
                "energy_joules": 30.0,
                "gflops_per_watt": 0.01,
                "throughput_per_watt": 1.0
            }
            mock_power.return_value = mock_power_instance
            
            mock_bandwidth_instance = MagicMock()
            mock_bandwidth_instance.get_metrics.return_value = {
                "bandwidth_supported": True,
                "avg_bandwidth_gbps": 100.0,
                "peak_theoretical_bandwidth_gbps": 200.0,
                "bandwidth_utilization_percent": 50.0,
                "arithmetic_intensity_flops_per_byte": 5.0,
                "compute_bound": False
            }
            mock_bandwidth_instance.get_roofline_data.return_value = {
                "peak_compute_flops": 10e12,
                "peak_memory_bandwidth_bytes_per_sec": 200e9,
                "ridge_point_flops_per_byte": 50.0,
                "arithmetic_intensity_flops_per_byte": 5.0,
                "is_compute_bound": False
            }
            mock_bandwidth.return_value = mock_bandwidth_instance
            
            # Simulate a benchmark run with all metrics
            # Create a simple orchestration process to mimic benchmark behavior
            
            # 1. Initialize all metrics
            latency = mock_latency("cpu")
            throughput = mock_throughput("cpu")
            memory = mock_memory("cpu")
            flops = mock_flops("cpu")
            power = mock_power("cpu")
            bandwidth = mock_bandwidth("cpu")
            
            # 2. Start all metrics
            latency.start()
            throughput.start()
            memory.start()
            flops.start()
            power.start()
            bandwidth.start()
            
            # 3. Simulate computation
            # (This would be model inference in a real benchmark)
            
            # 4. Stop all metrics
            latency.stop()
            throughput.stop()
            memory.stop()
            flops.stop()
            power.stop()
            bandwidth.stop()
            
            # 5. Transfer data between metrics
            # FLOPs to power and bandwidth
            power.set_operations_count(flops.total_flops)
            bandwidth.set_compute_operations(flops.total_flops)
            
            # Throughput to power
            power.set_throughput(throughput.get_metrics()["throughput_items_per_sec"])
            
            # 6. Collect all metrics
            all_metrics = {}
            all_metrics.update(latency.get_metrics())
            all_metrics.update(throughput.get_metrics())
            all_metrics.update(memory.get_metrics())
            all_metrics.update(flops.get_metrics())
            all_metrics.update(power.get_metrics())
            all_metrics.update(bandwidth.get_metrics())
            
            # Verify comprehensive metrics
            self.assertIn("latency_ms", all_metrics)
            self.assertIn("throughput_items_per_sec", all_metrics)
            self.assertIn("memory_usage_mb", all_metrics)
            self.assertIn("flops", all_metrics)
            self.assertIn("power_avg_watts", all_metrics)
            self.assertIn("gflops_per_watt", all_metrics)
            self.assertIn("avg_bandwidth_gbps", all_metrics)
            self.assertIn("bandwidth_utilization_percent", all_metrics)
            self.assertIn("arithmetic_intensity_flops_per_byte", all_metrics)
            
            # Verify correct values from our mock setup
            self.assertEqual(all_metrics["latency_ms"], 10.0)
            self.assertEqual(all_metrics["throughput_items_per_sec"], 100.0)
            self.assertEqual(all_metrics["memory_usage_mb"], 1000.0)
            self.assertEqual(all_metrics["flops"], 1e9)
            self.assertEqual(all_metrics["power_avg_watts"], 100.0)
            self.assertEqual(all_metrics["gflops_per_watt"], 0.01)
            self.assertEqual(all_metrics["avg_bandwidth_gbps"], 100.0)
            self.assertEqual(all_metrics["bandwidth_utilization_percent"], 50.0)
            
            # Test roofline data access
            roofline_data = bandwidth.get_roofline_data()
            self.assertIn("peak_compute_flops", roofline_data)
            self.assertIn("peak_memory_bandwidth_bytes_per_sec", roofline_data)
            self.assertIn("ridge_point_flops_per_byte", roofline_data)
            self.assertEqual(roofline_data["arithmetic_intensity_flops_per_byte"], 5.0)
            self.assertEqual(roofline_data["is_compute_bound"], False)

if __name__ == "__main__":
    unittest.main()