#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for timing metrics.

This script tests the timing metrics and factory with different device types.
"""

import os
import sys
import unittest
import time
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from metrics.timing import LatencyMetric, ThroughputMetric, TimingMetricFactory
import hardware

class TestTimingMetrics(unittest.TestCase):
    """Test cases for timing metrics."""
    
    def test_latency_metric_cpu(self):
        """Test CPU latency metric."""
        # Create CPU latency metric
        metric = LatencyMetric(device_type="cpu")
        
        # Test metric
        metric.start()
        
        # Simulate inference latency
        for _ in range(3):
            time.sleep(0.01)  # Sleep for 10ms
            metric.record_step()
        
        # Stop metric
        metric.stop()
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertIn("latency_ms", metrics)
        self.assertIn("latency_min_ms", metrics)
        self.assertIn("latency_max_ms", metrics)
        self.assertIn("latency_std_ms", metrics)
        self.assertIn("latency_median_ms", metrics)
        self.assertIn("latency_p90_ms", metrics)
        self.assertIn("latency_p95_ms", metrics)
        self.assertIn("latency_p99_ms", metrics)
        
        # Check that latencies are roughly in the expected range
        self.assertGreaterEqual(metrics["latency_ms"], 5.0)  # At least 5ms
        self.assertLessEqual(metrics["latency_ms"], 30.0)  # At most 30ms
        
        # Get latency distribution
        distribution = metric.get_latency_distribution()
        
        # Validate distribution
        self.assertIn("latencies_ms", distribution)
        self.assertIn("timestamps", distribution)
        self.assertEqual(len(distribution["latencies_ms"]), 3)
    
    def test_throughput_metric_cpu(self):
        """Test CPU throughput metric."""
        # Create CPU throughput metric
        batch_size = 16
        metric = ThroughputMetric(batch_size=batch_size, device_type="cpu")
        
        # Test metric
        metric.start()
        
        # Simulate batch processing
        for _ in range(5):
            time.sleep(0.01)  # Sleep for 10ms
            metric.update()
        
        # Stop metric
        metric.stop()
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertIn("throughput_items_per_sec", metrics)
        self.assertIn("throughput_batches_per_sec", metrics)
        self.assertIn("total_items_processed", metrics)
        self.assertIn("time_per_batch_ms", metrics)
        
        # Check that the total items processed is correct
        self.assertEqual(metrics["total_items_processed"], 5 * batch_size)
        
        # Check that the batch time is roughly correct
        self.assertGreaterEqual(metrics["time_per_batch_ms"], 8.0)  # At least 8ms
        self.assertLessEqual(metrics["time_per_batch_ms"], 30.0)  # At most 30ms
        
        # Get batch times
        batch_times = metric.get_batch_times()
        
        # Validate batch times
        self.assertEqual(len(batch_times), 5)
    
    def test_throughput_with_custom_items(self):
        """Test throughput metric with custom items per batch."""
        # Create throughput metric with custom items per batch
        batch_size = 2
        items_per_batch = 100  # 100 tokens per batch, for example
        metric = ThroughputMetric(batch_size=batch_size, items_per_batch=items_per_batch)
        
        # Test metric
        metric.start()
        
        # Simulate batch processing with varying items per batch
        items_counts = [90, 110, 95, 105]
        for items in items_counts:
            time.sleep(0.01)  # Sleep for 10ms
            metric.update(items_processed=items)
        
        # Stop metric
        metric.stop()
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertEqual(metrics["total_items_processed"], sum(items_counts))
    
    def test_cuda_metrics_if_available(self):
        """Test CUDA metrics if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create CUDA latency metric
        latency_metric = LatencyMetric(device_type="cuda")
        
        # Check that synchronization is available
        self.assertTrue(latency_metric.can_synchronize)
        
        # Create CUDA throughput metric
        throughput_metric = ThroughputMetric(batch_size=16, device_type="cuda")
        
        # Check that synchronization is available
        self.assertTrue(throughput_metric.can_synchronize)
    
    def test_timing_factory_with_torch_device(self):
        """Test timing metric factory with torch.device."""
        # Create metrics with torch.device
        cpu_device = torch.device("cpu")
        latency_metric = TimingMetricFactory.create_latency_metric(cpu_device)
        throughput_metric = TimingMetricFactory.create_throughput_metric(cpu_device, batch_size=16)
        
        # Validate device types
        self.assertEqual(latency_metric.device_type, "cpu")
        self.assertEqual(throughput_metric.device_type, "cpu")
        
        # Create metrics with CUDA device if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            latency_metric = TimingMetricFactory.create_latency_metric(cuda_device)
            throughput_metric = TimingMetricFactory.create_throughput_metric(cuda_device, batch_size=16)
            
            # Validate device types
            self.assertEqual(latency_metric.device_type, "cuda")
            self.assertEqual(throughput_metric.device_type, "cuda")
    
    def test_timing_factory_with_hardware_backend(self):
        """Test timing metric factory with hardware backend device."""
        # Create metrics with CPU hardware backend
        cpu_backend = hardware.get_hardware_backend("cpu")
        cpu_device = cpu_backend.initialize()
        latency_metric = TimingMetricFactory.create_latency_metric(cpu_device)
        throughput_metric = TimingMetricFactory.create_throughput_metric(cpu_device, batch_size=16)
        
        # Validate device types
        self.assertEqual(latency_metric.device_type, "cpu")
        self.assertEqual(throughput_metric.device_type, "cpu")
        
        # Create metrics with CUDA hardware backend if available
        if "cuda" in hardware.get_available_hardware():
            cuda_backend = hardware.get_hardware_backend("cuda")
            cuda_device = cuda_backend.initialize()
            latency_metric = TimingMetricFactory.create_latency_metric(cuda_device)
            throughput_metric = TimingMetricFactory.create_throughput_metric(cuda_device, batch_size=16)
            
            # Validate device types
            self.assertEqual(latency_metric.device_type, "cuda")
            self.assertEqual(throughput_metric.device_type, "cuda")

if __name__ == "__main__":
    unittest.main()