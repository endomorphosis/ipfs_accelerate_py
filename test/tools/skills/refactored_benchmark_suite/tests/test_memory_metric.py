#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for memory metric.

This script tests the memory metric and factory with different device types.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from metrics.memory import MemoryMetric, MemoryMetricFactory
import hardware

class TestMemoryMetric(unittest.TestCase):
    """Test cases for memory metric."""
    
    def test_cpu_memory_metric(self):
        """Test CPU memory metric."""
        # Create CPU memory metric
        metric = MemoryMetric(device_type="cpu")
        
        # Test metric
        metric.start()
        
        # Allocate some memory
        large_tensor = torch.ones(1000, 1000)
        
        # Record memory
        metric.record_memory()
        
        # Allocate more memory
        another_tensor = torch.ones(2000, 2000)
        
        # Stop metric
        metric.stop()
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertIn("memory_usage_mb", metrics)
        
        # Check for CPU memory tracking
        if metric.has_psutil:
            self.assertIn("cpu_memory_start_mb", metrics)
            self.assertIn("cpu_memory_end_mb", metrics)
            self.assertIn("cpu_memory_growth_mb", metrics)
        
        # Get timeline
        timeline = metric.get_memory_timeline()
        
        # Validate timeline
        self.assertEqual(len(timeline), 3)  # start, record, stop
        for entry in timeline:
            self.assertIn("timestamp", entry)
    
    def test_cuda_memory_metric_if_available(self):
        """Test CUDA memory metric if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create CUDA memory metric
        metric = MemoryMetric(device_type="cuda")
        
        # Validate tracking capabilities
        self.assertTrue(metric.can_track_device_memory)
        self.assertTrue(metric.can_track_peak_memory)
        
        # Test metric
        metric.start()
        
        # Allocate some GPU memory
        large_tensor = torch.ones(1000, 1000, device="cuda")
        
        # Record memory
        metric.record_memory()
        
        # Allocate more GPU memory
        another_tensor = torch.ones(2000, 2000, device="cuda")
        
        # Stop metric
        metric.stop()
        
        # Get metrics
        metrics = metric.get_metrics()
        
        # Validate metrics
        self.assertIn("memory_peak_mb", metrics)
        self.assertIn("memory_allocated_start_mb", metrics)
        self.assertIn("memory_allocated_end_mb", metrics)
        self.assertIn("memory_reserved_start_mb", metrics)
        self.assertIn("memory_reserved_end_mb", metrics)
        self.assertIn("memory_growth_mb", metrics)
        
        # Get timeline
        timeline = metric.get_memory_timeline()
        
        # Validate timeline
        self.assertEqual(len(timeline), 3)  # start, record, stop
        for entry in timeline:
            self.assertIn("timestamp", entry)
            self.assertIn("allocated_mb", entry)
            self.assertIn("reserved_mb", entry)
    
    def test_factory_with_torch_device(self):
        """Test memory metric factory with torch.device."""
        # Create metric with torch.device
        cpu_device = torch.device("cpu")
        metric = MemoryMetricFactory.create(cpu_device)
        
        # Validate device type
        self.assertEqual(metric.device_type, "cpu")
        
        # Create metric with CUDA device if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            metric = MemoryMetricFactory.create(cuda_device)
            
            # Validate device type
            self.assertEqual(metric.device_type, "cuda")
    
    def test_factory_with_hardware_backend(self):
        """Test memory metric factory with hardware backend device."""
        # Create metric with CPU hardware backend
        cpu_backend = hardware.get_hardware_backend("cpu")
        cpu_device = cpu_backend.initialize()
        metric = MemoryMetricFactory.create(cpu_device)
        
        # Validate device type
        self.assertEqual(metric.device_type, "cpu")
        
        # Create metric with CUDA hardware backend if available
        if "cuda" in hardware.get_available_hardware():
            cuda_backend = hardware.get_hardware_backend("cuda")
            cuda_device = cuda_backend.initialize()
            metric = MemoryMetricFactory.create(cuda_device)
            
            # Validate device type
            self.assertEqual(metric.device_type, "cuda")

if __name__ == "__main__":
    unittest.main()