#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision model hardware metrics integration tests.

This module tests the integration of hardware-aware metrics with the enhanced
vision model adapter, focusing on power efficiency and memory bandwidth metrics.
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
from models.vision_models import VisionModelAdapter, apply_hardware_optimizations
from metrics.power import PowerMetricFactory
from metrics.bandwidth import BandwidthMetricFactory

# Skip tests if torch is not available
if not torch.cuda.is_available():
    HAS_CUDA = False
else:
    HAS_CUDA = True

# Skip tests if MPS is not available
HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

class TestVisionHardwareMetrics(unittest.TestCase):
    """Tests for hardware-aware metrics with vision models."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a temp directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_hardware_optimization_function(self):
        """Test the hardware optimization function with a simple vision model."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10)
        )
        
        # Test with CPU optimizations
        optimized_model = apply_hardware_optimizations(model, "cpu")
        self.assertIsNotNone(optimized_model)
        
        # Test with CUDA optimizations if available
        if HAS_CUDA:
            optimized_model = apply_hardware_optimizations(model, "cuda")
            self.assertIsNotNone(optimized_model)
        
        # Test with Flash Attention flag
        optimized_model = apply_hardware_optimizations(model, "cpu", use_flash_attention=True)
        self.assertIsNotNone(optimized_model)
        
        # Test with torch.compile flag (this may fail if torch < 2.0)
        try:
            optimized_model = apply_hardware_optimizations(model, "cpu", use_torch_compile=True)
            self.assertIsNotNone(optimized_model)
        except Exception:
            # Skip test if torch.compile not available
            pass
    
    def test_vision_adapter_model_type_detection(self):
        """Test model type detection in vision adapter."""
        # Test model type detection for different vision models
        sam_adapter = VisionModelAdapter("facebook/sam-vit-base", task="image-segmentation")
        self.assertTrue(sam_adapter.is_sam)
        self.assertEqual(sam_adapter.task, "image-segmentation")
        
        detr_adapter = VisionModelAdapter("facebook/detr-resnet-50", task="object-detection")
        self.assertTrue(detr_adapter.is_detr)
        self.assertEqual(detr_adapter.task, "object-detection")
        
        dino_adapter = VisionModelAdapter("facebook/dino-vitb16", task="image-classification")
        self.assertTrue(dino_adapter.is_dino)
        self.assertEqual(dino_adapter.task, "image-classification")
        
        # Test auto task detection
        swin_adapter = VisionModelAdapter("microsoft/swin-base-patch4-window7-224")
        self.assertTrue(swin_adapter.is_swin)
        self.assertEqual(swin_adapter.task, "image-classification")
        
        # Test auto task detection for object detection
        det_adapter = VisionModelAdapter("facebook/detr-resnet-50")
        self.assertEqual(det_adapter.task, "object-detection")
    
    def test_vision_adapter_input_preparation(self):
        """Test input preparation for different vision models."""
        # Test using mock processor
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            # Mock the processor
            mock_proc = MagicMock()
            mock_proc.size = 224
            mock_processor.return_value = mock_proc
            
            # Test standard vision model
            adapter = VisionModelAdapter("google/vit-base-patch16-224")
            
            # Mock the processor call
            mock_proc.return_value = {"pixel_values": torch.rand(2, 3, 224, 224)}
            
            # Try to prepare inputs
            inputs = adapter.prepare_inputs(batch_size=2)
            self.assertIn("pixel_values", inputs)
            self.assertEqual(inputs["pixel_values"].shape[0], 2)  # Batch size
            self.assertEqual(inputs["pixel_values"].shape[1], 3)  # Channels
    
    def test_power_metric_integration(self):
        """Test integration of power metrics with vision models."""
        # Create power metric
        device = torch.device("cpu")
        power_metric = PowerMetricFactory.create(device)
        
        # Test power metric lifecycle
        power_metric.start()
        
        # Create a simple model and run inference
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 112 * 112, 10)
        ).to(device)
        
        # Run some inference
        inputs = torch.rand(1, 3, 224, 224).to(device)
        for _ in range(5):
            model(inputs)
        
        # Set some operations count
        operations_count = 1e9  # 1 GFLOP
        power_metric.set_operations_count(operations_count)
        
        # Set throughput
        power_metric.set_throughput(10.0)  # 10 items/sec
        
        # Stop the metric
        power_metric.stop()
        
        # Get metrics
        metrics = power_metric.get_metrics()
        
        # Validate metrics
        self.assertIn("power_supported", metrics)
        if metrics["power_supported"]:
            self.assertIn("power_avg_watts", metrics)
            self.assertIn("energy_joules", metrics)
            self.assertIn("ops_per_watt", metrics)
            self.assertIn("gflops_per_watt", metrics)
            self.assertIn("throughput_per_watt", metrics)
    
    def test_bandwidth_metric_integration(self):
        """Test integration of bandwidth metrics with vision models."""
        # Create bandwidth metric
        device = torch.device("cpu")
        bandwidth_metric = BandwidthMetricFactory.create(device)
        
        # Test bandwidth metric lifecycle
        bandwidth_metric.start()
        
        # Create a simple model and run inference
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 112 * 112, 10)
        ).to(device)
        
        # Run some inference
        inputs = torch.rand(1, 3, 224, 224).to(device)
        for _ in range(5):
            model(inputs)
        
        # Set compute operations
        operations_count = 1e9  # 1 GFLOP
        bandwidth_metric.set_compute_operations(operations_count)
        
        # Set memory transfers
        memory_transfers = 1024 * 1024 * 1024  # 1 GB
        bandwidth_metric.set_memory_transfers(memory_transfers)
        
        # Stop the metric
        bandwidth_metric.stop()
        
        # Get metrics
        metrics = bandwidth_metric.get_metrics()
        
        # Validate metrics
        self.assertIn("bandwidth_supported", metrics)
        if metrics["bandwidth_supported"]:
            self.assertIn("avg_bandwidth_gbps", metrics)
            self.assertIn("peak_theoretical_bandwidth_gbps", metrics)
            self.assertIn("arithmetic_intensity_flops_per_byte", metrics)
            self.assertIn("compute_bound", metrics)
        
        # Test roofline data
        roofline_data = bandwidth_metric.get_roofline_data()
        self.assertIn("arithmetic_intensity_flops_per_byte", roofline_data)
        self.assertIn("peak_compute_flops", roofline_data)
        self.assertIn("peak_memory_bandwidth_bytes_per_sec", roofline_data)
        self.assertIn("ridge_point_flops_per_byte", roofline_data)
    
    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    def test_gpu_hardware_metrics(self):
        """Test hardware metrics on GPU if available."""
        # Create metrics for CUDA
        device = torch.device("cuda")
        power_metric = PowerMetricFactory.create(device)
        bandwidth_metric = BandwidthMetricFactory.create(device)
        
        # Start metrics
        power_metric.start()
        bandwidth_metric.start()
        
        # Create a simple model and run inference
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 112 * 112, 10)
        ).to(device)
        
        # Run some inference
        inputs = torch.rand(1, 3, 224, 224).to(device)
        for _ in range(10):
            model(inputs)
            torch.cuda.synchronize()  # Ensure CUDA operations are complete
        
        # Set operations count
        operations_count = 1e9  # 1 GFLOP
        power_metric.set_operations_count(operations_count)
        bandwidth_metric.set_compute_operations(operations_count)
        
        # Set memory transfers
        memory_transfers = 1024 * 1024 * 1024  # 1 GB
        bandwidth_metric.set_memory_transfers(memory_transfers)
        
        # Stop metrics
        power_metric.stop()
        bandwidth_metric.stop()
        
        # Get metrics
        power_metrics = power_metric.get_metrics()
        bandwidth_metrics = bandwidth_metric.get_metrics()
        
        # Validate power metrics
        self.assertIn("power_supported", power_metrics)
        
        # Validate bandwidth metrics
        self.assertIn("bandwidth_supported", bandwidth_metrics)
        
        # If NVIDIA SMI is available, we should have power readings
        if power_metric.has_nvidia_smi:
            self.assertTrue(power_metrics["power_supported"])
            self.assertIn("power_avg_watts", power_metrics)
            self.assertGreater(power_metrics["power_avg_watts"], 0)

if __name__ == '__main__':
    unittest.main()