#!/usr/bin/env python3
"""
Test utilities for HuggingFace model testing.

This module provides common utilities, assertions, and helpers
for testing HuggingFace models across different hardware platforms.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from unittest.mock import MagicMock


class ModelTestUtils:
    """Utilities for testing HuggingFace models."""
    
    @staticmethod
    def assert_model_loaded(model: Any, model_name: str) -> None:
        """Assert that a model is properly loaded.
        
        Args:
            model: The loaded model
            model_name: Name of the model for error messages
        """
        assert model is not None, f"{model_name} model is None"
        assert not isinstance(model, MagicMock), f"{model_name} is a mock, not real model"
        assert hasattr(model, 'forward') or hasattr(model, '__call__'), \
            f"{model_name} doesn't have forward() or __call__()"
    
    @staticmethod
    def assert_tokenizer_loaded(tokenizer: Any, model_name: str) -> None:
        """Assert that a tokenizer is properly loaded.
        
        Args:
            tokenizer: The loaded tokenizer
            model_name: Name of the model for error messages
        """
        assert tokenizer is not None, f"{model_name} tokenizer is None"
        assert not isinstance(tokenizer, MagicMock), f"{model_name} tokenizer is a mock"
        assert hasattr(tokenizer, '__call__'), f"{model_name} tokenizer not callable"
    
    @staticmethod
    def assert_tensor_valid(tensor: torch.Tensor, name: str = "tensor") -> None:
        """Assert that a tensor is valid (no NaN, Inf, reasonable values).
        
        Args:
            tensor: The tensor to validate
            name: Name for error messages
        """
        assert tensor is not None, f"{name} is None"
        assert isinstance(tensor, torch.Tensor), f"{name} is not a tensor"
        assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contains Inf values"
    
    @staticmethod
    def assert_output_shape(output: torch.Tensor, expected_shape: Tuple, 
                           allow_batch_dim: bool = True) -> None:
        """Assert that output has expected shape.
        
        Args:
            output: The output tensor
            expected_shape: Expected shape (can include -1 for any dimension)
            allow_batch_dim: If True, allow first dimension to vary
        """
        assert output is not None, "Output is None"
        
        if allow_batch_dim:
            # Check all dimensions except batch
            assert len(output.shape) == len(expected_shape), \
                f"Shape mismatch: got {output.shape}, expected {expected_shape}"
            for i, (got, expected) in enumerate(zip(output.shape[1:], expected_shape[1:])):
                if expected != -1:  # -1 means any size
                    assert got == expected, \
                        f"Dimension {i+1} mismatch: got {got}, expected {expected}"
        else:
            # Check exact shape
            for i, (got, expected) in enumerate(zip(output.shape, expected_shape)):
                if expected != -1:
                    assert got == expected, \
                        f"Dimension {i} mismatch: got {got}, expected {expected}"
    
    @staticmethod
    def assert_device_correct(tensor: torch.Tensor, expected_device: str) -> None:
        """Assert that tensor is on expected device.
        
        Args:
            tensor: The tensor to check
            expected_device: Expected device string (e.g., "cuda", "cpu", "mps")
        """
        actual_device = tensor.device.type
        assert actual_device == expected_device, \
            f"Tensor on wrong device: got {actual_device}, expected {expected_device}"
    
    @staticmethod
    def measure_inference_time(model: Any, inputs: Dict, 
                               warmup_runs: int = 3,
                               test_runs: int = 10) -> Dict[str, float]:
        """Measure model inference time.
        
        Args:
            model: The model to test
            inputs: Model inputs
            warmup_runs: Number of warmup iterations
            test_runs: Number of test iterations
            
        Returns:
            Dict with timing statistics
        """
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Synchronize if using CUDA
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(test_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
        }
    
    @staticmethod
    def measure_memory_usage(model: Any, inputs: Dict, 
                            device: str = "cuda") -> Dict[str, float]:
        """Measure model memory usage.
        
        Args:
            model: The model to test
            inputs: Model inputs
            device: Device to measure ("cuda" or "cpu")
            
        Returns:
            Dict with memory statistics (in MB)
        """
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Run inference
            with torch.no_grad():
                _ = model(**inputs)
            
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'peak_mb': max_allocated,
            }
        else:
            # CPU memory measurement would require tracemalloc
            import tracemalloc
            tracemalloc.start()
            
            with torch.no_grad():
                _ = model(**inputs)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return {
                'current_mb': current / (1024 ** 2),
                'peak_mb': peak / (1024 ** 2),
            }
    
    @staticmethod
    def create_sample_text_inputs(tokenizer: Any, 
                                  texts: Optional[List[str]] = None,
                                  max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Create sample text inputs for testing.
        
        Args:
            tokenizer: The tokenizer to use
            texts: List of texts (or None for defaults)
            max_length: Maximum sequence length
            
        Returns:
            Dict of tokenized inputs
        """
        if texts is None:
            texts = [
                "This is a test sentence.",
                "The quick brown fox jumps over the lazy dog.",
            ]
        
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
    
    @staticmethod
    def create_sample_image_inputs(processor: Any, 
                                   size: Tuple[int, int] = (224, 224),
                                   batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Create sample image inputs for testing.
        
        Args:
            processor: The image processor to use
            size: Image size (height, width)
            batch_size: Number of images in batch
            
        Returns:
            Dict of processed inputs
        """
        from PIL import Image
        
        # Create random images
        images = [
            Image.new('RGB', size, color=(
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ))
            for _ in range(batch_size)
        ]
        
        return processor(images=images, return_tensors="pt")
    
    @staticmethod
    def compare_outputs(output1: torch.Tensor, output2: torch.Tensor,
                       rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Compare two model outputs for similarity.
        
        Args:
            output1: First output tensor
            output2: Second output tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if outputs are similar
        """
        return torch.allclose(output1, output2, rtol=rtol, atol=atol)


class HardwareTestUtils:
    """Utilities for hardware-specific testing."""
    
    @staticmethod
    def get_available_devices() -> List[str]:
        """Get list of available devices for testing.
        
        Returns:
            List of device strings
        """
        devices = ["cpu"]
        
        if torch.cuda.is_available():
            devices.append("cuda")
            # Add specific CUDA devices
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
            if torch.mps.is_available():
                devices.append("mps")
        
        return devices
    
    @staticmethod
    def assert_model_works_on_device(model: Any, inputs: Dict, 
                                    device: str) -> None:
        """Assert that model works on specified device.
        
        Args:
            model: The model to test
            inputs: Model inputs
            device: Device string
        """
        # Move model to device
        model = model.to(device)
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Validate outputs are on correct device
        if hasattr(outputs, 'last_hidden_state'):
            ModelTestUtils.assert_device_correct(outputs.last_hidden_state, device)
        elif isinstance(outputs, torch.Tensor):
            ModelTestUtils.assert_device_correct(outputs, device)


class PerformanceTestUtils:
    """Utilities for performance testing."""
    
    @staticmethod
    def assert_inference_time_within_threshold(actual_time: float,
                                              baseline_time: float,
                                              threshold_factor: float = 1.2) -> None:
        """Assert that inference time is within acceptable threshold.
        
        Args:
            actual_time: Actual inference time
            baseline_time: Baseline inference time
            threshold_factor: Maximum allowed factor (e.g., 1.2 = 20% slower)
        """
        max_allowed = baseline_time * threshold_factor
        assert actual_time <= max_allowed, \
            f"Inference too slow: {actual_time:.4f}s > {max_allowed:.4f}s " \
            f"(baseline: {baseline_time:.4f}s, threshold: {threshold_factor}x)"
    
    @staticmethod
    def assert_memory_within_threshold(actual_memory: float,
                                      baseline_memory: float,
                                      threshold_factor: float = 1.2) -> None:
        """Assert that memory usage is within acceptable threshold.
        
        Args:
            actual_memory: Actual memory usage (MB)
            baseline_memory: Baseline memory usage (MB)
            threshold_factor: Maximum allowed factor
        """
        max_allowed = baseline_memory * threshold_factor
        assert actual_memory <= max_allowed, \
            f"Memory usage too high: {actual_memory:.2f}MB > {max_allowed:.2f}MB " \
            f"(baseline: {baseline_memory:.2f}MB, threshold: {threshold_factor}x)"
    
    @staticmethod
    def create_performance_report(model_name: str, 
                                 timing_stats: Dict,
                                 memory_stats: Optional[Dict] = None) -> str:
        """Create a formatted performance report.
        
        Args:
            model_name: Name of the model
            timing_stats: Timing statistics from measure_inference_time
            memory_stats: Optional memory statistics
            
        Returns:
            Formatted report string
        """
        report = [
            f"\n{'='*60}",
            f"Performance Report: {model_name}",
            f"{'='*60}",
            "\nTiming Statistics:",
            f"  Mean:   {timing_stats['mean']*1000:.2f}ms",
            f"  Median: {timing_stats['median']*1000:.2f}ms",
            f"  Min:    {timing_stats['min']*1000:.2f}ms",
            f"  Max:    {timing_stats['max']*1000:.2f}ms",
            f"  Std:    {timing_stats['std']*1000:.2f}ms",
        ]
        
        if memory_stats:
            report.extend([
                "\nMemory Statistics:",
                f"  Allocated: {memory_stats.get('allocated_mb', 0):.2f}MB",
                f"  Peak:      {memory_stats.get('peak_mb', 0):.2f}MB",
            ])
        
        report.append(f"{'='*60}\n")
        return '\n'.join(report)
