#!/usr/bin/env python3
"""
WebGPU Quantization Implementation for LLMs

This module implements 4-bit quantization techniques for WebGPU-based language models,
enabling efficient inference in memory-constrained browser environments.

Key features:
- 4-bit weight quantization for significant memory reduction
- Mixed precision inference with 4-bit weights and 16-bit activations
- Layer-specific quantization (attention layers at higher precision)
- Adaptive precision adjustment based on runtime memory monitoring
- Dynamic loading of quantized weights for large models
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("webgpu_quantization")

class WebGPUQuantizer:
    """
    Implements quantization techniques for WebGPU acceleration,
    focusing on 4-bit inference for large language models.
    """
    
    def __init__(self, 
                 default_bits: int = 4,
                 mixed_precision: bool = True,
                 adaptive_precision: bool = True,
                 memory_threshold_mb: int = 2000):
        """
        Initialize the WebGPU quantizer.
        
        Args:
            default_bits: Default quantization bits (4, 8, or 16)
            mixed_precision: Whether to use mixed precision (different bits for different layers)
            adaptive_precision: Whether to adjust precision based on memory availability
            memory_threshold_mb: Memory threshold for adaptive precision in MB
        """
        self.default_bits = default_bits
        self.mixed_precision = mixed_precision
        self.adaptive_precision = adaptive_precision
        self.memory_threshold_mb = memory_threshold_mb
        
        # Calibration statistics for quantization
        self.calibration_stats = {}
        
        # Memory usage tracking
        self.memory_usage = {
            "peak_mb": 0,
            "current_mb": 0,
            "saved_mb": 0,
            "model_size_mb": 0
        }
        
        # Performance tracking
        self.performance_stats = {
            "quantization_time_ms": 0,
            "dequantization_time_ms": 0,
            "inference_time_ms": 0,
            "memory_savings_percent": 0
        }
        
        # Layer-specific quantization bits
        self.layer_bits = {
            "embedding": 8,      # Keep embeddings at higher precision
            "attention": 8,      # Higher precision for attention layers
            "feedforward": 4,    # Low precision for FF networks
            "norm": 16,          # Full precision for normalization layers
            "head": 8            # Higher precision for output layer
        }
        
        logger.info(f"Initialized WebGPU quantizer with {default_bits}-bit default precision")
        if mixed_precision:
            logger.info(f"Mixed precision enabled: {self.layer_bits}")
        if adaptive_precision:
            logger.info(f"Adaptive precision enabled with {memory_threshold_mb}MB threshold")
    
    def quantize_model(self, model_weights: Dict[str, np.ndarray], bits: Optional[int] = None) -> Dict[str, Any]:
        """
        Quantize model weights for WebGPU deployment.
        
        Args:
            model_weights: Dictionary of model weight tensors
            bits: Quantization bits (4, 8, or 16). If None, uses default_bits.
            
        Returns:
            Dictionary with quantized weights and quantization parameters
        """
        if bits is None:
            bits = self.default_bits
            
        start_time = time.time()
        quantized_model = {}
        
        # Calculate original model size
        original_size_bytes = 0
        for name, tensor in model_weights.items():
            original_size_bytes += tensor.size * tensor.itemsize
        
        self.memory_usage["model_size_mb"] = original_size_bytes / (1024 * 1024)
        logger.info(f"Original model size: {self.memory_usage['model_size_mb']:.2f}MB")
        
        # Quantize each layer based on layer type
        quantized_size_bytes = 0
        for name, tensor in model_weights.items():
            # Determine layer type from name
            layer_type = self._determine_layer_type(name)
            
            # Get bits for this layer (mixed precision)
            layer_bits = bits
            if self.mixed_precision:
                layer_bits = self.layer_bits.get(layer_type, bits)
            
            # Quantize tensor
            quantized_tensor, quant_params = self._quantize_tensor(tensor, layer_bits)
            
            # Store quantized tensor and parameters
            quantized_model[name] = {
                "data": quantized_tensor,
                "bits": layer_bits,
                "params": quant_params,
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "layer_type": layer_type
            }
            
            # Update size tracking
            quantized_size_bytes += quantized_tensor.size * quantized_tensor.itemsize
            if "scale" in quant_params:
                # Add size of quantization parameters
                quantized_size_bytes += quant_params["scale"].size * quant_params["scale"].itemsize
                if "zero_point" in quant_params:
                    quantized_size_bytes += quant_params["zero_point"].size * quant_params["zero_point"].itemsize
        
        # Calculate memory savings
        quantized_size_mb = quantized_size_bytes / (1024 * 1024)
        memory_saved_mb = self.memory_usage["model_size_mb"] - quantized_size_mb
        memory_savings_percent = (memory_saved_mb / self.memory_usage["model_size_mb"]) * 100
        
        self.memory_usage["current_mb"] = quantized_size_mb
        self.memory_usage["saved_mb"] = memory_saved_mb
        self.performance_stats["memory_savings_percent"] = memory_savings_percent
        
        # Record quantization time
        quantization_time = (time.time() - start_time) * 1000
        self.performance_stats["quantization_time_ms"] = quantization_time
        
        logger.info(f"Quantized model size: {quantized_size_mb:.2f}MB")
        logger.info(f"Memory reduction: {memory_savings_percent:.2f}%")
        logger.info(f"Quantization time: {quantization_time:.2f}ms")
        
        # Add metadata to the quantized model
        quantized_model["__metadata__"] = {
            "original_size_mb": self.memory_usage["model_size_mb"],
            "quantized_size_mb": quantized_size_mb,
            "memory_savings_percent": memory_savings_percent,
            "default_bits": self.default_bits,
            "mixed_precision": self.mixed_precision,
            "adaptive_precision": self.adaptive_precision,
            "layer_bits": self.layer_bits.copy(),
            "quantization_time_ms": quantization_time
        }
        
        return quantized_model
    
    def dequantize_for_inference(self, quantized_model: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Dequantize model for inference.
        
        In a real WebGPU implementation, this would happen on the GPU directly.
        This simulation demonstrates how dequantization would work.
        
        Args:
            quantized_model: Quantized model from quantize_model
            
        Returns:
            Dictionary with dequantized weights
        """
        start_time = time.time()
        dequantized_weights = {}
        
        for name, layer_data in quantized_model.items():
            if name == "__metadata__":
                continue
                
            # Get tensor and params
            quantized_tensor = layer_data["data"]
            quant_params = layer_data["params"]
            bits = layer_data["bits"]
            
            # Dequantize tensor
            dequantized_tensor = self._dequantize_tensor(quantized_tensor, quant_params, bits)
            dequantized_weights[name] = dequantized_tensor
        
        # Record dequantization time
        dequantization_time = (time.time() - start_time) * 1000
        self.performance_stats["dequantization_time_ms"] = dequantization_time
        
        logger.info(f"Dequantization time: {dequantization_time:.2f}ms")
        return dequantized_weights
    
    def _quantize_tensor(self, tensor: np.ndarray, bits: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Quantize a single tensor.
        
        Args:
            tensor: Input tensor to quantize
            bits: Number of bits for quantization
            
        Returns:
            Tuple of (quantized_tensor, quantization_parameters)
        """
        if bits == 16:
            # Use FP16 (no quantization)
            return tensor.astype(np.float16), {"original_dtype": str(tensor.dtype)}
            
        # Calculate scale and zero point for the tensor
        if bits == 8:
            # 8-bit quantization (int8)
            qmin, qmax = -128, 127
            dtype = np.int8
        elif bits == 4:
            # 4-bit quantization - simulate using int8 and masks
            qmin, qmax = -8, 7
            dtype = np.int8  # We'll pack 4-bit values later
        elif bits == 2:
            # 2-bit quantization - simulate using int8 and masks
            qmin, qmax = -2, 1
            dtype = np.int8  # We'll pack 2-bit values later
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
        
        # Calculate min and max values
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        # Calculate scale and zero_point
        scale = (tensor_max - tensor_min) / (qmax - qmin)
        zero_point = qmin - np.round(tensor_min / scale) if scale != 0 else 0
        
        # Quantize tensor
        quantized = np.clip(np.round(tensor / scale) + zero_point, qmin, qmax)
        
        # For 4-bit, pack values
        if bits == 4:
            # Pack two 4-bit values per int8
            quantized = self._pack_4bit_values(quantized.astype(dtype))
        elif bits == 2:
            # Pack four 2-bit values per int8
            quantized = self._pack_2bit_values(quantized.astype(dtype))
        else:
            quantized = quantized.astype(dtype)
        
        # Return quantized tensor and parameters
        return quantized, {
            "scale": scale.astype(np.float32),
            "zero_point": np.array(zero_point, dtype=np.int32),
            "bits": bits,
            "original_dtype": str(tensor.dtype)
        }
    
    def _dequantize_tensor(self, quantized_tensor: np.ndarray, quant_params: Dict[str, np.ndarray], bits: int) -> np.ndarray:
        """
        Dequantize a single tensor.
        
        Args:
            quantized_tensor: Quantized tensor
            quant_params: Quantization parameters
            bits: Number of bits for dequantization
            
        Returns:
            Dequantized tensor
        """
        if bits == 16:
            # Already in FP16
            return quantized_tensor
        
        # Get scale and zero point
        scale = quant_params["scale"]
        zero_point = quant_params["zero_point"]
        
        # For 4-bit, unpack values
        if bits == 4:
            # Unpack two 4-bit values per int8
            unpacked = self._unpack_4bit_values(quantized_tensor)
        elif bits == 2:
            # Unpack four 2-bit values per int8
            unpacked = self._unpack_2bit_values(quantized_tensor)
        else:
            unpacked = quantized_tensor
        
        # Dequantize
        dequantized = (unpacked.astype(np.float32) - zero_point) * scale
        
        return dequantized
    
    def _pack_4bit_values(self, tensor: np.ndarray) -> np.ndarray:
        """
        Pack 4-bit values into 8-bit storage.
        
        Args:
            tensor: Int8 tensor with values between -8 and 7
            
        Returns:
            Packed tensor with two 4-bit values per byte
        """
        # Ensure values are in valid range
        tensor = np.clip(tensor, -8, 7)
        
        # Reshape for packing (ensure even length)
        original_shape = tensor.shape
        flat = tensor.flatten()
        
        # Pad to even length if needed
        if len(flat) % 2 != 0:
            flat = np.pad(flat, (0, 1), 'constant')
        
        # Reshape to pack 2 values per byte
        pairs = flat.reshape(-1, 2)
        
        # Convert to unsigned for bitwise operations
        pairs = pairs + 8  # Now 0-15
        
        # Pack pairs into bytes (first value in low bits, second in high bits)
        packed = pairs[:, 0] | (pairs[:, 1] << 4)
        
        # Return as int8
        return packed.astype(np.int8)
    
    def _unpack_4bit_values(self, packed_tensor: np.ndarray) -> np.ndarray:
        """
        Unpack 4-bit values from 8-bit storage.
        
        Args:
            packed_tensor: Tensor with two 4-bit values per byte
            
        Returns:
            Unpacked tensor with original values
        """
        # Convert to unsigned for bitwise operations
        packed_unsigned = packed_tensor.astype(np.uint8)
        
        # Extract low and high bits
        low_bits = packed_unsigned & 0x0F
        high_bits = (packed_unsigned >> 4) & 0x0F
        
        # Interleave low and high bits
        unpacked = np.empty(len(packed_unsigned) * 2, dtype=np.int8)
        unpacked[0::2] = low_bits
        unpacked[1::2] = high_bits
        
        # Convert back to signed (-8 to 7 range)
        return unpacked.astype(np.int8) - 8
    
    def _pack_2bit_values(self, tensor: np.ndarray) -> np.ndarray:
        """
        Pack 2-bit values into 8-bit storage.
        
        Args:
            tensor: Int8 tensor with values between -2 and 1
            
        Returns:
            Packed tensor with four 2-bit values per byte
        """
        # Ensure values are in valid range
        tensor = np.clip(tensor, -2, 1)
        
        # Reshape for packing (ensure multiple of 4 length)
        original_shape = tensor.shape
        flat = tensor.flatten()
        
        # Pad to multiple of 4 length if needed
        if len(flat) % 4 != 0:
            flat = np.pad(flat, (0, 4 - (len(flat) % 4)), 'constant')
        
        # Reshape to pack 4 values per byte
        quads = flat.reshape(-1, 4)
        
        # Convert to unsigned for bitwise operations
        quads = quads + 2  # Now 0-3
        
        # Pack quads into bytes
        packed = (quads[:, 0] | 
                 (quads[:, 1] << 2) | 
                 (quads[:, 2] << 4) | 
                 (quads[:, 3] << 6))
        
        # Return as int8
        return packed.astype(np.int8)
    
    def _unpack_2bit_values(self, packed_tensor: np.ndarray) -> np.ndarray:
        """
        Unpack 2-bit values from 8-bit storage.
        
        Args:
            packed_tensor: Tensor with four 2-bit values per byte
            
        Returns:
            Unpacked tensor with original values
        """
        # Convert to unsigned for bitwise operations
        packed_unsigned = packed_tensor.astype(np.uint8)
        
        # Extract 2-bit values
        bits_0 = packed_unsigned & 0x03
        bits_1 = (packed_unsigned >> 2) & 0x03
        bits_2 = (packed_unsigned >> 4) & 0x03
        bits_3 = (packed_unsigned >> 6) & 0x03
        
        # Interleave values
        unpacked = np.empty(len(packed_unsigned) * 4, dtype=np.int8)
        unpacked[0::4] = bits_0
        unpacked[1::4] = bits_1
        unpacked[2::4] = bits_2
        unpacked[3::4] = bits_3
        
        # Convert back to signed (-2 to 1 range)
        return unpacked.astype(np.int8) - 2
    
    def _determine_layer_type(self, layer_name: str) -> str:
        """
        Determine layer type from layer name.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Layer type as string
        """
        layer_name = layer_name.lower()
        
        if "embed" in layer_name:
            return "embedding"
        elif "attention" in layer_name or "attn" in layer_name:
            return "attention"
        elif "mlp" in layer_name or "ffn" in layer_name or "feed_forward" in layer_name:
            return "feedforward"
        elif "norm" in layer_name or "ln" in layer_name or "layer_norm" in layer_name:
            return "norm"
        elif "head" in layer_name or "output" in layer_name or "classifier" in layer_name:
            return "head"
        else:
            return "other"
    
    def adjust_precision_for_memory(self, current_memory_mb: float) -> Dict[str, int]:
        """
        Dynamically adjust quantization precision based on available memory.
        
        Args:
            current_memory_mb: Current memory usage in MB
            
        Returns:
            Updated layer_bits dictionary
        """
        if not self.adaptive_precision:
            return self.layer_bits
        
        # Check if we're below the memory threshold
        if current_memory_mb < self.memory_threshold_mb:
            # We have enough memory, use higher precision
            updated_bits = {
                "embedding": 8,
                "attention": 8,
                "feedforward": 4,
                "norm": 16,
                "head": 8
            }
        else:
            # Memory constraint, use lower precision
            memory_pressure = current_memory_mb / self.memory_threshold_mb
            
            if memory_pressure > 2.0:
                # Severe memory pressure, use minimum precision
                updated_bits = {
                    "embedding": 4,
                    "attention": 4,
                    "feedforward": 2,
                    "norm": 8,
                    "head": 4
                }
            elif memory_pressure > 1.5:
                # High memory pressure
                updated_bits = {
                    "embedding": 4,
                    "attention": 4,
                    "feedforward": 4,
                    "norm": 8,
                    "head": 4
                }
            else:
                # Moderate memory pressure
                updated_bits = {
                    "embedding": 8,
                    "attention": 4,
                    "feedforward": 4,
                    "norm": 8,
                    "head": 8
                }
        
        # Update our layer bits
        self.layer_bits = updated_bits
        
        logger.info(f"Adjusted precision for memory usage {current_memory_mb:.2f}MB: {updated_bits}")
        return updated_bits
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return self.memory_usage
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return self.performance_stats


# Example usage function
def example_4bit_quantization():
    """Demonstrate 4-bit quantization for a simple model."""
    # Create a simple model with random weights
    model_weights = {
        "embedding.weight": np.random.randn(10000, 768).astype(np.float32),
        "encoder.layer.0.attention.self.query.weight": np.random.randn(768, 768).astype(np.float32),
        "encoder.layer.0.attention.self.key.weight": np.random.randn(768, 768).astype(np.float32),
        "encoder.layer.0.attention.self.value.weight": np.random.randn(768, 768).astype(np.float32),
        "encoder.layer.0.attention.output.dense.weight": np.random.randn(768, 768).astype(np.float32),
        "encoder.layer.0.intermediate.dense.weight": np.random.randn(768, 3072).astype(np.float32),
        "encoder.layer.0.output.dense.weight": np.random.randn(3072, 768).astype(np.float32),
        "encoder.layer.0.attention.output.LayerNorm.weight": np.random.randn(768).astype(np.float32),
        "encoder.layer.0.attention.output.LayerNorm.bias": np.random.randn(768).astype(np.float32),
        "encoder.layer.0.output.LayerNorm.weight": np.random.randn(768).astype(np.float32),
        "encoder.layer.0.output.LayerNorm.bias": np.random.randn(768).astype(np.float32),
        "pooler.dense.weight": np.random.randn(768, 768).astype(np.float32),
        "pooler.dense.bias": np.random.randn(768).astype(np.float32),
    }
    
    # Initialize quantizer
    quantizer = WebGPUQuantizer(default_bits=4, mixed_precision=True)
    
    # Quantize model
    quantized_model = quantizer.quantize_model(model_weights)
    
    # Print memory savings
    memory_usage = quantizer.get_memory_usage()
    performance_stats = quantizer.get_performance_stats()
    
    print(f"Original model size: {memory_usage['model_size_mb']:.2f} MB")
    print(f"Quantized model size: {memory_usage['current_mb']:.2f} MB")
    print(f"Memory savings: {memory_usage['saved_mb']:.2f} MB ({performance_stats['memory_savings_percent']:.2f}%)")
    
    # Test dequantization
    dequantized_weights = quantizer.dequantize_for_inference(quantized_model)
    
    # Check accuracy loss for a sample weight
    original = model_weights["encoder.layer.0.attention.self.query.weight"]
    dequantized = dequantized_weights["encoder.layer.0.attention.self.query.weight"]
    
    abs_error = np.abs(original - dequantized).mean()
    rel_error = abs_error / np.abs(original).mean() * 100
    
    print(f"Mean absolute error: {abs_error:.6f}")
    print(f"Mean relative error: {rel_error:.2f}%")


if __name__ == "__main__":
    example_4bit_quantization()