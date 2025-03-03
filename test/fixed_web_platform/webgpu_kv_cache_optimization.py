#!/usr/bin/env python3
"""
WebGPU KV-Cache Optimization for LLMs (April 2025)

This module implements memory-efficient Key-Value cache management for 
large language models in WebGPU environments. It reduces memory usage
during LLM inference by optimizing KV cache storage and retrieval.

Features:
- Sliding window KV cache for memory-constrained environments
- Memory-efficient attention for long contexts
- 4-bit quantized KV cache implementation
- Optimized block-wise cache management
- Dynamic cache pruning for long-running inference

Usage:
    from fixed_web_platform.webgpu_kv_cache_optimization import (
        WebGPUKVCacheManager,
        setup_kv_cache_for_llm,
        generate_kv_cache_shaders
    )
    
    # Create and use a KV cache manager
    kv_manager = WebGPUKVCacheManager(max_seq_length=2048, head_dim=128)
    cache_id = kv_manager.initialize_cache(batch_size=1, num_heads=32)
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webgpu_kv_cache")

try:
    # Try to import the quantization module if available
    from fixed_web_platform.webgpu_quantization import WebGPUQuantizer
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logger.warning("WebGPU quantization module not available, KV cache quantization will be disabled")

class WebGPUKVCacheManager:
    """Memory-efficient KV cache manager for LLMs in WebGPU."""
    
    def __init__(self, max_seq_length=2048, head_dim=64, 
                 max_memory_mb=1000, enable_quantization=True, 
                 sliding_window=True, window_size=None,
                 enable_pruning=True):
        """
        Initialize the KV cache manager.
        
        Args:
            max_seq_length: Maximum sequence length
            head_dim: Dimension of each attention head
            max_memory_mb: Maximum memory allowed for KV cache in MB
            enable_quantization: Whether to enable 4-bit quantization for KV cache
            sliding_window: Whether to use sliding window approach
            window_size: Size of the sliding window (default is 1/4 of max_seq_length)
            enable_pruning: Whether to enable dynamic pruning for long contexts
        """
        self.max_seq_length = max_seq_length
        self.head_dim = head_dim
        self.max_memory_mb = max_memory_mb
        self.enable_quantization = enable_quantization and QUANTIZATION_AVAILABLE
        self.sliding_window = sliding_window
        self.window_size = window_size or (max_seq_length // 4)
        self.enable_pruning = enable_pruning
        
        # Cache storage
        self.cache_instances = {}
        
        # Quantizer for 4-bit KV cache
        if self.enable_quantization:
            self.quantizer = WebGPUQuantizer(bits=4, group_size=32, scheme="symmetric")
        
        # Memory usage statistics
        self.memory_stats = {
            "current_memory_mb": 0,
            "peak_memory_mb": 0,
            "total_tokens_processed": 0,
            "pruned_tokens_count": 0,
            "cache_efficiency": 0.0,
            "cache_hit_ratio": 0.0
        }
        
        logger.info(f"Initialized WebGPU KV cache manager with max_seq_length={max_seq_length}, "
                   f"head_dim={head_dim}, max_memory_mb={max_memory_mb}, "
                   f"quantization={'enabled' if self.enable_quantization else 'disabled'}, "
                   f"sliding_window={'enabled' if self.sliding_window else 'disabled'}")
    
    def initialize_cache(self, batch_size=1, num_heads=16, model_name=None):
        """
        Initialize a new KV cache instance.
        
        Args:
            batch_size: Batch size for inference
            num_heads: Number of attention heads
            model_name: Optional name for the model
            
        Returns:
            Cache ID for the initialized cache
        """
        # Generate a unique ID for this cache instance
        cache_id = f"kv_cache_{model_name or 'model'}_{batch_size}_{num_heads}_{self.head_dim}_{int(time.time())}"
        
        # Calculate memory requirements
        keys_shape = (batch_size, num_heads, self.max_seq_length, self.head_dim)
        values_shape = (batch_size, num_heads, self.max_seq_length, self.head_dim)
        
        element_size = 4  # float32 = 4 bytes
        if self.enable_quantization:
            element_size = 1  # 4-bit = 1 byte (packed 2 values per byte)
        
        # Calculate memory usage
        keys_memory_mb = np.prod(keys_shape) * element_size / (1024 * 1024)
        values_memory_mb = np.prod(values_shape) * element_size / (1024 * 1024)
        total_memory_mb = keys_memory_mb + values_memory_mb
        
        # Check if memory exceeds limit
        if total_memory_mb > self.max_memory_mb:
            # Apply sliding window if enabled
            if self.sliding_window:
                window_keys_shape = (batch_size, num_heads, self.window_size, self.head_dim)
                window_values_shape = (batch_size, num_heads, self.window_size, self.head_dim)
                
                window_keys_memory_mb = np.prod(window_keys_shape) * element_size / (1024 * 1024)
                window_values_memory_mb = np.prod(window_values_shape) * element_size / (1024 * 1024)
                total_memory_mb = window_keys_memory_mb + window_values_memory_mb
                
                logger.info(f"Sliding window KV cache enabled: {self.window_size} tokens (reduced from {self.max_seq_length})")
                
                # Update stored shapes
                keys_shape = window_keys_shape
                values_shape = window_values_shape
            else:
                logger.warning(f"KV cache memory requirement ({total_memory_mb:.2f}MB) exceeds limit ({self.max_memory_mb}MB)")
        
        # Initialize cache instance
        cache_instance = {
            "config": {
                "batch_size": batch_size,
                "num_heads": num_heads,
                "max_seq_length": self.max_seq_length if not self.sliding_window else self.window_size,
                "head_dim": self.head_dim,
                "model_name": model_name,
                "quantized": self.enable_quantization,
                "sliding_window": self.sliding_window,
                "window_size": self.window_size if self.sliding_window else None,
                "pruning_enabled": self.enable_pruning
            },
            "memory_mb": total_memory_mb,
            "keys_shape": keys_shape,
            "values_shape": values_shape,
            "keys": None,  # Will be allocated on first use
            "values": None,  # Will be allocated on first use
            "current_length": 0,
            "position_map": {},  # Maps original positions to cache positions if using sliding window
            "pruning_scores": [],  # Used for token pruning
            "usage_counts": [],  # Tracks how frequently each token is accessed
            "last_access": []  # Tracks when each token was last accessed
        }
        
        self.cache_instances[cache_id] = cache_instance
        
        # Update memory statistics
        self.memory_stats["current_memory_mb"] += total_memory_mb
        self.memory_stats["peak_memory_mb"] = max(self.memory_stats["peak_memory_mb"], self.memory_stats["current_memory_mb"])
        
        logger.info(f"Initialized KV cache instance {cache_id} with {batch_size} batch size, "
                   f"{num_heads} heads, {self.head_dim} head dimension")
        logger.info(f"KV cache memory usage: {total_memory_mb:.2f}MB")
        
        return cache_id
    
    def update_cache(self, cache_id, keys, values, position):
        """
        Update the KV cache with new key-value pairs.
        
        Args:
            cache_id: ID of the cache to update
            keys: New key tensors to add
            values: New value tensors to add
            position: Position in the sequence
            
        Returns:
            Updated cache statistics
        """
        if cache_id not in self.cache_instances:
            raise ValueError(f"Cache ID {cache_id} not found")
        
        cache = self.cache_instances[cache_id]
        
        # First-time initialization
        if cache["keys"] is None:
            self._initialize_cache_tensors(cache_id)
        
        # Calculate cache position based on strategy
        cache_position = self._get_cache_position(cache_id, position)
        
        # Quantize keys and values if enabled
        if self.enable_quantization:
            keys = self._quantize_tensor(keys)
            values = self._quantize_tensor(values)
        
        # Update cache with new key-value pairs
        batch_size = keys.shape[0]
        num_heads = keys.shape[1]
        
        # Store keys and values at the calculated position
        for b in range(batch_size):
            for h in range(num_heads):
                # Update keys
                cache["keys"][b, h, cache_position] = keys[b, h]
                # Update values
                cache["values"][b, h, cache_position] = values[b, h]
        
        # Update position mapping
        cache["position_map"][position] = cache_position
        
        # Update access tracking
        if len(cache["usage_counts"]) <= cache_position:
            # Extend arrays if needed
            cache["usage_counts"].extend([0] * (cache_position - len(cache["usage_counts"]) + 1))
            cache["last_access"].extend([0] * (cache_position - len(cache["last_access"]) + 1))
            cache["pruning_scores"].extend([0] * (cache_position - len(cache["pruning_scores"]) + 1))
        
        cache["usage_counts"][cache_position] = 1
        cache["last_access"][cache_position] = time.time()
        
        # Update current length if needed
        cache["current_length"] = max(cache["current_length"], cache_position + 1)
        
        # Update memory statistics
        self.memory_stats["total_tokens_processed"] += 1
        
        return {
            "cache_id": cache_id,
            "position": position,
            "cache_position": cache_position,
            "current_length": cache["current_length"],
            "success": True
        }
    
    def get_cache_entries(self, cache_id, positions):
        """
        Retrieve KV pairs from cache.
        
        Args:
            cache_id: ID of the cache to retrieve from
            positions: List of positions to retrieve
            
        Returns:
            Dictionary containing keys and values for the requested positions
        """
        if cache_id not in self.cache_instances:
            raise ValueError(f"Cache ID {cache_id} not found")
        
        cache = self.cache_instances[cache_id]
        
        # Return empty result if cache is not yet initialized
        if cache["keys"] is None or cache["values"] is None:
            return {"keys": None, "values": None, "found": False}
        
        # Map original positions to cache positions
        cache_positions = []
        for pos in positions:
            if pos in cache["position_map"]:
                cache_positions.append(cache["position_map"][pos])
                # Update usage count and last access time
                cache_pos = cache["position_map"][pos]
                if cache_pos < len(cache["usage_counts"]):
                    cache["usage_counts"][cache_pos] += 1
                    cache["last_access"][cache_pos] = time.time()
            else:
                # Position not in cache
                return {"keys": None, "values": None, "found": False, "missing_position": pos}
        
        # Retrieve keys and values
        batch_size = cache["config"]["batch_size"]
        num_heads = cache["config"]["num_heads"]
        head_dim = cache["config"]["head_dim"]
        
        # Allocate tensors for the results
        result_keys = np.zeros((batch_size, num_heads, len(positions), head_dim), dtype=np.float32)
        result_values = np.zeros((batch_size, num_heads, len(positions), head_dim), dtype=np.float32)
        
        # Fill tensors with cache entries
        for i, cache_pos in enumerate(cache_positions):
            # Copy keys and values for all batches and heads
            for b in range(batch_size):
                for h in range(num_heads):
                    # Get from cache
                    cached_key = cache["keys"][b, h, cache_pos]
                    cached_value = cache["values"][b, h, cache_pos]
                    
                    # Dequantize if needed
                    if self.enable_quantization:
                        cached_key = self._dequantize_tensor(cached_key)
                        cached_value = self._dequantize_tensor(cached_value)
                    
                    # Store in result
                    result_keys[b, h, i] = cached_key
                    result_values[b, h, i] = cached_value
        
        # Update cache statistics
        self._update_cache_statistics(cache_id)
        
        return {
            "keys": result_keys,
            "values": result_values,
            "found": True,
            "positions": positions,
            "cache_positions": cache_positions
        }
    
    def clear_cache(self, cache_id):
        """
        Clear the KV cache.
        
        Args:
            cache_id: ID of the cache to clear
            
        Returns:
            Success status
        """
        if cache_id not in self.cache_instances:
            return {"success": False, "error": f"Cache ID {cache_id} not found"}
        
        # Get cache details for logging
        cache = self.cache_instances[cache_id]
        memory_freed = cache.get("memory_mb", 0)
        
        # Remove the cache
        del self.cache_instances[cache_id]
        
        # Update memory statistics
        self.memory_stats["current_memory_mb"] -= memory_freed
        
        logger.info(f"Cleared KV cache {cache_id}, freed {memory_freed:.2f}MB")
        
        return {"success": True, "memory_freed_mb": memory_freed}
    
    def prune_cache(self, cache_id, strategy="least_used"):
        """
        Prune the KV cache to reduce memory usage.
        
        Args:
            cache_id: ID of the cache to prune
            strategy: Pruning strategy ('least_used', 'least_recent', 'importance')
            
        Returns:
            Statistics about pruning operation
        """
        if not self.enable_pruning:
            return {"success": False, "reason": "Pruning is disabled"}
        
        if cache_id not in self.cache_instances:
            return {"success": False, "error": f"Cache ID {cache_id} not found"}
        
        cache = self.cache_instances[cache_id]
        
        # Only prune if we have a significant number of tokens
        if cache["current_length"] < 16:
            return {"success": False, "reason": "Cache too small to prune"}
        
        # Calculate tokens to keep (half of current length)
        tokens_to_keep = max(16, cache["current_length"] // 2)
        tokens_to_prune = cache["current_length"] - tokens_to_keep
        
        # Skip if nothing to prune
        if tokens_to_prune <= 0:
            return {"success": False, "reason": "No tokens to prune"}
        
        # Calculate pruning scores
        if strategy == "least_used":
            # Prune based on usage count (least used tokens first)
            scores = [-(count + 1) for count in cache["usage_counts"][:cache["current_length"]]]
        elif strategy == "least_recent":
            # Prune based on last access time (oldest first)
            current_time = time.time()
            scores = [-(current_time - last_time) for last_time in cache["last_access"][:cache["current_length"]]]
        elif strategy == "importance":
            # Use predetermined importance scores (e.g., from attention weights)
            scores = cache["pruning_scores"][:cache["current_length"]]
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")
        
        # Find indices to keep (highest scores)
        if len(scores) <= tokens_to_keep:
            # Nothing to prune
            return {"success": False, "reason": "No tokens to prune after scoring"}
        
        indices_to_keep = np.argsort(scores)[-tokens_to_keep:]
        indices_to_keep = sorted(indices_to_keep)  # Sort in ascending order
        
        # Create new position mapping
        new_position_map = {}
        for orig_pos, cache_pos in cache["position_map"].items():
            if cache_pos in indices_to_keep:
                # Get new position in the pruned cache
                new_pos = indices_to_keep.index(cache_pos)
                new_position_map[orig_pos] = new_pos
        
        # Create pruned cache tensors
        batch_size = cache["config"]["batch_size"]
        num_heads = cache["config"]["num_heads"]
        head_dim = cache["config"]["head_dim"]
        
        pruned_keys = np.zeros((batch_size, num_heads, tokens_to_keep, head_dim), dtype=np.float32)
        pruned_values = np.zeros((batch_size, num_heads, tokens_to_keep, head_dim), dtype=np.float32)
        
        # Copy data to pruned tensors
        for i, old_idx in enumerate(indices_to_keep):
            for b in range(batch_size):
                for h in range(num_heads):
                    pruned_keys[b, h, i] = cache["keys"][b, h, old_idx]
                    pruned_values[b, h, i] = cache["values"][b, h, old_idx]
        
        # Update usage statistics
        pruned_usage_counts = [cache["usage_counts"][i] for i in indices_to_keep]
        pruned_last_access = [cache["last_access"][i] for i in indices_to_keep]
        pruned_scores = [cache["pruning_scores"][i] if i < len(cache["pruning_scores"]) else 0 for i in indices_to_keep]
        
        # Update cache
        cache["keys"] = pruned_keys
        cache["values"] = pruned_values
        cache["position_map"] = new_position_map
        cache["current_length"] = tokens_to_keep
        cache["usage_counts"] = pruned_usage_counts
        cache["last_access"] = pruned_last_access
        cache["pruning_scores"] = pruned_scores
        
        # Update statistics
        self.memory_stats["pruned_tokens_count"] += tokens_to_prune
        
        logger.info(f"Pruned KV cache {cache_id}: removed {tokens_to_prune} tokens, kept {tokens_to_keep} tokens")
        
        return {
            "success": True,
            "tokens_pruned": tokens_to_prune,
            "tokens_kept": tokens_to_keep,
            "strategy": strategy
        }
    
    def get_cache_statistics(self, cache_id=None):
        """
        Get statistics for a specific cache or all caches.
        
        Args:
            cache_id: Optional ID of specific cache to get statistics for
            
        Returns:
            Dictionary of cache statistics
        """
        if cache_id:
            if cache_id not in self.cache_instances:
                return {"error": f"Cache ID {cache_id} not found"}
            
            cache = self.cache_instances[cache_id]
            
            return {
                "cache_id": cache_id,
                "batch_size": cache["config"]["batch_size"],
                "num_heads": cache["config"]["num_heads"],
                "head_dim": cache["config"]["head_dim"],
                "max_seq_length": cache["config"]["max_seq_length"],
                "current_length": cache["current_length"],
                "memory_mb": cache["memory_mb"],
                "quantized": cache["config"]["quantized"],
                "sliding_window": cache["config"]["sliding_window"],
                "window_size": cache["config"]["window_size"],
                "positions_cached": len(cache["position_map"]),
                "usage_stats": self._calculate_usage_stats(cache_id)
            }
        else:
            # Return global statistics
            num_caches = len(self.cache_instances)
            total_memory = sum(cache.get("memory_mb", 0) for cache in self.cache_instances.values())
            total_tokens = sum(cache.get("current_length", 0) for cache in self.cache_instances.values())
            
            return {
                "num_caches": num_caches,
                "total_memory_mb": total_memory,
                "current_memory_mb": self.memory_stats["current_memory_mb"],
                "peak_memory_mb": self.memory_stats["peak_memory_mb"],
                "total_tokens_cached": total_tokens,
                "total_tokens_processed": self.memory_stats["total_tokens_processed"],
                "total_tokens_pruned": self.memory_stats["pruned_tokens_count"],
                "cache_efficiency": self.memory_stats["cache_efficiency"],
                "cache_hit_ratio": self.memory_stats["cache_hit_ratio"],
                "cache_ids": list(self.cache_instances.keys())
            }
    
    def _initialize_cache_tensors(self, cache_id):
        """Initialize tensors for a KV cache instance."""
        cache = self.cache_instances[cache_id]
        
        keys_shape = cache["keys_shape"]
        values_shape = cache["values_shape"]
        
        # Allocate tensors
        cache["keys"] = np.zeros(keys_shape, dtype=np.float32)
        cache["values"] = np.zeros(values_shape, dtype=np.float32)
        
        # Initialize tracking arrays
        cache["usage_counts"] = [0] * keys_shape[2]  # Sequence length
        cache["last_access"] = [0] * keys_shape[2]  # Sequence length
        cache["pruning_scores"] = [0] * keys_shape[2]  # Sequence length
        
        logger.debug(f"Initialized KV cache tensors for {cache_id} with shapes {keys_shape} and {values_shape}")
    
    def _get_cache_position(self, cache_id, position):
        """Calculate cache position based on strategy."""
        cache = self.cache_instances[cache_id]
        
        if self.sliding_window:
            # Calculate position within sliding window
            max_len = cache["config"]["max_seq_length"]
            
            if position < max_len:
                # Direct mapping for positions within window size
                return position
            else:
                # For positions beyond window size, use circular buffer strategy
                return position % max_len
        else:
            # Direct mapping (position = cache position)
            return position
    
    def _quantize_tensor(self, tensor):
        """Quantize a tensor to 4-bit precision if quantization is enabled."""
        if not self.enable_quantization:
            return tensor
        
        try:
            quantized = self.quantizer.quantize_tensor(tensor)
            return quantized["data"]
        except Exception as e:
            logger.error(f"Error quantizing tensor: {e}")
            return tensor
    
    def _dequantize_tensor(self, quantized_tensor):
        """Dequantize a tensor from 4-bit precision if quantization is enabled."""
        if not self.enable_quantization:
            return quantized_tensor
        
        try:
            # Create a dummy quantized tensor dict for the dequantizer
            dummy_quantized = {
                "data": quantized_tensor,
                "scales": np.array([1.0], dtype=np.float32),  # Default scale
                "zero_points": None,
                "bits": 4,
                "group_size": 32,
                "scheme": "symmetric",
                "original_shape": quantized_tensor.shape,
                "original_dtype": "float32"
            }
            
            dequantized = self.quantizer.dequantize_tensor(dummy_quantized)
            return dequantized
        except Exception as e:
            logger.error(f"Error dequantizing tensor: {e}")
            return quantized_tensor
    
    def _update_cache_statistics(self, cache_id):
        """Update cache statistics after operations."""
        cache = self.cache_instances[cache_id]
        
        # Calculate cache hit ratio
        total_accesses = sum(cache["usage_counts"])
        total_positions = len(cache["position_map"])
        
        if total_accesses > 0:
            hit_ratio = total_positions / total_accesses
        else:
            hit_ratio = 0.0
        
        # Calculate cache efficiency
        total_space = cache["config"]["max_seq_length"]
        current_used = cache["current_length"]
        
        if total_space > 0:
            efficiency = current_used / total_space
        else:
            efficiency = 0.0
        
        # Update global statistics
        self.memory_stats["cache_hit_ratio"] = hit_ratio
        self.memory_stats["cache_efficiency"] = efficiency
    
    def _calculate_usage_stats(self, cache_id):
        """Calculate usage statistics for a cache instance."""
        cache = self.cache_instances[cache_id]
        
        # Skip if no usage data
        if not cache["usage_counts"]:
            return {
                "average_usage": 0,
                "max_usage": 0,
                "min_usage": 0
            }
        
        # Calculate usage statistics
        usage_counts = cache["usage_counts"][:cache["current_length"]]
        
        avg_usage = sum(usage_counts) / len(usage_counts) if usage_counts else 0
        max_usage = max(usage_counts) if usage_counts else 0
        min_usage = min(usage_counts) if usage_counts else 0
        
        return {
            "average_usage": avg_usage,
            "max_usage": max_usage,
            "min_usage": min_usage
        }

def setup_kv_cache_for_llm(model_name, max_seq_length=2048, head_dim=64, 
                          num_heads=16, batch_size=1, max_memory_mb=1000,
                          enable_quantization=True, sliding_window=True,
                          window_size=None):
    """
    Set up a KV cache manager for LLM inference.
    
    Args:
        model_name: Name of the model
        max_seq_length: Maximum sequence length
        head_dim: Dimension of each attention head
        num_heads: Number of attention heads
        batch_size: Batch size for inference
        max_memory_mb: Maximum memory allowed for KV cache in MB
        enable_quantization: Whether to enable 4-bit quantization
        sliding_window: Whether to use sliding window approach
        window_size: Size of the sliding window
        
    Returns:
        Tuple of (KV cache manager, cache ID)
    """
    # Create KV cache manager
    kv_manager = WebGPUKVCacheManager(
        max_seq_length=max_seq_length,
        head_dim=head_dim,
        max_memory_mb=max_memory_mb,
        enable_quantization=enable_quantization,
        sliding_window=sliding_window,
        window_size=window_size
    )
    
    # Initialize cache
    cache_id = kv_manager.initialize_cache(
        batch_size=batch_size,
        num_heads=num_heads,
        model_name=model_name
    )
    
    logger.info(f"Set up KV cache for {model_name} with {num_heads} heads, "
               f"max sequence length {max_seq_length}, head dimension {head_dim}")
    
    return kv_manager, cache_id

def generate_kv_cache_shaders(seq_length=2048, num_heads=16, head_dim=64, 
                             use_4bit=True, causal=True):
    """
    Generate WebGPU compute shaders for efficient KV cache operations.
    
    Args:
        seq_length: Maximum sequence length
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        use_4bit: Whether to use 4-bit precision
        causal: Whether to use causal attention masking
        
    Returns:
        Dictionary containing shader code for different operations
    """
    # Determine workgroup size
    workgroup_size = 128
    
    # Create shader template for KV cache access
    kv_access_shader = f"""
    // KV Cache Access Compute Shader for WebGPU
    // Configuration: seq_length={seq_length}, heads={num_heads}, head_dim={head_dim}, 
    // use_4bit={use_4bit}, causal={causal}
    
    struct Params {{
        seq_length: u32,
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
        current_length: u32,
        causal: u32,
        position: u32,
    }};
    
    @group(0) @binding(0) var<storage, read> input_q: array<f32>;
    @group(0) @binding(1) var<storage, read> cache_k: array<{"u8" if use_4bit else "f32"}>;
    @group(0) @binding(2) var<storage, read> cache_v: array<{"u8" if use_4bit else "f32"}>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    @group(0) @binding(5) var<storage, read> cache_scales: array<f32>;
    
    // Shared memory for tiles
    var<workgroup> tile_q: array<f32, {workgroup_size * head_dim}>;
    var<workgroup> tile_k: array<{"u8" if use_4bit else "f32"}, {workgroup_size * head_dim}>;
    var<workgroup> tile_v: array<{"u8" if use_4bit else "f32"}, {workgroup_size * head_dim}>;
    
    // Helper functions for 4-bit operations
    fn dequantize_4bit(value: u8, scale: f32, idx: u32) -> f32 {{
        // Extract the 4-bit value from packed byte
        var nibble: u32;
        if (idx % 2 == 0) {{
            // Extract lower 4 bits
            nibble = u32(value & 0x0F);
        }} else {{
            // Extract upper 4 bits
            nibble = u32((value >> 4) & 0x0F);
        }}
        
        // Convert to signed int in range [-8, 7]
        var signed_val: i32 = i32(nibble);
        if (signed_val > 7) {{
            signed_val = signed_val - 16;
        }}
        
        // Dequantize with scale
        return f32(signed_val) * scale;
    }}
    
    @compute @workgroup_size({workgroup_size}, 1, 1)
    fn main_kv_cache_access(
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
        let seq_idx = global_id.x; // Token index in sequence
        let head_idx = global_id.y; // Attention head index
        let batch_idx = global_id.z; // Batch index
        
        // Early exit if out of bounds
        if (seq_idx >= params.seq_length || head_idx >= params.num_heads || batch_idx >= params.batch_size) {{
            return;
        }}
        
        // Initialize output accumulators
        var output_vec: array<f32, {head_dim}>;
        for (var d = 0u; d < params.head_dim; d++) {{
            output_vec[d] = 0.0;
        }}
        
        // Load query vector for current token
        let q_offset = (batch_idx * params.num_heads * params.seq_length + 
                       head_idx * params.seq_length + 
                       seq_idx) * params.head_dim;
        
        // Load query vector into shared memory
        for (var d = 0u; d < params.head_dim; d++) {{
            tile_q[local_id.x * params.head_dim + d] = input_q[q_offset + d];
        }}
        
        // Compute attention using KV cache
        // ... KV cache access implementation ...
        
        // Write output
        let output_offset = (batch_idx * params.num_heads * params.seq_length + 
                           head_idx * params.seq_length + 
                           seq_idx) * params.head_dim;
        
        for (var d = 0u; d < params.head_dim; d++) {{
            output[output_offset + d] = output_vec[d];
        }}
    }}
    """
    
    # Shader for updating KV cache
    kv_update_shader = f"""
    // KV Cache Update Compute Shader for WebGPU
    // Configuration: seq_length={seq_length}, heads={num_heads}, head_dim={head_dim}, 
    // use_4bit={use_4bit}, causal={causal}
    
    struct Params {{
        seq_length: u32,
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
        position: u32,
        cache_position: u32,
    }};
    
    @group(0) @binding(0) var<storage, read> input_k: array<f32>;
    @group(0) @binding(1) var<storage, read> input_v: array<f32>;
    @group(0) @binding(2) var<storage, read_write> cache_k: array<{"u8" if use_4bit else "f32"}>;
    @group(0) @binding(3) var<storage, read_write> cache_v: array<{"u8" if use_4bit else "f32"}>;
    @group(0) @binding(4) var<uniform> params: Params;
    @group(0) @binding(5) var<storage, read_write> cache_scales: array<f32>;
    
    // Quantization helper function
    fn quantize_4bit(value: f32, scale: ptr<function, f32>) -> u8 {{
        // Determine scale if not provided
        if (*scale == 0.0) {{
            *scale = abs(value) / 7.0;
            if (*scale == 0.0) {{
                *scale = 1.0; // Avoid division by zero
            }}
        }}
        
        // Quantize to 4-bit signed integer (-8 to 7)
        var int_val = i32(round(value / *scale));
        int_val = clamp(int_val, -8, 7);
        
        // Convert to unsigned 4-bit (0-15)
        var uint_val = u32(int_val & 0xF);
        if (int_val < 0) {{
            uint_val = u32(int_val + 16);
        }}
        
        return u8(uint_val);
    }}
    
    @compute @workgroup_size({workgroup_size}, 1, 1)
    fn main_kv_cache_update(
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
        let head_dim_idx = global_id.x; // Index into head dimension
        let head_idx = global_id.y; // Attention head index
        let batch_idx = global_id.z; // Batch index
        
        // Early exit if out of bounds
        if (head_dim_idx >= params.head_dim || head_idx >= params.num_heads || batch_idx >= params.batch_size) {{
            return;
        }}
        
        // Compute input offsets
        let k_offset = (batch_idx * params.num_heads + head_idx) * params.head_dim + head_dim_idx;
        let v_offset = (batch_idx * params.num_heads + head_idx) * params.head_dim + head_dim_idx;
        
        // Compute cache offsets
        let cache_k_offset = (batch_idx * params.num_heads * params.seq_length + 
                             head_idx * params.seq_length + 
                             params.cache_position) * params.head_dim + head_dim_idx;
        let cache_v_offset = (batch_idx * params.num_heads * params.seq_length + 
                             head_idx * params.seq_length + 
                             params.cache_position) * params.head_dim + head_dim_idx;
        
        // Get input key and value
        let k_val = input_k[k_offset];
        let v_val = input_v[v_offset];
        
        // Process based on precision format
        if ({use_4bit}) {{
            // Calculate scale indices
            let k_scale_idx = (batch_idx * params.num_heads * params.seq_length + 
                              head_idx * params.seq_length + 
                              params.cache_position);
            let v_scale_idx = (batch_idx * params.num_heads * params.seq_length + 
                              head_idx * params.seq_length + 
                              params.cache_position) + (params.batch_size * params.num_heads * params.seq_length);
            
            // Get existing scales
            var k_scale = cache_scales[k_scale_idx];
            var v_scale = cache_scales[v_scale_idx];
            
            // Compute packed byte index and bit shift
            let k_byte_idx = cache_k_offset / 2;
            let k_shift = (cache_k_offset % 2) * 4; // 0 or 4 bits
            
            let v_byte_idx = cache_v_offset / 2;
            let v_shift = (cache_v_offset % 2) * 4; // 0 or 4 bits
            
            // Quantize to 4-bit
            var k_quant = quantize_4bit(k_val, &k_scale);
            var v_quant = quantize_4bit(v_val, &v_scale);
            
            // Update scales
            cache_scales[k_scale_idx] = k_scale;
            cache_scales[v_scale_idx] = v_scale;
            
            // Pack two 4-bit values into a byte (pair-wise packing)
            if (head_dim_idx % 2 == 0) {{
                // Even indices: initialize byte
                cache_k[k_byte_idx] = k_quant;
                cache_v[v_byte_idx] = v_quant;
            }} else {{
                // Odd indices: update upper 4 bits of previous byte
                var existing_k = cache_k[k_byte_idx];
                var existing_v = cache_v[v_byte_idx];
                
                cache_k[k_byte_idx] = (existing_k & 0x0F) | (k_quant << 4);
                cache_v[v_byte_idx] = (existing_v & 0x0F) | (v_quant << 4);
            }}
        }} else {{
            // Store in full precision
            cache_k[cache_k_offset] = k_val;
            cache_v[cache_v_offset] = v_val;
        }}
    }}
    """
    
    # Return shader code
    return {
        "kv_access": {
            "shader_code": kv_access_shader,
            "entry_point": "main_kv_cache_access",
            "workgroup_size": workgroup_size,
            "configuration": {
                "seq_length": seq_length,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "use_4bit": use_4bit,
                "causal": causal
            }
        },
        "kv_update": {
            "shader_code": kv_update_shader,
            "entry_point": "main_kv_cache_update",
            "workgroup_size": workgroup_size,
            "configuration": {
                "seq_length": seq_length,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "use_4bit": use_4bit,
                "causal": causal
            }
        }
    }

if __name__ == "__main__":
    # Example usage
    print("WebGPU KV Cache Optimization Module")
    print("===================================")
    
    # Example 1: Set up KV cache manager for LLM
    print("\nExample 1: Setting up KV cache manager")
    model_name = "llama-7b"
    kv_manager, cache_id = setup_kv_cache_for_llm(
        model_name=model_name,
        max_seq_length=2048,
        head_dim=128,
        num_heads=32,
        batch_size=1,
        max_memory_mb=1000,
        enable_quantization=True
    )
    
    # Example 2: Update KV cache with new tokens
    print("\nExample 2: Updating KV cache with new token")
    keys = np.random.randn(1, 32, 128).astype(np.float32)  # [batch_size, num_heads, head_dim]
    values = np.random.randn(1, 32, 128).astype(np.float32)  # [batch_size, num_heads, head_dim]
    
    result = kv_manager.update_cache(cache_id, keys, values, position=0)
    print(f"Updated cache at position {result['position']}, cache position {result['cache_position']}")
    
    # Example 3: Get entries from KV cache
    print("\nExample 3: Retrieving KV cache entries")
    entries = kv_manager.get_cache_entries(cache_id, positions=[0])
    print(f"Retrieved cache entries: found={entries['found']}")
    
    # Example 4: Get cache statistics
    print("\nExample 4: Cache statistics")
    stats = kv_manager.get_cache_statistics(cache_id)
    print(f"Cache memory: {stats['memory_mb']:.2f}MB")
    print(f"Current length: {stats['current_length']}")
    
    # Example 5: Generate shader code
    print("\nExample 5: Generate shader code")
    shaders = generate_kv_cache_shaders(seq_length=2048, num_heads=32, head_dim=128, use_4bit=True)
    print(f"Generated shaders for KV cache operations: {list(shaders.keys())}")