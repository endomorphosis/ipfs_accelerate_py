#\!/usr/bin/env python3
"""
Ultra-Low Precision Quantization for WebGPU (July 2025)

This module implements 2-bit and 3-bit quantization for WebGPU with adaptive precision,
specialized compute shaders, and mixed precision capabilities:

- 2-bit and 3-bit matrix multiplication kernels
- Adaptive precision for critical model layers
- Mixed precision across different components
- Quantization calibration and configuration
- Accuracy-performance tradeoff analysis
- Memory-aware precision adaptation

Usage:
    from fixed_web_platform.webgpu_ultra_low_precision import (
        setup_ultra_low_precision,
        create_2bit_compute_shaders,
        create_3bit_compute_shaders,
        quantize_model_mixed_precision,
        MixedPrecisionConfig,
        analyze_accuracy_performance_tradeoff
    )
    
    # Setup ultra-low precision configuration
    config = setup_ultra_low_precision(model, bits=2, adaptive=True)
    
    # Create specialized compute shaders
    shaders = create_2bit_compute_shaders()
    
    # Use the intelligent precision configuration 
    precision_config = MixedPrecisionConfig(model_type="transformer")
    
    # Optimize based on available memory
    precision_config.optimize_memory_usage(available_memory_mb=2048)
    
    # Quantize model with adaptive mixed precision
    quantized_model = quantize_model_mixed_precision(
        model, 
        precision_config=precision_config.precision_map
    )
    
    # Analyze accuracy-performance tradeoffs
    tradeoff_results = analyze_accuracy_performance_tradeoff(
        model=model,
        precision_configs=[
            {"embedding": 8, "attention": 4, "feed_forward": 2},  # Config A
            {"embedding": 8, "attention": 3, "feed_forward": 2},  # Config B
            {"embedding": 4, "attention": 3, "feed_forward": 2},  # Config C
        ],
        dataset=validation_dataset,
        metric_fn=calculate_accuracy
    )
"""

import os
import time
import math
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webgpu_ultra_low_precision")

def setup_ultra_low_precision(
    model: Any, 
    bits: int = 2, 
    adaptive: bool = True,
    group_size: int = 64,
    scheme: str = "symmetric",
    critical_layers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Set up ultra-low precision quantization configuration.
    
    Args:
        model: The model to quantize
        bits: Bit width for quantization (2 or 3)
        adaptive: Whether to use adaptive precision
        group_size: Group size for quantization
        scheme: Quantization scheme (symmetric or asymmetric)
        critical_layers: List of critical layers to use higher precision
    
    Returns:
        Configuration dictionary
    """
    if bits not in [2, 3]:
        raise ValueError("Ultra-low precision must be 2 or 3 bits")
    
    # Set default critical layers if not provided
    if critical_layers is None:
        critical_layers = [
            "attention.query", 
            "attention.key", 
            "lm_head",
            "embeddings"
        ]
    
    # Create configuration
    config = {
        "bits": bits,
        "group_size": group_size,
        "scheme": scheme,
        "adaptive_precision": adaptive,
        "critical_layers": critical_layers,
        "memory_reduction": 87.5 if bits == 2 else 81.25  # vs FP16
    }
    
    # Add advanced configuration
    if adaptive:
        config["precision_map"] = {
            layer: 8 for layer in critical_layers  # Use 8-bit for critical layers
        }
        config["default_precision"] = bits
        
        # Calculate effective memory reduction with adaptive precision
        critical_ratio = 0.15  # Approximately 15% of model parameters are in critical layers
        effective_bits = (bits * (1 - critical_ratio) + 8 * critical_ratio)
        config["effective_bits"] = effective_bits
        config["effective_memory_reduction"] = (16 - effective_bits) / 16 * 100
    
    # Add shader configuration
    if bits == 2:
        config["use_specialized_kernels"] = True
        config["dequant_cache_size"] = 256  # Dequantization cache size (MB)
        config["compute_shader_config"] = _get_2bit_shader_config()
    else:  # 3-bit
        config["use_specialized_kernels"] = True
        config["dequant_cache_size"] = 128  # Smaller cache for 3-bit
        config["compute_shader_config"] = _get_3bit_shader_config()
    
    logger.info(f"Ultra-low precision configuration: {bits}-bit, adaptive={adaptive}, group_size={group_size}")
    return config

def create_2bit_compute_shaders() -> Dict[str, str]:
    """
    Create specialized WebGPU compute shaders for 2-bit quantized operations.
    
    Returns:
        Dictionary of shader code by operation type
    """
    # Note: In a real implementation, these would be complete WGSL shader code
    # Here we just provide template entries
    
    shaders = {
        "matmul": _get_2bit_matmul_shader(),
        "dequantize": _get_2bit_dequantize_shader(),
        "attention": _get_2bit_attention_shader()
    }
    
    return shaders

def create_3bit_compute_shaders() -> Dict[str, str]:
    """
    Create specialized WebGPU compute shaders for 3-bit quantized operations.
    
    Returns:
        Dictionary of shader code by operation type
    """
    # Note: In a real implementation, these would be complete WGSL shader code
    # Here we just provide template entries
    
    shaders = {
        "matmul": _get_3bit_matmul_shader(),
        "dequantize": _get_3bit_dequantize_shader(),
        "attention": _get_3bit_attention_shader()
    }
    
    return shaders

def quantize_weights_2bit(
    weights: np.ndarray, 
    group_size: int = 64, 
    scheme: str = "symmetric"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize weights to 2-bit precision.
    
    Args:
        weights: Weight tensor to quantize
        group_size: Group size for quantization
        scheme: Quantization scheme (symmetric or asymmetric)
        
    Returns:
        Tuple of (quantized_weights, scales)
    """
    # This is a simplified implementation for demonstration
    # A real implementation would handle different tensor shapes and optimizations
    
    # Flatten weights for processing
    original_shape = weights.shape
    weights_flat = weights.reshape(-1)
    
    # Calculate number of groups
    num_elements = weights_flat.shape[0]
    num_groups = math.ceil(num_elements / group_size)
    
    # Create output arrays
    quantized = np.zeros(num_elements, dtype=np.uint8)
    scales = np.zeros(num_groups, dtype=np.float32)
    
    # Process each group
    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, num_elements)
        group = weights_flat[group_start:group_end]
        
        # Compute scale based on scheme
        if scheme == "symmetric":
            # Use abs max for symmetric quantization
            scale = np.max(np.abs(group))
            scales[group_idx] = scale
            
            # Skip empty or zero groups
            if scale == 0:
                continue
                
            # Quantize to 2-bit symmetric [-1.5, -0.5, 0.5, 1.5] * scale
            normalized = group / scale
            
            # Quantize to values 0, 1, 2, 3
            quant_values = np.clip(np.round(normalized / 0.5 + 2), 0, 3).astype(np.uint8)
            quantized[group_start:group_end] = quant_values
            
        else:  # asymmetric
            # Use min/max for asymmetric quantization
            min_val = np.min(group)
            max_val = np.max(group)
            scale = (max_val - min_val) / 3.0
            
            # Skip empty or constant groups
            if scale == 0:
                scales[group_idx] = 0
                continue
                
            scales[group_idx] = scale
            
            # Quantize to 2-bit range [0, 1, 2, 3] mapping to [min_val, min_val+scale, ..., max_val]
            normalized = (group - min_val) / scale
            quant_values = np.clip(np.round(normalized), 0, 3).astype(np.uint8)
            quantized[group_start:group_end] = quant_values
    
    # Reshape quantized weights back to original shape
    quantized = quantized.reshape(original_shape)
    
    return quantized, scales

def quantize_weights_3bit(
    weights: np.ndarray, 
    group_size: int = 128, 
    scheme: str = "symmetric"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize weights to 3-bit precision.
    
    Args:
        weights: Weight tensor to quantize
        group_size: Group size for quantization
        scheme: Quantization scheme (symmetric or asymmetric)
        
    Returns:
        Tuple of (quantized_weights, scales)
    """
    # This is a simplified implementation for demonstration
    # A real implementation would handle different tensor shapes and optimizations
    
    # Flatten weights for processing
    original_shape = weights.shape
    weights_flat = weights.reshape(-1)
    
    # Calculate number of groups
    num_elements = weights_flat.shape[0]
    num_groups = math.ceil(num_elements / group_size)
    
    # Create output arrays
    quantized = np.zeros(num_elements, dtype=np.uint8)
    scales = np.zeros(num_groups, dtype=np.float32)
    
    # Process each group
    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, num_elements)
        group = weights_flat[group_start:group_end]
        
        # Compute scale based on scheme
        if scheme == "symmetric":
            # Use abs max for symmetric quantization
            scale = np.max(np.abs(group))
            scales[group_idx] = scale
            
            # Skip empty or zero groups
            if scale == 0:
                continue
                
            # Quantize to 3-bit symmetric range (-3.5, -2.5, ... 3.5) * scale/4
            normalized = group / (scale / 4)
            
            # Quantize to values 0-7
            quant_values = np.clip(np.round(normalized + 4), 0, 7).astype(np.uint8)
            quantized[group_start:group_end] = quant_values
            
        else:  # asymmetric
            # Use min/max for asymmetric quantization
            min_val = np.min(group)
            max_val = np.max(group)
            scale = (max_val - min_val) / 7.0
            
            # Skip empty or constant groups
            if scale == 0:
                scales[group_idx] = 0
                continue
                
            scales[group_idx] = scale
            
            # Quantize to 3-bit range [0-7] mapping to [min_val, min_val+scale, ..., max_val]
            normalized = (group - min_val) / scale
            quant_values = np.clip(np.round(normalized), 0, 7).astype(np.uint8)
            quantized[group_start:group_end] = quant_values
    
    # Reshape quantized weights back to original shape
    quantized = quantized.reshape(original_shape)
    
    return quantized, scales

def quantize_model_mixed_precision(
    model: Any,
    precision_config: Dict[str, int]
) -> Dict[str, Any]:
    """
    Quantize a model with mixed precision across different components.
    
    Args:
        model: The model to quantize
        precision_config: Dict mapping layer patterns to bit widths
        
    Returns:
        Quantized model with mixed precision
    """
    # This is a simplified implementation for demonstration
    # A real implementation would work with actual model architectures
    
    # Track quantization stats
    stats = {
        "total_params": 0,
        "memory_reduction": 0,
        "layer_stats": {},
        "bit_distribution": {2: 0, 3: 0, 4: 0, 8: 0, 16: 0}
    }
    
    # Track memory for each precision
    memory_by_precision = {2: 0, 3: 0, 4: 0, 8: 0, 16: 0}
    
    # Simulate quantization for parameter groups
    # In a real implementation, this would iterate through actual model layers
    for layer_name, params in model.items():
        # Skip non-parameter entries
        if not isinstance(params, dict) or "weight" not in params:
            continue
            
        # Get weight tensor
        weight = params["weight"]
        num_params = np.prod(weight.shape)
        stats["total_params"] += num_params
        
        # Determine precision for this layer
        precision = _get_precision_for_layer(layer_name, precision_config)
        
        # Quantize with appropriate precision
        if precision == 2:
            # 2-bit quantization
            quant_weight, scales = quantize_weights_2bit(weight)
            memory_bytes = (num_params * 2) / 8  # 2 bits per parameter
        elif precision == 3:
            # 3-bit quantization
            quant_weight, scales = quantize_weights_3bit(weight)
            memory_bytes = (num_params * 3) / 8  # 3 bits per parameter
        elif precision == 4:
            # 4-bit quantization (simplified)
            quant_weight, scales = weight, None  # Placeholder
            memory_bytes = (num_params * 4) / 8  # 4 bits per parameter
        elif precision == 8:
            # 8-bit quantization (simplified)
            quant_weight, scales = weight, None  # Placeholder
            memory_bytes = num_params  # 8 bits per parameter
        else:
            # FP16 (no quantization)
            quant_weight, scales = weight, None
            memory_bytes = num_params * 2  # 16 bits per parameter
            precision = 16
        
        # Update stats
        memory_by_precision[precision] += memory_bytes
        stats["bit_distribution"][precision] += num_params
        
        # Store layer stats
        stats["layer_stats"][layer_name] = {
            "precision": precision,
            "params": num_params,
            "memory_bytes": memory_bytes
        }
    
    # Calculate overall memory reduction vs FP16
    fp16_memory = stats["total_params"] * 2  # 16 bits per parameter
    quantized_memory = sum(memory_by_precision.values())
    memory_reduction = (fp16_memory - quantized_memory) / fp16_memory * 100
    
    # Update final stats
    stats["memory_reduction"] = memory_reduction
    stats["quantized_memory_mb"] = quantized_memory / (1024 * 1024)
    stats["original_memory_mb"] = fp16_memory / (1024 * 1024)
    
    # Convert bit distribution to percentages
    for precision in stats["bit_distribution"]:
        stats["bit_distribution"][precision] = (
            stats["bit_distribution"][precision] / stats["total_params"] * 100
        )
    
    logger.info(f"Model quantized with mixed precision. Memory reduction: {memory_reduction:.2f}%")
    return {
        "model": model,  # In reality, this would be the quantized model
        "stats": stats
    }

def analyze_accuracy_performance_tradeoff(
    model: Any,
    precision_configs: List[Dict[str, int]],
    dataset: Any,
    metric_fn: Callable
) -> Dict[str, Any]:
    """
    Analyze the accuracy-performance tradeoff for different precision configurations.
    
    Args:
        model: The model to analyze
        precision_configs: List of precision configurations to test
        dataset: Evaluation dataset
        metric_fn: Function to compute accuracy metric
        
    Returns:
        Analysis results
    """
    # This is a simplified implementation for demonstration
    # A real implementation would actually run the model on a dataset
    
    results = []
    
    for i, config in enumerate(precision_configs):
        # Simulate quantizing the model with this config
        quantized = quantize_model_mixed_precision(model, config)
        
        # Simulate evaluation
        start_time = time.time()
        time.sleep(0.1)  # Simulate evaluation time
        elapsed = time.time() - start_time
        
        # Simulate accuracy drop based on precision config
        # Lower precision -> more accuracy drop
        accuracy_drop = _estimate_accuracy_drop(config)
        
        # Collect results
        results.append({
            "config_id": i,
            "precision_config": config,
            "memory_reduction": quantized["stats"]["memory_reduction"],
            "accuracy_drop": accuracy_drop,
            "eval_time": elapsed,
            "bit_distribution": quantized["stats"]["bit_distribution"]
        })
    
    # Find Pareto optimal configurations
    pareto_optimal = _find_pareto_optimal_configs(results)
    
    # Return comprehensive analysis
    return {
        "all_configs": results,
        "pareto_optimal": pareto_optimal,
        "recommended_config": _find_recommended_config(results)
    }

def _get_precision_for_layer(layer_name: str, precision_config: Dict[str, int]) -> int:
    """
    Determine the precision to use for a layer based on precision config.
    
    Args:
        layer_name: Name of the layer
        precision_config: Dict mapping layer patterns to bit widths
        
    Returns:
        Bit width to use for the layer
    """
    # Default to 16-bit if no match
    default_precision = 16
    
    # Check for exact match
    if layer_name in precision_config:
        return precision_config[layer_name]
    
    # Check for pattern match
    for pattern, precision in precision_config.items():
        if pattern in layer_name:
            return precision
    
    return default_precision

def _estimate_accuracy_drop(precision_config: Dict[str, int]) -> float:
    """
    Estimate accuracy drop based on precision configuration.
    
    Args:
        precision_config: Dict mapping layer patterns to bit widths
        
    Returns:
        Estimated accuracy drop percentage
    """
    # Base accuracy drops for different bit widths
    base_drops = {
        2: 8.0,   # 2-bit has significant drop
        3: 4.0,   # 3-bit has moderate drop
        4: 2.5,   # 4-bit has small drop
        8: 1.0,   # 8-bit has very small drop
        16: 0.0   # 16-bit has no drop (reference)
    }
    
    # Count parameters at each precision level (simplified estimate)
    precision_counts = {2: 0, 3: 0, 4: 0, 8: 0, 16: 0}
    
    # In a real implementation, this would consider the actual parameter counts
    # Here we just use the number of layer patterns as a proxy
    for _, precision in precision_config.items():
        precision_counts[precision] += 1
    
    # Normalize counts to get distribution
    total_count = sum(precision_counts.values())
    if total_count == 0:
        return 0.0
        
    precision_dist = {p: count / total_count for p, count in precision_counts.items()}
    
    # Calculate weighted accuracy drop
    weighted_drop = 0.0
    for precision, dist in precision_dist.items():
        weighted_drop += base_drops[precision] * dist
    
    return weighted_drop

def _find_pareto_optimal_configs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find Pareto optimal configurations from results.
    
    Args:
        results: List of configuration results
        
    Returns:
        List of Pareto optimal configurations
    """
    pareto_optimal = []
    
    for i, config_i in enumerate(results):
        is_dominated = False
        
        for j, config_j in enumerate(results):
            if i == j:
                continue
                
            # Check if config_j dominates config_i
            if (config_j["memory_reduction"] >= config_i["memory_reduction"] and
                config_j["accuracy_drop"] <= config_i["accuracy_drop"] and
                (config_j["memory_reduction"] > config_i["memory_reduction"] or 
                 config_j["accuracy_drop"] < config_i["accuracy_drop"])):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_optimal.append(config_i)
    
    return pareto_optimal

def _find_recommended_config(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find recommended configuration based on balanced accuracy and memory.
    
    Args:
        results: List of configuration results
        
    Returns:
        Recommended configuration
    """
    # Normalize metrics
    max_memory_reduction = max(r["memory_reduction"] for r in results)
    max_accuracy_drop = max(r["accuracy_drop"] for r in results)
    
    # Avoid division by zero
    if max_memory_reduction == 0 or max_accuracy_drop == 0:
        return results[0]
    
    best_score = -float('inf')
    best_config = None
    
    for config in results:
        # Normalize metrics to [0, 1]
        norm_memory = config["memory_reduction"] / max_memory_reduction
        norm_accuracy = 1.0 - (config["accuracy_drop"] / max_accuracy_drop)
        
        # Compute balanced score (weight accuracy more)
        score = 0.4 * norm_memory + 0.6 * norm_accuracy
        
        if score > best_score:
            best_score = score
            best_config = config
    
    return best_config

def _get_2bit_matmul_shader() -> str:
    """
    Get 2-bit matrix multiplication shader code for WebGPU.
    
    Returns:
        WGSL shader code for 2-bit matrix multiplication
    """
    return """
    // 2-bit matrix multiplication shader for WebGPU (June 2025)
    // Optimized for memory efficiency and computation speed
    
    @group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
    @group(0) @binding(1) var<storage, read> weight_quantized: array<u32>;
    @group(0) @binding(2) var<storage, read> weight_scales: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;
    
    struct Params {
        M: u32,  // Batch size or output rows
        N: u32,  // Output dimension
        K: u32,  // Input dimension
        group_size: u32,  // Quantization group size
        use_cache: u32,   // Whether to use dequant cache
    }
    @group(0) @binding(4) var<uniform> params: Params;
    
    // Constants for 2-bit quantization
    const BITS_PER_VALUE: u32 = 2u;
    const VALUES_PER_WORD: u32 = 16u;  // 32 bits / 2 bits per value
    const QUANT_MASK: u32 = 3u;  // 0b11
    
    // Shared memory for cached matrix tiles and dequantized weights
    var<workgroup> tile_a: array<f32, 8 * 32>;  // Input tile cache
    var<workgroup> dequant_cache: array<f32, 32 * 32>;  // Dequantized weights cache
    
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) group_id: vec3<u32>) {
        
        let row = global_id.x;
        let col = global_id.y;
        let local_row = local_id.x;
        let local_col = local_id.y;
        
        // Early exit for out-of-bounds threads
        if (row >= params.M || col >= params.N) {
            return;
        }
        
        var sum: f32 = 0.0;
        
        // Process input in tiles for better cache locality
        for (var tile_start: u32 = 0u; tile_start < params.K; tile_start += 32u) {
            // Load input tile into shared memory
            if (local_col < 4u) {  // Each thread loads 4 elements
                for (var i: u32 = 0u; i < 4u; i++) {
                    let k_idx = tile_start + local_col * 4u + i;
                    if (k_idx < params.K) {
                        tile_a[local_row * 32u + local_col * 4u + i] = input_tensor[row * params.K + k_idx];
                    } else {
                        tile_a[local_row * 32u + local_col * 4u + i] = 0.0;
                    }
                }
            }
            
            // Load and dequantize weights tile cooperatively
            if (local_row * 8u + local_col < 32u) {
                let thread_idx = local_row * 8u + local_col;
                let weights_idx = tile_start + thread_idx;
                
                // Each thread dequantizes 16 weight values (one 32-bit word)
                if (weights_idx < params.K) {
                    let word_idx = weights_idx;
                    let packed_word = weight_quantized[word_idx];
                    
                    // Determine quantization group and scale
                    let group_idx = weights_idx / params.group_size;
                    let scale = weight_scales[group_idx];
                    
                    // Dequantize 16 weight values
                    for (var i: u32 = 0u; i < 16u; i++) {
                        let bit_offset = i * BITS_PER_VALUE;
                        let quant_value = (packed_word >> bit_offset) & QUANT_MASK;
                        
                        // Dequantize: 0->-1.5, 1->-0.5, 2->0.5, 3->1.5
                        // This symmetric quantization reduces quantization error
                        let weight_value = (f32(quant_value) - 1.5) * scale;
                        
                        // Store in shared memory cache
                        let cache_idx = thread_idx * 16u + i;
                        if (cache_idx < 32u * 32u) {
                            dequant_cache[cache_idx] = weight_value;
                        }
                    }
                }
            }
            
            // Sync to ensure all shared memory writes are complete
            workgroupBarrier();
            
            // Compute partial matrix multiplication for this tile
            for (var k: u32 = 0u; k < 32u; k++) {
                if (tile_start + k < params.K) {
                    // Use cached input and dequantized weight values
                    let input_val = tile_a[local_row * 32u + k];
                    let weight_val = dequant_cache[k * 16u + col % 16u];
                    sum += input_val * weight_val;
                }
            }
            
            // Sync before loading next tile
            workgroupBarrier();
        }
        
        // Write result to output
        output_tensor[row * params.N + col] = sum;
    }
    """

def _get_3bit_matmul_shader() -> str:
    """
    Get 3-bit matrix multiplication shader code for WebGPU.
    
    Returns:
        WGSL shader code for 3-bit matrix multiplication
    """
    return """
    // 3-bit matrix multiplication shader for WebGPU (June 2025)
    // Optimized for memory efficiency and computation speed
    
    @group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
    @group(0) @binding(1) var<storage, read> weight_quantized: array<u32>;
    @group(0) @binding(2) var<storage, read> weight_scales: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;
    
    struct Params {
        M: u32,  // Batch size or output rows
        N: u32,  // Output dimension
        K: u32,  // Input dimension
        group_size: u32,  // Quantization group size
        use_cache: u32,   // Whether to use dequant cache
    }
    @group(0) @binding(4) var<uniform> params: Params;
    
    // Constants for 3-bit quantization
    const BITS_PER_VALUE: u32 = 3u;
    const VALUES_PER_WORD: u32 = 10u;  // Approx 10 complete 3-bit values per 32-bit word
    const QUANT_MASK: u32 = 7u;  // 0b111
    
    // Shared memory for cached matrix tiles and dequantized weights
    var<workgroup> tile_a: array<f32, 8 * 32>;  // Input tile cache
    var<workgroup> dequant_cache: array<f32, 32 * 32>;  // Dequantized weights cache
    
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) group_id: vec3<u32>) {
        
        let row = global_id.x;
        let col = global_id.y;
        let local_row = local_id.x;
        let local_col = local_id.y;
        
        // Early exit for out-of-bounds threads
        if (row >= params.M || col >= params.N) {
            return;
        }
        
        var sum: f32 = 0.0;
        
        // Process input in tiles for better cache locality
        for (var tile_start: u32 = 0u; tile_start < params.K; tile_start += 32u) {
            // Load input tile into shared memory
            if (local_col < 4u) {  // Each thread loads 4 elements
                for (var i: u32 = 0u; i < 4u; i++) {
                    let k_idx = tile_start + local_col * 4u + i;
                    if (k_idx < params.K) {
                        tile_a[local_row * 32u + local_col * 4u + i] = input_tensor[row * params.K + k_idx];
                    } else {
                        tile_a[local_row * 32u + local_col * 4u + i] = 0.0;
                    }
                }
            }
            
            // Load and dequantize weights tile cooperatively
            // 3-bit packing is more complex than 2-bit: need to handle crossing boundaries
            if (local_row * 8u + local_col < 32u) {
                let thread_idx = local_row * 8u + local_col;
                let weights_start_idx = tile_start + thread_idx * 10u; // Each thread handles ~10 values
                
                // Each thread processes up to 10 weight values from potentially multiple 32-bit words
                for (var i: u32 = 0u; i < 10u; i++) {
                    let weight_idx = weights_start_idx + i;
                    
                    if (weight_idx < params.K) {
                        // 3-bit values can cross 32-bit word boundaries
                        // Calculate which 32-bit word contains this value's starting bits
                        let bit_pos = weight_idx * BITS_PER_VALUE;
                        let word_idx = bit_pos / 32u;
                        let bit_offset = bit_pos % 32u;
                        
                        // Get the quantized value, handling potential word boundary crossing
                        var quant_value: u32;
                        
                        if (bit_offset <= 29u) {
                            // Value fits within a single word
                            quant_value = (weight_quantized[word_idx] >> bit_offset) & QUANT_MASK;
                        } else {
                            // Value crosses word boundary
                            // Get lower bits from current word
                            let lower_bits = (weight_quantized[word_idx] >> bit_offset);
                            // Get upper bits from next word
                            let upper_bits = (weight_quantized[word_idx + 1u] << (32u - bit_offset));
                            // Combine and mask
                            quant_value = (lower_bits | (upper_bits & (QUANT_MASK << (3u - (32u - bit_offset))))) & QUANT_MASK;
                        }
                        
                        // Determine quantization group and scale
                        let group_idx = weight_idx / params.group_size;
                        let scale = weight_scales[group_idx];
                        
                        // Dequantize: map 0-7 to -3.5 to 3.5 in steps of 1.0
                        // This symmetric quantization reduces quantization error
                        let weight_value = (f32(quant_value) - 3.5) * (scale / 4.0);
                        
                        // Store in shared memory cache
                        let cache_idx = thread_idx * 10u + i;
                        if (cache_idx < 32u * 32u) {
                            dequant_cache[cache_idx] = weight_value;
                        }
                    }
                }
            }
            
            // Sync to ensure all shared memory writes are complete
            workgroupBarrier();
            
            // Compute partial matrix multiplication for this tile
            for (var k: u32 = 0u; k < 32u; k++) {
                if (tile_start + k < params.K) {
                    // Use cached input and dequantized weight values
                    let input_val = tile_a[local_row * 32u + k];
                    
                    // Determine which thread's cache contains this weight
                    let thread_idx = k / 10u;
                    let value_idx = k % 10u;
                    let cache_idx = thread_idx * 10u + value_idx;
                    
                    if (cache_idx < 32u * 32u) {
                        let weight_val = dequant_cache[cache_idx];
                        sum += input_val * weight_val;
                    }
                }
            }
            
            // Sync before loading next tile
            workgroupBarrier();
        }
        
        // Write result to output
        output_tensor[row * params.N + col] = sum;
    }
    """

def _get_2bit_dequantize_shader() -> str:
    """Get 2-bit dequantization shader code for WebGPU."""
    # Template for dequantization shader
    return """
    // 2-bit dequantization shader for WebGPU
    // This is a template - a real implementation would have complete shader code
    
    @group(0) @binding(0) var<storage, read> quantized: array<u32>;
    @group(0) @binding(1) var<storage, read> scales: array<f32>;
    @group(0) @binding(2) var<storage, read_write> dequantized: array<f32>;
    
    struct Params {
        num_elements: u32,
        group_size: u32,
    }
    @group(0) @binding(3) var<uniform> params: Params;
    
    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        
        if (idx >= params.num_elements) {
            return;
        }
        
        let group_idx = idx / params.group_size;
        let scale = scales[group_idx];
        
        // Get quantized value (packed in 32-bit words)
        let values_per_word = 16u;  // 32 bits / 2 bits per value
        let word_idx = idx / values_per_word;
        let bit_offset = (idx % values_per_word) * 2u;
        
        let packed = quantized[word_idx];
        let quant_value = (packed >> bit_offset) & 3u;
        
        // Dequantize based on symmetric 2-bit quantization
        // 0 -> -1.5, 1 -> -0.5, 2 -> 0.5, 3 -> 1.5
        let value = (f32(quant_value) - 1.5) * scale;
        
        dequantized[idx] = value;
    }
    """

def _get_3bit_dequantize_shader() -> str:
    """Get 3-bit dequantization shader code for WebGPU."""
    # Template for dequantization shader
    return """
    // 3-bit dequantization shader for WebGPU
    // This is a template - a real implementation would have complete shader code
    
    @group(0) @binding(0) var<storage, read> quantized: array<u32>;
    @group(0) @binding(1) var<storage, read> scales: array<f32>;
    @group(0) @binding(2) var<storage, read_write> dequantized: array<f32>;
    
    struct Params {
        num_elements: u32,
        group_size: u32,
    }
    @group(0) @binding(3) var<uniform> params: Params;
    
    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        
        if (idx >= params.num_elements) {
            return;
        }
        
        let group_idx = idx / params.group_size;
        let scale = scales[group_idx];
        
        // 3-bit packing is more complex than 2-bit
        // One 32-bit word contains 10 complete 3-bit values, with 2 bits remaining
        // This requires careful handling of values that cross word boundaries
        
        // Simplified approach for template - real implementation would be more complex
        let values_per_word = 10u;  // Approximate - real version handles boundary crossing
        let word_idx = idx / values_per_word;
        let bit_offset = (idx % values_per_word) * 3u;
        
        let packed = quantized[word_idx];
        let quant_value = (packed >> bit_offset) & 7u;
        
        // Dequantize: map 0-7 to -3.5 to 3.5 in steps of 1.0
        let value = (f32(quant_value) - 3.5) * (scale / 4.0);
        
        dequantized[idx] = value;
    }
    """

def _get_2bit_attention_shader() -> str:
    """Get 2-bit attention computation shader code for WebGPU."""
    # Template for attention shader with 2-bit weights
    return """
    // 2-bit quantized attention shader for WebGPU
    // This is a template - a real implementation would have complete shader code
    
    // Various bindings for attention computation
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> q_weight_quantized: array<u32>;
    @group(0) @binding(2) var<storage, read> q_weight_scales: array<f32>;
    @group(0) @binding(3) var<storage, read> k_weight_quantized: array<u32>;
    @group(0) @binding(4) var<storage, read> k_weight_scales: array<f32>;
    @group(0) @binding(5) var<storage, read> v_weight_quantized: array<u32>;
    @group(0) @binding(6) var<storage, read> v_weight_scales: array<f32>;
    @group(0) @binding(7) var<storage, read_write> output: array<f32>;
    
    struct Params {
        batch_size: u32,
        seq_length: u32,
        num_heads: u32,
        head_dim: u32,
        group_size: u32,
    }
    @group(0) @binding(8) var<uniform> params: Params;
    
    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        // Attention computation with 2-bit quantized weights
        // Simplified template - real implementation would be more complex
        
        // ... code for computing attention with 2-bit weights ...
    }
    """

def _get_3bit_attention_shader() -> str:
    """Get 3-bit attention computation shader code for WebGPU."""
    # Template for attention shader with 3-bit weights
    return """
    // 3-bit quantized attention shader for WebGPU
    // This is a template - a real implementation would have complete shader code
    
    // Various bindings for attention computation
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> q_weight_quantized: array<u32>;
    @group(0) @binding(2) var<storage, read> q_weight_scales: array<f32>;
    @group(0) @binding(3) var<storage, read> k_weight_quantized: array<u32>;
    @group(0) @binding(4) var<storage, read> k_weight_scales: array<f32>;
    @group(0) @binding(5) var<storage, read> v_weight_quantized: array<u32>;
    @group(0) @binding(6) var<storage, read> v_weight_scales: array<f32>;
    @group(0) @binding(7) var<storage, read_write> output: array<f32>;
    
    struct Params {
        batch_size: u32,
        seq_length: u32,
        num_heads: u32,
        head_dim: u32,
        group_size: u32,
    }
    @group(0) @binding(8) var<uniform> params: Params;
    
    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        // Attention computation with 3-bit quantized weights
        // Simplified template - real implementation would be more complex
        
        // ... code for computing attention with 3-bit weights ...
    }
    """

def _get_2bit_shader_config() -> Dict[str, Any]:
    """Get shader configuration for 2-bit quantized operations."""
    return {
        "workgroup_size": (8, 8, 1),
        "shared_memory_bytes": 8192,
        "values_per_byte": 4,
        "values_per_word": 16,
        "use_unroll": True,
        "use_shared_memory": True,
        "use_dequant_cache": True
    }

def _get_3bit_shader_config() -> Dict[str, Any]:
    """Get shader configuration for 3-bit quantized operations."""
    return {
        "workgroup_size": (8, 8, 1),
        "shared_memory_bytes": 8192,
        "values_per_byte": 2.67,  # Approximate
        "values_per_word": 10.67,  # Approximate
        "use_unroll": True,
        "use_shared_memory": True,
        "use_dequant_cache": True
    }

class MixedPrecisionConfig:
    """
    Configuration for mixed precision quantization across model components.
    
    This class handles the intelligent distribution of precision across
    different model components based on their importance and sensitivity.
    
    July 2025 Update:
    - Added memory-aware optimization
    - Added browser-specific optimizations 
    - Added accuracy-performance tradeoff analyzer
    - Added support for browser capabilities detection
    """
    
    def __init__(self, model_type="transformer", default_bits=2):
        """
        Initialize mixed precision configuration.
        
        Args:
            model_type: Type of model (transformer, vision, audio, etc.)
            default_bits: Default bit width for quantization
        """
        self.model_type = model_type.lower()
        self.default_bits = default_bits
        self.critical_layers = self._get_critical_layers()
        self.precision_map = self._create_precision_map()
        
    def _get_critical_layers(self):
        """
        Identify critical layers based on model type.
        
        Returns:
            Dictionary mapping layer patterns to importance scores (0-10)
        """
        # Base critical layers for all transformer models
        critical_layers = {
            "embedding": 9,  # Embeddings are critical
            "lm_head": 9,    # Output projections are critical
            "attention.query": 8,
            "attention.key": 8,
            "attention.value": 7,
            "layer_norm": 7,  # Layer norms need higher precision
            "feed_forward": 3,  # Feed forward are less sensitive
        }
        
        # Add model-specific critical layers
        if self.model_type == "vision":
            critical_layers.update({
                "vision_projection": 9,
                "patch_embedding": 8,
                "pooler": 7,
            })
        elif self.model_type == "audio":
            critical_layers.update({
                "feature_extractor": 9,
                "spectrogram": 8,
                "conv_layers": 7,
            })
        elif self.model_type == "multimodal":
            critical_layers.update({
                "vision_encoder": 9,
                "cross_attention": 8,
                "projection": 8,
            })
            
        return critical_layers
    
    def _create_precision_map(self):
        """
        Create precision map for model components.
        
        Returns:
            Dictionary mapping layer patterns to bit widths
        """
        precision_map = {}
        
        # Convert importance scores to precision bits
        for layer, importance in self.critical_layers.items():
            if importance >= 9:
                # Most critical layers use 8-bit
                precision_map[layer] = 8
            elif importance >= 7:
                # Important layers use 4-bit
                precision_map[layer] = 4
            elif importance >= 5:
                # Moderately important layers use 3-bit
                precision_map[layer] = 3
            else:
                # Less critical layers use default precision
                precision_map[layer] = self.default_bits
                
        return precision_map
    
    def get_precision_for_layer(self, layer_name):
        """
        Get precision for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Precision in bits
        """
        # First check for exact match
        if layer_name in self.precision_map:
            return self.precision_map[layer_name]
            
        # Then check for partial matches
        for pattern, bits in self.precision_map.items():
            if pattern in layer_name:
                return bits
                
        # Default to the global default precision
        return self.default_bits
    
    def optimize_memory_usage(self, available_memory_mb):
        """
        Optimize precision configuration based on available memory.
        
        Args:
            available_memory_mb: Available memory in MB
            
        Returns:
            Optimized precision map
        """
        optimized_map = self.precision_map.copy()
        
        # For very constrained memory, reduce precision of less critical layers
        if available_memory_mb < 500:
            for layer, importance in self.critical_layers.items():
                if importance < 7:
                    # Lower precision for non-critical layers
                    optimized_map[layer] = min(optimized_map[layer], 2)
        
        # For even more constrained memory, also reduce some important layers
        if available_memory_mb < 250:
            for layer, importance in self.critical_layers.items():
                if importance < 9:
                    # Further reduce precision for moderately important layers
                    optimized_map[layer] = min(optimized_map[layer], 3)
        
        return optimized_map
    
    def get_memory_reduction(self):
        """
        Estimate memory reduction compared to FP16.
        
        Returns:
            Dictionary with memory reduction statistics
        """
        # Count layers per precision
        precision_counts = {2: 0, 3: 0, 4: 0, 8: 0}
        total_layers = len(self.critical_layers)
        
        for layer, importance in self.critical_layers.items():
            precision = self.get_precision_for_layer(layer)
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
            
        # Calculate weighted average precision
        weighted_bits = 0
        for bits, count in precision_counts.items():
            weighted_bits += bits * (count / total_layers)
            
        # Calculate memory reduction vs FP16
        reduction_percentage = (16 - weighted_bits) / 16 * 100
        
        return {
            "precision_distribution": {
                f"{bits}-bit": f"{count/total_layers*100:.1f}%" 
                for bits, count in precision_counts.items() if count > 0
            },
            "average_bits": weighted_bits,
            "memory_reduction_percent": reduction_percentage,
            "effective_compression_ratio": 16 / weighted_bits
        }
    
    def to_dict(self):
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "model_type": self.model_type,
            "default_bits": self.default_bits,
            "precision_map": self.precision_map,
            "memory_reduction": self.get_memory_reduction()
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            MixedPrecisionConfig instance
        """
        config = cls(
            model_type=config_dict.get("model_type", "transformer"),
            default_bits=config_dict.get("default_bits", 2)
        )
        
        # Override precision map if provided
        if "precision_map" in config_dict:
            config.precision_map = config_dict["precision_map"]
            
        return config


def optimize_mixed_precision_for_model(
    model, 
    model_type="transformer", 
    target_memory_mb=None,
    browser_capabilities=None,
    accuracy_target=None
):
    """
    Create optimized mixed precision configuration for a model.
    
    Args:
        model: Model to optimize
        model_type: Type of model
        target_memory_mb: Target memory usage in MB, or None for automatic
        browser_capabilities: Dictionary of browser capabilities
        accuracy_target: Target accuracy (percentage as float), None for auto
        
    Returns:
        Optimized MixedPrecisionConfig
    """
    # Create base configuration
    config = MixedPrecisionConfig(model_type=model_type)
    
    # If target memory specified, optimize for it
    if target_memory_mb is not None:
        config.precision_map = config.optimize_memory_usage(target_memory_mb)
    
    # Apply browser-specific optimizations
    if browser_capabilities is not None:
        config = _apply_browser_optimizations(config, browser_capabilities)
    
    # Balance precision for accuracy if target specified
    if accuracy_target is not None:
        config = _balance_precision_for_accuracy(config, model, accuracy_target)
    
    return config

def _apply_browser_optimizations(config, browser_capabilities):
    """
    Apply browser-specific optimizations to precision config.
    
    Args:
        config: MixedPrecisionConfig to optimize
        browser_capabilities: Dictionary of browser capabilities
        
    Returns:
        Optimized MixedPrecisionConfig
    """
    # Get browser name and version
    browser_name = browser_capabilities.get("browser_name", "").lower()
    browser_version = browser_capabilities.get("browser_version", 0)
    
    # Apply browser-specific adjustments
    if browser_name == "safari":
        # Safari has better performance with 3-bit minimum precision
        for layer, bits in config.precision_map.items():
            if bits < 3:
                config.precision_map[layer] = 3
    
    elif browser_name == "firefox" and browser_capabilities.get("compute_shaders_supported", False):
        # Firefox has optimized compute shaders for audio processing
        if config.model_type == "audio":
            # Can use lower precision for some layers due to optimized shaders
            audio_layers = [l for l in config.precision_map if "feature_extractor" in l or "conv" in l]
            for layer in audio_layers:
                config.precision_map[layer] = max(2, config.precision_map[layer] - 1)
    
    # Check for specific hardware capabilities
    if browser_capabilities.get("gpu_memory_gb", 0) < 2:
        # Low GPU memory - further optimize
        config.default_bits = min(config.default_bits, 2)
        for layer, bits in config.precision_map.items():
            if "feed_forward" in layer or "intermediate" in layer:
                config.precision_map[layer] = 2
    
    return config

def _balance_precision_for_accuracy(config, model, accuracy_target):
    """
    Balance precision configuration to meet accuracy target.
    
    Args:
        config: MixedPrecisionConfig to optimize
        model: Model to optimize for
        accuracy_target: Target accuracy percentage
        
    Returns:
        Optimized MixedPrecisionConfig
    """
    # Simple heuristic based on accuracy target
    if accuracy_target > 95:
        # High accuracy requirement - increase precision for critical layers
        for layer in config.critical_layers:
            if config.critical_layers[layer] >= 7:
                config.precision_map[layer] = max(config.precision_map[layer], 4)
    elif accuracy_target < 90:
        # Lower accuracy requirement - can reduce precision
        for layer in config.critical_layers:
            if config.critical_layers[layer] <= 5:
                config.precision_map[layer] = min(config.precision_map[layer], 2)
    
    return config


if __name__ == "__main__":
    print("Ultra-Low Precision WebGPU Quantization Module")
    
    # Example model (dictionary for demonstration)
    example_model = {
        "layer1": {"weight": np.random.randn(128, 128).astype(np.float32)},
        "layer2": {"weight": np.random.randn(128, 256).astype(np.float32)},
        "attention.query": {"weight": np.random.randn(128, 128).astype(np.float32)},
        "attention.key": {"weight": np.random.randn(128, 128).astype(np.float32)},
        "attention.value": {"weight": np.random.randn(128, 128).astype(np.float32)},
        "lm_head": {"weight": np.random.randn(256, 512).astype(np.float32)},
    }
    
    # Example 1: Setup 2-bit quantization
    config = setup_ultra_low_precision(example_model, bits=2, adaptive=True)
    print(f"Configuration: {config}")
    
    # Example 2: Create 2-bit compute shaders
    shaders = create_2bit_compute_shaders()
    print(f"Created {len(shaders)} specialized 2-bit compute shaders")
    
    # Example 3: Quantize with mixed precision
    precision_config = {
        "embedding": 8,
        "attention.query": 3,
        "attention.key": 3,
        "attention.value": 3,
        "feed_forward": 2,
        "layer_norm": 8,
        "lm_head": 4
    }
    
    result = quantize_model_mixed_precision(example_model, precision_config)
    print(f"Mixed precision quantization complete. Memory reduction: {result['stats']['memory_reduction']:.2f}%")
    print(f"Bit distribution: {result['stats']['bit_distribution']}")
    
    # Example 4: Use MixedPrecisionConfig
    mixed_config = MixedPrecisionConfig(model_type="transformer", default_bits=2)
    print(f"\nMixed Precision Configuration for transformer:")
    print(f"Memory reduction: {mixed_config.get_memory_reduction()}")
    print(f"Precision map: {mixed_config.precision_map}")
    
    # Example 5: Optimize for different model types
    vision_config = MixedPrecisionConfig(model_type="vision", default_bits=3)
    audio_config = MixedPrecisionConfig(model_type="audio", default_bits=2)
    multimodal_config = MixedPrecisionConfig(model_type="multimodal", default_bits=2)
    
    print("\nMemory reduction by model type:")
    for model_type, config in [
        ("Transformer", mixed_config),
        ("Vision", vision_config),
        ("Audio", audio_config),
        ("Multimodal", multimodal_config)
    ]:
        reduction = config.get_memory_reduction()
        print(f"{model_type}: {reduction['memory_reduction_percent']:.1f}% reduction, " 
              f"avg bits: {reduction['average_bits']:.1f}")
        
    # Example 6: Memory-constrained optimization
    print("\nMemory-constrained optimization:")
    for memory_mb in [1000, 500, 250, 100]:
        optimized_config = MixedPrecisionConfig(model_type="transformer")
        optimized_config.precision_map = optimized_config.optimize_memory_usage(memory_mb)
        reduction = optimized_config.get_memory_reduction()
        print(f"{memory_mb}MB target: {reduction['memory_reduction_percent']:.1f}% reduction, "
              f"avg bits: {reduction['average_bits']:.1f}")
