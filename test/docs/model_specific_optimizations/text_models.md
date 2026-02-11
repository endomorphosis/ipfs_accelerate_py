# Text Model Optimization Guide

**Last Updated:** March 6, 2025

This guide provides optimization recommendations for text models (BERT, T5, LLaMA, GPT, etc.) across different hardware platforms.

## Overview

Text models, including embeddings and generative models, form the foundation of many NLP tasks. The framework supports a wide range of text models from simple embedding models like BERT to large generative models like LLaMA.

## Hardware Compatibility

| Model Size | CUDA | ROCm | MPS | OpenVINO | Qualcomm | WebNN | WebGPU |
|------------|------|------|-----|----------|----------|-------|--------|
| Tiny (<100M) | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High |
| Small (100M-1B) | ✅ High | ✅ High | ✅ High | ✅ High | ✅ Medium | ✅ High | ✅ High |
| Medium (1B-7B) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| Large (>7B) | ✅ High | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ❌ None | ❌ None | ❌ None |

## Performance Optimization Techniques

### CUDA Optimization

CUDA provides the best overall performance for text models, especially for larger models.

#### Recommended Configurations

```python
# Embedding models (BERT, etc.)
platform = UnifiedWebPlatform(
    model_name="bert-base-uncased",
    model_type="text",
    platform="cuda",
    config={
        "precision": "fp16",
        "batch_size": 32,
        "kv_cache_enabled": False,  # Not needed for embedding models
        "cuda_optimization_level": "highest"
    }
)

# Generative models (LLaMA, GPT, etc.)
platform = UnifiedWebPlatform(
    model_name="llama-7b",
    model_type="text",
    platform="cuda",
    config={
        "precision": "int8",  # or fp16 for higher accuracy
        "batch_size": 4,      # Adjust based on GPU memory
        "kv_cache_enabled": True,
        "kv_cache_max_length": 2048,
        "cuda_optimization_level": "highest",
        "tensor_parallelism": 1,  # Increase for multi-GPU setups
    }
)
```

#### Memory Optimization

For large models on CUDA:

- Use tensor parallelism for models >13B parameters
- Enable gradient checkpointing for training
- Use flash attention for improved performance and reduced memory usage
- Use int8 or int4 quantization for inference when appropriate

### ROCm (AMD) Optimization

ROCm provides good performance for small to medium-sized text models.

#### Recommended Configurations

```python
# Embedding models on ROCm
platform = UnifiedWebPlatform(
    model_name="bert-base-uncased", 
    model_type="text",
    platform="rocm",
    config={
        "precision": "fp16",
        "batch_size": 24,  # Slightly smaller than CUDA
        "rocm_optimization_level": "high"
    }
)

# Generative models on ROCm
platform = UnifiedWebPlatform(
    model_name="llama-7b",
    model_type="text",
    platform="rocm",
    config={
        "precision": "int8",  # Lower precision for larger models
        "batch_size": 2,      # Start small and test
        "kv_cache_enabled": True,
        "kv_cache_max_length": 1024,  # Smaller context windows
        "rocm_optimization_level": "medium"
    }
)
```

#### ROCm-Specific Considerations

- ROCm performance is more sensitive to batch sizes; start with smaller batches
- For models >7B, use model sharding or quantization
- Enable HIP graph capture for repetitive inference patterns
- Consider using a smaller context window for improved throughput

### MPS (Apple Silicon) Optimization

MPS provides excellent performance and efficiency for small to medium text models on Apple devices.

#### Recommended Configurations

```python
# Embedding models on MPS
platform = UnifiedWebPlatform(
    model_name="bert-base-uncased",
    model_type="text",
    platform="mps",
    config={
        "precision": "fp16",
        "batch_size": 16,
        "use_mps_graph": True
    }
)

# Generative models on MPS
platform = UnifiedWebPlatform(
    model_name="llama-7b",
    model_type="text",
    platform="mps",
    config={
        "precision": "int8",
        "batch_size": 1,  # MPS works best with smaller batch sizes
        "kv_cache_enabled": True,
        "kv_cache_max_length": 512,
        "use_mps_graph": True,
        "power_efficient": True  # For battery life on laptops
    }
)
```

#### MPS-Specific Considerations

- Use MPS Graph mode for significant performance improvements
- For M1/M2 MacBooks, enable power efficient mode for better battery life
- Use int8 quantization for models >2B parameters
- For M1 Pro/Max/Ultra, up to 7B models can run efficiently
- For M3 chips, models up to 13B can be run with quantization

### Qualcomm AI Engine Optimization

Qualcomm hardware excels at power-efficient inference for mobile and edge devices.

#### Recommended Configurations

```python
# Embedding models on Qualcomm
platform = UnifiedWebPlatform(
    model_name="bert-tiny",  # Smaller models preferred
    model_type="text",
    platform="qualcomm",
    config={
        "precision": "int8",
        "batch_size": 1,
        "power_mode": "efficient",
        "hexagon_enabled": True  # Use Hexagon DSP
    }
)

# Small generative models on Qualcomm
platform = UnifiedWebPlatform(
    model_name="opt-125m",  # Small models only
    model_type="text",
    platform="qualcomm",
    config={
        "precision": "int4",  # Ultra-low precision
        "batch_size": 1,
        "kv_cache_enabled": True,
        "kv_cache_max_length": 256,  # Smaller context
        "power_mode": "efficient",
        "hexagon_enabled": True,
        "sparse_inference": True  # Optimized for sparsity
    }
)
```

#### Qualcomm-Specific Considerations

- Use weight clustering quantization for optimal performance (see [Advanced Quantization Guide](../QUALCOMM_IMPLEMENTATION_GUIDE.md))
- Use smaller variants for generative models (<1B parameters)
- Enable Hexagon DSP for maximum performance
- Set power_mode to "efficient" for mobile devices or "performance" for edge devices
- Consider layer pruning for large models

### WebNN Optimization

WebNN provides CPU-accelerated inference in browser environments where WebGPU isn't available.

#### Recommended Configurations

```python
# Embedding models on WebNN
platform = UnifiedWebPlatform(
    model_name="bert-tiny",
    model_type="text",
    platform="webnn",
    config={
        "precision": "fp16",
        "batch_size": 4,
        "use_wasm_threads": True
    }
)

# Generative models on WebNN
platform = UnifiedWebPlatform(
    model_name="opt-125m",  # Small models only
    model_type="text",
    platform="webnn",
    config={
        "precision": "int8",
        "batch_size": 1,
        "kv_cache_enabled": True,
        "kv_cache_max_length": 256,
        "use_wasm_threads": True,
        "use_simd": True
    }
)
```

#### WebNN-Specific Considerations

- WebNN performance varies significantly by browser and device
- Enable WASM threads and SIMD for multi-core performance
- Use small model variants (distilled or quantized)
- Consider using adaptive generation to reduce perceived latency
- Implement progressive loading for improved user experience

### WebGPU Optimization

WebGPU provides GPU-accelerated inference in modern browsers, with performance approaching native hardware for smaller models.

#### Recommended Configurations

```python
# Embedding models on WebGPU
platform = UnifiedWebPlatform(
    model_name="bert-base-uncased",
    model_type="text",
    platform="webgpu",
    config={
        "precision": "fp16",
        "batch_size": 8,
        "shader_precompile": True,
        "optimize_for_browser": "auto"  # Automatically detect and optimize
    }
)

# Generative models on WebGPU
platform = UnifiedWebPlatform(
    model_name="llama-2-7b",  # Up to 7B with quantization
    model_type="text",
    platform="webgpu",
    config={
        "precision": "int4",
        "batch_size": 1,
        "kv_cache_enabled": True,
        "kv_cache_max_length": 512,
        "shader_precompile": True,
        "optimize_for_browser": "auto",
        "compute_shader_optimizations": True
    }
)

# WebGPU streaming for text generation
from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference

streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "quantization": "int4",
        "kv_cache_optimization": True,
        "optimize_compute_transfer": True,
        "optimize_for_latency": True,
        "browser_optimizations": True
    }
)
```

#### WebGPU-Specific Considerations

- Enable shader precompilation for faster first inference
- Use int4 quantization for models >1B parameters
- Implement compute/transfer overlap for streaming generation
- For Firefox, use custom workgroup sizes (256x1x1) for optimal performance
- Check memory usage and implement fallback mechanisms for large models

## Model-Specific Optimization

### BERT and Embedding Models

Embedding models like BERT benefit from:

1. Higher batch sizes for throughput
2. FP16 precision for most platforms
3. Caching results for repeated inputs
4. Removing unused layers (pooler, etc.) for pure embedding use cases

### T5 and Encoder-Decoder Models

Encoder-decoder models like T5 benefit from:

1. Separate encoder-decoder configurations
2. Caching encoder outputs for repetitive tasks
3. Beam search optimization for generation tasks
4. Weight sharing where possible

### LLaMA and Large Language Models

Large language models benefit from:

1. KV-cache optimization
2. Low-precision inference (int8, int4)
3. Flash attention implementation
4. Streaming token generation
5. Context compression for longer contexts

## Memory-Constrained Environments

For memory-constrained environments:

1. Use quantization (int8/int4)
2. Implement sliding window attention
3. Consider use of distilled models
4. Reduce context length
5. Disable model components not needed for the task
6. Implement attention pruning or sparse attention

## Mobile and Edge Optimization

For deployment on mobile or edge devices:

1. Use Qualcomm AI Engine for Android devices
2. Use CoreML/MPS for iOS devices
3. Implement weight pruning and quantization
4. Use small model variants (<500M parameters)
5. Cache common operations and results
6. Implement adaptive power management
7. Use progressive loading with user feedback

## Cross-Platform Considerations

When deploying across multiple platforms:

1. Implement platform detection and automatic configuration
2. Create fallback strategies for unsupported features
3. Test on representative devices for each platform
4. Consider maintaining separate model variants optimized for each platform
5. Implement adaptive generation parameters based on hardware capabilities

## Streaming Text Generation

For optimal streaming text generation:

1. Use WebGPU streaming inference with compute/transfer overlap
2. Implement token prefetching for reduced perceived latency
3. Use browser-specific optimizations (especially Firefox for audio models)
4. Implement adaptive batch sizing for consistent performance
5. Use error recovery mechanisms for browser interruptions

## Related Documentation

- [WebGPU Streaming Documentation](../WEBGPU_STREAMING_DOCUMENTATION.md)
- [WebGPUStreamingInference API Reference](../api_reference/webgpu_streaming_inference.md)
- [Qualcomm Implementation Guide](../QUALCOMM_IMPLEMENTATION_GUIDE.md)
- [Hardware Selection Guide](../HARDWARE_SELECTION_GUIDE.md)