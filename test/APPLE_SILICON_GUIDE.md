# Apple Silicon (MPS) Acceleration Guide

## Overview

This guide explains how to effectively use Apple Silicon (M1/M2/M3) GPU acceleration via Metal Performance Shaders (MPS) in the IPFS Accelerate Python Framework. The framework now provides 100% coverage for all 13 key model types on Apple Silicon, including specialized implementations for complex multimodal models.

## Hardware Support

The MPS (Metal Performance Shaders) backend supports all Apple Silicon devices:

- Apple M1/M1 Pro/M1 Max/M1 Ultra
- Apple M2/M2 Pro/M2 Max/M2 Ultra
- Apple M3/M3 Pro/M3 Max/M3 Ultra
- Any macOS device running macOS 12.3 or later with PyTorch MPS support

## Implementation Status

As of March 2025, all 13 key model categories have complete MPS support:

| Model Category | Status | Implementation Type | Notes |
|----------------|--------|---------------------|-------|
| Text Embedding (BERT) |  Complete | Native PyTorch MPS | High performance |
| Text Generation (T5, LLAMA) |  Complete | Native PyTorch MPS | Good performance |
| Vision (ViT, CLIP) |  Complete | Native PyTorch MPS | Excellent performance |
| Audio (Whisper, Wav2Vec2, CLAP) |  Complete | Native PyTorch MPS | Good performance |
| Multimodal (LLaVA, LLaVA-Next) |  Complete | Optimized PyTorch MPS | Specialized implementation with optimizations |
| Object Detection (DETR) |  Complete | Native PyTorch MPS | Good performance |
| Video (XCLIP) |  Complete | Native PyTorch MPS | Good performance |

## LLaVA and LLaVA-Next Optimizations

The framework includes special optimizations for running complex multimodal models like LLaVA and LLaVA-Next on Apple Silicon:

1. **Half-Precision Support**: Uses `torch.float16` to reduce memory usage while maintaining accuracy
2. **MPS Synchronization**: Properly synchronizes MPS operations for increased stability
3. **Alternative Loading Methods**: Multiple loading strategies to handle different model sizes
4. **Graceful Degradation**: Automatic fallback mechanisms when operations aren't supported
5. **Error Handling**: Robust error recovery with CPU fallbacks when needed

## Usage

### Basic Usage

```python
# Use MPS for any model by specifying the platform
python scripts/generators/benchmark_scripts/generators/run_model_benchmarks.py --model bert --hardware mps

# Test LLaVA models on MPS
python scripts/generators/benchmark_scripts/generators/run_model_benchmarks.py --model llava --hardware mps

# Comprehensive testing across hardware platforms
python test/benchmark_all_key_models.py --model llava --hardware cpu,mps,cuda
```

### Specific Options for Multimodal Models

For LLaVA and LLaVA-Next models, you can use specialized options:

```python
# Set precision for MPS (default is float16)
python scripts/generators/benchmark_scripts/generators/run_model_benchmarks.py --model llava --hardware mps --precision float16

# Configure memory-efficient loading
python scripts/generators/benchmark_scripts/generators/run_model_benchmarks.py --model llava --hardware mps --memory-efficient

# Force alternative loading method (CPU first, then transfer)
python scripts/generators/benchmark_scripts/generators/run_model_benchmarks.py --model llava --hardware mps --alternative-loading
```

## Performance Guidelines

Apple Silicon performance varies by model type:

1. **Vision Models**: 3-5x faster than CPU for ViT, CLIP, etc.
2. **Text Models**: 2-3x faster than CPU for BERT, T5, etc.
3. **Audio Models**: 4-7x faster than CPU for Whisper, Wav2Vec2, etc.
4. **Multimodal Models**: Performance is highly dependent on model size and memory requirements

## Troubleshooting

### Common Issues

1. **Out of Memory**: Try reducing batch size or using a smaller model variant
   ```python
   python scripts/generators/benchmark_scripts/generators/run_model_benchmarks.py --model llava --hardware mps --batch-size 1 --small-model
   ```

2. **MPS Operation Not Implemented**: Some operations may not be supported
   ```
   Error: operation 'xxx' not implemented for 'mps'
   ```
   The framework will automatically fall back to CPU for these operations.

3. **Model too large for memory**: Try using memory-efficient loading
   ```python
   python scripts/generators/benchmark_scripts/generators/run_model_benchmarks.py --model llava --hardware mps --memory-efficient
   ```

### Memory Optimization

For large models on systems with limited memory:

1. Use half-precision (`torch.float16`) 
2. Enable memory-efficient loading
3. Consider quantized models when available
4. Limit batch size to 1 for very large models

## Implementation Details

The MPS support is implemented in several key components:

1. **Template Files**: See `hardware_test_templates/template_llava.py` and others
2. **Hardware Detection**: MPS detection in `scripts/generators/hardware/hardware_detection.py`
3. **Test Generators**: All generators support MPS platform in generated tests

## Further Reading

For more information about hardware support and benchmarks:
- [HARDWARE_IMPLEMENTATION_SUMMARY.md](HARDWARE_IMPLEMENTATION_SUMMARY.md)
- [CROSS_PLATFORM_TEST_COVERAGE.md](CROSS_PLATFORM_TEST_COVERAGE.md)
- [HARDWARE_BENCHMARKING_GUIDE.md](HARDWARE_BENCHMARKING_GUIDE.md)