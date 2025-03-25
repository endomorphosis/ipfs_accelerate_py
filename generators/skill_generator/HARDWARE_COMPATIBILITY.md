# Hardware Compatibility Matrix

This document details the compatibility between model architectures and hardware backends in the IPFS Accelerate Python Framework.

## Overview

The IPFS Accelerate Python Framework supports multiple hardware backends, each with different capabilities and limitations. This document provides information about which model architectures are compatible with which hardware backends, along with notes on performance, memory requirements, and special considerations.

## Hardware Backends

The framework supports the following hardware backends:

1. **CPU**: Available on all systems
   - Pros: Universal compatibility, stable
   - Cons: Slowest performance, high memory usage for large models

2. **CUDA**: NVIDIA GPU acceleration via PyTorch
   - Pros: High performance, optimized for deep learning
   - Cons: Requires NVIDIA GPU, high power consumption

3. **ROCm**: AMD GPU acceleration via PyTorch+ROCm
   - Pros: Excellent performance on AMD hardware, comprehensive model support
   - Cons: Slightly fewer optimizations than CUDA, requires newer AMD GPUs

4. **MPS**: Apple Silicon acceleration via Metal Performance Shaders
   - Pros: Good performance on Apple M-series chips, energy efficient
   - Cons: Memory limitations, some operations not supported

5. **OpenVINO**: Intel acceleration via OpenVINO framework
   - Pros: Optimized for Intel CPUs, GPUs, and VPUs
   - Cons: Requires model conversion, not all models supported

6. **QNN**: Qualcomm Neural Network acceleration
   - Pros: Optimized for Qualcomm Snapdragon processors
   - Cons: Limited model support, requires specialized knowledge

## Compatibility Matrix

| Architecture Type | CPU | CUDA | ROCm | MPS | OpenVINO | QNN |
|-------------------|-----|------|------|-----|----------|-----|
| encoder-only      | ✅  | ✅   | ✅   | ✅  | ✅       | ✅  |
| decoder-only      | ✅  | ✅   | ✅   | ✅  | ✅       | ❌  |
| encoder-decoder   | ✅  | ✅   | ✅   | ✅  | ✅       | ❌  |
| vision            | ✅  | ✅   | ✅   | ✅  | ✅       | ✅  |
| vision-encoder-text-decoder | ✅ | ✅ | ✅ | ✅ | ✅     | ✅  |
| speech            | ✅  | ✅   | ✅   | ✅  | ✅       | ✅  |
| multimodal        | ✅  | ✅   | ✅   | ✅  | ✅       | ❌  |
| diffusion         | ✅  | ✅   | ✅   | ✅  | ✅       | ❌  |
| mixture-of-experts| ✅  | ✅   | ✅   | ❌  | ❌       | ❌  |
| state-space       | ✅  | ✅   | ✅   | ✅  | ❌       | ❌  |
| rag               | ✅  | ✅   | ✅   | ✅  | ❌       | ❌  |

## Detailed Compatibility Notes

### Encoder-Only Models (BERT, RoBERTa, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Medium      | Medium       | Works well with batch size 1-8 |
| CUDA             | ✅ Full      | High        | Low          | Optimal batch size 16-64 |
| ROCm             | ✅ Full      | High        | Low          | Excellent performance, all models supported |
| MPS              | ✅ Full      | Medium-High | Low          | Good on Apple M1/M2 chips |
| OpenVINO         | ✅ Full      | Medium-High | Low          | Good with INT8 quantization |
| QNN              | ✅ Full      | Medium      | Low          | Good with 8-bit quantization |

### Decoder-Only Models (GPT-2, LLaMA, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Low         | High         | Very slow for large models |
| CUDA             | ✅ Full      | High        | Medium-High  | Benefits from Tensor Cores |
| ROCm             | ✅ Full      | High        | Medium-High  | Comparable to CUDA, half-precision support |
| MPS              | ✅ Full      | Medium      | High         | Memory limited on smaller devices |
| OpenVINO         | ✅ Full      | Medium      | Medium       | Good for smaller models (<7B) |
| QNN              | ❌ Limited   | N/A         | N/A          | Limited support for large LLMs |

### Vision Models (ViT, ResNet, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Low-Medium  | Medium       | Works well for inference |
| CUDA             | ✅ Full      | High        | Low          | Excellent for batch processing |
| ROCm             | ✅ Full      | High        | Low          | Excellent performance, optimized for vision tasks |
| MPS              | ✅ Full      | Medium-High | Low          | Very good on Apple Silicon |
| OpenVINO         | ✅ Full      | High        | Low          | Excellent with INT8 quantization |
| QNN              | ✅ Full      | Medium-High | Low          | Good with optimized models |

### Vision-Encoder-Text-Decoder Models (CLIP, BLIP, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Low-Medium  | Medium       | Text-image similarity works well |
| CUDA             | ✅ Full      | High        | Medium       | Excellent performance |
| ROCm             | ✅ Full      | High        | Medium       | Full model support, great for CLIP/BLIP |
| MPS              | ✅ Full      | Medium      | Medium       | Good on Apple Silicon |
| OpenVINO         | ✅ Full      | Medium-High | Low-Medium   | Good with optimizations |
| QNN              | ✅ Full      | Medium      | Low-Medium   | Works well with CLIP |

### Speech Models (Whisper, Wav2Vec2, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Low         | Medium       | Usable for short audio |
| CUDA             | ✅ Full      | High        | Medium       | Real-time transcription possible |
| ROCm             | ✅ Full      | Medium-High | Medium       | Good performance |
| MPS              | ✅ Full      | Medium      | Medium       | Decent on Apple Silicon |
| OpenVINO         | ✅ Full      | Medium      | Low-Medium   | Good with optimizations |
| QNN              | ✅ Full      | Medium      | Low-Medium   | Works well with Whisper-tiny/small |

### Multimodal Models (FLAVA, LLaVA, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Low         | High         | Very slow |
| CUDA             | ✅ Full      | Medium-High | High         | Works well with 16+ GB VRAM |
| ROCm             | ✅ Full      | Medium      | High         | Similar to CUDA |
| MPS              | ✅ Limited   | Low-Medium  | High         | Memory limited on Apple Silicon |
| OpenVINO         | ✅ Limited   | Low-Medium  | Medium-High  | Works for some models |
| QNN              | ❌ Not supported | N/A     | N/A          | Too complex for current QNN |

### Mixture-of-Experts Models (Mixtral, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Very Low    | Very High    | Usable but very slow |
| CUDA             | ✅ Full      | Medium      | High         | Requires 24GB+ VRAM |
| ROCm             | ✅ Limited   | Low-Medium  | High         | Limited optimizations |
| MPS              | ❌ Not supported | N/A     | N/A          | Memory requirements exceed capabilities |
| OpenVINO         | ❌ Not supported | N/A     | N/A          | Not currently supported |
| QNN              | ❌ Not supported | N/A     | N/A          | Not currently supported |

### State-Space Models (Mamba, RWKV, etc.)

| Hardware Backend | Compatibility | Performance | Memory Usage | Notes |
|------------------|---------------|-------------|--------------|-------|
| CPU              | ✅ Full      | Low         | Medium-High  | Slower than transformer models |
| CUDA             | ✅ Full      | High        | Medium       | Good with specialized kernels |
| ROCm             | ✅ Limited   | Medium      | Medium       | Limited kernel support |
| MPS              | ✅ Limited   | Low-Medium  | Medium       | Basic operations only |
| OpenVINO         | ❌ Not supported | N/A     | N/A          | No specialized optimizations |
| QNN              | ❌ Not supported | N/A     | N/A          | Not currently supported |

## Performance Recommendations

For optimal performance with different model types:

### Encoder-Only Models
- Best hardware: CUDA > ROCm > OpenVINO > MPS > QNN > CPU
- Recommendation: These models work well on all hardware. Use CUDA/ROCm when available.

### Decoder-Only Models
- Best hardware: CUDA > ROCm > MPS > OpenVINO > CPU
- Recommendation: For large LLMs (>7B), use CUDA/ROCm. For smaller models, all backends work adequately.

### Vision Models
- Best hardware: CUDA > ROCm > OpenVINO > MPS > QNN > CPU
- Recommendation: Vision models are quite efficient. All backends provide acceptable performance.

### Vision-Text Models
- Best hardware: CUDA > ROCm > OpenVINO > MPS > QNN > CPU
- Recommendation: CLIP models work well on all hardware. Use CUDA/ROCm for BLIP and other more complex models.

### Speech Models
- Best hardware: CUDA > ROCm > OpenVINO > MPS > QNN > CPU
- Recommendation: For real-time transcription with Whisper, use CUDA/ROCm. For offline transcription, all backends work.

### Multimodal Models
- Best hardware: CUDA > ROCm > OpenVINO > MPS > CPU
- Recommendation: These are demanding models. Use the highest performance hardware available.

### Mixture-of-Experts Models
- Best hardware: CUDA > ROCm > CPU
- Recommendation: Use only with high-end NVIDIA GPUs (24GB+ VRAM). CPU is usable but very slow.

### State-Space Models
- Best hardware: CUDA > ROCm > MPS > CPU
- Recommendation: CUDA is significantly better due to specialized kernels. Other backends have limited optimizations.

## Memory Requirements

Estimated VRAM/RAM requirements for different model sizes:

| Model Size | CPU (RAM) | CUDA/ROCm (VRAM) | MPS (Unified) | OpenVINO | QNN |
|------------|-----------|------------------|---------------|----------|-----|
| Tiny (<100M) | 500MB-1GB | 300-700MB | 300-700MB | 200-500MB | 100-300MB |
| Small (100M-500M) | 1-3GB | 0.5-2GB | 0.5-2GB | 0.4-1.5GB | 0.3-1GB |
| Base (500M-1B) | 3-6GB | 2-4GB | 2-4GB | 1.5-3GB | 1-2GB |
| Large (1B-7B) | 6-20GB | 4-16GB | 4-16GB | 3-12GB | Not supported |
| XL (7B-13B) | 20-40GB | 16-30GB | Not supported | Not supported | Not supported |
| XXL (>13B) | 40GB+ | 30GB+ | Not supported | Not supported | Not supported |

Note: These are estimates and actual memory usage depends on:
- Implementation details
- Batch size
- Sequence length
- Use of techniques like quantization and attention optimizations