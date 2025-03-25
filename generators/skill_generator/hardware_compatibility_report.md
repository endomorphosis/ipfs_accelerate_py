# Hardware Compatibility Report

Generated on: 2025-03-24 15:54:48

**Version:** 1.0.0
**Last Updated:** 2025-03-24
**Description:** Compatibility matrix between model architectures and hardware backends

## Compatibility Overview

| Architecture | cpu | cuda | rocm | mps | openvino | qnn |
| --- | --- | --- | --- | --- | --- | --- |
| **encoder-only** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | ✅ Good | ✓ Compatible | 
| **decoder-only** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✓ Moderate | ✓ Moderate | ❌ No | 
| **encoder-decoder** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | ✅ Good | ❌ No | 
| **vision** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | 
| **vision-encoder-text-decoder** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | ✅ Good | ✓ Moderate | 
| **speech** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | ✅ **Excellent** | ✅ Good | 
| **multimodal** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✓ Moderate | ✓ Moderate | ❌ No | 
| **diffusion** | ⚠️ Poor | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | ✓ Moderate | ❌ No | 
| **mixture-of-experts** | ⚠️ Poor | ✅ **Excellent** | ✅ **Excellent** | ❌ No | ❌ No | ❌ No | 
| **state-space** | ✓ Moderate | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | ❌ No | ❌ No | 
| **rag** | ✓ Baseline | ✅ **Excellent** | ✅ **Excellent** | ✅ Good | ❌ No | ❌ No | 


## Detailed Architecture Compatibility

### Encoder-only Architecture

**Example Models:** `roberta-base`, `mobilebert-uncased`, `distilbert-base-uncased`, `bert-base-uncased`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | good | Well supported on Apple Silicon | half_precision, unified_memory |
| **OPENVINO** | ✅ Yes | good | Well optimized for Intel hardware | int8_quantization, model_optimization |
| **QNN** | ✅ Yes | varies | Support varies by model size and complexity | fixed_shapes, quantization |

### Decoder-only Architecture

**Example Models:** `phi-1.5`, `gpt2`, `llama-7b`, `opt-350m`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn, flash_attention |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | moderate | Works well for smaller models, memory limited for larger ones | half_precision, unified_memory |
| **OPENVINO** | ✅ Yes | moderate | Works best with smaller to medium sized models | int8_quantization, model_optimization |
| **QNN** | ❌ No | n/a | Most decoder-only models exceed current QNN capabilities |  |

### Encoder-decoder Architecture

**Example Models:** `flan-t5-base`, `t5-small`, `bart-base`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | good | Well supported on Apple Silicon for small to medium models | half_precision, unified_memory |
| **OPENVINO** | ✅ Yes | good | Well optimized for Intel hardware | int8_quantization, model_optimization |
| **QNN** | ❌ No | n/a | Complex architecture not well-suited for current QNN capabilities |  |

### Vision Architecture

**Example Models:** `facebook/deit-base-patch16-224`, `google/vit-base-patch16-224`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | excellent | Very well optimized on Apple Silicon | half_precision, unified_memory, metal_performance_shaders |
| **OPENVINO** | ✅ Yes | excellent | Vision models are highly optimized for Intel hardware | int8_quantization, model_optimization, winograd_convolution |
| **QNN** | ✅ Yes | good | Vision models are well-suited for QNN hardware | fixed_shapes, quantization, npu_kernels |

### Vision-encoder-text-decoder Architecture

**Example Models:** `nlpconnect/vit-gpt2-image-captioning`, `openai/clip-vit-base-patch32`, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | good | Well supported on Apple Silicon | half_precision, unified_memory, metal_performance_shaders |
| **OPENVINO** | ✅ Yes | good | Well optimized for Intel hardware | int8_quantization, model_optimization |
| **QNN** | ✅ Yes | moderate | Vision encoder works well, text decoder may be limited | fixed_shapes, quantization |

### Speech Architecture

**Example Models:** `facebook/wav2vec2-base-960h`, `openai/whisper-tiny`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | good | Well supported on Apple Silicon | half_precision, unified_memory |
| **OPENVINO** | ✅ Yes | excellent | Speech models are highly optimized for Intel hardware | int8_quantization, model_optimization, specialized_kernels |
| **QNN** | ✅ Yes | good | Speech models benefit from QNN's specialized audio processing | fixed_shapes, quantization, audio_acceleration |

### Multimodal Architecture

**Example Models:** `llava-hf/llava-1.5-7b-hf`, `facebook/flava-full`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn, flash_attention |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | moderate | Works with smaller models, may be memory-limited for larger ones | half_precision, unified_memory |
| **OPENVINO** | ✅ Yes | moderate | Variable performance based on model complexity | int8_quantization, model_optimization |
| **QNN** | ❌ No | n/a | Most multimodal models exceed current QNN capabilities |  |

### Diffusion Architecture

**Example Models:** `stabilityai/stable-diffusion-2-1-base`, `runwayml/stable-diffusion-v1-5`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | poor | Very slow but functional | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with highly optimized kernels | half_precision, tensor_cores, cudnn, memory_efficient_attention |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | good | Good performance on Apple Silicon with optimizations | half_precision, unified_memory, mps_optimized_kernels |
| **OPENVINO** | ✅ Yes | moderate | Works but with reduced performance compared to CUDA/ROCm | int8_quantization, model_optimization |
| **QNN** | ❌ No | n/a | Diffusion models exceed current QNN capabilities |  |

### Mixture-of-experts Architecture

**Example Models:** `google/switch-base-8`, `mistralai/Mixtral-8x7B-v0.1`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | poor | Very slow but functional | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels for sparse computation | half_precision, tensor_cores, expert_parallelism, flash_attention |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization, expert_parallelism |
| **MPS** | ❌ No | n/a | MoE models typically exceed MPS memory limits |  |
| **OPENVINO** | ❌ No | n/a | MoE models not well supported on current OpenVINO |  |
| **QNN** | ❌ No | n/a | MoE models exceed current QNN capabilities |  |

### State-space Architecture

**Example Models:** `BlinkDL/RWKV-4-Pile-430M`, `state-spaces/mamba-2.8b`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | moderate | State-space models can be efficient even on CPU | quantization, threading, recurrent_optimizations |
| **CUDA** | ✅ Yes | excellent | Full support with specialized kernels for state-space operations | half_precision, tensor_cores, selective_scan_kernels |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | good | Well supported on Apple Silicon | half_precision, unified_memory |
| **OPENVINO** | ❌ No | n/a | State-space models not well supported on current OpenVINO |  |
| **QNN** | ❌ No | n/a | State-space models not well supported on current QNN |  |

### Rag Architecture

**Example Models:** `facebook/rag-sequence-nq`, `facebook/rag-token-nq`

| Hardware | Compatibility | Performance | Notes | Optimizations |
| --- | --- | --- | --- | --- |
| **CPU** | ✅ Yes | baseline | Always supported, serves as fallback | quantization, threading |
| **CUDA** | ✅ Yes | excellent | Full support with optimized kernels | half_precision, tensor_cores, cudnn |
| **ROCM** | ✅ Yes | excellent | Full support via HIP or CUDA compatibility layer | half_precision, memory_optimization |
| **MPS** | ✅ Yes | good | Well supported on Apple Silicon | half_precision, unified_memory |
| **OPENVINO** | ❌ No | n/a | RAG models not well supported on current OpenVINO |  |
| **QNN** | ❌ No | n/a | RAG models not well supported on current QNN |  |

## Hardware Backend Details

### CPU

CPU is the default fallback for all models. It offers universal compatibility but typically with baseline performance.

**Key Benefits:**
- Universal compatibility
- No special hardware required
- Predictable behavior

**Limitations:**
- Slower inference compared to accelerated hardware
- Memory constraints for larger models

### CUDA (NVIDIA GPUs)

CUDA provides excellent performance for all model types on NVIDIA GPUs.

**Key Benefits:**
- Excellent performance across all model types
- Mature ecosystem with optimized kernels
- Wide range of supporting libraries

**Limitations:**
- Requires NVIDIA GPU hardware
- Memory constraints for very large models

### ROCm (AMD GPUs)

ROCm provides excellent performance for all model types on AMD GPUs through the HIP API or CUDA compatibility layer.

**Key Benefits:**
- Excellent performance on AMD GPU hardware
- CUDA compatibility layer for broad support
- Growing ecosystem of optimized kernels

**Limitations:**
- Requires AMD GPU hardware
- Some advanced optimizations may lag behind CUDA

### MPS (Apple Silicon)

Apple's Metal Performance Shaders (MPS) provides good to excellent performance on Apple Silicon hardware.

**Key Benefits:**
- Excellent performance on Apple Silicon
- Unified memory architecture
- Power-efficient inference

**Limitations:**
- Limited to Apple hardware
- Memory constraints for large models
- Limited support for very large models

### OpenVINO (Intel)

Intel's OpenVINO provides optimized inference on Intel CPUs, GPUs, and specialized hardware.

**Key Benefits:**
- Optimized for Intel hardware
- INT8 quantization support
- Model optimization capabilities

**Limitations:**
- Limited support for specialized architectures
- Best performance on Intel hardware

### QNN (Qualcomm)

Qualcomm Neural Network provides acceleration on Qualcomm NPUs, primarily for vision and simpler text models.

**Key Benefits:**
- Power-efficient inference on mobile
- Specialized for edge devices
- Good performance for vision models

**Limitations:**
- Limited support for large and complex models
- Requires fixed input shapes
- Limited availability outside Qualcomm hardware

