# WebNN and WebGPU Model Coverage

This document provides a comprehensive overview of WebNN and WebGPU support for HuggingFace model classes in the IPFS Accelerate framework. It includes details on quantization options, browser compatibility, and performance characteristics.

## Table of Contents

- [Introduction](#introduction)
- [Quantization Summary](#quantization-summary)
- [Text Models](#text-models)
- [Text Generation Models](#text-generation-models)
- [Vision Models](#vision-models)
- [Audio Models](#audio-models)
- [Multimodal Models](#multimodal-models)
- [Testing Resources](#testing-resources)

## Introduction

WebNN and WebGPU implementations in our framework support over 300 HuggingFace model classes, providing a comprehensive solution for browser-based inference. This document details our testing results, compatibility matrix, and specific recommendations for each model family.

### Implementation Status Summary

| Technology | Browser Support | Quantization | Best Use Case | Key Advantage |
|------------|----------------|--------------|---------------|---------------|
| WebNN | Chrome, Edge, Safari | 8-bit (standard)<br>4-bit, 2-bit (experimental) | Text and vision models | Native browser APIs, hardware acceleration |
| WebGPU | Chrome, Edge, Firefox, Safari | 2-bit through 16-bit | All model types | Advanced quantization, custom optimizations |

## Quantization Summary

### Memory Reduction

| Precision | Memory Reduction | WebNN Support | WebGPU Support | Typical Accuracy Drop |
|-----------|------------------|---------------|----------------|----------------------|
| FP16 (16-bit) | 0% (baseline) | ✅ All browsers | ✅ All browsers | 0% |
| INT8 (8-bit) | 50% | ✅ All browsers | ✅ All browsers | 0.5-1% |
| INT4 (4-bit) | 75% | ⚠️ Experimental | ✅ All browsers | 1-3% |
| INT3 (3-bit) | 81.25% | ❌ Not supported | ✅ All browsers | 2-4% |
| INT2 (2-bit) | 87.5% | ⚠️ Experimental | ✅ All browsers | 3-8% |

### Mixed Precision

Mixed precision provides better accuracy while maintaining memory savings by using:

- **Higher precision (8-bit)**:
  - Attention layers (query, key, value)
  - Embedding layers
  - Output projection layers
  - Layer normalization

- **Lower precision (4-bit or 2-bit)**:
  - Feed-forward networks
  - Intermediate projections
  - Less sensitive matrices

## Text Models

### BERT Family

| Model | WebNN (8-bit) | WebNN (4-bit) | WebGPU (4-bit) | WebGPU (2-bit) | Notes |
|-------|---------------|---------------|----------------|----------------|-------|
| bert-base-uncased | ✅ Good | ⚠️ Experimental | ✅ Excellent | ✅ Good | Best overall compatibility |
| bert-large-uncased | ✅ Good | ⚠️ Experimental | ✅ Good | ⚠️ Limited | Larger size may impact WebGPU performance |
| bert-base-cased | ✅ Good | ⚠️ Experimental | ✅ Excellent | ✅ Good | Similar to bert-base-uncased |
| prajjwal1/bert-tiny | ✅ Excellent | ✅ Experimental | ✅ Excellent | ✅ Excellent | Small size works well on all platforms |
| distilbert-base-uncased | ✅ Excellent | ⚠️ Experimental | ✅ Excellent | ✅ Good | Distilled models work well |
| albert-base-v2 | ✅ Good | ⚠️ Experimental | ✅ Good | ⚠️ Limited | Parameter sharing benefits quantization |

**Recommendations:**
- For maximum compatibility: Use `bert-base-uncased` with WebNN 8-bit
- For best memory efficiency: Use `prajjwal1/bert-tiny` with WebGPU 4-bit
- For Firefox: Always use WebGPU with 4-bit precision

**Example Command:**
```bash
python implement_comprehensive_webnn_webgpu.py --model bert-base-uncased --platform webgpu --bits 4
```

### T5 Family

| Model | WebNN (8-bit) | WebNN (4-bit) | WebGPU (4-bit) | WebGPU (2-bit) | Notes |
|-------|---------------|---------------|----------------|----------------|-------|
| t5-small | ✅ Good | ⚠️ Experimental | ✅ Excellent | ✅ Good | Good general performance |
| t5-base | ✅ Good | ❌ Too large | ✅ Good | ⚠️ Limited | Larger size impacts WebNN |
| t5-efficient-tiny | ✅ Excellent | ✅ Experimental | ✅ Excellent | ✅ Excellent | Best for memory-constrained |
| google/flan-t5-small | ✅ Good | ⚠️ Experimental | ✅ Good | ⚠️ Limited | Instruction tuning preserved |

**Recommendations:**
- For instruction-tuned models: Use `google/flan-t5-small` with WebGPU 4-bit
- For memory efficiency: Use `t5-efficient-tiny` with WebGPU 4-bit or 2-bit
- Enable mixed precision with `--mixed-precision` for best results with larger variants

### Other Text Models

| Model Family | WebNN | WebGPU | Recommended Setup |
|--------------|-------|--------|-------------------|
| RoBERTa | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| XLM | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| ELECTRA | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| DeBERTa | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| BART | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit, Firefox |

## Text Generation Models

### GPT-2 Family

| Model | WebNN (8-bit) | WebNN (4-bit) | WebGPU (4-bit) | WebGPU (2-bit) | Notes |
|-------|---------------|---------------|----------------|----------------|-------|
| gpt2 | ⚠️ Limited | ❌ Too large | ✅ Good | ⚠️ Limited | Memory constraints in WebNN |
| gpt2-medium | ❌ Too large | ❌ Too large | ⚠️ Limited | ❌ Too large | Only works with WebGPU 4-bit |
| distilgpt2 | ⚠️ Limited | ❌ Too large | ✅ Good | ⚠️ Limited | Distilled model helps with size |

**Recommendations:**
- Always use WebGPU for GPT-2 models 
- Use distilgpt2 for better browser compatibility
- Always enable mixed precision with `--mixed-precision`

### LLAMA and OPT Family

| Model | WebNN (8-bit) | WebNN (4-bit) | WebGPU (4-bit) | WebGPU (2-bit) | Notes |
|-------|---------------|---------------|----------------|----------------|-------|
| facebook/opt-125m | ⚠️ Limited | ❌ Too large | ✅ Good | ✅ Limited | Small variant works well |
| facebook/opt-350m | ❌ Too large | ❌ Too large | ⚠️ Limited | ❌ Too large | Only with WebGPU 4-bit |
| TinyLlama/TinyLlama-1.1B | ❌ Too large | ❌ Too large | ⚠️ Limited | ❌ Too large | Only with WebGPU 4-bit and mixed precision |

**Recommendations:**
- Use the smallest variant possible (opt-125m works best)
- Always use mixed precision for these models
- WebGPU with 4-bit is the only viable option for most variants

**Example Command:**
```bash
python implement_comprehensive_webnn_webgpu.py --model facebook/opt-125m --platform webgpu --bits 4 --mixed-precision
```

### Other Text Generation Models

| Model Family | WebNN | WebGPU | Recommended Setup |
|--------------|-------|--------|-------------------|
| Bloom | ⚠️ 8-bit (small variants) | ✅ 4-bit+Mixed | WebGPU 4-bit+Mixed |
| Falcon | ❌ Too large | ⚠️ 4-bit (limited) | WebGPU 4-bit+Mixed, small variants only |
| Gemma | ❌ Too large | ⚠️ 4-bit (limited) | WebGPU 4-bit+Mixed, small variants only |
| Qwen | ⚠️ 8-bit (small variants) | ✅ 4-bit+Mixed | WebGPU 4-bit+Mixed |

## Vision Models

### Vision Transformer (ViT) Family

| Model | WebNN (8-bit) | WebNN (4-bit) | WebGPU (4-bit) | WebGPU (2-bit) | Notes |
|-------|---------------|---------------|----------------|----------------|-------|
| google/vit-base-patch16-224 | ✅ Good | ⚠️ Experimental | ✅ Excellent | ✅ Good | Great overall compatibility |
| google/vit-large-patch16-224 | ✅ Good | ❌ Too large | ✅ Good | ⚠️ Limited | Larger size limits WebNN |
| WinKawaks/vit-small-patch16-224 | ✅ Excellent | ✅ Experimental | ✅ Excellent | ✅ Excellent | Small variant works everywhere |

**Recommendations:**
- ViT models work well with WebGPU 4-bit across all browsers
- Enable shader precompilation for faster first inference
- For Edge browser, WebNN 8-bit is also a good option

### Other Vision Models

| Model Family | WebNN | WebGPU | Recommended Setup |
|--------------|-------|--------|-------------------|
| ResNet | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| Swin | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| DeiT | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| ConvNeXT | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| BEiT | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| DETR | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| SegFormer | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| YOLOS | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |

**Example Command:**
```bash
python implement_comprehensive_webnn_webgpu.py --model google/vit-base-patch16-224 --platform webgpu --bits 4
```

## Audio Models

### Whisper Family

| Model | WebNN (8-bit) | WebNN (4-bit) | WebGPU (4-bit) | WebGPU (2-bit) | Notes |
|-------|---------------|---------------|----------------|----------------|-------|
| openai/whisper-tiny | ✅ Good | ⚠️ Experimental | ✅ Excellent | ✅ Good | Firefox gives ~20% better performance |
| openai/whisper-base | ✅ Good | ❌ Too large | ✅ Good | ⚠️ Limited | Larger size impacts WebNN support |
| openai/whisper-small | ⚠️ Limited | ❌ Too large | ✅ Limited | ❌ Too large | Only WebGPU 4-bit is viable |

**Recommendations:**
- Use Firefox with WebGPU for all audio models
- Firefox's specialized compute shaders provide ~20% better performance
- Whisper-tiny has the best compatibility across platforms

### Other Audio Models

| Model Family | WebNN | WebGPU | Recommended Setup |
|--------------|-------|--------|-------------------|
| Wav2Vec2 | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |
| CLAP | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |
| HuBERT | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |
| Speech-to-Text | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |

**Example Command:**
```bash
python implement_comprehensive_webnn_webgpu.py --model openai/whisper-tiny --platform webgpu --browser firefox --bits 4
```

## Multimodal Models

### CLIP Family

| Model | WebNN (8-bit) | WebNN (4-bit) | WebGPU (4-bit) | WebGPU (2-bit) | Notes |
|-------|---------------|---------------|----------------|----------------|-------|
| openai/clip-vit-base-patch32 | ⚠️ Limited | ❌ Too large | ✅ Good | ⚠️ Limited | WebGPU preferred for CLIP |
| openai/clip-vit-large-patch14 | ❌ Too large | ❌ Too large | ⚠️ Limited | ❌ Too large | Only WebGPU 4-bit with mixed precision |

**Recommendations:**
- Always use WebGPU for CLIP models
- Enable parallel loading for 30-45% faster initialization
- Use mixed precision for better performance/accuracy tradeoff

### LLaVA and Other Multimodal Models

| Model Family | WebNN | WebGPU | Recommended Setup |
|--------------|-------|--------|-------------------|
| LLaVA | ❌ Too large | ⚠️ 4-bit (limited) | WebGPU 4-bit+Mixed, small variants only |
| BLIP | ⚠️ 8-bit (limited) | ✅ 4-bit | WebGPU 4-bit+Mixed |
| FLAVA | ❌ Too large | ⚠️ 4-bit (limited) | WebGPU 4-bit+Mixed, small variants only |
| GIT | ⚠️ 8-bit (limited) | ✅ 4-bit | WebGPU 4-bit+Mixed |
| XCLIP | ❌ Too large | ⚠️ 4-bit (limited) | WebGPU 4-bit+Mixed, small variants only |

**Example Command:**
```bash
python implement_comprehensive_webnn_webgpu.py --model openai/clip-vit-base-patch32 --platform webgpu --bits 4 --mixed-precision
```

## Testing Resources

### Automated Testing Script

Use the comprehensive testing script to validate any HuggingFace model:

```bash
python implement_comprehensive_webnn_webgpu.py --model MODEL_NAME
```

This script will:
1. Auto-detect model type and family
2. Test with appropriate quantization settings
3. Apply browser-specific optimizations
4. Work with both WebNN and WebGPU

### Generating Compatibility Matrix

Generate a full compatibility matrix by running:

```bash
# Test all families and generate matrix
python implement_comprehensive_webnn_webgpu.py --test-all-families --db-path results.db
python implement_comprehensive_webnn_webgpu.py --generate-matrix --db-path results.db --output matrix.md
```

### Browser Compatibility Testing

To test on different browsers:

```bash
python implement_comprehensive_webnn_webgpu.py --model bert-base-uncased --browsers chrome firefox edge
```

### Further Resources

- [WEBNN_WEBGPU_QUANTIZATION_GUIDE.md](WEBNN_WEBGPU_QUANTIZATION_GUIDE.md): Complete quantization guide
- [WEBNN_WEBGPU_QUANTIZATION_MARCH2025_UPDATE.md](WEBNN_WEBGPU_QUANTIZATION_MARCH2025_UPDATE.md): Latest updates
- [WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md](WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md): Technical summary