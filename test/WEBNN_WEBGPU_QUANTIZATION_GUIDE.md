# WebNN and WebGPU Quantization Guide

## Introduction

This comprehensive guide explains how to use WebNN and WebGPU for model quantization in the IPFS Accelerate framework. Quantization reduces model size and improves inference performance by representing weights with lower numerical precision. Our framework now supports **all HuggingFace model classes** with comprehensive quantization options.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Quantization Options](#quantization-options)
4. [WebNN Implementation](#webnn-implementation)
5. [WebGPU Implementation](#webgpu-implementation)
6. [Browser Compatibility](#browser-compatibility)
7. [HuggingFace Model Coverage](#huggingface-model-coverage)
8. [Real-World Usage](#real-world-usage)
9. [Advanced Testing Tools](#advanced-testing-tools)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

## Overview

The IPFS Accelerate framework provides comprehensive support for model quantization using WebNN and WebGPU:

| Platform | Supported Precision | Memory Reduction | Primary Use Case |
|----------|---------------------|------------------|------------------|
| WebGPU   | 2-bit, 3-bit, 4-bit, 8-bit, 16-bit | Up to 87.5% | Memory-constrained environments |
| WebNN    | 8-bit, 16-bit (standard)<br>4-bit, 2-bit (experimental) | Up to 50% (standard)<br>Up to 75% (experimental) | Hardware-accelerated inference |

## Quick Start

The main entry point for testing WebNN and WebGPU quantization is `run_real_webgpu_webnn.py`:

```bash
# Test WebGPU with 4-bit precision
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 4

# Test WebNN with 8-bit precision
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 8

# Test WebNN with experimental 4-bit precision
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 4 --experimental-precision

# Test both platforms with default settings
python run_real_webgpu_webnn.py --platform both

# Test a specific model with auto-detection
python run_real_webgpu_webnn.py --model bert-base-uncased
```

## Quantization Options

### Bit Precision

The `--bits` parameter controls quantization precision:

| Bits | Memory Reduction | Accuracy Impact | Support |
|------|------------------|-----------------|---------|
| 16   | Baseline (FP16)  | None            | WebNN, WebGPU |
| 8    | ~50%             | ~1%             | WebNN, WebGPU |
| 4    | ~75%             | ~2-3%           | WebGPU, WebNN (experimental) |
| 3    | ~81.25%          | ~3-5%           | WebGPU only |
| 2    | ~87.5%           | ~5-8%           | WebGPU, WebNN (experimental) |

### Mixed Precision

Enable mixed precision with the `--mixed-precision` flag:

```bash
python run_real_webgpu_webnn.py --platform webgpu --bits 4 --mixed-precision
```

Mixed precision uses:
- Higher precision (8-bit) for critical layers (attention, embedding)
- Lower precision (4-bit/2-bit) for less sensitive layers (feed-forward)
- Dynamic adjustment based on model architecture

### Experimental Precision for WebNN

WebNN officially supports 8-bit quantization. To test experimental lower precision modes:

```bash
python run_real_webgpu_webnn.py --platform webnn --bits 4 --experimental-precision
```

WebNN offers two operational modes:

1. **Standard Mode** (default):
   - Automatically upgrades 4-bit/2-bit requests to 8-bit
   - Provides silent fallback for compatibility
   - No precision-related errors

2. **Experimental Mode**:
   - Attempts true 4-bit/2-bit precision
   - Reports detailed browser errors
   - Useful for research and debugging

## WebNN Implementation

WebNN provides efficient neural network acceleration through native browser APIs:

### Standard Mode Usage

```bash
# Test WebNN with 8-bit precision (native support)
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 8

# Test WebNN with 4-bit request (auto-upgrades to 8-bit)
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 4
```

### Experimental Mode Usage

```bash
# Test experimental 4-bit precision with error reporting
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 4 --experimental-precision

# Test experimental 2-bit precision with error reporting
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 2 --experimental-precision
```

### WebNN Architecture

The WebNN implementation:
- Uses browser's native ML acceleration APIs
- Integrates with ONNX Runtime Web for optimal performance
- Leverages hardware acceleration when available
- Adapts to device capabilities automatically

## WebGPU Implementation

WebGPU provides more flexible quantization options through custom shaders:

### Basic Usage

```bash
# Test WebGPU with 4-bit precision
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 4

# Test WebGPU with 2-bit ultra-low precision
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 2
```

### Advanced Features

```bash
# Mixed precision with critical layer optimization
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 4 --mixed-precision

# Firefox optimized audio model testing
python run_real_webgpu_webnn.py --platform webgpu --browser firefox --model whisper-tiny --bits 4
```

### WebGPU Architecture

The WebGPU implementation:
- Uses custom compute shaders for matrix multiplication
- Implements efficient packing for 2-bit and 4-bit values
- Provides specialized kernels for different model types
- Optimizes for browser-specific WebGPU implementations
- Includes Firefox-specific audio model optimizations

## Browser Compatibility

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| WebNN Support | ✅ Good (v122+) | ✅ Excellent | ❌ Not supported | ⚠️ Limited |
| WebGPU Support | ✅ Good | ✅ Good | ✅ Excellent | ⚠️ Limited |
| 8-bit Quantization | ✅ Both | ✅ Both | ✅ WebGPU only | ✅ Both (limited) |
| 4-bit Quantization | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU | ⚠️ WebGPU (limited) |
| 3-bit Quantization | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU | ❌ Not supported |
| 2-bit Quantization | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU | ❌ Not supported |
| Mixed Precision | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU | ⚠️ Limited |
| Audio Optimization | ⚠️ Limited | ⚠️ Limited | ✅ Excellent | ❌ Poor |

## HuggingFace Model Coverage

Our framework now provides comprehensive support for all HuggingFace model classes with WebNN and WebGPU quantization.

### Text Models

| Model Family | Examples | WebNN | WebGPU | Recommended Format |
|--------------|----------|-------|--------|-------------------|
| BERT | bert-base-uncased | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| T5 | t5-small | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| RoBERTa | roberta-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| DistilBERT | distilbert-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| ALBERT | albert-base-v2 | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| XLM | xlm-mlm-100-1280 | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| ELECTRA | electra-small | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |

### Text Generation Models

| Model Family | Examples | WebNN | WebGPU | Recommended Format |
|--------------|----------|-------|--------|-------------------|
| GPT-2 | gpt2 | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| LLAMA | opt-125m (tiny llama) | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| OPT | facebook/opt-125m | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| GPT-Neo | EleutherAI/gpt-neo-125m | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| BLOOM | bigscience/bloom-560m | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| Falcon | tiiuae/falcon-7b | ❌ | ⚠️ 4-bit** | WebGPU 4-bit+Mixed |
| Gemma | google/gemma-2b | ❌ | ⚠️ 4-bit** | WebGPU 4-bit+Mixed |
| Qwen | Qwen/Qwen1.5-0.5B | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |

\* Limited by memory, works with smaller variants only  
\** Large models may exceed browser memory limits, use smaller variants

### Vision Models

| Model Family | Examples | WebNN | WebGPU | Recommended Format |
|--------------|----------|-------|--------|-------------------|
| ViT | google/vit-base-patch16-224 | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| ResNet | microsoft/resnet-50 | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| Swin | microsoft/swin-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| DeiT | facebook/deit-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| ConvNeXT | facebook/convnext-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| BEiT | microsoft/beit-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| DETR | facebook/detr-resnet-50 | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| SegFormer | nvidia/segformer-b0 | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |
| YOLOS | hustvl/yolos-small | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit |

### Audio Models

| Model Family | Examples | WebNN | WebGPU | Recommended Format |
|--------------|----------|-------|--------|-------------------|
| Whisper | openai/whisper-tiny | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |
| Wav2Vec2 | facebook/wav2vec2-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |
| CLAP | laion/clap-htsat-unfused | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |
| HuBERT | facebook/hubert-base | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |
| Speech-to-Text | facebook/s2t-small | ✅ 8-bit | ✅ 4-bit | WebGPU 4-bit (Firefox) |

### Multimodal Models

| Model Family | Examples | WebNN | WebGPU | Recommended Format |
|--------------|----------|-------|--------|-------------------|
| CLIP | openai/clip-vit-base-patch32 | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| LLaVA | llava-hf/llava-1.5-7b-hf | ❌ | ⚠️ 4-bit** | WebGPU 4-bit+Mixed |
| BLIP | Salesforce/blip-image-captioning-base | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| FLAVA | facebook/flava-full | ❌ | ⚠️ 4-bit** | WebGPU 4-bit+Mixed |
| GIT | microsoft/git-base | ⚠️ 8-bit* | ✅ 4-bit | WebGPU 4-bit+Mixed |
| XCLIP | microsoft/xclip-base-patch32 | ❌ | ⚠️ 4-bit** | WebGPU 4-bit+Mixed |

\* Limited by memory, works with smaller variants only  
\** Large models may exceed browser memory limits, use smaller variants

## Real-World Usage

### Model Type Recommendations

1. **Text Models (BERT, T5)**:
   - Chrome/Firefox: WebGPU with 4-bit precision
   - Edge: WebNN with 8-bit for better browser integration
   - Recommended Flag: `--bits 4 --platform webgpu`

2. **Vision Models (ViT, CLIP)**:
   - All browsers: WebGPU with 4-bit or 8-bit
   - Mixed precision recommended for larger models
   - Recommended Flag: `--bits 4 --mixed-precision --platform webgpu`

3. **Audio Models (Whisper, Wav2Vec2)**:
   - Firefox: WebGPU with optimized compute shaders
   - ~20% better performance on Firefox for audio
   - Recommended Flag: `--browser firefox --platform webgpu --bits 4`

4. **Large Language Models (LLAMA, Qwen2)**:
   - WebGPU with mixed precision (critical: 8-bit, other: 4-bit)
   - Memory constraints often require 4-bit or lower
   - Recommended Flag: `--bits 4 --mixed-precision --platform webgpu`

### Production Deployment Tips

1. **Feature Detection**:
   - Test for browser support before using WebNN/WebGPU
   - Implement automatic fallbacks between technologies
   - Default to 8-bit when running on unknown devices

2. **Multi-browser Strategy**:
   - Chrome/Edge: Try WebNN first, then WebGPU
   - Firefox: Use WebGPU directly (no WebNN)
   - Safari: Try WebNN first with higher precision

3. **Performance Optimization**:
   - Use mixed precision for large models
   - Use dedicated audio optimizations on Firefox
   - Implement progressive loading for large models

## Advanced Testing Tools

We've introduced a new comprehensive testing tool to validate WebNN and WebGPU compatibility across all HuggingFace model classes:

```bash
# Test a specific model with automatic type detection
python implement_comprehensive_webnn_webgpu.py --model bert-base-uncased

# Test with specific quantization
python implement_comprehensive_webnn_webgpu.py --model llama-7b --platform webgpu --bits 4

# Mixed precision quantization
python implement_comprehensive_webnn_webgpu.py --model llava --mixed-precision

# WebNN experimental mode
python implement_comprehensive_webnn_webgpu.py --model bert-base-uncased --platform webnn --bits 4 --experimental-precision

# Test all model families
python implement_comprehensive_webnn_webgpu.py --test-all-families

# Generate compatibility matrix
python implement_comprehensive_webnn_webgpu.py --generate-matrix --output webnn_webgpu_matrix.md
```

### Database Integration

The tool can store test results in a DuckDB database:

```bash
# Run test and store results in database
python implement_comprehensive_webnn_webgpu.py --model bert-base-uncased --db-path results.db

# Generate matrix from database
python implement_comprehensive_webnn_webgpu.py --generate-matrix --db-path results.db --output matrix.md
```

## Troubleshooting

### Common WebNN Issues

1. **Precision Errors**:
   - Error: `Failed to execute 'buildSync' on 'MLGraphBuilder': The operator is not supported with the requested precision`
   - Solution: Use standard mode instead of experimental, or increase precision to 8-bit

2. **Browser Compatibility**:
   - Error: `WebNN API not found`
   - Solution: Use Chrome 122+ or Edge, Firefox doesn't support WebNN

3. **Memory Issues**:
   - Error: `Failed to allocate memory`
   - Solution: Reduce model size or use lower precision

### Common WebGPU Issues

1. **Shader Compilation Errors**:
   - Error: `Shader compilation failed`
   - Solution: Use higher precision (4-bit instead of 2-bit) or try a different browser

2. **Device Limits**:
   - Error: `WebGPU device limits exceeded`
   - Solution: Use smaller models or reduce batch size

3. **Browser Support**:
   - Error: `WebGPU not available`
   - Solution: Use a supported browser or enable flags in experimental browsers

4. **Model Too Large**:
   - Error: `Out of memory`
   - Solution: Use a smaller model variant, lower precision, or enable mixed precision

### Model-Specific Issues

1. **Large LLMs (7B+)**:
   - Issue: Browser memory limits exceeded
   - Solution: Use smaller variants (125M-1B) or consider server-side inference

2. **LLaVA and other multimodal models**:
   - Issue: High memory usage from multiple components
   - Solution: Enable parallel loading for faster initialization and mixed precision for lower memory usage

3. **Audio models on Chrome/Edge**:
   - Issue: Suboptimal compute shader configuration
   - Solution: Use Firefox for audio models (~20% better performance)

## API Reference

### Command Line Options

| Option | Description | Values |
|--------|-------------|--------|
| `--platform` | Platform to use | `webgpu`, `webnn`, `both` |
| `--browser` | Browser to use | `chrome`, `firefox`, `edge`, `safari` |
| `--bits` | Precision level | `2`, `3`, `4`, `8`, `16` |
| `--mixed-precision` | Enable mixed precision | Flag |
| `--experimental-precision` | Enable experimental WebNN precision | Flag |
| `--model` | Model to test | e.g., `bert-base-uncased` |
| `--model-type` | Model type | `auto`, `text`, `vision`, `audio`, `multimodal`, `text_generation` |
| `--headless` | Run in headless mode | Flag |
| `--no-simulation` | Disable simulation fallback | Flag |
| `--verbose` | Enable verbose logging | Flag |
| `--db-path` | Database path for storing results | Path |
| `--test-family` | Test a specific model family | Family name |
| `--test-all-families` | Test all model families | Flag |
| `--generate-matrix` | Generate compatibility matrix | Flag |
| `--output` | Output file path | Path |

### Python API

```python
import anyio
from implement_comprehensive_webnn_webgpu import run_webnn_webgpu_test

# Run WebNN test
async def test_webnn():
    args = argparse.Namespace(
        platform="webnn",
        model="bert-base-uncased",
        model_type="text",
        browser="edge",
        visible_browser=False,
        bits=8,
        scheme="symmetric",
        mixed_precision=False,
        experimental_precision=False
    )
    result = await run_webnn_webgpu_test(args)
    print(f"WebNN result: {result}")

# Run WebGPU test
async def test_webgpu():
    args = argparse.Namespace(
        platform="webgpu",
        model="bert-base-uncased",
        model_type="text",
        browser="chrome",
        visible_browser=False,
        bits=4,
        scheme="symmetric",
        mixed_precision=True,
        experimental_precision=False
    )
    result = await run_webnn_webgpu_test(args)
    print(f"WebGPU result: {result}")

# Run tests
anyio.run(test_webnn)
anyio.run(test_webgpu)
```

## Further Resources

- [WEBNN_WEBGPU_QUANTIZATION_MARCH2025_UPDATE.md](WEBNN_WEBGPU_QUANTIZATION_MARCH2025_UPDATE.md): March 2025 update details
- [WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md](WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md): Technical details of quantization
- [WEBNN_WEBGPU_DOCS_INDEX.md](WEBNN_WEBGPU_DOCS_INDEX.md): Documentation index
- [WEBNN_WEBGPU_MODEL_COVERAGE.md](WEBNN_WEBGPU_MODEL_COVERAGE.md): Detailed model coverage report