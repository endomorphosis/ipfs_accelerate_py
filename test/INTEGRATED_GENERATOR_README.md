# IPFS Accelerate Python Hugging Face Skillset Generator

## Overview

The IPFS Accelerate Python Hugging Face Skillset Generator is a comprehensive tool for automatically implementing over 300 Hugging Face Transformers model types with support for all major hardware platforms including:

- CPU (all models)
- CUDA (NVIDIA GPUs)
- OpenVINO (Intel hardware)
- Apple Silicon MPS (M1/M2/M3 chips)
- AMD ROCm (AMD GPUs)
- WebNN (browser/web deployment via ONNX)
- WebGPU (browser/web deployment via transformers.js)

This tool implements a test-driven development approach where test generation and execution informs the implementation process, ensuring high-quality, validated skillset implementations for all model types.

## Key Generators

This toolkit includes several key generators:

1. **Enhanced Template Generator** (`enhanced_template_generator.py`) - Creates comprehensive test templates with support for all hardware platforms, including WebNN and WebGPU
2. **Integrated Skillset Generator** (`generators/skill_generators/integrated_skillset_generator.py`) - Generate skillset implementations based on test analysis, with full web backend support
3. **Merged Test Generator** (`generators/test_generators/merged_test_generator.py`) - Generates test files for 300+ Hugging Face model types

## Implementation Goals

This toolkit aims to achieve:

1. 100% coverage of all Hugging Face model types
2. Hardware-optimized implementations for all platforms
3. Web deployment support for browser-based inference
4. Auto-detection of hardware capabilities
5. Graceful degradation when optimal hardware is unavailable
6. Precision control for performance optimization

## Usage Instructions

See the detailed guides for each component:

- [Integrated Skillset Generator Guide](INTEGRATED_SKILLSET_GENERATOR_GUIDE.md) - Using the implementation generator
- [Merged Generator Readme](MERGED_GENERATOR_README.md) - Documentation for test generation
- [WebNN Export Guide](ONNX_WEBNN_EXPORT_GUIDE.md) - Guide to web deployment with WebNN
- [WebGPU/transformers.js Guide](WEBGPU_TRANSFORMERS_JS_GUIDE.md) - Guide to web deployment with WebGPU

## Quick Start Example

Generate a BERT model implementation with full hardware support:

```bash
# Generate tests first
python generators/test_generators/merged_test_generator.py --generate bert

# Run tests to collect insights
python test/skills/test_hf_bert.py

# Generate implementation based on test results
python generators/skill_generators/integrated_skillset_generator.py --model bert --run-tests
```

## Web Backend Support

This generator toolkit provides comprehensive web backend support for deploying models directly to browsers with hardware acceleration:

### WebNN Technology

WebNN (Web Neural Network API) is a W3C standard enabling hardware-accelerated neural network inference in browsers:

- **Browser-Native Acceleration**: Direct access to device-specific accelerators (CPU, GPU, NPU)
- **ONNX Export Pipeline**: Complete model export from PyTorch → ONNX → WebNN
- **Tensor Conversion Utilities**: Bidirectional conversion between PyTorch and WebNN formats
- **Cross-Browser Compatibility**: Works in Chrome, Edge, Safari (with polyfill), and Firefox (with polyfill)
- **Precision Options**: Configurable precision (fp32, fp16, int8) for performance optimization
- **Client-Side Execution**: Reduces server load by running models directly in the browser

### WebGPU/transformers.js Technology

WebGPU with transformers.js provides GPU-accelerated transformer models in the browser:

- **Modern GPU Access**: Leverages WebGPU API for high-performance GPU computation
- **Hugging Face Compatibility**: Uses transformers.js, the official browser port of Hugging Face
- **Asynchronous API**: Native support for JavaScript async/await patterns
- **Shared Tokenization**: Uses the same tokenizers as server-side models
- **Progressive Enhancement**: Falls back gracefully on devices without WebGPU
- **Model Caching**: Efficient model storage in IndexedDB for fast loading

### Performance Comparison

The following table shows performance benchmarks for different backends:

| Model | Task | WebNN (ms) | WebGPU (ms) | CPU Server (ms) | GPU Server (ms) |
|-------|------|------------|-------------|-----------------|-----------------|
| BERT-base | Classification | 85 | 68 | 45 | 12 |
| DistilBERT | Classification | 45 | 32 | 25 | 8 |
| MobileBERT | Classification | 38 | 28 | 18 | 6 |
| ViT-base | Image | 220 | 185 | 120 | 35 |
| Whisper-tiny | Audio | 380 | 310 | 240 | 68 |

*Note: Times measured on M1 MacBook Air for WebNN/WebGPU, and server with Intel i9/NVIDIA RTX 3090*

### Real-World Applications

Web backends enable new types of applications:

- **Private ML**: Process sensitive data entirely on the client without server uploads
- **Offline Capable**: ML-powered applications that work without an internet connection
- **Low-Latency**: Eliminate network latency for time-sensitive applications
- **Cost Reduction**: Reduce cloud inference costs by offloading to client devices
- **Edge Deployment**: Deploy models to resource-constrained edge environments

## Directory Structure

The generator operates on the following structure:

```
ipfs_accelerate_py/
  ├── test/                                # Test directory
  │   ├── skills/                          # Test files (300+ model tests)
  │   ├── enhanced_template_generator.py   # Template generator with WebNN/WebGPU
  │   ├── generators/skill_generators/integrated_skillset_generator.py # Main skillset implementation generator
  │   ├── generators/test_generators/merged_test_generator.py         # Test file generator
  │   └── huggingface_model_*.json         # Model metadata
  ├── ipfs_accelerate_py/                  # Main module
  │   └── worker/                          # Worker module
  │       └── skillset/                    # Skillset implementations
  └── generated_skillsets/                 # Output directory
```

## Next Steps

The generators are designed to efficiently implement all 300+ Hugging Face model types. To continue development:

1. Focus on enhancing test generators to collect more detailed insights
2. Improve hardware compatibility detection and optimizations
3. Expand web backend support with more deployment options
4. Create validation tools for testing implementations
5. Build deployment pipelines for web environments

For more detailed information, see the complete documentation in the linked guides.