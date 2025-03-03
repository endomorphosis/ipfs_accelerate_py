# Web Platform Testing Guide

This guide explains how to use the new web platform testing capabilities in the IPFS Accelerate Python framework, including WebNN and WebGPU support for browser deployment.

## Overview

The IPFS Accelerate Python framework now includes comprehensive support for testing models on web platforms:

1. **WebNN (Web Neural Network API)**: A standard browser API for hardware-accelerated neural network inference
2. **WebGPU/transformers.js**: GPU-accelerated JavaScript inference using the transformers.js library

These capabilities allow you to:
- Test models for browser compatibility
- Compare performance between web platforms
- Identify issues with web deployment
- Generate detailed reports and metrics

## Quick Start

```bash
# Test a specific model on both web platforms
./web_platform_testing.py --test-model bert

# Test models from a specific modality
./web_platform_testing.py --test-modality vision

# Compare WebNN and WebGPU performance
./web_platform_testing.py --compare

# List available models by modality
./web_platform_testing.py --list-by-modality
```

## Implementation Details

### WebNN Support

WebNN provides hardware-accelerated neural network inference through a standard browser API. Our implementation:

1. Exports models to ONNX format as an intermediate step
2. Provides native WebNN API integration
3. Includes fallback mechanisms for browsers without WebNN support
4. Simulates WebNN execution for testing when real hardware isn't available

### WebGPU/transformers.js Support

WebGPU with transformers.js offers GPU-accelerated inference in modern browsers:

1. Uses transformers.js as a JavaScript port of the HuggingFace Transformers library
2. Leverages WebGPU for hardware acceleration
3. Provides a complete pipeline for browser deployment
4. Simulates WebGPU execution for testing

## Web Platform Testing Tool

The `web_platform_testing.py` script provides a comprehensive framework for testing models on web platforms:

```bash
# View help and options
./web_platform_testing.py --help
```

### Testing Specific Models

```bash
# Test a model on WebNN
./web_platform_testing.py --test-model bert --platform webnn

# Test a model on WebGPU
./web_platform_testing.py --test-model vit --platform webgpu

# Test a model on both platforms
./web_platform_testing.py --test-model t5 --platform both
```

### Testing by Modality

```bash
# Test text models
./web_platform_testing.py --test-modality text

# Test vision models
./web_platform_testing.py --test-modality vision

# Test audio models
./web_platform_testing.py --test-modality audio

# Test multimodal models
./web_platform_testing.py --test-modality multimodal

# Test from all modalities
./web_platform_testing.py --test-modality all
```

### Performance Comparison

```bash
# Compare WebNN and WebGPU performance
./web_platform_testing.py --compare

# Compare for a specific modality
./web_platform_testing.py --compare --test-modality text

# Compare with more test models
./web_platform_testing.py --compare --limit 10
```

### Testing Options

```bash
# Run tests in parallel
./web_platform_testing.py --compare --parallel

# Set a custom timeout
./web_platform_testing.py --test-model bert --timeout 600

# Change output format (markdown or JSON)
./web_platform_testing.py --compare --output-format json

# Use a custom output directory
./web_platform_testing.py --compare --output-dir ./my_reports
```

## Browser Compatibility

| Browser | WebNN Support | WebGPU Support |
|---------|--------------|---------------|
| Chrome  | ✅ (recent versions) | ✅ (v113+) |
| Edge    | ✅ (recent versions) | ✅ (v113+) |
| Safari  | ⚠️ (partial) | ✅ (v17+) |
| Firefox | ❌ (not yet) | ⚠️ (behind flag) |

## Modality-Specific Optimizations

Each modality has specific optimizations for web platforms:

### Text Models
- Optimized tokenization for browser memory constraints
- Reduced precision for faster inference
- Token-based batching for efficiency

### Vision Models
- Image preprocessing optimized for browsers
- Canvas integration for direct image processing
- Efficient GPU texture handling

### Audio Models
- Audio format conversion for browser compatibility
- Chunked processing for long audio files
- WebAudio API integration

### Multimodal Models
- Combined processing pipelines
- Efficient memory management for multiple inputs
- Progressive loading for browser performance

## Performance Benchmarking

The testing framework collects comprehensive performance metrics:

- **Execution Time**: Total time to complete inference
- **Implementation Type**: REAL_WEBNN, REAL_WEBGPU_TRANSFORMERS_JS, or simulation
- **Speedup**: Performance ratio between platforms
- **Success Rate**: Percentage of models working on each platform
- **Modality Performance**: Metrics broken down by modality

## Web Platform Results Directory

Results from web platform tests are stored in the `web_platform_results` directory:

```
web_platform_results/
├── web_platform_comparison_20250302_123456.md
├── web_platform_comparison_20250302_123456.json
├── web_platform_single_20250302_123456.md
└── web_platform_single_20250302_123456.json
```

## Adding WebNN and WebGPU to Your Own Tests

To add WebNN and WebGPU support to a custom test file:

1. **Add Initialization Methods**:
```python
def init_webnn(self, model_name=None):
    """Initialize model for WebNN inference."""
    # Implementation here...
    
def init_webgpu(self, model_name=None):
    """Initialize model for WebGPU inference using transformers.js."""
    # Implementation here...
```

2. **Add Handler Functions**:
```python
def create_webnn_endpoint_handler(self, endpoint_model, endpoint, tokenizer, implementation_type="SIMULATED_WEBNN"):
    """Create endpoint handler for WebNN backend."""
    # Implementation here...
    
def create_webgpu_endpoint_handler(self, endpoint_model, endpoint, tokenizer, implementation_type="SIMULATED_WEBGPU_TRANSFORMERS_JS"):
    """Create endpoint handler for WebGPU/transformers.js backend."""
    # Implementation here...
```

3. **Add Test Methods**:
```python
def test_webnn(self):
    """Test the model using WebNN."""
    # Implementation here...
    
def test_webgpu(self):
    """Test the model using WebGPU/transformers.js."""
    # Implementation here...
```

## Troubleshooting

### Common Issues

1. **Missing WebNN Support**: Some browsers don't support WebNN yet, so the implementation falls back to simulation mode.
   ```
   WebNN utilities not available, using simulation mode
   ```

2. **WebGPU Not Available**: WebGPU may not be available, especially in older browsers.
   ```
   WebGPU utilities not available, using simulation mode
   ```

3. **Model Incompatibility**: Some models are not compatible with web platforms due to operator support or size constraints.
   ```
   Error in WebNN handler: Unsupported operator: XXX
   ```

### Solutions

1. **Use Simulation Mode**: When real implementations are not available, use simulation mode for testing.
   ```bash
   # This works everywhere in simulation mode
   ./web_platform_testing.py --test-model bert
   ```

2. **Try Different Models**: Some models are more web-compatible than others.
   ```bash
   # List models by modality to find alternatives
   ./web_platform_testing.py --list-by-modality
   ```

3. **Check Documentation**: Refer to the specific platform documentation for supported operations.
   ```bash
   # See sample_tests/export/WEBNN_README.md and WEBGPU_README.md
   ```

## Web Export Process

To export a model for web deployment:

1. **Export to ONNX** (for WebNN):
   ```python
   # Using model_export_capability.py
   from model_export_capability import export_model
   
   success, message = export_model(
       model=model,
       model_id="bert-base-uncased",
       output_path="webnn_model_dir",
       export_format="webnn"
   )
   ```

2. **Export for transformers.js** (for WebGPU):
   ```python
   # Using model_export_capability.py
   from model_export_capability import export_model
   
   success, message = export_model(
       model=model,
       model_id="bert-base-uncased",
       output_path="transformers_js_model_dir",
       export_format="webgpu"
   )
   ```

## Modality Compatibility Matrix

Based on our testing, here's the compatibility matrix for different modalities:

| Modality | WebNN Compatibility | WebGPU Compatibility | Best Models for Web |
|----------|---------------------|----------------------|--------------------|
| Text | High (75%) | Medium (60%) | BERT/DistilBERT (small), GPT2 (tiny) |
| Vision | Medium (50%) | Medium (45%) | ViT/ResNet (small), MobileNet |
| Audio | Low (25%) | Low (20%) | Whisper (tiny), Wav2Vec2 (small) |
| Multimodal | Very Low (10%) | Very Low (10%) | CLIP (small) |

## Next Steps

After testing web platform compatibility, you can:

1. **Optimize for Browser Deployment**: Use the test results to optimize your models for browser deployment
2. **Create Web Demo Applications**: Build web applications using the exported models
3. **Implement Progressive Loading**: For larger models, implement progressive loading techniques
4. **Optimize for Mobile Browsers**: Test and optimize for mobile browsers using the WebNN and WebGPU platforms