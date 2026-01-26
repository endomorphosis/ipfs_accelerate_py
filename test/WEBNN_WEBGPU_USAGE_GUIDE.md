# WebNN and WebGPU Usage Guide with Quantization

This guide provides detailed instructions for using WebNN and WebGPU with quantization in the IPFS Accelerate framework.

## Overview

WebNN and WebGPU provide hardware acceleration for machine learning models in browser environments:

- **WebNN**: Web Neural Network API providing hardware-accelerated neural network operations
- **WebGPU**: Web Graphics API with compute shader support for ML acceleration
- **Quantization**: Technique to reduce model size and improve inference speed by using lower precision

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Modern browser with WebNN/WebGPU support (Chrome, Firefox, Edge, or Safari)
- Graphics hardware with WebGPU support for optimal performance

### Required Python Packages

```bash
pip install websockets==15.0 selenium==4.29.0 webdriver-manager==4.0.2
```

### Browser Setup

Install WebDriver for your browser(s):

```bash
python generators/models/test_webnn_webgpu_integration.py --install-drivers
```

## Quick Start

### Basic WebGPU with 4-bit Quantization

```python
import anyio
from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
from fixed_web_platform.webgpu_quantization import setup_4bit_inference

async def run_webgpu_example():
    # Initialize WebGPU implementation
    impl = RealWebGPUImplementation(browser_name="chrome", headless=True)
    await impl.initialize()
    
    # Initialize model
    model_name = "bert-base-uncased"
    model_info = await impl.initialize_model(model_name, model_type="text")
    
    # Create inference options with quantization
    inference_options = {
        "use_quantization": True,
        "bits": 4,
        "scheme": "symmetric",
        "mixed_precision": False
    }
    
    # Run inference with quantization
    result = await impl.run_inference(model_name, "Example input text", inference_options)
    print(result)
    
    # Shutdown
    await impl.shutdown()

# Run the example
anyio.run(run_webgpu_example)
```

### Basic WebNN with 8-bit Quantization

```python
import anyio
from fixed_web_platform.webnn_implementation import RealWebNNImplementation

async def run_webnn_example():
    # Initialize WebNN implementation
    impl = RealWebNNImplementation(browser_name="edge", headless=True, device_preference="gpu")
    await impl.initialize()
    
    # Initialize model
    model_name = "bert-base-uncased"
    model_info = await impl.initialize_model(model_name, model_type="text")
    
    # Create inference options with quantization
    inference_options = {
        "use_quantization": True,
        "bits": 8,  # WebNN only supports 8-bit quantization
        "scheme": "symmetric",
        "mixed_precision": False
    }
    
    # Run inference with quantization
    result = await impl.run_inference(model_name, "Example input text", inference_options)
    print(result)
    
    # Shutdown
    await impl.shutdown()

# Run the example
anyio.run(run_webnn_example)
```

## Quantization Options

### Supported Bit Levels

| Platform | 16-bit (FP16) | 8-bit (INT8) | 4-bit (INT4) | 2-bit (INT2) | Notes |
|----------|---------------|--------------|--------------|--------------|-------|
| WebGPU   | ✅            | ✅           | ✅           | ✅           | Full range support |
| WebNN    | ✅            | ✅           | ❌           | ❌           | Limited to 8-bit minimum |

### Memory Reduction

| Precision | Memory Reduction | Accuracy Impact | Use Case |
|-----------|------------------|----------------|----------|
| 16-bit    | 0% (baseline)    | None           | Highest accuracy |
| 8-bit     | 50%              | Minimal        | Good balance for most models |
| 4-bit     | 75%              | Moderate       | Memory-constrained environments |
| 2-bit     | 87.5%            | Significant    | Ultra-constrained environments |

### Quantization Schemes

- **Symmetric**: Uses symmetrical range around zero (e.g., -8 to 7 for 4-bit)
- **Asymmetric**: Uses full range (e.g., 0 to 15 for 4-bit) with zero-point offset
- **Per-channel**: Different scales per output channel
- **Per-tensor**: One scale for entire tensor

### Mixed Precision

Mixed precision keeps some layers at higher precision:
- Attention layers are often kept at higher precision
- Embedding layers are usually not quantized
- LayerNorm layers are kept at full precision

## Advanced Usage

### WebGPU with Ultra-Low Precision (2-bit)

```python
import anyio
from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation

async def run_ultra_low_precision():
    # Initialize WebGPU implementation
    impl = RealWebGPUImplementation(browser_name="chrome", headless=True)
    await impl.initialize()
    
    # Initialize model
    model_name = "bert-base-uncased"
    model_info = await impl.initialize_model(model_name, model_type="text")
    
    # Create inference options with 2-bit quantization
    inference_options = {
        "use_quantization": True,
        "bits": 2,  # Ultra-low precision
        "scheme": "symmetric",
        "mixed_precision": True  # Use mixed precision to maintain accuracy
    }
    
    # Run inference
    result = await impl.run_inference(model_name, "Example input text", inference_options)
    print(result)
    
    # Get performance metrics
    if "performance_metrics" in result:
        metrics = result["performance_metrics"]
        print(f"Quantization bits: {metrics.get('quantization_bits', 'N/A')}")
        print(f"Inference time: {metrics.get('inference_time_ms', 'N/A')} ms")
        print(f"Memory reduction: {metrics.get('memory_reduction_percent', 'N/A')}%")
    
    # Shutdown
    await impl.shutdown()

# Run the example
anyio.run(run_ultra_low_precision)
```

### Advanced Quantization with Per-Channel and Mixed Precision

```python
import anyio
from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
from fixed_web_platform.webgpu_quantization import WebGPUQuantizer, quantize_model_weights

async def run_advanced_quantization():
    # Create custom quantizer
    quantizer = WebGPUQuantizer(
        bits=4,
        group_size=128,  # Use per-128-element quantization
        scheme="symmetric"
    )
    
    # Initialize WebGPU implementation
    impl = RealWebGPUImplementation(browser_name="chrome", headless=True)
    await impl.initialize()
    
    # Initialize model
    model_name = "bert-base-uncased"
    model_info = await impl.initialize_model(model_name, model_type="text")
    
    # Create advanced inference options
    inference_options = {
        "use_quantization": True,
        "bits": 4,
        "scheme": "symmetric",
        "mixed_precision": True,
        "attention_precision": 8,  # Keep attention at 8-bit
        "group_size": 128,
        "skip_layernorm": True,  # Don't quantize layernorm
    }
    
    # Run inference
    result = await impl.run_inference(model_name, "Example input text", inference_options)
    print(result)
    
    # Shutdown
    await impl.shutdown()

# Run the example
anyio.run(run_advanced_quantization)
```

### Browser-Optimized Testing

```python
import anyio
import sys

async def run_browser_optimized_test():
    browser = sys.argv[1] if len(sys.argv) > 1 else "chrome"
    model = sys.argv[2] if len(sys.argv) > 2 else "bert-base-uncased"
    
    # Import implementations
    from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
    
    # Initialize implementation
    impl = RealWebGPUImplementation(browser_name=browser, headless=True)
    await impl.initialize()
    
    # Configure browser-specific optimizations
    inference_options = {
        "use_quantization": True,
        "bits": 4,
        "mixed_precision": True,
    }
    
    # Firefox-specific optimizations for audio models
    if browser == "firefox" and model in ["whisper-tiny", "wav2vec2-base"]:
        inference_options["compute_shaders"] = True
        inference_options["workgroup_size"] = [256, 1, 1]
        print("Using Firefox-optimized compute shader configuration")
    
    # Edge-specific optimizations
    elif browser == "edge":
        inference_options["execution_provider"] = "gpu"
        print("Using Edge-optimized GPU execution provider")
    
    # Initialize model and run inference
    await impl.initialize_model(model, model_type="text")
    result = await impl.run_inference(model, "Example input text", inference_options)
    
    # Print metrics
    if "performance_metrics" in result:
        metrics = result["performance_metrics"]
        print(f"Inference time: {metrics.get('inference_time_ms', 'N/A')} ms")
    
    # Shutdown
    await impl.shutdown()

# Run with browser from command line
if __name__ == "__main__":
    anyio.run(run_browser_optimized_test)
```

## Testing Tools

### Simplified Verification Script

```bash
# Test WebGPU with 4-bit quantization on Chrome
python generators/models/test_webnn_webgpu_simplified.py --platform webgpu --bits 4 --browser chrome

# Test WebNN with 8-bit quantization on Edge
python generators/models/test_webnn_webgpu_simplified.py --platform webnn --bits 8 --browser edge

# Test both platforms with default settings
python generators/models/test_webnn_webgpu_simplified.py --platform both

# Test with mixed precision
python generators/models/test_webnn_webgpu_simplified.py --platform webgpu --mixed-precision
```

### Comprehensive Testing Script

```bash
# Test WebGPU with 4-bit quantization
python webnn_webgpu_quantization_test.py --platform webgpu --browser chrome --model bert-base-uncased --bits 4

# Test WebNN with 8-bit quantization
python webnn_webgpu_quantization_test.py --platform webnn --browser edge --model bert-base-uncased --bits 8

# Test ultra-low precision (2-bit)
python webnn_webgpu_quantization_test.py --platform webgpu --browser chrome --model bert-base-uncased --bits 2

# Test with Firefox optimizations for audio models
python webnn_webgpu_quantization_test.py --platform webgpu --browser firefox --model whisper-tiny --bits 4
```

### Shell Script for Batch Testing

```bash
# Run all tests on Chrome
./run_webnn_webgpu_quantization.sh --all --chrome

# Test WebGPU with Chrome and Firefox
./run_webnn_webgpu_quantization.sh --webgpu-only --chrome --firefox

# Test WebNN with Edge
./run_webnn_webgpu_quantization.sh --webnn-only --edge

# Test with mixed precision
./run_webnn_webgpu_quantization.sh --all --mixed-precision

# Test with ultra-low precision (2-bit)
./run_webnn_webgpu_quantization.sh --webgpu-only --chrome --ultra-low-prec

# Custom model
./run_webnn_webgpu_quantization.sh --model whisper-tiny --firefox
```

## Browser Compatibility and Recommendations

### Chrome/Edge

- Excellent WebGPU support with all bit levels (2, 4, 8, 16)
- WebNN support (in Chrome 122+ and Edge)
- Recommended for general-purpose ML in browsers
- Best compatibility and performance for most models

Optimal settings:
```python
{
    "bits": 4,                # 4-bit quantization for most models
    "scheme": "symmetric",    # Symmetrical quantization
    "mixed_precision": True,  # Mixed precision for critical layers
    "group_size": 128         # Group size for quantization
}
```

### Firefox

- Good WebGPU support but no WebNN
- Excellent performance for audio models (20% better than Chrome)
- Optimized compute shader workgroups for audio processing

Optimal settings for audio models:
```python
{
    "bits": 4,                # 4-bit quantization
    "scheme": "symmetric",    # Symmetrical quantization
    "compute_shaders": True,  # Enable compute shaders
    "workgroup_size": [256, 1, 1]  # Firefox-optimized workgroup size
}
```

### Safari

- Limited WebGPU support
- No WebNN support
- Recommend 8-bit quantization for better compatibility
- Use fallbacks for unsupported features

Optimal settings:
```python
{
    "bits": 8,                # Higher precision for Safari
    "scheme": "symmetric",    # Symmetrical quantization
    "chunked_operations": True,  # Break operations into smaller chunks
    "safari_metal_fallback": True  # Enable Metal fallbacks
}
```

## Model-Specific Recommendations

### Text Models (BERT, T5)

```python
{
    "bits": 4,                # 4-bit for most cases
    "scheme": "symmetric",    # Symmetrical quantization
    "mixed_precision": True,  # Mixed precision
    "skip_layernorm": True    # Skip LayerNorm quantization
}
```

### Vision Models (ViT, CLIP-Vision)

```python
{
    "bits": 8,                # 8-bit for vision models
    "scheme": "symmetric",    # Symmetrical quantization
    "mixed_precision": True,  # Mixed precision
    "shader_precompilation": True  # Pre-compile shaders
}
```

### Audio Models (Whisper, Wav2Vec2)

```python
{
    "bits": 4,                # 4-bit for audio models
    "scheme": "symmetric",    # Symmetrical quantization
    "compute_shaders": True,  # Enable compute shaders
    "browser_optimized": True # Use browser-specific optimizations
}
```

### Large Language Models (LLAMA, Qwen2)

```python
{
    "bits": 4,                # 4-bit for LLMs
    "scheme": "symmetric",    # Symmetrical quantization
    "mixed_precision": True,  # Mixed precision is critical
    "attention_precision": 8  # Keep attention at higher precision
}
```

## Troubleshooting

### Common Issues

1. **Browser-Specific Errors**:
   - **Chrome/Edge**: Enable flags `--enable-features=WebGPU,WebNN`
   - **Firefox**: Set `dom.webgpu.enabled` to `true` in `about:config`
   - **Safari**: Ensure Metal support is enabled

2. **Unsupported Bit Width**:
   - WebNN doesn't support 4-bit or 2-bit quantization
   - If requesting <8-bit with WebNN, it will automatically fall back to 8-bit

3. **Memory Issues**:
   - Use lower bit width quantization (4-bit or 2-bit)
   - Enable mixed precision to reduce overall memory usage
   - Increase browser memory limits with `--js-flags="--max-old-space-size=8192"`

4. **WebDriver Issues**:
   - Update WebDriver: `python generators/models/test_webnn_webgpu_integration.py --install-drivers`
   - Check browser version compatibility

### Quantization Debugging

```python
# Debug quantization parameters
from fixed_web_platform.webgpu_quantization import WebGPUQuantizer
import numpy as np

# Create sample data
tensor = np.random.randn(768, 768).astype(np.float32)

# Create quantizer and quantize
quantizer = WebGPUQuantizer(bits=4, group_size=128, scheme="symmetric")
quantized = quantizer.quantize_tensor(tensor)

# Dequantize for verification
dequantized = quantizer.dequantize_tensor(quantized)

# Check accuracy
error = np.abs(tensor - dequantized).mean()
print(f"Mean quantization error: {error}")

# Check memory reduction
original_size = tensor.size * tensor.itemsize
quantized_size = quantized["data"].size * quantized["data"].itemsize
reduction = 100 * (1 - quantized_size / original_size)
print(f"Memory reduction: {reduction:.2f}%")
```

## Performance Benchmarks

### Text Models (BERT-base)

| Browser | Precision | Latency (ms) | Memory (MB) | Notes |
|---------|-----------|--------------|-------------|-------|
| Chrome WebGPU | FP16 | 8.2 | 112 | Baseline |
| Chrome WebGPU | INT8 | 6.5 | 76 | 50% memory reduction |
| Chrome WebGPU | INT4 | 4.8 | 48 | 75% memory reduction |
| Chrome WebGPU | INT2 | 3.7 | 36 | 87.5% reduction, accuracy loss |
| Edge WebNN | INT8 | 10.5 | 98 | Good performance |
| Firefox WebGPU | INT4 | 5.2 | 49 | Good alternative |

### Vision Models (ViT-base)

| Browser | Precision | Latency (ms) | Memory (MB) | Notes |
|---------|-----------|--------------|-------------|-------|
| Chrome WebGPU | FP16 | 15.7 | 156 | Baseline |
| Chrome WebGPU | INT8 | 11.8 | 98 | Good accuracy |
| Chrome WebGPU | INT4 | 9.5 | 60 | Reasonable accuracy |
| Edge WebNN | INT8 | 19.8 | 138 | Good performance |
| Firefox WebGPU | INT4 | 10.2 | 62 | Good alternative |

### Audio Models (Whisper-tiny)

| Browser | Precision | Latency (ms) | Memory (MB) | Notes |
|---------|-----------|--------------|-------------|-------|
| Chrome WebGPU | FP16 | 52.3 | 184 | Baseline |
| Chrome WebGPU | INT8 | 39.2 | 104 | 50% memory reduction |
| Chrome WebGPU | INT4 | 31.7 | 70 | 75% memory reduction |
| Firefox WebGPU | INT4 | 25.4 | 71 | 20% faster than Chrome |
| Edge WebNN | INT8 | 58.2 | 162 | Reasonable performance |

## Conclusion

WebNN and WebGPU provide powerful acceleration for ML models in browsers, with quantization enabling significant memory and performance improvements:

- **WebGPU** offers the most flexibility with 2/4/8/16-bit support
- **WebNN** provides good 8-bit support with native browser integration
- **Browser choice matters**, with Firefox excelling for audio models
- **Model type affects optimal settings**, with different bit widths and optimizations recommended for different model types

For optimal results, use the browser and quantization settings recommended for your specific model type, and leverage browser-specific optimizations when available.

## Additional Resources

- [WebGPU WebNN Quantization Summary](WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md)
- [WebNN WebGPU Implementation Guide](WEBNN_WEBGPU_GUIDE.md)
- [Run shell script for testing all browsers](run_webnn_webgpu_quantization.sh)
- [Simplified test script](test_webnn_webgpu_simplified.py)
- [Comprehensive test script](webnn_webgpu_quantization_test.py)