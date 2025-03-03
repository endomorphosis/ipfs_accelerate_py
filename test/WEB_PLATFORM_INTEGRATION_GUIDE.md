# Web Platform Integration Guide

## Overview

The ResourcePool system now provides comprehensive support for web-based deployment using WebNN and WebGPU. This guide covers how to use the enhanced ResourcePool with web platforms for efficient browser-based inference.

## Web Platform Capabilities

The framework supports two primary web acceleration technologies:

1. **WebNN (Web Neural Network API)** - A standard API for accelerated neural network inference on the web
2. **WebGPU** - A modern graphics and compute API for the web that can be used for ML inference

These technologies enable deploying machine learning models directly in the browser with hardware acceleration.

## Features Added for Web Platform Support

1. **WebNN/WebGPU Detection**: Automatic detection of web platform capabilities
2. **Web Deployment Subfamilies**: Special model subfamily definitions for web deployment
3. **Browser-Optimized Settings**: Configurations specifically for browser environments
4. **Simulation Mode**: Testing web platform features outside of actual browsers
5. **Model Size Limitations**: Special handling for browser-compatible model sizes
6. **Family-Based Optimizations**: Different strategies for different model types
7. **Resilient Error Handling**: Graceful fallbacks when web acceleration isn't available
8. **Comprehensive Testing**: Dedicated test suite for web platform integration

## Using Web Platform Features

### Hardware Preferences for Web Deployment

```python
from resource_pool import get_global_resource_pool
from hardware_detection import WEBNN, WEBGPU, CPU

# Get resource pool
pool = get_global_resource_pool()

# Create web-specific hardware preferences for embedding models
web_embedding_prefs = {
    "priority_list": [WEBNN, WEBGPU, CPU],
    "model_family": "embedding",
    "subfamily": "web_deployment",
    "fallback_to_simulation": True,
    "browser_optimized": True
}

# Create web-specific hardware preferences for vision models
web_vision_prefs = {
    "priority_list": [WEBGPU, WEBNN, CPU],
    "model_family": "vision",
    "subfamily": "web_deployment",
    "fallback_to_simulation": True,
    "browser_optimized": True
}

# Create web-specific hardware preferences for small text generation models
web_text_prefs = {
    "priority_list": [WEBNN, CPU],
    "model_family": "text_generation",
    "subfamily": "web_deployment",
    "fallback_to_simulation": True,
    "browser_optimized": True,
    "max_model_size": "tiny"  # Limit to small models for browser
}

# Load models with web deployment preferences
embedding_model = pool.get_model(
    "embedding", 
    "prajjwal1/bert-tiny",
    constructor=lambda: create_bert_model(),
    hardware_preferences=web_embedding_prefs
)

vision_model = pool.get_model(
    "vision", 
    "google/vit-base-patch16-224",
    constructor=lambda: create_vision_model(),
    hardware_preferences=web_vision_prefs
)

text_model = pool.get_model(
    "text_generation", 
    "google/t5-efficient-tiny",
    constructor=lambda: create_t5_model(),
    hardware_preferences=web_text_prefs
)
```

### Testing Web Platform Integration

The framework includes dedicated tests for web platform features:

```bash
# Run web platform specific test
python test_resource_pool.py --test web --debug

# Enable simulation mode for testing in non-browser environments
python test_resource_pool.py --test web --simulation --debug

# Run with hardware testing for more comprehensive verification
python test_resource_pool.py --test hardware --web-platform --debug
```

### Web Platform Compatibility Matrix

Below is a compatibility matrix for different model families with web platforms:

| Model Family | WebNN | WebGPU | Examples | Notes |
|--------------|-------|--------|----------|-------|
| Embedding | ✅ High | ✅ Medium | BERT, RoBERTa | Best on WebNN |
| Vision | ✅ Medium | ✅ High | ViT, ResNet | Best on WebGPU |
| Text Generation (small) | ✅ Medium | ❌ Low | T5 (tiny/small) | Limited to small models, WebNN preferred |
| Audio | ❌ | ❌ | Whisper | Not currently supported in browsers |
| Multimodal | ❌ | ❌ | LLaVA, CLIP | Not currently supported in browsers |

### Web Platform Simulation Mode

For development and testing outside of actual browser environments, the framework provides a simulation mode:

```python
import os

# Enable simulation mode
os.environ["WEBNN_SIMULATION"] = "1"
os.environ["WEBGPU_SIMULATION"] = "1"

# Now ResourcePool will simulate WebNN/WebGPU capabilities
# This is useful for testing browser deployment configurations
# without an actual browser environment
```

You can also enable simulation mode for tests:

```bash
python test_resource_pool.py --test web --simulation --debug
```

### Error Handling for Web Platforms

The ResourcePool provides special error handling for web platforms:

```python
try:
    # Try to load a model that may not be compatible with web platforms
    model = pool.get_model(
        "audio", 
        "openai/whisper-large-v3",
        constructor=lambda: create_whisper_model(),
        hardware_preferences={"priority_list": [WEBNN, WEBGPU, CPU]}
    )
except Exception as e:
    # The ResourcePool provides helpful error messages for web platform issues
    if "web platform" in str(e) or "browser" in str(e):
        print("This model is not compatible with web deployment")
        # Handle the error appropriately for your application
    else:
        # Handle other types of errors
        raise
```

## Implementation Details

### WebNN Detection

WebNN is detected through several methods:

1. Checking for Node.js with ONNX export capabilities
2. Looking for the WebNN API in the browser environment
3. Checking for onnxruntime-web with WebNN backend

### WebGPU Detection

WebGPU is detected through:

1. Checking for Node.js with transformers.js
2. Looking for WebGPU API in the browser
3. Checking for ONNX export capabilities

### Model Family Classification for Web Platforms

The model family classifier has been enhanced to understand web platform compatibility:

```python
# Import the classifier
from model_family_classifier import classify_model

# Analyze with web platform compatibility
model_info = classify_model(
    model_name="prajjwal1/bert-tiny",
    model_class="BertModel",
    hw_compatibility={
        "webnn": {"compatible": True, "memory_usage": {"peak": 100}},
        "webgpu": {"compatible": True, "memory_usage": {"peak": 120}}
    }
)

# Check web platform compatibility
if model_info.get("subfamily") == "web_deployment":
    print("Model is optimized for web deployment")
```

## Best Practices for Web Deployment

1. **Use Small Models**: Choose efficient, small models for browser deployment
2. **Match Models to Platforms**: Use BERT-family models with WebNN, vision models with WebGPU
3. **Enable Fallbacks**: Always set `fallback_to_simulation: True` for graceful degradation
4. **Test in Simulation**: Use simulation mode during development and testing
5. **Consider Size Limits**: Set appropriate `max_model_size` for browser deployment
6. **Use Browser Optimizations**: Enable `browser_optimized: True` for web-specific optimizations
7. **Implement Robust Error Handling**: Be prepared for platforms not being available
8. **Optimize for Mobile**: Consider mobile browser constraints with lower memory limits
9. **Quantize When Possible**: Use quantized models where available for better performance
10. **Progressive Enhancement**: Implement basic CPU fallbacks for universal compatibility

## Future Enhancements

1. **WebGPU Compute Shaders**: Enhanced support for GPU compute operations
2. **Unified Web Priority Lists**: Smarter platform selection across model families
3. **Automatic Quantization**: On-the-fly quantization for web deployment
4. **Progressive Loading**: Partial model loading for faster startup times
5. **Browser Fingerprinting**: More detailed capability detection in browsers