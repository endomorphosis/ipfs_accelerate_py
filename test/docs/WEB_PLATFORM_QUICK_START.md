# Web Platform Quick Start Guide (March 2025)

This guide helps you quickly get started with running machine learning models in web browsers using our optimized web platform integration.

## 5-Minute Quick Start

### Step 1: Choose the right configuration for your model

Select the appropriate configuration based on your model type:

```python
from fixed_web_platform.browser_capability_detector import BrowserCapabilityDetector

# Detect browser capabilities
detector = BrowserCapabilityDetector()
capabilities = detector.get_capabilities()
browser = capabilities["browser_info"]["name"]

# Configure based on model type
if model_type == "audio":  # Whisper, Wav2Vec2, CLAP
    # Firefox has ~20% better performance for audio
    os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
    if browser == "firefox":
        os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"

elif model_type == "multimodal":  # CLIP, LLaVA
    # Enable parallel loading for multimodal models
    os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
    os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"

else:  # Text/Vision models (BERT, ViT, etc.)
    # Shader precompilation for faster first inference
    os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
```

### Step 2: Initialize the model with optimizations

```python
# Initialize with appropriate optimizations
result = init_webgpu(
    model=model,
    model_name=model_name,
    web_api_mode="simulation",  # or "browser" in actual web environment
    compute_shaders=model_type == "audio",
    parallel_loading=model_type == "multimodal",
    precompile_shaders=True  # beneficial for all models except in Safari
)
```

### Step 3: Handle browser compatibility with fallback

```python
# Use WebAssembly fallback for browsers without WebGPU (e.g., Safari)
if not capabilities["webgpu"]["available"]:
    from fixed_web_platform.webgpu_wasm_fallback import dispatch_operation
    
    # Dispatch to WebAssembly fallback
    result = dispatch_operation(
        operation="inference",
        inputs={"input_data": input_data},
        webgpu_available=False
    )
```

## Model Setup Examples

### Audio Model (Whisper) with Firefox Optimization

```python
import os
from fixed_web_platform.browser_capability_detector import BrowserCapabilityDetector
from fixed_web_platform.webgpu_audio_compute_shaders import optimize_for_firefox

# Detect browser
detector = BrowserCapabilityDetector()
browser = detector.get_capabilities()["browser_info"]["name"]

# Enable optimizations (Firefox has 20% better audio performance)
os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
if browser == "firefox":
    os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
    firefox_config = optimize_for_firefox({
        "model_name": "whisper-tiny",
        "workgroup_size": "256x1x1"
    })
    audio_processor = firefox_config["processor"]

# Initialize WebGPU with optimizations
model = WhisperModel("whisper-tiny")  # Your model loading code
result = init_webgpu(
    model=model,
    compute_shaders=True,
    precompile_shaders=True
)

# Process audio
transcription = model.transcribe("audio.mp3")
```

### Large Language Model with 4-bit Quantization

```python
from fixed_web_platform.webgpu_ultra_low_precision import configure_precision

# Configure 4-bit precision for large model
precision_config = configure_precision(
    model_name="llama-7b",
    default_bits=4,  # Use 4-bit for most layers
    attention_bits=8,  # Higher precision for attention
    feed_forward_bits=4  # 4-bit for feed forward
)

# Initialize with 4-bit precision
model = LLaMAModel("llama-7b")  # Your model loading code
result = init_webgpu(
    model=model,
    precision_config=precision_config,
    progressive_loading=True  # Load components progressively
)

# Generate text
response = model.generate("Tell me about WebGPU")
```

### Multimodal Model with Parallel Loading

```python
# Enable parallel loading for multimodal models
os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"

# Initialize with parallel loading
model = CLIPModel("clip-vit-base-patch32")  # Your model loading code
result = init_webgpu(
    model=model,
    parallel_loading=True,
    precompile_shaders=True
)

# Process image and text
similarity = model.compute_similarity("image.jpg", "A description of the image")
```

## Browser-Specific Optimizations

Use these browser-specific optimizations for optimal performance:

### Firefox
- For audio models, use 256x1x1 workgroup size
- Enable MOZ_WEBGPU_ADVANCED_COMPUTE environment variable
- Use compute shaders without precompilation if there are errors

### Chrome/Edge
- Standard configuration works well for most models
- Use shader precompilation for faster first inference
- Enable parallel loading for multimodal models

### Safari
- Use WebAssembly fallback with 8-bit precision
- Enable progressive loading to reduce memory pressure
- Avoid compute shaders and shader precompilation

## Command-Line Testing

Use these commands to test your web platform implementation:

```bash
# Test Firefox audio optimization
python test/test_web_platform_optimizations.py --compute-shaders --model whisper --browser firefox

# Test Safari with WebAssembly fallback
python test/test_web_platform_optimizations.py --browser safari --use-wasm-fallback

# Test memory efficiency with 4-bit quantization
python test/test_web_platform_optimizations.py --model llama --precision 4 --progressive-loading

# Compare browser performance
python test/test_web_platform_optimizations.py --compare-browsers --models bert,whisper,clip
```

## Next Steps and Additional Resources

For more detailed information, refer to these guides:

- **[Web Platform Integration Guide](WEB_PLATFORM_INTEGRATION_GUIDE.md)** - Comprehensive guide for integrating web platform capabilities
- **[WebGPU Implementation Guide](WEBGPU_IMPLEMENTATION_GUIDE.md)** - Detailed guide to WebGPU integration
- **[WebGPU Shader Precompilation Guide](WEBGPU_SHADER_PRECOMPILATION.md)** - Guide to 30-45% faster first inference with shader precompilation
- **[Browser-Specific Optimizations](browser_specific_optimizations.md)** - Tailored configurations for different browsers
- **[Firefox Audio Optimization Guide](WEB_PLATFORM_FIREFOX_AUDIO_GUIDE.md)** - Special optimizations for audio models in Firefox (~20% better performance)
- **[Developer Tutorial](DEVELOPER_TUTORIAL.md)** - Step-by-step tutorials with complete working examples
- **[Error Handling Guide](ERROR_HANDLING_GUIDE.md)** - Comprehensive error handling strategies

For a complete list of all documentation, see the [Documentation Index](DOCUMENTATION_INDEX.md).