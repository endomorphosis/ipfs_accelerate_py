# Web Platform Integration Quick Reference Guide

This quick reference provides key commands, configurations, and best practices for integrating machine learning models with the web platform using WebNN and WebGPU.

> **KEY POINT:** For audio models, Firefox provides ~20% better performance than Chrome with optimized workgroup size of 256x1x1.

## Key Recommendations by Model Type

*Use this table for quick decision-making on optimization strategy*

| Model Type | Recommended Optimizations | Best Browser | Key Settings |
|------------|---------------------------|--------------|-------------|
| **Audio** (Whisper, Wav2Vec2) | Compute Shaders | Firefox (20% faster) | `WEBGPU_COMPUTE_SHADERS_ENABLED=1`<br>`MOZ_WEBGPU_ADVANCED_COMPUTE=1`<br>Workgroup: 256x1x1 |
| **Vision** (ViT, ResNet) | Shader Precompilation | Chrome/Edge | `WEBGPU_SHADER_PRECOMPILE_ENABLED=1` |
| **Text** (BERT, T5) | Shader Precompilation | Chrome/Edge | `WEBGPU_SHADER_PRECOMPILE_ENABLED=1` |
| **Multimodal** (CLIP, LLaVA) | Parallel Loading<br>Shader Precompilation | Chrome/Edge | `WEB_PARALLEL_LOADING_ENABLED=1`<br>`WEBGPU_SHADER_PRECOMPILE_ENABLED=1` |
| **Large LLMs** (>1B params) | 4-bit Quantization<br>Progressive Loading | Chrome/Edge | Configure precision to 4-bit<br>Enable `progressive_loading=True` |

## Essential Code Snippets

### Browser Detection & Capability Check

```python
from fixed_web_platform.browser_capability_detector import BrowserCapabilityDetector

detector = BrowserCapabilityDetector()
capabilities = detector.get_capabilities()
browser = capabilities["browser_info"]["name"]
webgpu_available = capabilities["webgpu"]["available"]
```

### Audio Model with Firefox Optimization

```python
# Enable optimizations for audio models
os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"

# Firefox-specific optimizations (20% better audio performance)
if browser == "firefox":
    os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
    from fixed_web_platform.webgpu_audio_compute_shaders import optimize_for_firefox
    firefox_config = optimize_for_firefox({
        "model_name": "whisper",
        "workgroup_size": "256x1x1"
    })
```

### WebAssembly Fallback for Safari

```python
from fixed_web_platform.webgpu_wasm_fallback import dispatch_operation

if not webgpu_available:  # Safari or other browsers without WebGPU
    result = dispatch_operation(
        operation="matmul",
        inputs={"a": input_tensor, "b": weights},
        webgpu_available=False
    )
```

### Multimodal Model with Parallel Loading

```python
# Enable parallel loading for multimodal models
os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"

# Initialize with parallel loading
result = init_webgpu(
    model=model,
    model_name="clip-vit-base-patch32",
    parallel_loading=True,
    precompile_shaders=True
)
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| WebGPU not available | Use WebAssembly fallback |
| Safari low performance | Use WebAssembly with 8-bit precision |
| Memory constraints | Use 4-bit or 2-bit precision for large models |
| Slow first inference | Enable shader precompilation |
| Slow multimodal loading | Enable parallel loading |
| Firefox shader errors | Use compute shaders without precompilation |

## Command-Line Testing

```bash
# Test Firefox audio optimization
python test/test_web_platform_optimizations.py --compute-shaders --model whisper --browser firefox

# Test Safari with WebAssembly fallback
python test/test_web_platform_optimizations.py --browser safari --use-wasm-fallback

# Compare Firefox vs Chrome for audio models
python test/test_web_platform_optimizations.py --compare-browsers --models whisper,wav2vec2,clap

# Test with browser automation
./test/run_web_platform_tests.sh --use-browser-automation --browser firefox --enable-compute-shaders --model whisper
```

## Key Performance Facts

- **Firefox Audio Advantage**: ~20% faster for audio models (Whisper, Wav2Vec2, CLAP)
- **Shader Precompilation**: 30-45% faster first inference time
- **Parallel Loading**: 30-45% faster loading for multimodal models
- **4-bit Precision**: 75% memory reduction compared to FP16
- **WebAssembly Fallback**: 60-80% of WebGPU performance on Safari

## Memory Usage Guide by Model Size

| Model Size | Parameters | Standard | 4-bit | 2-bit | Recommended Approach |
|------------|------------|----------|-------|-------|---------------------|
| Tiny | <100M | <250MB | <120MB | <80MB | FP16 or 4-bit |
| Small | 100M-500M | 500MB | 250MB | 150MB | 4-bit |
| Medium | 500M-1B | 1GB | 400MB | 250MB | 4-bit |
| Large | 1B-7B | 2-4GB | 800MB-1.6GB | 500MB-1GB | 4-bit |
| X-Large | >7B | >8GB | 2-4GB | 1.2-2.5GB | 2-bit |

## Browser Compatibility

| Browser | WebNN | WebGPU | Best For | Limitations |
|---------|-------|--------|----------|-------------|
| Chrome/Edge | ✅ | ✅ | General purpose | None major |
| Firefox | ❌ | ✅ | Audio models | No WebNN, limited precompilation |
| Safari | ⚠️ | ⚠️ | Use WASM fallback | Limited WebGPU support |

## Firefox vs Chrome Audio Performance

| Audio Duration | Chrome | Firefox | Firefox Advantage |
|----------------|--------|---------|-------------------|
| 5 seconds | 83ms | 67ms | 19.3% |
| 15 seconds | 211ms | 169ms | 19.9% |
| 30 seconds | 407ms | 321ms | 21.1% |
| 60 seconds | 798ms | 622ms | 22.1% |

*Note: Advantage increases with longer audio samples*

## Model Setup Checklist

- [ ] Detect browser capabilities
- [ ] Select optimal backend (WebGPU or WebAssembly)
- [ ] Configure appropriate precision based on model size
- [ ] Apply model-specific optimizations (compute shaders, parallel loading)
- [ ] Apply browser-specific settings (Firefox workgroup size, Safari fallback)
- [ ] Enable shader precompilation for faster startup
- [ ] Monitor memory usage and performance
- [ ] Implement progressive loading for large models

For full details, refer to the [Web Platform Integration Guide](web_platform_integration_guide.md).