# WebGPU and WebNN Quantization Troubleshooting Guide

This guide provides solutions to common issues encountered when using WebGPU and WebNN with different quantization levels for HuggingFace models.

## Common Issues and Solutions

### 1. Memory Errors with Low Precision

#### Symptoms
- Out of memory errors during model loading
- Browser tab crashes
- Model initialization fails with memory-related errors

#### Solutions
- **Increase precision for large models**: Use 8-bit instead of 4-bit for models >1B parameters
- **Enable mixed precision**: Critical layers should use higher precision
  ```python
  platform = UnifiedWebPlatform(
      model_name="large-model",
      model_type="text_generation",
      platform="webgpu",
      bits=4,
      mixed_precision=True  # Crucial for large models
  )
  ```
- **Reduce batch size**: Lower batch sizes require less memory
- **Enable progressive loading**: Load model components separately
  ```python
  platform = UnifiedWebPlatform(
      model_name="large-model",
      model_type="text_generation",
      platform="webgpu",
      bits=4,
      progressive_loading=True
  )
  ```
- **Try sharded loading**: Split model across multiple tabs if available

### 2. Accuracy Degradation

#### Symptoms
- Noticeably worse model outputs
- Garbage outputs or random tokens
- Classification errors significantly higher than baseline

#### Solutions
- **Increase precision**: Use 8-bit instead of 4-bit or 4-bit mixed instead of 2-bit
- **Identify critical layers**: Ensure attention and embedding layers use higher precision
  ```python
  precision_config = MixedPrecisionConfig(
      model_type="text",
      default_bits=4,
      attention_bits=8,    # Critical for performance
      embedding_bits=8,    # Important for token representation
      output_bits=8        # Ensures clean outputs
  )
  ```
- **Adjust block size**: Try larger block sizes for more stable quantization
  ```python
  config = setup_quantization_config(
      model_type="text",
      bits=4,
      block_size=256,  # Larger block size (default is 128)
      mixed_precision=True
  )
  ```
- **Verify model compatibility**: Some models are more sensitive to quantization

### 3. Browser Compatibility Issues

#### Symptoms
- Works in one browser but not another
- Features missing in certain browsers
- Performance varies significantly across browsers

#### Solutions
- **Chrome/Edge**: Best overall support, use for most models
  ```python
  config = setup_quantization_config(
      model_type="text",
      bits=4,
      optimize_for_browser="chrome"
  )
  ```
- **Firefox**: Superior for audio models with compute shaders
  ```python
  config = setup_quantization_config(
      model_type="audio",
      bits=8,
      optimize_for_browser="firefox"
  )
  ```
- **Safari**: Most limited, requires higher precision
  ```python
  config = setup_quantization_config(
      model_type="text",
      bits=8,  # Minimum for Safari
      mixed_precision=False,  # Safari WebGPU implementation is more limited
      conservative_mode=True  # Safer settings for Safari
  )
  ```
- **Feature detection**: Always check capabilities at runtime
  ```python
  from fixed_web_platform.browser_detection import detect_browser_capabilities
  
  capabilities = detect_browser_capabilities()
  if capabilities["webgpu"] and capabilities["advanced_quantization"]:
      bits = 4
  else:
      bits = 8  # Fallback to safer option
  ```

### 4. Performance Issues

#### Symptoms
- Quantization doesn't improve performance as expected
- Latency higher than anticipated
- Memory usage not significantly reduced

#### Solutions
- **Enable shader precompilation**: Faster startup and inference
  ```python
  platform = UnifiedWebPlatform(
      model_name="model-name",
      model_type="text",
      platform="webgpu",
      bits=4,
      shader_precompilation=True
  )
  ```
- **Use browser-specific optimizations**: Each browser has different optimal settings
- **Check WebGPU adapter type**: Performance varies by GPU
  ```python
  from fixed_web_platform.browser_detection import get_webgpu_adapter_info
  
  adapter_info = get_webgpu_adapter_info()
  print(f"Adapter: {adapter_info['name']}, Type: {adapter_info['type']}")
  ```
- **Try different quantization schemes**: Symmetric vs Asymmetric
  ```python
  config = setup_quantization_config(
      model_type="text",
      bits=4,
      scheme="asymmetric"  # Try "symmetric" if performance isn't good
  )
  ```
- **Optimize compute kernels**: Use specialized matrix multiplication kernels
  ```python
  from fixed_web_platform.webgpu_4bit_kernels import optimize_kernels_for_hardware
  
  config = setup_quantization_config(
      model_type="text",
      bits=4,
      mixed_precision=True
  )
  config = optimize_kernels_for_hardware(config)
  ```

### 5. Model-Specific Issues

#### Text Models (BERT, T5)
- Best Configuration: 4-bit mixed precision
- Common Issue: Attention mechanism needs higher precision
- Solution: Use 8-bit for attention layers, 4-bit elsewhere

#### Vision Models (ViT, DETR)
- Best Configuration: 8-bit uniform
- Common Issue: Visual artifacts with 4-bit
- Solution: Stick with 8-bit or use very careful 4-bit mixed precision

#### Audio Models (Whisper, Wav2Vec2)
- Best Configuration: 8-bit with Firefox compute shaders
- Common Issue: Audio quality degradation with low precision
- Solution: Use Firefox with 8-bit and optimized audio compute shaders

#### Large Language Models (LLaMA, Qwen2)
- Best Configuration: 4-bit mixed with KV cache optimization
- Common Issue: Limited context length and memory errors
- Solution: KV cache optimization + 4-bit mixed precision

## Advanced Debugging

### WebGPU Debugging Tools

```python
from fixed_web_platform.webgpu_debug import WebGPUDebugger

# Create a debugger instance
debugger = WebGPUDebugger()

# Capture trace of quantized operations
with debugger.trace() as trace:
    platform = UnifiedWebPlatform(
        model_name="bert-base-uncased",
        model_type="text",
        platform="webgpu",
        bits=4,
        mixed_precision=True
    )
    result = platform.run_inference({"input_text": "Sample text"})

# Analyze the trace
debugger.analyze_trace(trace)
debugger.analyze_memory_usage(trace)
debugger.analyze_shader_performance(trace)

# Export trace for browser debugging tools
debugger.export_trace("webgpu_trace.json")
```

### Precision Impact Analysis

```python
from fixed_web_platform.precision_analysis import analyze_precision_impact

# Analyze the impact of different precision levels on specific layers
results = analyze_precision_impact(
    model_name="bert-base-uncased",
    model_type="text",
    test_input={"input_text": "Sample text"},
    precision_configurations=[
        {"all": 16},                         # Baseline FP16
        {"all": 8},                          # INT8 all layers
        {"all": 4},                          # INT4 all layers
        {"attention": 8, "others": 4},       # Mixed precision
        {"attention": 8, "feedforward": 2, "others": 4}  # Ultra mixed
    ]
)

# Print layer-by-layer analysis
for layer_name, layer_results in results["layer_analysis"].items():
    print(f"Layer: {layer_name}")
    for precision, metrics in layer_results.items():
        print(f"  Precision {precision}: Accuracy {metrics['accuracy']:.4f}, Speed {metrics['relative_speed']:.2f}x")
```

## Browser-Specific Optimization Guide

### Chrome/Edge
- 4-bit mixed precision works well
- Shader precompilation provides significant speedup
- Excellent WebGPU compatibility across features

### Firefox
- Superior for audio models with compute shaders
- 8-bit precision for maximum stability
- Use 256x1x1 workgroup size for optimal performance

### Safari
- Limited WebGPU implementation
- 8-bit minimum recommended
- Be cautious with advanced features
- Test thoroughly on actual devices