# Real WebNN/WebGPU Benchmarking Guide

## Introduction

This guide provides detailed instructions for running **real** WebNN and WebGPU benchmarks using actual browsers with hardware acceleration. The benchmarking system is designed to provide accurate performance measurements by using real browser instances rather than simulation.

## Key Features

- **Real Hardware Detection**: Automatically distinguishes between real hardware acceleration and simulation
- **Browser-Specific Optimizations**: Uses the optimal browser for each model type (Firefox for audio, Edge for WebNN, etc.)
- **March 2025 Optimizations**: Support for compute shader optimization, shader precompilation, and parallel loading
- **Database Integration**: Stores results in DuckDB for efficient analysis
- **Cross-Browser Testing**: Supports Chrome, Firefox, Edge, and Safari
- **Comprehensive Model Support**: Tests all high-priority model types across different hardware platforms
- **Advanced WebSocket Bridge**: Reliable communication between Python and browsers

## Prerequisites

Before running benchmarks, ensure you have the following installed:

```bash
# Install required packages
pip install selenium websockets duckdb

# Make sure you have at least one of these browsers installed:
# - Chrome
# - Firefox 
# - Microsoft Edge
# - Safari (limited support)
```

## Quick Start

The easiest way to run real WebNN/WebGPU benchmarks is using the `run_real_web_benchmarks.py` script:

```bash
# Run WebGPU benchmark with Chrome
python test/run_real_web_benchmarks.py --platform webgpu --browser chrome --model bert

# Run WebNN benchmark with Edge (best WebNN support)
python test/run_real_web_benchmarks.py --platform webnn --browser edge --model bert

# Run audio model with Firefox and compute shader optimization (Firefox is ~20% faster)
python test/run_real_web_benchmarks.py --model whisper --browser firefox --compute-shaders

# Run comprehensive benchmarks with optimal configurations for all models
python test/run_real_web_benchmarks.py --comprehensive
```

## Optimal Browser Selection

Different browsers have strengths with different model types and platforms:

| Model Type | Optimal Browser | Optimal Platform | Recommended Optimizations |
|------------|-----------------|------------------|---------------------------|
| Text (BERT, T5) | Edge/Chrome | WebNN/WebGPU | Shader Precompilation |
| Vision (ViT, CLIP) | Chrome | WebGPU | Shader Precompilation |
| Audio (Whisper, Wav2Vec2) | Firefox | WebGPU | Compute Shaders |
| Multimodal (CLIP) | Chrome | WebGPU | Parallel Loading, Shader Precompilation |

For automatic optimal selection, use:

```bash
python test/run_real_web_benchmarks.py --model bert --browser auto
```

## March 2025 Optimizations

Three major optimizations were added in March 2025:

### 1. Compute Shader Optimization

Significantly improves performance for audio models, especially in Firefox:

```bash
# Enable compute shader optimization
python test/run_real_web_benchmarks.py --model whisper --compute-shaders
```

This optimization:
- Uses optimized compute shader workgroups (256x1x1 in Firefox vs 128x2x1 in Chrome)
- Shows ~20-25% better performance in Firefox for audio models
- Improves integration between WebGPU and WebAudio contexts

### 2. Shader Precompilation

Reduces first-inference latency by precompiling WebGPU shaders:

```bash
# Enable shader precompilation
python test/run_real_web_benchmarks.py --model vit --shader-precompile
```

This optimization:
- Reduces first-inference latency by 30-45%
- Particularly effective for vision models
- Works across all browsers with WebGPU support

### 3. Parallel Model Loading

Accelerates loading time for multimodal models:

```bash
# Enable parallel model loading
python test/run_real_web_benchmarks.py --model clip --parallel-loading
```

This optimization:
- Reduces loading time by 30-45% for multimodal models
- Loads multiple model components simultaneously
- Particularly effective for models with separate encoders

Enable all optimizations at once with:

```bash
python test/run_real_web_benchmarks.py --all-optimizations
```

## Advanced Usage

### Database Integration

Store benchmark results in DuckDB for efficient analysis:

```bash
# Use database integration with custom path
python test/run_real_web_benchmarks.py --db-path ./benchmark_db.duckdb

# Store results only in database (no JSON files)
python test/run_real_web_benchmarks.py --db-only

# Use database path from environment variable
export BENCHMARK_DB_PATH=./benchmark_db.duckdb
python test/run_real_web_benchmarks.py
```

### Visualization Options

Generate visual reports of benchmark results:

```bash
# Generate visualizations
python test/run_real_web_benchmarks.py --visualize

# Select specific output format
python test/run_real_web_benchmarks.py --format html
```

### Advanced Testing Options

```bash
# Run multiple iterations for more reliable results
python test/run_real_web_benchmarks.py --runs 5

# Use smaller model variants for faster testing
python test/run_real_web_benchmarks.py --small-models

# Run with browser visible (not headless)
python test/run_real_web_benchmarks.py --visible

# Allow simulation if real hardware not available
python test/run_real_web_benchmarks.py --allow-simulation

# Run warmup iteration before benchmarking
python test/run_real_web_benchmarks.py --warmup
```

## Using the Low-Level API

For more control, you can use the lower-level API directly:

```python
import asyncio
from fixed_web_platform.browser_automation import BrowserAutomation
from fixed_web_platform.websocket_bridge import create_websocket_bridge

async def run_custom_benchmark():
    # Create WebSocket bridge
    bridge = await create_websocket_bridge(port=8765)
    if not bridge:
        print("Failed to create WebSocket bridge")
        return
    
    # Create browser automation
    automation = BrowserAutomation(
        platform="webgpu",
        browser_name="firefox",
        headless=False,
        compute_shaders=True,
        test_port=8765
    )
    
    # Launch browser
    success = await automation.launch()
    if not success:
        print("Failed to launch browser")
        await bridge.stop()
        return
    
    try:
        # Wait for connection
        connected = await bridge.wait_for_connection(timeout=30)
        if not connected:
            print("WebSocket connection timed out")
            return
        
        # Get browser capabilities
        capabilities = await bridge.get_browser_capabilities()
        print(f"Browser capabilities: {capabilities}")
        
        # Initialize model
        init_response = await bridge.initialize_model(
            model_name="whisper-tiny",
            model_type="audio",
            platform="webgpu",
            options={"precision": {"bits": 8}}
        )
        
        # Run inference
        input_data = {"audio": "test.mp3"}
        inference_response = await bridge.run_inference(
            model_name="whisper-tiny",
            input_data=input_data,
            platform="webgpu"
        )
        
        print(f"Inference result: {inference_response}")
        
    finally:
        # Clean up
        await automation.close()
        await bridge.stop()

# Run the benchmark
asyncio.run(run_custom_benchmark())
```

## ResourcePoolBridge Integration (March 2025)

For concurrent execution of models across multiple browser instances, use the ResourcePoolBridge:

```python
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridge

# Create resource pool bridge
bridge = ResourcePoolBridge(max_connections=4)

# Initialize with model configurations
await bridge.initialize([
    {
        'model_id': 'vision-model',
        'model_path': 'https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.onnx',
        'backend': 'webgpu',
        'family': 'vision'
    },
    {
        'model_id': 'text-model',
        'model_path': 'https://huggingface.co/bert-base-uncased/resolve/main/model.onnx',
        'backend': 'webnn',
        'family': 'text_embedding'
    }
])

# Run parallel inference
vision_result, text_result = await bridge.run_parallel([
    ('vision-model', {'input': image_data}),
    ('text-model', {'input_ids': text_data})
])

# Close when done
await bridge.close()
```

## Troubleshooting

### WebSocket Connection Issues

If you encounter WebSocket connection problems:

1. Check firewall settings that might block WebSocket connections
2. Ensure port 8765 (or your custom port) is available
3. Try running with `--visible` to see browser output
4. Check browser console for JavaScript errors

### Browser Detection Issues

If browsers are not detected correctly:

1. Ensure browser executables are in standard locations
2. Use `--browser` to specify a browser explicitly
3. Check browser version compatibility (WebGPU requires recent browsers)

### Hardware Acceleration Issues

If real hardware acceleration is not detected:

1. Ensure your browser and GPU support WebGPU/WebNN
2. Update GPU drivers to latest version
3. Check browser settings to ensure hardware acceleration is enabled
4. Run with `--allow-simulation` if real hardware is not available

### Other Common Issues

1. **Selenium errors**: Ensure you have the correct version of Selenium installed (`pip install selenium`)
2. **File access errors**: Ensure the script has permission to create and access temporary files
3. **Database errors**: Check DuckDB permissions and path
4. **Browser crashes**: Try updating your browser to the latest version

## Implementation Details

The benchmarking system consists of these key components:

1. **WebSocketBridge**: Handles communication between Python and the browser via WebSockets
2. **BrowserAutomation**: Manages browser launching and control with optimal settings
3. **ResourcePoolBridge**: Provides concurrent execution of models across multiple browser instances
4. **WebnnWebgpuBenchmarker**: High-level API for running complete benchmarks

## Best Practices

1. **Use browser-specific optimizations** - Firefox for audio models, Edge for WebNN
2. **Run multiple iterations** (`--runs 5`) for more stable results
3. **Enable appropriate optimizations** for each model type
4. **Store results in the database** for better analysis
5. **Run warmup iterations** for more accurate measurements
6. **Verify real hardware acceleration** before benchmarking
7. **Use small models** during development and testing
8. **Compare browsers** to find the best performance for each model

## Understanding Benchmark Results

The benchmark results include:

- **Latency (ms)**: Time for a single inference (lower is better)
- **Throughput (items/sec)**: Number of inferences per second (higher is better)
- **Memory (MB)**: Memory usage during inference
- **Is Real Implementation**: Whether real hardware acceleration was used
- **Browser**: Which browser was used
- **Platform**: WebNN or WebGPU
- **Optimization Flags**: Which optimizations were enabled

## Conclusion

Real WebNN/WebGPU benchmarking provides accurate performance measurements using actual browsers with hardware acceleration. By using the optimal browser and enabling appropriate optimizations, you can achieve significant performance improvements for different model types.

The integrated database storage and visualization tools help you analyze results and make informed decisions about which platforms and optimizations to use for your specific use case.