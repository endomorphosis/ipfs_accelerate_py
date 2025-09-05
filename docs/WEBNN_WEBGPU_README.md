# WebNN/WebGPU Integration with IPFS Acceleration

This module provides a comprehensive integration between WebNN/WebGPU browser-based acceleration
and IPFS content delivery for efficient model inference.

## Key Features

- **Browser Hardware Acceleration**: Leverage browser WebNN and WebGPU APIs for hardware-accelerated machine learning
- **IPFS Content Delivery**: Use IPFS content addressing for efficient model storage and retrieval
- **Optimal Browser Selection**: Automatically select the best browser for different model types
- **Precision Control**: Support for 4/8/16/32-bit precision with mixed-precision options
- **Resource Pooling**: Efficient browser connection management for concurrent inference
- **Hardware-specific Optimizations**:
  - Firefox for audio models (better compute shader performance)
  - Edge for text models (best WebNN support)
  - Chrome for vision models (excellent WebGPU support)
- **Database Integration**: Comprehensive metrics tracking with DuckDB
- **Cross-platform Support**: Works on Windows, macOS, and Linux

## Installation

### Prerequisites

- Python 3.8+ with asyncio support
- Working IPFS node (optional, but recommended)
- WebSockets support: `pip install websockets`
- Database support (optional): `pip install duckdb`
- Browser automation (optional): `pip install playwright` or `pip install selenium`

### Install

```bash
# For basic functionality
pip install ipfs_accelerate_py

# For WebNN/WebGPU features with browser support
pip install -e ".[webnn]"
```

## Quick Start

```python
from ipfs_accelerate_py.webnn_webgpu_integration import accelerate_with_browser

# Run inference with WebGPU
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    platform="webgpu",
    browser="chrome",
    precision=16
)

print(f"Inference time: {result['inference_time']:.3f}s")
print(f"Output: {result['output']}")
```

## Browser Integration

The WebNN/WebGPU integration supports both a simulation mode and a real browser mode:

### Simulation Mode

In simulation mode, the system simulates hardware acceleration without launching a real browser.
This is useful for testing and development.

```python
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    use_real_browser=False  # Use simulation mode
)
```

### Real Browser Mode

In real browser mode, the system launches an actual browser instance and uses WebNN/WebGPU
APIs for hardware acceleration. This requires additional dependencies (playwright or selenium).

```python
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    use_real_browser=True  # Use a real browser
)
```

## Configuration Options

### Platform Selection

Choose between WebNN and WebGPU based on your model's needs:

```python
# Use WebGPU for graphics-intensive models
result = accelerate_with_browser(
    model_name="vit-base-patch16-224",
    inputs=pixel_values,
    platform="webgpu"
)

# Use WebNN for neural network models
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs,
    platform="webnn"
)
```

### Browser Selection

Automatically select the optimal browser or specify one:

```python
# Let the system choose the best browser for the model type
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs
)

# Specify a browser
result = accelerate_with_browser(
    model_name="whisper-small",
    inputs=audio_inputs,
    browser="firefox"  # Firefox has better audio performance
)
```

### Precision Control

Control the precision of computations:

```python
# Use 16-bit precision (default)
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs,
    precision=16
)

# Use 4-bit precision for higher performance
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs,
    precision=4
)

# Use mixed precision for better accuracy/performance tradeoff
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs,
    precision=16,
    mixed_precision=True
)
```

### Performance Optimizations

Various optimizations can be enabled:

```python
# Use compute shader optimizations (good for audio models)
result = accelerate_with_browser(
    model_name="whisper-small",
    inputs=audio_inputs,
    compute_shaders=True
)

# Use shader precompilation (reduces startup time)
result = accelerate_with_browser(
    model_name="vit-base-patch16-224",
    inputs=pixel_values,
    precompile_shaders=True
)

# Enable parallel loading (good for multimodal models)
result = accelerate_with_browser(
    model_name="clip-vit-base-patch32",
    inputs=multimodal_inputs,
    parallel_loading=True
)
```

## Browser Bridge

The browser bridge provides communication between Python and the browser environment:

```python
from ipfs_accelerate_py.browser_bridge import create_browser_bridge

# Create and start browser bridge
bridge = await create_browser_bridge(browser_name="chrome", headless=True)

# Get browser capabilities
capabilities = await bridge.get_browser_capabilities()
print(f"WebGPU support: {capabilities.get('webgpu')}")
print(f"WebNN support: {capabilities.get('webnn')}")

# Run inference
result = await bridge.request_inference(
    model="bert-base-uncased",
    inputs={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    model_type="text_embedding"
)

# Stop browser
await bridge.stop()
```

## IPFS Integration

The system integrates with IPFS for content-addressed model storage and retrieval:

```python
# Enable IPFS acceleration
result = accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs,
    enable_ipfs=True
)

# Check if IPFS was used
if result["ipfs_accelerated"]:
    print(f"Model CID: {result['cid']}")
    print(f"Cache hit: {result['ipfs_cache_hit']}")
```

## Database Integration

Store and analyze results with DuckDB integration:

```python
# Enable database storage
accelerator = get_accelerator(db_path="benchmark_results.duckdb")

# Run inference with DB storage
result = await accelerator.accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs
)

# Query results (example)
# conn = duckdb.connect("benchmark_results.duckdb")
# results = conn.execute("SELECT * FROM webnn_webgpu_results").fetchall()
```

## Advanced Usage

### Custom Accelerator Configuration

```python
from ipfs_accelerate_py.webnn_webgpu_integration import get_accelerator

# Create custom accelerator
accelerator = get_accelerator(
    db_path="results.duckdb",
    max_connections=8,
    headless=True,
    enable_ipfs=True
)

# Use the accelerator
result = await accelerator.accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs=text_inputs
)

# Close resources when done
accelerator.close()
```

### Benchmark Multiple Models

```python
async def benchmark_models():
    models = [
        "bert-base-uncased",
        "vit-base-patch16-224",
        "whisper-small"
    ]
    
    results = {}
    for model in models:
        result = await accelerate_with_browser(
            model_name=model,
            inputs=create_inputs_for_model(model),
            platform="webgpu"
        )
        results[model] = result
    
    return results
```

## Debugging

Enable logging for more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Supported Model Types

The integration supports various model types with optimized configurations:

- **Text Embedding**: BERT, RoBERTa, MPNet, etc.
- **Text Generation**: LLaMA, GPT, Phi, Mistral, etc.
- **Text-to-Text**: T5, mT5, BART, etc.
- **Vision**: ViT, CLIP, DETR, etc.
- **Audio**: Whisper, Wav2Vec2, CLAP, etc.
- **Multimodal**: LLaVA, BLIP, Fuyu, etc.

## Example Applications

Check the `examples` directory for sample applications:

- `demo_webnn_webgpu.py`: Command-line demo with benchmarking capabilities
- (More examples coming soon)

## Troubleshooting

### Common Issues

1. **Browser not found**: Ensure the browser is installed and in a standard location
2. **WebGPU not available**: Use a compatible browser (Chrome 113+, Edge 113+, Firefox 116+)
3. **WebNN not available**: Use Chrome or Edge (Firefox has limited WebNN support)
4. **CORS errors**: When using real browsers, ensure your server allows cross-origin requests

### Browser Compatibility

| Browser | WebGPU Support | WebNN Support | Audio Performance | Text Performance | Vision Performance |
|---------|---------------|--------------|------------------|-----------------|-------------------|
| Chrome  | Excellent     | Good         | Good             | Good            | Excellent         |
| Firefox | Good          | Limited      | Excellent        | Limited         | Good              |
| Edge    | Excellent     | Excellent    | Good             | Excellent       | Good              |
| Safari  | Limited       | Experimental | Limited          | Limited         | Limited           |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.