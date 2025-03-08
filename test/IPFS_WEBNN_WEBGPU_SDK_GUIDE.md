# IPFS Acceleration with WebNN/WebGPU SDK Guide

This guide shows how to use the IPFS Accelerate Python SDK with WebNN and WebGPU hardware acceleration support. This integration combines the power of P2P-optimized content delivery with browser-based hardware acceleration.

## Installation

Ensure you have the latest version of the SDK installed:

```bash
pip install ipfs_accelerate_py>=0.3.0
```

## Basic Usage

The SDK provides an `accelerate()` function that combines IPFS content delivery with hardware acceleration:

```python
import ipfs_accelerate_py

# Initialize the SDK
sdk = ipfs_accelerate_py.ipfs_accelerate_py()

# Accelerate inference with WebGPU 
result = sdk.accelerate(
    model_name="bert-base-uncased",
    content="This is a test sentence",
    config={
        "platform": "webgpu",
        "browser": "chrome",
        "precision": 8,
        "mixed_precision": False
    }
)

# Print results
print(f"Processing time: {result['processing_time']:.3f} seconds")
print(f"Total time: {result['total_time']:.3f} seconds")
print(f"IPFS source: {result['ipfs_source']}")
print(f"P2P optimized: {result['p2p_optimized']}")
print(f"Memory usage: {result['memory_usage_mb']:.2f} MB")
print(f"Throughput: {result['throughput_items_per_sec']:.2f} items/sec")
```

## Browser-Specific Optimizations

Different browsers excel at different tasks. Use these browser-specific configurations for optimal performance:

### Firefox Audio Optimizations

Firefox provides exceptional performance for audio models with specialized compute shader workgroups:

```python
# Process audio with Firefox's optimized compute shaders
whisper_result = sdk.accelerate(
    model_name="whisper-tiny",
    content={"audio_path": "audio.mp3"},
    config={
        "platform": "webgpu",
        "browser": "firefox",
        "use_firefox_optimizations": True
    }
)
```

### Edge WebNN for Text Models

Edge provides the best WebNN implementation for text models:

```python
# Process text with Edge's WebNN implementation
bert_result = sdk.accelerate(
    model_name="bert-base-uncased",
    content="This is a test sentence",
    config={
        "platform": "webnn",
        "browser": "edge"
    }
)
```

### Chrome for Vision Models

Chrome provides excellent WebGPU performance for vision models:

```python
# Process images with Chrome's WebGPU implementation
vit_result = sdk.accelerate(
    model_name="vit-base-patch16-224",
    content={"image_path": "image.jpg"},
    config={
        "platform": "webgpu",
        "browser": "chrome"
    }
)
```

## Precision Control

The SDK supports various precision levels to balance performance and accuracy:

```python
# 4-bit quantization with mixed precision
low_precision_result = sdk.accelerate(
    model_name="bert-base-uncased",
    content="This is a test sentence",
    config={
        "platform": "webgpu",
        "precision": 4,
        "mixed_precision": True
    }
)

# 8-bit quantization (good balance)
balanced_result = sdk.accelerate(
    model_name="bert-base-uncased",
    content="This is a test sentence",
    config={
        "platform": "webgpu",
        "precision": 8
    }
)

# 16-bit quantization (higher accuracy)
high_precision_result = sdk.accelerate(
    model_name="bert-base-uncased",
    content="This is a test sentence",
    config={
        "platform": "webgpu",
        "precision": 16
    }
)
```

## Configuration Options

| Option | Description | Default | Values |
|--------|-------------|---------|--------|
| `platform` | Hardware acceleration platform | `"webgpu"` | `"webnn"`, `"webgpu"` |
| `browser` | Target browser | `"chrome"` | `"chrome"`, `"firefox"`, `"edge"`, `"safari"` |
| `is_real_hardware` | Whether real hardware is available | `False` | `True`, `False` |
| `precision` | Bit precision for computation | `8` | `4`, `8`, `16`, `32` |
| `mixed_precision` | Use mixed precision | `False` | `True`, `False` |
| `use_firefox_optimizations` | Use Firefox audio optimizations | `False` | `True`, `False` |

## Return Values

The `accelerate()` function returns a dictionary with detailed information:

```python
{
    "model_name": "bert-base-uncased",        # Model name
    "model_type": "text",                     # Model type (text, vision, audio, multimodal)
    "platform": "webgpu",                     # Hardware acceleration platform
    "browser": "chrome",                      # Browser used
    "is_real_hardware": True,                 # Whether real hardware was used
    "precision": 8,                           # Precision level
    "mixed_precision": False,                 # Whether mixed precision was used
    "processing_time": 0.025,                 # Model processing time in seconds
    "total_time": 0.134,                      # Total execution time in seconds
    "ipfs_cache_hit": True,                   # Whether model was found in cache
    "ipfs_source": "p2p",                     # Source of model (cache, p2p, ipfs)
    "ipfs_load_time": 100.5,                  # Load time in milliseconds
    "optimizations": ["4bit_optimization"],   # Applied optimizations
    "memory_usage_mb": 256.5,                 # Estimated memory usage
    "throughput_items_per_sec": 40.0,         # Items processed per second
    "p2p_optimized": True                     # Whether P2P optimization was used
}
```

## Browser Recommendations

For optimal performance, we recommend the following browser/platform combinations:

1. **Text Models (BERT, T5)**:
   - **Best**: Edge with WebNN
   - **Good**: Chrome with WebGPU

2. **Vision Models (ViT, CLIP)**:
   - **Best**: Chrome with WebGPU
   - **Good**: Firefox with WebGPU

3. **Audio Models (Whisper, Wav2Vec2)**:
   - **Best**: Firefox with WebGPU + audio optimizations
   - **Good**: Chrome with WebGPU

4. **Multimodal Models (LLaVA, CLIP)**:
   - **Best**: Chrome with WebGPU
   - **Good**: Firefox with WebGPU

## Advanced Features

### P2P Network Analytics

You can access P2P network analytics to understand content delivery performance:

```python
# Get P2P network analytics
p2p_analytics = sdk.get_p2p_network_analytics()

print(f"Peer count: {p2p_analytics['peer_count']}")
print(f"Network efficiency: {p2p_analytics['network_efficiency']:.2f}")
print(f"Optimization score: {p2p_analytics['optimization_score']:.2f}")
print(f"Network health: {p2p_analytics['network_health']}")
```

### Custom File Handling

Direct file handling for custom workflows:

```python
# Add a file to IPFS
add_result = sdk.add_file("model.onnx")
cid = add_result["cid"]

# Get a file from IPFS with P2P optimization
get_result = sdk.get_file(cid, output_path="model_retrieved.onnx", use_p2p=True)
```

## Benchmark Integration

The SDK integrates with the benchmarking system for performance testing:

```python
# Benchmark WebNN/WebGPU acceleration
from benchmark_ipfs_acceleration import benchmark_ipfs_webnn_webgpu

results = benchmark_ipfs_webnn_webgpu(
    models=["bert-base-uncased", "whisper-tiny", "vit-base-patch16-224"],
    platforms=["webnn", "webgpu"],
    browsers=["chrome", "firefox", "edge"],
    precisions=[4, 8, 16],
    db_path="./benchmark_db.duckdb"
)
```

## Testing

Test the integration using the provided test scripts:

```bash
# Simple integration test
python test_ipfs_accelerate_webnn_webgpu.py

# Comprehensive test with real browser automation
python test_ipfs_accelerate_with_real_webnn_webgpu.py --browser firefox --platform webgpu --model whisper-tiny --optimize-audio

# Test with all browsers and platforms
python test_ipfs_accelerate_with_real_webnn_webgpu.py --comprehensive
```

## Troubleshooting

### Common Issues

1. **Browser Automation Failures**:
   - Ensure you have the correct browser drivers installed
   - Check browser version compatibility
   - Use `--visible` flag to see browser automation

2. **WebNN/WebGPU Not Available**:
   - Check browser capabilities with `check_browser_webnn_webgpu.py`
   - Update your browser to the latest version
   - Enable WebGPU in browser flags if needed

3. **Performance Issues**:
   - Try different precision levels
   - Enable browser-specific optimizations
   - Monitor memory usage during execution

### Getting Help

For detailed information on WebNN/WebGPU integration:
- See [IPFS_WEBNN_WEBGPU_INTEGRATION.md](IPFS_WEBNN_WEBGPU_INTEGRATION.md)
- See [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md)
- See [REAL_WEBNN_WEBGPU_IMPLEMENTATION.md](REAL_WEBNN_WEBGPU_IMPLEMENTATION.md)