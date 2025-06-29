# IPFS Acceleration with WebNN/WebGPU Integration

This document provides an overview of the WebNN and WebGPU integration in the IPFS Accelerate Python package, which enables hardware-accelerated AI model inference on web platforms.

> **IMPORTANT NOTE**: All WebGPU/WebNN implementations will be moved to a dedicated `ipfs_accelerate_js` folder once all tests pass. Import paths and references in this document will be updated accordingly after the migration.

## Overview

The IPFS Accelerate package now integrates with WebNN and WebGPU, allowing hardware-accelerated AI model inference for various model types (text, vision, audio, multimodal) across different browsers (Chrome, Firefox, Edge, Safari) with optimization options including:

- Browser-specific optimizations (Firefox for audio, Edge for WebNN)
- Precision control (4-bit, 8-bit, 16-bit)
- Mixed precision support
- Real hardware detection vs. simulation

## Feature Highlights

1. **WebNN and WebGPU Support**: Hardware acceleration across modern browsers
2. **Browser-Specific Optimizations**:
   - Firefox audio optimizations (256x1x1 workgroup size for ~20-25% better performance)
   - Edge-optimized WebNN implementation 
   - Chrome with good all-around WebGPU support
3. **P2P-Optimized Content Delivery**: IPFS content delivery with peer-to-peer optimization
4. **Quantization Support**: 4-bit, 8-bit, and 16-bit precision with mixed precision options
5. **Real Hardware Detection**: Proper detection of real hardware implementations vs. simulation

## Usage

```python
import ipfs_accelerate_py

# Basic usage
result = ipfs_accelerate_py.accelerate(
    model_name="bert-base-uncased",
    content="Text to process with the model",
    config={
        "platform": "webgpu",  # or "webnn"
        "browser": "chrome",   # or "firefox", "edge", "safari"
        "is_real_hardware": True,  # Whether real hardware is available
        "precision": 8,        # Bit precision (4, 8, 16)
        "mixed_precision": False, 
        "use_firefox_optimizations": False
    }
)

# Firefox audio optimization for Whisper models
result = ipfs_accelerate_py.accelerate(
    model_name="whisper-tiny",
    content={"audio_path": "audio.mp3"},
    config={
        "platform": "webgpu",
        "browser": "firefox",
        "use_firefox_optimizations": True
    }
)

# Edge-optimized WebNN for text models
result = ipfs_accelerate_py.accelerate(
    model_name="bert-base-uncased",
    content="Text to process with the model",
    config={
        "platform": "webnn",
        "browser": "edge"
    }
)

# 4-bit quantization with mixed precision
result = ipfs_accelerate_py.accelerate(
    model_name="t5-small",
    content="Text to translate",
    config={
        "platform": "webgpu",
        "precision": 4,
        "mixed_precision": True
    }
)
```

## Configuration Options

| Option | Description | Default | Values |
|--------|-------------|---------|--------|
| `platform` | The hardware acceleration platform | `"webgpu"` | `"webnn"`, `"webgpu"` |
| `browser` | The browser to use | `"chrome"` | `"chrome"`, `"firefox"`, `"edge"`, `"safari"` |
| `is_real_hardware` | Whether real hardware is available | `False` | `True`, `False` |
| `precision` | Bit precision for computation | `8` | `4`, `8`, `16`, `32` |
| `mixed_precision` | Use mixed precision | `False` | `True`, `False` |
| `use_firefox_optimizations` | Use Firefox audio optimizations | `False` | `True`, `False` |

## Return Value

The `accelerate()` function returns a dictionary with the following information:

```python
{
    "model_name": "bert-base-uncased",
    "model_type": "text",  # "text", "vision", "audio", or "multimodal"
    "platform": "webgpu",
    "browser": "chrome",
    "is_real_hardware": True,
    "precision": 8,
    "mixed_precision": False,
    "processing_time": 0.025,  # Model processing time in seconds
    "total_time": 0.134,       # Total execution time in seconds
    "ipfs_cache_hit": True,    # Whether model was found in cache
    "ipfs_source": "p2p",      # Source of model: "cache", "p2p", or "ipfs"
    "ipfs_load_time": 100.5,   # Load time in milliseconds
    "optimizations": ["4bit_optimization"],  # Applied optimizations
    "memory_usage_mb": 256.5,  # Estimated memory usage
    "throughput_items_per_sec": 40.0,  # Items processed per second
    "p2p_optimized": True,     # Whether P2P optimization was used
}
```

## Browser-Specific Recommendations

For optimal performance, we recommend:

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

## Testing

A simple test script is provided to verify the integration:

```bash
python generators/models/test_ipfs_accelerate_webnn_webgpu.py
```

For comprehensive testing with real browser automation:

```bash
python generators/models/test_ipfs_accelerate_with_real_webnn_webgpu.py --browser firefox --platform webgpu --model whisper-tiny --optimize-audio
```

## Technical Details

### WebNN/WebGPU Detection

The system properly detects real WebNN and WebGPU implementations versus simulation:

- Chrome, Edge, Firefox: WebGPU supported with compute shader capabilities
- Edge: Best support for WebNN
- Firefox: Specialized audio optimizations for WebGPU

### Firefox Audio Optimizations

Firefox provides specialized optimizations for audio models using optimized compute shader workgroup sizes (256x1x1 vs Chrome's 128x2x1):

- ~20-25% better performance than Chrome for audio models
- Particularly effective for Whisper, Wav2Vec2, and CLAP models
- Enabled with `use_firefox_optimizations=True`

### Quantization Support

The system supports various precision levels:

- **4-bit**: Fastest, lowest memory usage, may reduce accuracy
- **8-bit**: Good balance of speed, memory usage, and accuracy 
- **16-bit**: Higher accuracy, more memory usage, slower
- **Mixed precision**: Uses higher precision for critical layers

## Upcoming Migration to ipfs_accelerate_js

All WebGPU/WebNN implementations will be moved from the current location to a dedicated `ipfs_accelerate_js` folder once all tests pass successfully. This migration will:

1. **Create a clearer separation between JavaScript and Python components**
2. **Provide a more intuitive structure for WebGPU/WebNN implementations**
3. **Make the codebase easier to navigate and maintain**
4. **Simplify future JavaScript SDK development**

### Import Path Changes After Migration

Current imports like:
```python
from fixed_web_platform.webgpu_audio_compute_shaders import optimize_for_firefox
```

Will change to:
```python
from ipfs_accelerate_js.webgpu_audio_compute_shaders import optimize_for_firefox
```

The main `accelerate()` function API will remain unchanged to ensure backward compatibility.

## Version History

- **0.4.0** - Planned migration to ipfs_accelerate_js (Coming Soon)
- **0.3.0** - Added WebNN/WebGPU integration
- **0.2.0** - Added P2P network optimization
- **0.1.0** - Initial release