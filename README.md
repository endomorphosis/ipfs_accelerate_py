# IPFS Accelerate Python

A Python framework for hardware-accelerated machine learning inference with IPFS network-based distribution and acceleration.

## Features

- **Hardware Acceleration**:
  - CPU optimization (x86, ARM)
  - GPU acceleration (CUDA, ROCm)
  - Intel Neural Compute (OpenVINO)
  - Apple Silicon (MPS)
  - WebNN/WebGPU for browser-based acceleration
  - Automatic hardware detection and optimization

- **IPFS Integration**:
  - Content-addressed model storage and distribution
  - Efficient caching and retrieval
  - P2P content distribution
  - Reduced bandwidth for frequently used models

- **Model Support**:
  - Text generation models
  - Embedding/encoding models
  - Vision models
  - Audio models
  - Multimodal models

- **Framework Compatibility**:
  - HuggingFace Transformers
  - PyTorch
  - ONNX
  - Custom model formats

- **Browser Integration**:
  - WebNN hardware acceleration
  - WebGPU acceleration
  - Browser-specific optimizations
  - Cross-browser model sharding
  - Cross-model tensor sharing

## Installation

```bash
# Basic installation
pip install ipfs_accelerate_py

# With WebNN/WebGPU support
pip install ipfs_accelerate_py[webnn]

# With visualization tools
pip install ipfs_accelerate_py[viz]

# Full installation with all dependencies
pip install ipfs_accelerate_py[all]
```

## Quick Start

### Basic Usage

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Initialize with default settings
accelerator = ipfs_accelerate_py({}, {})

# Get optimal hardware backend for a model
optimal_backend = accelerator.get_optimal_backend("bert-base-uncased", "text_embedding")

# Run inference with automatic hardware selection
result = accelerator.run_model(
    "bert-base-uncased",
    {"input_ids": [101, 2054, 2003, 2026, 2171, 2024, 2059, 2038, 102]},
    "text_embedding"
)

# Access the output
embedding = result["embedding"]
```

### WebNN/WebGPU Acceleration

```python
from ipfs_accelerate_py import accelerate_with_browser

# Run inference with WebGPU in browser
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

### Custom Configuration

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Initialize with custom config
config = {
    "ipfs": {
        "gateway": "http://localhost:8080/ipfs/",
        "local_node": "http://localhost:5001",
        "timeout": 30
    },
    "hardware": {
        "prefer_cuda": True,
        "allow_openvino": True,
        "precision": "fp16",
        "mixed_precision": True
    },
    "db_path": "benchmark_results.duckdb"
}

accelerator = ipfs_accelerate_py(config, {})

# Accelerate model with custom configuration
result = accelerator.run_model(
    "llama-7b",
    {"prompt": "Explain quantum computing in simple terms"},
    "text_generation",
    max_length=100
)
```

## Documentation

For detailed documentation on all components and features, please refer to:

- [General Usage Guide](docs/USAGE.md)
- [WebNN/WebGPU Integration](WEBNN_WEBGPU_README.md)
- [API Reference](docs/API.md)
- [Hardware Optimization](docs/HARDWARE.md)
- [IPFS Integration](docs/IPFS.md)
- [Examples](examples/README.md)

## Browser Integration

The IPFS Accelerate Python framework provides comprehensive browser integration for hardware-accelerated inference:

```python
from ipfs_accelerate_py import get_accelerator

# Create an accelerator with WebNN/WebGPU support
accelerator = get_accelerator(enable_ipfs=True)

# Run vision model on WebGPU
result = await accelerator.accelerate_with_browser(
    model_name="vit-base-patch16-224",
    inputs={"pixel_values": image_tensor},
    model_type="vision",
    platform="webgpu",
    browser="chrome",
    precision=16
)

# Run text model on WebNN
result = await accelerator.accelerate_with_browser(
    model_name="bert-base-uncased",
    inputs={"input_ids": token_ids},
    model_type="text_embedding",
    platform="webnn",
    browser="edge",
    precision=16
)
```

For more information on WebNN/WebGPU integration, see the [WebNN/WebGPU README](WEBNN_WEBGPU_README.md).

## Benchmarking and Optimization

Measure performance across hardware platforms and get optimization recommendations:

```python
from ipfs_accelerate_py.benchmark import run_benchmark
from ipfs_accelerate_py.optimization import get_optimization_recommendations

# Run benchmark across all available hardware
results = run_benchmark(
    model_name="bert-base-uncased",
    inputs={"input_ids": [101, 2054, 2003, 2026, 2171, 102]},
    model_type="text_embedding",
    hardware=["cpu", "cuda", "openvino", "webgpu"],
    batch_sizes=[1, 8, 32],
    precision=["fp32", "fp16"],
    num_runs=5
)

# Generate visualization
results.to_visualization("benchmark_results.html")

# Get hardware-specific optimization recommendations
recommendations = get_optimization_recommendations(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    batch_size=8,
    current_precision="fp32"
)

# Export optimization recommendations
from test.optimization_recommendation.optimization_exporter import OptimizationExporter

exporter = OptimizationExporter(output_dir="./optimizations")
export_result = exporter.export_optimization(
    model_name="bert-base-uncased",
    hardware_platform="cuda"
)

# Create ZIP archive of exported files
archive_data = exporter.create_archive(export_result)
with open("optimization_exports.zip", "wb") as f:
    f.write(archive_data.getvalue())
```

## Examples

The `examples` directory contains practical examples for various use cases:

- [WebNN/WebGPU Demo](examples/demo_webnn_webgpu.py) - Browser-based acceleration with WebNN and WebGPU
- [Transformers Integration](examples/transformers_example.py) - HuggingFace Transformers integration
- [MCP Integration](examples/mcp_integration_example.py) - Model Control Protocol integration

Additional examples are available in the benchmarks directory for performance testing and hardware optimization.

## License

This project is licensed under the [MIT License](LICENSE).