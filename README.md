# IPFS Accelerate Python

A Python framework for hardware-accelerated machine learning inference with IPFS network-based distribution and acceleration.

## Features

- **Hardware Acceleration**:
  - **Mojo/MAX Engine**: High-performance compilation and deployment (✅ **Real Integration**)
  - CPU optimization (x86, ARM) with SIMD vectorization
  - GPU acceleration (CUDA, ROCm, Metal)
  - Intel Neural Compute (OpenVINO)
  - Apple Silicon (MPS, Neural Engine)
  - WebNN/WebGPU for browser-based acceleration
  - Automatic hardware detection and optimization
  - Device-specific optimization and deployment

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

- **Model Context Protocol (MCP) Integration**:
  - ✅ **Production Ready**: Full JSON-RPC 2.0 protocol support  
  - ✅ **VS Code Compatible**: Works seamlessly with VS Code MCP extension
  - ✅ **8 Tools Available**: Hardware detection, IPFS operations, data processing
  - ✅ **Dual Protocol Support**: JSON-RPC 2.0 + HTTP REST endpoints
  - Real-time AI assistant integration for accelerated workflows

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

### Mojo/MAX Hardware Backend ✅

Experience real high-performance compilation and inference:

```bash
# Check Mojo/MAX integration status
python -c "
from src.backends.modular_backend import ModularEnvironment
env = ModularEnvironment()
print(f'🔥 Mojo available: {env.mojo_available}')
print(f'⚡ MAX available: {env.max_available}')
print(f'🖥️ Devices detected: {len(env.devices)}')
for i, device in enumerate(env.devices):
    print(f'  {i+1}. {device[\"name\"]} ({device[\"type\"]})')
"

# Run inference comparison test (100% PyTorch matching)
python test_real_inference_comparison.py

# Test HuggingFace model compatibility (367+ models supported)
python test_enhanced_huggingface_mojo_max.py --limit 5
```

**Real Integration Features**:
- ✅ **ModularEnvironment**: Hardware detection and capability assessment
- ✅ **MojoBackend**: Model compilation with O0/O1/O2/O3 optimization levels
- ✅ **MaxBackend**: Model deployment and high-performance inference serving
- ✅ **Inference Matching**: 100% output compatibility with PyTorch
- ✅ **Device Support**: CPU (AVX2/AVX-512), NVIDIA GPU, AMD GPU detection
- ✅ **Graceful Degradation**: Simulation mode when SDK unavailable

### MCP Server Integration ✅

Start the production-ready MCP server for AI assistant integration:

```bash
# Start the MCP server (fully operational with JSON-RPC 2.0)
python final_mcp_server.py --host 127.0.0.1 --port 8004 --debug

# Test server connectivity
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}' \
  http://127.0.0.1:8004/jsonrpc

# List available tools (8 tools registered)
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":2}' \
  http://127.0.0.1:8004/jsonrpc
```

**Features**:
- ✅ JSON-RPC 2.0 protocol for VS Code MCP extension
- ✅ 8 production tools: hardware detection, IPFS operations, data processing  
- ✅ Dual protocol support (JSON-RPC + HTTP REST)
- ✅ Enterprise-grade error handling and logging

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

- **Mojo/MAX Integration** ✅:
  - [Mojo Hardware Backends Documentation](docs/MOJO_HARDWARE_BACKENDS.md) - **Complete implementation guide**
  - [Modular Backend API Reference](src/backends/modular_backend.py) - Real backend implementation
  - [Test Suite Documentation](tests/mojo/) - Comprehensive testing infrastructure
  - [Performance Benchmarking](docs/MOJO_PERFORMANCE.md) - Hardware optimization guide

- **MCP Integration** ✅:
  - [MCP Connection Resolution Status](MCP_CONNECTION_RESOLUTION_STATUS.md) - **All issues resolved**
  - [MCP Integration Guide](IPFS_ACCELERATE_MCP_INTEGRATION_GUIDE.md) - Updated troubleshooting
  - [MCP Server README](mcp/README.md) - Updated architecture and usage
  - [Comprehensive MCP Guide](ipfs_accelerate_py/mcp/COMPREHENSIVE_MCP_GUIDE.md) - Complete documentation

- **General Documentation**:
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

- [Basic Usage](examples/basic_usage.py)
- [WebNN/WebGPU Demo](examples/demo_webnn_webgpu.py)
- [Multi-Model Pipeline](examples/multi_model_pipeline.py)
- [Hardware Benchmarking](examples/hardware_benchmark.py)
- [IPFS Content Addressing](examples/ipfs_content_addressing.py)

## License

This project is licensed under the [MIT License](LICENSE).