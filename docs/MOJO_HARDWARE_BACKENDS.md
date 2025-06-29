# Mojo Hardware Backends Documentation

## Overview

The IPFS Accelerate project includes comprehensive integration with Modular's Mojo programming language and MAX Engine for high-performance AI/ML model inference and compilation. This document provides detailed information about the hardware backend implementation, capabilities, and usage.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Hardware Backend Components](#hardware-backend-components)
3. [Installation Requirements](#installation-requirements)
4. [Device Detection and Support](#device-detection-and-support)
5. [Compilation Pipeline](#compilation-pipeline)
6. [Deployment and Inference](#deployment-and-inference)
7. [Performance Optimization](#performance-optimization)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Architecture Overview

The Mojo hardware backend is built around three core components:

```
┌─────────────────────┐
│   ModularBackend    │  ← Main integration interface
├─────────────────────┤
│ ModularEnvironment  │  ← Hardware detection & capabilities
│    MojoBackend      │  ← Compilation and optimization  
│    MaxBackend       │  ← Deployment and inference
└─────────────────────┘
```

### Key Features

- **Real Hardware Detection**: Automatic detection of CPU, GPU, and specialized AI accelerators
- **Dynamic Compilation**: Mojo code generation and compilation with multiple optimization levels
- **Inference Engine**: MAX Engine integration for high-performance model serving
- **Graceful Degradation**: Falls back to simulation when Modular SDK is not available
- **PyTorch Compatibility**: 100% inference output matching with PyTorch models

## Hardware Backend Components

### 1. ModularEnvironment

The `ModularEnvironment` class handles hardware detection and capability assessment:

```python
from src.backends.modular_backend import ModularEnvironment

env = ModularEnvironment()
print(f"Mojo available: {env.mojo_available}")
print(f"MAX available: {env.max_available}")
print(f"Detected devices: {len(env.devices)}")
```

**Capabilities:**
- Detects Mojo compiler installation
- Identifies MAX Engine availability
- Enumerates available compute devices (CPU, GPU, accelerators)
- Determines SIMD capabilities and optimization flags
- Reports Modular platform version

### 2. MojoBackend

The `MojoBackend` class provides model compilation services:

```python
from src.backends.modular_backend import MojoBackend

mojo = MojoBackend(env)
result = await mojo.compile_model(
    model_path="model.py",
    output_path="optimized_model.mojo.bin",
    optimization="O3",
    target_device="gpu"
)
```

**Features:**
- Dynamic Mojo code generation from PyTorch models
- Multiple optimization levels (O0, O1, O2, O3)
- SIMD vectorization and parallelization
- Target-specific optimization (CPU, GPU, specialized hardware)
- Comprehensive error handling and reporting

### 3. MaxBackend

The `MaxBackend` class manages model deployment and inference:

```python
from src.backends.modular_backend import MaxBackend

max_engine = MaxBackend(env)
await max_engine.initialize()

result = await max_engine.deploy_model(
    model_path="optimized_model.max",
    model_id="my_model",
    target_hardware="gpu"
)
```

**Features:**
- Model deployment with resource allocation
- High-performance inference serving
- Dynamic batching and optimization
- Endpoint management and scaling
- Performance monitoring and metrics

## Installation Requirements

### Option 1: With Full Modular SDK (Recommended for Production)

1. **Install Modular CLI**:
   ```bash
   curl -s https://get.modular.com | sh -
   ```

2. **Authenticate and Install Mojo/MAX**:
   ```bash
   modular auth
   modular install mojo
   modular install max
   ```

3. **Verify Installation**:
   ```bash
   modular --version
   mojo --version
   max --version
   ```

### Option 2: Development Mode (Simulation)

The backend works in simulation mode without the Modular SDK installed:

```python
# Will automatically detect absence of SDK and use simulation
env = ModularEnvironment()
# env.mojo_available = False
# env.max_available = False
```

### Common Installation Issues

**Wrong Mojo Installation**: If you see Juju-related errors, you have the Ubuntu snap "mojo" package:
```bash
# Remove wrong mojo
sudo snap remove mojo

# Install correct Modular Mojo (see Option 1 above)
```

**Permission Issues**: Ensure proper permissions for Modular installation:
```bash
# Fix common permission issues
sudo chown -R $USER:$USER ~/.modular
```

## Device Detection and Support

### Supported Hardware

| Device Type | Support Level | Features |
|-------------|---------------|----------|
| **CPU** | Full | SIMD optimization, multi-threading, vectorization |
| **NVIDIA GPU** | Full | CUDA acceleration, Tensor Cores, mixed precision |
| **AMD GPU** | Partial | ROCm acceleration, OpenCL fallback |
| **Intel GPU** | Planned | Intel XPU support via oneAPI |
| **Apple Silicon** | Partial | Metal Performance Shaders, Neural Engine |
| **Specialized AI** | Future | TPU, Habana Gaudi, other accelerators |

### Device Detection Code

```python
env = ModularEnvironment()

for device in env.devices:
    print(f"Device: {device['name']}")
    print(f"  Type: {device['type']}")
    print(f"  Cores: {device.get('cores', 'N/A')}")
    print(f"  Memory: {device.get('memory_mb', 'N/A')} MB")
    print(f"  Features: {device.get('supported_dtypes', [])}")
```

### CPU Capabilities

The system automatically detects CPU SIMD capabilities:

- **AVX-512**: 16-wide SIMD vectors
- **AVX2**: 8-wide SIMD vectors  
- **AVX**: 4-wide SIMD vectors
- **SSE2**: 2-wide SIMD vectors (minimum)

### GPU Detection

Automatic detection includes:

```python
# NVIDIA GPU detection
nvidia_devices = [d for d in env.devices if d.get('vendor') == 'nvidia']

# AMD GPU detection  
amd_devices = [d for d in env.devices if d.get('vendor') == 'amd']
```

## Compilation Pipeline

### Mojo Code Generation

The backend generates optimized Mojo code from PyTorch models:

```python
# Example generated Mojo code structure
struct ModelName:
    fn __init__(inout self):
        # Model initialization
        
    fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Optimized forward pass with SIMD vectorization
        
    @always_inline
    fn _process_vectorized(self, input: Tensor, inout output: Tensor):
        # Vectorized operations using Mojo's SIMD capabilities
```

### Optimization Levels

| Level | Optimizations | Use Case |
|-------|---------------|----------|
| **O0** | None | Debugging, development |
| **O1** | Basic | Quick compilation |
| **O2** | Vectorization, loop optimization | Production (default) |
| **O3** | Aggressive inlining, auto-parallelization | Maximum performance |

### Compilation Process

```python
# Complete compilation workflow
result = await mojo_backend.compile_model(
    model_path="pytorch_model.py",
    output_path="optimized_model.mojo.bin", 
    optimization="O2",
    target_device="gpu"
)

if result.success:
    print(f"Compilation time: {result.compilation_time}s")
    print(f"Optimizations: {result.optimizations_applied}")
    print(f"Output size: {os.path.getsize(result.compiled_path)} bytes")
```

## Deployment and Inference

### Model Deployment

```python
# Deploy compiled model
deployment = await max_backend.deploy_model(
    model_path="optimized_model.max",
    model_id="production_model_v1",
    target_hardware="gpu"
)

if deployment.success:
    print(f"Endpoint: {deployment.endpoint_url}")
    print(f"Resources: {deployment.allocated_resources}")
```

### Inference Serving

```python
# High-performance inference
async def run_inference(input_data):
    response = await http_client.post(
        deployment.endpoint_url,
        json={"inputs": input_data.tolist()}
    )
    return response.json()["outputs"]
```

### Performance Monitoring

```python
# Benchmark deployed model
benchmark = await backend.benchmark_modular(
    model_id="production_model_v1",
    workload_type="inference"
)

print(f"Throughput: {benchmark['throughput_tokens_per_sec']} tokens/sec")
print(f"Latency P95: {benchmark['latency_ms']['p95']} ms")
print(f"Memory usage: {benchmark['memory_usage_mb']} MB")
```

## Performance Optimization

### SIMD Vectorization

Mojo automatically applies SIMD optimizations:

```python
# Mojo vectorization example
@parameter
fn vectorized_op[simd_width: Int](idx: Int):
    let input_vec = input.load[simd_width](idx)
    let processed = input_vec * 2.0 + 1.0
    output.store[simd_width](idx, processed)

vectorize[vectorized_op, simd_width_of[DType.float32]()](
    input.num_elements()
)
```

### Memory Optimization

- **Zero-copy operations**: Direct tensor memory access
- **Memory pooling**: Reuse allocated buffers
- **Garbage collection**: Automatic memory management
- **NUMA awareness**: CPU affinity optimization

### Parallelization

- **Auto-parallelization**: Automatic parallel loop detection
- **Thread pooling**: Efficient thread management
- **Work stealing**: Dynamic load balancing
- **GPU kernels**: Optimized CUDA/ROCm kernels

## API Reference

### ModularBackend Class

```python
class ModularBackend:
    async def initialize() -> None
    async def compile_mojo(model_id: str, **kwargs) -> Dict[str, Any]
    async def deploy_max(model_id: str, **kwargs) -> Dict[str, Any] 
    async def benchmark_modular(model_id: str, **kwargs) -> Dict[str, Any]
    def get_hardware_info() -> Dict[str, Any]
    async def health_check() -> Dict[str, Any]
```

### Environment Detection

```python
class ModularEnvironment:
    mojo_available: bool
    max_available: bool
    modular_version: Optional[str]
    devices: List[Dict[str, Any]]
```

### Compilation Results

```python
@dataclass
class CompilationResult:
    success: bool
    compiled_path: Optional[str]
    compilation_time: Optional[float]
    optimizations_applied: Optional[List[str]]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]]
```

### Deployment Results

```python
@dataclass
class DeploymentResult:
    success: bool
    endpoint_url: Optional[str]
    model_id: Optional[str]
    target_hardware: Optional[str]
    allocated_resources: Optional[Dict[str, Any]]
    error_message: Optional[str]
```

## Troubleshooting

### Common Issues

**1. Mojo Not Found**
```
Error: Mojo compiler not found
Solution: Install Modular SDK or verify PATH
```

**2. Compilation Failures**
```
Error: Compilation failed with exit code 1
Solution: Check model compatibility and optimization level
```

**3. MAX Engine Unavailable**
```
Error: MAX runtime not available
Solution: Run `modular install max`
```

**4. GPU Not Detected**
```
Error: No CUDA devices found
Solution: Install NVIDIA drivers and CUDA toolkit
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('modular_backend').setLevel(logging.DEBUG)

env = ModularEnvironment()
# Detailed device detection logs will be shown
```

### Health Check

```python
backend = ModularBackend()
health = await backend.health_check()

if health["status"] != "healthy":
    print("Issues detected:")
    for component, status in health["components"].items():
        if status["status"] == "error":
            print(f"  {component}: {status['error']}")
```

## Examples

### Basic Usage

```python
import asyncio
from src.backends.modular_backend import ModularBackend

async def main():
    # Initialize backend
    backend = ModularBackend()
    await backend.initialize()
    
    # Get hardware info
    hw_info = backend.get_hardware_info()
    print(f"Mojo available: {hw_info['mojo_available']}")
    print(f"Devices: {len(hw_info['detected_devices'])}")
    
    # Compile model
    compilation = await backend.compile_mojo(
        model_id="my_transformer",
        optimization_level="O2"
    )
    
    if compilation["success"]:
        print(f"Compiled in {compilation['compilation_time']}s")
        
        # Deploy with MAX
        deployment = await backend.deploy_max(
            model_id="my_transformer",
            target_hardware="auto"
        )
        
        if deployment["success"]:
            print(f"Model deployed at: {deployment['endpoint_url']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### HuggingFace Integration

```python
from generators.models.mojo_max_support import MojoMaxTargetMixin

class MyModelSkill(MojoMaxTargetMixin):
    def __init__(self):
        super().__init__()
        self.device = self.get_default_device_with_mojo_max()
    
    def process(self, inputs):
        if self.device in ["mojo_max", "max", "mojo"]:
            return self.process_with_mojo_max(inputs, "BertModel")
        else:
            return self.process_with_pytorch(inputs)
```

### Performance Benchmarking

```python
async def benchmark_model():
    backend = ModularBackend()
    await backend.initialize()
    
    # Run comprehensive benchmark
    results = await backend.benchmark_modular(
        model_id="bert_base",
        workload_type="inference"
    )
    
    print("Performance Results:")
    print(f"  Throughput: {results['throughput_tokens_per_sec']} tokens/sec")
    print(f"  Latency P50: {results['latency_ms']['p50']} ms")
    print(f"  Latency P95: {results['latency_ms']['p95']} ms")
    print(f"  Memory usage: {results['memory_usage_mb']} MB")
    print(f"  Energy consumption: {results['energy_consumption_watts']} W")
    
    # Performance comparison
    comparison = results.get('comparison_vs_baseline', {})
    print(f"  Speedup vs PyTorch: {comparison.get('speedup', '1.0x')}")
    print(f"  Memory reduction: {comparison.get('memory_reduction', '0%')}")

asyncio.run(benchmark_model())
```

## Integration Testing

The project includes comprehensive test suites:

- **Unit Tests**: `tests/mojo/test_modular_backend.py`
- **Integration Tests**: `tests/e2e/test_mojo_integration.py`
- **Performance Tests**: `tests/benchmarks/test_mojo_performance.py`
- **Compatibility Tests**: `tests/integration/test_huggingface_mojo.py`

Run tests:
```bash
# Unit tests
pytest tests/mojo/ -v

# Integration tests  
pytest tests/e2e/ -v

# Full test suite
pytest tests/ -v --cov=src
```

## Contributing

To contribute to the Mojo backend:

1. **Setup Development Environment**:
   ```bash
   git clone https://github.com/endomorphosis/ipfs_accelerate
   cd ipfs_accelerate
   pip install -e .[dev]
   ```

2. **Run Tests**:
   ```bash
   python organize_project.py  # Organize project structure
   pytest tests/mojo/ -v       # Run Mojo-specific tests
   ```

3. **Code Style**:
   ```bash
   black src/backends/
   isort src/backends/
   mypy src/backends/
   ```

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

*Last updated: June 29, 2025*
*Documentation version: 1.0.0*
