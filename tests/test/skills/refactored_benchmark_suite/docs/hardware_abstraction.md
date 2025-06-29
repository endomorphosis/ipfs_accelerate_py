# Hardware Abstraction Layer

The Hardware Abstraction Layer (HAL) provides a consistent interface for benchmarking models across various hardware platforms. It handles hardware detection, initialization, and cleanup, allowing benchmarks to run on different devices transparently.

## Architecture

The HAL is built around the `HardwareBackend` base class, with specific implementations for different hardware types:

```
HardwareBackend
  ├── CPUBackend
  ├── CUDABackend
  ├── MPSBackend
  ├── ROCmBackend
  ├── OpenVINOBackend
  ├── WebNNBackend
  └── WebGPUBackend
```

Each backend provides consistent methods for hardware detection, information retrieval, initialization, and cleanup.

## Supported Hardware Platforms

The HAL currently supports the following hardware platforms:

- **CPU**: General-purpose CPU computing
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon GPUs using Metal Performance Shaders
- **ROCm**: AMD GPUs with ROCm support
- **OpenVINO**: Intel hardware accelerators via OpenVINO
- **WebNN**: Neural network acceleration via the WebNN API
- **WebGPU**: GPU acceleration via the WebGPU API

## Hardware Capabilities

In addition to basic hardware types, the HAL also supports hardware-specific capabilities:

- **cuda_tensor_cores**: NVIDIA Tensor Cores for accelerated matrix operations (available on Volta+ GPUs)

## Usage

### Basic Hardware Detection

```python
from refactored_benchmark_suite import hardware

# Get available hardware
available_hardware = hardware.get_available_hardware()
print(f"Available hardware: {available_hardware}")

# Get detailed hardware information
hw_info = hardware.get_hardware_info()
print(f"CUDA info: {hw_info['cuda_info'] if hw_info['cuda'] else 'Not available'}")
```

### Hardware Initialization

```python
from refactored_benchmark_suite import hardware

# Initialize CPU
cpu_device = hardware.initialize_hardware("cpu")

# Initialize CUDA (with specific device)
cuda_device = hardware.initialize_hardware("cuda", device_index=0)

# Initialize MPS
mps_device = hardware.initialize_hardware("mps")
```

### Direct Backend Access

```python
from refactored_benchmark_suite.hardware import CUDABackend

# Create CUDA backend
cuda_backend = CUDABackend(device_index=0)

# Initialize
device = cuda_backend.initialize()

# Run operations on the device
# ...

# Clean up
cuda_backend.cleanup()
```

## Hardware Fallbacks

The HAL implements automatic fallbacks for handling hardware unavailability:

1. If the requested hardware is not available, it will fall back to CPU
2. If CUDA is requested with a non-existent device index, it will fall back to device 0
3. If WebNN GPU is not available, it will fall back to WebNN CPU

## Hardware Information

Each backend provides detailed hardware information:

### CPU Information

- Processor model
- Architecture
- Physical and logical core count
- System information

### CUDA Information

- Device count and names
- Compute capabilities
- Memory information
- Multi-processor count
- CUDNN version and status

### MPS Information

- Apple Silicon detection
- Platform information
- CPU brand information

### ROCm Information

- Device count and names
- Memory information
- ROCm version

### OpenVINO Information

- Available devices
- Version information
- Device-specific capabilities

### WebNN Information

- Available backends
- Browser information
- WebNN version

### WebGPU Information

- Adapter information
- Features and limits
- Browser information

## Extending the HAL

To add support for a new hardware platform:

1. Create a new backend class inheriting from `HardwareBackend`
2. Implement the required methods: `is_available()`, `get_info()`, `initialize()`, and `cleanup()`
3. Add the backend to the `HARDWARE_BACKENDS` dictionary in `hardware/__init__.py`

```python
from hardware.base import HardwareBackend

class NewHardwareBackend(HardwareBackend):
    name = "new_hardware"
    
    @classmethod
    def is_available(cls) -> bool:
        # Implement availability check
        return False
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        # Implement info retrieval
        return {"available": cls.is_available()}
    
    def initialize(self) -> Any:
        # Implement initialization
        return None
    
    def cleanup(self) -> None:
        # Implement cleanup
        pass
```