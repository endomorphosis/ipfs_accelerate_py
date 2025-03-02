# Multi-GPU Support and Custom Device Mapping

This document explains how to use the multi-GPU support and custom device mapping features in the IPFS Accelerate Python framework.

## Overview

The multi-GPU support system provides comprehensive functionality for distributing models across multiple GPUs, enabling efficient inference and deployment of large language models and other deep learning models that may not fit on a single GPU.

Key features include:

1. Hardware detection for CUDA, ROCm, and Apple Silicon GPUs
2. Custom device mapping for model layers across multiple GPUs
3. Automated memory optimization for large models
4. Container deployment with multi-GPU support
5. Tensor parallel and pipeline parallel configuration
6. AMD ROCm hardware support alongside NVIDIA CUDA
7. Memory estimation for optimized device allocation
8. Docker container integration for production deployments

## Core Components

The implementation consists of several key components:

1. **DeviceMapper** (`utils/device_mapper.py`): Core class for hardware detection and device mapping
2. **Multi-GPU Utilities** (`utils/multi_gpu_utils.py`): High-level functions for model loading and pipeline creation
3. **Container Deployment** (`deploy_multi_gpu_container.py`): Script for deploying models in containers with multi-GPU support

## Device Mapping Strategies

The framework supports multiple device mapping strategies:

1. **Auto**: Intelligently distributes model parts across GPUs based on memory constraints
2. **Balanced**: Evenly distributes model layers across available GPUs
3. **Sequential**: Fills one GPU before moving to the next

## Usage Examples

### Loading a Model with Device Mapping

```python
from utils.multi_gpu_utils import load_model_with_device_map

# Auto strategy - automatically distributes the model
model, device_map = load_model_with_device_map(
    model_id="facebook/opt-6.7b",
    strategy="auto"
)

# Balanced strategy with specific devices
model, device_map = load_model_with_device_map(
    model_id="facebook/opt-6.7b",
    strategy="balanced",
    devices=["cuda:0", "cuda:1"]
)
```

### Creating an Optimized Pipeline

```python
from utils.multi_gpu_utils import create_optimized_pipeline

# Create a text generation pipeline with auto device mapping
pipe = create_optimized_pipeline(
    model_id="facebook/opt-1.3b",
    pipeline_type="text-generation",
    strategy="auto"
)

# Generate text with the pipeline
result = pipe("Once upon a time", max_length=100)
```

### Tensor Parallel Loading (for VLLM)

```python
from utils.multi_gpu_utils import load_model_with_tensor_parallel

# Load a model with tensor parallelism
model = load_model_with_tensor_parallel(
    model_id="facebook/opt-6.7b",
    tensor_parallel_size=2
)
```

### Deploying a Container with Multi-GPU Support

```bash
# Deploy with automatic device selection
python deploy_multi_gpu_container.py --model facebook/opt-1.3b --auto-select

# Deploy with specific devices
python deploy_multi_gpu_container.py --model facebook/opt-1.3b --devices cuda:0 cuda:1

# Deploy with specific container configuration
python deploy_multi_gpu_container.py --model facebook/opt-1.3b \
  --api-type tgi \
  --image huggingface/text-generation-inference:latest \
  --port 8080 \
  --devices cuda:0 cuda:1
```

## Detection and Device Selection

The system automatically detects available hardware:

```python
from utils.device_mapper import DeviceMapper

# Create a device mapper
mapper = DeviceMapper()

# Get detected hardware information
hardware = mapper.device_info
print(f"Detected hardware: {hardware}")

# Get recommended device for a model
device = mapper.get_recommended_device("facebook/opt-1.3b")
print(f"Recommended device: {device}")
```

## Memory Estimation

The system can estimate memory requirements for models:

```python
from utils.device_mapper import DeviceMapper

# Create a device mapper
mapper = DeviceMapper()

# Estimate memory requirements for a model
memory_req = mapper.estimate_model_memory("facebook/opt-6.7b")
print(f"Memory requirements: {memory_req}")
```

## Container Deployment Options

The container deployment script supports multiple options:

```bash
# Get help
python deploy_multi_gpu_container.py --help

# Run device detection only
python deploy_multi_gpu_container.py --detect-only

# Dry run (just print the command)
python deploy_multi_gpu_container.py --model facebook/opt-1.3b --dry-run

# Deploy a Text Generation Inference (TGI) container
python deploy_multi_gpu_container.py --model facebook/opt-1.3b --api-type tgi

# Deploy a Text Embedding Inference (TEI) container
python deploy_multi_gpu_container.py --model sentence-transformers/all-MiniLM-L6-v2 --api-type tei

# Deploy a VLLM container
python deploy_multi_gpu_container.py --model facebook/opt-1.3b --api-type vllm --image vllm/vllm-openai:latest
```

## AMD ROCm Support

The system supports AMD GPUs with ROCm:

```python
from utils.device_mapper import DeviceMapper

# Create a device mapper with ROCm preference
mapper = DeviceMapper(prefer_rocm=True)

# Get detected hardware information
hardware = mapper.device_info
print(f"Detected hardware: {hardware}")

# Create device map for ROCm
device_map = mapper.create_device_map("facebook/opt-1.3b", strategy="auto")
print(f"Device map: {device_map}")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**: Try using the `balanced` strategy or specify only GPUs with sufficient memory
2. **Model Loading Failures**: Ensure PyTorch is installed with CUDA/ROCm support
3. **Container Deployment Issues**: Check Docker installation and GPU driver compatibility

### Debugging Tips

1. Use `--detect-only` with the deployment script to verify hardware detection
2. Use `--dry-run` to check the generated Docker command
3. Set the logging level to DEBUG for more detailed output

## Advanced Configuration

### Custom Device Maps

You can create a custom device map:

```python
# Create a custom device map
custom_map = {
    "embeddings": "cuda:0",
    "layer.0": "cuda:0",
    "layer.1": "cuda:0",
    "layer.2": "cuda:1",
    "layer.3": "cuda:1",
    "head": "cuda:1"
}

# Apply the custom map to a model
from utils.device_mapper import DeviceMapper
mapper = DeviceMapper()
mapper.apply_device_map(model, custom_map)
```

### Container Environment Variables

You can specify additional environment variables for container deployment:

```bash
python deploy_multi_gpu_container.py --model facebook/opt-1.3b \
  --env MAX_BATCH_SIZE=32 ALLOW_TENSORFLOW=true
```

## Performance Considerations

1. **Memory Bandwidth**: Distributing a model across multiple GPUs may introduce latency due to inter-GPU communication
2. **Tensor vs Pipeline Parallelism**: For inference, tensor parallelism often performs better than pipeline parallelism
3. **Optimal Layer Distribution**: The `auto` strategy tries to minimize communication overhead by keeping related layers on the same device

## Implementation Details

### Hardware Detection

The system detects hardware using PyTorch's CUDA API to provide a unified interface for GPU discovery across platforms:

```python
import torch

# Check for CUDA
if torch.cuda.is_available():
    cuda_count = torch.cuda.device_count()
    for i in range(cuda_count):
        device_name = torch.cuda.get_device_name(i)
        device_mem = torch.cuda.get_device_properties(i).total_memory
        # Convert to GB
        device_mem_gb = device_mem / (1024**3)
        # Get compute capability for CUDA-specific optimizations
        compute_capability = f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
        
# Check for ROCm
if hasattr(torch, '_C') and hasattr(torch._C, '_rocm_is_available') and torch._C._rocm_is_available():
    # ROCm uses the CUDA API in PyTorch
    rocm_count = torch.cuda.device_count()
    for i in range(rocm_count):
        device_name = torch.cuda.get_device_name(i)
        device_mem = torch.cuda.get_device_properties(i).total_memory
        # Convert to GB
        device_mem_gb = device_mem / (1024**3)
    
# Check for MPS (Apple Silicon)
if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
    # MPS is available
    # Estimate memory from system memory as MPS doesn't provide direct memory reporting
    import psutil
    system_memory = psutil.virtual_memory().total / (1024**3)
    # Estimate 70% of system memory is available for MPS
    estimated_mem = system_memory * 0.7
```

### Device Mapping Algorithm

The core algorithm for automatic device mapping uses a load-balancing approach based on estimated memory requirements:

```python
def create_auto_map(model_id, memory_req, devices):
    """Create an auto device map based on memory constraints."""
    device_map = {}
    
    # If only one device, put everything there
    if len(devices) == 1:
        return {"": devices[0]}
    
    # Sort devices by memory capacity (descending)
    device_capacities = sorted([
        (device, device_memory.get(device, float('inf')) if device != "cpu" else float('inf'))
        for device in devices
    ], key=lambda x: x[1], reverse=True)
    
    sorted_devices = [d[0] for d in device_capacities]
    
    # Track memory usage on each device
    device_usage = {device: 0.0 for device in sorted_devices}
    
    # Assign embeddings to first device
    device_map["embeddings"] = sorted_devices[0]
    device_usage[sorted_devices[0]] += memory_req["embeddings"]
    
    # Distribute layers by finding device with least used memory percentage
    for i, layer_mem in enumerate(memory_req["layers"]):
        best_device = min(
            sorted_devices,
            key=lambda d: device_usage[d] / (device_memory.get(d, float('inf')) if d != "cpu" else float('inf'))
        )
        
        device_map[f"layer.{i}"] = best_device
        device_usage[best_device] += layer_mem
    
    # Assign head to device with most layers
    layer_counts = {}
    for i in range(len(memory_req["layers"])):
        device = device_map[f"layer.{i}"]
        layer_counts[device] = layer_counts.get(device, 0) + 1
    
    head_device = max(layer_counts.items(), key=lambda x: x[1])[0]
    device_map["head"] = head_device
    
    return device_map
```

### Memory Requirement Estimation

The system estimates memory requirements for models based on their architecture and size:

```python
def estimate_model_memory(model_id, layers=None):
    """Estimate memory requirements for model parts."""
    # Default estimates based on model type
    if "gpt2" in model_id.lower():
        base_size = 0.5  # GB
        per_layer = 0.1  # GB per layer
    elif "bert" in model_id.lower():
        base_size = 0.4
        per_layer = 0.05
    elif "t5" in model_id.lower():
        base_size = 0.8
        per_layer = 0.15
    elif "llama" in model_id.lower() or "mistral" in model_id.lower():
        base_size = 1.2
        per_layer = 0.25
    else:
        # Default estimates
        base_size = 0.5
        per_layer = 0.1
    
    # Estimate number of layers if not provided
    if layers is None:
        if "small" in model_id.lower():
            layers = 6
        elif "base" in model_id.lower():
            layers = 12
        elif "large" in model_id.lower():
            layers = 24
        elif "xl" in model_id.lower() or "xxl" in model_id.lower():
            layers = 36
        else:
            layers = 12  # Default
    
    # Calculate memory requirements
    total_mem = base_size + (layers * per_layer)
    
    return {
        "total": total_mem,
        "embeddings": base_size * 0.3,
        "layers": [(per_layer * 0.8) for _ in range(layers)],
        "head": base_size * 0.2
    }
```

### Container GPU Configuration

The system generates Docker commands with GPU arguments for deploying containerized models:

```python
def get_docker_gpu_args(devices):
    """Get Docker GPU arguments for container deployment."""
    # Get device indices
    device_indices = []
    for device in devices:
        parts = device.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            device_indices.append(int(parts[1]))
    
    # Sort device indices
    device_indices.sort()
    
    # Create GPU argument string
    if not device_indices:
        gpu_arg = ""
    elif len(device_indices) == 1:
        gpu_arg = f"--gpus device={device_indices[0]}"
    else:
        gpu_arg = f"--gpus all"
    
    # Create environment variables for containerized deployment
    env_vars = {
        "NUM_SHARD": len(device_indices) if device_indices else 1
    }
    
    # If specific devices, add CUDA_VISIBLE_DEVICES
    if device_indices:
        env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_indices))
    
    return gpu_arg, env_vars
```

### Tensor Parallelism Integration

For backends that support tensor parallelism (like VLLM), the system provides configuration:

```python
def get_tensor_parallel_config(model_id, devices):
    """Get tensor parallel configuration for models that support it."""
    # Get device indices
    device_indices = []
    for device in devices:
        parts = device.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            device_indices.append(int(parts[1]))
    
    # Configuration for tensor parallelism
    config = {
        "tensor_parallel_size": len(device_indices),
        "gpu_ids": device_indices,
        "max_parallel_loading_workers": min(8, len(device_indices) * 2)
    }
    
    return config
```

## Future Improvements

1. **Dynamic Memory Management**: Adjust memory allocation during runtime based on layer utilization
2. **Quantization Integration**: Add support for mixed precision and quantization alongside device mapping
3. **Benchmark-Guided Optimization**: Use benchmarking to determine optimal device mapping
4. **Multi-Node Support**: Extend device mapping across multiple machines
5. **Automated Kernel Optimization**: Apply custom CUDA kernels based on detected hardware capabilities
6. **Dynamic Load Balancing**: Adapt resource allocation during inference based on actual GPU utilization
7. **Expert Mixture Routing**: Support for Mixture-of-Experts models with efficient routing across GPUs
8. **Zero Redundancy Optimizer**: Implement ZeRO-style optimizations for inference
9. **Hybrid CPU-GPU Pipeline**: Allow memory-intensive layers to offload to CPU when necessary
10. **FlashAttention Integration**: Automatically substitute attention implementations with optimized versions

## References

- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate/index)
- [Text Generation Inference Repository](https://github.com/huggingface/text-generation-inference)
- [VLLM Documentation](https://vllm.readthedocs.io/)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Megatron-LM: Training Multi-Billion Parameter Models](https://github.com/NVIDIA/Megatron-LM)
- [FlashAttention](https://github.com/HazyResearch/flash-attention)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Docker with GPUs](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [NVIDIA Multi-Instance GPU (MIG) Documentation](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)

## Compatibility Matrix

| Strategy              | NVIDIA CUDA | AMD ROCm | Apple MPS | Multi-GPU | Cross-Platform |
|-----------------------|-------------|----------|-----------|-----------|---------------|
| Auto                  | ✅          | ✅       | ✅        | ✅        | ❌            |
| Balanced              | ✅          | ✅       | ✅        | ✅        | ❌            |
| Sequential            | ✅          | ✅       | ✅        | ✅        | ❌            |
| Tensor Parallel       | ✅          | ✅       | ❌        | ✅        | ❌            |
| Pipeline Parallel     | ✅          | ✅       | ❌        | ✅        | ❌            |
| ZeRO-Inference        | ✅          | ✅       | ❌        | ✅        | ❌            |
| Custom Device Map     | ✅          | ✅       | ✅        | ✅        | ❌            |
| Container Deployment  | ✅          | ✅       | ❌        | ✅        | ❌            |