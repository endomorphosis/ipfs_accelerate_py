# IPFS Accelerate Python - Hardware Optimization Guide

This guide covers hardware-specific optimization strategies for the IPFS Accelerate Python framework.

## Table of Contents

- [Hardware Detection](#hardware-detection)
- [CPU Optimization](#cpu-optimization)
- [CUDA Acceleration](#cuda-acceleration)
- [AMD ROCm Support](#amd-rocm-support)
- [Intel OpenVINO](#intel-openvino)
- [Apple Metal Performance Shaders (MPS)](#apple-metal-performance-shaders-mps)
- [Qualcomm Acceleration](#qualcomm-acceleration)
- [WebNN/WebGPU](#webnnwebgpu)
- [Performance Tuning](#performance-tuning)
- [Memory Management](#memory-management)
- [Benchmarking](#benchmarking)

## Hardware Detection

The framework automatically detects available hardware acceleration options during initialization.

### Automatic Detection

```python
from ipfs_accelerate_py import ipfs_accelerate_py

accelerator = ipfs_accelerate_py({}, {})

# Check detected hardware
hardware_info = accelerator.hardware_detection.detect_all_hardware()
print(hardware_info)
```

Expected output:
```python
{
    "cpu": {"available": True, "cores": 8, "architecture": "x86_64"},
    "cuda": {"available": True, "devices": 1, "memory": "8GB", "compute_capability": "8.6"},
    "openvino": {"available": True, "version": "2023.1", "devices": ["CPU", "GPU"]},
    "mps": {"available": True, "unified_memory": "16GB"},
    "rocm": {"available": False},
    "qualcomm": {"available": False},
    "webnn": {"available": True, "providers": ["DirectML", "OpenVINO"]},
    "webgpu": {"available": True, "adapters": ["NVIDIA", "Intel"]}
}
```

### Manual Hardware Configuration

```python
config = {
    "hardware": {
        "prefer_cuda": True,        # Prioritize CUDA if available
        "allow_openvino": True,     # Enable Intel OpenVINO
        "allow_mps": True,          # Enable Apple MPS
        "allow_rocm": True,         # Enable AMD ROCm
        "allow_qualcomm": False,    # Disable Qualcomm (default)
        "precision": "fp16",        # Use half precision
        "mixed_precision": True,    # Enable mixed precision training
        "batch_size": 1,            # Default batch size
        "max_memory": "4GB"         # Limit memory usage
    }
}

accelerator = ipfs_accelerate_py(config, {})
```

## CPU Optimization

### x86/x64 Optimization

For Intel and AMD processors:

```python
cpu_config = {
    "hardware": {
        "prefer_cuda": False,       # Use CPU as primary
        "cpu_optimization": {
            "use_mkl": True,        # Use Intel Math Kernel Library
            "num_threads": 8,       # Number of CPU threads
            "enable_avx": True,     # Enable AVX instructions
            "enable_avx512": True   # Enable AVX-512 if available
        }
    }
}

accelerator = ipfs_accelerate_py(cpu_config, {})
```

### ARM Optimization

For ARM-based processors (including Apple Silicon):

```python
arm_config = {
    "hardware": {
        "cpu_optimization": {
            "use_neon": True,       # Enable ARM NEON SIMD
            "num_threads": 4,       # Optimize for ARM core count
            "memory_pool": True     # Enable memory pooling
        }
    }
}
```

### CPU Performance Tips

1. **Thread Configuration**: Set `num_threads` to match your CPU cores
2. **Memory Allocation**: Use memory pooling for frequent allocations
3. **SIMD Instructions**: Enable AVX/NEON for vectorized operations
4. **Process Affinity**: Pin processes to specific CPU cores for consistency

```python
import os

# Set CPU affinity (Linux/macOS)
os.sched_setaffinity(0, {0, 1, 2, 3})  # Use first 4 cores

# Set environment variables for optimized CPU libraries
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
```

## CUDA Acceleration

### NVIDIA GPU Configuration

```python
cuda_config = {
    "hardware": {
        "prefer_cuda": True,
        "cuda_optimization": {
            "device_id": 0,             # GPU device ID
            "memory_fraction": 0.8,     # Use 80% of GPU memory
            "allow_growth": True,       # Allow memory growth
            "precision": "fp16",        # Use half precision
            "enable_tensorrt": True,    # Enable TensorRT optimization
            "enable_cudnn": True        # Enable cuDNN
        }
    }
}

accelerator = ipfs_accelerate_py(cuda_config, {})
```

### Multi-GPU Configuration

```python
multi_gpu_config = {
    "hardware": {
        "prefer_cuda": True,
        "cuda_optimization": {
            "devices": [0, 1, 2, 3],    # Use multiple GPUs
            "strategy": "data_parallel", # Parallelization strategy
            "memory_fraction": 0.7,      # Per-GPU memory limit
            "enable_peer_to_peer": True  # Enable P2P memory transfer
        }
    }
}
```

### CUDA Performance Optimization

```python
# Example with specific CUDA optimizations
cuda_optimized = ipfs_accelerate_py({
    "hardware": {
        "prefer_cuda": True,
        "precision": "fp16",
        "mixed_precision": True,
        "cuda_optimization": {
            "enable_flash_attention": True,  # Enable Flash Attention
            "enable_memory_efficient": True, # Memory efficient attention
            "compile_model": True,           # Compile model for speed
            "use_triton": True              # Use Triton kernels
        }
    }
}, {})

# Benchmark CUDA performance
import time
start_time = time.time()

result = cuda_optimized.process(
    model="bert-base-uncased",
    input_data={"input_ids": [101] + list(range(1000, 1500)) + [102]},
    endpoint_type="text_embedding"
)

cuda_time = time.time() - start_time
print(f"CUDA inference time: {cuda_time:.3f}s")
```

## AMD ROCm Support

### ROCm Configuration

```python
rocm_config = {
    "hardware": {
        "prefer_cuda": False,
        "allow_rocm": True,
        "rocm_optimization": {
            "device_id": 0,
            "memory_fraction": 0.8,
            "precision": "fp16",
            "enable_miopen": True,      # Enable MIOpen
            "hip_visible_devices": "0"  # Specify visible devices
        }
    }
}

accelerator = ipfs_accelerate_py(rocm_config, {})
```

### ROCm Environment Setup

```python
import os

# Set ROCm environment variables
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCR_VISIBLE_DEVICES'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # For compatibility

# Initialize with ROCm
accelerator = ipfs_accelerate_py(rocm_config, {})
```

## Intel OpenVINO

### OpenVINO Configuration

```python
openvino_config = {
    "hardware": {
        "prefer_cuda": False,
        "allow_openvino": True,
        "openvino_optimization": {
            "device": "CPU",            # CPU, GPU, or AUTO
            "precision": "FP16",        # FP32, FP16, or INT8
            "num_streams": 4,           # Number of inference streams
            "enable_cache": True,       # Enable model caching
            "cache_dir": "./openvino_cache"
        }
    }
}

accelerator = ipfs_accelerate_py(openvino_config, {})
```

### OpenVINO Multi-Device

```python
# Use multiple OpenVINO devices
multi_device_config = {
    "hardware": {
        "allow_openvino": True,
        "openvino_optimization": {
            "device": "MULTI:CPU,GPU",  # Use both CPU and GPU
            "precision": "FP16",
            "batch_size": 4,
            "enable_dynamic_shapes": True
        }
    }
}
```

### Intel GPU with OpenVINO

```python
# Specific Intel GPU configuration
intel_gpu_config = {
    "hardware": {
        "allow_openvino": True,
        "openvino_optimization": {
            "device": "GPU",
            "precision": "FP16",
            "gpu_device_id": 0,
            "enable_gpu_throttling": False,
            "gpu_memory_optimization": True
        }
    }
}
```

## Apple Metal Performance Shaders (MPS)

### MPS Configuration

```python
mps_config = {
    "hardware": {
        "prefer_cuda": False,
        "allow_mps": True,
        "mps_optimization": {
            "precision": "fp16",        # Use half precision
            "unified_memory": True,     # Leverage unified memory
            "memory_fraction": 0.8,     # Use 80% of available memory
            "enable_amp": True          # Automatic Mixed Precision
        }
    }
}

accelerator = ipfs_accelerate_py(mps_config, {})
```

### MPS Performance Tips

```python
# Optimize for Apple Silicon
apple_silicon_config = {
    "hardware": {
        "allow_mps": True,
        "mps_optimization": {
            "precision": "fp16",
            "batch_size": 1,            # Apple Silicon optimized for single batch
            "enable_memory_mapping": True,
            "use_neural_engine": True,  # Leverage Neural Engine when possible
            "memory_pool_size": "2GB"
        }
    }
}

# Example usage
accelerator = ipfs_accelerate_py(apple_silicon_config, {})

# Benchmark on Apple Silicon
result = accelerator.process(
    model="bert-base-uncased", 
    input_data={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    endpoint_type="text_embedding"
)
```

## Qualcomm Acceleration

### Qualcomm Configuration

```python
qualcomm_config = {
    "hardware": {
        "allow_qualcomm": True,
        "qualcomm_optimization": {
            "device": "dsp",            # dsp, gpu, or cpu
            "precision": "int8",        # Quantized precision
            "enable_htp": True,         # Hexagon Tensor Processor
            "performance_mode": "high", # high, balanced, or power_save
            "enable_caching": True
        }
    }
}

accelerator = ipfs_accelerate_py(qualcomm_config, {})
```

### Qualcomm Performance Modes

```python
# High performance mode
high_perf_config = {
    "hardware": {
        "allow_qualcomm": True,
        "qualcomm_optimization": {
            "device": "dsp",
            "performance_mode": "high",
            "cpu_fallback": False,      # Disable CPU fallback
            "enable_profiling": True    # Enable performance profiling
        }
    }
}

# Power-efficient mode
power_save_config = {
    "hardware": {
        "allow_qualcomm": True,
        "qualcomm_optimization": {
            "device": "dsp",
            "performance_mode": "power_save",
            "precision": "int8",
            "enable_dynamic_batching": False
        }
    }
}
```

## WebNN/WebGPU

### WebNN Configuration

```python
webnn_config = {
    "hardware": {
        "webnn_optimization": {
            "provider": "DirectML",     # DirectML, OpenVINO, CoreML
            "device": "gpu",           # cpu, gpu, or npu
            "precision": "fp16",
            "enable_graph_optimization": True,
            "batch_size": 1
        }
    }
}
```

### WebGPU Configuration

```python
webgpu_config = {
    "hardware": {
        "webgpu_optimization": {
            "adapter": "discrete",      # integrated, discrete, or software
            "precision": "fp16",
            "enable_shader_caching": True,
            "memory_management": "automatic",
            "enable_compute_shaders": True
        }
    }
}
```

For detailed WebNN/WebGPU usage, see the [WebNN/WebGPU README](../../features/webnn-webgpu/WEBNN_WEBGPU_README.md).

## Performance Tuning

### Batch Size Optimization

```python
def find_optimal_batch_size(accelerator, model, sample_input):
    """Find optimal batch size for a model."""
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # Create batched input
            batched_input = {
                key: [value] * batch_size 
                for key, value in sample_input.items()
            }
            
            start_time = time.time()
            result = accelerator.process(
                model=model,
                input_data=batched_input,
                endpoint_type="auto"
            )
            inference_time = time.time() - start_time
            
            results[batch_size] = {
                "time_per_sample": inference_time / batch_size,
                "total_time": inference_time,
                "throughput": batch_size / inference_time
            }
            
        except Exception as e:
            results[batch_size] = {"error": str(e)}
    
    return results

# Example usage
sample_input = {"input_ids": [101, 2054, 2003, 102]}
batch_results = find_optimal_batch_size(accelerator, "bert-base-uncased", sample_input)
```

### Precision Optimization

```python
# Compare different precisions
precisions = ["fp32", "fp16", "int8"]
precision_results = {}

for precision in precisions:
    config = {
        "hardware": {
            "precision": precision,
            "mixed_precision": precision == "fp16"
        }
    }
    
    test_accelerator = ipfs_accelerate_py(config, {})
    
    start_time = time.time()
    result = test_accelerator.process(
        model="bert-base-uncased",
        input_data={"input_ids": [101, 2054, 2003, 102]},
        endpoint_type="text_embedding"
    )
    inference_time = time.time() - start_time
    
    precision_results[precision] = {
        "inference_time": inference_time,
        "result_shape": len(result.get("embedding", [])),
        "memory_usage": "estimate"  # Would need actual memory profiling
    }

print("Precision Comparison:", precision_results)
```

## Memory Management

### Memory Pool Configuration

```python
memory_config = {
    "hardware": {
        "memory_management": {
            "enable_pooling": True,     # Enable memory pooling
            "pool_size": "2GB",         # Memory pool size
            "enable_garbage_collection": True,
            "gc_threshold": 0.8,        # Trigger GC at 80% usage
            "enable_memory_mapping": True
        }
    }
}

accelerator = ipfs_accelerate_py(memory_config, {})
```

### Memory Monitoring

```python
import psutil
import time

def monitor_memory_usage(func, *args, **kwargs):
    """Monitor memory usage during function execution."""
    process = psutil.Process()
    
    # Get initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Get final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "result": result,
        "execution_time": end_time - start_time,
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "memory_delta_mb": final_memory - initial_memory
    }

# Example usage
def inference_task():
    return accelerator.process(
        model="bert-base-uncased",
        input_data={"input_ids": [101] + list(range(1000, 1500)) + [102]},
        endpoint_type="text_embedding"
    )

memory_stats = monitor_memory_usage(inference_task)
print(f"Memory usage: {memory_stats['memory_delta_mb']:.1f} MB")
print(f"Execution time: {memory_stats['execution_time']:.3f} seconds")
```

## Benchmarking

### Comprehensive Hardware Benchmark

```python
import time
import json

def comprehensive_benchmark():
    """Benchmark all available hardware configurations."""
    
    # Define test configurations
    configs = {
        "cpu": {"hardware": {"prefer_cuda": False, "allow_openvino": False}},
        "cuda": {"hardware": {"prefer_cuda": True}},
        "openvino": {"hardware": {"allow_openvino": True, "prefer_cuda": False}},
        "mps": {"hardware": {"allow_mps": True, "prefer_cuda": False}}
    }
    
    test_cases = [
        {
            "name": "small_text",
            "model": "bert-base-uncased",
            "input": {"input_ids": [101, 2054, 2003, 102]},
            "endpoint_type": "text_embedding"
        },
        {
            "name": "large_text", 
            "model": "bert-base-uncased",
            "input": {"input_ids": [101] + list(range(1000, 1400)) + [102]},
            "endpoint_type": "text_embedding"
        }
    ]
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"Benchmarking {config_name}...")
        results[config_name] = {}
        
        try:
            accelerator = ipfs_accelerate_py(config, {})
            
            for test_case in test_cases:
                times = []
                
                # Warm-up
                accelerator.process(
                    model=test_case["model"],
                    input_data=test_case["input"],
                    endpoint_type=test_case["endpoint_type"]
                )
                
                # Benchmark runs
                for _ in range(5):
                    start_time = time.time()
                    result = accelerator.process(
                        model=test_case["model"],
                        input_data=test_case["input"],
                        endpoint_type=test_case["endpoint_type"]
                    )
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                results[config_name][test_case["name"]] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "std_dev": (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times))**0.5
                }
                
        except Exception as e:
            results[config_name]["error"] = str(e)
    
    return results

# Run benchmark
benchmark_results = comprehensive_benchmark()

# Save results
with open("hardware_benchmark_results.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("Benchmark Results:")
for config, tests in benchmark_results.items():
    print(f"\n{config.upper()}:")
    if "error" in tests:
        print(f"  Error: {tests['error']}")
    else:
        for test_name, metrics in tests.items():
            print(f"  {test_name}: {metrics['avg_time']:.3f}s (±{metrics['std_dev']:.3f})")
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_inference():
    """Profile inference performance."""
    
    def inference_task():
        accelerator = ipfs_accelerate_py({
            "hardware": {"prefer_cuda": True, "precision": "fp16"}
        }, {})
        
        for _ in range(10):
            result = accelerator.process(
                model="bert-base-uncased",
                input_data={"input_ids": [101, 2054, 2003, 102]},
                endpoint_type="text_embedding"
            )
        
        return result
    
    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = inference_task()
    
    profiler.disable()
    
    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result

# Run profiling
profiled_result = profile_inference()
```

For more benchmarking tools and examples, see the [benchmarks directory](../data/benchmarks/).

## Hardware-Specific Tips

### General Optimization Principles

1. **Memory Hierarchy**: Optimize for your hardware's memory hierarchy
2. **Parallelism**: Leverage available parallelism (SIMD, GPU cores, etc.)
3. **Data Types**: Use appropriate precision for your hardware
4. **Batch Processing**: Optimize batch sizes for throughput
5. **Caching**: Enable caching for frequently used models

### Debugging Hardware Issues

```python
def diagnose_hardware():
    """Diagnose hardware acceleration issues."""
    
    accelerator = ipfs_accelerate_py({}, {})
    hardware_info = accelerator.hardware_detection.detect_all_hardware()
    
    print("Hardware Diagnosis:")
    print("==================")
    
    for hardware, info in hardware_info.items():
        print(f"{hardware.upper()}:")
        if info.get("available", False):
            print(f"  ✓ Available")
            for key, value in info.items():
                if key != "available":
                    print(f"  - {key}: {value}")
        else:
            print(f"  ✗ Not available")
        print()
    
    # Test basic inference on each available hardware
    test_configs = []
    if hardware_info["cuda"]["available"]:
        test_configs.append(("CUDA", {"prefer_cuda": True}))
    if hardware_info["openvino"]["available"]:
        test_configs.append(("OpenVINO", {"allow_openvino": True, "prefer_cuda": False}))
    if hardware_info["mps"]["available"]:
        test_configs.append(("MPS", {"allow_mps": True, "prefer_cuda": False}))
    
    for name, config in test_configs:
        try:
            test_accelerator = ipfs_accelerate_py({"hardware": config}, {})
            result = test_accelerator.process(
                model="bert-base-uncased",
                input_data={"input_ids": [101, 102]},
                endpoint_type="text_embedding"
            )
            print(f"{name}: ✓ Working")
        except Exception as e:
            print(f"{name}: ✗ Error - {e}")

# Run diagnosis
diagnose_hardware()
```

## Related Documentation

- [Usage Guide](../../archive/USAGE.md) - General usage patterns
- [API Reference](../../api/overview.md) - Complete API documentation  
- [IPFS Integration](../../features/ipfs/IPFS.md) - IPFS-specific features
- [WebNN/WebGPU README](../../features/webnn-webgpu/WEBNN_WEBGPU_README.md) - Browser acceleration
- [Benchmarks](../../../data/benchmarks/README.md) - Performance benchmarking tools