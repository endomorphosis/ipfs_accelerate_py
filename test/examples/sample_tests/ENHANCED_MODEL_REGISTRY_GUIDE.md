# Enhanced Model Registry Guide

This guide explains how to use the enhanced model registry with multi-precision and multi-hardware support in the IPFS Accelerate Python framework.

## Overview

The enhanced model registry provides:

1. **Multi-Hardware Support**: Seamless model initialization across CPUs, NVIDIA GPUs, AMD ROCm GPUs, Apple Silicon, and more
2. **Multi-Precision Support**: Flexible precision formats including FP32, FP16, BF16, INT8, INT4, and more
3. **Automatic Detection**: Smart detection of available hardware and optimal precision types
4. **Performance Benchmarking**: Tools to measure and compare performance across hardware and precision combinations

## Quick Start

```python
from auto_hardware_detection import detect_all_hardware, determine_precision_for_all_hardware
from transformers import AutoModel

# Detect hardware and optimal precision
hardware = detect_all_hardware()
precision_info = determine_precision_for_all_hardware(hardware)

# Get primary hardware and its optimal precision
primary_hw = next((hw for hw, info in hardware.items() if info.detected), "cpu")
optimal_precision = precision_info.get(primary_hw).optimal if primary_hw in precision_info else "fp32"

# Convert precision string to torch dtype
precision_map = {
    "fp32": torch.float32,
    "fp16": torch.float16, 
    "bf16": torch.bfloat16
}
torch_dtype = precision_map.get(optimal_precision, torch.float32)

# Initialize model with optimal settings
model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch_dtype)

# Move to appropriate device
device = "cuda" if primary_hw in ["cuda", "amd"] else primary_hw
model = model.to(device)
```

## Features

### 1. Hardware Detection

```python
from auto_hardware_detection import detect_all_hardware

# Detect all hardware
hardware = detect_all_hardware()

# Check for specific hardware
has_cuda = hardware.get("cuda", {}).get("detected", False)
has_amd = hardware.get("amd", {}).get("detected", False)
has_mps = hardware.get("mps", {}).get("detected", False)

# Get hardware details
if has_cuda:
    cuda_info = hardware["cuda"]
    gpu_count = cuda_info.count
    gpu_names = cuda_info.names
    cuda_version = cuda_info.api_version
    
if has_amd:
    amd_info = hardware["amd"]
    gpu_count = amd_info.count
    gpu_names = amd_info.names
    rocm_version = amd_info.driver_version
```

### 2. Precision Optimization

```python
from auto_hardware_detection import detect_all_hardware, determine_precision_for_all_hardware

# Detect hardware and get precision info
hardware = detect_all_hardware()
precision_info = determine_precision_for_all_hardware(hardware)

# Get supported precision types for AMD hardware
if "amd" in precision_info:
    amd_precision = precision_info["amd"]
    supported_precisions = [p for p, is_supported in amd_precision.supported.items() if is_supported]
    optimal_precision = amd_precision.optimal
    print(f"AMD GPU supports: {supported_precisions}")
    print(f"Optimal precision: {optimal_precision}")
    
    # Performance ranking from best to worst
    if amd_precision.performance_ranking:
        print(f"Performance ranking: {' > '.join(amd_precision.performance_ranking)}")
```

### 3. Benchmarking

```python
from benchmark_precision_hardware import run_benchmark_suite

# Run benchmarks across different hardware and precision types
results = run_benchmark_suite(
    model_names=["bert-base-uncased", "t5-small"],
    hardware_types=["cpu", "cuda", "amd"],
    precision_types=["fp32", "fp16", "int8"],
    batch_sizes=[1, 8],
    output_file="benchmark_results.json",
    generate_charts=True
)

# Print benchmark summary
results.print_summary()

# Generate visualization charts
results.generate_charts(output_dir="charts")
```

### 4. Dependency Management

```python
from install_hardware_dependencies import detect_and_install

# Auto-detect hardware and install dependencies
detect_and_install(
    install_torch=True,
    install_transformers_pkgs=True,
    install_openvino_pkgs=False,
    install_quantization=True,
    install_monitoring=False
)
```

## Working with AMD ROCm

AMD GPUs require some special considerations:

1. **PyTorch ROCm Integration**
   - PyTorch uses the CUDA API for AMD GPUs
   - The device should be set to "cuda" when using AMD GPUs with ROCm

2. **Initialization and Detection**
   ```python
   # Check for AMD GPU with ROCm support
   has_amd = False
   try:
       import torch.utils.hip
       has_amd = torch.utils.hip.is_available()
   except ImportError:
       pass
   
   # Device and precision setup
   if has_amd:
       device = "cuda"  # ROCm uses CUDA API
       dtype = torch.float16  # FP16 is generally best on AMD
   else:
       device = "cpu"
       dtype = torch.float32
   ```

3. **ROCm Version Compatibility**
   - Different ROCm versions support different PyTorch versions
   - Install the appropriate version:
     ```
     pip install torch==2.0.0+rocm5.4.2 -f https://download.pytorch.org/whl/rocm5.4.2/torch/
     ```

4. **Memory Management**
   - AMD GPUs use HIP/ROCm but PyTorch operations use CUDA API:
     ```python
     # Clear cache
     torch.cuda.empty_cache()
     
     # Monitor memory usage
     total_mem = torch.cuda.get_device_properties(0).total_memory
     reserved_mem = torch.cuda.memory_reserved(0)
     allocated_mem = torch.cuda.memory_allocated(0)
     free_mem = total_mem - reserved_mem
     ```

## Precision Type Selection

Choose the right precision type based on your task:

1. **FP32 (float32)**
   - Best for: High-precision workloads, research tasks
   - Hardware: Supported on all platforms
   - Usage:
     ```python
     model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float32)
     ```

2. **FP16 (float16)**
   - Best for: Most inference tasks, balancing speed and accuracy
   - Hardware: NVIDIA, AMD, Apple Silicon
   - Usage:
     ```python
     model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16)
     ```

3. **BF16 (bfloat16)**
   - Best for: Training and mixed-precision tasks
   - Hardware: NVIDIA Ampere+, AMD CDNA2+, CPUs with AVX2
   - Usage:
     ```python
     model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.bfloat16)
     ```

4. **INT8 (8-bit integer)**
   - Best for: Deployment, memory-constrained environments
   - Hardware: Most GPUs and CPUs
   - Usage:
     ```python
     model = AutoModel.from_pretrained("bert-base-uncased", load_in_8bit=True)
     ```

5. **INT4 (4-bit integer)**
   - Best for: Extreme memory optimization
   - Hardware: NVIDIA GPUs, OpenVINO
   - Usage:
     ```python
     from transformers import BitsAndBytesConfig
     quantization_config = BitsAndBytesConfig(load_in_4bit=True)
     model = AutoModel.from_pretrained("bert-base-uncased", quantization_config=quantization_config)
     ```

## Using Hardware Config Files

The auto-detection tool can generate a configuration file that you can use throughout your application:

```python
import json
from transformers import AutoModel
import torch

# Load hardware configuration
with open('hardware_config.json', 'r') as f:
    config = json.load(f)

# Get model-specific configuration (e.g., for BERT)
bert_config = config.get("models", {}).get("bert", {})
hardware = bert_config.get("hardware", "cpu")
precision = bert_config.get("precision", "fp32")
batch_size = bert_config.get("batch_size", 1)

# Map precision to torch dtype
precision_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}
torch_dtype = precision_map.get(precision, torch.float32)

# Initialize model with optimal settings
model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch_dtype)

# Determine device
device = "cuda" if hardware in ["cuda", "amd"] else hardware
model = model.to(device)
```

## Advanced Usage

### Testing Multiple Hardware Platforms

```python
import torch
from transformers import AutoModel, AutoTokenizer
from benchmark_precision_hardware import detect_available_hardware

# Detect available hardware
hardware = detect_available_hardware()
hardware_types = [hw for hw, available in hardware.items() if available and hw != "openvino"]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "This is a test sentence for hardware comparison."
inputs = tokenizer(text, return_tensors="pt")

# Test model on different hardware
results = {}
for hw in hardware_types:
    # Set device
    if hw == "cpu":
        device = "cpu"
    elif hw == "cuda" or hw == "amd":  # Both use CUDA API
        device = "cuda"
    elif hw == "mps":
        device = "mps"
    else:
        continue
        
    # Initialize model
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    
    # Move inputs to device
    hw_inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**hw_inputs)
    end_time = time.time()
    
    # Store result
    results[hw] = {
        "time": end_time - start_time,
        "hidden_states_shape": outputs.last_hidden_state.shape
    }

# Compare results
for hw, result in results.items():
    print(f"{hw}: {result['time']:.4f} seconds")
```

### Custom Hardware Priority

```python
def get_optimal_device_and_precision():
    """Get optimal device and precision based on custom priority"""
    # Custom hardware priority order
    hw_priority = ["amd", "cuda", "mps", "cpu"]
    
    # Detect available hardware
    hardware = detect_available_hardware()
    precision_info = determine_precision_for_all_hardware(hardware)
    
    # Find first available hardware in priority list
    for hw in hw_priority:
        if hardware.get(hw, {}).get("detected", False):
            # Get optimal precision
            if hw in precision_info:
                optimal_precision = precision_info[hw].optimal
            else:
                # Default precision if not in precision info
                optimal_precision = "fp16" if hw != "cpu" else "fp32"
                
            # Map hardware to device string
            device = "cuda" if hw in ["cuda", "amd"] else hw
            
            return device, optimal_precision
    
    # Fallback to CPU with fp32
    return "cpu", "fp32"

# Use in model initialization
device, precision = get_optimal_device_and_precision()
# Map precision to torch dtype
precision_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
torch_dtype = precision_map.get(precision, torch.float32)

model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch_dtype).to(device)
```

## Best Practices

1. **Always detect hardware first**
   - Don't assume specific hardware is available
   - Use the auto-detection tools to dynamically adapt

2. **Choose the right precision for your task**
   - FP32: When accuracy is critical
   - FP16: For most inference tasks
   - INT8: For deployment and memory constraints
   - Let the auto-detection suggest the optimal precision

3. **Benchmark before deployment**
   - Different models perform differently on the same hardware
   - Use the benchmarking tool to find the optimal configuration
   - Test with your specific workload patterns

4. **Manage dependencies carefully**
   - Use the dependency installer to get the right packages
   - Be aware of version compatibility between PyTorch and ROCm

5. **Handle fallbacks gracefully**
   - Always have a CPU fallback for when specific hardware is unavailable
   - Check precision support before trying to use specific formats

## Troubleshooting

### Precision Errors

```python
def safe_load_model(model_name, device, precision):
    """Safely load model with fallback to lower precision if needed"""
    try:
        # Try loading with requested precision
        precision_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        torch_dtype = precision_map.get(precision, torch.float32)
        
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        return model.to(device), precision
    
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"Failed to load with {precision}: {str(e)}")
        
        # Fallback path
        if precision == "bf16":
            print("Falling back to fp16")
            return safe_load_model(model_name, device, "fp16")
        elif precision == "fp16":
            print("Falling back to fp32")
            return safe_load_model(model_name, device, "fp32")
        else:
            # If fp32 fails, it's a more serious issue
            raise e
```

### Device Errors

```python
def get_safe_device():
    """Get a safe device with fallbacks"""
    if torch.cuda.is_available():
        try:
            # Test CUDA device
            x = torch.zeros(1).cuda()
            del x
            return "cuda"
        except RuntimeError:
            print("CUDA device reported as available but failed to initialize")
    
    # Try MPS (Apple Silicon)
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            x = torch.zeros(1).to("mps")
            del x
            return "mps"
    except:
        print("MPS device reported as available but failed to initialize")
    
    # Fallback to CPU
    return "cpu"
```

## Examples

### Complete BERT Example

```python
import torch
import json
import time
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# First detect hardware and optimal settings
from auto_hardware_detection import detect_all_hardware, determine_precision_for_all_hardware

def initialize_optimized_model(model_name):
    # Detect hardware and precision
    hardware = detect_all_hardware()
    precision_info = determine_precision_for_all_hardware(hardware)
    
    # Find primary hardware
    detected_hw = [hw for hw, info in hardware.items() if info.detected]
    if not detected_hw:
        print("No hardware detected, using CPU")
        return AutoModel.from_pretrained(model_name).to("cpu")
    
    # Get hardware with priority
    hw_priority = ["cuda", "amd", "mps", "cpu"]
    primary_hw = next((hw for hw in hw_priority if hw in detected_hw), "cpu")
    
    # Get optimal precision
    optimal_precision = None
    if primary_hw in precision_info:
        optimal_precision = precision_info[primary_hw].optimal
    
    # Default precisions if not determined
    if not optimal_precision:
        if primary_hw in ["cuda", "amd", "mps"]:
            optimal_precision = "fp16"
        else:
            optimal_precision = "fp32"
    
    print(f"Using {primary_hw} with {optimal_precision} precision")
    
    # Map precision to torch dtype
    precision_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    torch_dtype = precision_map.get(optimal_precision, torch.float32)
    
    # Handle special case of int8 quantization
    if optimal_precision == "int8":
        model = AutoModel.from_pretrained(model_name, load_in_8bit=True)
    else:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    
    # Set device (AMD GPUs use CUDA device in PyTorch)
    device = "cuda" if primary_hw in ["cuda", "amd"] else primary_hw
    
    return model.to(device)

# Initialize model with optimal settings
model = initialize_optimized_model("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Use the model
text = "Testing the optimized model initialization."
inputs = tokenizer(text, return_tensors="pt")

# Move inputs to the same device as model
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference
with torch.no_grad():
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()

print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
print(f"Output shape: {outputs.last_hidden_state.shape}")
```

## Further Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)