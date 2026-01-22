# AMD Hardware Support with Multi-Precision Benchmarking & ONNX/WebNN Export

This document describes the enhanced AMD ROCm support, multi-precision benchmarking capabilities, and ONNX/WebNN export features added to the IPFS Accelerate Python framework for optimal deployment across platforms.

## Overview

The framework now includes comprehensive support for AMD GPUs with ROCm, alongside multiple precision formats and deployment options. It provides tools for:

1. **Hardware Auto-Detection**: Automatically discover available hardware and optimal precision formats
2. **Performance Benchmarking**: Compare model performance across hardware platforms and precision types
3. **Optimized Dependency Installation**: Automatically install the right dependencies for your hardware
4. **ONNX Model Export**: Convert PyTorch models to ONNX format with AMD-specific optimizations
5. **WebNN JavaScript Implementation**: Generate ready-to-use JavaScript code for web deployment 
6. **Cross-Platform Deployment**: Seamlessly move from AMD hardware training to browser or Node.js deployment

## Key Components

### 1. Hardware Detection (auto_generators/hardware/hardware_detection.py)

Automatically identifies available hardware platforms:

- CPU features and capabilities
- NVIDIA GPUs with CUDA
- AMD GPUs with ROCm
- Apple Silicon with MPS
- Intel hardware with OpenVINO
- Qualcomm AI hardware

For each platform, it determines:
- Hardware specifications
- Supported precision formats
- Optimal configurations for different model types
- Recommended software dependencies

**Usage:**
```bash
python auto_generators/hardware/hardware_detection.py --generate-config --generate-requirements
```

### 2. Precision Benchmarking (benchmark_precision_hardware.py)

Benchmarks models across hardware platforms and precision formats:

- Tests models on CPU, CUDA (NVIDIA), AMD (ROCm), and MPS (Apple)
- Measures performance with different precision types (fp32, fp16, bf16, int8, int4, etc.)
- Measures inference speed, memory usage, throughput, and energy consumption
- Generates detailed performance charts and comparison reports

**Usage:**
```bash
# Benchmark BERT model on all available hardware
python duckdb_api/core/benchmark_precision_hardware.py --models bert-base-uncased

# Benchmark specific models with specific precision types
python duckdb_api/core/benchmark_precision_hardware.py --models bert-base-uncased t5-small --precision fp32 fp16 int8 --hardware cpu cuda amd
```

### 3. Dependency Installation (install_hardware_dependencies.py)

Automatically installs dependencies based on detected hardware:

- Installs PyTorch with appropriate hardware support (CUDA, ROCm, MPS)
- Manages precision-specific dependencies (quantization, etc.)
- Handles hardware-specific requirements for optimal performance
- Provides flexible installation options

**Usage:**
```bash
# Auto-detect hardware and install appropriate dependencies
python install_hardware_dependencies.py

# Install with specific options
python install_hardware_dependencies.py --install-openvino --install-monitoring
```

### 4. Model Export Capability (model_export_capability.py)

Exports models to ONNX and WebNN formats with comprehensive metadata:

- Exports PyTorch models to ONNX with hardware-specific optimizations
- Generates WebNN-compatible models via ONNX intermediates
- Captures detailed model architecture information
- Provides JavaScript/Node.js implementation templates
- Includes pre/post-processing specifications

**Usage:**
```bash
# Export BERT model to ONNX with AMD optimizations
python model_export_capability.py --model bert-base-uncased --format onnx --hardware amd --output model.onnx

# Export to WebNN format with browser deployment templates
python model_export_capability.py --model bert-base-uncased --format webnn --output web_model_dir

# Analyze model export compatibility
python model_export_capability.py --model bert-base-uncased --analyze
```

## AMD ROCm Support Details

AMD GPUs are now fully supported with:

1. **Hardware Detection**:
   - Automatic detection of AMD GPUs using PyTorch's ROCm support
   - Identification of ROCm version and driver details
   - Integration with `rocm-smi` for detailed GPU information

2. **Precision Support**:
   - FP32 (full precision)
   - FP16 (half precision)
   - BF16 (bfloat16) on select hardware
   - INT8 (8-bit quantization)

3. **Optimizations**:
   - Hardware-aware initialization to utilize AMD-specific features
   - ROCm version-specific optimizations
   - Performance tuning for different precision formats

## Precision Types Explained

| Precision | Bits | Description | Best For |
|-----------|------|-------------|----------|
| FP32      | 32   | Standard full precision | Accuracy-critical tasks |
| FP16      | 16   | Half precision | Balance of speed and accuracy |
| BF16      | 16   | Brain floating point | Neural network training |
| INT8      | 8    | 8-bit integer quantization | Inference optimization |
| INT4      | 4    | 4-bit integer quantization | Memory-constrained deployment |
| UINT4     | 4    | Unsigned 4-bit quantization | Small model deployment |
| FP8       | 8    | 8-bit floating point | Newer NVIDIA GPUs |
| FP4       | 4    | 4-bit floating point | Experimental applications |

## Hardware Compatibility Matrix

| Hardware | FP32 | FP16 | BF16 | INT8 | INT4 | UINT4 | FP8 | FP4 |
|----------|------|------|------|------|------|-------|-----|-----|
| CPU      | ✓    | ×    | ⚪¹   | ✓    | ⚪²   | ⚪²    | ×   | ×   |
| NVIDIA (CUDA) | ✓ | ✓  | ⚪³   | ✓    | ✓    | ✓     | ⚪⁴  | ×   |
| AMD (ROCm) | ✓  | ✓    | ⚪⁵   | ✓    | ×    | ×     | ×   | ×   |
| Apple Silicon | ✓ | ✓  | ×    | ✓    | ×    | ×     | ×   | ×   |
| OpenVINO | ✓   | ✓    | ×    | ✓    | ✓    | ✓     | ×   | ×   |

¹ Requires AVX2 support  
² Requires transformers library with quantization support  
³ Ampere architecture or newer (compute capability 8.0+)  
⁴ Hopper architecture only (compute capability 9.0+)  
⁵ CDNA2 architecture or newer  

## Getting Started

1. **Detect your hardware**:
   ```bash
   python auto_generators/hardware/hardware_detection.py
   ```

2. **Install appropriate dependencies**:
   ```bash
   python install_hardware_dependencies.py
   ```

3. **Benchmark models**:
   ```bash
   python duckdb_api/core/benchmark_precision_hardware.py --models bert-base-uncased
   ```

4. **Use the optimal configuration**:
   - The hardware detection tool generates a JSON configuration file
   - Use this configuration to initialize models with the best settings
   - Load the configuration with:
   ```python
   with open('hardware_config.json') as f:
       config = json.load(f)
   ```

## Advanced Usage

### Custom Precision Settings

To manually specify precision settings:

```python
import torch
from transformers import AutoModel

# For FP16 precision
model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16)

# For INT8 quantization
model = AutoModel.from_pretrained("bert-base-uncased", load_in_8bit=True)

# For INT4 quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModel.from_pretrained("bert-base-uncased", quantization_config=quantization_config)
```

### AMD-Specific Optimizations

```python
import torch
from transformers import AutoModel

# Check for AMD GPU
has_amd = False
try:
    import torch.utils.hip
    has_amd = torch.utils.hip.is_available()
except ImportError:
    pass

# Load model with appropriate settings
if has_amd:
    # ROCm uses CUDA device for AMD GPUs in PyTorch
    device = "cuda"
    # FP16 is generally best on AMD GPUs
    model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16).to(device)
else:
    # Fallback to CPU
    model = AutoModel.from_pretrained("bert-base-uncased").to("cpu")
```

## Troubleshooting

### Common AMD ROCm Issues

1. **ROCm installation problems**:
   - Make sure you have the correct ROCm version installed
   - Check system compatibility with `rocm-smi --showproductname`

2. **PyTorch with ROCm support**:
   - Install PyTorch with specific ROCm version:
     ```
     pip install torch==2.0.0+rocm5.4.2 -f https://download.pytorch.org/whl/rocm5.4.2/torch/
     ```

3. **Memory errors**:
   - Reduce batch size
   - Use lower precision (FP16 instead of FP32)
   - Enable gradient checkpointing

### Performance Optimization

1. **Setting optimal batch size**:
   - Use the auto-detection tool's recommended batch sizes
   - Generally, AMD GPUs perform best with larger batch sizes than CPU

2. **Choosing precision**:
   - FP16 is usually the best balance of speed and accuracy on AMD GPUs
   - INT8 provides significant speed improvements with some accuracy loss
   - BF16 is available on newer AMD hardware (CDNA2 architecture)

3. **Memory management**:
   - Use `torch.cuda.empty_cache()` between inferences
   - Monitor memory with `rocm-smi`

## Contributing

Contributions to improve AMD support and precision benchmarking are welcome:

1. Bug reports and feature requests: Open an issue in the repository
2. Code contributions: Submit a pull request
3. Documentation improvements: Update README or documentation files

## License

This project is licensed under the MIT License - see the LICENSE file for details.