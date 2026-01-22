# Model Compression and Optimization Guide

## Overview

The model compression system provides a comprehensive toolkit for compressing and optimizing models using various techniques including quantization, pruning, knowledge distillation, and graph optimization. This system helps reduce model size, improve inference speed, and optimize models for deployment on specific hardware platforms.

## Key Components

The model compression system consists of these primary components:

1. **Model Compressor** (`model_compression.py`): Core component that applies various compression techniques
2. **Compression Recommendations**: System that suggests optimal compression methods based on model family and target hardware
3. **Compression Validation**: Tools for validating compression results with performance and accuracy metrics
4. **Compression Reporting**: Detailed reporting on compression results and performance impact

## Model Compressor

The `model_compression.py` script provides the core compression functionality:

### Features

- **Multiple Compression Methods**: Supports quantization, pruning, knowledge distillation, and graph optimization
- **Hardware-Aware Optimization**: Tailors compression to specific target hardware
- **Model Family Awareness**: Uses model family classification to select optimal compression strategies
- **Comprehensive Metrics**: Measures size reduction, inference speedup, and memory usage
- **ResourcePool Integration**: Uses the ResourcePool for efficient model loading and caching
- **Validation System**: Validates compressed models against original models
- **Detailed Reporting**: Generates comprehensive reports on compression results
- **Multiple Output Formats**: Supports PyTorch and ONNX output formats

### Usage

```bash
# Basic usage with default settings (get recommendations)
python model_compression.py --model bert-base-uncased

# List available compression methods
python model_compression.py --list-methods

# Get compression recommendations without applying them
python model_compression.py --model bert-base-uncased --recommend

# Apply specific compression methods
python model_compression.py --model bert-base-uncased --methods quantization:dynamic pruning:magnitude

# Specify target hardware
python model_compression.py --model bert-base-uncased --target-hardware cuda

# Validate compression results
python model_compression.py --model bert-base-uncased --validate

# Save in ONNX format
python model_compression.py --model bert-base-uncased --format onnx

# Enable debug logging
python model_compression.py --model bert-base-uncased --debug
```

### Key Parameters

- `--model`: Model name or path (required)
- `--output-dir`: Output directory for compressed models
- `--cache-dir`: Cache directory for model downloads
- `--model-type`: Model type (embedding, text_generation, vision, audio, multimodal)
- `--methods`: Compression methods to apply (e.g., quantization:dynamic pruning:magnitude)
- `--params`: JSON string with parameters for compression methods
- `--target-hardware`: Target hardware (cpu, cuda, mps, openvino)
- `--list-methods`: List available compression methods
- `--validate`: Validate compression with basic metrics
- `--format`: Output format (pytorch, onnx)
- `--recommend`: Get compression recommendations without applying them

## Compression Methods

The model compression system supports the following compression methods:

### Quantization

Quantization reduces the precision of model weights and activations:

| Method | Description | Target Hardware |
|--------|-------------|----------------|
| `quantization:dynamic` | Dynamic quantization (post-training) | CPU |
| `quantization:static` | Static quantization (requires calibration data) | CPU |
| `quantization:qat` | Quantization-aware training | CPU |
| `quantization:onnx` | ONNX-based quantization | CPU, specialized hardware |
| `quantization:int8` | Int8 precision (8-bit weights and activations) | CPU, specialized hardware |
| `quantization:int4` | Int4 precision (4-bit weights) | CPU, specialized hardware |
| `quantization:fp16` | Mixed precision (16-bit floating point) | CUDA, ROCm |
| `quantization:bf16` | BFloat16 precision (brain floating point format) | CUDA (Ampere+), TPUs |

Examples:

```bash
# Apply dynamic quantization for CPU deployment
python model_compression.py --model bert-base-uncased --methods quantization:dynamic

# Apply FP16 quantization for CUDA deployment
python model_compression.py --model gpt2 --methods quantization:fp16 --target-hardware cuda

# Apply Int8 quantization for edge devices
python model_compression.py --model distilbert-base-uncased --methods quantization:int8
```

### Pruning

Pruning removes unnecessary weights from the model:

| Method | Description | Target Hardware |
|--------|-------------|----------------|
| `pruning:magnitude` | Magnitude-based weight pruning | All |
| `pruning:structured` | Structured pruning (removes entire channels/neurons) | All |
| `pruning:progressive` | Progressive pruning (gradual pruning during training) | All |

Examples:

```bash
# Apply magnitude-based pruning with 30% sparsity
python model_compression.py --model bert-base-uncased --methods pruning:magnitude --params '{"magnitude": {"sparsity": 0.3}}'

# Apply structured pruning
python model_compression.py --model vit-base-patch16-224 --methods pruning:structured
```

### Distillation

Knowledge distillation trains a smaller model using a larger model:

| Method | Description | Target Hardware |
|--------|-------------|----------------|
| `distillation:standard` | Standard knowledge distillation | All |
| `distillation:self` | Self-distillation | All |
| `distillation:token` | Token-level distillation (for sequence models) | All |

Examples:

```bash
# Apply standard distillation (requires training data)
python model_compression.py --model bert-large-uncased --methods distillation:standard

# Apply token-level distillation for transformer models
python model_compression.py --model t5-base --methods distillation:token
```

### Graph Optimization

Graph optimization restructures the model's computational graph:

| Method | Description | Target Hardware |
|--------|-------------|----------------|
| `graph_optimization:fusion` | Operator fusion | All |
| `graph_optimization:constant_folding` | Constant folding | All |
| `graph_optimization:onnx_graph` | ONNX graph optimizations | ONNX-compatible hardware |

Examples:

```bash
# Apply ONNX graph optimizations for OpenVINO
python model_compression.py --model bert-base-uncased --methods graph_optimization:onnx_graph --params '{"onnx_graph": {"target": "openvino"}}'

# Apply operator fusion
python model_compression.py --model t5-small --methods graph_optimization:fusion
```

## Compression Recommendations

The model compression system can recommend optimal compression methods based on model family and target hardware:

```bash
# Get compression recommendations for CPU deployment
python model_compression.py --model bert-base-uncased --target-hardware cpu --recommend

# Get compression recommendations for CUDA deployment
python model_compression.py --model gpt2 --target-hardware cuda --recommend
```

Example recommendations:

```
Compression Recommendations:

Model: bert-base-uncased
Type: embedding
Target Hardware: cpu

Recommended Methods:
  - quantization:dynamic
  - pruning:magnitude

Recommended Parameters:
  - dynamic: {'dtype': 'qint8'}
  - magnitude: {'sparsity': 0.3}

To apply these recommendations, run:
python model_compression.py --model bert-base-uncased --methods quantization:dynamic pruning:magnitude --params '{"dynamic": {"dtype": "qint8"}, "magnitude": {"sparsity": 0.3}}' --target-hardware cpu
```

## Compression Validation

The model compression system includes a validation system for comparing the performance of compressed models against the original models:

```bash
# Validate compression results
python model_compression.py --model bert-base-uncased --validate
```

The validation system measures:

1. **Latency**: Inference speed comparison between original and compressed models
2. **Memory Usage**: Memory usage comparison, including peak memory and model size
3. **Accuracy**: Accuracy comparison (when validation data is provided)

## Compression Reports

The model compression system generates comprehensive reports on compression results:

### JSON Results

Detailed compression results are saved in JSON format:

```json
{
  "model_name": "bert-base-uncased",
  "model_type": "embedding",
  "compression_steps": [
    {
      "method": "quantization:dynamic",
      "parameters": {
        "dtype": "qint8"
      },
      "success": true,
      "original_size": 438272000,
      "compressed_size": 109568000,
      "compression_ratio": 4.0
    },
    {
      "method": "pruning:magnitude",
      "parameters": {
        "sparsity": 0.3
      },
      "success": true,
      "original_size": 109568000,
      "compressed_size": 76697600,
      "compression_ratio": 1.43
    }
  ],
  "methods_applied": [
    "quantization:dynamic",
    "pruning:magnitude"
  ],
  "original_size": 438272000,
  "compressed_size": 76697600,
  "overall_compression_ratio": 5.72
}
```

### Markdown Reports

Human-readable compression reports are generated in Markdown format:

```markdown
# Model Compression Report

- **Model:** bert-base-uncased
- **Type:** embedding
- **Date:** 2025-03-02 14:30:45

## Compression Summary

- **Overall Compression Ratio:** 5.72x
- **Original Size:** 418.00 MB
- **Compressed Size:** 73.15 MB
- **Size Reduction:** 82.5%
- **Methods Applied:** quantization:dynamic, pruning:magnitude

## Compression Steps

### Step 1: quantization:dynamic

#### Parameters:
- **dtype:** qint8

#### Results:
- **Success:** Yes
- **Compression Ratio:** 4.00x
- **Size Before:** 418.00 MB
- **Size After:** 104.50 MB

### Step 2: pruning:magnitude

#### Parameters:
- **sparsity:** 0.3

#### Results:
- **Success:** Yes
- **Compression Ratio:** 1.43x
- **Size Before:** 104.50 MB
- **Size After:** 73.15 MB

## Validation Results

### Latency
- **Reference Model:** 0.0124 seconds
- **Compressed Model:** 0.0082 seconds
- **Speedup:** 1.51x

### Memory Usage
- **Reference Model Size:** 418.00 MB
- **Compressed Model Size:** 73.15 MB
- **Compression Ratio:** 5.72x

#### Peak Memory Usage (CUDA)
- **Reference Model:** 560.25 MB
- **Compressed Model:** 183.45 MB
- **Reduction Ratio:** 3.05x

## Recommendations

Based on the compression results, we recommend:

- ✅ The model has been significantly compressed. Consider deploying this compressed version.
- ✅ The compressed model shows significant speedup. Good for latency-critical applications.
```

## Integration with Resource Pool

The model compression system integrates with the ResourcePool for efficient model loading:

```python
# Using ResourcePool for model loading
pool = get_global_resource_pool()

# Load model using ResourcePool
model = pool.get_model(
    model_type=self.model_type,
    model_name=model_name,
    constructor=create_model,
    hardware_preferences=hardware_preferences
)
```

## Integration with Hardware Detection

The model compression system uses hardware detection for target-specific optimizations:

```python
# Get hardware capabilities
hardware_info = detect_hardware_with_comprehensive_checks()

# Check for specific hardware features
if hardware_info.get("cuda", False):
    # CUDA-specific compression strategies
    ...
elif hardware_info.get("mps", False):
    # MPS-specific compression strategies
    ...
```

## Integration with Model Family Classification

The model compression system uses model family classification for optimal compression strategies:

```python
# Classify model to understand its characteristics
classification = classify_model(model_name=model_name)
model_family = classification.get("family")

# Use model family information for compression strategy
if model_family == "embedding":
    # Optimal compression for embedding models
    ...
elif model_family == "text_generation":
    # Optimal compression for text generation models
    ...
```

## Best Practices

### Compression Strategy Selection

1. **Know Your Target Hardware**: Choose compression techniques compatible with your deployment hardware
2. **Consider Model Family**: Different model families respond differently to compression techniques
3. **Start with Quantization**: Quantization often gives the best compression-to-performance ratio
4. **Layer-wise Analysis**: Consider which layers contribute most to model size
5. **Combine Methods**: Often a combination of methods provides the best results
6. **Order Matters**: Apply pruning before quantization for best results

### Quantization Tips

1. **Dynamic Quantization**: Good first approach for CPU deployment
2. **Static Quantization**: Use when you have representative calibration data
3. **INT8 vs FP16**: Use INT8 for CPU, FP16 for GPU
4. **BFloat16**: Use for newer NVIDIA GPUs (Ampere/A100 and newer) or TPUs
5. **Weight-Only Quantization**: Consider when activation quantization causes accuracy problems

### Pruning Tips

1. **Start Conservative**: Begin with lower sparsity (20-30%)
2. **Structured vs Unstructured**: Structured pruning yields better speed-ups but more accuracy loss
3. **Layer-wise Pruning**: Apply different sparsity levels to different layers
4. **Iterative Pruning**: Gradually increase sparsity with fine-tuning between steps
5. **Attention Heads**: For transformers, consider pruning entire attention heads

### Validation Best Practices

1. **Representative Data**: Use representative data for validation
2. **Meaningful Metrics**: Choose metrics relevant to your application
3. **Latency Measurement**: Measure under realistic batch sizes and sequence lengths
4. **Accuracy Thresholds**: Define acceptable accuracy loss thresholds
5. **Multiple Hardware**: Test on all target deployment hardware

## Troubleshooting

### Common Issues

1. **Accuracy Degradation**: Reduce compression ratio or try different methods
2. **Slow Inference**: Check if the compression method is optimal for your hardware
3. **ONNX Export Failures**: Check for unsupported operations in your model
4. **Out of Memory**: Use smaller batch sizes or reduce model size further
5. **Hardware Compatibility**: Ensure compression method is supported by target hardware

### Debugging

```bash
# Enable debug logging
python model_compression.py --model bert-base-uncased --debug

# Check hardware compatibility
python model_compression.py --model bert-base-uncased --recommend --target-hardware cuda
```

## Advanced Usage

### Compression Parameter Tuning

Fine-tune compression parameters for optimal results:

```bash
# Fine-tune quantization parameters
python model_compression.py --model bert-base-uncased --methods quantization:dynamic --params '{"dynamic": {"dtype": "qint8", "modules": ["Linear", "LayerNorm"]}}'

# Fine-tune pruning parameters
python model_compression.py --model bert-base-uncased --methods pruning:magnitude --params '{"magnitude": {"sparsity": 0.3, "min_magnitude": 1e-5}}'
```

### Custom Compression Workflows

Combine multiple compression methods with specific parameters:

```bash
# Custom compression workflow
python model_compression.py --model bert-base-uncased --methods pruning:magnitude quantization:dynamic graph_optimization:fusion --params '{"magnitude": {"sparsity": 0.3}, "dynamic": {"dtype": "qint8"}, "fusion": {"enabled": true}}'
```

### Hardware-Specific Compression

Optimize for specific hardware platforms:

```bash
# Optimize for NVIDIA GPUs
python model_compression.py --model gpt2 --methods quantization:fp16 --target-hardware cuda

# Optimize for Intel CPUs with OpenVINO
python model_compression.py --model bert-base-uncased --methods quantization:int8 graph_optimization:onnx_graph --params '{"onnx_graph": {"target": "openvino"}}' --target-hardware openvino

# Optimize for Apple Silicon
python model_compression.py --model bert-base-uncased --methods quantization:dynamic --target-hardware mps
```

## Further Resources

- [ResourcePool Guide](RESOURCE_POOL_GUIDE.md): Details on the ResourcePool for model caching
- [Hardware Detection Guide](HARDWARE_DETECTION_GUIDE.md): Information on the hardware detection system
- [Model Family Classifier Guide](MODEL_FAMILY_CLASSIFIER_GUIDE.md): Details on model family classification
- [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md): Guide to performance benchmarking
- [Hardware Platform Test Guide](HARDWARE_PLATFORM_TEST_GUIDE.md): Guide to hardware platform testing
- [Hardware Model Validation Guide](HARDWARE_MODEL_VALIDATION_GUIDE.md): Guide to model validation

---

*This guide is part of the IPFS Accelerate Python Framework documentation*