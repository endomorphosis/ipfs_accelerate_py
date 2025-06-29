# ONNX and WebNN Export Guide

This comprehensive guide covers how to export models from the IPFS Accelerate Python framework to ONNX and WebNN formats, with a particular focus on JavaScript/Node.js implementation for cross-platform deployment.

## Overview

The enhanced model registry now includes comprehensive export capabilities that allow you to:

1. **Export to ONNX format** - Universal model format supported by numerous runtime environments
2. **Export to WebNN format** - For deployment in web browsers and edge devices
3. **Perform hardware-specific optimizations** - Optimize exports for target hardware including AMD GPUs
4. **Apply precision-specific settings** - Configure exports for different precision needs
5. **Generate JavaScript inference code** - Ready-to-use JS/Node.js implementation templates
6. **Access complete model architecture details** - All parameters needed for proper conversion and deployment

## Getting Started with Model Export

### Basic ONNX Export

```python
from model_export_capability import export_model
from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained("bert-base-uncased")

# Export to ONNX
success, message = export_model(
    model=model,
    model_id="bert-base-uncased",
    output_path="model.onnx",
    export_format="onnx"
)

if success:
    print(f"Success: {message}")
else:
    print(f"Failed: {message}")
```

### Hardware-Specific Export

```python
from model_export_capability import export_model
from transformers import AutoModel
from auto_hardware_detection import detect_all_hardware

# First detect available hardware
hardware = detect_all_hardware()
detected_hw = [hw for hw, info in hardware.items() if info.detected]
primary_hw = detected_hw[0] if detected_hw else "cpu"

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Export with hardware-specific optimizations
success, message = export_model(
    model=model,
    model_id="bert-base-uncased",
    output_path="optimized_model.onnx",
    export_format="onnx",
    hardware_target=primary_hw,
    precision="fp16" if primary_hw in ["cuda", "amd"] else "fp32"
)
```

### WebNN Export for Web Deployment

```python
from model_export_capability import export_model
from transformers import AutoModelForSequenceClassification

# Load a smaller model suitable for web deployment
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Export to WebNN format (via ONNX)
success, message = export_model(
    model=model,
    model_id="distilbert-base-uncased",
    output_path="web_model_dir",  # Directory to store WebNN files
    export_format="webnn",
    hardware_target="cpu",  # WebNN typically targets CPU or WASM
    precision="fp16"  # Use FP16 for smaller model size
)
```

## Analyzing Model Export Compatibility

Before exporting, you can analyze if a model is compatible with your target export format:

```python
from model_export_capability import analyze_model_export_compatibility
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Analyze compatibility
report = analyze_model_export_compatibility(
    model=model,
    model_id="bert-base-uncased",
    formats=["onnx", "webnn"]
)

# Check ONNX compatibility
if report["formats"]["onnx"]["compatible"]:
    print("Model is ONNX compatible")
    print("Recommended hardware:", report["formats"]["onnx"]["recommended_hardware"])
    print("Recommended settings:", report["formats"]["onnx"]["recommended_config"])
else:
    print("Model has ONNX compatibility issues:")
    for issue in report["formats"]["onnx"]["issues"]:
        print(f"- {issue}")

# Check WebNN compatibility
if report["formats"]["webnn"]["compatible"]:
    print("Model is WebNN compatible")
else:
    print("Model has WebNN compatibility issues:")
    for issue in report["formats"]["webnn"]["issues"]:
        print(f"- {issue}")
```

## Export Configuration Options

The export system supports comprehensive configuration options:

```python
from model_export_capability import ExportConfig, export_model
from transformers import AutoModel

# Create custom export configuration
config = ExportConfig(
    format="onnx",
    opset_version=13,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    },
    optimization_level=2,
    target_hardware="cuda",
    precision="fp16",
    quantize=False,
    simplify=True,
    constant_folding=True,
    verbose=True,
    additional_options={"custom_setting": True}
)

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Export with custom configuration
export_model(
    model=model,
    model_id="bert-base-uncased",
    output_path="custom_model.onnx",
    export_format="onnx",
    custom_config=config
)
```

## AMD GPU-Specific Optimizations

The export system includes special optimizations for AMD GPUs with ROCm:

```python
from model_export_capability import export_model
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Export with AMD-specific optimizations
success, message = export_model(
    model=model,
    model_id="bert-base-uncased",
    output_path="amd_optimized.onnx",
    export_format="onnx",
    hardware_target="amd",
    precision="fp16"  # FP16 is generally best on AMD GPUs
)
```

Key AMD-specific optimizations include:
- FP16 precision settings optimized for AMD hardware
- Dynamic shape handling compatible with ROCm
- Optimized performance for AMD's architecture

## Supported Input/Output Specifications

The export system captures detailed information about model inputs and outputs:

```python
from model_export_capability import get_model_export_capability
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Get export capability information
capability = get_model_export_capability("bert-base-uncased", model)

# Display input information
print("Model inputs:")
for i, inp in enumerate(capability.inputs):
    print(f"Input {i}: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Data type: {inp.dtype}")
    print(f"  Dynamic: {inp.is_dynamic}")
    print(f"  Required: {inp.is_required}")
    print(f"  Typical shape: {inp.typical_shape}")

# Display output information
print("\nModel outputs:")
for i, out in enumerate(capability.outputs):
    print(f"Output {i}: {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Data type: {out.dtype}")
    print(f"  Dynamic: {out.is_dynamic}")
    print(f"  Typical shape: {out.typical_shape}")
```

## Precision Types

Different precision types are supported for export optimization:

| Precision | Description | Hardware Support |
|-----------|-------------|------------------|
| fp32 | Full 32-bit precision | All platforms |
| fp16 | Half precision (16-bit) | CUDA, AMD, MPS |
| bf16 | Brain floating point | CUDA (Ampere+), AMD (CDNA2+), CPU (with AVX2) |
| int8 | 8-bit integer quantization | CPU, CUDA, AMD, OpenVINO |
| int4 | 4-bit integer quantization | CUDA, OpenVINO |
| uint4 | Unsigned 4-bit quantization | CUDA, OpenVINO |

Each precision type is automatically matched to hardware capabilities:

```python
from model_export_capability import get_optimized_export_config

# Get optimized config for AMD with FP16
amd_config = get_optimized_export_config(
    model_id="bert-base-uncased",
    export_format="onnx",
    hardware_target="amd",
    precision="fp16"
)

# Get optimized config for OpenVINO with INT8
openvino_config = get_optimized_export_config(
    model_id="bert-base-uncased",
    export_format="onnx",
    hardware_target="openvino",
    precision="int8"
)
```

## WebNN Export Details

WebNN export is a two-step process:

1. Export to ONNX as an intermediate format
2. Convert ONNX to WebNN-compatible format

WebNN has some limitations:
- Limited operation support compared to ONNX
- Model size constraints
- Precision limitations (mainly FP32 and FP16)

```python
from model_export_capability import export_to_webnn
from transformers import AutoModelForSequenceClassification
import torch

# Load a small model for web deployment
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Prepare sample inputs
inputs = {
    "input_ids": torch.ones(1, 64, dtype=torch.long),
    "attention_mask": torch.ones(1, 64, dtype=torch.long)
}

# Export directly to WebNN
success, message = export_to_webnn(
    model=model,
    inputs=inputs,
    output_dir="webnn_model"
)
```

## Mixing AMD Hardware with WebNN Export

When using AMD hardware to prepare models for WebNN export, some special considerations apply:

```python
from model_export_capability import export_model, ExportConfig
from transformers import AutoModel
import torch

# Load a model
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Create custom configuration for AMD optimization + WebNN compatibility
config = ExportConfig(
    format="webnn",
    opset_version=12,  # WebNN works best with opset 12
    target_hardware="amd",
    precision="fp16",
    simplify=True,
    constant_folding=True,
    additional_options={
        "optimize_for_web": True,
        "minimize_model_size": True
    }
)

# Export to WebNN using AMD hardware for preparation
export_model(
    model=model,
    model_id="distilbert-base-uncased",
    output_path="amd_optimized_webnn",
    export_format="webnn",
    custom_config=config
)
```

## Command-Line Export Tool

The export capability can be used from the command line:

```bash
# Export a model to ONNX format
python model_export_capability.py --model bert-base-uncased --format onnx --output model.onnx

# Export with hardware and precision specifications
python model_export_capability.py --model bert-base-uncased --format onnx --hardware amd --precision fp16 --output amd_model.onnx

# Just analyze compatibility without exporting
python model_export_capability.py --model bert-base-uncased --format webnn --analyze
```

## Best Practices

1. **Start with compatibility analysis**:
   Always run `analyze_model_export_compatibility()` before attempting export to identify potential issues.

2. **Match precision to hardware**:
   Use the appropriate precision for your target hardware:
   - AMD GPUs: FP16 for best performance
   - NVIDIA GPUs: FP16 or BF16 (Ampere+)
   - CPU: FP32 or INT8 (with quantization)
   - Web/Edge: FP16 for most web deployments

3. **Use dynamic axes for variable inputs**:
   Always specify dynamic axes for dimensions that can change (batch size, sequence length).

4. **Understand WebNN limitations**:
   WebNN has stricter compatibility requirements than ONNX:
   - Model size should be under 100MB for good performance
   - Not all operations are supported
   - Quantized models might have limited support

5. **Test the exported model**:
   Always validate exported models with the target runtime (ONNX Runtime, WebNN).

## Troubleshooting

### Common Issues with ONNX Export

1. **Dynamic shape errors**:
   - Symptom: "Broadcasting is not supported with dynamic shapes"
   - Solution: Specify explicit dynamic_axes in ExportConfig

2. **Unsupported operations**:
   - Symptom: "Error: Unsupported operation: CustomOperation"
   - Solution: Check if your model uses custom PyTorch operations and consider alternatives

3. **Memory issues with large models**:
   - Symptom: Out of memory during export
   - Solution: Export on a machine with more RAM or use quantization to reduce model size

### Common Issues with WebNN Export

1. **Model size limitations**:
   - Symptom: "Model too large for efficient WebNN deployment"
   - Solution: Use a smaller model or apply quantization/pruning

2. **Operation support**:
   - Symptom: "Operation X is not supported in WebNN"
   - Solution: Use simpler architectures or consider model modification

3. **Precision compatibility**:
   - Symptom: "Parameter has dtype that may not be supported in WebNN"
   - Solution: Ensure model uses FP32 or FP16 precision

## Conclusion

With the export capabilities added to the IPFS Accelerate Python framework, you can now easily deploy models across different platforms, optimizing for specific hardware and precision requirements. The system's integration with AMD hardware detection ensures optimal export settings for AMD GPUs, making it easier to prepare models for deployment on AMD hardware or for WebNN-compatible environments.

## JavaScript Implementation with WebNN

The enhanced model registry now includes ready-to-use JavaScript code templates for implementing WebNN inference. This makes it much easier to convert your PyTorch model to WebNN and build JavaScript implementations:

```javascript
// Example of using the generated JavaScript code
import { bertPipeline } from './generated_bert_code.js';

// Load the model and run inference
async function main() {
  const text = "This is a test sentence for the BERT model.";
  const modelPath = "model.onnx";
  
  try {
    // Run the full pipeline
    const results = await bertPipeline(text, modelPath);
    
    // Process results
    console.log("Inference results:", results);
    
    // Access the output tensor (e.g., last_hidden_state)
    const outputTensor = results.last_hidden_state;
    console.log("Output shape:", outputTensor.dims);
    console.log("Output data:", outputTensor.data.slice(0, 10)); // First 10 values
  } catch (error) {
    console.error("Inference failed:", error);
  }
}

main();
```

### Node.js Implementation

For server-side or Node.js environments, the model registry provides Node.js specific templates:

```javascript
// Node.js implementation for ONNX models
const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

// Load model and run inference
async function runInference(modelPath, inputData) {
  try {
    // Load the model
    const session = await ort.InferenceSession.create(modelPath);
    
    // Prepare input tensors
    const feeds = {};
    for (const [name, data] of Object.entries(inputData)) {
      feeds[name] = new ort.Tensor(
        name.includes('input_ids') ? 'int64' : 'float32',
        data,
        Array.isArray(data) ? [1, data.length] : data.shape
      );
    }
    
    // Run inference
    const results = await session.run(feeds);
    return results;
  } catch (error) {
    console.error('Inference failed:', error);
    throw error;
  }
}

module.exports = { runInference };
```

The model registry provides all the necessary information for both browser and Node.js implementations, ensuring you can easily deploy your models in any JavaScript environment.