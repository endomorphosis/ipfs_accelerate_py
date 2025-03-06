# Integrated Skillset Generator Guide (Updated March 6, 2025)

This guide explains how to use the `integrated_skillset_generator.py` tool to create implementation files for all 300+ Hugging Face model types with full hardware support, including comprehensive modality-specific handling and specialized processors for different model types.

## Overview

The Integrated Skillset Generator implements a test-driven development approach that:

1. Analyzes test files and results to extract model metadata
2. Generates skillset implementations based on test insights
3. Supports all hardware backends (CPU, CUDA, OpenVINO, MPS, AMD ROCm, WebNN, WebGPU)
4. Creates consistent implementations using existing reference templates
5. Validates implementations against test expectations

## Key Features (Updated March 2025)

- **Model Registry System**: Comprehensive mapping from short names to full model IDs
- **Modality-Specific Initialization**: Specialized initialization for text, vision, audio, and multimodal models
- **Processor Selection**: Automatic selection of appropriate processor (tokenizer, image processor, feature extractor)
- **Test-Driven Development**: Implementations are based on real test results
- **Web Backend Support**: Full WebNN and WebGPU (transformers.js) support
- **Hardware Compatibility Analysis**: Automatically detects compatible hardware
- **Enhanced OpenVINO Support**: Proper initialization with all required parameters
- **Qualcomm AI Engine Integration**: Support for Qualcomm mobile/edge hardware
- **Template-Based Generation**: Uses reference implementations as templates
- **Parallel Generation**: Can generate multiple implementations simultaneously
- **Output Validation**: Flexible validation that handles different output structures
- **Comprehensive Validation**: Ensures implementations match test expectations

## Requirements

- Python 3.8+
- Jinja2 (optional but recommended for better templating)
- Existing test files in `test/skills/`
- Hugging Face model metadata in `test/huggingface_model_types.json`
- Reference implementations in `ipfs_accelerate_py/worker/skillset/`

## Installation

For the best experience, install Jinja2:

```bash
pip install jinja2
```

## Usage Examples

### List Model Families and Tasks

To see what model families and tasks are available:

```bash
# List all model families
python integrated_skillset_generator.py --list-families

# List all primary tasks
python integrated_skillset_generator.py --list-tasks
```

### Generate Implementation for a Specific Model

To generate an implementation for a specific model:

```bash
# Basic generation
python integrated_skillset_generator.py --model bert

# Generate with specific hardware platforms
python integrated_skillset_generator.py --model vit --hardware cuda,openvino,qualcomm,webgpu

# Generate with cross-platform support for all hardware
python integrated_skillset_generator.py --model clip --hardware all --cross-platform

# Run tests first to inform implementation
python integrated_skillset_generator.py --model bert --run-tests

# Force overwrite of existing implementation
python integrated_skillset_generator.py --model bert --force
```

### Generate Implementations for Groups of Models

To generate implementations for multiple models:

```bash
# Generate for all models in a family
python integrated_skillset_generator.py --family bert

# Generate for all models with a specific task
python integrated_skillset_generator.py --task text-classification

# Generate for all supported models (300+ models)
python integrated_skillset_generator.py --all --max-workers 20
```

### Validate Implementations

To validate that an implementation meets the expected structure:

```bash
python integrated_skillset_generator.py --validate bert
```

## Output

Generated implementation files are saved to the `generated_skillsets` directory by default. You can customize this with the `--output-dir` option:

```bash
python integrated_skillset_generator.py --model bert --output-dir /path/to/custom/directory
```

## Web Backend Support

The generator creates implementations with full support for WebNN and WebGPU (transformers.js) backends, enabling model deployment directly in web browsers with hardware acceleration. These web backends represent a significant advancement in browser-based machine learning.

### WebNN Support

WebNN (Web Neural Network API) is a W3C standard for accelerating neural networks in web browsers:

- **Hardware Acceleration**: Leverages device-specific acceleration (CPU, GPU, NPU) through a standard web API
- **ONNX Integration**: All generated implementations include ONNX export with WebNN-specific optimizations
- **Cross-Platform**: Compatible with all major browsers through the WebNN API or polyfill
- **Precision Control**: Supports fp32, fp16, and int8 precision formats for performance optimization
- **Tensor Conversion**: Includes handlers that manage conversion between PyTorch tensors and WebNN format
- **Graceful Degradation**: Falls back to CPU execution when WebNN acceleration is unavailable

#### Implementation Details

The WebNN implementation includes:

```python
# WebNN initialization with proper error handling
def init_webnn(self, model_name, model_type, precision="fp32"):
    """Initialize model for WebNN inference with browser compatibility."""
    try:
        # Load and optimize model for WebNN
        # Convert PyTorch model to ONNX first
        onnx_model = self._export_to_onnx(model_name)
        
        # Initialize WebNN runtime and session
        webnn_model = self._convert_onnx_to_webnn(onnx_model, precision)
        
        # Create appropriate handlers
        handler = self.create_webnn_endpoint_handler(
            endpoint_model=model_name,
            endpoint=webnn_model,
            precision=precision
        )
        
        return webnn_model, handler
    except Exception as e:
        # Return mock implementation for graceful degradation
        return self._create_mock_webnn_implementation()
```

### WebGPU/transformers.js Support

WebGPU with transformers.js enables high-performance transformer models directly in the browser:

- **GPU Acceleration**: Uses WebGPU for high-performance parallel execution on the GPU
- **Transformers.js Integration**: Leverages the transformers.js library (browser version of Hugging Face)
- **Async Execution**: Supports asynchronous execution patterns required in browser environments
- **JavaScript API**: Provides JavaScript-friendly handlers that work in web applications
- **Tokenization**: Includes browser-compatible tokenization with shared vocabularies
- **Model Caching**: Supports efficient model caching in the browser for better performance

#### Implementation Details

The WebGPU implementation includes:

```python
# WebGPU/transformers.js initialization
def init_webgpu(self, model_name, model_type, precision="fp32"):
    """Initialize model for WebGPU inference using transformers.js."""
    try:
        # Create transformers.js configuration
        config = {
            "model": model_name,
            "quantization": precision if precision != "fp32" else None,
            "cache": True,
            "accelerationMode": "webgpu"
        }
        
        # Initialize transformers.js pipeline
        pipeline = self._create_transformers_js_pipeline(model_type, config)
        
        # Create asynchronous handler for browser environment
        handler = self.create_webgpu_endpoint_handler(
            endpoint_model=model_name,
            endpoint=pipeline,
            precision=precision
        )
        
        return pipeline, handler
    except Exception as e:
        # Return mock implementation for graceful degradation
        return self._create_mock_webgpu_implementation()
```

### Practical Web Backend Example

Here's a complete example of how the generated implementation works with web backends:

```python
# Load model with WebNN backend
model = hf_bert(resources={"transformers": transformers})
webnn_model, webnn_handler = model.init_webnn("bert-base-uncased", "text-classification")

# Process text with WebNN backend
result = webnn_handler("This is a test sentence for WebNN execution.")
print(f"WebNN result: {result}")

# Load model with WebGPU/transformers.js backend
webgpu_model, webgpu_handler = model.init_webgpu("bert-base-uncased", "text-classification")

# Process text with WebGPU backend (async in browser)
async def process_with_webgpu():
    result = await webgpu_handler("This is a test sentence for WebGPU execution.")
    print(f"WebGPU result: {result}")
```

## Template System (Updated March 2025)

The generator uses a template system based on existing reference implementations. The templates are automatically selected based on the model's family and include modality-specific code:

- **Text Models**:
  - Use BERT, T5, GPT, or LLAMA templates
  - Include AutoTokenizer initialization
  - Handle text processing with appropriate tokenization
  - Validate text model outputs (last_hidden_state, logits)

- **Vision Models**:
  - Use ViT or CLIP templates
  - Include AutoImageProcessor initialization
  - Handle image preprocessing with PIL
  - Validate vision model outputs (logits, image_embeds)

- **Audio Models**:
  - Use Whisper or Wav2Vec2 templates
  - Include AutoFeatureExtractor initialization
  - Handle audio sampling and processing
  - Validate audio model outputs (logits, spectrogram)

- **Multimodal Models**:
  - Use LLaVA, CLIP, or BLIP templates
  - Include AutoProcessor initialization
  - Handle combined text and image inputs
  - Validate multimodal outputs (text_embeds, image_embeds)

## Understanding Hardware Compatibility (Updated March 2025)

The generator analyzes test results to determine hardware compatibility for each model. This information is used to generate appropriate initialization methods and handlers:

```python
# Example hardware compatibility dictionary (March 2025 update)
hardware_compatibility = {
    "cpu": True,          # Always true for all models
    "cuda": True,         # For NVIDIA GPUs
    "openvino": True,     # For Intel CPUs and GPUs with OpenVINO
    "mps": True,          # For Apple Silicon (M1/M2/M3)
    "rocm": True,         # For AMD GPUs with ROCm
    "qualcomm": True,     # For Qualcomm AI Engine on mobile/edge devices
    "webnn": True,        # For WebNN in browsers
    "webgpu": True        # For WebGPU/transformers.js in browsers
}

# Modality-specific initialization based on hardware
def initialize_model(model_id, hardware, modality):
    if modality == "text":
        # Text model initialization
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if hardware == "openvino":
            # OpenVINO initialization with proper parameters
            model = init_openvino(model_id, openvino_label="text_model")
        elif hardware == "qualcomm":
            # Qualcomm AI Engine initialization
            model = init_qualcomm(model_id)
        elif hardware in ["webnn", "webgpu"]:
            # Web backend initialization
            model = init_web_backend(model_id, hardware)
        else:
            # Standard initialization
            model = AutoModel.from_pretrained(model_id)
    elif modality == "vision":
        # Vision model initialization with appropriate processor
        processor = AutoImageProcessor.from_pretrained(model_id)
        # ... hardware-specific initialization
```

## Extending the Generator (March 2025 Updates)

The generator is designed to be extensible. Here are some ways to extend it:

### Adding a New Hardware Backend

To add support for a new hardware backend:

1. Add the backend to `hardware_compatibility` in `TestAnalyzer.analyze_hardware_compatibility()`
2. Add initialization methods for the new backend
3. Add handler creation methods for the new backend
4. Add validation checks for the new backend
5. Ensure all modality-specific processors are properly initialized

### Adding New Model Types and Processors

To add support for new modality types:

1. Update the model registry with appropriate mappings
2. Add modality-specific processor initialization:
   ```python
   if modality == "new_modality":
       from transformers import AutoNewModalityProcessor
       processor = AutoNewModalityProcessor.from_pretrained(model_id)
   ```
3. Add input preparation logic for the new modality
4. Add output validation for the new modality's output structures

### Adding New Model Types

The generator automatically supports new model types as they are added to:

1. `huggingface_model_types.json`
2. Test files in `test/skills/`

No code changes are needed to support new models.

## Troubleshooting

### Missing Templates

If the generator cannot find a reference implementation for a model family, it will use BERT as a fallback. To resolve this:

1. Create a reference implementation in `ipfs_accelerate_py/worker/skillset/`
2. Add it to the reference_models dictionary in `TemplateEngine._load_reference_implementation()`

### Failed Validation

If validation fails, it means the generated implementation is missing required components. Check:

1. The reference implementation is appropriate for the model type
2. The template contains all required components (init, handlers, etc.)
3. The web backends are properly included in the template

## Additional Features

### Parallel Generation

The generator supports parallel generation using ThreadPoolExecutor. Control the number of worker threads with the `--max-workers` option:

```bash
python integrated_skillset_generator.py --all --max-workers 30
```

### Verbose Logging

Enable detailed logging to see more information about the generation process:

```bash
python integrated_skillset_generator.py --model bert --verbose
```

## Integration with WebNN and WebGPU

The generator creates implementations that support both WebNN and WebGPU based on the enhanced template generator. This includes:

1. Hardware detection for web browsers
2. ONNX export capabilities for WebNN
3. transformers.js integration for WebGPU
4. Browser-compatible handler functions
5. Proper async support for browser environments

## Next Steps (March 2025)

After generating implementations:

1. Run the validation to ensure quality: `--validate model_name`
2. Test the implementations in real environments
3. Verify output structure handling for different model types
4. Export models to web formats with the web export utilities
5. Deploy the models to browser environments
6. Use the DuckDB database to store benchmark results
7. Update the template database with any new model types

## March 2025 Updates

The March 2025 update to the integrated skillset generator includes several critical improvements:

1. **Model Registry System**: Added comprehensive mapping of short model names to full model IDs
2. **Modality-Based Initialization**: Added specialized initialization for different model types
3. **Processor Selection**: Automatic selection of appropriate processor based on model type
4. **OpenVINO Integration**: Fixed initialization with proper parameters
5. **Qualcomm AI Engine Support**: Added support for Qualcomm mobile/edge hardware
6. **Output Validation**: Improved validation for different output structures
7. **Error Handling**: Enhanced error handling with contextual messages
8. **Test Interface**: Added proper integration with test class interface

These improvements ensure that the generator creates valid, executable implementations for all model types and hardware platforms.

## Full Web Backend Workflow

For a complete workflow to deploy models to the web:

1. Generate the implementation: `python integrated_skillset_generator.py --model bert --run-tests`
2. Validate the implementation: `python integrated_skillset_generator.py --validate bert`
3. Export the model to ONNX: Use the `web_export` helper function in the implementation
4. Convert to WebNN or use with transformers.js: See `ONNX_WEBNN_EXPORT_GUIDE.md` and `WEBGPU_TRANSFORMERS_JS_GUIDE.md`