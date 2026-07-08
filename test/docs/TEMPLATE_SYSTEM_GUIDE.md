# IPFS Accelerate Test Template System Guide

This document describes the test template system for the IPFS Accelerate project, which enables the creation of standardized test files for models, hardware platforms, and APIs.

## Overview

The template system provides a structured way to generate test files with consistent patterns, appropriate imports, and standardized testing methodologies. This ensures that tests across the project follow the same conventions and can be easily maintained.

The system includes:

1. **Base Template Class**: Provides common functionality for all templates
2. **Specialized Templates**:
   - **Model Test Template**: For testing ML models (text, vision, audio, multimodal)
   - **Hardware Test Template**: For testing hardware platforms (WebGPU, WebNN, CUDA, ROCm)
   - **API Test Template**: For testing API endpoints and clients

## Directory Structure

The template system follows this directory structure:

```
test/
├── template_system/
│   ├── __init__.py
│   ├── generate_test.py               # CLI tool for generating tests
│   └── templates/
│       ├── __init__.py
│       ├── base_template.py           # Base template class
│       ├── model_test_template.py     # Model-specific template
│       ├── hardware_test_template.py  # Hardware-specific template
│       └── api_test_template.py       # API-specific template
└── docs/
    └── TEMPLATE_SYSTEM_GUIDE.md       # This guide
```

## Using the Template System

### Command-Line Interface

The most straightforward way to use the template system is through the command-line interface provided by the `generate_test.py` script:

```bash
# Generate a model test
python -m test.template_system.generate_test model \
  --model-name bert-base-uncased \
  --model-type text \
  --framework transformers

# Generate a hardware test
python -m test.template_system.generate_test hardware \
  --hardware-platform webgpu \
  --test-name webgpu_matmul \
  --test-operation matmul \
  --test-category compute

# Generate an API test
python -m test.template_system.generate_test api \
  --api-name openai \
  --test-name openai_client \
  --api-type openai
```

### Python API

You can also use the template system programmatically in your Python code:

```python
from test.template_system.templates.model_test_template import ModelTestTemplate
from test.template_system.templates.hardware_test_template import HardwareTestTemplate
from test.template_system.templates.api_test_template import APITestTemplate

# Generate a model test
model_template = ModelTestTemplate(
    model_name="bert-base-uncased",
    model_type="text",
    framework="transformers"
)
model_test_path = model_template.generate()

# Generate a hardware test
hardware_template = HardwareTestTemplate(
    parameters={
        "hardware_platform": "webgpu",
        "test_name": "webgpu_matmul",
        "test_operation": "matmul",
        "test_category": "compute"
    },
    output_dir="test"
)
hardware_test_path = hardware_template.write()

# Generate an API test
api_template = APITestTemplate(
    parameters={
        "api_name": "openai",
        "test_name": "openai_client",
        "api_type": "openai"
    },
    output_dir="test"
)
api_test_path = api_template.write()
```

## Template Types

### Model Test Template

The Model Test Template generates tests for ML models with specific handling for different model types:

- **Text Models**: BERT, T5, GPT, etc.
- **Vision Models**: ViT, ResNet, etc.
- **Audio Models**: Whisper, Wav2Vec, etc.
- **Multimodal Models**: CLIP, LayoutLM, etc.

**Parameters**:

- **model_name**: Name of the model (e.g., `bert-base-uncased`)
- **model_type**: Type of model (`text`, `vision`, `audio`, `multimodal`)
- **framework**: Framework used (`transformers`, `torch`, `tensorflow`, `onnx`)
- **batch_size**: Batch size for testing (default: 1)
- **output_dir**: Output directory (default: derived from model type and name)
- **overwrite**: Whether to overwrite existing files (default: False)

**Example**:

```python
from test.template_system.templates.model_test_template import ModelTestTemplate

template = ModelTestTemplate(
    model_name="bert-base-uncased",
    model_type="text",
    framework="transformers",
    batch_size=2
)
test_path = template.generate()
```

### Hardware Test Template

The Hardware Test Template generates tests for specific hardware platforms, including device detection, computation tests, and hardware-specific capabilities.

**Parameters**:

- **hardware_platform**: Hardware platform to test (`webgpu`, `webnn`, `cuda`, `rocm`, `cpu`)
- **test_name**: Name for the test (e.g., `webgpu_matmul`)
- **test_operation**: Operation to test (`matmul`, `conv`, `inference`)
- **test_category**: Category of test (`compute`, `memory`, `throughput`, `latency`)
- **output_dir**: Output directory (default: `test`)

**Example**:

```python
from test.template_system.templates.hardware_test_template import HardwareTestTemplate

template = HardwareTestTemplate(
    parameters={
        "hardware_platform": "webgpu",
        "test_name": "webgpu_matmul",
        "test_operation": "matmul",
        "test_category": "compute"
    },
    output_dir="test"
)
test_path = template.write()
```

### API Test Template

The API Test Template generates tests for API endpoints, clients, and integrations with specific handling for different API types.

**Parameters**:

- **api_name**: Name of the API (e.g., `openai`)
- **test_name**: Name for the test (e.g., `openai_client`)
- **api_type**: Type of API (`openai`, `hf_tei`, `hf_tgi`, `ollama`, `vllm`, `claude`, `internal`)
- **output_dir**: Output directory (default: `test`)

**Example**:

```python
from test.template_system.templates.api_test_template import APITestTemplate

template = APITestTemplate(
    parameters={
        "api_name": "openai",
        "test_name": "openai_client",
        "api_type": "openai"
    },
    output_dir="test"
)
test_path = template.write()
```

## Generated Tests

The template system generates test files with the following structure:

1. **File Header**: Includes file description and auto-generation note
2. **Imports**: Framework-specific imports and common utilities
3. **Fixtures**: Test fixtures for setup and resources
4. **Test Class**: Main test class with test methods
5. **Main Section**: For running the test directly

The generated tests include:

- **Basic Tests**: Basic loading and inference tests
- **Batch Tests**: Tests for batch processing
- **Device Tests**: Tests for different hardware devices
- **Mock Tests**: Tests with mock objects for API tests

## Customizing Templates

To customize the generated tests, you can:

1. **Subclass Existing Templates**: Create custom templates that inherit from the base templates
2. **Override Template Methods**: Override specific methods to change behavior
3. **Add Hooks**: Use the `before_generate` and `after_generate` hooks to add custom behavior

**Example**:

```python
from test.template_system.templates.model_test_template import ModelTestTemplate

class CustomModelTemplate(ModelTestTemplate):
    def generate_imports(self) -> str:
        """Add custom imports."""
        imports = super().generate_imports()
        imports += "\n# Custom imports\nimport custom_module\n"
        return imports
    
    def after_generate(self) -> None:
        """Custom actions after generation."""
        super().after_generate()
        print("Custom template generated")
```

## Adding New Template Types

To add a new template type:

1. Create a new Python module in the `test/template_system/templates/` directory
2. Subclass `BaseTemplate` or another template class
3. Override the necessary methods
4. Add support to the `generate_test.py` script

## Best Practices

When using the template system:

1. **Use the CLI for Simple Cases**: The CLI is the easiest way to generate tests
2. **Use the Python API for Complex Cases**: For batch generation or custom logic
3. **Follow Naming Conventions**: Use consistent names for tests
4. **Organize Tests Logically**: Place tests in the appropriate directories
5. **Add Appropriate Markers**: Use pytest markers to categorize tests
6. **Document Custom Templates**: Document any custom templates you create

## Troubleshooting

Common issues and solutions:

1. **Import Errors**: Ensure the Python path includes the project root
2. **Permission Errors**: Ensure you have write permission to the output directory
3. **File Exists**: Use the `--overwrite` flag to overwrite existing files
4. **Template Not Found**: Ensure the template module is in the Python path
5. **Hardware Detection**: Ensure hardware detection is set up correctly for hardware tests

## Conclusion

The template system provides a powerful way to generate standardized test files for the IPFS Accelerate project. By using the templates, you can ensure that your tests follow project conventions and include all necessary components.

For more information, see the source code in the `test/template_system/` directory.