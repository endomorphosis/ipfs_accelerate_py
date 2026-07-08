# IPFS Accelerate Testing Framework Refactoring Plan

## Overview

This document outlines the comprehensive plan for refactoring the IPFS Accelerate testing framework. The goal is to create a more cohesive, maintainable, and discoverable test organization with a single entry point for running tests in CI/CD environments.

## Current Challenges

1. **Fragmented Test Organization**: Tests are scattered across multiple directories without a consistent structure.
2. **Multiple Entry Points**: Different test suites have different entry points, making CI/CD integration complex.
3. **Duplicate Utilities**: Common functionality is duplicated across test files.
4. **Inconsistent Hardware Detection**: Hardware capability detection is implemented differently across tests.
5. **Poor Test Discovery**: Difficult to locate and understand available tests.
6. **Limited CI/CD Integration**: CI pipeline configuration is complex due to the fragmented structure.

## Refactoring Goals

1. **Unified Directory Structure**: Organize tests by logical category, improving discoverability.
2. **Single Entry Point**: Create a unified test runner that simplifies test execution.
3. **Standardized Utilities**: Develop common utilities for hardware detection, model handling, etc.
4. **Consistent Test Patterns**: Establish consistent patterns for test implementation.
5. **Improved CI/CD Integration**: Simplify CI/CD configuration and execution.
6. **Backward Compatibility**: Ensure existing tests continue to work during migration.
7. **Support for Distributed Testing**: Maintain compatibility with distributed testing infrastructure.

## New Directory Structure

```
test/
├── run.py                    # Unified test runner
├── conftest.py               # Global pytest configuration
├── pytest.ini                # Pytest configuration
├── common/                   # Common utilities
│   ├── __init__.py
│   ├── hardware_detection.py # Hardware detection utilities
│   ├── model_helpers.py      # Model utilities
│   └── fixtures.py           # Common test fixtures
├── models/                   # Model-specific tests
│   ├── __init__.py
│   ├── text/                 # Text models
│   │   ├── bert/             # BERT models
│   │   ├── t5/               # T5 models
│   │   └── gpt/              # GPT models
│   ├── vision/               # Vision models
│   │   ├── vit/              # ViT models
│   ├── audio/                # Audio models
│   │   ├── whisper/          # Whisper models
│   └── multimodal/           # Multimodal models
├── hardware/                 # Hardware-specific tests
│   ├── __init__.py
│   ├── webgpu/               # WebGPU tests
│   │   ├── compute_shaders/  # Compute shader tests
│   │   └── tensor_sharing/   # Tensor sharing tests
│   ├── webnn/                # WebNN tests
│   ├── cuda/                 # CUDA tests
│   └── rocm/                 # ROCm tests
├── api/                      # API tests
│   ├── __init__.py
│   ├── openai/               # OpenAI API tests
│   ├── hf_tei/               # HuggingFace TensorRT-LLM Endpoint Interface tests
│   └── hf_tgi/               # HuggingFace Text Generation Interface tests
├── integration/              # Integration tests
│   ├── __init__.py
│   ├── browser/              # Browser integration tests
│   └── database/             # Database integration tests
├── template_system/          # Template system for generating tests
│   ├── __init__.py
│   ├── templates/            # Test templates
│   │   ├── __init__.py
│   │   ├── base_template.py  # Base class for templates
│   │   ├── model_test_template.py  # Template for model tests
│   │   └── hardware_test_template.py  # Template for hardware tests
└── docs/                     # Documentation
    ├── TEST_REFACTORING_PLAN.md  # This file
    └── MIGRATION_GUIDE.md    # Guide for migrating tests
```

## Implementation Details

### 1. Unified Test Runner (`run.py`)

The unified test runner provides a single entry point for running all tests with comprehensive command-line options:

```python
python run.py [options]

# Examples:
python run.py --test-type model  # Run all model tests
python run.py --test-type hardware --platform webgpu  # Run WebGPU tests
python run.py --platform cuda  # Run all tests on CUDA
```

Key features of the test runner:
- Flexible test selection based on test type, hardware platform, and model
- Support for distributed testing
- CI/CD integration
- Detailed output formatting options
- Automatic test report generation

### 2. Common Utilities

#### Hardware Detection (`common/hardware_detection.py`)

Standard utilities for detecting hardware capabilities:

```python
from common.hardware_detection import detect_hardware, skip_if_no_cuda

# Get hardware capabilities
hardware_info = detect_hardware()
if hardware_info['platforms']['webgpu']['available']:
    # WebGPU is available
    ...

# Skip tests if CUDA is not available
@skip_if_no_cuda
def test_cuda_function():
    # This test will be skipped if CUDA is not available
    ...
```

#### Model Helpers (`common/model_helpers.py`)

Utilities for working with ML models:

```python
from common.model_helpers import load_model, get_sample_inputs_for_model

# Load a model
model = load_model("bert-base-uncased", framework="transformers")

# Generate sample inputs
inputs = get_sample_inputs_for_model("bert", batch_size=2)
```

#### Common Fixtures (`common/fixtures.py`)

Reusable pytest fixtures:

```python
from common.fixtures import cuda_device, bert_model, webgpu_browser

def test_bert_inference(bert_model, cuda_device):
    # Use bert_model and cuda_device fixtures
    ...
```

### 3. Global Pytest Configuration (`conftest.py`)

Global pytest configuration and fixtures:

```python
# Importing global fixtures
from common.fixtures import hardware_info, cuda_available, cuda_device, ...

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "webgpu: mark test as requiring WebGPU")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    ...

# Skip tests based on hardware availability
def pytest_runtest_setup(item):
    markers = [mark.name for mark in item.iter_markers()]
    if 'webgpu' in markers and not hardware_info['platforms']['webgpu']['available']:
        pytest.skip("Test requires WebGPU")
    ...
```

### 4. Test Templates

Templates for generating consistent test files:

#### Model Test Template

```python
from template_system.templates.model_test_template import ModelTestTemplate

# Create a model test
template = ModelTestTemplate(
    template_name="bert_test",
    output_dir="test",
    parameters={
        'model_name': 'bert-base-uncased',
        'model_type': 'text',
        'test_name': 'bert_base_uncased'
    }
)

# Write the test file
template.write()
```

#### Hardware Test Template

```python
from template_system.templates.hardware_test_template import HardwareTestTemplate

# Create a hardware test
template = HardwareTestTemplate(
    template_name="webgpu_test",
    output_dir="test",
    parameters={
        'hardware_platform': 'webgpu',
        'test_name': 'webgpu_matmul',
        'test_category': 'compute_shaders',
        'test_operation': 'matmul'
    }
)

# Write the test file
template.write()
```

### 5. Test Organization Patterns

#### Model Tests

Model tests are organized by modality and model family:

```
models/
├── text/                  # Text models
│   ├── bert/              # BERT models
│   │   ├── test_bert_base_uncased.py  # Test for bert-base-uncased
│   │   └── test_bert_large_uncased.py  # Test for bert-large-uncased
│   ├── t5/                # T5 models
│   └── gpt/               # GPT models
├── vision/                # Vision models
├── audio/                 # Audio models
└── multimodal/            # Multimodal models
```

#### Hardware Tests

Hardware tests are organized by platform and capability:

```
hardware/
├── webgpu/               # WebGPU tests
│   ├── compute_shaders/  # Compute shader tests
│   │   ├── test_webgpu_matmul.py  # Matrix multiplication test
│   │   └── test_webgpu_conv.py  # Convolution test
│   └── tensor_sharing/   # Tensor sharing tests
├── webnn/                # WebNN tests
├── cuda/                 # CUDA tests
└── rocm/                 # ROCm tests
```

### 6. CI/CD Integration

CI/CD pipeline configuration uses the unified test runner:

```yaml
# GitHub Actions workflow
name: Run Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-type: [model, hardware, api, integration]
        platform: [cpu, cuda]
        exclude:
          - test-type: hardware
            platform: cpu
          - test-type: api
            platform: cuda
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        cd test
        python run.py --test-type ${{ matrix.test-type }} --platform ${{ matrix.platform }} --junit-xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results-${{ matrix.test-type }}-${{ matrix.platform }}
        path: test/test-results.xml
```

## Migration Strategy

The migration will be performed gradually to ensure minimal disruption:

1. **Create the new directory structure**: Set up the new directory hierarchy
2. **Implement common utilities**: Develop the shared utilities
3. **Create the test runner**: Implement the unified test entry point
4. **Migrate tests incrementally**:
   - Use the `migrate_tests.py` script to analyze and move tests
   - Start with a subset of tests to validate the approach
   - Update imports and references
5. **Update CI/CD workflows**: Transition to the new test runner
6. **Validate the migration**: Verify all tests work as expected

The `migrate_tests.py` script will assist with:
- Classifying test files based on content
- Determining the appropriate target location
- Moving or copying files to the new structure
- Creating `__init__.py` files
- Updating import statements

## Backward Compatibility

To maintain backward compatibility during migration:

1. **Test Adapter Layer**: Create adapters for existing entry points
2. **Dual CI/CD Pipelines**: Run both old and new test structures in CI
3. **Import Compatibility**: Ensure imports work in both structures
4. **Documentation**: Provide clear migration guides for developers

## Timeline

| Phase | Description | Timeline |
|-------|-------------|----------|
| 1 | Create new directory structure and common utilities | Week 1 |
| 2 | Implement test runner and templates | Week 1-2 |
| 3 | Migration script development | Week 2 |
| 4 | Initial test migration (subset) | Week 2-3 |
| 5 | CI/CD integration | Week 3 |
| 6 | Complete migration of all tests | Week 3-4 |
| 7 | Validation and documentation | Week 4 |

## Potential Challenges

1. **Complex Test Dependencies**: Some tests may have complex dependencies that are difficult to migrate
2. **Browser Automation**: WebGPU/WebNN tests that use browser automation may require special handling
3. **CI/CD Integration**: Ensuring CI/CD workflows continue to work during transition
4. **Distributed Testing**: Maintaining compatibility with distributed testing infrastructure

## Conclusion

This refactoring plan provides a comprehensive approach to reorganizing the IPFS Accelerate testing framework. By implementing a unified directory structure, standardized utilities, and a single entry point, we will create a more maintainable, discoverable, and efficient testing system that integrates seamlessly with CI/CD pipelines and supports the project's distributed testing capabilities.