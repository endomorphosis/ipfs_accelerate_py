# IPFS Accelerate Test Framework

Welcome to the IPFS Accelerate Test Framework documentation. This document provides an overview of the test framework architecture, usage, and development guidelines.

## Overview

The IPFS Accelerate Test Framework is designed to provide a unified, organized, and extensible way to test the IPFS Accelerate Python Framework across multiple model types, hardware platforms, and integration scenarios.

The framework is organized into logical test categories and provides common utilities for hardware detection, model handling, and test environment setup.

## Directory Structure

The test framework is organized into the following categories:

```
test/
├── models/                   # Tests for specific model types
│   ├── text/                 # Text models (BERT, T5, GPT, etc.)
│   ├── vision/               # Vision models (ViT, DETR, etc.)
│   ├── audio/                # Audio models (Whisper, etc.)
│   └── multimodal/           # Multimodal models (CLIP, etc.)
├── hardware/                 # Tests for specific hardware platforms
│   ├── webgpu/               # WebGPU tests
│   ├── webnn/                # WebNN tests
│   ├── cuda/                 # CUDA tests
│   └── ...
├── api/                      # Tests for API integrations
│   ├── llm_providers/        # LLM API providers (OpenAI, Claude, etc.)
│   ├── huggingface/          # HuggingFace API tests
│   ├── local_servers/        # Local server tests (Ollama, vLLM, etc.)
│   └── ...
├── integration/              # Cross-component integration tests
│   ├── browser/              # Browser integration tests
│   ├── database/             # Database integration tests
│   ├── distributed/          # Distributed testing
│   └── ...
├── common/                   # Shared utilities and fixtures
│   ├── hardware_detection.py # Hardware platform detection
│   ├── model_helpers.py      # Model loading and input utilities
│   └── fixtures.py           # Common test fixtures
├── run.py                    # Unified test runner
├── docs/                     # Documentation
└── ...
```

## Getting Started

### Prerequisites

* Python 3.8+
* pytest
* Additional requirements based on testing needs

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up the test environment:

```bash
./setup_test_env.sh
```

3. Verify the test environment:

```bash
python verify_test_environment.py
```

## Running Tests

The framework provides a unified entry point (`run.py`) for running tests:

```bash
# Run all tests
python run.py

# Run specific test types
python run.py --test-type model
python run.py --test-type hardware
python run.py --test-type api
python run.py --test-type integration

# Run tests for specific models/platforms
python run.py --test-type model --model bert
python run.py --test-type hardware --platform webgpu

# Run tests with specific markers
python run.py --markers "slow or webgpu"

# Generate HTML report
python run.py --report
```

## Hardware Detection

The framework includes utilities for detecting available hardware platforms and skipping tests when required hardware is not available. This allows tests to be written once and executed conditionally based on the test environment.

Example usage:

```python
import pytest
from common.hardware_detection import skip_if_no_webgpu

@skip_if_no_webgpu
def test_webgpu_matmul():
    # This test will be skipped if WebGPU is not available
    ...
```

## Common Fixtures

The framework provides common fixtures for hardware setup, model loading, and test environment configuration:

```python
import pytest
from common.fixtures import cuda_device, bert_model

def test_bert_inference(bert_model, cuda_device):
    # Use bert_model and cuda_device fixtures
    ...
```

## Test Templates

The framework includes template-based test generation for common test patterns:

```python
from template_system.templates.model_test_template import ModelTestTemplate

class BertTest(ModelTestTemplate):
    model_name = "bert-base-uncased"
    model_type = "text"
    
    def test_inference(self):
        # Test inference with the model
        ...
```

## Distributed Testing

The framework supports distributed testing across multiple workers:

```bash
# Run tests in distributed mode
python run.py --distributed --worker-count 4
```

## Visualization and Reporting

The framework includes tools for visualizing test results and generating reports:

```bash
# Generate HTML report
python run.py --report

# Generate JUnit XML report for CI
python run.py --junit-xml
```

## Migration

If you're migrating from the old test structure, see the [Migration Guide](MIGRATION_GUIDE.md) for detailed instructions.

## Contributing

See the [Contributing Guide](CONTRIBUTING.md) for guidelines on contributing to the test framework.

## Additional Documentation

* [Template System Guide](TEMPLATE_SYSTEM_GUIDE.md): Detailed information on the template-based test generation system
* [Migration Guide](MIGRATION_GUIDE.md): Instructions for migrating from the old test structure
* [CI/CD Integration](CICD_INTEGRATION.md): Information on integrating with CI/CD pipelines