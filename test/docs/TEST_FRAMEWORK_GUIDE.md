# IPFS Accelerate Test Framework Guide

## Introduction

The IPFS Accelerate Test Framework provides a comprehensive, structured approach to testing across different types of models, hardware platforms, and APIs. This guide explains the structure, components, and usage of the test framework.

## Framework Structure

The test framework is organized as follows:

```
test/
├── models/                  # Model-specific tests
│   ├── text/                # Text model tests
│   ├── vision/              # Vision model tests
│   ├── audio/               # Audio model tests
│   └── multimodal/          # Multimodal model tests
├── hardware/                # Hardware-specific tests
│   ├── webgpu/              # WebGPU tests
│   ├── webnn/               # WebNN tests
│   ├── cuda/                # CUDA tests
│   └── ...                  # Other hardware platforms
├── api/                     # API-specific tests
│   ├── openai/              # OpenAI API tests
│   ├── hf_tei/              # HuggingFace TEI tests
│   └── ...                  # Other API tests
├── integration/             # Integration tests
├── common/                  # Shared utilities and helpers
│   ├── fixtures.py          # Test fixtures
│   ├── hardware_detection.py # Hardware detection utilities
│   └── model_helpers.py     # Model loading utilities
├── template_system/         # Template-based test generation
│   ├── templates/           # Test templates
│   └── generate_test.py     # Test generator script
├── docs/                    # Documentation
├── conftest.py              # Pytest configuration
├── pytest.ini               # Pytest settings
├── run.py                   # Test runner
└── run.sh                   # CI/CD entry point
```

## Key Components

### Test Templates

The framework uses a template-based approach to generate consistent test files. The following templates are available:

- `BaseTemplate`: Base class for all templates with common functionality
- `ModelTestTemplate`: Template for model tests with different modalities
- `HardwareTestTemplate`: Template for hardware platform tests
- `APITestTemplate`: Template for API tests

### Utilities

Common utilities provide shared functionality:

- `hardware_detection.py`: Detect available hardware and provide skip decorators
- `model_helpers.py`: Load models and provide sample inputs
- `fixtures.py`: Common test fixtures for hardware, models, and browsers

### Test Runner

The `run.py` script provides a flexible command-line interface for running tests:

```bash
# Run all tests
python run.py

# Run model tests
python run.py --test-type model

# Run specific model tests
python run.py --test-type model --model bert

# Run hardware tests
python run.py --test-type hardware --platform webgpu

# Run API tests
python run.py --test-type api --api openai
```

For CI/CD environments, the `run.sh` script provides additional functionality:

```bash
# Set up environment and run all tests
./run.sh --setup-env --install-deps

# Run tests with reports
./run.sh --generate-reports

# Run distributed tests
./run.sh --distributed --worker-count 4
```

## Working with the Framework

### Creating a New Test

The easiest way to create a new test is using the template system:

```bash
# Create a model test
python template_system/generate_test.py model --model-name bert-base-uncased --model-type text

# Create a hardware test
python template_system/generate_test.py hardware --hardware-platform webgpu --test-name matmul_performance

# Create an API test
python template_system/generate_test.py api --api-name openai --test-name chat_completion
```

### Running Tests

To run tests, use the `run.py` script:

```bash
# Run all tests
python run.py

# Run with more verbosity
python run.py -v

# Run tests with specific markers
python run.py --markers "webgpu or cuda"

# Generate an HTML report
python run.py --report
```

For CI/CD environments, use the `run.sh` script:

```bash
# Set up environment and run all tests
./run.sh --setup-env --install-deps

# Run tests with reports
./run.sh --generate-reports
```

### Migrating Existing Tests

To migrate existing tests to the new structure:

1. Analyze existing tests:
   ```bash
   python migrate_tests.py --source-dir old_tests --analyze-only
   ```

2. Migrate the tests:
   ```bash
   python migrate_tests.py --source-dir old_tests --output-dir test
   ```

3. Track migration progress:
   ```bash
   python track_migration_progress.py --analysis-report migration_analysis.json --migrated-dir test
   ```

## Writing Tests

### Test Types

The framework supports different types of tests:

1. **Model Tests**: Test model loading, inference, and performance

   ```python
   @pytest.mark.model
   @pytest.mark.text
   class TestBertBaseUncased:
       def test_load_model(self, cpu_device):
           model, tokenizer = load_text_model("bert-base-uncased", device=cpu_device)
           assert model is not None
           assert tokenizer is not None
   ```

2. **Hardware Tests**: Test hardware platforms and operations

   ```python
   @pytest.mark.hardware
   @pytest.mark.webgpu
   class TestWebGPUMatmul:
       def test_matmul_performance(self, webgpu_browser):
           # Test WebGPU matrix multiplication
           pass
   ```

3. **API Tests**: Test API clients and services

   ```python
   @pytest.mark.api
   @pytest.mark.openai
   class TestOpenAIChat:
       def test_chat_completion(self, api_key):
           # Test OpenAI chat completion
           pass
   ```

### Fixtures

The framework provides common fixtures for tests:

```python
# Hardware fixtures
def test_with_cuda(cuda_device):
    # Use CUDA device
    pass

# Model fixtures
def test_with_bert(bert_model):
    model, tokenizer = bert_model
    # Use BERT model
    pass

# Browser fixtures
def test_with_webgpu(webgpu_browser):
    # Use WebGPU browser
    pass
```

### Markers

Use markers to categorize tests:

```python
@pytest.mark.model           # Mark as model test
@pytest.mark.text            # Mark as text model test
@pytest.mark.hardware        # Mark as hardware test
@pytest.mark.webgpu          # Mark as WebGPU test
@pytest.mark.api             # Mark as API test
@pytest.mark.slow            # Mark as slow test
@pytest.mark.distributed     # Mark as distributed test
```

## Advanced Features

### Distributed Testing

The framework supports distributed testing:

```bash
# Run tests in distributed mode
python run.py --distributed --worker-count 4
```

### Test Reports

Generate test reports:

```bash
# Generate HTML report
python run.py --report

# Generate JUnit XML report
python run.py --junit-xml
```

### Conditional Testing

Skip tests based on hardware availability:

```python
@skip_if_no_cuda
def test_cuda_specific():
    # Only runs if CUDA is available
    pass
```

## Best Practices

1. **Use the template system** for creating new tests to ensure consistency
2. **Use fixtures** for common setup and teardown operations
3. **Use markers** to categorize tests and control execution
4. **Follow the directory structure** to keep tests organized
5. **Write modular tests** that focus on specific functionality
6. **Include hardware checks** to skip tests on unsupported platforms
7. **Generate sample inputs** using the model helpers

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Run `verify_test_environment.py` to check dependencies
2. **Import errors**: Ensure the project root is in the Python path
3. **Hardware not available**: Use skip decorators to handle unavailable hardware
4. **Browser issues**: Check Selenium setup and browser configurations
5. **Model loading errors**: Verify model availability and credentials

### Verification

Use the verification script to check the environment:

```bash
python verify_test_environment.py
```

## Contributing

When contributing to the test framework:

1. Follow the existing structure and naming conventions
2. Use templates to generate new tests
3. Add appropriate markers and fixtures
4. Update documentation for new features
5. Ensure tests are modular and focused
6. Include hardware checks for platform-specific tests

## Further Resources

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md): Guide for migrating existing tests
- [TEMPLATE_SYSTEM_GUIDE.md](TEMPLATE_SYSTEM_GUIDE.md): Guide for the template system
- [DISTRIBUTED_TESTING_GUIDE.md](../DISTRIBUTED_TESTING_GUIDE.md): Guide for distributed testing