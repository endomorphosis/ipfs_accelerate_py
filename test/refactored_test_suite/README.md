# Refactored Test Suite for IPFS Accelerate Python Framework

This directory contains the refactored test suite for the IPFS Accelerate Python Framework. The goal of this refactoring is to create a more maintainable, structured test suite that follows best practices for Python testing.

## Structure

The test suite is organized into the following directories:

- `api/`: Tests for API functionality
- `browser/`: Tests for browser integration
- `hardware/`: Tests for hardware-specific features
  - `webgpu/`: Tests for WebGPU integration
  - `webnn/`: Tests for WebNN integration
  - `platform/`: Tests for other hardware platforms (CUDA, MPS, ROCm, etc.)
- `models/`: Tests for model functionality
  - `text/`: Tests for text models
  - `vision/`: Tests for vision models
  - `audio/`: Tests for audio models
  - `other/`: Tests for other model types or general model functionality
- `resource_pool/`: Tests for resource pool functionality
- `integration/`: Integration tests
- `tests/`: Miscellaneous tests and test utilities

## Base Classes

The test suite uses a hierarchy of base classes to avoid code duplication and ensure consistent testing patterns:

- `BaseTest`: The root class for all tests
- `ModelTest`: For testing models
- `HardwareTest`: For testing hardware-specific features
- `BrowserTest`: For testing browser integration
- `APITest`: For testing API functionality

## Usage

To run the refactored test suite, use the `run_refactored_test_suite.py` script:

```bash
# Run all tests
python run_refactored_test_suite.py --init

# Run tests in specific directories
python run_refactored_test_suite.py --init --subdirs api models/text
```

The `--init` flag creates the necessary `__init__.py` files for proper module imports.

## Migration Progress

For more information about the migration progress, see [MIGRATION_PROGRESS.md](MIGRATION_PROGRESS.md).

## Adding New Tests

When adding new tests, follow these guidelines:

1. Place the test in the appropriate directory based on its category
2. Use the appropriate base class for the test type
3. Use absolute imports rather than relative imports
4. Follow the naming convention `test_*.py` for test files
5. Include proper documentation for the test

Example:

```python
from refactored_test_suite.model_test import ModelTest

class TestMyModel(ModelTest):
    """Test class for my model."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model_name = "my-model"
        
    def test_inference(self):
        """Test inference with the model."""
        model = self.load_model()
        # Test code here
```