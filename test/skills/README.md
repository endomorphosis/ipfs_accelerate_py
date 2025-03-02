# Hugging Face Model Test Suite

## Complete Test Coverage for All 300 Model Types

This test suite aims to provide comprehensive testing for all 300 Hugging Face model types as listed in `huggingface_model_types.json`.

## Current Status (March 1, 2025)

- **Total Model Types Required**: 300
- **Implemented Test Files**: 175
- **Coverage**: 58.3%
- **Remaining Model Types**: 125

## Directory Contents

- **Test Files**: Individual test implementations for each model type (`test_hf_*.py`)
- **Automation Tools**: Scripts for generating and running tests
- **Results Directory**: Collected test results by model and hardware
- **Expected Results**: Reference results for verification
- **Documentation**: Implementation guides and test coverage reports

## Getting Started

### Prerequisites

The test suite requires the following dependencies:
- Python 3.8+
- PyTorch
- Transformers
- Model-specific dependencies (specified in each test file)

### Running Tests

To run tests for a specific model:

```bash
# List available models
python test_hf_bert.py --list-models

# Run test with default model
python test_hf_bert.py

# Run test with specific model
python test_hf_bert.py --model bert-base-uncased

# Test on all available hardware
python test_hf_bert.py --all-hardware

# Save results to file
python test_hf_bert.py --save
```

### Generating Missing Tests

We provide tools to automatically generate test files for missing model types:

```bash
# Generate test for a specific model type
python generate_missing_hf_tests.py --generate MODEL_TYPE

# Generate multiple tests in batch
python generate_missing_hf_tests.py --batch 10

# Generate all missing tests
python generate_all_missing_tests.py --all
```

## Implementation Strategy

The test suite implements consistent testing across all model types:

1. **Pipeline API Testing**: Tests using the transformers pipeline API
2. **Direct Model Testing**: Tests using the model's from_pretrained() method
3. **Hardware Testing**: Tests across CPU, CUDA, and OpenVINO backends
4. **Performance Metrics**: Measures inference time, load time, and memory usage

## Test Structure

Each test file follows a standardized structure:

```python
# Model registry with specific models for each family
MODEL_REGISTRY = {...}

class TestModelFamily:
    # Test initialization
    def __init__(self, model_id=None):
        ...
        
    # Pipeline API testing
    def test_pipeline(self, device="auto"):
        ...
        
    # Direct model testing
    def test_from_pretrained(self, device="auto"):
        ...
        
    # OpenVINO testing
    def test_with_openvino(self):
        ...
        
    # Run all tests
    def run_tests(self, all_hardware=False):
        ...
```

## Documentation

- **[TEST_AUTOMATION.md](TEST_AUTOMATION.md)**: Guide to test automation and batch generation
- **[test_report.md](test_report.md)**: Current test results and status
- **[HF_COVERAGE_REPORT.md](HF_COVERAGE_REPORT.md)**: Analysis of test coverage
- **[CLAUDE.md](/test/CLAUDE.md)**: Development guide with implementation plans

## Next Steps

1. Complete generation of all missing test files (125 model types)
2. Verify implementation across hardware backends
3. Generate comprehensive performance benchmarks
4. Expand test coverage to model variants
5. Implement parallel test execution

## Contributors

- IPFS Accelerate Python Team