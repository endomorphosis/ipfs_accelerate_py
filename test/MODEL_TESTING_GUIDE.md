# IPFS Accelerate Model Testing Guide

This guide documents the comprehensive testing framework for HuggingFace models in the IPFS Accelerate Python library. It provides instructions for running tests, generating new tests, and understanding test coverage reporting.

## Current Test Coverage Status (March 2025)

- **Total Hugging Face model types**: 300
- **Tests implemented**: 127+ (42.3% coverage)
- **Recent additions**: kosmos-2, grounding-dino, tapas, and various vision models
- **High-priority models remaining**: swinv2, vit_mae, layoutlmv2, nougat

### Coverage by Category

| Category | Implemented | Total | Coverage |
|----------|-------------|-------|----------|
| Language Models | 65+ | 92 | 70.7% |
| Vision Models | 32+ | 51 | 62.7% |
| Audio Models | 15+ | 20 | 75.0% |
| Multimodal Models | 15+ | 19 | 78.9% |

## Running Tests

### Individual Tests

To run an individual test:

```bash
# Basic test run
python skills/test_hf_bert.py

# With platform selection
python skills/test_hf_bert.py --platform cpu
python skills/test_hf_bert.py --platform cuda
python skills/test_hf_bert.py --platform openvino

# With model override
python skills/test_hf_bert.py --model bert-base-uncased
```

### Batch Testing

To run multiple tests at once:

```bash
# Run all tests
python run_skills_tests.py --all

# Run tests for specific models
python run_skills_tests.py --models bert,roberta,gpt2

# Run tests by category
python run_skills_tests.py --category language
python run_skills_tests.py --category vision
python run_skills_tests.py --category audio
python run_skills_tests.py --category multimodal
```

## Generating New Tests

The test framework includes advanced scripts for generating high-quality tests for new models.

### Using the Recommended Generator

The `generate_model_tests.py` script is the recommended way to generate new tests:

```bash
# List all models missing test implementations
python generate_model_tests.py --list-only

# Generate tests for specific models
python generate_model_tests.py --models kosmos-2 tapas layoutlmv2

# Generate tests by category
python generate_model_tests.py --category vision --limit 5
python generate_model_tests.py --category audio --limit 10

# Generate tests with a custom output directory
python generate_model_tests.py --output-dir custom_tests --limit 3
```

### Test Generator Features

The test generator automatically:

1. **Categorizes models** based on their pipeline tasks
2. **Selects appropriate examples** for each model type
3. **Creates specialized test inputs** for different model capabilities
4. **Implements batch testing** for comprehensive validation
5. **Handles hardware-specific implementations** (CPU, CUDA, OpenVINO)
6. **Incorporates robust error handling** with graceful degradation
7. **Collects performance metrics** when possible
8. **Creates expected results** for validation

## Test Structure

Each test follows a consistent structure:

1. **Imports and Setup**: Imports libraries and sets up the test environment
2. **Model Implementation**: Either imports the real model implementation or creates a mock
3. **Test Class**: Contains test methods and utilities
4. **Platform-Specific Testing**: Tests for CPU, CUDA, and OpenVINO
5. **Result Collection**: Saves outputs in standardized JSON format
6. **Performance Metrics**: Records inference time and resource usage

## Result Validation

Test results are stored in two locations:

- **Collected Results**: Current test outputs in `skills/collected_results/`
- **Expected Results**: Baseline outputs in `skills/expected_results/`

The test framework compares current results with expected results to identify regressions.

## Implementation Types

Tests will report one of the following implementation types:

- **REAL**: Using actual model weights with real inference
- **MOCK**: Using simulated outputs (when model or dependencies are unavailable)
- **PARTIAL**: Using partially real implementation (some components are mocked)

## Troubleshooting

### Common Issues

1. **Missing dependencies**:
   - Tests automatically mock missing dependencies
   - Install required packages for real testing: `pip install torch transformers`

2. **CUDA unavailable**:
   - Tests will report "CUDA not available" and skip CUDA tests
   - Ensure CUDA is properly configured if you want to test CUDA support

3. **Memory errors**:
   - Use smaller model variants with the `--model` parameter
   - Free memory between tests when running multiple models

4. **Download failures**:
   - Tests include fallback mechanisms for models that fail to download
   - Check your internet connection and HuggingFace account if needed

### Fixing Failed Tests

If a test fails:

1. Check the error messages in the test output
2. Verify that required dependencies are installed
3. Try with a smaller model variant using `--model`
4. Update the expected results if the implementation has changed

## Creating Custom Tests

While the test generator handles most cases, you can create custom tests:

1. Copy a similar test file as a template
2. Update the model name and dependencies
3. Modify the test inputs for your specific model
4. Adjust the processor and model initialization
5. Run the test to generate expected results

## High-Priority Test Targets

The highest priority models for new test implementation are:

1. **NOUGAT**: Document understanding model for academic papers
2. **SwinV2**: Advanced vision transformer for image understanding
3. **ViTMAE**: Vision transformer with masked autoencoder pretraining
4. **LayoutLMv2**: Document understanding model with spatial layout
5. **Depth-Anything**: Depth estimation model
6. **OLMo**: Open language model

## Contributing New Tests

When contributing new tests:

1. Use the test generator for consistency
2. Follow the established test structure
3. Include both regular and batch testing
4. Test on all available platforms
5. Document any special handling in comments
6. Update the test coverage documentation

## Further Resources

- [MODEL_TESTING_README.md](MODEL_TESTING_README.md): Detailed information about model tests
- [MODEL_IMPLEMENTATION_PROGRESS.md](MODEL_IMPLEMENTATION_PROGRESS.md): Current implementation status
- [huggingface_test_implementation_plan.md](huggingface_test_implementation_plan.md): Overall implementation plan