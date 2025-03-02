# IPFS Accelerate Model Testing Guide

This guide documents the comprehensive testing framework for HuggingFace models in the IPFS Accelerate Python library. It provides instructions for running tests, generating new tests, and understanding test coverage reporting.

## Current Test Coverage Status (March 2025)

- **Total Hugging Face model types**: 300
- **Tests implemented**: 70+ (23.3% coverage)
- **Real implementations**: 13 model types have real implementations (18.5% of tested models)
- **Recent improvements**: New smart model selection and error handling
- **High-priority models for real implementation**: Sam, Phi3, QWen3, Mamba

### Coverage by Category

| Category | Total Models | Implemented Tests | Real Implementations |
|----------|--------------|-------------------|---------------------|
| Language Models | 150+ | 30+ (20%) | 5 (BERT, T5, LLaMA, GPT2, QWen2) |
| Vision Models | 70+ | 15+ (21%) | 4 (CLIP, ViT, DETR, XCLIP) |
| Audio Models | 30+ | 10+ (33%) | 3 (Whisper, Wav2Vec2, CLAP) |
| Multimodal Models | 30+ | 10+ (33%) | 1 (LLaVA) |
| Specialized Models | 20+ | 5+ (25%) | 0 |

## Running Tests

### Individual Tests

To run an individual test:

```bash
# Basic test run
cd skills && python test_hf_bert.py

# With verbose output
cd skills && python test_hf_bert.py --verbose

# With model override (choose a smaller model for faster testing)
cd skills && python test_hf_bert.py --model bert-base-uncased
```

### Using the Unified Test Runner

Our new unified test runner supports parallel execution and detailed reporting:

```bash
# Run all model tests
python run_unified_tests.py --type model

# Run tests for specific models
python run_unified_tests.py --type model --models bert t5 llama

# Run tests with increased parallelism (faster)
python run_unified_tests.py --workers 8 --type all

# Run tests by implementation type
python run_unified_tests.py --type model --impl-type real
python run_unified_tests.py --type model --impl-type mock

# Generate a detailed test report
python run_unified_tests.py --type model --report model_test_report.md
```

## Generating New Tests

The test framework includes advanced scripts for generating high-quality tests for new models.

### Using the Test Generators

The framework provides two recommended test generators:

#### 1. Basic Test Generator (Simple & Fast)

The `generate_basic_tests.py` script is the recommended way to quickly generate new tests:

```bash
# Generate tests for specific models with automatic task selection
python generate_basic_tests.py bert t5 llama clip

# Generate tests with explicit task
python generate_basic_tests.py detr --task object-detection
python generate_basic_tests.py whisper --task automatic-speech-recognition

# Check models by type for proper test generation
python generate_basic_tests.py --list-models
```

#### 2. Comprehensive Test Generator (Advanced)

For more comprehensive tests with advanced options, use `generate_unified_test.py`:

```bash
# List all models missing test implementations
python generate_unified_test.py --type model --list-missing

# Generate tests for specific models
python generate_unified_test.py --type model --models bert t5 llama

# Generate tests by category
python generate_unified_test.py --type model --category vision --limit 5
python generate_unified_test.py --type model --category audio --limit 10

# Generate tests with a custom output directory
python generate_unified_test.py --type model --output-dir custom_tests --limit 3
```

### Test Generator Features

Our test generators automatically:

1. **Categorizes models** by task type (language, vision, audio, multimodal)
2. **Selects appropriate test models** based on model type (e.g., bert-base-uncased for BERT)
3. **Creates specialized test inputs** for each model's needs (text, images, audio)
4. **Implements intelligent output handling** for different return types
5. **Handles hardware-specific tests** (CPU, CUDA, OpenVINO)
6. **Provides robust error handling** with graceful degradation to mock implementations
7. **Collects performance metrics** including inference time and memory usage
8. **Creates standardized test outputs** in JSON format for easy comparison

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

## High-Priority Implementation Targets

The highest priority models for real implementation (beyond test coverage) are:

1. **Sam**: Segment Anything Model for advanced image segmentation
2. **Phi3**: Microsoft's newest language model with strong performance
3. **QWen3**: Alibaba's powerful multilingual language model
4. **Mamba**: State-space model with efficient sequence modeling
5. **Depth-Anything**: Universal depth estimation model
6. **Mistral-Next**: Latest generation of the Mistral language model family

These models already have test coverage but need real implementations to replace the mock versions.

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