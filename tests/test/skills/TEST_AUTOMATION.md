# Hugging Face Model Test Automation Guide

## Complete Test Coverage for All 300 Model Types

This document outlines the process for achieving 100% test coverage for all 300 Hugging Face model types as specified in `huggingface_model_types.json`.

## Current Status (March 19, 2025)

- **Total model types required**: 300
- **Current implementation**: 179 model types
- **Current coverage**: 59.7%
- **Remaining to implement**: 121 model types
- **Last update**: March 19, 2025 - Fixed indentation issues and refactored test generator

## Automation Tools

We have implemented a suite of automation tools to streamline the test generation process:

### 1. Test Discovery and Analysis

The `generate_missing_hf_tests.py` script provides comprehensive analysis of current test coverage:

```bash
# List all missing test files
python generate_missing_hf_tests.py --list-missing

# Generate coverage report
python generate_missing_hf_tests.py --report
```

### 2. Individual Test Generation

```bash
# Generate test for a specific model type
python generate_missing_hf_tests.py --generate MODEL_TYPE

# Example
python generate_missing_hf_tests.py --generate vits
```

### 3. Batch Test Generation

```bash
# Generate a batch of missing tests
python generate_missing_hf_tests.py --batch NUMBER

# Example - generate 10 test files
python generate_missing_hf_tests.py --batch 10
```

### 4. Complete Test Generation

The `generate_all_missing_tests.py` script automates the generation of all remaining test files:

```bash
# Generate all missing tests in batches
python generate_all_missing_tests.py --all

# Generate a specific batch size
python generate_all_missing_tests.py --batch-size 20

# Verify coverage after generation
python generate_all_missing_tests.py --verify
```

## Test Implementation Structure

Each test file follows a standardized structure:

1. **Registry Definition**: Model-specific registry with default models and architecture type
2. **Test Class**: Standardized test class for the model family
3. **Pipeline Testing**: Test using transformers pipeline() API
4. **Direct Model Testing**: Test using from_pretrained() API
5. **Hardware Testing**: CPU, CUDA, and OpenVINO hardware acceleration testing
6. **Result Collection**: Comprehensive results with performance metrics

### Architecture-Aware Testing (Updated March 2025)

Our test generator is architecture-aware and handles different model families appropriately through a modular template system. This allows for proper indentation, consistent formatting, and architecture-specific handling:

1. **Encoder-Only Models** (BERT, ViT, etc.)
   - Examples: `bert-base-uncased`, `google/vit-base-patch16-224`
   - Key features: Bidirectional attention, no autoregressive behavior
   - Task types: fill-mask, feature-extraction, sequence-classification
   - Template: `encoder_only_template.py` with proper indentation standards

2. **Decoder-Only Models** (GPT-2, LLaMA, etc.)
   - Examples: `gpt2`, `meta-llama/Llama-2-7b`
   - Key features: Autoregressive behavior, padding token configuration 
   - Special handling: `tokenizer.pad_token = tokenizer.eos_token`
   - Task types: text-generation, causal-lm, text-completion
   - Template: `decoder_only_template.py` with autoregressive generation settings

3. **Encoder-Decoder Models** (T5, BART, etc.)
   - Examples: `t5-small`, `facebook/bart-base`
   - Key features: Separate encoder and decoder components
   - Special handling: Empty decoder inputs (`decoder_input_ids`) required
   - Task types: translation, summarization, question-answering
   - Template: `encoder_decoder_template.py` with both components set up

4. **Vision Models** (ViT, Swin, etc.)
   - Examples: `google/vit-base-patch16-224` 
   - Key features: Image processing, specific tensor shapes
   - Special handling: Proper image tensor shape (batch_size, channels, height, width)
   - Task types: image-classification, object-detection, semantic-segmentation
   - Template: `vision_model_template.py` with image processing components

Each template follows standard Python indentation (4 spaces per level) and includes:
- Hardware detection with CPU/GPU/MPS/OpenVINO support
- Mock implementations for graceful degradation
- Error classification and reporting
- Standardized performance metrics collection

## Running Tests

### Individual Model Test

```bash
# List available models
python generators/models/test_hf_MODEL_TYPE.py --list-models

# Test with default model
python generators/models/test_hf_MODEL_TYPE.py

# Test specific model
python generators/models/test_hf_MODEL_TYPE.py --model MODEL_ID

# Test with all hardware backends
python generators/models/test_hf_MODEL_TYPE.py --all-hardware

# Save results to file
python generators/models/test_hf_MODEL_TYPE.py --save
```

### Batch Testing

```bash
# Test all model types
python generators/models/test_all_models.py

# Test specific model families
python generators/models/test_all_models.py --models bert,gpt2,t5
```

## Comprehensive Testing Plan

To fully verify all 300 model types across hardware platforms:

1. **Phase 1: Generate All Test Files**
   ```bash
   python generate_all_missing_tests.py --all
   ```

2. **Phase 2: Verify Each Model Type on CPU**
   ```bash
   python generators/models/test_all_models.py --cpu-only
   ```

3. **Phase 3: Test GPU Acceleration**
   ```bash
   python generators/models/test_all_models.py --cuda-only
   ```

4. **Phase 4: Test OpenVINO Acceleration**
   ```bash
   python generators/models/test_all_models.py --openvino-only
   ```

5. **Phase 5: Generate Comprehensive Report**
   ```bash
   python generate_test_report.py
   ```

## Implementation Tips

1. **Architecture Identification**: Ensure models are correctly identified by architecture type
2. **Model-Specific Inputs**: Adjust test inputs based on the model's expected format
3. **Architecture-Specific Handling**: Use the appropriate handling for each architecture:
   - Decoder-only: Set padding token
   - Encoder-decoder: Provide decoder inputs
   - Vision models: Use proper image tensor shape
4. **Hardware Acceleration**: Test on all available hardware backends
5. **Error Handling**: Implement robust error classification
6. **Performance Metrics**: Collect inference time, memory usage, and load time metrics
7. **Documentation**: Update test_report.md with new test results

## Comprehensive Test Toolkit (New!)

We've developed a unified toolkit to streamline all test-related operations:

```bash
# View available commands
python test_integration.py --help

# List available model families
python test_integration.py --list

# Generate a test for a specific model
python test_integration.py --generate --models bert

# Test a specific model
python test_integration.py --run --models bert

# Run tests for all core models
python test_integration.py --run --core

# Verify syntax of all test files
python test_integration.py --verify

# Generate test coverage report
python test_integration.py --report

# Fix indentation in test files
python test_integration.py --fix

# Run the end-to-end integration
python test_integration.py --all --core

# Generate tests in batches for specific architecture
python test_integration.py --generate --arch encoder-only
```

This toolkit provides a consistent, easy-to-use interface for all test-related tasks, ensuring reliable results while reducing the learning curve for new contributors.

See `INTEGRATION_README.md` for complete documentation and `HF_TEST_TROUBLESHOOTING_GUIDE.md` for solving common issues.

### Architecture-Aware Testing (Enhanced March 2025)

Our test generator is now fully architecture-aware with comprehensive indentation fixing capabilities. The system handles different model families appropriately through a modular template system with proper indentation, consistent formatting, and architecture-specific handling.

For details on the fixes and improvements, see `TESTING_FIXES_SUMMARY.md`. For the phased implementation plan to achieve 100% coverage, see `INTEGRATION_PLAN.md`.

## Expected Timeline

- **Week 1**: Complete generation of all missing test files
- **Week 2**: Verify CPU implementation for all model types
- **Week 3**: Test GPU acceleration for compatible models
- **Week 4**: Test OpenVINO acceleration and generate final report

## Generator Test Suite

To ensure the test generator produces valid and consistent files, we've implemented a comprehensive test suite:

```bash
# Run the test suite for the generator
python test_generator_test_suite.py

# Run specific test cases
python test_generator_test_suite.py TestGeneratorTestCase.test_file_generation
```

The test suite validates:

1. **Syntax Correctness**: Ensures generated files pass Python syntax checks
2. **Architecture Specifics**: Validates that each model family includes proper architecture-specific code
3. **Hardware Detection**: Confirms hardware detection code is included
4. **Mock Imports**: Tests that graceful degradation with MagicMock is implemented
5. **Template Consistency**: Verifies consistent structure across all templates

## CI/CD Integration

Our CI/CD pipeline now includes automatic syntax validation for generated test files:

1. **Pre-commit Hook**: The git pre-commit hook runs `test_generator_test_suite.py` to validate any changes to the generator
2. **GitHub Actions**: Workflow automatically runs the test suite on all PRs
3. **Nightly Tests**: Complete test generation and validation runs nightly on all 300 model types
4. **Syntax Enforcement**: Standardized indentation and code style is enforced by the CI/CD pipeline

The CI/CD workflow includes these steps:
- Run the generator test suite
- Run syntax validation on all generated files
- Perform limited functionality tests on key models
- Generate coverage report
- Update documentation with latest coverage metrics

## Resources

- **API Documentation**: transformers.huggingface.co/docs
- **Model Hub**: huggingface.co/models
- **Model JSON**: huggingface_model_types.json (full list of 300 model types)
- **Test Results**: /test/skills/collected_results
- **Coverage Report**: /test/skills/HF_COVERAGE_REPORT.md
- **Test Suite**: /test/skills/test_generator_test_suite.py
- **Generator**: /test/test_generator.py