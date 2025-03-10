# Hugging Face Model Test Automation Guide

## Complete Test Coverage for All 300 Model Types

This document outlines the process for achieving 100% test coverage for all 300 Hugging Face model types as specified in `huggingface_model_types.json`.

## Current Status (March 1, 2025)

- **Total model types required**: 300
- **Current implementation**: 175 model types
- **Current coverage**: 58.3%
- **Remaining to implement**: 125 model types

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

1. **Registry Definition**: Model-specific registry with default models
2. **Test Class**: Standardized test class for the model family
3. **Pipeline Testing**: Test using transformers pipeline() API
4. **Direct Model Testing**: Test using from_pretrained() API
5. **Hardware Testing**: CPU, CUDA, and OpenVINO hardware acceleration testing
6. **Result Collection**: Comprehensive results with performance metrics

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

1. **Model-Specific Inputs**: Adjust test inputs based on the model's expected format
2. **Hardware Acceleration**: Test on all available hardware backends
3. **Error Handling**: Implement robust error classification
4. **Performance Metrics**: Collect inference time, memory usage, and load time metrics
5. **Documentation**: Update test_report.md with new test results

## Expected Timeline

- **Week 1**: Complete generation of all missing test files
- **Week 2**: Verify CPU implementation for all model types
- **Week 3**: Test GPU acceleration for compatible models
- **Week 4**: Test OpenVINO acceleration and generate final report

## Resources

- **API Documentation**: transformers.huggingface.co/docs
- **Model Hub**: huggingface.co/models
- **Model JSON**: huggingface_model_types.json (full list of 300 model types)
- **Test Results**: /test/skills/collected_results
- **Coverage Report**: /test/skills/HF_COVERAGE_REPORT.md