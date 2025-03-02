# Running Model Tests with the Comprehensive Test Framework

This guide provides step-by-step instructions for running model tests with the new comprehensive test framework.

## Prerequisites

Before running tests, ensure you have:

1. **Required Libraries**:
   ```bash
   pip install transformers torch numpy
   
   # For vision models
   pip install pillow
   
   # For audio models
   pip install librosa soundfile
   
   # For OpenVINO tests
   pip install openvino optimum[openvino]
   ```

2. **Test Data Files**:
   ```bash
   # Ensure test files exist
   ls -la test.jpg test.mp3
   
   # Create them if they don't exist
   # For image test
   wget -O test.jpg https://github.com/huggingface/transformers/raw/main/docs/source/imgs/transformers_logo_name.png
   
   # For audio test (optional)
   wget -O test.mp3 https://github.com/librosa/librosa/raw/main/tests/data/test1_44100.wav
   ```

## Quick Start Guide

### Running a Single Test

To run a test for a specific model across all hardware backends:

```bash
# Navigate to the test directory
cd /home/barberb/ipfs_accelerate_py/test

# Run the test for a specific model
python skills/test_hf_bert_base_uncased.py
```

### Testing Specific Hardware

To test only on specific hardware backends:

```bash
# CPU only (fastest)
python skills/test_hf_bert_base_uncased.py --cpu-only

# CUDA only (if GPU available)
python skills/test_hf_bert_base_uncased.py --cuda-only

# OpenVINO only (if installed)
python skills/test_hf_bert_base_uncased.py --openvino-only
```

### Generating Tests for New Models

To create a test file for a model that doesn't have one yet:

```bash
# Generate a test for a specific model
python generate_comprehensive_tests.py --model bert-base-uncased

# The test will be created at:
# skills/test_hf_bert_base_uncased.py
```

### Batch Testing Multiple Models

To test multiple models in sequence:

```bash
# Run a batch of tests
for model in bert-base-uncased gpt2 t5-small; do
  echo "Testing $model..."
  python generate_comprehensive_tests.py --model $model
  python skills/test_hf_$(echo $model | tr '-/.' '_').py --cpu-only
done
```

## Advanced Usage

### Performance Testing

To run comprehensive performance tests across hardware:

```bash
# Full performance benchmark
python skills/test_hf_bert_base_uncased.py --verbose

# View the results with a JSON viewer
cat skills/collected_results/test_bert_base_uncased_*.json | python -m json.tool
```

### Batch Processing Tests

To test batch processing capabilities:

```bash
# Run with batch processing (default)
python skills/test_hf_bert_base_uncased.py

# Disable batch processing
python skills/test_hf_bert_base_uncased.py --no-batch
```

### Parallel Testing

To control parallel test execution:

```bash
# Enable parallel testing (default)
python skills/test_hf_bert_base_uncased.py

# Disable parallel testing
python skills/test_hf_bert_base_uncased.py --no-parallel
```

## Generated Test File Structure

The generated test files follow a consistent structure:

1. **Imports and Setup**: Sets up environment and imports the `ComprehensiveModelTester`
2. **Model Class**: Creates a test class for the specific model
3. **Test Methods**:
   - `test()`: Main test method that runs comprehensive tests
   - `run_tests()`: Legacy method for compatibility
   - `__test__()`: Default test entry point

## Interpreting Test Results

After running tests, results are stored in JSON files in `skills/collected_results/`:

```bash
# List test results
ls -la skills/collected_results/

# Examine a specific test result
cat skills/collected_results/test_bert_base_uncased_*.json | less
```

Key metrics to look for:

1. **Implementation Status**: Look for `"implementation_type": "REAL"` vs `"MOCK"`
2. **Performance**: Check `"average_time"`, `"min_time"`, and `"max_time"`
3. **Memory Usage**: Monitor `"memory_usage"` for each device
4. **Success Rates**: Check `"success": true` entries

## Summary Command Sheet

```bash
# Generate test for a model
python generate_comprehensive_tests.py --model MODEL_NAME

# Run test with all hardware
python skills/test_hf_MODEL_NAME.py

# Run test with specific hardware
python skills/test_hf_MODEL_NAME.py --cpu-only
python skills/test_hf_MODEL_NAME.py --cuda-only
python skills/test_hf_MODEL_NAME.py --openvino-only

# Control test behavior
python skills/test_hf_MODEL_NAME.py --no-batch --no-parallel --verbose

# View results
cat skills/collected_results/test_MODEL_NAME_*.json | python -m json.tool
```

Replace `MODEL_NAME` with the normalized model name (e.g., `bert_base_uncased` for `bert-base-uncased`).