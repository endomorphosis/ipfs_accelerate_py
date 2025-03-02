# Advanced Testing for IPFS Accelerate

This directory contains advanced testing capabilities for the IPFS Accelerate framework, focusing on batch inference and model quantization features.

## Test Files

1. **test_batch_inference.py**: Tests batch processing capabilities across different model types, platforms, and batch sizes
2. **test_quantization.py**: Tests model quantization (FP16, INT8) with CUDA and OpenVINO backends
3. **run_advanced_tests.py**: Unified test runner script that can execute both tests and generate combined reports

## Usage

### Running All Tests

To run both batch inference and quantization tests:

```bash
python run_advanced_tests.py
```

Results will be saved in their respective directories:
- Batch inference: `/test/batch_inference_results/`
- Quantization: `/test/quantization_results/`
- Combined reports: `/test/performance_results/`

### Running Specific Tests

To run only batch inference tests:

```bash
python run_advanced_tests.py --tests batch
```

To run only quantization tests:

```bash
python run_advanced_tests.py --tests quantization
```

### Customizing Batch Inference Tests

```bash
python run_advanced_tests.py --tests batch \
    --model-types bert,t5,clip \
    --batch-sizes 1,2,4,8 \
    --platforms cpu,cuda \
    --specific-model bert:prajjwal1/bert-tiny \
    --fp16
```

### Options

- `--tests`: Which tests to run (`batch`, `quantization`, or `all`)
- `--output-dir`: Custom directory to save test results
- `--model-types`: Comma-separated list of model types to test
- `--batch-sizes`: Comma-separated list of batch sizes to test
- `--platforms`: Comma-separated list of platforms to test (`cpu`, `cuda`, `openvino`)
- `--specific-model`: Specify a model for a given type (format: `type:model_name`)
- `--fp16`: Use FP16 precision for CUDA tests

## Test Requirements

These tests have the following requirements:

1. PyTorch with CUDA support (for GPU tests)
2. OpenVINO runtime (for OpenVINO tests)
3. Transformers library
4. Access to test models (small, open-access models are used by default)
5. Test audio and image files in the test directory (`test.mp3` and `test.jpg`)

## Expected Output

The tests will generate detailed reports in Markdown format that include:

1. **Batch Inference Results**:
   - Success rates by model type, platform, and batch size
   - Throughput scaling analysis with batch size
   - Memory usage statistics
   - Performance recommendations

2. **Quantization Results**:
   - Success rates by model type and precision
   - Performance comparisons between precisions
   - Memory reduction analysis
   - Model-specific recommendations

3. **Combined Summary Report**:
   - Overall test statistics
   - Implementation status assessment
   - Production readiness evaluation
   - Strategic recommendations for deployment