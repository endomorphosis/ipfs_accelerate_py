# Improved HuggingFace Model Tests

This directory contains improved, pytest-compatible tests for HuggingFace models.

## Overview

These tests demonstrate a better testing approach compared to the existing test_hf_*.py files:

### Key Improvements

1. **Proper Pytest Structure**
   - Uses `test_*` functions that pytest can discover
   - Organized into test classes
   - Uses pytest fixtures for setup/teardown
   
2. **Real Assertions**
   - Tests actually validate behavior with `assert` statements
   - No more returning `{"success": True}` dictionaries
   - Tests can fail meaningfully
   
3. **Hardware Testing**
   - Tests across multiple hardware platforms (CPU, CUDA, MPS, ROCm)
   - Uses pytest markers (`@pytest.mark.cuda`, etc.)
   - Automatically skips unavailable hardware
   
4. **Performance Benchmarks**
   - Measures inference latency
   - Tracks memory usage
   - Can detect performance regressions
   
5. **Error Handling**
   - Tests invalid inputs
   - Validates error messages
   - Tests edge cases

## Structure

```
test/improved/
├── __init__.py
├── test_hf_bert_improved.py        # Improved BERT tests
├── test_hf_clip_improved.py        # Improved CLIP tests (TODO)
├── test_hf_llama_improved.py       # Improved LLaMA tests (TODO)
└── README.md                        # This file

test/common/
├── test_utils.py                    # Test utilities and assertions
└── test_template_improved.py       # Template for new tests
```

## Running Tests

### Run all improved tests:
```bash
pytest test/improved/ -v
```

### Run specific test file:
```bash
pytest test/improved/test_hf_bert_improved.py -v
```

### Run specific test class:
```bash
pytest test/improved/test_hf_bert_improved.py::TestBERTInference -v
```

### Run specific test function:
```bash
pytest test/improved/test_hf_bert_improved.py::TestBERTInference::test_forward_pass -v
```

### Run tests with specific markers:
```bash
# Run only CUDA tests
pytest test/improved/ -m cuda -v

# Run only CPU tests
pytest test/improved/ -m cpu -v

# Run hardware tests
pytest test/improved/ -m hardware -v

# Run benchmark tests
pytest test/improved/ -m benchmark -v

# Skip slow tests
pytest test/improved/ -m "not slow" -v
```

### Run with coverage:
```bash
pytest test/improved/ --cov=ipfs_accelerate_py --cov-report=html -v
```

## Test Structure Example

```python
import pytest
from test.common.test_utils import ModelTestUtils

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model once for all tests."""
    model = load_model("bert-base-uncased")
    tokenizer = load_tokenizer("bert-base-uncased")
    return model, tokenizer

@pytest.mark.model
@pytest.mark.text
class TestModelInference:
    """Test model inference."""
    
    def test_forward_pass(self, model_and_tokenizer):
        """Test basic forward pass."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("test", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Assertions validate behavior
        assert outputs is not None
        ModelTestUtils.assert_tensor_valid(outputs.last_hidden_state)

@pytest.mark.hardware
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    """Test on CUDA devices."""
    
    def test_cuda_inference(self, model_and_tokenizer):
        """Test inference on CUDA."""
        model, tokenizer = model_and_tokenizer
        model = model.to("cuda")
        
        inputs = tokenizer("test", return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        ModelTestUtils.assert_device_correct(outputs.last_hidden_state, "cuda")
```

## Test Utilities

The `test/common/test_utils.py` module provides helper functions:

### ModelTestUtils

- `assert_model_loaded(model, name)` - Validate model loaded correctly
- `assert_tokenizer_loaded(tokenizer, name)` - Validate tokenizer loaded
- `assert_tensor_valid(tensor, name)` - Check tensor has no NaN/Inf
- `assert_output_shape(output, expected_shape)` - Validate output shape
- `assert_device_correct(tensor, device)` - Check tensor on correct device
- `measure_inference_time(model, inputs)` - Benchmark inference speed
- `measure_memory_usage(model, inputs)` - Measure memory consumption
- `create_sample_text_inputs(tokenizer, texts)` - Create test inputs
- `create_sample_image_inputs(processor, size)` - Create test images
- `compare_outputs(output1, output2)` - Compare model outputs

### HardwareTestUtils

- `get_available_devices()` - List available hardware
- `assert_model_works_on_device(model, inputs, device)` - Test device compatibility

### PerformanceTestUtils

- `assert_inference_time_within_threshold(actual, baseline, threshold)` - Performance regression check
- `assert_memory_within_threshold(actual, baseline, threshold)` - Memory regression check
- `create_performance_report(model_name, timing_stats, memory_stats)` - Generate report

## Creating New Tests

1. **Copy the template:**
   ```bash
   cp test/common/test_template_improved.py test/improved/test_hf_newmodel_improved.py
   ```

2. **Update model configuration:**
   ```python
   MODEL_ID = "model/name"
   MODEL_NAME = "NewModel"
   TASK_TYPE = "task-type"
   ```

3. **Customize tests:**
   - Add model-specific tests
   - Adjust expected values (hidden_size, num_layers, etc.)
   - Add special features tests

4. **Run tests:**
   ```bash
   pytest test/improved/test_hf_newmodel_improved.py -v
   ```

## Pytest Markers

Available markers (defined in pytest.ini):

- `@pytest.mark.model` - Model test
- `@pytest.mark.hardware` - Hardware compatibility test
- `@pytest.mark.integration` - Integration test
- `@pytest.mark.benchmark` - Performance benchmark
- `@pytest.mark.slow` - Slow test (>30s)
- `@pytest.mark.cpu` - Requires CPU
- `@pytest.mark.cuda` - Requires CUDA
- `@pytest.mark.rocm` - Requires ROCm
- `@pytest.mark.mps` - Requires Apple MPS
- `@pytest.mark.openvino` - Requires OpenVINO
- `@pytest.mark.text` - Text model
- `@pytest.mark.vision` - Vision model
- `@pytest.mark.audio` - Audio model
- `@pytest.mark.multimodal` - Multimodal model

## Benefits Over Existing Tests

| Aspect | Old Tests (test_hf_*.py) | New Tests (improved/) |
|--------|--------------------------|----------------------|
| **Pytest Compatible** | ❌ No (class methods, no assertions) | ✅ Yes (test_* functions) |
| **Discoverable** | ❌ No (not found by pytest) | ✅ Yes (auto-discovered) |
| **Assertions** | ❌ Returns dictionaries | ✅ Uses assert statements |
| **Hardware Testing** | ⚠️ Limited | ✅ Comprehensive |
| **Performance** | ⚠️ Load time only | ✅ Full benchmarks |
| **Error Handling** | ❌ Minimal | ✅ Extensive |
| **Fixtures** | ❌ No reuse | ✅ Pytest fixtures |
| **Coverage** | ❌ Unknown | ✅ Trackable with pytest-cov |
| **CI Integration** | ⚠️ Difficult | ✅ Easy |

## Next Steps

1. **Convert existing tests:**
   - Use improved tests as examples
   - Convert top priority models first
   - Gradually migrate test_hf_*.py files

2. **Add more models:**
   - CLIP (multimodal)
   - LLaMA (large language model)
   - Whisper (audio)
   - ViT (vision)
   - T5 (encoder-decoder)

3. **Enhance coverage:**
   - Add more edge cases
   - Test quantization
   - Test distributed inference
   - Add memory leak detection

4. **CI/CD Integration:**
   - Run improved tests in CI
   - Generate coverage reports
   - Track performance trends
   - Alert on regressions

## Contributing

When adding new improved tests:

1. Follow the template structure
2. Use test utilities from `test/common/test_utils.py`
3. Add appropriate pytest markers
4. Include docstrings for all test functions
5. Test on multiple hardware platforms
6. Add performance benchmarks
7. Include error handling tests

## Questions?

See:
- `test/common/test_template_improved.py` - Full template with all test types
- `docs/HF_MODEL_TESTING_REVIEW.md` - Comprehensive testing infrastructure review
- `pytest.ini` - Pytest configuration
- `conftest.py` - Global pytest fixtures
