# Mock Detection Guide for IPFS Accelerate Testing Framework

This guide explains how to use and implement mock detection in test files, ensuring consistent handling of environments with and without required dependencies.

## Overview

The mock detection system ensures tests can run in both:
- **Real Inference Mode** - when all dependencies are available (torch, transformers, etc.)
- **Mock Mode** - when running in CI/CD or environments without dependencies

This system provides clear indicators (🚀 for real inference, 🔷 for mocks) and standardized reporting.

## Implementation

### Basic Implementation

Add these lines to your test file:

```python
# Define flags for dependency detection
HAS_TORCH = True
try:
    import torch
    if torch.__version__:
        pass  # Successfully imported torch
except (ImportError, AttributeError):
    HAS_TORCH = False

HAS_TRANSFORMERS = True
try:
    import transformers
    if transformers.__version__:
        pass  # Successfully imported transformers
except (ImportError, AttributeError):
    HAS_TRANSFORMERS = False

HAS_TOKENIZERS = True
try:
    import tokenizers
    if tokenizers.__version__:
        pass  # Successfully imported tokenizers
except (ImportError, AttributeError):
    HAS_TOKENIZERS = False

HAS_SENTENCEPIECE = True
try:
    import sentencepiece
    if sentencepiece.__version__:
        pass  # Successfully imported sentencepiece
except (ImportError, AttributeError):
    HAS_SENTENCEPIECE = False

# Check for mock status
using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
```

### Print Status Indicators

Add this code to display indicators:

```python
print("\nTEST RESULTS SUMMARY:")

if using_real_inference and not using_mocks:
    print(f"🚀 Using REAL INFERENCE with actual models")
else:
    print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
    print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
```

### Conditional Test Logic

Use the detection flags to implement conditional behavior:

```python
if using_real_inference and not using_mocks:
    # Run with real models and full functionality
    model = AutoModel.from_pretrained(model_id)
    result = model(**inputs)
else:
    # Use mocked objects or simplified testing
    model = MagicMock()
    model.return_value = {"mock_output": "test_value"}
    result = model(**inputs)
```

## Visualization Tools

The framework includes visualization tools for mock detection status:

```bash
# Generate all visualizations
python mock_detection_visualization.py

# Generate specific visualizations
python mock_detection_visualization.py --no-dashboard --no-heatmap

# Analyze custom test directory
python mock_detection_visualization.py --test-dir path/to/tests
```

## Visualization Types

1. **Implementation Heatmap** - Shows which test files have mock detection
2. **Implementation Summary** - Overall implementation statistics
3. **Model Family Analysis** - Implementation rates by model family
4. **Test Result Analysis** - Compares success rates with real vs. mock dependencies
5. **Interactive Dashboard** - Combines key visualizations in one view
6. **Markdown Report** - Generated report with findings and recommendations

## Mock Testing

You can test your mock detection implementation with:

```bash
# Test simple mock detection
python simple_mock_test.py

# Test with specific dependencies mocked
python mock_test_demo.py --test-file skills/fixed_tests/test_hf_bert.py --mock torch transformers

# Run comprehensive mock detection tests
python test_mock_detection.py
```

## Best Practices

1. **Consistent Implementation** - Include mock detection in all test files
2. **Complete Dependency Checks** - Check all relevant dependencies
3. **Clear Indicators** - Use standard emoji indicators (🚀/🔷)
4. **Granular Conditionals** - Implement specific handling for different dependencies
5. **Metadata in Results** - Include mock status in test result metadata
6. **Descriptive Outputs** - Provide clear information about mock status

## Examples

See these example files:
- `simple_mock_test.py` - Basic implementation
- `manual_mock_test.py` - Manual testing of indicators
- `mock_test_demo.py` - Dynamic mocking with command-line options
- `test_mock_detection.py` - Comprehensive test suite

## Integration with Test Generator

The test generator automatically adds mock detection to all generated test files. When using the test generator, mock detection is included in the generated output.

```bash
python skills/test_generator_fixed.py --model-id bert-base-uncased
```