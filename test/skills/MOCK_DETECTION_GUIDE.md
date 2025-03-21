# Mock Detection System Guide

This document provides a comprehensive guide to the mock detection system implemented across HuggingFace model tests in the IPFS Accelerate Python Framework.

## Overview

The mock detection system provides clear visibility into whether tests are running with real model inference or mock objects. This is particularly important for CI/CD environments where dependencies might be unavailable or should be mocked for faster testing.

## Key Features

1. **Environment Variable Control**: Tests can simulate missing dependencies through environment variables
2. **Visual Indicators**: Colorized output clearly shows whether real inference or mocks are being used
3. **Metadata Tracking**: Test results include information about which dependencies were available
4. **Conditional Imports**: Tests automatically fall back to mocks when dependencies are unavailable

## Environment Variables

The following environment variables control the mock detection system:

| Variable | Purpose | Default |
|----------|---------|---------|
| `MOCK_TORCH` | Force tests to mock PyTorch | `False` |
| `MOCK_TRANSFORMERS` | Force tests to mock Transformers | `False` |
| `MOCK_TOKENIZERS` | Force tests to mock Tokenizers | `False` |
| `MOCK_SENTENCEPIECE` | Force tests to mock SentencePiece | `False` |

Setting any of these to `True` will cause the corresponding dependency to be mocked, simulating that the dependency is unavailable.

Example:
```bash
MOCK_TORCH=True python test_hf_bert.py
```

## Visual Indicators

The tests display colorized output to clearly indicate which mode they are running in:

- 🚀 **REAL INFERENCE** (green): Tests are using actual models and running real inference
- 🔷 **MOCK OBJECTS** (blue): Tests are using mock objects for CI/CD testing only

## Implementation Details

### Import Pattern

Each test file implements the following pattern for importing dependencies:

```python
# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'

# Try to import torch
try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")
```

### Detection Logic

The detection logic determines whether real inference or mocks are being used:

```python
# Determine if real inference or mock objects were used
using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
```

### Visual Output

Tests display colorized output to indicate the mode they're running in:

```python
# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

# Indicate real vs mock inference clearly
if using_real_inference and not using_mocks:
    print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
else:
    print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
    print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
```

## Verification Tools

The following tools are available to verify the mock detection system:

- `verify_mock_detection.py`: Verifies individual files with different environment variable combinations
- `verify_all_mock_detection.py`: Comprehensive tool for checking, fixing, and verifying all test files
- `add_mock_detection_to_templates.py`: Adds mock detection to template files
- `add_env_mock_support.py`: Adds environment variable-based mocking support
- `add_colorized_output.py`: Adds colorized visual indicators
- `fix_mock_detection_errors.py`: Fixes common errors in mock detection implementation

## Best Practices

1. **Always check mock detection**: Verify that new test files correctly implement the mock detection system
2. **Include all dependencies**: Ensure all required dependencies are imported with proper mock fallbacks
3. **Use environment variables**: Control mock behavior through environment variables, not hard-coded flags
4. **Maintain visual indicators**: Keep the colorized output to clearly indicate which mode the test is running in
5. **Verify changes**: After modifying test files, verify that mock detection still works correctly

## Running Mock Detection Verification

To verify the mock detection system across all test files:

```bash
# Check mock detection in all files without fixing
python verify_all_mock_detection.py --check-only

# Check and fix mock detection in all files
python verify_all_mock_detection.py --fix

# Check, fix, and verify mock detection with different environment variables
python verify_all_mock_detection.py --fix --verify

# Verify mock detection in a specific file
python verify_all_mock_detection.py --file path/to/test_file.py --fix --verify
```

A verification report will be generated with details about each file.

## Troubleshooting

If mock detection is not working correctly, check the following:

1. **Missing imports**: Ensure all necessary dependencies are imported with proper mock fallbacks
2. **Environment variables**: Check that environment variables are being read correctly
3. **Detection logic**: Verify that the detection logic correctly determines when mocks are being used
4. **Visual indicators**: Ensure the colorized output is properly implemented

For automated fixing, use the `fix_mock_detection_errors.py` script:

```bash
python fix_mock_detection_errors.py --file path/to/test_file.py
```

## Summary

The mock detection system provides critical visibility into how tests are running, making it clear when real inference is being performed versus when mock objects are being used. This is especially important for CI/CD environments and developers working with incomplete dependencies.