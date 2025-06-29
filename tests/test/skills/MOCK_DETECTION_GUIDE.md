# Hugging Face Test Mock Detection System

This guide explains the mock detection system implemented in our Hugging Face test files, how to use it, and how to maintain it.

## Overview

The mock detection system provides a clear visual indication of whether tests are running with real model inference or mock objects. This is crucial for:

1. CI/CD environments where dependencies may be unavailable
2. Testing on systems with limited resources
3. Quickly identifying the test execution mode
4. Ensuring consistent behavior across different environments

## Key Features

- **Environment Variable Control**: Force mocking of specific dependencies
- **Visual Indicators**: Colorized terminal output for clear distinction
- **Detailed Metadata**: Comprehensive information in test results
- **Verification Tools**: Scripts to check, fix, and verify implementation

## Implementation Details

### Environment Variables

The system supports the following environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `MOCK_TORCH` | Mock PyTorch when set to "true" | "false" |
| `MOCK_TRANSFORMERS` | Mock Transformers when set to "true" | "false" |
| `MOCK_TOKENIZERS` | Mock Tokenizers when set to "true" | "false" |
| `MOCK_SENTENCEPIECE` | Mock SentencePiece when set to "true" | "false" |

### Import Pattern

Each dependency is imported with environment variable control:

```python
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

Tests determine whether they're using real inference or mock objects:

```python
# Determine if real inference or mock objects were used
using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
```

### Visual Indicators

The test output clearly indicates the execution mode:

- **Real Inference**: 🚀 Using REAL INFERENCE with actual models
- **Mock Objects**: 🔷 Using MOCK OBJECTS for CI/CD testing only

### Metadata

Test results include detailed information about dependency availability:

```python
"metadata": {
    "has_transformers": HAS_TRANSFORMERS,
    "has_torch": HAS_TORCH,
    "has_tokenizers": HAS_TOKENIZERS,
    "has_sentencepiece": HAS_SENTENCEPIECE,
    "using_real_inference": using_real_inference,
    "using_mocks": using_mocks,
    "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
}
```

## Tools for Implementation and Maintenance

### Single File Fix (`fix_single_file.py`)

A lightweight Python script that:

1. Adds missing environment variable declarations for mock control
2. Adds mock checking to imports
3. Adds missing mock detection logic
4. Ensures syntax is correct before applying changes

Usage:

```bash
# Fix a specific file
python fix_single_file.py --file fixed_tests/test_hf_bert.py
```

### Comprehensive Fix (`fix_all_tests.py`) 

A more comprehensive script that:

1. Tries the lightweight fix first
2. Falls back to more complex fixes if needed
3. Fixes hyphenated model names
4. Fixes mock detection errors
5. Applies manual fixes as a last resort
6. Verifies syntax after all changes

Usage:

```bash
# Fix a specific file
python fix_all_tests.py --file fixed_tests/test_hf_bert.py

# Fix all files in a directory
python fix_all_tests.py --dir fixed_tests

# Fix and rename files (replace hyphens with underscores)
python fix_all_tests.py --dir fixed_tests --rename-files

# Fix with extra verification
python fix_all_tests.py --dir fixed_tests --verify
```

### Verify and Fix Script (`verify_and_fix_all_tests.sh`)

A shell script that:

1. Checks all test files for issues
2. Applies fixes where needed
3. Verifies fixed files with different mock configurations
4. Generates detailed reports

Usage:

```bash
# Run the full verification and fix process
./verify_and_fix_all_tests.sh
```

### Additional Specialized Tools

- `add_colorized_output.py`: Add colorized terminal output for better visibility
- `add_env_mock_support.py`: Add environment variable control for dependency mocking
- `add_mock_detection_to_templates.py`: Add mock detection to template files
- `fix_mock_detection_errors.py`: Fix specific mock detection issues

## Running Tests with Mock Control

Use environment variables to control test behavior:

```bash
# Run with all real dependencies (if available)
python fixed_tests/test_hf_bert.py

# Mock PyTorch only
MOCK_TORCH=true python fixed_tests/test_hf_bert.py

# Mock Transformers only
MOCK_TRANSFORMERS=true python fixed_tests/test_hf_bert.py

# Mock multiple dependencies
MOCK_TORCH=true MOCK_TOKENIZERS=true python fixed_tests/test_hf_bert.py
```

## CI/CD Integration

See the `ci_templates` directory for ready-to-use CI/CD workflow configurations for:

- GitHub Actions
- GitLab CI
- Jenkins

These templates include:

1. Syntax Validation: Verify correct mock detection implementation
2. Configuration Testing: Test with different mock configurations
3. Real Inference Testing: Optional stage for environments with all dependencies

## Best Practices

1. **Always include mock detection** in new test files
2. **Verify changes** with `comprehensive_test_fix.py` after modifying test files
3. **Document environment variables** in test documentation
4. **Use visual indicators** consistently for clarity
5. **Run tests with different configurations** to ensure flexibility

## Troubleshooting

If mock detection is not working correctly:

1. Check that all required environment variables are defined
2. Verify that import blocks include the mock control logic
3. Ensure visual indicators are present in the main function
4. Run with `--check-only` to identify specific issues
5. Apply fixes automatically with `comprehensive_test_fix.py`

## Support and Maintenance

This mock detection system is maintained as part of the test framework. For issues or enhancements:

1. Check the existing issues in the repository
2. Run the verification tools to diagnose problems
3. Contribute improvements that maintain backward compatibility