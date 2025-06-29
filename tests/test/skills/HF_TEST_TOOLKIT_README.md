# HuggingFace Test Toolkit

A comprehensive toolkit for managing HuggingFace model tests in the IPFS Accelerate Python framework.

> **HIGH PRIORITY OBJECTIVE:** Achieving 100% test coverage for all 300+ HuggingFace model classes with end-to-end validation is a critical priority. Current coverage is only 57.6% (114/198 tracked models).

## Overview

This toolkit provides a unified interface for all test-related operations:
- Generating tests for new model architectures
- Running and validating existing tests
- Batch generating multiple tests
- Analyzing test coverage
- Managing CI/CD integration
- Ensuring code quality and consistency

## Quick Start

```bash
# Generate a test for a specific model family
./test_toolkit.py generate bert

# Run a test for a specific model
./test_toolkit.py test bert --model-id bert-base-uncased

# Generate a coverage report
./test_toolkit.py coverage

# Run the test suite
./test_toolkit.py suite

# Batch generate multiple tests
./test_toolkit.py batch 10

# Run tests for all core models
./test_toolkit.py all-tests

# Verify syntax for all test files
./test_toolkit.py verify

# Install the pre-commit hook
./test_toolkit.py install-hook

# Regenerate core model tests
./test_toolkit.py regenerate
```

## Components

The toolkit includes the following components:

### 1. Test Generator

Generates test files for different model architectures with proper indentation and structure.

```bash
./test_toolkit.py generate MODEL_FAMILY [--output DIR] [--template MODEL]
```

### 2. Test Runner

Runs tests for specific model families with customizable options.

```bash
./test_toolkit.py test MODEL_FAMILY [--model-id ID] [--cpu-only]
```

### 3. Test Suite

Validates the test generator functionality and ensures generated files meet quality standards.

```bash
./test_toolkit.py suite
```

### 4. Coverage Analyzer

Generates reports on test coverage across model architectures.

```bash
./test_toolkit.py coverage
```

### 5. Batch Generator

Generates tests for multiple model families in a single operation.

```bash
./test_toolkit.py batch BATCH_SIZE [--all]
```

### 6. Validation Tools

Verifies syntax and functionality of all test files.

```bash
./test_toolkit.py verify
```

### 7. CI/CD Integration

Installs pre-commit hooks and integrates with GitHub Actions and GitLab CI.

```bash
./test_toolkit.py install-hook
```

## Architecture Support

The toolkit provides architecture-specific templates for different model families:

- **Encoder-Only Models**: BERT, RoBERTa, ALBERT, etc.
- **Decoder-Only Models**: GPT-2, LLaMA, Falcon, etc.
- **Encoder-Decoder Models**: T5, BART, Pegasus, etc.
- **Vision Models**: ViT, Swin, BEiT, etc.
- **Multimodal Models**: CLIP, BLIP, LLaVA, etc.
- **Audio Models**: Whisper, Wav2Vec2, HuBERT, etc.

## Documentation

Comprehensive documentation is available for all aspects of testing:

- [HF_TEST_DEVELOPMENT_GUIDE.md](HF_TEST_DEVELOPMENT_GUIDE.md): Guide for test developers
- [HF_TEST_IMPLEMENTATION_CHECKLIST.md](HF_TEST_IMPLEMENTATION_CHECKLIST.md): Checklist for test implementation
- [HF_TEST_CICD_INTEGRATION.md](HF_TEST_CICD_INTEGRATION.md): CI/CD integration guide
- [HF_TEST_TROUBLESHOOTING_GUIDE.md](HF_TEST_TROUBLESHOOTING_GUIDE.md): Troubleshooting common issues
- [TEST_AUTOMATION.md](TEST_AUTOMATION.md): Test automation overview

## Directory Structure

```
/test/
├── test_generator.py               # Main test generator
├── test_hf_{MODEL_FAMILY}.py       # Generated test files
│
├── skills/
│   ├── test_toolkit.py             # Unified toolkit interface
│   ├── test_generator_test_suite.py # Test suite for the generator
│   ├── visualize_test_coverage.py  # Coverage visualization tool
│   ├── generate_batch_tests.py     # Batch test generator
│   ├── regenerate_model_tests.sh   # Script to regenerate tests
│   ├── pre-commit                  # Git pre-commit hook
│   ├── install_pre_commit.sh       # Pre-commit hook installer
│   │
│   ├── ci_templates/               # CI/CD templates
│   │   ├── github-actions-test-validation.yml
│   │   └── gitlab-ci.yml
│   │
│   ├── coverage_visualizations/    # Generated coverage reports
│   ├── backups/                    # Backup files
│   └── temp_generated/             # Temporary files
```

## Development Workflow

1. **Check Test Coverage**:
   ```bash
   ./test_toolkit.py coverage
   ```

2. **Identify Missing Tests**:
   ```bash
   cd /test/skills
   python generate_batch_tests.py --list-missing
   ```

3. **Generate Tests for Missing Models**:
   ```bash
   ./test_toolkit.py batch 10
   ```

4. **Verify Generated Tests**:
   ```bash
   ./test_toolkit.py verify
   ```

5. **Run Tests**:
   ```bash
   ./test_toolkit.py all-tests
   ```

6. **Install CI/CD Integration**:
   ```bash
   ./test_toolkit.py install-hook
   ```

## Best Practices

### 1. NEVER Edit Generated Files Directly

> **CRITICAL:** Always modify generators and templates, NEVER edit the generated test files directly.

❌ **INCORRECT**: Editing test files in the output directory
```bash
vim test_hf_bert.py                 # WRONG!
vim fixed_tests/test_hf_gpt_j.py    # WRONG!
```

✅ **CORRECT**: Modify the generator and templates, then regenerate
```bash
# Edit the generator source
vim skills/test_generator_fixed.py  # Correct!
vim skills/templates/encoder_only_template.py  # Correct!

# Then regenerate affected tests
./test_toolkit.py regenerate
```

### 2. Always Reference Official Documentation

When implementing or fixing test files, always reference:
- **Official Transformers Documentation**: `/home/barberb/ipfs_accelerate_py/test/doc-builder/build`
- **Current Implementation Status**: `reports/missing_models.md`

### 3. Core Best Practices

- **Regenerate After Changes**: After modifying the test generator, regenerate all core tests:
  ```bash
  ./test_toolkit.py regenerate
  ```

- **Run Test Suite Regularly**: Validate the test generator functionality:
  ```bash
  ./test_toolkit.py suite
  ```

- **Keep Coverage Updated**: Generate updated coverage reports after adding new tests:
  ```bash
  ./test_toolkit.py coverage
  ```

- **Use Pre-commit Hooks**: Install and use the pre-commit hook to ensure code quality:
  ```bash
  ./test_toolkit.py install-hook
  ```

- **Batch Generation**: When implementing many model families, use batch generation:
  ```bash
  ./test_toolkit.py batch 20 --priority high
  ```

## Contributing

When adding new tests or enhancing existing ones:

1. **Focus on High Priority Models First:**
   - Consult current coverage report to identify high priority missing models
   - Use `./test_toolkit.py batch --priority high` to implement high priority models

2. **Follow the Proper Modification Process:**
   - NEVER edit generated test files directly
   - Always modify generators and templates, then regenerate tests
   - Apply architecture-specific guidance in [HF_TEST_DEVELOPMENT_GUIDE.md](HF_TEST_DEVELOPMENT_GUIDE.md)

3. **Ensure Changes Are Properly Propagated:**
   - Update the implementation checklist in [HF_TEST_IMPLEMENTATION_CHECKLIST.md](HF_TEST_IMPLEMENTATION_CHECKLIST.md)
   - Run the test suite to validate your changes
   - Generate an updated coverage report
   - Verify that fixes apply consistently across model architectures
   - Document any special cases in the generator code comments

4. **End-to-End Validation:**
   - When possible, validate models with real weights (not just mocks)
   - Follow guidance in [GENERATOR_MODIFICATION_GUIDE.md](GENERATOR_MODIFICATION_GUIDE.md)
   - Document transformers version compatibilities

5. **Submit your PR**:
   - Ensure all changes are made to generators and templates
   - Include updated coverage reports
   - Document major modifications to the generator infrastructure

## Troubleshooting

If you encounter issues, consult the [HF_TEST_TROUBLESHOOTING_GUIDE.md](HF_TEST_TROUBLESHOOTING_GUIDE.md) or run:

```bash
./test_toolkit.py help
```

---

Last updated: March 19, 2025