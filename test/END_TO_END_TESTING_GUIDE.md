# End-to-End Testing Guide

This guide explains the new end-to-end testing approach implemented in the IPFS Accelerate Python Framework.

## Introduction

Traditional testing approaches test components (skills, tests, benchmarks) individually, which can lead to integration issues when they're combined. Our new end-to-end testing framework instead:

1. Generates all components together
2. Tests them as a unified whole
3. Compares results against known good outputs
4. Generates documentation automatically

This approach focuses on fixing issues at the generator/template level rather than in individual files, improving maintenance efficiency and ensuring consistency across the codebase.

## Getting Started

The end-to-end testing framework is located in `/generators/runners/end_to_end/`. To run a basic test:

```bash
cd /home/barberb/ipfs_accelerate_py/test
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cpu
```

This will:
1. Generate skill, test, and benchmark files for bert-base-uncased on CPU
2. Run the test and collect results
3. Compare with expected results
4. Generate a summary report

## Core Components

### Directory Structure

```
generators/
├── expected_results/        # Expected outputs for regression testing
├── collected_results/       # Actual test results with timestamps
├── model_documentation/     # Generated documentation
└── runners/
    └── end_to_end/          # End-to-end testing framework scripts
```

### Key Scripts

- `run_e2e_tests.py`: Main script for running tests
- `template_validation.py`: Validation and comparison logic
- `model_documentation_generator.py`: Documentation generator
- `simple_utils.py`: Utility functions

## Common Workflows

### Testing a New Model

```bash
# First generate expected results
python generators/runners/end_to_end/run_e2e_tests.py --model your-new-model --hardware cpu --update-expected

# Then run the test normally
python generators/runners/end_to_end/run_e2e_tests.py --model your-new-model --hardware cpu

# Generate documentation
python generators/runners/end_to_end/run_e2e_tests.py --model your-new-model --hardware cpu --generate-docs
```

### Testing Across Multiple Hardware Platforms

```bash
# Test on multiple platforms at once
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cpu,cuda,webgpu

# Test on priority hardware platforms
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --priority-hardware

# Test on all hardware platforms
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --all-hardware
```

### Working with Model Families

```bash
# Test all models in a family
python generators/runners/end_to_end/run_e2e_tests.py --model-family text-embedding --hardware cpu

# Test all supported models
python generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware
```

### Maintenance Tasks

```bash
# Clean up old test results
python generators/runners/end_to_end/run_e2e_tests.py --clean-old-results --days 14

# Clean up failed test results too
python generators/runners/end_to_end/run_e2e_tests.py --clean-old-results --days 14 --clean-failures
```

## Best Practices

### When to Update Expected Results

Expected results should be updated when:
- A model implementation has changed intentionally
- Hardware optimizations have improved performance
- The model API has changed

To update expected results:

```bash
python generators/runners/end_to_end/run_e2e_tests.py --model your-model --hardware your-hardware --update-expected
```

### How to Fix Issues

When tests fail, the framework generates detailed reports showing the differences between expected and actual results. To fix issues:

1. **Examine the Differences**: Look at the details in the test report
2. **Identify the Root Cause**: Determine if the issue is in the template, generator, or model implementation
3. **Fix at the Generator Level**: Instead of modifying individual test files, fix the generator or template
4. **Update Expected Results**: After fixing the issue, update the expected results

### Documentation Generation

Documentation is automatically generated when running tests with the `--generate-docs` flag:

```bash
python generators/runners/end_to_end/run_e2e_tests.py --model your-model --hardware your-hardware --generate-docs
```

Generated documentation provides:
- Model implementation details
- Hardware-specific optimizations
- Performance characteristics
- Expected behavior
- Code examples

Documentation is stored in `/generators/model_documentation/`.

## Advanced Features

### Verbose Logging

```bash
python generators/runners/end_to_end/run_e2e_tests.py --model your-model --hardware your-hardware --verbose
```

### Keep Temporary Files

```bash
python generators/runners/end_to_end/run_e2e_tests.py --model your-model --hardware your-hardware --keep-temp
```

### Quick Tests

```bash
python generators/runners/end_to_end/run_e2e_tests.py --model your-model --hardware your-hardware --quick-test
```

## Extending the Framework

### Adding New Models

1. Update the `MODEL_FAMILY_MAP` in `run_e2e_tests.py`
2. Create appropriate templates in the template database
3. Run tests with `--update-expected` to generate baseline results

### Adding New Hardware Platforms

1. Update the `SUPPORTED_HARDWARE` list in `run_e2e_tests.py`
2. Add hardware-specific logic to the component generators
3. Add hardware-specific notes to the documentation generator
4. Run tests with `--update-expected` to generate baseline results

## Troubleshooting

### Tests Failing Without Clear Reason

If tests are failing but the implementation looks correct:
- Check if expected results need updating: `--update-expected`
- Examine the differences in the test report
- Check if hardware configuration has changed

### Documentation Issues

If documentation is not generating correctly:
- Check that all components are being generated correctly
- Verify the model and hardware names are correct
- Check the template extractors in `model_documentation_generator.py`

### Framework Errors

If the framework itself is failing:
- Check Python dependencies
- Verify file paths and permissions
- Check the log output with `--verbose`

## References

- [CLAUDE.md](CLAUDE.md) - Project status and current focus areas
- [Model Documentation README](/generators/model_documentation/README.md) - Documentation format and purpose
- [End-to-End Testing Framework README](/generators/runners/end_to_end/README.md) - Detailed framework documentation
- [Expected Results README](/generators/expected_results/README.md) - Expected results format and usage
- [Collected Results README](/generators/collected_results/README.md) - Test result format and interpretation