# End-to-End Testing Framework

## Overview

The End-to-End Testing Framework is a comprehensive solution for testing model implementations across different hardware platforms. Rather than testing individual components separately, this framework focuses on:

1. Generating all components (skill, test, benchmark) together
2. Testing components as a unified whole
3. Comparing results against expected outputs
4. Generating comprehensive documentation

This approach ensures that models work correctly across the entire pipeline and helps identify issues at the generator level rather than in individual files.

## Key Features

- **Joint Component Generation**: Creates skill, test, and benchmark files together
- **Cross-Platform Testing**: Tests models across different hardware platforms
- **Result Validation**: Compares actual results with expected results
- **Automatic Documentation**: Generates Markdown documentation for each implementation
- **Template-Driven Approach**: Focuses fixes on generators rather than individual files
- **Test Result Persistence**: Stores test results for trend analysis and regression detection

## Usage

### Basic Usage

```bash
# Test a single model on a single hardware platform
python run_e2e_tests.py --model bert-base-uncased --hardware cpu

# Test with documentation generation
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs

# Update expected results
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --update-expected

# Enable verbose logging
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --verbose
```

### Advanced Usage

```bash
# Test a model on multiple hardware platforms
python run_e2e_tests.py --model bert-base-uncased --hardware cpu,cuda,webgpu

# Test all models in a family
python run_e2e_tests.py --model-family text-embedding --hardware cpu

# Test all supported models
python run_e2e_tests.py --all-models --priority-hardware

# Clean up old test results
python run_e2e_tests.py --clean-old-results --days 14
```

## Directory Structure

```
generators/
├── expected_results/        # Expected outputs for regression testing
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── expected_result.json
│   │   └── ...
│   └── ...
├── collected_results/       # Actual test results with timestamps
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── 20250310_120000/
│   │   └── ...
│   └── summary/             # Summary reports from test runs
├── model_documentation/     # Generated documentation
│   ├── bert-base-uncased/
│   │   ├── cpu_implementation.md
│   │   └── ...
│   └── ...
└── runners/
    └── end_to_end/          # End-to-end testing framework scripts
        ├── run_e2e_tests.py             # Main script for running tests
        ├── template_validation.py       # Validation and comparison logic
        ├── model_documentation_generator.py  # Documentation generator
        └── simple_utils.py              # Utility functions
```

## Command Line Options

```
usage: run_e2e_tests.py [-h] [--model MODEL] [--model-family MODEL_FAMILY]
                        [--all-models] [--hardware HARDWARE]
                        [--priority-hardware] [--all-hardware] [--quick-test]
                        [--update-expected] [--generate-docs] [--keep-temp]
                        [--clean-old-results] [--days DAYS]
                        [--clean-failures] [--verbose]

End-to-End Testing Framework for IPFS Accelerate

options:
  -h, --help            show this help message and exit
  --model MODEL         Specific model to test
  --model-family MODEL_FAMILY
                        Model family to test (e.g., text-embedding, vision)
  --all-models          Test all supported models
  --hardware HARDWARE   Hardware platforms to test, comma-separated (e.g.,
                        cpu,cuda,webgpu)
  --priority-hardware   Test on priority hardware platforms (cpu, cuda,
                        openvino, webgpu)
  --all-hardware        Test on all supported hardware platforms
  --quick-test          Run a quick test with minimal validation
  --update-expected     Update expected results with current test results
  --generate-docs       Generate markdown documentation for models
  --keep-temp           Keep temporary directories after tests
  --clean-old-results   Clean up old collected results
  --days DAYS           Number of days to keep results when cleaning (default:
                        14)
  --clean-failures      Clean failed test results too
  --verbose             Enable verbose logging
```

## Workflow

1. **Test Execution**:
   - The framework generates skill, test, and benchmark files based on templates
   - It runs tests and collects results
   - It compares results with expected results
   - It generates documentation if requested

2. **Result Management**:
   - Expected results are stored in the `expected_results` directory
   - Actual results are stored in timestamped directories under `collected_results`
   - Summary reports are generated for each test run

3. **Maintenance**:
   - When models or hardware change, update expected results
   - Fix issues in the generators rather than in individual files
   - Clean up old test results periodically

## Adding New Models or Hardware

To add a new model or hardware platform to the testing framework:

1. Update the `MODEL_FAMILY_MAP` in `run_e2e_tests.py` to include the new model
2. Update the `SUPPORTED_HARDWARE` list to include the new hardware platform
3. Run tests with the `--update-expected` flag to generate expected results
4. Run tests normally to verify the implementation works correctly

## Best Practices

1. **Always update expected results after significant changes**:
   ```bash
   python run_e2e_tests.py --model your-model --hardware your-hardware --update-expected
   ```

2. **Generate documentation to understand implementation**:
   ```bash
   python run_e2e_tests.py --model your-model --hardware your-hardware --generate-docs
   ```

3. **Clean up old results periodically**:
   ```bash
   python run_e2e_tests.py --clean-old-results --days 14
   ```

4. **Focus fixes on generators rather than individual files**

## Integration with CI/CD

This framework can be integrated with CI/CD pipelines to:

1. Test all models and hardware platforms automatically
2. Update expected results when approved by reviewers
3. Generate documentation as part of the build process
4. Alert on test failures or regressions

Sample CI/CD configuration:

```yaml
jobs:
  e2e_tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run end-to-end tests
        run: |
          python generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware
```

## Troubleshooting

1. **Tests are failing but the implementation looks correct**:
   - Check if expected results need updating: `--update-expected`
   - Examine the differences in the test report

2. **Documentation is not generating correctly**:
   - Check that all components are being generated correctly
   - Verify the model and hardware names are correct

3. **Clean old results**:
   - If the collected_results directory is getting too large, use `--clean-old-results`

## Contact

If you have questions or need support with the end-to-end testing framework, please contact the infrastructure team.