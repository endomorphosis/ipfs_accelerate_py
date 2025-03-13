# Basic WebGPU/WebNN Resource Pool Fault Tolerance Test

This tool provides a simple, standalone test case for the WebGPU/WebNN Resource Pool fault tolerance system. It's designed to be easy to run and understand, making it ideal for CI/CD environments or quick validation tests.

## Features

- Tests basic fault tolerance capabilities with minimal dependencies
- Supports multiple fault scenarios (connection loss, component failure, browser crash, multiple failures)
- Tests different recovery strategies (simple, progressive, coordinated)
- Provides clear, formatted output for easy interpretation
- Saves detailed JSON reports for further analysis

## Usage

### Basic Usage

```bash
# Run with default settings (all scenarios, progressive recovery)
python test_basic_resource_pool_fault_tolerance.py

# Test a specific scenario
python test_basic_resource_pool_fault_tolerance.py --scenario browser_crash

# Test with a specific model
python test_basic_resource_pool_fault_tolerance.py --model vit-base-patch16-224

# Test a specific recovery strategy
python test_basic_resource_pool_fault_tolerance.py --recovery-strategy coordinated

# Specify a custom output directory
python test_basic_resource_pool_fault_tolerance.py --output-dir ./my_test_results
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name to test | bert-base-uncased |
| `--scenario` | Specific scenario to test | (All scenarios) |
| `--recovery-strategy` | Recovery strategy to use | progressive |
| `--output-dir` | Directory for output files | ./fault_tolerance_basic_test_results |

### Available Scenarios

| Scenario | Description |
|----------|-------------|
| `connection_lost` | Tests recovery from browser connection loss |
| `component_failure` | Tests recovery from model component failure |
| `browser_crash` | Tests recovery from complete browser crash |
| `multiple_failures` | Tests recovery from multiple simultaneous failures |

### Recovery Strategies

| Strategy | Description |
|----------|-------------|
| `simple` | Basic recovery approach that attempts to recover each component independently |
| `progressive` | Two-phase approach that first reconnects browsers, then recovers components |
| `coordinated` | Ensures critical components recover together as a group |

## Output

The test generates two types of output:

1. **Terminal Output**: A formatted display of test results including:
   - Overall test status
   - Results for each scenario
   - Phase-by-phase breakdown with timing information
   - Summary statistics

2. **JSON Report**: A detailed JSON file containing:
   - Complete test configuration
   - Detailed results for each scenario
   - Timing information for each test phase
   - Diagnostic information about the test environment
   - Full error details if any occurred

## Examples

### Example Terminal Output

```
================================================================================
FAULT TOLERANCE TEST RESULTS - bert-base-uncased
Recovery Strategy: progressive
Timestamp: 20250313_002247
================================================================================

SCENARIO: browser_crash - ✅ SUCCESS
Description: Complete browser crash scenario

Test Phases:
  ✅ initialize           - 0.26s    - Initialization succeeded
  ✅ initial_inference    - 0.00s    - Initial inference successful
  ✅ fault_injection      - 0.00s    - Fault injection success
  ✅ fault_inference      - 0.00s    - Inference with fault succeeded (fault tolerance working)
  ✅ recovery             - 0.00s    - Recovery succeeded
  ✅ recovery_inference   - 0.00s    - Post-recovery inference successful

================================================================================
SUMMARY:
Scenarios: 1/1 successful
Success Rate: 100.0%
Overall Result: ✅ SUCCESS
================================================================================
```

### Example JSON Report

The JSON report is saved to the specified output directory with a timestamp in the filename. It contains detailed information about the test execution, including:

- Test configuration (model, browsers, recovery strategy)
- Results for each scenario
- Phase-by-phase breakdown with timing information
- Diagnostic information about model components and browsers
- Error details if any occurred

## Integration with CI/CD

This test is designed to work well in CI/CD environments:

1. It returns exit code 0 for success and 1 for failure, making it easy to integrate with CI/CD pipelines
2. The test has minimal dependencies, using only the mock implementation to simulate browser behavior
3. Output is stored as JSON, making it easy to parse and analyze in automated workflows
4. The test can be configured to run specific scenarios, making it suitable for different testing stages

Example CI integration:

```yaml
# Example GitHub Actions workflow
test-fault-tolerance:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run fault tolerance tests
      run: |
        python test_basic_resource_pool_fault_tolerance.py --output-dir ./test-results
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: fault-tolerance-results
        path: ./test-results
```

## Extending the Test

The test can be extended in several ways:

1. Add new scenarios by extending the `fault_scenarios` dictionary
2. Add new recovery strategies by modifying the MockCrossBrowserModelShardingManager
3. Add new test phases to the `test_scenario` method
4. Enhance the output formatting in the `format_results_for_display` function

## Troubleshooting

If you encounter issues:

1. Check that the `fixed_mock_cross_browser_sharding.py` file is available
2. Ensure you have the correct Python version (3.7+)
3. If the mock implementation fails to import, check the error message for details
4. Look for JSON reports in the output directory for detailed error information