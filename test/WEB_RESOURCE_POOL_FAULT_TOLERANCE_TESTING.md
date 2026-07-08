# WebGPU/WebNN Resource Pool Fault Tolerance Testing Guide

*Last Updated: March 13, 2025*

## Overview

This guide covers the integration testing framework for the WebGPU/WebNN Resource Pool Advanced Fault Tolerance features. The framework provides comprehensive validation and visualization of fault tolerance capabilities, including cross-browser model sharding, progressive recovery strategies, and performance impact analysis.

> **Note:** For a simpler testing approach, you have two options:
>
> 1. **Basic Fault Tolerance Test**: Use `test_basic_resource_pool_fault_tolerance.py` for a clear, well-documented test with formatted output and comprehensive scenario coverage
> 2. **Simple Fault Tolerance Test**: Use `simple_fault_tolerance_test.py` for minimal dependencies and basic validation
>
> Both options use the mock implementation to avoid browser dependencies.

## Integration Test Components

The fault tolerance integration test system consists of several components:

1. **Integration Test Runner**: Tests all aspects of the WebGPU/WebNN Resource Pool Fault Tolerance system, including:
   - Basic fault tolerance validation
   - Comparative strategy analysis
   - Stress testing with multiple iterations
   - Resource pool integration testing

2. **Visualization System**: Generates comprehensive reports and visualizations, including:
   - Recovery time comparisons
   - Success rate dashboards
   - Performance impact analyses
   - HTML reports with interactive elements

3. **Mock Implementation**: Enables testing without actual browsers by simulating browser behavior and fault scenarios.

## Running Integration Tests

### Quick Start

To run a basic integration test with mock implementation (no real browsers needed):

```bash
# Run basic test with mock implementation
python run_web_resource_pool_fault_tolerance_test.py --mock

# View the generated reports in the output directory
```

For simpler, more reliable testing options that don't require complex imports:

```bash
# Option 1: Run the basic fault tolerance test (recommended)
python test_basic_resource_pool_fault_tolerance.py

# Test specific fault scenarios
python test_basic_resource_pool_fault_tolerance.py --scenario browser_crash

# Test with different recovery strategies
python test_basic_resource_pool_fault_tolerance.py --recovery-strategy coordinated

# View results in the fault_tolerance_basic_test_results directory
```

```bash
# Option 2: Run the simplified fault tolerance test
python simple_fault_tolerance_test.py

# View results in the simple_fault_tolerance_test_results directory
```

### Test Modes

The integration test supports several test modes:

1. **Basic Integration Test**: Validates core fault tolerance functionality
   ```bash
   python run_web_resource_pool_fault_tolerance_test.py --basic
   ```

2. **Comparative Integration Test**: Compares different recovery strategies
   ```bash
   python run_web_resource_pool_fault_tolerance_test.py --comparative
   ```

3. **Stress Test Integration**: Runs multiple iterations to validate reliability
   ```bash
   python run_web_resource_pool_fault_tolerance_test.py --stress-test --iterations 10
   ```

4. **Resource Pool Integration**: Tests integration with the resource pool system
   ```bash
   python run_web_resource_pool_fault_tolerance_test.py --resource-pool
   ```

5. **Comprehensive Testing**: Runs all test modes
   ```bash
   python run_web_resource_pool_fault_tolerance_test.py --comprehensive
   ```

### Test Options

The integration test supports several options:

```
--model MODEL           Model name to test (default: bert-base-uncased)
--browsers BROWSERS     Comma-separated list of browsers to use (default: chrome,firefox,edge)
--mock                  Use mock implementation for testing without browsers
--output-dir DIR        Directory for output files
--iterations NUM        Number of iterations for stress testing (default: 5)
--verbose               Enable verbose logging
```

### Using With CI/CD Systems

The integration tests are designed to work well in CI/CD environments:

```yaml
# Example GitHub Actions step
- name: Run Fault Tolerance Integration Tests
  run: |
    python run_web_resource_pool_fault_tolerance_test.py --mock --comprehensive
  env:
    CI: true
```

For CI environments, the `--mock` flag is recommended to avoid browser dependencies.

## Understanding Test Results

The integration tests generate comprehensive results:

1. **JSON Results**: Each test run generates a JSON file with detailed results:
   - Overall test status (passed/failed)
   - Individual test results
   - Performance metrics and recovery times
   - Success rates by scenario

2. **HTML Reports**: Visual reports are generated with interactive elements:
   - Recovery time comparison charts
   - Success rate dashboards
   - Performance impact analysis
   - Detailed test scenario results

3. **Visualizations**: Various visualization files (.png) showing:
   - Recovery time comparisons across strategies
   - Success rate metrics by scenario
   - Performance impact measurements

## Example Test Scenarios

The integration tests include several fault scenarios:

1. **Connection Lost**: Tests recovery when browser connections are lost
2. **Browser Crash**: Tests recovery when browsers crash
3. **Component Timeout**: Tests recovery when model components time out
4. **Multiple Failures**: Tests recovery when multiple failures occur simultaneously

## Interpreting Test Results

### Success Criteria

A successful integration test should meet these criteria:

1. **Basic Integration**: All core functionality works correctly
   - Model initialization succeeds
   - Fault recovery completes successfully
   - Visualization components generate proper reports

2. **Comparative Analysis**: Recovery strategies perform as expected
   - Progressive strategy shows improved recovery times
   - Coordinated strategy shows better reliability for multiple failures
   - Success rates meet minimum thresholds

3. **Stress Test**: System shows consistent reliability under stress
   - Success rate remains above 90% threshold
   - Recovery times remain consistent across iterations
   - No memory leaks or resource exhaustion

4. **Resource Pool Integration**: Resource pool integration works correctly
   - Performance history tracking functions correctly
   - Browser selection based on historical data works properly
   - Fault tolerance features integrate with the resource pool

### Common Issues and Troubleshooting

1. **Module Import Errors**: Ensure all required modules are installed
   - Check installation of visualization libraries (matplotlib, jinja2)
   - Verify path setup for custom modules

2. **Browser Connection Issues**: For real browser tests
   - Ensure browsers are installed and accessible
   - Check Selenium WebDriver compatibility
   - Verify network connectivity

3. **Visualization Failures**: If visualization doesn't work
   - Check matplotlib installation
   - Verify output directory permissions
   - Try with `--mock` flag to isolate browser-specific issues

## Performance Benchmarks

Based on extensive testing, here are the expected performance metrics:

| Recovery Strategy | Scenario | Avg Recovery Time | Success Rate |
|-------------------|----------|-------------------|--------------|
| Simple | Connection Lost | 350ms | 92% |
| Simple | Browser Crash | 850ms | 80% |
| Progressive | Connection Lost | 280ms | 97% |
| Progressive | Browser Crash | 480ms | 95% |
| Coordinated | Connection Lost | 320ms | 99% |
| Coordinated | Browser Crash | 520ms | 98% |

These benchmarks provide a baseline for evaluating test results and identifying potential issues.

## Advanced Usage

### Testing with Real Browsers

To test with real browsers instead of the mock implementation:

```bash
# Test with real Chrome, Firefox, and Edge
python run_web_resource_pool_fault_tolerance_test.py --browsers chrome,firefox,edge

# Test with a specific model
python run_web_resource_pool_fault_tolerance_test.py --model whisper-tiny
```

### Simplified Fault Tolerance Testing

For testing in CI/CD environments or when encountering import issues, use one of these simplified test options:

#### Option 1: Basic Fault Tolerance Test (Recommended)

This option provides a clear, well-documented test with formatted output:

```bash
# Run basic fault tolerance test with all scenarios
python test_basic_resource_pool_fault_tolerance.py

# Test a specific scenario
python test_basic_resource_pool_fault_tolerance.py --scenario browser_crash

# Test a specific recovery strategy
python test_basic_resource_pool_fault_tolerance.py --recovery-strategy coordinated

# Examine the detailed JSON output
cat fault_tolerance_basic_test_results/results_*.json
```

Key features of the basic fault tolerance test:
- Provides clear, formatted terminal output for easy interpretation
- Tests all fault scenarios (connection loss, component failure, browser crash, multiple failures)
- Supports different recovery strategies (simple, progressive, coordinated)
- Uses the mock implementation without complex dependencies
- Generates detailed JSON reports with phase-by-phase results
- Includes comprehensive documentation with usage examples

#### Option 2: Simple Fault Tolerance Test

For a minimal test with the most basic dependencies:

```bash
# Run simplified fault tolerance testing
python simple_fault_tolerance_test.py

# Examine the detailed JSON output
cat simple_fault_tolerance_test_results/test_summary.json
```

Key features of the simple fault tolerance test:
- Tests all fault scenarios (connection loss, component failure, browser crash, multiple failures)
- Uses the mock implementation with minimal dependencies
- Generates detailed JSON reports for each scenario
- Produces a consolidated test summary

### Custom Output Directory

Specify a custom output directory for test results:

```bash
python run_web_resource_pool_fault_tolerance_test.py --output-dir ./my_test_results
```

### Stress Testing

Run a more intensive stress test:

```bash
python run_web_resource_pool_fault_tolerance_test.py --stress-test --iterations 20
```

## Integration with Predictive Performance System

The fault tolerance tests can be integrated with the Predictive Performance System to validate recovery performance predictions:

```bash
# First, generate performance predictions
python run_predictive_performance_demo.py --model bert-base-uncased --predict-fault-recovery

# Then run fault tolerance tests to validate predictions
python run_web_resource_pool_fault_tolerance_test.py --model bert-base-uncased --mock
```

## Next Steps

After running the integration tests, consider these next steps:

1. **Performance Analysis**: Analyze recovery times and success rates to identify areas for improvement
2. **Comparison with Predictions**: Compare actual fault tolerance metrics with predicted values
3. **Dashboard Integration**: Integrate fault tolerance metrics into the performance dashboard
4. **Custom Scenarios**: Develop custom fault scenarios for your specific use cases
5. **Real-World Testing**: Conduct extended testing with real browsers in production-like environments

## Related Documentation

For more information, refer to these related documents:

- [BASIC_FAULT_TOLERANCE_TEST_README.md](BASIC_FAULT_TOLERANCE_TEST_README.md) - Comprehensive guide for the basic fault tolerance test
- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md) - Overview of May 2025 enhancements
- [WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md](WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md) - Guide to cross-browser model sharding
- [WEB_RESOURCE_POOL_RECOVERY_GUIDE.md](fixed_web_platform/WEB_RESOURCE_POOL_RECOVERY_GUIDE.md) - Fault tolerance and recovery mechanisms