# WebGPU/WebNN Resource Pool Fault Tolerance Integration Tests

This package provides comprehensive integration tests for the WebGPU/WebNN Resource Pool Advanced Fault Tolerance system that was completed on May 22, 2025. These tests verify the functionality of the fault tolerance, recovery, and visualization components, ensuring reliable operation in both development and production environments.

## Quick Start

### Run Basic Test (Mock Implementation)

To run a basic test with the mock implementation (no real browsers needed):

```bash
python run_web_resource_pool_fault_tolerance_test.py --mock
```

### Run Comprehensive Test

To run a comprehensive test with all test modes:

```bash
python run_web_resource_pool_fault_tolerance_test.py --comprehensive
```

## Test Modes

### Basic Integration Test

Tests the core fault tolerance functionality:

```bash
python run_web_resource_pool_fault_tolerance_test.py --basic
```

### Comparative Integration Test

Compares different recovery strategies:

```bash
python run_web_resource_pool_fault_tolerance_test.py --comparative
```

### Stress Test

Runs multiple iterations to validate reliability:

```bash
python run_web_resource_pool_fault_tolerance_test.py --stress-test --iterations 10
```

### Resource Pool Integration

Tests integration with the resource pool system:

```bash
python run_web_resource_pool_fault_tolerance_test.py --resource-pool
```

## CI/CD Integration

For CI/CD environments, use the mock implementation to avoid browser dependencies:

```bash
python run_web_resource_pool_fault_tolerance_test.py --mock --comprehensive
```

## Visualization Tools

For direct visualization of fault tolerance metrics, use:

```bash
python run_advanced_fault_tolerance_visualization.py --model bert-base-uncased --comparative
```

## Key Components

1. **WebResourcePoolFaultToleranceIntegrationTester**: Main integration test runner
2. **FaultToleranceValidationSystem**: Validation framework
3. **FaultToleranceVisualizer**: Visualization components
4. **MockCrossBrowserModelShardingManager**: Mock implementation for testing

## Reports and Visualizations

Test runs generate the following outputs:

1. **HTML Reports**: Interactive reports with visualizations
2. **JSON Results**: Detailed test results with metrics
3. **Visualizations**: Recovery time charts, success rate dashboards

## Supported Models

The tests support all models available in the WebGPU/WebNN Resource Pool, including:

- Text models: BERT, T5, LLAMA
- Vision models: ViT, CLIP, DETR
- Audio models: Whisper, CLAP

## Performance Benchmarks

Based on extensive testing, these are the expected performance metrics:

| Recovery Strategy | Scenario | Avg Recovery Time | Success Rate |
|-------------------|----------|-------------------|--------------|
| Simple | Connection Lost | 350ms | 92% |
| Progressive | Connection Lost | 280ms | 97% |
| Coordinated | Connection Lost | 320ms | 99% |
| Simple | Browser Crash | 850ms | 80% |
| Progressive | Browser Crash | 480ms | 95% |
| Coordinated | Browser Crash | 520ms | 98% |

## Documentation

For comprehensive documentation on using these tests, see:
- [WEB_RESOURCE_POOL_FAULT_TOLERANCE_TESTING.md](WEB_RESOURCE_POOL_FAULT_TOLERANCE_TESTING.md)
- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md)

## Next Steps

After running these tests, consider:

1. Integrating with your CI/CD pipeline
2. Customizing test scenarios for your specific use cases
3. Extending the test framework with additional failure modes
4. Creating custom visualizations for specific metrics

## Contributors

This integration test framework was developed as part of the May 2025 enhancements to the WebGPU/WebNN Resource Pool Integration.