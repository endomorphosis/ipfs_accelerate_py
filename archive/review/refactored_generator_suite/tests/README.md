# Refactored Generator Suite Tests

This directory contains integration and unit tests for the refactored generator suite. The tests are designed to validate the functionality of the entire system and its individual components.

## Test Structure

The test suite is organized as follows:

- `test_integration.py` - End-to-end integration tests for the entire generation pipeline
- `test_model_selection.py` - Tests for the model selection system
- `test_templates.py` - Tests for the template system
- `test_hardware.py` - Tests for the hardware detection system
- (Additional test files for other components)

## Running Tests

Tests can be run using the provided `run_all_tests.py` script or via the Makefile.

### Using the run_all_tests.py script

```bash
# Run all tests
./run_all_tests.py

# Run tests with increased verbosity
./run_all_tests.py -v

# Run only integration tests
./run_all_tests.py --integration

# Run only unit tests
./run_all_tests.py --unit

# Run tests matching a specific pattern
./run_all_tests.py --pattern hardware
```

### Using the Makefile

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with specific arguments
make test ARGS="-v --pattern=hardware"
```

## Test Coverage

The tests aim to provide comprehensive coverage of the refactored generator suite:

1. **Integration Tests**:
   - Complete generation pipeline
   - Hardware-aware test generation
   - Batch generation
   - Syntax validation and fixing
   - Mock environment support

2. **Model Selection Tests**:
   - Model registry functionality
   - Selection filters (task, hardware, size, framework)
   - Model architecture mapping
   - Selection with complex criteria

3. **Template Tests**:
   - Base template functionality
   - Architecture-specific templates
   - Template rendering
   - Conditional sections
   - Template extension

4. **Hardware Detection Tests**:
   - Individual hardware detectors
   - Hardware detection manager
   - Exception handling
   - Device selection logic

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new test file named `test_<component>.py`
2. Extend `unittest.TestCase` for each test class
3. Use descriptive test method names: `test_<functionality>`
4. Add appropriate docstrings
5. Organize tests into logical classes
6. Use mocks for external dependencies
7. Add your tests to this README

## Mock Usage

The tests use mocks to simulate external dependencies and hardware. This allows tests to run in any environment without requiring specific hardware or installations.

Example mock usage:

```python
with patch("torch.cuda.is_available", return_value=True), \
     patch("torch.version.cuda", "11.7"):
    result = detector.detect()
    self.assertTrue(result["available"])
```

## Continuous Integration

These tests are designed to run in the CI/CD pipeline to ensure code quality. The test runner uses appropriate exit codes to indicate test success or failure.