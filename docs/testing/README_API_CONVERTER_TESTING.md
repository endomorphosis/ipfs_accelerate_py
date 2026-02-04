# API Backend Converter Testing Infrastructure

This documentation describes the testing infrastructure for the API Backend Converter, which converts Python API backends to TypeScript implementations.

## Overview

The testing infrastructure consists of:

1. **Unit Tests** - Test individual components of the converter
2. **Integration Tests** - Test end-to-end conversion of backends
3. **TypeScript Validation** - Verify the syntax of generated TypeScript code
4. **Test Generation** - Generate Jest tests for the TypeScript implementations

## Test Files

- `test_api_backend_converter.py` - Unit tests for the converter
- `test_api_backend_converter_integration.py` - Integration tests with real files
- `run_api_converter_tests.py` - Script to run all tests
- `generate_api_backend_test.py` - Generate Jest tests for TypeScript backends

## Running Tests

### Running All Tests

```bash
# Run all tests
python run_api_converter_tests.py

# Run tests with verbose output
python run_api_converter_tests.py -v

# Run tests matching a specific pattern
python run_api_converter_tests.py -p TestAPIBackendConverter
```

### Running Unit Tests

```bash
# Run unit tests directly
python -m unittest test_api_backend_converter.py

# Run specific test case
python -m unittest test_api_backend_converter.TestAPIBackendConverter.test_parse_python_file
```

### Running Integration Tests

```bash
# Run integration tests directly
python -m unittest test_api_backend_converter_integration.py

# Run only TypeScript validation tests
python -m unittest test_api_backend_converter_integration.TestValidTypeScriptOutput
```

## Generating TypeScript Tests

After converting Python backends to TypeScript, you can automatically generate Jest tests:

```bash
# Generate tests for a specific backend
python generate_api_backend_test.py --backend ollama

# Generate tests for all converted backends
python generate_api_backend_test.py --all

# Specify custom directories
python generate_api_backend_test.py --all --ts-dir ./path/to/typescript --test-dir ./path/to/tests
```

## Test Coverage

The test suite covers the following aspects of the converter:

### Unit Tests:

1. **Initialization**
   - Proper setup of converter instance
   - Correct parsing of backend names
   - Proper output file path generation

2. **Type Conversion**
   - Python to TypeScript type mapping
   - Complex type handling (generics, optionals, etc.)
   - Default value conversion

3. **Code Analysis**
   - Parsing Python files with AST
   - Handling syntax errors
   - Extracting method signatures, parameters, and return types

4. **TypeScript Generation**
   - Generate proper class structure
   - Generate method implementations
   - Generate type definitions
   - Backend-specific customizations

### Integration Tests:

1. **End-to-end Conversion**
   - Converting real backend files
   - Verifying the output file structure
   - Checking the content of generated files

2. **TypeScript Validation**
   - Validating TypeScript syntax
   - Ensuring correct imports
   - Checking class and method signatures

3. **CLI Operation**
   - Command-line execution
   - Argument parsing
   - Processing multiple backends

## Adding New Tests

### Adding Unit Tests

To add new unit tests, add test methods to the `TestAPIBackendConverter` class in `test_api_backend_converter.py`:

```python
def test_new_feature(self):
    """Test a new feature of the converter"""
    # Test setup
    result = self.converter.some_method()
    # Assertions
    self.assertTrue(result)
```

### Adding Integration Tests

To add new integration tests, add test methods to the relevant classes in `test_api_backend_converter_integration.py`:

```python
def test_new_integration_scenario(self):
    """Test a new integration scenario"""
    # Test setup
    # ...
    # Run the converter
    # ...
    # Verify the results
    self.assertTrue(os.path.exists(expected_file))
```

## Test Data

The tests use the following types of test data:

1. **Sample Python Files** - Created in-memory during unit tests
2. **Real Backend Files** - Used from the project during integration tests
3. **Generated TypeScript Files** - Created during the tests and validated

## Continuous Integration

To run the tests as part of CI/CD:

```bash
# Run all tests and fail if any test fails
python run_api_converter_tests.py || exit 1

# Generate TypeScript tests for all backends
python generate_api_backend_test.py --all
```

## Future Improvements

- Add code coverage reporting
- Expand tests for edge cases and error handling
- Add property-based testing for type conversions
- Add performance benchmarks
- Implement E2E tests with actual TypeScript compilation
- Add TypeScript linting tests