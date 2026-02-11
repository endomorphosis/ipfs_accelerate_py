# Refactored Tests Suite

This directory contains the refactored test suite for the IPFS Accelerate Python project. The test suite has been reorganized to improve maintainability, reduce duplication, and increase testing efficiency.

## Directory Structure

```
refactored_tests/
├── common/             # Common test utilities and base classes
│   ├── base_test.py    # Base test class for all tests
│   ├── model_test.py   # Base class for model tests
│   ├── hardware_test.py # Base class for hardware tests
│   ├── api_test.py     # Base class for API tests
│   ├── browser_test.py # Base class for browser tests
│   ├── test_fixtures.py # Common test fixtures
│   ├── test_assertions.py # Common test assertions
│   ├── test_mocks.py   # Common test mocks
│   └── hardware_detection.py # Hardware detection utilities
├── unit/               # Unit tests for components
├── integration/        # Integration tests between components
├── models/             # ML model tests
│   ├── text/           # Text model tests
│   ├── vision/         # Vision model tests
│   └── audio/          # Audio model tests
├── hardware/           # Hardware compatibility tests
│   ├── webgpu/         # WebGPU tests
│   ├── webnn/          # WebNN tests
│   └── platform/       # Platform-specific tests
├── browser/            # Browser-specific tests
├── api/                # API tests
└── e2e/                # End-to-end tests
```

## Base Test Classes

The refactored test suite uses a hierarchy of base test classes to provide common functionality:

- `BaseTest`: Base class for all tests, provides logging, timing, and utility methods
- `ModelTest`: Base class for ML model tests, adds model loading/unloading and verification
- `HardwareTest`: Base class for hardware compatibility tests, adds hardware detection
- `APITest`: Base class for API tests, adds API client setup and request handling
- `BrowserTest`: Base class for browser tests, adds browser setup and control

## Running Tests

To run the refactored tests:

```bash
python test/run_refactored_tests.py
```

To run specific test types:

```bash
# Run model tests only
python test/run_refactored_tests.py test/refactored_tests/models

# Run specific model test
python test/run_refactored_tests.py test/refactored_tests/models/text/test_bert_model.py

# Run with verbose output
python test/run_refactored_tests.py -v
```

## Migrating Tests

For information on migrating existing tests to the refactored structure, see the [REFACTORED_TEST_MIGRATION_GUIDE.md](REFACTORED_TEST_MIGRATION_GUIDE.md).

## Documentation

- [COMPREHENSIVE_TEST_REFACTORING_PLAN.md](COMPREHENSIVE_TEST_REFACTORING_PLAN.md): Complete refactoring strategy with timeline and implementation plan
- [README_TEST_REFACTORING_IMPLEMENTATION.md](README_TEST_REFACTORING_IMPLEMENTATION.md): Detailed implementation plan for Phase 1 of the refactoring
- [TEST_REFACTORING_SUMMARY.md](TEST_REFACTORING_SUMMARY.md): Summary of the completed test refactoring analysis