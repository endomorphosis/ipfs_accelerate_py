# IPFS Accelerate Test Framework

## Overview

The IPFS Accelerate Test Framework provides a comprehensive, structured approach to testing across different types of models, hardware platforms, and APIs. The framework is designed to enable efficient test discovery, consistent test patterns, and a clear migration path from legacy tests.

## Key Features

- **Template-based test generation** for consistent test structure
- **Hierarchical directory organization** for clear test categorization
- **Hardware detection and platform-specific testing** for cross-platform compatibility
- **Distributed testing support** for efficient test execution
- **Comprehensive test reporting** with visualizations
- **CI/CD integration** for automated testing
- **Migration tools** for transitioning from legacy tests

## Directory Structure

```
test/
├── models/                  # Model-specific tests
│   ├── text/                # Text model tests
│   ├── vision/              # Vision model tests
│   ├── audio/               # Audio model tests
│   └── multimodal/          # Multimodal model tests
├── hardware/                # Hardware-specific tests
│   ├── webgpu/              # WebGPU tests
│   ├── webnn/               # WebNN tests
│   ├── cuda/                # CUDA tests
│   └── ...                  # Other hardware platforms
├── api/                     # API-specific tests
│   ├── openai/              # OpenAI API tests
│   ├── hf_tei/              # HuggingFace TEI tests
│   └── ...                  # Other API tests
├── integration/             # Integration tests
├── common/                  # Shared utilities and helpers
│   ├── fixtures.py          # Test fixtures
│   ├── hardware_detection.py # Hardware detection utilities
│   └── model_helpers.py     # Model loading utilities
├── template_system/         # Template-based test generation
│   ├── templates/           # Test templates
│   └── generate_test.py     # Test generator script
├── distributed_testing/     # Distributed testing framework
│   ├── coordinator.py       # Test coordinator
│   └── worker.py            # Test worker
├── docs/                    # Documentation
├── visualizations/          # Visualization output
├── conftest.py              # Pytest configuration
├── pytest.ini               # Pytest settings
├── run.py                   # Test runner
└── run.sh                   # CI/CD entry point
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (install with `pip install -r requirements_test.txt`)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-organization/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install dependencies
pip install -r test/requirements_test.txt
```

### Running Tests

To run tests, use the `run.py` script:

```bash
# Run all tests
python test/run.py

# Run model tests
python test/run.py --test-type model

# Run specific model tests
python test/run.py --test-type model --model bert

# Run hardware tests
python test/run.py --test-type hardware --platform webgpu

# Run API tests
python test/run.py --test-type api --api openai

# Run tests with specific markers
python test/run.py --markers "slow or webgpu"

# Run tests in distributed mode
python test/run.py --distributed --worker-count 4
```

For CI/CD environments, use the `run.sh` script:

```bash
# Set up environment and run all tests
./test/run.sh --setup-env --install-deps

# Run tests with reports
./test/run.sh --generate-reports

# Run distributed tests
./test/run.sh --distributed --worker-count 4
```

### Generating Tests

To generate new tests, use the `generate_test.py` script:

```bash
# Generate a model test
python test/template_system/generate_test.py model --model-name bert-base-uncased --model-type text

# Generate a hardware test
python test/template_system/generate_test.py hardware --hardware-platform webgpu --test-name matmul_performance

# Generate an API test
python test/template_system/generate_test.py api --api-name openai --test-name chat_completion
```

To generate example tests across all categories:

```bash
python test/generate_example_tests.py --output-dir test
```

### Migrating Legacy Tests

To migrate legacy tests to the new structure:

```bash
# Analyze existing tests
python test/migrate_tests.py --source-dir old_tests --analyze-only

# Migrate tests
python test/migrate_tests.py --source-dir old_tests --output-dir test

# Track migration progress
python test/track_migration_progress.py --analysis-report migration_analysis.json --migrated-dir test
```

## Documentation

For detailed documentation, see the following guides:

- [Test Framework Guide](docs/TEST_FRAMEWORK_GUIDE.md): Comprehensive guide to the test framework
- [Distributed Testing Framework](docs/DISTRIBUTED_TESTING_FRAMEWORK.md): Guide to distributed testing
- [Migration Guide](docs/MIGRATION_GUIDE.md): Guide for migrating legacy tests
- [Template System Guide](docs/TEMPLATE_SYSTEM_GUIDE.md): Guide to the template system

## Components

### Test Templates

The framework uses templates to generate consistent test files:

- `BaseTemplate`: Base class for all templates
- `ModelTestTemplate`: Template for model tests
- `HardwareTestTemplate`: Template for hardware tests
- `APITestTemplate`: Template for API tests

### Utilities

Common utilities provide shared functionality:

- `hardware_detection.py`: Detect available hardware
- `model_helpers.py`: Load models and provide sample inputs
- `fixtures.py`: Common test fixtures

### Distributed Testing

The distributed testing framework enables efficient test execution across multiple workers:

- `coordinator.py`: Manages test distribution and worker coordination
- High availability mode with leader election
- Hardware-aware test routing
- Performance visualization

### CI/CD Integration

The framework integrates with CI/CD systems:

- GitHub Actions workflow support
- Jenkins integration
- GitLab CI support
- Customizable reporting

## Best Practices

1. **Use the template system** for creating new tests
2. **Follow the directory structure** to keep tests organized
3. **Use fixtures** for common setup and teardown
4. **Use markers** to categorize tests
5. **Include hardware checks** for platform-specific tests
6. **Write modular tests** that focus on specific functionality
7. **Use distributed testing** for efficient test execution

## Contributing

When contributing to the test framework:

1. Follow the existing structure and naming conventions
2. Use templates to generate new tests
3. Add appropriate markers and fixtures
4. Update documentation for new features
5. Ensure tests are modular and focused

## License

This project is licensed under the MIT License - see the LICENSE file for details.
