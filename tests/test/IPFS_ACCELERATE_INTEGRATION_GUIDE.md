# IPFS Accelerate Python Integration Guide

## Introduction

The IPFS Accelerate Python package provides functionality for accelerating IPFS operations in Python applications. This guide explains how to integrate the package into your projects and use its core capabilities.

## Installation

### Regular Installation
Install the package using pip:

```bash
pip install ipfs_accelerate_py
```

### Developer Installation (for HuggingFace Model Testing)
For development and testing with the 300+ HuggingFace model classes, install directly from the repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/ipfs_accelerate_py.git

# Install in development mode
cd ipfs_accelerate_py
pip install -e .
```

This ensures that the `api_backends` module is available, which is required for comprehensive model testing.

## Basic Usage

### Importing the Package

```python
import ipfs_accelerate_py
```

### Configuration

The package uses a configuration system that looks for a `config.toml` file. You can create this file in your working directory:

```toml
# Basic configuration for IPFS Accelerate Python
[general]
debug = true
log_level = "INFO"

[cache]
enabled = true
max_size_mb = 1000
path = "./cache"

[endpoints]
default = "local"

[endpoints.local]
host = "localhost"
port = 8000
```

### Core Components

The package provides several core components:

1. **Config** - Configuration management
2. **Backends** - Backend container operations
3. **ipfs_accelerate** - Core functionality module
4. **load_checkpoint_and_dispatch** - Loading and dispatching functionality

### Example: Accessing Core Components

```python
import ipfs_accelerate_py

# Access the config class
config_class = ipfs_accelerate_py.config

# Access the backends class
backends_class = ipfs_accelerate_py.backends

# Access the dispatch function
dispatch_func = ipfs_accelerate_py.load_checkpoint_and_dispatch
```

## Advanced: HuggingFace Model Testing

For comprehensive testing with 300+ HuggingFace model classes, additional setup is required:

1. Use the developer installation method described above
2. Ensure the `api_backends` module is properly configured
3. Run comprehensive tests with the appropriate model flags

### API Backends Configuration

The `api_backends` module requires API credentials for external services. Create a `api_credentials.json` file:

```json
{
  "openai_api_key": "your_openai_key",
  "claude_api_key": "your_claude_key",
  "gemini_api_key": "your_gemini_key"
}
```

### Running Comprehensive Model Tests

Execute the comprehensive test script with model flags:

```bash
# Test a specific model
python test/test_ipfs_accelerate.py --models bert-base-uncased

# Test multiple models
python test/test_ipfs_accelerate.py --models bert-base-uncased,t5-small,vit-base

# Test all models (requires significant resources)
python test/test_ipfs_accelerate.py --all-models
```

These tests run at the end of each development phase to ensure compatibility across model classes and hardware platforms.

## Performance Considerations

The package is designed for high performance:

- **Fast Loading:** Modules load in under 4ms
- **Efficient Operations:** Basic operations execute in negligible time
- **Good Parallelism:** The package handles parallel loading well

For best performance in multi-threaded applications, consider pre-loading the modules.

## Testing

You can test the package's functionality and performance using the provided test scripts:

```bash
# Run minimal functionality test
python test/test_ipfs_accelerate_minimal.py

# Run performance benchmark
python test/benchmark_ipfs_acceleration.py

# Check compatibility between installed package and test framework
python test/compatibility_check.py
```

## Troubleshooting

### Missing Configuration File

If you see an error like:

```
no config file found
make sure config.toml is in the working directory
or specify path using --config
```

Create a `config.toml` file in your working directory as shown in the Configuration section.

### Import Errors with api_backends

If you encounter errors like:

```
ModuleNotFoundError: No module named 'ipfs_accelerate_py.api_backends'
```

This indicates you're trying to run comprehensive tests with the regular package installation. Use the developer installation method to access the `api_backends` module.

### Hardware Platform Issues

If model tests fail with specific hardware platforms, ensure:

1. The hardware is properly detected and configured
2. Required drivers are installed
3. The model is compatible with the hardware platform

Consult the hardware compatibility matrix in `CLAUDE.md` for supported model-hardware combinations.

## Additional Resources

- For more detailed information, see the [IPFS Acceleration Benchmark Report](IPFS_ACCELERATION_BENCHMARK_REPORT.md)
- For hardware compatibility details, see [CLAUDE.md](CLAUDE.md)

## Contributing

Contributions to the IPFS Accelerate Python package are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

The IPFS Accelerate Python package is licensed under [MIT License](LICENSE).