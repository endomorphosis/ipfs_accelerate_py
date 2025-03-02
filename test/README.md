# IPFS Accelerate Python - Test Framework

This directory contains the comprehensive testing framework for the IPFS Accelerate Python library, with a focus on validating model functionality, API integrations, and hardware acceleration capabilities.

## Overview

The test framework includes:

1. **Model Tests** - Validation for 127+ HuggingFace model types across different hardware platforms
2. **API Tests** - Integration tests for various AI API providers
3. **Hardware Tests** - Validation of CPU, CUDA, and OpenVINO acceleration
4. **Endpoint Tests** - Tests for local inference endpoints
5. **Performance Tests** - Benchmarking across hardware configurations

## Test Documentation

- [MODEL_TESTING_GUIDE.md](MODEL_TESTING_GUIDE.md) - Complete guide to model testing
- [MODEL_TESTING_README.md](MODEL_TESTING_README.md) - Detailed model test documentation
- [MODEL_TESTING_PROGRESS.md](MODEL_TESTING_PROGRESS.md) - Implementation status and progress
- [MODEL_DEPENDENCIES.md](MODEL_DEPENDENCIES.md) - Model dependency tracking matrix
- [huggingface_test_implementation_plan.md](huggingface_test_implementation_plan.md) - Implementation priorities
- [API_TESTING_README.md](API_TESTING_README.md) - API testing documentation
- [API_IMPLEMENTATION_STATUS.md](API_IMPLEMENTATION_STATUS.md) - API implementation status

## Test Generation Tools

The framework includes several tools for generating new tests:

- [generate_model_tests.py](generate_model_tests.py) - Primary test generator for HuggingFace models
- [simple_model_test_generator.py](simple_model_test_generator.py) - Enhanced generator with dependency tracking
- [generate_missing_test_files.py](generate_missing_test_files.py) - Older test generator
- [generate_comprehensive_tests.py](generate_comprehensive_tests.py) - Advanced test generator (WIP)

## Running Tests

### Model Tests

```bash
# Run a specific model test
python skills/test_hf_bert.py

# Run multiple tests in parallel
python run_skills_tests.py --models bert,roberta,gpt2

# Run tests for a specific category
python run_skills_tests.py --category language
```

### API Tests

```bash
# Run all API tests
python check_api_implementation.py 

# Test a specific API
python test_single_api.py [api_name]
```

### Hardware Tests

```bash
# Test hardware backends
python test_hardware_backend.py
```

## Current Test Coverage (March 2025)

- **HuggingFace Models**: 137 of 299 models (45.8%)
- **API Backends**: 11 API types with comprehensive testing
- **Hardware Support**: CPU (100%), CUDA (93.8%), OpenVINO (89.6%)
- **Test Quality**: Dual-method testing, dependency tracking, remote code support

## Generating New Tests

To generate tests for missing models:

```bash
# Using the primary generator
python generate_model_tests.py --list-only
python generate_model_tests.py --models layoutlmv2 nougat swinv2 vit_mae
python generate_model_tests.py --category vision --limit 5

# Using the enhanced generator with dependency tracking
python simple_model_test_generator.py --batch  # Generate batch of tests with dependency tracking
python simple_model_test_generator.py --model llama-3-70b-instruct --task text-generation  # Specific model
```

## Recent Improvements

Recent improvements to the testing framework include:

1. **Dual-Method Testing** - Tests now cover both pipeline() and from_pretrained() methods
2. **Dependency Tracking** - Automatic detection and documentation of model dependencies
3. **Remote Code Support** - Proper handling of models requiring custom code execution
4. **Enhanced Mock Objects** - Sophisticated mock implementations for various dependencies
5. **Batch Testing Support** - All generated tests now include batch processing validation
6. **Advanced Error Handling** - Improved degradation mechanisms for handling missing dependencies
7. **Performance Monitoring** - Added inference time and memory usage tracking
8. **Documentation Updates** - Expanded documentation with dependency matrix and progress tracking

## Contributing

When adding new tests:

1. Use the `generate_model_tests.py` script when possible
2. Follow the established test structure
3. Include both single and batch testing
4. Test on all available hardware platforms
5. Update the documentation to reflect new tests

## License

This test framework follows the same license as the main IPFS Accelerate Python library.