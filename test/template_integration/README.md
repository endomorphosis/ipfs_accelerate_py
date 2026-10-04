# Template Integration System

A comprehensive solution for generating standardized test files for machine learning models in the IPFS Accelerate Python framework.

## Status: COMPLETED

The Template Integration project is now complete with:
- 6 template types fully implemented and validated
- A comprehensive test generator that covers all HuggingFace Transformers classes

## Components

### Template System

The template system provides standardized test templates for different types of models:

1. **Vision Template**: For image classification, object detection, etc. (ViT, DeiT, etc.)
2. **Encoder-Only Template**: For text classification, token classification, etc. (BERT, RoBERTa, etc.)
3. **Decoder-Only Template**: For text generation, causal language modeling, etc. (GPT2, LLaMA, etc.)
4. **Encoder-Decoder Template**: For translation, summarization, etc. (T5, BART, etc.)
5. **Speech/Audio Template**: For speech recognition, audio classification, etc. (Whisper, Wav2Vec2, etc.)
6. **Multimodal Template**: For image-text tasks, etc. (CLIP, BLIP, FLAVA, etc.)

Each template provides comprehensive test coverage:
- Model loading tests
- Pipeline API tests
- Direct inference tests
- Hardware compatibility tests (CPU, CUDA, MPS)
- OpenVINO integration tests

### Comprehensive Test Generator

The `comprehensive_test_generator.py` script provides a single entry point for generating tests for all HuggingFace Transformers classes:

- Automatically discovers all model classes with `from_pretrained` support
- Categorizes models by architecture type
- Maps models to appropriate pipeline tasks
- Generates and validates test files
- Produces comprehensive coverage reports

### Utilities

The system includes several utilities to streamline test file generation and validation:

- `batch_generate_tests.py`: Batch generation of test files for multiple models
- `validate_test_files.py`: Validation of generated test file syntax and structure
- `run_test_generator.sh`: Convenience script for running the comprehensive generator

## Key Features

1. **Standardized Tests**: All generated tests follow a consistent pattern with proper class inheritance, setup, and test methods.

2. **Hardware Detection**: Tests automatically detect and use the best available hardware (CPU, CUDA, MPS).

3. **Dependency Mocking**: Tests support mocking dependencies for CI/CD environments.

4. **Model Registries**: Each template includes a registry of supported models with their configurations.

5. **Comprehensive Coverage**: Tests cover from_pretrained, pipeline API, direct inference, and hardware-specific methods.

6. **Validation**: Integrated validation ensures generated tests have correct syntax and structure.

7. **Scaling**: Built-in support for batch generation and parallel processing.

## Usage

### Basic Template Usage

```bash
# Generate a test file for a specific model
python template_integration_workflow.py --model openai/clip-vit-base-patch32 --architecture multimodal
```

### Batch Generation

```bash
# Generate test files for multiple models by architecture
python batch_generate_tests.py --architectures vision multimodal
```

### Comprehensive Test Generation

```bash
# Generate tests for all HuggingFace Transformers classes
./comprehensive_test_generator.py

# Generate tests for specific categories
./comprehensive_test_generator.py --categories vision multimodal

# Just discover classes without generating tests
./comprehensive_test_generator.py --discover-only --discovery-output classes.json
```

### Using the Convenience Script

```bash
# Discover all HuggingFace classes
./run_test_generator.sh discover

# Generate tests for vision models
./run_test_generator.sh vision

# Generate tests for multimodal models with custom options
./run_test_generator.sh multimodal --dry-run --workers 8
```

### Validation

```bash
# Validate all test files in a directory
python validate_test_files.py --directory ../refactored_test_suite/models/multimodal
```

## Documentation

- `TEMPLATE_INTEGRATION_COMPLETED.md`: Status of template integration project
- `template_integration_summary.md`: Summary of completed templates
- `COMPREHENSIVE_TEST_GENERATOR.md`: Guide for the comprehensive test generator
- `MULTIMODAL_TEMPLATE_COMPLETION.md`: Details of multimodal template implementation

## Integration with Refactored Test Suite

All templates properly integrate with the refactored test suite architecture:

- All templates inherit from the `ModelTest` base class
- All templates follow standardized test methods and naming conventions
- All templates properly detect and support hardware acceleration
- All templates provide integrated dependency mocking for CI/CD