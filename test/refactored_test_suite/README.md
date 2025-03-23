# IPFS Accelerate Refactored Test Suite

This directory contains a comprehensive test framework for validating HuggingFace model support in the IPFS Accelerate Python framework. The test suite is designed around the `ModelTest` pattern with architecture-specific base classes.

## Overview

The test framework is structured to facilitate:

- **Standardized Testing**: Common test patterns across all model architectures
- **Architecture-Specific Handling**: Specialized tests for each model architecture (encoder-only, decoder-only, etc.)
- **Test Generation**: Automated creation of test files from templates
- **CI/CD Integration**: Mock support for running tests in CI environments
- **Comprehensive Validation**: Tools to verify test completeness and compliance

## Directory Structure

```
refactored_test_suite/
├── generated_tests/         # Generated test files
├── generators/              # Test generation components
│   ├── architecture_detector.py  # Model architecture detection
│   └── test_generator.py    # Test file generator
├── models/                  # Manually written model tests
│   ├── audio/               # Speech model tests
│   ├── multimodal/          # Multimodal model tests
│   ├── text/                # Text model tests
│   └── vision/              # Vision model tests
├── reports/                 # Test and validation reports
├── templates/               # Templates for each architecture
│   ├── decoder_only_template.py
│   ├── encoder_decoder_template.py
│   ├── encoder_only_template.py
│   ├── multimodal_template.py
│   ├── speech_template.py
│   ├── vision_template.py
│   └── vision_text_template.py
├── validation/              # Validation components
│   └── test_validator.py    # Test file validator
├── model_test_base.py       # Base classes for tests
├── run_comprehensive_test_suite.py  # Main runner for all steps
├── run_integration_tests.py # Integration test runner
├── run_test_generation.py   # Test generation script
├── run_validation.py        # Validation script
└── track_implementation_progress.py # Progress tracking
```

## Key Components

### ModelTest Base Classes

The `model_test_base.py` file defines the foundational test architecture:

- `ModelTest`: Abstract base class with common functionality
- `EncoderOnlyModelTest`: For models like BERT, RoBERTa
- `DecoderOnlyModelTest`: For models like GPT-2, LLaMA
- `EncoderDecoderModelTest`: For models like T5, BART
- `VisionModelTest`: For models like ViT, Swin
- `SpeechModelTest`: For models like Whisper, Wav2Vec2
- `VisionTextModelTest`: For models like CLIP, BLIP
- `MultimodalModelTest`: For models like LLaVA, FLAVA

### Architecture Detection

The architecture detector (`generators/architecture_detector.py`) provides:

- Pattern-based model architecture detection
- Fallback to HuggingFace config inspection
- Model name normalization
- Metadata extraction for models

### Test Generator

The test generator (`generators/test_generator.py`) offers:

- Template-based test file generation
- Architecture-specific template handling
- Syntax and pattern validation of generated files
- Batch generation for multiple models

### Validation System

The validation system (`validation/test_validator.py`) ensures:

- Python syntax validation
- ModelTest pattern compliance verification
- Required method implementation checking
- Detailed reporting on validation results

## Running the Test Suite

### Comprehensive Test Suite

To run all steps of the test suite:

```bash
python run_comprehensive_test_suite.py --test-dir ./generated_tests --report-dir ./reports
```

Options:
- `--priority [high|medium|low|all]`: Set model priority for generation
- `--mock`: Use mocked dependencies for integration tests
- `--force`: Overwrite existing test files
- `--validate`: Run only validation step
- `--generate`: Run only test generation step
- `--integrate`: Run only integration tests step
- `--track`: Run only implementation tracking step

### Individual Components

#### Test Generation

Generate tests for specific model priorities:

```bash
python run_test_generation.py --priority high --output-dir ./generated_tests
```

#### Validation

Validate existing tests:

```bash
python run_validation.py --test-dir ./generated_tests --report-dir ./reports
```

#### Integration Tests

Run tests against actual models:

```bash
python run_integration_tests.py --test-dir ./generated_tests --output-dir ./reports
```

Use `--mock` flag for CI environments to mock dependencies.

#### Implementation Tracking

Track progress on implementing tests for all required models:

```bash
python track_implementation_progress.py --dirs ./generated_tests ./models --output ./reports/implementation_progress.md
```

## CI/CD Integration

For CI/CD environments, use the mock system to avoid downloading large model files:

```bash
# Set environment variables
export MOCK_TORCH=true
export MOCK_TRANSFORMERS=true
export MOCK_TOKENIZERS=true
export MOCK_SENTENCEPIECE=true

# Or use the --mock flag with the comprehensive script
python run_comprehensive_test_suite.py --mock
```

## Adding a New Model Test

### Using the Generator

Generate a test for a new model:

```bash
python run_test_generation.py --model bert
```

### Manual Creation

1. Determine the model's architecture type
2. Use the appropriate template from the `templates/` directory
3. Implement required methods:
   - `get_default_model_id()`: Return the default model ID
   - `run_all_tests()`: Run tests for the model

### Validation

Validate your test file:

```bash
python run_validation.py --test-dir ./your_test_directory
```

## Architecture Guidelines

When implementing a new test, follow these guidelines:

1. **Inherit from the correct architecture-specific class**:
   - Text models should inherit from `EncoderOnlyModelTest`, `DecoderOnlyModelTest`, or `EncoderDecoderModelTest`
   - Vision models should inherit from `VisionModelTest`
   - Speech models should inherit from `SpeechModelTest`
   - Vision-text models should inherit from `VisionTextModelTest`
   - Multimodal models should inherit from `MultimodalModelTest`

2. **Set model-specific properties**:
   - Set `self.model_type` to the base model type
   - Set `self.task` to the appropriate task
   - Set `self.architecture_type` to the architecture type

3. **Implement required methods**:
   - Override `get_default_model_id()` to return the correct model ID
   - Implement `run_all_tests()` to run tests for the model
   - Optionally add model-specific tests

4. **Use the mock system for CI/CD**:
   - All tests should check for mock environment variables
   - Tests should function in both real and mocked modes

## Progress Tracking

Track implementation progress for high-priority models:

```bash
python track_implementation_progress.py
```

This generates a report showing:
- Overall implementation progress
- Implementation progress by priority and architecture
- Missing high-priority models
- Next steps for implementation

## CI/CD Automation

We have implemented GitHub Actions workflows to automate testing in CI environments. See `.github/workflows/model_tests.yml` for the configuration.

Key features include:
- Matrix testing across all model architectures
- Automatic generation of coverage reports
- Testing with mocked dependencies for CI environments
- Verification of test generation for new models

For more details on the CI/CD integration, see [CI_CD_INTEGRATION.md](CI_CD_INTEGRATION.md).

## Next Steps

1. **Performance Benchmarking**: Add performance testing to CI/CD workflow
2. **Test Result Visualization**: Create dashboards for visualizing test results
3. **Notification System**: Set up alerts for test failures
4. **Advanced Validation**: Implement model-specific validation rules
5. **Documentation Expansion**: Create detailed guides for contributors

For a complete list of next steps and implementation details, see [CI_CD_INTEGRATION.md](CI_CD_INTEGRATION.md).