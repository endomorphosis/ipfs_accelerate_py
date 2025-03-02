# Template Test Generator for IPFS Accelerate Models

A robust test file generator that creates test implementations compatible with the ipfs_accelerate_py worker/skillset module structure. This generator ensures consistent test coverage across all 300+ Hugging Face model types while maintaining alignment with the actual implementation structure.

## Overview

The Template Test Generator (`template_test_generator.py`) addresses a critical need for standardized test file generation that properly aligns with the ipfs_accelerate_py worker/skillset module structure. This tool was developed after observing inconsistencies in the existing test generation process, where generated tests did not properly match the implementation structure required by the skillset modules.

The generator produces test files with the correct:

- Class structure and initialization that mirrors the actual implementation
- Handler creation methods for multiple hardware platforms (CPU, CUDA, OpenVINO, Apple Silicon, Qualcomm AI)
- Robust hardware detection and platform-specific implementation strategies
- Comprehensive test methods that verify the entire interface and functionality
- Proper error handling and fallback mechanisms for different environments

## Key Features

- **Consistent Structure Alignment**: Ensures perfect alignment with the ipfs_accelerate_py worker/skillset implementation structure, including required methods and attributes
- **Model-Specific Task Support**: Automatically pulls task information from the huggingface_model_pipeline_map.json file to customize tests for each model's capabilities 
- **Reference Implementation Integration**: Intelligently checks for existing model implementations in the worker/skillset directory to use as templates
- **Complete Multi-Platform Support**: Creates robust tests for all supported hardware platforms: CPU, CUDA, OpenVINO, Apple Silicon (MPS), and Qualcomm AI
- **Mock Implementations with Fallbacks**: Provides sophisticated mock implementations with proper fallback mechanisms for all components
- **Hardware-Aware Testing**: Automatically detects available hardware and adapts tests accordingly, ensuring tests work in any environment
- **Handler Method Verification**: Thoroughly tests all handler creation methods, verifying proper interface adherence
- **Standardized Results Collection**: Implements consistent result collection and reporting across all models

## Usage

### Basic Usage

Generate a test for a specific model:

```bash
python template_test_generator.py --model <model_name> --output-dir <output_directory>
```

For example:

```bash
python template_test_generator.py --model bert --output-dir ./test_output
```

### Options

- `--model`: The model type to generate a test for (required)
- `--output-dir`: Directory to save the test file (defaults to test/skills)
- `--force`: Force overwrite if the file already exists

### Examples

Generate a test for a language model:

```bash
python template_test_generator.py --model llama --output-dir ./language_models
```

Generate a test for a vision model:

```bash
python template_test_generator.py --model vit --output-dir ./vision_models
```

Generate a test for an audio model:

```bash
python template_test_generator.py --model whisper --output-dir ./audio_models
```

## Helper Scripts

Several helper scripts are provided to demonstrate the capabilities of the template generator:

### Generate Sample Tests

The `generate_sample_tests.py` script generates tests for a representative set of models from different categories:

```bash
python generate_sample_tests.py
```

This generates tests for:
- bert (Language)
- vit (Vision)
- whisper (Audio)
- llava (Multimodal)

### Generate Tests for All Worker/Skillset Models

The `generate_tests_for_all_skillset_models.py` script generates tests for all models found in the worker/skillset directory:

```bash
python generate_tests_for_all_skillset_models.py
```

## Generated Test Structure

The generated test files include:

1. **Class Definition**: A class with the correct naming convention (`hf_<model_type>`)
2. **Initialization Method**: Proper initialization with resources and metadata
3. **Handler Creation Methods**: Methods to create handlers for different hardware platforms
4. **Initialization Methods**: Methods to initialize the model on different hardware platforms
5. **Test Method**: A `__test__` method to validate the implementation
6. **Run Helper**: A `run_test()` function to easily run the test from the command line

## Customizing for Tasks

The generator automatically detects the appropriate tasks for each model from the huggingface_model_pipeline_map.json file. This ensures that the test matches the expected capabilities of the model.

For models where task information is unavailable, the generator defaults to "text-generation" as the primary task.

## Known Issues and Future Improvements

1. **Torch Tensor Attribute Issue**: The generated code attempts to set attributes on torch tensors, which are not writable. The handlers currently fall back to dictionaries when this occurs, which works but isn't ideal. In future versions, this will be fixed by using a proper wrapper class that maintains both the tensor functionality and custom attributes.

2. **Task-Specific Templates**: While the generator pulls task information, it doesn't yet have fully customized templates for all 300+ task types. Future versions will include more specialized templates for each model family and task combination.

3. **Real Implementation Testing**: The current implementation uses mocks for testing. Future versions will better integrate with real model implementations when available while still providing robust mocks for testing in environments without dependencies.

4. **Dynamic Templating System**: A more sophisticated templating system is needed to handle the wide variety of model architectures and tasks, which will be implemented in future versions.

## Integration Path

The `template_test_generator.py` was designed as a reference implementation that demonstrates the correct approach for generating test files. The recommended integration path is:

1. Use this generator as a reference for updating the `merged_test_generator.py`
2. Apply the structural patterns from this generator to the merged generator
3. Maintain the registry and model discovery capabilities of the merged generator
4. Incorporate the hardware support and handler creation methods from this generator

## Contributing

To improve the template generator:

1. **Fix Torch Tensor Issue**: Implement a wrapper class or dictionary approach to solve the tensor attribute issue
2. **Expand Task Templates**: Add specialized templates for different model tasks and architectures
3. **Enhance Real Implementation Testing**: Improve integration with real implementations where available
4. **Refine Hardware Detection**: Enhance the hardware platform detection and adaptation mechanisms
5. **Improve Error Handling**: Add more sophisticated error handling and reporting
6. **Add Batch Testing Capabilities**: Implement more efficient batch testing across multiple models
7. **Integrate with CI/CD**: Enable integration with continuous testing pipelines

## Further Reading

For more information, see:
- [RECOMMENDATION_FOR_TEST_GENERATOR.md](RECOMMENDATION_FOR_TEST_GENERATOR.md): Detailed recommendations for improving the test generator
- [SUMMARY_OF_IMPROVEMENTS.md](SUMMARY_OF_IMPROVEMENTS.md): Summary of the improvements made with this implementation