# Enhanced Template Test Generator for IPFS Accelerate Models

A robust test file generator that creates test implementations compatible with the ipfs_accelerate_py worker/skillset module structure. This generator ensures consistent test coverage across all 300+ Hugging Face model types while maintaining alignment with the actual implementation structure.

**NEW**: The generator now includes comprehensive model registry information with detailed input/output data types, endpoint handler parameters, and dependency requirements.

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
- **Enhanced Model Registry**: Detailed model registry with comprehensive metadata about input/output formats, helper functions, and dependencies
- **Model-Specific Task Support**: Automatically pulls task information from the huggingface_model_pipeline_map.json file to customize tests for each model's capabilities 
- **Reference Implementation Integration**: Intelligently checks for existing model implementations in the worker/skillset directory to use as templates
- **Complete Multi-Platform Support**: Creates robust tests for all supported hardware platforms: CPU, CUDA, OpenVINO, Apple Silicon (MPS), and Qualcomm AI
- **Detailed Helper Functions**: Documents each helper function with its description, required arguments, and return types
- **Endpoint Handler Parameters**: Specifies required and optional parameters for endpoint handlers with types and default values
- **Dependency Management**: Lists all required Python packages, version requirements, and platform-specific dependencies
- **Mock Implementations with Fallbacks**: Provides sophisticated mock implementations with proper fallback mechanisms for all components
- **Hardware-Aware Testing**: Automatically detects available hardware and adapts tests accordingly, ensuring tests work in any environment
- **Handler Method Verification**: Thoroughly tests all handler creation methods, verifying proper interface adherence
- **Standardized Results Collection**: Implements consistent result collection and reporting across all models

## Available Generators

The framework now includes three test generators, each with different levels of detail and complexity:

1. **Template Test Generator** (`template_test_generator.py`): The original generator that maintains compatibility with the worker/skillset structure.

2. **Improved Template Generator** (`improved_template_generator.py`): Enhanced generator with comprehensive model registry information.

3. **Basic Test Generator** (`generate_basic_test.py`): Simplified generator that still includes the enhanced model metadata.

## Usage

### Template Test Generator (Original)

Generate a test for a specific model:

```bash
python template_test_generator.py --model <model_name> --output-dir <output_directory>
```

For example:

```bash
python template_test_generator.py --model bert --output-dir ./test_output
```

### Improved Template Generator (Enhanced)

The enhanced generator includes detailed model registry information:

```bash
python improved_template_generator.py --model <model_name> --output-dir <output_directory>
```

For example:

```bash
python improved_template_generator.py --model bert --output-dir ./sample_tests
```

### Basic Test Generator (Simplified)

A simplified generator that still includes the enhanced model metadata:

```bash
python generate_basic_test.py <model_name> [<output_directory>]
```

For example:

```bash
python generate_basic_test.py bert-base-uncased ./sample_tests
```

### Options (Template and Improved Generators)

- `--model`: The model type to generate a test for (required)
- `--output-dir`: Directory to save the test file (defaults to test/skills or sample_tests)
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
3. **Enhanced Model Registry**: Comprehensive model metadata including input/output specifications
4. **Handler Creation Methods**: Methods to create handlers for different hardware platforms
5. **Initialization Methods**: Methods to initialize the model on different hardware platforms
6. **Test Method**: A `__test__` method to validate the implementation
7. **Run Helper**: A `run_test()` function to easily run the test from the command line

## Enhanced Model Registry Structure

The improved generator includes a detailed model registry with comprehensive metadata:

```python
MODEL_REGISTRY = {
    "model-name": {
        "description": "Description of the model",
        
        # Model dimensions and capabilities
        "embedding_dim": 768,
        "sequence_length": 512,
        "model_precision": "float32", 
        "supports_half_precision": True,
        "default_batch_size": 1,
        
        # Hardware compatibility
        "hardware_compatibility": {
            "cpu": True,
            "cuda": True,
            "openvino": True,
            "apple": True,
            "qualcomm": False
        },
        
        # Input/Output specifications
        "input": {
            "format": "text",  # text, image, audio, multimodal
            "tensor_type": "int64",
            "uses_attention_mask": True,
            "uses_position_ids": False,
            "typical_shapes": ["batch_size, sequence_length"]
        },
        "output": {
            "format": "embedding",  # text, image, audio, embedding
            "tensor_type": "float32",
            "typical_shapes": ["batch_size, sequence_length, hidden_size"]
        },
        
        # Required helper functions with arguments
        "helper_functions": {
            "tokenizer": {
                "description": "Tokenizes input text",
                "args": ["text", "max_length", "padding", "truncation"],
                "returns": "Dictionary with input_ids and attention_mask"
            },
            "model_loader": {
                "description": "Loads model from pretrained weights",
                "args": ["model_name", "cache_dir", "device"],
                "returns": "Loaded model instance"
            }
        },
        
        # Endpoint handler parameters
        "handler_params": {
            "text": {
                "description": "Input text to process",
                "type": "str or List[str]",
                "required": True
            },
            "max_length": {
                "description": "Maximum sequence length",
                "type": "int",
                "required": False,
                "default": 512
            }
        },
        
        # Dependencies
        "dependencies": {
            "python": ">=3.8,<3.11",
            "pip": ["torch>=1.12.0", "transformers>=4.26.0", "numpy>=1.20.0"],
            "system": [],
            "optional": {
                "cuda": ["nvidia-cuda-toolkit>=11.6", "nvidia-cudnn>=8.3"],
                "openvino": ["openvino>=2022.1.0"],
                "apple": ["torch>=1.12.0"],
                "qualcomm": ["qti-aisw>=1.8.0"]
            }
        }
    }
}

## Customizing for Tasks

The generator automatically detects the appropriate tasks for each model from the huggingface_model_pipeline_map.json file. This ensures that the test matches the expected capabilities of the model.

For models where task information is unavailable, the generator defaults to "text-generation" as the primary task.

## Known Issues and Future Improvements

1. **Task-Specific Templates**: While the generator pulls task information and now provides more detailed helper function metadata, we still need more specialized templates for all 300+ task types. Future versions will include even more customized templates for each model family and task combination.

2. **Real Implementation Testing**: The current implementation uses mocks for testing. Future versions will better integrate with real model implementations when available while still providing robust mocks for testing in environments without dependencies.

3. **Dynamic Templating System**: A more sophisticated templating system is needed to handle the wide variety of model architectures and tasks, which will be implemented in future versions.

4. **Automated Dependency Detection**: While the enhanced model registry includes dependency specifications, future improvements could include automated detection and installation of required dependencies.

5. **Input/Output Shape Inference**: The current implementation provides typical shapes, but future versions could automatically infer the exact shapes from the model architecture.

## Integration Path

The generator system now includes multiple options with different levels of detail. The recommended integration path is:

1. Use the `improved_template_generator.py` as the primary test generator for models that need detailed metadata
2. Use the `generate_basic_test.py` for quick test generation when full details aren't required
3. Apply the enhanced model registry structure to the `merged_test_generator.py`
4. Incorporate the detailed helper function documentation and endpoint parameter specifications into all generators

## Contributing

To improve the template generator:

1. **Enhance Model Registry**: Further improve the model registry with additional metadata fields
2. **Expand Task Templates**: Add specialized templates for different model tasks and architectures
3. **Enhance Real Implementation Testing**: Improve integration with real implementations where available
4. **Add Dependency Management**: Implement automatic dependency installation during test setup
5. **Improve Helper Function Documentation**: Expand documentation for helper functions with examples
6. **Add Input/Output Shape Inference**: Implement automatic shape inference based on model architecture
7. **Refine Hardware Detection**: Enhance the hardware platform detection and adaptation mechanisms
8. **Improve Error Handling**: Add more sophisticated error handling and reporting
9. **Add Batch Testing Capabilities**: Implement more efficient batch testing across multiple models
10. **Integrate with CI/CD**: Enable integration with continuous testing pipelines

## Further Reading

For more information, see:
- [RECOMMENDATION_FOR_TEST_GENERATOR.md](RECOMMENDATION_FOR_TEST_GENERATOR.md): Detailed recommendations for improving the test generator
- [SUMMARY_OF_IMPROVEMENTS.md](SUMMARY_OF_IMPROVEMENTS.md): Summary of the improvements made with this implementation

### Sample Generated Test Files

The following sample test files have been generated with the enhanced test generators:

- [test_hf_bert_base_uncased.py](sample_tests/test_hf_bert_base_uncased.py): BERT text model with enhanced model registry
- [test_hf_t5_small.py](sample_tests/test_hf_t5_small.py): T5 text generation model with enhanced model registry
- [test_hf_vit.py](sample_tests/test_hf_vit.py): Vision Transformer model with image-specific helper functions
- [test_hf_whisper.py](sample_tests/test_hf_whisper.py): Audio model with audio-specific dependencies and helper functions