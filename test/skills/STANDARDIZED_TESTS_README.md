# Standardized HuggingFace Model Testing Framework

This is a framework for creating standardized test files for HuggingFace transformer models of various architectures. The standardized tests implement consistent methods for loading and testing models while supporting model-specific adaptations through helper methods.

## Features

- Consistent testing approach across model architectures
- Support for hyphenated model names (e.g., gpt-j, bert-base)
- Flexible helper methods for model-specific adaptations
- Hardware detection for CUDA, MPS, and CPU
- Dependency checking with graceful degradation
- Performance measurement and statistics collection
- Error classification and handling
- Results collection and storage

## Usage

### Creating a New Standardized Test File

```bash
python create_standardized_test.py <model_family> <output_path>
```

Examples:
```bash
# Create a test file for BERT models
python create_standardized_test.py bert fixed_tests/test_hf_bert_standardized.py

# Create a test file for GPT-2 models
python create_standardized_test.py gpt2 fixed_tests/test_hf_gpt2_standardized.py

# Create a test file for T5 models
python create_standardized_test.py t5 fixed_tests/test_hf_t5_standardized.py

# Create a test file for ViT (Vision Transformer) models
python create_standardized_test.py vit fixed_tests/test_hf_vit_standardized.py

# Create a test file for models with hyphenated names (like GPT-J)
python create_standardized_test.py gpt-j fixed_tests/test_hf_gpt_j_standardized.py
```

### Running a Standardized Test

Once created, the standardized test files can be run as standalone Python scripts:

```bash
# Run a test for BERT with default model
python fixed_tests/test_hf_bert_standardized.py

# Run a test for a specific model
python fixed_tests/test_hf_bert_standardized.py --model google-bert/bert-base-uncased

# Save test results to a file
python fixed_tests/test_hf_bert_standardized.py --save

# Test on a specific device
python fixed_tests/test_hf_bert_standardized.py --device cuda
```

## Test Methods

Each standardized test file implements the following methods:

1. `test_pipeline()`: Tests the model using the HuggingFace pipeline API
2. `test_from_pretrained()`: Tests the model by directly loading it with the from_pretrained method
3. `run_tests()`: Runs all tests and collects results

## Helper Methods for Model-Specific Adaptations

These methods can be customized for different model architectures:

1. `get_model_class()`: Returns the appropriate model class based on model type
2. `prepare_test_input()`: Prepares appropriate test input for the model type
3. `process_model_output()`: Processes model output based on model type

## Supported Model Architectures

The framework currently has built-in support for these model architectures:

- **BERT** and other masked language models (encoder-only)
- **GPT-2** and other causal language models (decoder-only)
- **T5** and other sequence-to-sequence models (encoder-decoder)
- **ViT** and other vision models
- **GPT-J** and other large language models

Additional architectures can be supported by adding entries to the `MODEL_INFO` dictionary in the script.

## Results Collection

Test results are stored in JSON format with detailed information about:

- Test success/failure and error types
- Performance statistics (inference time, load time)
- Example inputs and outputs
- Model metadata
- Hardware configuration
- Dependency availability

## Supporting New Model Types

To add support for a new model architecture, add an entry to the `MODEL_INFO` dictionary in the `create_standardized_test.py` script:

```python
MODEL_INFO = {
    # Existing entries...
    
    'new_model_family': {
        'default_model': 'model-org/model-name',
        'default_class': 'ModelClass',
        'architecture': 'architecture-type',  # encoder-only, decoder-only, encoder-decoder, vision, etc.
        'task': 'task-name',  # fill-mask, text-generation, etc.
        'test_input': 'Example input text for testing'
    }
}
```