# Hugging Face Model Testing Framework

This document provides a comprehensive overview of the class-based testing framework for Hugging Face models in the IPFS Accelerate Python project.

## Overview

The testing framework is designed around a class-based architecture that groups models by their transformer architecture type rather than testing each model individually. This approach provides significant advantages:

- **Reduced code duplication**: One test file handles multiple related models
- **Improved maintainability**: Updates only need to be made once per model family
- **Consistent testing**: All models in a family are tested in the same way
- **Efficient test discovery**: Easy to find which test file handles a particular model
- **Hardware optimization**: Testing can be adapted to available hardware

## Architecture

Each test file corresponds to a specific model architecture type and can test all models of that type:

| Test File | Architecture Class | Example Models |
|-----------|-------------------|----------------|
| `test_hf_bert.py` | BertForMaskedLM | bert-base-uncased, distilbert-base-uncased, roberta-base |
| `test_hf_gpt.py` | GPT2LMHeadModel | gpt2, gpt2-medium, distilgpt2 |
| `test_hf_t5.py` | T5ForConditionalGeneration | t5-small, t5-base, flan-t5-base |
| `test_hf_llama.py` | LlamaForCausalLM | meta-llama/Llama-2-7b, meta-llama/Llama-2-13b |
| `test_hf_clip.py` | CLIPModel | openai/clip-vit-base-patch32, laion/CLIP-ViT-L-14-laion2B-s32B-b82K |
| `test_hf_whisper.py` | WhisperForConditionalGeneration | openai/whisper-tiny, openai/whisper-small |
| `test_hf_vit.py` | ViTForImageClassification | google/vit-base-patch16-224, facebook/deit-base-patch16-224 |
| `test_hf_sam.py` | SamModel | facebook/sam-vit-base, facebook/sam-vit-large |

## Test Structure

Each test file follows a consistent structure:

1. **Model Registry**: A dictionary mapping specific model IDs to their configurations
2. **Base Test Class**: A class for testing any model in this architecture family
3. **Hardware Detection**: Code to identify available hardware resources
4. **Command-Line Interface**: Tools for running tests from the command-line

### Example Test Class

```python
class TestBertModels:
    """Base test class for all BERT-family models."""
    
    def __init__(self, model_id=None):
        # Model initialization and configuration
        self.model_id = model_id or "bert-base-uncased"
        # ...
        
    def test_pipeline(self, device="auto"):
        # Test using transformers pipeline
        # ...
        
    def test_from_pretrained(self, device="auto"):
        # Test using direct model loading
        # ...
        
    def test_with_openvino(self):
        # Test using OpenVINO integration
        # ...
        
    def run_tests(self, all_hardware=False):
        # Run all appropriate tests
        # ...
```

## Test Capabilities

Each test class includes the following core capabilities:

### 1. Multiple Testing Methods

- **Pipeline API Testing**: Tests the model using the high-level Transformers pipeline
- **Direct API Testing**: Tests the model using direct from_pretrained loading
- **OpenVINO Integration**: Tests the model with OpenVINO optimizations when available

### 2. Multiple Hardware Support

- **CPU Testing**: Always available baseline testing
- **CUDA Testing**: Automatically used when available
- **MPS Testing**: Supported on Apple Silicon devices
- **OpenVINO Testing**: Available when OpenVINO is installed

### 3. Comprehensive Metrics

- **Load Times**: Measures model and tokenizer loading times
- **Inference Speed**: Runs multiple trials to get accurate timing
- **Memory Usage**: Tracks model size and memory footprint
- **Prediction Quality**: Records model output quality and accuracy

### 4. Error Detection and Diagnosis

- **Dependency Checking**: Verifies all required libraries are available
- **Error Classification**: Categorizes errors by type (CUDA, OOM, dependency)
- **Detailed Reporting**: Provides stack traces and error context
- **Hardware Compatibility**: Tests compatibility across devices

## Usage Guide

### Basic Usage

```bash
# Test a specific model
python test_hf_bert.py --model bert-base-uncased

# Test all models of a particular type
python test_hf_bert.py --all-models

# List available models of this type
python test_hf_bert.py --list-models

# Save detailed test results
python test_hf_bert.py --model bert-base-uncased --save
```

### Hardware-Specific Testing

```bash
# Test on all available hardware
python test_hf_bert.py --model bert-base-uncased --all-hardware

# Force CPU-only testing
python test_hf_bert.py --model bert-base-uncased --cpu-only
```

### Batch Testing

```bash
# Test all models and save results
python test_all_models.py --category bert

# Test all models across all architectures
python test_all_models.py --all
```

## Model Registry

Each test file contains a registry of supported models with their specific configurations:

```python
BERT_MODELS_REGISTRY = {
    "bert-base-uncased": {
        "description": "BERT base model (uncased)",
        "class": "BertForMaskedLM",
        "vocab_size": 30522,
        "dependency_note": None
    },
    "distilbert-base-uncased": {
        "description": "DistilBERT base model (uncased)",
        "class": "DistilBertForMaskedLM",
        "vocab_size": 30522,
        "dependency_note": None
    },
    # ...
}
```

This registry approach allows:
- Documenting specific model variations
- Tracking special requirements
- Setting appropriate expectations for each model

## Results Format

Test results are saved in JSON format with the following structure:

```json
{
  "results": {
    "pipeline_cuda": {
      "model": "bert-base-uncased",
      "device": "cuda",
      "pipeline_success": true,
      "pipeline_avg_time": 0.010031,
      "pipeline_error_type": "none"
    },
    "from_pretrained_cuda": {
      "model": "bert-base-uncased",
      "device": "cuda",
      "from_pretrained_success": true,
      "from_pretrained_avg_time": 0.008558,
      "model_size_mb": 417.764,
      "predictions": [...]
    }
  },
  "examples": [...],
  "performance": {...},
  "hardware": {...},
  "metadata": {...}
}
```

## Implementation Status

Current implementation status by model family:

| Model Family | Test File | Status | Examples Covered |
|--------------|-----------|--------|------------------|
| BERT | test_hf_bert.py | ✅ Complete | 7 models |
| GPT | test_hf_gpt.py | 🔄 In Progress | 3 models |
| T5 | test_hf_t5.py | 🔄 In Progress | 2 models |
| LLaMA | test_hf_llama.py | 🔄 In Progress | 2 models |
| Vision Transformers | test_hf_vit.py | 🔄 In Progress | 3 models |
| CLIP | test_hf_clip.py | 🔄 In Progress | 2 models |
| Whisper | test_hf_whisper.py | 🔄 In Progress | 2 models |
| SAM | test_hf_sam.py | 📝 Planned | - |
| LLaVA | test_hf_llava.py | 📝 Planned | - |

## Adding a New Model Family

To add tests for a new model family:

1. Create a new test file based on the existing templates
2. Define a model registry with supported models
3. Create a test class with appropriate test methods
4. Add hardware detection and command-line interface
5. Register the test in the main test_all_models.py file

## Dependency Management

Dependencies are managed on a per-model-family basis, with appropriate mocking for missing dependencies. Each test file:

1. Detects available dependencies
2. Creates appropriate mocks for missing dependencies
3. Skips tests that require unavailable dependencies
4. Reports dependency issues clearly

## Next Steps

1. **Complete remaining model families**: Finish implementation for all major model types
2. **Enhance test coverage**: Add more comprehensive tests for each model family
3. **Improve performance reporting**: Add detailed performance comparison tools
4. **Add model-specific test cases**: Include specialized tests for particular model features
5. **Create unified test runner**: Build a comprehensive tool to run all tests

## Conclusion

The class-based testing approach provides a powerful, maintainable framework for testing hundreds of Hugging Face models with minimal code duplication. By organizing tests by architecture family rather than individual models, we create a more sustainable testing framework that can easily adapt to new models.