# Refactored Model Tests

## Overview

This directory contains test files for various model types, organized by modality and following the refactored test suite structure. These tests are designed to provide comprehensive validation of model functionality while maintaining a consistent structure and reducing code duplication.

## Directory Structure

- `audio/`: Tests for audio models like Whisper, Wav2Vec2, etc.
- `multimodal/`: Tests for multimodal models like CLIP, LLaVA, etc.
- `text/`: Tests for text models:
  - Encoder-only (BERT, RoBERTa, etc.)
  - Decoder-only (GPT, LLaMA, etc.) 
  - Encoder-decoder (T5, BART, etc.)
- `vision/`: Tests for vision models like ViT, DETR, etc.
- `other/`: Tests for specialized or custom models

## Test Structure

All test files follow a standardized structure based on the `ModelTest` base class:

```python
class TestModelName(ModelTest):
    def setUp(self):
        super().setUp()
        # Model-specific setup
        
    def tearDown(self):
        super().tearDown()
        # Model-specific cleanup
        
    def load_model(self):
        # Model loading logic
        
    def test_model_loading(self):
        # Tests that model loads correctly
        
    def test_basic_inference(self):
        # Tests basic inference capability
        
    def test_hardware_compatibility(self):
        # Tests model on different hardware
```

## Template Generation

Test files can be automatically generated using the template system. To generate a new test file:

```bash
cd test/template_integration
python debug_template.py --model-type bert --model-id bert-large-uncased
```

Available model types:
- `vision`: Vision models
- `bert`: Encoder-only models like BERT
- `gpt`: Decoder-only models like GPT

## Inheritance Hierarchy

```
BaseTest
└── ModelTest
    ├── TestVitModel (Vision)
    ├── TestBertModel (Encoder-only)
    ├── TestGptModel (Decoder-only)
    └── Other model test classes...
```

## Running Tests

To run the refactored tests:

```bash
cd test
python run_refactored_test_suite.py --model vision
```

## Documentation

For more information on the refactored test suite:
- [TEST_REFACTORING_PLAN.md](../../TEST_REFACTORING_PLAN.md): Overall refactoring plan
- [TEMPLATE_REFACTORING_GUIDE.md](../../template_integration/TEMPLATE_REFACTORING_GUIDE.md): Guide for template integration
- [TEMPLATE_INTEGRATION_COMPLETED.md](../../template_integration/TEMPLATE_INTEGRATION_COMPLETED.md): Template integration completion report