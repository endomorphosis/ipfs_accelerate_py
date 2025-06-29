# Minimal HuggingFace Test Files

This directory contains minimal test files for various HuggingFace model families. These files are designed to have correct indentation and syntax, while providing the essential functionality for testing models.

## Overview

The minimal test files:

1. Are correctly indented according to Python syntax rules
2. Pass Python syntax validation
3. Include the core functionality for testing models with transformers
4. Support different model architectures (encoder-only, decoder-only, encoder-decoder, vision)
5. Offer a cleaner alternative to the more complex test files

## Usage

Each test file follows this pattern:

```bash
python test_hf_bert.py --model bert-base-uncased --save
```

## Architecture-Aware Testing

These minimal test files are generated with architecture-specific customizations:

- **Encoder-only models** (bert, roberta, etc.): Focused on masked inputs and encoding
- **Decoder-only models** (gpt2, llama, etc.): Support for text generation
- **Encoder-decoder models** (t5, bart, etc.): Translation and sequence-to-sequence tasks
- **Vision models** (vit, swin, etc.): Image processing and classification

## File Structure

Each test file includes:

1. Import section with proper error handling
2. Hardware detection for CUDA/MPS/CPU
3. Model registry for specific models
4. TestModelFamily class with:
   - `__init__`: Model and environment setup
   - `test_pipeline`: Basic pipeline testing
   - `run_tests`: Unified test execution
5. Utility functions for saving results and CLI interface

## Creating Additional Test Files

To create test files for other model families, use the `manage_test_files.py` script:

```bash
python manage_test_files.py create <family> <output_path>
python manage_test_files.py batch-create <family1> <family2> ... --output-dir <dir>
```

## Supported Model Families

The following model families are supported:

- **Base families** (with dedicated templates):
  - bert, gpt2, t5, vit

- **Extended families** (using architecture-based templates):
  - Encoder-only: roberta, distilbert, albert, electra, etc.
  - Decoder-only: gpt_neo, gpt_neox, llama, bloom, etc.
  - Encoder-decoder: bart, pegasus, mbart, etc.
  - Vision: detr, swin, convnext, etc.

To get a complete list of supported families:

```bash
python manage_test_files.py list
```