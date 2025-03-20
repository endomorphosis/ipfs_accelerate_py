# HuggingFace Test Generator (Fixed)

This document describes the fixed version of the HuggingFace test generator, which addresses indentation issues and implements architecture-aware test generation.

## Overview

The test generator creates test files for HuggingFace transformers models with proper indentation and architecture-specific implementations. It supports different model architectures and hardware backends.

## Key Components

1. **regenerate_tests_with_fixes.py**: Main script for generating test files with fixes
2. **complete_indentation_fix.py**: Tool for fixing indentation issues
3. **test_integration.py**: End-to-end integration framework
4. **cleanup_integration.py**: Environment preparation script
5. **templates/**: Directory containing architecture-specific templates

## Architecture Support

The generator supports the following architectures:

- **Encoder-Only** (BERT, RoBERTa, DistilBERT, etc.)
- **Decoder-Only** (GPT-2, LLaMA, Mistral, etc.)
- **Encoder-Decoder** (T5, BART, Pegasus, etc.)
- **Vision** (ViT, Swin, DeiT, etc.)
- **Speech** (Whisper, Wav2Vec2, HuBERT, etc.)
- **Multimodal** (LLaVA, CLIP, BLIP, etc.)

## Usage

### Generate a Test File

```bash
python regenerate_tests_with_fixes.py --single bert
```

### Fix Indentation in a Test File

```bash
python complete_indentation_fix.py test_hf_bert.py
```

### Run the Integration Framework

```bash
python test_integration.py --all --core
```

### Prepare Environment

```bash
python cleanup_integration.py --templates
```

## Architecture-Specific Templates

The generator uses different templates for each architecture type:

### Encoder-Only Template

Special handling for:
- Mask token replacement in input text
- Bidirectional attention mechanism
- Token classification and sequence classification tasks

### Decoder-Only Template

Special handling for:
- Padding token configuration (pad_token = eos_token)
- Autoregressive generation parameters
- Text generation tasks

### Encoder-Decoder Template

Special handling for:
- Decoder input initialization
- Translation and summarization tasks
- Both encoder and decoder components

### Vision Template

Special handling for:
- Image tensor shape (batch_size, channels, height, width)
- Image preprocessing
- Classification and detection tasks

### Speech Template

Special handling for:
- Audio input preprocessing
- Sampling rates and feature extraction
- Transcription and recognition tasks

### Multimodal Template

Special handling for:
- Multiple input modalities
- Cross-modal attention
- Multimodal tasks like image captioning

## Indentation Fixing

The indentation fixer addresses common issues:

- Misaligned method definitions
- Incorrect spacing in class definitions
- Improperly indented dependency checks
- Broken try/except blocks
- Incorrect bracket and parenthesis matching

## Hardware Support

The generator includes code for detecting and using different hardware backends:

- CPU (always available)
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- OpenVINO (Intel hardware)

## Documentation

For more information, see:

- **INTEGRATION_README.md**: Complete documentation of the integration framework
- **HF_TEST_TROUBLESHOOTING_GUIDE.md**: Troubleshooting guidance
- **INTEGRATION_PLAN.md**: Phased implementation plan
- **TESTING_FIXES_SUMMARY.md**: Summary of fixes and improvements

## Next Steps

1. Generate tests for all 300+ HuggingFace model architectures
2. Integrate with CI/CD pipelines
3. Create comprehensive test coverage report
4. Implement nightly test runs

## License

Apache 2.0