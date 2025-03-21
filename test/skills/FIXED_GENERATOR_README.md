# HuggingFace Test Generator (Fixed)

This document describes the fixed version of the HuggingFace test generator, which addresses indentation issues and implements architecture-aware test generation.

## Overview

The test generator creates test files for HuggingFace transformers models with proper indentation and architecture-specific implementations. It supports different model architectures and hardware backends.

## Key Components

1. **test_generator_fixed.py**: Fixed generator with architecture awareness and proper indentation
2. **regenerate_tests_with_fixes.py**: Main script for generating test files with fixes
3. **fix_indentation_and_apply_template.py**: Direct script for applying templates to specific model types
4. **integrate_generator_fixes.py**: Script to apply fixes to the original generator
5. **templates/**: Directory containing architecture-specific templates

## New Direct Fixing Approach (March 2025)

We've developed a faster and more reliable approach to fixing test files:

1. **Direct Template Application**: The `fix_indentation_and_apply_template.py` script can directly fix model-specific tests without complex regeneration logic.
2. **Generator Integration**: The fixes have been integrated into the generator itself (`test_generator_fixed.py`).
3. **Architecture Detection**: Automatic architecture type detection based on model name.
4. **Fixed Class References**: Corrected class references for different architectures:
   - `AutoModelForCausalLM` for decoder-only models (was incorrectly using `AutoModelLMHeadModel`)
   - Fixed capitalization for vision model classes (e.g., `ViTForImageClassification`)
5. **Architecture-Specific Processing**: Added specialized processing for each architecture type.

## Architecture Support

The generator supports the following architectures:

- **Encoder-Only** (BERT, RoBERTa, DistilBERT, etc.)
- **Decoder-Only** (GPT-2, LLaMA, Mistral, etc.)
- **Encoder-Decoder** (T5, BART, Pegasus, etc.)
- **Vision** (ViT, Swin, DeiT, etc.)
- **Speech** (Whisper, Wav2Vec2, HuBERT, etc.)
- **Multimodal** (LLaVA, CLIP, BLIP, etc.)

## Using the Fixed Approach

### Direct Template Application (Recommended)

This approach is fastest and most reliable:

```bash
# Fix a specific model test
python fix_indentation_and_apply_template.py bert --output-dir fixed_tests --verify

# Fix multiple model tests
python fix_indentation_and_apply_template.py gpt2 --output-dir fixed_tests --verify
python fix_indentation_and_apply_template.py t5 --output-dir fixed_tests --verify
python fix_indentation_and_apply_template.py vit --output-dir fixed_tests --verify
```

### Use the Fixed Generator

Generate tests using the fixed generator:

```bash
python test_generator_fixed.py --model bert
python test_generator_fixed.py --model gpt2
```

### Apply Fixes to Original Generator

```bash
python integrate_generator_fixes.py --generator test_generator.py
```

### Legacy Approach

```bash
# Still available but not recommended
python regenerate_tests_with_fixes.py --single bert
python complete_indentation_fix.py test_hf_bert.py
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

## Recent Updates (March 2025)

We've successfully:

1. **Fixed Core Model Tests**:
   - `test_hf_bert.py` - Encoder-only architecture (verified and working)
   - `test_hf_gpt2.py` - Decoder-only architecture (verified and working)
   - `test_hf_t5.py` - Encoder-decoder architecture (verified and working)
   - `test_hf_vit.py` - Vision architecture (verified and working)

2. **Key Bug Fixes**:
   - Fixed GPT-2 model class (`AutoModelLMHeadModel` → `AutoModelForCausalLM`)
   - Fixed ViT capitalization (`VitForImageClassification` → `ViTForImageClassification`)
   - Added padding token handling for decoder-only models
   - Added decoder input initialization for encoder-decoder models
   - Added architecture detection for proper model class selection

3. **Enhanced Test Result Clarity**:
   - Added comprehensive mock detection system to clearly indicate when tests are using mock objects vs. real inference
   - Implemented visual indicators (🚀 for real inference, 🔷 for mock objects) with detailed dependency reporting
   - Added granular dependency tracking in test metadata to improve transparency (`has_transformers`, `has_torch`, `has_tokenizers`, `has_sentencepiece`)
   - Enhanced test output to show complete environment information and dependency status
   - Added test type indicator in metadata (`REAL INFERENCE` vs. `MOCK OBJECTS (CI/CD)`)

4. **Reduced Code Debt**:
   - Integrated fixes directly into the generator
   - Created a direct template application approach
   - Added architecture-aware processing at the source

## Next Steps

1. Generate tests for all 300+ HuggingFace model architectures
2. Add specialized templates for speech and multimodal models
3. Integrate with CI/CD pipelines
4. Create comprehensive test coverage report
5. Implement nightly test runs

## License

Apache 2.0