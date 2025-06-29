# Architecture-Specific Templates for HuggingFace Tests

> **HIGH PRIORITY OBJECTIVE:** Achieving 100% test coverage for all 300+ HuggingFace model classes with validated end-to-end testing is a critical priority. Current coverage is only 57.6% (114/198 tracked models).

This directory contains architecture-specific templates used by the test generator to create properly formatted test files for different model families. These templates have been enhanced with proper indentation, robust error handling, and comprehensive mock detection capabilities.

## CRITICAL: MODIFY TEMPLATES, NOT GENERATED FILES

> **CRITICAL RULE:** Always modify these templates and the generator, NEVER edit the generated test files directly.

Editing the templates in this directory ensures changes are properly propagated to all generated files. Directly editing generated test files will result in lost changes when tests are regenerated and create inconsistencies across the test suite.

## Latest Improvements (March 21, 2025)

Recent enhancements to the templates include:

1. **Proper Indentation**: All templates have been fixed to ensure consistent and correct indentation, with particular focus on:
   - Class method indentation with proper spacing
   - Consistent indentation in nested blocks (try/except, if/else, for loops)
   - Proper alignment of multi-line statements
   - Correct spacing between class and function definitions
   - Fixed OVT5Wrapper class indentation in encoder_decoder_template.py
   - Properly aligned MockSentencePieceProcessor class in all relevant templates
   - Fixed vision_template.py with corrected indentation throughout the file
   - Improved MockImage and MockRequests class indentation in vision_template.py
   - Fixed speech_template.py with proper spacing and indentation for all methods
   - Fixed multimodal_template.py with corrected indentation and proper method nesting

2. **Enhanced Error Reporting**: Better error classification and detailed error reporting.
3. **Robust Hardware Detection**: Improved detection of CUDA, ROCm, MPS, WebGPU, and WebNN.
4. **Comprehensive Mock Detection**: Advanced detection of mock objects vs real inference.
5. **Environment Variable Controls**: Added environment variable control for mocking dependencies.
6. **Hardware-Aware Testing**: Improved testing on different hardware configurations.
7. **Expanded Model Registries**: More comprehensive model registries for each architecture type.
8. **Task-Specific Input Handling**: Better generation of appropriate test inputs for each task type.
9. **Unified Result Reporting**: Standardized result collection across all architecture types.

## Available Templates

The following templates are available:

1. **encoder_only_template.py** - For encoder-only models (BERT, RoBERTa, etc.)
2. **decoder_only_template.py** - For decoder-only models (GPT-2, LLaMA, etc.)
3. **encoder_decoder_template.py** - For encoder-decoder models (T5, BART, etc.)
4. **vision_template.py** - For vision models (ViT, Swin, etc.)
5. **vision_text_template.py** - For vision-text models (CLIP, BLIP, etc.)
6. **speech_template.py** - For speech models (Whisper, Wav2Vec2, etc.)
7. **multimodal_template.py** - For multimodal models (LLaVA, etc.)
8. **minimal_bert_template.py** - Simplified template for quick BERT model testing
9. **minimal_vision_template.py** - Simplified template for quick vision model testing

## Architecture-Specific Features

Each template implements architecture-specific features:

### Encoder-Only Models

Template: `encoder_only_template.py`

Special handling for:
- Bidirectional attention patterns
- Mask token handling for masked language modeling
- Token prediction extraction
- Consistent tokenizer interface
- Compatible with BERT, RoBERTa, Albert, DistilBERT, Electra, etc.

### Decoder-Only Models

Template: `decoder_only_template.py`

Special handling for:
- Autoregressive behavior
- Padding token configuration (often setting pad_token = eos_token)
- Causal attention patterns
- Text generation capabilities
- Compatible with GPT-2, LLaMA, Mistral, Falcon, etc.

### Encoder-Decoder Models

Template: `encoder_decoder_template.py`

Special handling for:
- Separate encoder and decoder components
- Decoder input initialization
- Sequence-to-sequence capabilities
- Translation task handling
- Compatible with T5, BART, Pegasus, mBART, etc.

### Vision Models

Template: `vision_template.py`

Special handling for:
- Image preprocessing with proper tensor shapes
- Image processor instead of tokenizer
- Pixel values handling
- Classification task implementation
- Compatible with ViT, Swin, DeiT, BEiT, ConvNeXT, etc.

### Vision-Text Models

Template: `vision_text_template.py`

Special handling for:
- Dual-stream architecture for images and text
- Combined image-text processing
- Contrastive learning implementations
- Cross-modal alignment
- Compatible with CLIP, BLIP, etc.

### Speech Models

Template: `speech_template.py`

Special handling for:
- Audio preprocessing and feature extraction
- Mel spectrogram conversion
- Automatic speech recognition task handling
- Audio processor configuration
- Synthetic audio generation for testing without audio files
- Sampling rate handling for different audio models
- Audio file detection with multiple format support
- Hardware-specific audio processing
- Compatible with Whisper, Wav2Vec2, HuBERT, EnCodec, MusicGen, SEW, etc.

### Multimodal Models

Template: `multimodal_template.py`

Special handling for:
- Multiple input modalities
- Cross-modal attention mechanisms
- Complex input processing pipelines
- Multimodal task handling
- Compatible with LLaVA, Video-LLaVA, etc.

## Core Features in All Templates

Each template includes the following core features:

1. **Hardware Detection**: Automatic detection of CPU, CUDA, MPS, OpenVINO, WebNN, and WebGPU.
2. **Dependency Management**: Graceful handling of missing dependencies with mock objects.
3. **Mock Detection System**: Clear indicators (🚀 vs. 🔷) for real inference vs. mock objects with dependency reporting.
4. **Model Registry**: Architecture-specific model registry with default configurations.
5. **Test Class Implementation**: Class-based testing framework with architecture-specific methods.
6. **Pipeline Testing**: Tests using the high-level `pipeline()` API.
7. **From_pretrained Testing**: Tests using the lower-level `from_pretrained()` API.
8. **Hardware-specific Testing**: Tests for specialized hardware like OpenVINO.
9. **Command-line Interface**: Flexible CLI with options for model selection, hardware selection, etc.
10. **Result Collection**: Structured output in JSON format with detailed metadata.

## Mock Detection Implementation

All templates implement the mock detection system which:

1. **Detects Dependencies**: Checks for key dependencies (transformers, torch, tokenizers, sentencepiece).
2. **Determines Inference Type**: Uses dependency status to determine if real inference is possible.
3. **Adds Visual Indicators**:
   - 🚀 for real inference with actual models
   - 🔷 for mock objects in CI/CD testing
4. **Adds Metadata**: Enriches the result JSON with detailed environment information:
   ```python
   {
       "metadata": {
           # ... existing fields ...
           "has_transformers": HAS_TRANSFORMERS,
           "has_torch": HAS_TORCH,
           "has_tokenizers": HAS_TOKENIZERS, 
           "has_sentencepiece": HAS_SENTENCEPIECE,
           "using_real_inference": using_real_inference,
           "using_mocks": using_mocks,
           "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
       }
   }
   ```

For more details on the mock detection system, see [MOCK_DETECTION_README.md](../../MOCK_DETECTION_README.md).

## Usage

These templates are used by the test generator to create test files. Several scripts can leverage these templates:

### Using regenerate_fixed_tests.py

```bash
python regenerate_fixed_tests.py --model bert --verify  # Uses encoder_only_template.py
python regenerate_fixed_tests.py --model gpt2 --verify  # Uses decoder_only_template.py
python regenerate_fixed_tests.py --model t5 --verify    # Uses encoder_decoder_template.py
python regenerate_fixed_tests.py --model vit --verify   # Uses vision_template.py
```

### Using generate_simple_test.py with Advanced Options

```bash
# Basic usage
python generate_simple_test.py --model-type bert

# With hardware constraints
python generate_simple_test.py --model-type gpt2 --hardware-profile gpu-small

# With task specification
python generate_simple_test.py --model-type t5 --task translation

# With size constraints
python generate_simple_test.py --model-type vit --max-size-mb 500

# With framework constraints
python generate_simple_test.py --model-type bert --framework pytorch

# Combining multiple constraints
python generate_simple_test.py --model-type bert --hardware-profile cpu-small --task fill-mask
```

### Architecture-Specific Usage Examples

#### For Encoder-Only Models
```bash
python generate_simple_test.py --model-type roberta --template encoder_only_template.py
```

#### For Decoder-Only Models
```bash
python generate_simple_test.py --model-type gpt-j --template decoder_only_template.py
```

#### For Vision Models
```bash
python generate_simple_test.py --model-type vit --template vision_template.py
```

#### For Speech Models
```bash
python generate_simple_test.py --model-type whisper --template speech_template.py
```

## Extending the Templates

To add a new architecture type or enhance existing templates:

### Adding a New Architecture Type

1. Create a new template file (e.g., `new_architecture_template.py`) based on the closest existing architecture
2. Update `ARCHITECTURE_TYPES` in `test_generator_fixed.py` to include the new architecture
3. Update `get_template_for_architecture()` to map model types to the new template
4. Add an architecture-specific model registry with default configurations
5. Implement architecture-specific test methods and input processing
6. Add a class-based testing framework with appropriate methods
7. Ensure all indentation is consistent throughout the file
8. Add comprehensive error handling and mock detection
9. Verify the template with the Python compiler

### Enhancing Existing Templates

When enhancing templates, follow these guidelines:

1. **Keep Indentation Consistent**: Ensure all indentation is well-formed and consistent
2. **Augment Mock Detection**: Enhance the mock detection system with new dependencies
3. **Extend Hardware Support**: Add support for new hardware types as they become available
4. **Expand Model Registries**: Add more models to the architecture-specific registries
5. **Add Task-Specific Handlers**: Implement specialized handling for specific tasks
6. **Maintain Backward Compatibility**: Ensure existing test scripts continue to work
7. **Document Changes**: Update this README with new features and capabilities

### Required Components for Each Template

All templates must include:

1. Hardware detection with fallbacks
2. Mock detection for all required dependencies
3. Environment variable controls for mocking
4. Architecture-specific model registry
5. Class-based test implementation
6. Pipeline and from_pretrained test methods
7. Command-line interface
8. Unified result collection
9. Consistent error handling and reporting

### Indentation and Formatting Guidelines

To maintain consistent and correct indentation across all templates, follow these rules:

1. **Class Definitions**:
   - Use 4-space indentation for class methods 
   - Add a blank line between method definitions
   - Ensure proper docstrings for each class and method

2. **Function Definitions**:
   - Outside of classes, functions should not be indented
   - Add a blank line between function definitions
   - Include proper docstrings

3. **Nested Blocks**:
   - Consistently use 4-space indentation for nested blocks
   - Ensure proper alignment of multi-line statements
   - Keep consistent indentation in try/except blocks

4. **Special Python Constructs**:
   - Ensure proper indentation for nested class definitions
   - Be careful with indentation in complex try/except blocks
   - Pay attention to list comprehensions and multi-line expressions

5. **Mock Object Definitions**:
   - Keep consistent indentation in mock class implementations
   - Ensure proper method indentation within mock classes

All templates should pass a syntax check using Python's built-in compiler:
```bash
python -m py_compile templates/encoder_only_template.py
```

## Template Verification and Testing

### Syntax Verification

All templates should be verified using the Python compiler to ensure they have valid syntax:

```bash
python -m py_compile templates/encoder_only_template.py
```

### Generated File Verification

The test generator includes a verification step to ensure that generated files have valid syntax:

```bash
python generate_simple_test.py --model-type bert --verify
```

### Manual Testing

To thoroughly test a template, run it with various models within its architecture type:

```bash
# For encoder-only template
python -m templates.encoder_only_template --model bert-base-uncased
python -m templates.encoder_only_template --model roberta-base

# For decoder-only template
python -m templates.decoder_only_template --model gpt2
python -m templates.decoder_only_template --model gpt2-medium

# For vision template
python -m templates.vision_template --model google/vit-base-patch16-224
```

### Dependency Testing

Test templates with various dependency configurations:

```bash
# With all dependencies
python -m templates.encoder_only_template --model bert-base-uncased

# Without transformers
MOCK_TRANSFORMERS=True python -m templates.encoder_only_template --model bert-base-uncased

# Without torch
MOCK_TORCH=True python -m templates.encoder_only_template --model bert-base-uncased

# Without any dependencies
MOCK_TRANSFORMERS=True MOCK_TORCH=True python -m templates.encoder_only_template --model bert-base-uncased
```
