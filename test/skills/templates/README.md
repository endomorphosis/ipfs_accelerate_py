# Architecture-Specific Templates for HuggingFace Tests

This directory contains architecture-specific templates used by the test generator to create properly formatted test files for different model families.

## Available Templates

The following templates are available:

1. **encoder_only_template.py** - For encoder-only models (BERT, RoBERTa, etc.)
2. **decoder_only_template.py** - For decoder-only models (GPT-2, LLaMA, etc.)
3. **encoder_decoder_template.py** - For encoder-decoder models (T5, BART, etc.)
4. **vision_template.py** - For vision models (ViT, Swin, etc.)
5. **vision_text_template.py** - For vision-text models (CLIP, BLIP, etc.)
6. **speech_template.py** - For speech models (Whisper, Wav2Vec2, etc.)
7. **multimodal_template.py** - For multimodal models (LLaVA, etc.)

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
- Compatible with Whisper, Wav2Vec2, HuBERT, etc.

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

These templates are used by the test generator to create test files:

```bash
python regenerate_fixed_tests.py --model bert --verify  # Uses encoder_only_template.py
python regenerate_fixed_tests.py --model gpt2 --verify  # Uses decoder_only_template.py
python regenerate_fixed_tests.py --model t5 --verify    # Uses encoder_decoder_template.py
python regenerate_fixed_tests.py --model vit --verify   # Uses vision_template.py
```

## Extending the Templates

To add a new architecture type:

1. Create a new template file (e.g., `new_architecture_template.py`)
2. Update `ARCHITECTURE_TYPES` in `test_generator_fixed.py` to include the new architecture
3. Update `get_template_for_architecture()` to map to the new template
4. Add any architecture-specific methods and configurations to the template

## Template Verification

All templates are verified using the Python compiler to ensure they have valid syntax:

```bash
python -m py_compile templates/encoder_only_template.py
```

The test generator also includes a verification step to ensure that generated files have valid syntax.
