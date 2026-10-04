# Test Generator with Fixed Templates

## Overview

This directory contains a set of tools for generating test files for HuggingFace models. The original test generator had issues with indentation and template handling, particularly for hyphenated model names like `gpt-j` and `xlm-roberta`. These issues have been fixed in the new implementation, and the generator now integrates with the advanced model selection system and architecture-specific templates.

## Latest Improvements (March 21, 2025)

1. **Complete Template Overhaul**: All templates have been fixed and standardized with proper indentation, error handling, and consistent interfaces.
2. **Comprehensive Indentation Fixes**: Fixed indentation issues in all template files with special attention to:
   - Properly indented class methods with consistent spacing
   - Correctly nested code blocks in try/except blocks
   - Fixed OVT5Wrapper and MockSentencePieceProcessor class indentation
   - Properly aligned method bodies and nested statements
   - Consistent spacing between class and function definitions
   - Fixed vision_template.py with corrected hardware detection and mock classes
   - Fixed speech_template.py with proper spacing and consistent method indentation
   - Fixed multimodal_template.py with corrected indentation and proper method nesting
   - Repaired multi-line statement indentation in test methods across all templates
3. **Architecture-Specific Templates**: Added specialized templates for each model architecture (encoder-only, decoder-only, etc.).
4. **Enhanced Mock Detection**: Improved mock detection system with environment variable controls and clear indicators.
5. **Advanced Model Selection**: Integrated with the advanced model selection capabilities for task-specific and hardware-aware model selection.
6. **Hardware-Aware Testing**: Better support for various hardware configurations including CPU, CUDA, ROCm, MPS, OpenVINO, WebNN, and WebGPU.
7. **Standardized Result Collection**: Unified result collection and JSON output format across all templates.
8. **Graceful Dependency Handling**: Improved handling of missing dependencies with sophisticated fallbacks.

## Fixed Issues

1. **Template Indentation**: Fixed indentation issues in all template files, ensuring consistent 4-space indentation and proper code structure.
2. **Nested Block Alignment**: Ensured proper alignment of nested blocks, particularly in complex try/except blocks and for loops.
3. **Hyphenated Model Names**: Properly handled hyphenated model names to ensure valid Python identifiers and variable names.
4. **Task-Specific Input**: Added appropriate input text for different model types and tasks.
5. **Syntax Validation**: Added syntax validation to ensure generated files are valid Python code.
6. **Advanced Model Selection**: Integrated with the advanced model selection capabilities for task-specific and hardware-aware model selection.
7. **Template Selection Logic**: Added architecture-aware template selection based on model type.
8. **Class and Function Definition Spacing**: Ensured proper spacing between class methods and between standalone functions.

## Files

- `generate_simple_test.py`: A simplified test generator that creates properly indented test files with advanced model selection.
- `templates/minimal_bert_template.py`: A fixed template file for BERT-like models with proper indentation.
- `templates/encoder_only_template.py`: A fixed template for encoder-only models.
- `templates/decoder_only_template.py`: A fixed template for decoder-only models.
- `templates/encoder_decoder_template.py`: A fixed template for encoder-decoder models.
- `templates/vision_template.py`: A fixed template for vision models.
- `templates/vision_text_template.py`: A fixed template for vision-text multimodal models.
- `templates/speech_template.py`: A fixed template for speech models.
- `templates/multimodal_template.py`: A fixed template for multimodal models.

## Basic Usage

```
python generate_simple_test.py --model MODEL_TYPE [--output-dir OUTPUT_DIR] [--template TEMPLATE_PATH]
```

The script generates a test file for the specified model type using the appropriate template.

## Advanced Usage

The test generator now includes integrated advanced model selection with the following options:

```
python generate_simple_test.py --model MODEL_TYPE [--task TASK] [--hardware HARDWARE_PROFILE] [--max-size MAX_SIZE_MB] [--framework FRAMEWORK]
```

### Arguments

- `--model`: Model type (e.g., bert, gpt2, t5, vit)
- `--task`: Specific task for model selection (e.g., fill-mask, text-generation)
- `--hardware`: Hardware profile for size constraints (e.g., cpu-small, gpu-medium)
- `--max-size`: Maximum model size in MB
- `--framework`: Framework compatibility (e.g., pytorch, tensorflow)

### List Available Options

```
# List available tasks
python generate_simple_test.py --list-tasks

# List available architecture types
python generate_simple_test.py --list-architectures
```

## Examples

### Basic Test Generation

```bash
# Generate a test for BERT with encoder-only template
python generate_simple_test.py --model bert --output-dir test_output

# Generate a test for GPT-2 with decoder-only template
python generate_simple_test.py --model gpt2 --output-dir test_output

# Generate a test for a hyphenated model name (gpt-j)
python generate_simple_test.py --model gpt-j --output-dir test_output

# Generate a test for T5 with encoder-decoder template
python generate_simple_test.py --model t5 --output-dir test_output

# Generate a test for Vision Transformer (ViT) with vision template
python generate_simple_test.py --model vit --output-dir test_output

# Generate a test for Whisper with speech template
python generate_simple_test.py --model whisper --output-dir test_output

# Generate a test for CLIP with vision-text template
python generate_simple_test.py --model clip --output-dir test_output
```

### Advanced Test Generation with Constraints

```bash
# Generate a test for BERT with task constraint
python generate_simple_test.py --model bert --task text-classification

# Generate a test for GPT-2 with hardware constraint (smaller model for CPU)
python generate_simple_test.py --model gpt2 --hardware cpu-small

# Generate a test for T5 with size constraint
python generate_simple_test.py --model t5 --max-size 500

# Generate a test for ViT with framework constraint
python generate_simple_test.py --model vit --framework pytorch

# Generate a test for Whisper with multiple constraints
python generate_simple_test.py --model whisper --task automatic-speech-recognition --hardware gpu-small
```

### Template Override Examples

```bash
# Explicitly use encoder_only_template.py for RoBERTa
python generate_simple_test.py --model roberta --template templates/encoder_only_template.py

# Explicitly use decoder_only_template.py for LLaMA
python generate_simple_test.py --model llama --template templates/decoder_only_template.py

# Explicitly use vision_template.py for Swin Transformer
python generate_simple_test.py --model swin --template templates/vision_template.py
```

### Advanced Testing Options

```bash
# Use CPU-only testing for all models
python generate_simple_test.py --model gpt2 --device cpu

# Test with all available hardware
python generate_simple_test.py --model bert --all-hardware

# Generate test with mock environment variables
MOCK_TRANSFORMERS=True python generate_simple_test.py --model bert

# Generate test with special input for a specific task
python generate_simple_test.py --model bert --task fill-mask --input "The [MASK] brown fox jumps over the lazy dog."

# Save test results to a specific location
python generate_simple_test.py --model t5 --save-results --output-dir my_results
```

### Architecture + Task Examples

```bash
# Encoder-only model for question answering
python generate_simple_test.py --model bert --task question-answering

# Decoder-only model for text generation
python generate_simple_test.py --model gpt2 --task text-generation

# Encoder-decoder model for translation
python generate_simple_test.py --model t5 --task translation

# Vision model for image classification
python generate_simple_test.py --model vit --task image-classification

# Speech model for speech recognition
python generate_simple_test.py --model whisper --task automatic-speech-recognition

# Multimodal model for zero-shot image classification
python generate_simple_test.py --model clip --task zero-shot-image-classification
```

## Hardware Profiles

The test generator includes pre-defined hardware profiles:

- `cpu-small`: Limited CPU environments (500 MB max)
- `cpu-medium`: Standard CPU environments (2 GB max)
- `cpu-large`: High-memory CPU environments (10 GB max)
- `gpu-small`: Entry-level GPUs (5 GB max)
- `gpu-medium`: Mid-range GPUs (15 GB max)
- `gpu-large`: High-end GPUs (50 GB max)

These profiles are used to select appropriate model sizes.

## Supported Tasks

The test generator supports a wide range of tasks including:

- `fill-mask`: Masked language modeling (BERT, RoBERTa)
- `text-generation`: Autoregressive text generation (GPT-2, LLaMA)
- `text2text-generation`: Encoder-decoder text generation (T5, BART)
- `image-classification`: Image classification (ViT, Swin)
- `automatic-speech-recognition`: Speech recognition (Whisper, Wav2Vec2)
- And many more (see `--list-tasks` for the full list)

## Architecture Types

Models are grouped by architecture type for template selection:

- `encoder-only`: BERT, RoBERTa, DistilBERT, etc.
- `decoder-only`: GPT-2, GPT-J, LLaMA, etc.
- `encoder-decoder`: T5, BART, PEGASUS, etc.
- `vision`: ViT, Swin, DeiT, etc.
- `speech`: Whisper, Wav2Vec2, HuBERT, etc.
- `multimodal`: CLIP, BLIP, LLaVA, etc.

## Integration with Advanced Model Selection

The test generator is now fully integrated with the advanced model selection capabilities from `advanced_model_selection.py`, providing:

- Task-specific model selection based on model capabilities
- Hardware-aware model selection to fit available resources
- Size-constrained selection for constrained environments
- Framework compatibility filtering (PyTorch, TensorFlow, etc.)
- Fallback mechanisms to ensure a valid model is always selected

The integration uses a tiered approach:
1. If advanced selection is available, it tries that first
2. If that fails, it falls back to hardware-specific presets
3. If that fails, it uses default models
4. As a final fallback, it generates a basic model name

## Recent Accomplishments (March 21, 2025)

1. ✅ **Fixed Template Indentation**: Corrected indentation issues in all template files
2. ✅ **Created Architecture-Specific Templates**: Developed specialized templates for each architecture type
3. ✅ **Enhanced Mock Detection**: Implemented comprehensive mock detection with environment variable controls
4. ✅ **Integrated Advanced Model Selection**: Connected the generator with the advanced model selection system
5. ✅ **Added Hardware Awareness**: Expanded hardware detection for various device types
6. ✅ **Implemented Robust Error Handling**: Added detailed error reporting across all templates
7. ✅ **Improved Task-Specific Input Generation**: Created appropriate inputs for different tasks

## Next Steps

1. **Create Specialized Class and Method Variations**: Add more specialized methods for different model variants within each architecture
2. **Enhance Multimodal Support**: Expand multimodal template with more sophisticated input handling for various input types
3. **Implement Framework-Specific Adaptations**: Add tailored code paths for PyTorch, TensorFlow, and JAX implementations
4. **Create Validation Test Suite**: Develop a comprehensive validation suite to ensure all templates work correctly
5. **Add Distributed Testing Support**: Integrate with the distributed testing framework for parallel execution
6. **Implement Performance Metrics Collection**: Enhance result collection with detailed performance metrics
7. **Add CI/CD Pipeline Integration Examples**: Create examples of how to integrate the generator in CI/CD pipelines
8. **Develop Visualization Tools**: Add tools to visualize test coverage and performance results
9. **Enhance Model Registry**: Further expand the model registries with detailed parameter information

## Long-Term Vision

The long-term goal for the template system is to create a comprehensive testing framework that can:

1. **Test Any HuggingFace Model**: Provide specialized templates for all model architectures
2. **Work in Any Environment**: From high-end GPUs to constrained edge devices
3. **Support Any Framework**: Work with PyTorch, TensorFlow, JAX, ONNX, and other frameworks
4. **Integrate with CI/CD Pipelines**: Seamlessly integrate with GitHub Actions, GitLab CI, and Jenkins
5. **Collect Comprehensive Metrics**: Report detailed performance and compatibility metrics
6. **Visualize Results**: Provide rich visualizations of test results and model performance
7. **Scale to Distributed Testing**: Support distributed testing across multiple machines and hardware types