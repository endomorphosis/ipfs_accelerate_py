# HuggingFace Model Test Generator Suite

A comprehensive framework for generating hardware-aware test files for HuggingFace Transformer models. The generator creates test files that detect available hardware, select appropriate model implementations, handle graceful degradation, and support specialized model architectures.

## Key Features

- **Hardware-Aware Testing**: Automatically detects and utilizes available hardware (CPU, CUDA, ROCm, MPS, OpenVINO, Qualcomm)
- **ROCm Support**: Full support for AMD GPUs via both the HIP API and CUDA compatibility layer
- **Architecture-Specific Templates**: Specialized templates for different model architectures
- **Graceful Degradation**: Generates code with robust fallback mechanisms
- **Mock Implementations**: Creates mock implementations when hardware is unavailable
- **Syntax Validation**: Validates and fixes syntax in generated code
- **Batch Generation**: Efficiently generates tests for multiple models

## Supported Hardware Backends

- **CPU**: Default fallback for all models
- **CUDA**: NVIDIA GPU support with half-precision optimization
- **ROCm**: AMD GPU support via both HIP API and CUDA compatibility layer
- **OpenVINO**: Intel hardware acceleration
- **MPS**: Apple Silicon (M1/M2/M3) acceleration
- **Qualcomm**: Qualcomm AI Engine support for mobile/edge devices

## Supported Model Architectures

- **Encoder-Only**: BERT, RoBERTa, ALBERT, etc.
- **Decoder-Only**: GPT-2, GPT-J, LLaMA, etc.
- **Encoder-Decoder**: T5, BART, etc.
- **Vision**: ViT, DeiT, etc.
- **Vision-Text**: CLIP, BLIP, etc.
- **Speech**: Wav2Vec2, Whisper, etc.
- **Multimodal**: FLAVA, etc.
- **RAG**: Retrieval-Augmented Generation models
- **Graph Neural Networks**: Graphormer, etc.
- **Time Series**: PatchTST, Informer, etc.
- **Object Detection**: DETR, Mask2Former, etc.
- **Mixture of Experts**: Mixtral, etc.
- **State Space Models**: Mamba, etc.

## ROCm Support Details

The suite includes comprehensive support for AMD GPUs through ROCm (Radeon Open Compute), including:

1. **Dual-path detection**: 
   - Via the dedicated HIP API (`torch.hip.is_available()`)
   - Via CUDA compatibility layer (detecting AMD GPUs in CUDA device names)

2. **Environment variable support**:
   - `HIP_VISIBLE_DEVICES` for controlling visible AMD GPUs
   - Fallback to `CUDA_VISIBLE_DEVICES` when using the CUDA compatibility layer

3. **Half-precision optimization**:
   - Tests for half-precision support on AMD GPUs
   - Graceful fallback to full precision when not supported

4. **Memory management**:
   - Reports available GPU memory
   - Implements proper VRAM cleanup after operations

5. **AMD-specific error handling**:
   - Robust error handling for AMD-specific failure modes
   - Graceful degradation to CPU when necessary

## Usage

### Basic Usage

```bash
# Generate a test file for a specific model
python test_generator_suite.py --model bert --output ./tests/test_bert.py

# Generate tests for all encoder-only models
python test_generator_suite.py --batch --architecture encoder-only --output-dir ./generated_tests/

# Generate tests from a list of models in a file
python test_generator_suite.py --batch --batch-file models.txt --output-dir ./generated_tests/

# Generate with a report
python test_generator_suite.py --model bert --report --report-file bert_report.md
```

### Example Script

The included `test_generator_example.py` demonstrates how to use the generator programmatically:

```bash
python test_generator_example.py
```

### Verifying ROCm Support

To verify ROCm detection and support:

```bash
# Check for ROCm environment
python test_rocm_detection.py

# Test loading a model on ROCm
python test_rocm_detection.py --run-model

# Verify ROCm support in all templates
python verify_templates.py
```

## Repository Structure

```
refactored_generator_suite/
├── test_generator_suite.py      # Main generator class
├── test_generator_example.py    # Example usage script
├── test_rocm_detection.py       # ROCm testing utility
├── verify_templates.py          # Template verification tool
├── generator_core/              # Core generator components
│   ├── generator.py             # Base generator class
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration management
│   └── registry.py              # Component registry
├── generators/                  # Model generators
│   ├── model_generator.py       # Model-specific generator
│   └── reference_model_generator.py # Reference model generator
├── templates/                   # Template files
│   ├── base.py                  # Base template
│   ├── encoder_only_template.py # Encoder-only template
│   ├── decoder_only_template.py # Decoder-only template
│   ├── hf_reference_template.py # Reference template
│   ├── rocm_hardware.py         # ROCm hardware template
│   └── ...                      # Other architecture templates
├── hardware/                    # Hardware detection
│   └── hardware_detection.py    # Hardware detector
├── model_selection/             # Model selection
│   ├── registry.py              # Model registry
│   └── selector.py              # Model selector
└── syntax/                      # Syntax validation and fixing
    ├── validator.py             # Syntax validator
    └── fixer.py                 # Syntax fixer
```

## Integration with Existing Pipeline

The generator suite integrates with the existing pipeline:

1. Templates are designed to work with the existing template composer
2. Hardware detection is compatible with the existing hardware detector
3. Generated files follow the same conventions as existing test files
4. ROCm support is implemented consistently across all templates

## Extending the Generator

To add support for a new model architecture:

1. Create a new template file in the `templates/` directory
2. Add the architecture to the `ARCHITECTURE_TYPES` list in `test_generator_suite.py`
3. Update the architecture mapping in the `get_model_info` method

To add support for a new hardware backend:

1. Add the hardware type to the `HARDWARE_BACKENDS` list
2. Implement detection logic in the `detect_hardware` method
3. Create the hardware template in the `templates/` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.