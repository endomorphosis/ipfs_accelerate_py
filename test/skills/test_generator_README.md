# Hugging Face Model Testing Framework

This directory contains test files and tools for comprehensive testing of Hugging Face models with the IPFS Accelerate Python Framework.

## Overview

The testing framework provides:

1. **Complete Model Ecosystem Coverage**: Tests for every major Hugging Face model architecture across all domains (language, vision, audio, multimodal, biomedical, etc.), covering all 76 model families and 169+ model variants.

2. **Multiple Hardware Backends**: Support for testing on CPU, CUDA, and OpenVINO hardware backends for all models.

3. **Dual API Approach Testing**: Testing for both `pipeline()` and `from_pretrained()` approaches with consistent validation.

4. **Comprehensive Performance Benchmarking**: Collection of inference time, memory usage, load time, and other performance metrics for every model.

5. **Fully Automated Test Infrastructure**: Tools to automatically discover, generate, and run tests for new model architectures as they are released.

## Test Generator

The `test_generator.py` script is a powerful tool for generating test files for different model families. It includes:

- A comprehensive registry of all model families with their configurations
- Templates for generating standardized test files for any model architecture
- Automatic model discovery and suggestion from Hugging Face Hub
- Batch processing of multiple models simultaneously
- Hardware detection and compatibility checks across devices

### Basic Usage

```bash
# List available model families (76 total)
python generators/models/test_generator.py --list-families

# Generate a test file for a specific model family
python generators/models/test_generator.py --generate bert

# Generate test files for all model families
python generators/models/test_generator.py --all

# Generate test files for a specific set of models
python generators/models/test_generator.py --batch-generate bert,gpt2,t5,vit,clip
```

### Advanced Features

```bash
# Discover and suggest new models to add
python generators/models/test_generator.py --suggest-models

# Generate a registry entry for a specific model
python generators/models/test_generator.py --generate-registry-entry sam

# Automatically discover and add new models to registry
python generators/models/test_generator.py --auto-add --max-models 5

# Update test_all_models.py with all current model families
python generators/models/test_generator.py --update-all-models

# Scan the transformers library for available models
python generators/models/test_generator.py --scan-transformers
```

## Running Tests

The generated test files can be run directly to test specific model families:

```bash
# Test a specific model family
python generators/models/test_hf_bert.py

# Test all hardware backends (CPU, CUDA, OpenVINO)
python generators/models/test_hf_bert.py --all-hardware

# Test a specific model
python generators/models/test_hf_bert.py --model bert-base-uncased

# Save detailed test results
python generators/models/test_hf_bert.py --save --output-dir collected_results
```

For running multiple tests at once, use the `test_all_models.py` script:

```bash
# Run tests for all models
python generators/models/test_all_models.py

# Run tests with all hardware
python generators/models/test_all_models.py --all-hardware

# Run tests for specific categories
python generators/models/test_all_models.py --categories language,vision,audio
```

## Test Results

Test results are stored in the `collected_results` directory as JSON files, which include:

- Success/failure information for each test
- Detailed performance metrics across hardware
- Hardware compatibility and acceleration support
- Model-specific information and metadata
- Example input/output pairs for verification

A comprehensive summary of test results is available in `test_report.md`.

## Adding New Models

To add a new model family to the testing framework:

1. Use the `--suggest-models` option to see available models
2. Use `--generate-registry-entry [model]` to generate a registry entry
3. Add the entry to the `MODEL_REGISTRY` in `test_generator.py`
4. Generate the test file with `--generate [model]`
5. Run the test to verify it works

## Supported Model Categories

The framework provides comprehensive coverage across all model domains:

### Language Models
- **Foundation LLMs**: BERT, GPT2, T5, RoBERTa, DistilBERT, LLaMA, Mistral, Phi (1/3/4), Mixtral, Gemma, Qwen2, DeepSeek, BART
- **Recent LLMs**: Command-R, Orca3, Claude3-Haiku, TinyLlama
- **State Space Models**: Mamba, Mamba2, RWKV
- **Code Models**: CodeLlama, StarCoder2

### Domain-Specific Models
- **Biomedical**: BioGPT, ESM (protein models)
- **Graph Neural Networks**: GraphSAGE

### Vision Models
- **Classification/Backbone**: ViT, Swin, ConvNeXT, DINOv2
- **Object Detection**: DETR, Grounding-DINO, OWL-ViT
- **Segmentation**: SegFormer, SAM
- **Depth Estimation**: ZoeDepth, Depth-Anything
- **Generative**: VQGAN
- **3D Understanding**: ULIP (point cloud)

### Vision-Language Models
- **CLIP Family**: CLIP, X-CLIP, SigLIP
- **BLIP Family**: BLIP, BLIP-2, InstructBLIP
- **Multimodal LLMs**: LLaVA, VisualBERT, PaLI-GEMMA, KOSMOS-2, ViLT, Qwen2-VL, CogVLM2
- **Advanced VLMs**: IDEFICS3, CM3
- **Video Understanding**: Video-LLaVA, LLaVA-NeXT-Video
- **Cross-Modal**: ImageBind

### Document Understanding
- **Layout Analysis**: LayoutLM (v2, v3), Donut
- **Markup Processing**: MarkupLM

### Audio & Speech Models
- **Speech Recognition**: Whisper, Wav2Vec2, Wav2Vec2-BERT, HuBERT, WavLM
- **Audio Generation**: MusicGen, EnCodec, Bark, AudioLDM2
- **Voice Processing**: Qwen2-Audio, Qwen2-Audio-Encoder, CLAP

### Time Series Models
- Time Series Transformer

## Hardware Support

All tests support the following hardware backends:

- **CPU**: Full support for all models with performance benchmarks
- **CUDA**: GPU acceleration with memory tracking and throughput measurement
- **OpenVINO**: Hardware-optimized inference with compatibility testing

## Continuous Updates

The testing framework is designed to stay current with the Hugging Face ecosystem:

- Regular model discovery to identify new architectures
- Auto-generation of tests for emerging model types
- Performance tracking across hardware generations
- Consistent API updates to match the transformers library

See the full comprehensive list of 76 model families by running `--list-families`.