# HuggingFace Test Generator

This README describes the test generator for HuggingFace models in the IPFS Accelerate Python Framework.

## Overview

The HuggingFace Test Generator is a tool for automatically generating standardized test files for HuggingFace models. The tests ensure comprehensive coverage across:

1. Multiple hardware backends (CPU, CUDA, OpenVINO)
2. Different API access methods (pipeline, from_pretrained)
3. Batch processing capabilities
4. Performance metrics collection

## Generator Scripts

The repository contains several test generation scripts:

1. **generate_remaining_hf_tests.py** - Our main script for generating new test files
   - Identifies missing tests
   - Generates test implementations for high-priority models
   - Updates documentation and status tracking files

2. **generate_model_tests.py** - Contains more detailed model-specific customizations
   - Currently has syntax issues that need to be addressed

3. **generate_comprehensive_tests.py** - Uses the ComprehensiveModelTester framework
   - Offers standardized testing across all hardware backends
   - Currently has syntax issues that need to be addressed

4. **generate_simple_test.py** - Basic script for generating simple test files
   - Currently has argument parsing issues

## Generating Additional Test Files

To generate more test files:

```bash
# Generate tests for high-priority models
python generate_remaining_hf_tests.py

# Options (modify the script to change these):
# - Generate non-priority models: Set priority_only=False in generate_test_files()
# - Adjust count: Change count parameter in generate_test_files()
```

## Current Implementation Status

Current test implementation progress:
- 299 total HuggingFace model types
- 136 implemented tests (45.5% coverage)
- 170 models remaining to implement

High-priority models for future implementation:
- udop
- vision-encoder-decoder
- chinese_clip
- glm
- focalnet
- convnextv2
- phi3
- fuyu
- paligemma
- grounding-dino

## Test Structure

Each generated test file follows the same structure:

1. **Module Import and Mock Implementation**:
   - Attempts to import the real implementation 
   - Creates a mock implementation if real one not available

2. **Test Infrastructure**:
   - Test input preparation
   - CPU/CUDA/OpenVINO implementation testing
   - Batch processing tests

3. **Result Handling**:
   - JSON result generation
   - Comparison with expected results
   - Result file storage

## Model Categories

Models are divided into four main categories:

1. **Language Models**:
   - text-generation, fill-mask, text-classification, etc.
   - Examples: BERT, T5, LLaMA, GPT2

2. **Vision Models**:
   - image-classification, object-detection, image-segmentation, etc.
   - Examples: ViT, CLIP, SegFormer, Swin

3. **Audio Models**:
   - automatic-speech-recognition, audio-classification, etc.
   - Examples: Whisper, Wav2Vec2, CLAP

4. **Multimodal Models**:
   - image-to-text, visual-question-answering, etc.
   - Examples: LLaVA, BLIP, Fuyu, Video-LLaVA

## How Tests Work

1. **Initialization**:
   - Tests try to import the model module from the codebase
   - Falls back to a mock implementation if not available

2. **Test Execution**:
   - Tests initialization on CPU
   - Tests initialization on CUDA (if available)
   - Tests initialization on OpenVINO (if available)
   - Tests batch processing capabilities

3. **Result Collection**:
   - Results are stored in JSON format
   - Includes implementation type (REAL vs. MOCK)
   - Stores performance metrics when available

## Improving the Generator

Areas for improvement:

1. **Fix syntax issues in other generator scripts**:
   - Resolve f-string nesting issues
   - Fix argument parsing problems

2. **Add model-specific configurations**:
   - Better default model selection per model type
   - Input parameter adaptation based on model type

3. **Categorize tests properly**:
   - Update the category counts in documentation
   - Properly classify each model by task and category

4. **Add auto-categorization of models**:
   - Implement automatic model category detection
   - Update statistics based on implementation