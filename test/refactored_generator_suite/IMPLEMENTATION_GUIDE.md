# Implementation Guide for Refactored Generator Suite

This document outlines the key implementation details for the refactored generator suite, focusing on the gap between the current implementation and the target implementation.

## Current vs. Target Implementation

### Naming Conventions

**Current Implementation:**
- Files: `{model_name}_{device}_skillset.py` (e.g., `bert_cpu_skillset.py`)
- Classes: `{ModelName}Skillset` (e.g., `BertSkillset`)

**Target Implementation:**
- Files: `hf_{model_type}.py` (e.g., `hf_bert.py`)
- Classes: `hf_{model_type}` (e.g., `hf_bert`)

Where `{model_type}` is retrieved from the model's autoconfig/config.json, not derived from the model name.

### Architecture Detection

**Current Implementation:**
- Uses model name-based patterns to determine architecture type
- May not correctly identify specialized variants of architectures
- Does not use autoconfig information from HuggingFace models

**Target Implementation:**
- Uses the model's actual config.json or autoconfig information
- Accurately identifies architecture based on model type and configuration
- Follows the same architecture detection methodology as in the refactored_test_suite

### Destination Directory

**Current Implementation:**
- Outputs files to a local `skillsets/` directory
- Does not integrate directly with the main framework

**Target Implementation:**
- Should output files to `../ipfs_accelerate_py/worker/skillset/` directory
- Direct integration with the main framework's skillset structure

### Inference Implementations

**Current Implementation:**
- Generic inference implementations that may not account for model-specific requirements
- Lack of specialized methods for different architecture types

**Target Implementation:**
- Model-specific inference implementations informed by HuggingFace documentation
- Leverages helper functions documented in transformers_docs_build/transformers_docs_built
- Specialized handling for each architecture type (encoder-only, decoder-only, vision, etc.)

## Reference Implementation Insights

The reference implementation in `../ipfs_accelerate_py/worker/skillset` provides several key patterns that should be adopted:

1. **Consistent Hardware Support**: Each model supports the same set of hardware backends with consistent interface methods
2. **Error Handling**: Comprehensive error handling with appropriate feedback and fallbacks
3. **Dynamic Loading**: Models are loaded dynamically based on available hardware
4. **Mock Support**: Consistent mock implementations for testing
5. **Hardware Abstraction**: Clear separation between model logic and hardware-specific code
6. **API Standardization**: Consistent API methods across all model types

## Implementation Steps

To align with the target implementation:

1. Update file and class naming conventions in the generator templates
2. Enhance architecture detection to use HuggingFace autoconfig information
3. Modify inference implementations to use model-specific patterns from HuggingFace documentation
4. Update the output directory path to target the main framework's skillset directory
5. Ensure all templates consistently implement the required methods for each hardware backend
6. Validate generated files against the reference implementation to ensure compatibility

## Using HuggingFace Documentation

The HuggingFace transformers documentation available in:
- `transformers_docs_build`: Documentation build scripts and raw content
- `transformers_docs_built`: Built documentation output

Contains valuable information for template generation, including:

1. **Model-Specific Classes**: The correct class to use for each model type
2. **Input Formatting**: How to properly format inputs for different models
3. **Output Processing**: How to process and interpret model outputs
4. **Special Tokens**: Handling of special tokens for different tokenizers
5. **Helper Functions**: Utility functions for different model types

Refer to this documentation when creating templates for different architecture types to ensure accurate implementation.