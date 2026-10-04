# Template Integration Guide

## Overview

This guide documents the process of integrating manually created model tests with the template system in the IPFS Accelerate Python framework. The template system ensures consistency across test files, properly handling hardware detection, mock objects, and result collection.

## Template Integration Process

The template integration process consists of the following steps:

1. **Analysis**: Analyze existing manually created test files to identify missing components and potential issues
2. **Generation**: Generate new test files using the template system with model-specific customizations
3. **Verification**: Verify that the generated files have valid syntax and include all required components
4. **Application**: Apply the changes to the main codebase

## Implementation Details

### Model Configuration

Model-specific customization is defined in the `MODEL_CONFIG` dictionary in `model_template_fixes.py`. Each model entry includes:

- `architecture`: The architecture type (e.g., "vision-encoder-text-decoder", "speech", "encoder-decoder")
- `model_id`: The HuggingFace model ID
- `class_name`: The model class name
- `task`: The model task (e.g., "document-question-answering", "text-to-speech")
- `test_inputs`: Input data for testing
- `processor_class`: The processor class to use
- `source_file`: The original file to convert
- `custom_imports`: Additional imports needed
- `special_handling`: Model-specific code to include

### Architecture Types

Models are organized by architecture type, which determines the template to use:

- `encoder-only`: BERT, RoBERTa, etc.
- `decoder-only`: GPT-2, LLaMA, etc.
- `encoder-decoder`: T5, BART, etc.
- `vision`: ViT, DeiT, etc.
- `vision-encoder-text-decoder`: CLIP, BLIP, LayoutLMv2, etc.
- `speech`: Whisper, Wav2Vec2, etc.
- `multimodal`: LLaVA, FLAVA, etc.

### Template Customization

The template customization process performs the following operations:

1. Basic replacements (model name, class name)
2. Registry entry addition
3. Custom imports handling
4. Special handling code insertion (with proper indentation)
5. Test input updates
6. Processor class updates

### Special Handling

Special handling code requires proper indentation to ensure valid syntax. The approach used is:

1. Detect the indentation level of the surrounding code
2. Format the special handling code with proper indentation
3. Insert the properly indented code at the appropriate location

For models with specific requirements:

- **Vision models** (layoutlmv2, layoutlmv3): Add dummy image creation code
- **Speech models** (clvp, seamless_m4t_v2): Add dummy audio creation code

## Troubleshooting

Common issues encountered during template integration:

1. **Indentation Errors**: Ensure proper indentation in special handling code, especially for conditional statements and blocks
2. **Import Conflicts**: Handle duplicate imports and ensure proper organization
3. **Template Mismatches**: Use the correct template for each model architecture
4. **Syntax Errors**: Verify syntax for all generated files

## Available Commands

The `model_template_fixes.py` script provides several commands:

- `--list-models`: List all configured models
- `--verify-model MODEL`: Verify a specific model test file
- `--generate-model MODEL`: Generate a test file for a specific model
- `--generate-all`: Generate test files for all models
- `--generate-specific`: Generate test files for specific problematic models
- `--verify`: Verify generated test files
- `--apply`: Apply changes to architecture types

## Example Usage

```bash
# List all configured models
python model_template_fixes.py --list-models

# Generate and verify all model tests
python model_template_fixes.py --generate-all --verify

# Generate and verify a specific model test
python model_template_fixes.py --generate-model layoutlmv2 --verify

# Focus on problematic models
python model_template_fixes.py --generate-specific --verify
```

## Integration Workflow

The `template_integration_workflow.py` script orchestrates the entire integration process:

```bash
# Run the complete integration workflow
python template_integration_workflow.py

# Skip analysis step
python template_integration_workflow.py --skip-analysis

# Skip generation step
python template_integration_workflow.py --skip-generation

# Skip verification step
python template_integration_workflow.py --skip-verification

# Apply changes to architecture types
python template_integration_workflow.py --apply
```

## Advanced Use: Quick Fix Tool

For focused fixes on specific model types, the `fix_template_issues.py` script provides a more direct approach:

```bash
# Fix indentation issues for problematic models
python fix_template_issues.py
```

This script focuses on properly indenting special handling code for vision and speech models.

## Next Steps

1. Run comprehensive tests on the regenerated files to ensure full functionality
2. Update model registry to include all converted models
3. Update documentation to reflect the standardized template approach
4. Consider other manually created model tests that might benefit from the template system