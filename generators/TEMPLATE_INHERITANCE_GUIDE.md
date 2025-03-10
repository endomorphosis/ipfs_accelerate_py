# Template Inheritance System Guide (Updated March 2025)

## Overview

The Template Inheritance System provides a powerful way to create specialized model templates that share common functionality while allowing for customization. This guide explains how to use and extend the template system for creating model implementations.

> **Update March 2025**: The template system has been enhanced with comprehensive validation capabilities, including hardware platform compatibility checks, generator compatibility validation, and template variable verification. See the [Template Validation Guide](TEMPLATE_VALIDATION_GUIDE.md) for detailed information on the new validation features.

## Key Components

1. **Model Template Registry** (`model_template_registry.py`): Central registry for all templates with inheritance resolution
2. **Base Template** (`hf_template.py`): Foundation template with core functionality for all models
3. **Family Templates**: Templates specialized for model families (embedding, text generation, etc.)
4. **Model Templates**: Templates for specific model architectures (BERT, T5, etc.)
5. **Template Verifier** (`template_verifier.py`): Tool for validating templates and their inheritance

## Template Structure

Each template consists of several key elements:

1. **Metadata**: Information about the template, its requirements, and capabilities
2. **Inheritance Declaration**: Which template it inherits from
3. **Sections**: Code blocks for different parts of the implementation

### Template Metadata

```python
# Template metadata
TEMPLATE_VERSION = "1.0.0"
TEMPLATE_DESCRIPTION = "Base template for all Hugging Face model implementations"
INHERITS_FROM = None  # Base template has no parent
SUPPORTS_HARDWARE = ["cpu", "cuda", "rocm", "mps"]  # Hardware types supported
COMPATIBLE_FAMILIES = []  # Compatible with all model families
MODEL_REQUIREMENTS = {}  # No specific model requirements
```

### Template Sections

Templates are divided into sections, which can be overridden by child templates:

```python
SECTION_IMPORTS = """import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
...
"""

SECTION_CLASS_DEFINITION = """class hf_{{ model_name }}:
    """{{ model_description }}"""
    
    # Model metadata
    MODEL_TYPE = "{{ model_type }}"
    MODEL_NAME = "{{ model_name }}"
    ...
"""
```

## Inheritance Mechanism

Templates inherit from parent templates and can override specific sections while keeping others. The inheritance chain is resolved at rendering time.

### Example: Embedding Model Template

```python
# Template metadata
TEMPLATE_VERSION = "1.0.0"
TEMPLATE_DESCRIPTION = "Template for embedding models like BERT"
INHERITS_FROM = "hf_template.py"  # Inherits from base template
SUPPORTS_HARDWARE = ["cpu", "cuda", "rocm", "mps"]
COMPATIBLE_FAMILIES = ["embedding"]  # Only compatible with embedding models
MODEL_REQUIREMENTS = {"embedding_dim": [768, 1024]}  # Common embedding dimensions

# Override just the methods section to specialize for embedding models
SECTION_METHODS = """    def encode(self, text, **kwargs):
        """
        Encode text input to embeddings with special handling for embedding models.
        
        Args:
            text: The text to encode
            **kwargs: Additional keyword arguments for encoding
            
        Returns:
            Embeddings for the input text
        """
        # Embedding-specific implementation
        ...
"""
```

## Using Templates

### Selecting Templates

The system can automatically select the best template for a model based on:

1. Model family
2. Available hardware
3. Specific model requirements

```python
from templates.model_template_registry import select_template

# Select template for embedding model
template_name = select_template(
    model_family="embedding",
    hardware_info={"cuda": True, "mps": False},
    model_requirements={"embedding_dim": 768}
)
```

### Rendering Templates

Templates are rendered with a context containing variables:

```python
from templates.model_template_registry import render_template

# Context with variables for template rendering
context = {
    "model_name": "bert",
    "model_type": "embedding",
    "model_description": "BERT model for text embeddings",
    "modality": "text",
    "supports_quantization": "True",
    "requires_gpu": "False"
}

# Render the template with the context
implementation = render_template(template_name, context)
```

## Creating New Templates

### Creating a Family Template

1. Create a new file with a descriptive name (e.g., `hf_vision_template.py`)
2. Specify the parent template in `INHERITS_FROM`
3. Define compatible model families in `COMPATIBLE_FAMILIES`
4. Override only the sections that need customization

### Creating a Model-Specific Template

1. Create a new file with a model-specific name (e.g., `hf_bert_template.py`)
2. Inherit from the appropriate family template
3. Make the template highly specific to the model architecture
4. Provide default values in the template for model-specific parameters

## Template Variables

Templates support variable substitution with the format `{{ variable_name }}`:

1. **Basic Variables**: `{{ model_name }}`, `{{ model_type }}`, etc.
2. **Nested Variables**: `{{ hardware.cuda.available }}`
3. **Control Flow**: Template renderer handles basic variable substitution

## Testing Templates

The Template Verifier tool can be used to validate templates:

```bash
python templates/template_verifier.py --template hf_embedding_template.py --test-family embedding
```

It can also generate comprehensive reports on template compatibility:

```bash
python templates/template_verifier.py --report --output template_report.json
```

## Best Practices

1. **Minimize Duplication**: Inherit from the most specific appropriate template
2. **Section Granularity**: Create well-defined sections that are easily overridable
3. **Hardware Awareness**: Specify hardware compatibility in template metadata
4. **Documentation**: Include clear docstrings and comments in your templates
5. **Testing**: Verify template rendering with different contexts
6. **Versioning**: Update the `TEMPLATE_VERSION` when making significant changes

## Real-world Examples

### Basic Usage: Embedding Model

```python
# Select template for BERT
template_name = select_template("embedding")

# Context with BERT-specific values
context = {
    "model_name": "bert-base-uncased",
    "model_type": "bert",
    "model_description": "BERT base uncased model for text embeddings",
    "modality": "text",
    "supports_quantization": "True",
    "requires_gpu": "False"
}

# Render implementation
implementation = render_template(template_name, context)
```

### Hardware-Specific Selection

```python
# Get hardware info
from hardware_detection import detect_available_hardware
hardware_info = detect_available_hardware()

# Select template appropriate for the hardware
template_name = select_template("text_generation", hardware_info)
```

## Extending the System

The template inheritance system can be extended in several ways:

1. **Add New Metadata Fields**: Enhance template selection with more metadata
2. **Create Domain-Specific Templates**: For specialized domains or applications
3. **Add Rendering Features**: Enhance the template renderer with more features
4. **Integration with CI/CD**: Automate template verification and testing
5. **Template Generators**: Create tools to generate templates from existing code

## Troubleshooting

### Common Issues

1. **Unresolved Variables**: Check that all template variables are provided in the context
2. **Inheritance Chain Errors**: Ensure the parent template exists and is properly referenced
3. **Conflicting Sections**: Check for section naming conflicts in the inheritance chain
4. **Hardware Compatibility**: Verify hardware compatibility metadata matches implementation

### Using the Verifier

The template verifier can help diagnose issues:

```bash
python templates/template_verifier.py --template problematic_template.py --verbose
```

## Next Steps

- Create specialized templates for more model families
- Integrate with model registry for automatic template selection
- Add template validation in CI/CD pipeline
- Develop template migration tools for version upgrades
