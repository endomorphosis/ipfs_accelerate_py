# Template Database Guide

## Overview

The Template Database system provides a way to store, retrieve, and manage test templates for different models and hardware platforms. This guide explains how to use and integrate with the Template Database.

## Features

- **Centralized Template Storage**: Store all test templates in a single DuckDB database
- **Model-Specific Templates**: Templates customized for specific model types
- **Platform-Specific Templates**: Hardware-specific test implementations
- **Template Inheritance**: Base templates can be extended for specific needs
- **Template Validation**: Tools to validate template integrity
- **Template Variables**: Support for variable substitution in templates

## Basic Usage

### Creating a Template Database

Use the `create_simple_template_db.py` script to create a new template database:

```bash
python create_simple_template_db.py
```

This will create a database with basic templates for text and vision models.

### Using Templates with the Generator

Use the enhanced `simple_test_generator.py` to generate tests using templates:

```bash
# Generate a test for bert using templates
python simple_test_generator.py -g bert -t

# Generate a test for vit with specific platforms
python simple_test_generator.py -g vit -p cuda,webgpu -t

# List available templates
python simple_test_generator.py --list-templates
```

### Detecting Available Hardware

The generator can detect available hardware platforms:

```bash
python simple_test_generator.py --detect-hardware
```

## Template Database Structure

The template database consists of several tables:

- **templates**: Stores the actual templates
  - id: Primary key
  - model_type: Type of model (bert, vision, etc.)
  - template_type: Type of template (test, base, etc.)
  - platform: Optional platform (cuda, webgpu, etc.)
  - template: The actual template content
  - created_at: Timestamp when the template was created
  - updated_at: Timestamp when the template was last updated

- **template_helpers**: Stores helper functions for templates
  - id: Primary key
  - helper_name: Name of the helper function
  - helper_code: The helper function code
  - created_at: Timestamp when the helper was created
  - updated_at: Timestamp when the helper was last updated

Other tables include template_versions, template_dependencies, template_variables, and template_validation.

## Template Variables

Templates support several variables that can be replaced when generating tests:

- `{{model_name}}`: Name of the model
- `{{model_category}}`: Category of the model (text, vision, etc.)
- `{{model_name.replace("-", "").capitalize()}}`: Capitalized class name

Example template with variables:

```python
class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    def setUp(self):
        self.model_name = "{{model_name}}"
        # ...
```

## Hardware Support

The template system and generator support a wide range of hardware platforms:

- **CPU**: Basic CPU execution
- **CUDA**: NVIDIA GPU acceleration
- **ROCm**: AMD GPU acceleration
- **MPS**: Apple Silicon GPU acceleration (Metal Performance Shaders)
- **OpenVINO**: Intel hardware acceleration
- **Qualcomm**: Qualcomm AI Engine and Hexagon DSP
- **WebNN**: Web Neural Network API for browser inference
- **WebGPU**: Web Graphics API for browser GPU acceleration

## Platform-Specific Templates

You can create templates specific to certain hardware platforms. These templates are selected when generating tests for those platforms.

Example:

```python
# Add a Qualcomm-specific template for bert
template_id = get_next_id(conn, "templates")
conn.execute(
    "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (?, 'bert', 'test', 'qualcomm', ?)",
    [template_id, qualcomm_bert_template]
)
```

## Advanced Features

### Template Inheritance

Templates can inherit from base templates, allowing for specialized behavior while maintaining common functionality:

```python
# Base template for all vision models
conn.execute(
    "INSERT INTO templates (id, model_type, template_type, template) VALUES (?, 'vision', 'test', ?)",
    [base_id, vision_template]
)

# Platform-specific template for vision models on WebGPU
conn.execute(
    "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (?, 'vision', 'test', 'webgpu', ?)",
    [specific_id, webgpu_vision_template]
)
```

### Template Validation

The template validator checks:

1. Syntax correctness
2. Required imports
3. Class structure
4. Method patterns
5. Hardware awareness and completeness

Run validation with:

```bash
python simple_template_validator.py --file template_file.py
python simple_template_validator.py --dir template_directory
```

## System Verification

You can verify that all template system components are working correctly by running the system check script:

```bash
python run_template_system_check.py
```

This script performs the following checks:
1. Template database creation
2. Template validation
3. Test generation for BERT model
4. Hardware platform detection
5. Test generation for vision models
6. Qualcomm AI Engine support
7. Test execution validation

## March 2025 Updates

### Qualcomm AI Engine Support

As of March 2025, all templates include support for Qualcomm AI Engine:

```python
# Qualcomm hardware detection
HAS_QUALCOMM = (
    importlib.util.find_spec("qnn_wrapper") is not None or 
    importlib.util.find_spec("qti") is not None or
    "QUALCOMM_SDK" in os.environ
)

def test_qualcomm(self):
    """Test model on Qualcomm AI Engine."""
    if not HAS_QUALCOMM:
        self.skipTest("Qualcomm AI Engine not available")
        
    # Load model with Qualcomm optimizations
    model = AutoModel.from_pretrained(self.model_name)
    # Run inference
    outputs = model(**inputs)
    # Validate results
    self.assertIsNotNone(outputs)
```

Generate a test with Qualcomm support:

```bash
python simple_test_generator.py -g bert -p qualcomm -o test_bert_qualcomm.py
```

### Enhanced Template Validator

The template validator has been improved to verify:
- Template variable correctness
- Hardware platform completeness
- DuckDB database integration

Run the enhanced validator:

```bash
python simple_template_validator.py --validate-db
```

## Completion Status

| Component | Progress | Status |
|-----------|----------|--------|
| Database Schema | 100% | ✅ Complete |
| Basic Templates | 100% | ✅ Complete |
| Hardware Platform Detection | 100% | ✅ Complete |
| Template Variables | 100% | ✅ Complete |
| Template DB Integration | 100% | ✅ Complete |
| Template Validation | 100% | ✅ Complete |
| Template Inheritance | 100% | ✅ Complete |
| Documentation | 100% | ✅ Complete |
| Qualcomm AI Engine Support | 100% | ✅ Complete |
| System Verification | 100% | ✅ Complete |

## Future Enhancements

1. Create a template management UI
2. Add performance benchmarking support to templates
3. Implement CI/CD integration for template validation
4. Develop analytics for template usage patterns
5. Create template visualization tools
