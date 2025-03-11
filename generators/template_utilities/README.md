# Template Utilities Package

A comprehensive toolkit for template management, validation, and rendering in the IPFS Accelerate Python Framework.

## Overview

The Template Utilities package provides a complete solution for working with templates in the IPFS Accelerate Python Framework, with these key components:

- **Placeholder Management**: Extract, validate, and substitute placeholders in templates with smart defaults
- **Template Validation**: Check template syntax, hardware support, and placeholder completeness
- **Template Inheritance**: Implement modality-based template hierarchy for code reuse and standardization
- **Database Operations**: Store, retrieve, and manage templates in a DuckDB database
- **Hardware Detection**: Identify available hardware and validate hardware support in templates
- **Context Generation**: Create context dictionaries with hardware awareness for template rendering
- **CLI Interface**: Command-line tools for all template operations

## Benefits

By using the Template Utilities package, you can:

- **Reduce Duplication**: Create templates that inherit common functionality from parent templates
- **Ensure Consistency**: Validate templates for syntax, hardware support, and required placeholders
- **Centralize Storage**: Store templates in a database for easier management and retrieval
- **Simplify Rendering**: Generate contexts with sensible defaults and hardware awareness
- **Automate Validation**: Check templates for errors before using them

## Quick Start

### Setting Up the Database

```bash
# Create a new template database
python -m template_utilities.cli db --create

# Add default parent templates (text, vision, audio, multimodal)
python -m template_utilities.cli db --add-defaults

# List all templates in the database
python -m template_utilities.cli db --list
```

### Working with Templates

```bash
# Get a template from the database
python -m template_utilities.cli template --get --model-type bert --template-type test --output bert_template.py

# Store a template in the database
python -m template_utilities.cli template --store --model-type bert --template-type test --input bert_template.py

# Validate a template
python -m template_utilities.cli validate --template bert_template.py --model-type bert --template-type test
```

### Using Placeholders

```bash
# List standard placeholders
python -m template_utilities.cli placeholder --list-standard

# Extract placeholders from a template
python -m template_utilities.cli placeholder --extract --template bert_template.py

# Generate context for a model
python -m template_utilities.cli placeholder --context --model-name bert-base-uncased --output context.json

# Render a template with context
python -m template_utilities.cli placeholder --render --template bert_template.py --model-name bert-base-uncased
```

### Managing Inheritance

```bash
# Get parent template for a model type
python -m template_utilities.cli inheritance --get-parent --model-type bert

# Get inheritance hierarchy for a model type
python -m template_utilities.cli inheritance --get-hierarchy --model-type bert

# Update all templates with inheritance information
python -m template_utilities.cli inheritance --update
```

### Hardware Operations

```bash
# Detect available hardware platforms
python -m template_utilities.cli hardware --detect

# Validate hardware support in a template
python -m template_utilities.cli hardware --validate --template bert_template.py
```

## Command Line Interface

The package includes a comprehensive CLI with multiple subcommands for different operations:

### Database Commands

```bash
# Create a new database
python -m template_utilities.cli db --create

# Check if database exists and has proper schema
python -m template_utilities.cli db --check

# List all templates in the database
python -m template_utilities.cli db --list

# Filter templates by model type
python -m template_utilities.cli db --list --model-type bert

# Add default parent templates
python -m template_utilities.cli db --add-defaults

# Use a custom database path
python -m template_utilities.cli db --list --db-path /path/to/custom_db.duckdb
```

### Template Commands

```bash
# Get a template from the database
python -m template_utilities.cli template --get --model-type bert --template-type test --output bert_template.py

# Get a hardware-specific template
python -m template_utilities.cli template --get --model-type bert --template-type test --hardware-platform cuda

# Store a template in the database
python -m template_utilities.cli template --store --model-type bert --template-type test --input bert_template.py

# Store with custom parent and modality
python -m template_utilities.cli template --store --model-type bert --template-type test --input bert_template.py --parent default_text --modality text

# Store without validation
python -m template_utilities.cli template --store --model-type bert --template-type test --input bert_template.py --no-validate
```

### Validation Commands

```bash
# Validate a template file
python -m template_utilities.cli validate --template bert_template.py --model-type bert --template-type test

# Validate for specific hardware platform
python -m template_utilities.cli validate --template bert_template.py --model-type bert --template-type test --hardware-platform cuda

# Save validation results to file
python -m template_utilities.cli validate --template bert_template.py --model-type bert --template-type test --output validation_results.json

# Validate all templates in the database
python -m template_utilities.cli validate --all --model-type bert
```

### Complete CLI Reference

For a complete list of commands and options:

```bash
# Show help for main CLI
python -m template_utilities.cli --help

# Show help for a specific subcommand
python -m template_utilities.cli db --help
python -m template_utilities.cli template --help
python -m template_utilities.cli validate --help
python -m template_utilities.cli inheritance --help
python -m template_utilities.cli placeholder --help
python -m template_utilities.cli hardware --help
```

## Python API Examples

### Basic Template Operations

```python
from template_utilities.template_database import get_template, store_template, list_templates

# Get a template from the database
template, parent, modality = get_template(
    db_path="./template_db.duckdb",
    model_type="bert",
    template_type="test"
)

# Store a template in the database
store_template(
    db_path="./template_db.duckdb",
    model_type="bert",
    template_type="test",
    template_content=template_content,
    hardware_platform="cuda",  # Optional: for hardware-specific templates
    parent_template="default_text",  # Optional: explicitly set parent template
    modality="text"  # Optional: explicitly set modality
)

# List all templates in the database
templates = list_templates(db_path="./template_db.duckdb")
for template in templates:
    print(f"{template['model_type']}/{template['template_type']}/{template['hardware_platform']}")
```

### Rendering Templates

```python
from template_utilities.placeholder_helpers import get_default_context, render_template

# Generate default context with hardware detection
context = get_default_context("bert-base-uncased")

# Add custom values to context
context["custom_value"] = "my_value"

# Render template with context
rendered = render_template(template, context)

# Render with strict mode (raises error for missing placeholders)
try:
    rendered = render_template(template, context, strict=True)
except ValueError as e:
    print(f"Missing placeholder: {e}")
```

### Template Validation

```python
from template_utilities.template_validation import validate_template_syntax, validate_hardware_support, validate_template

# Validate template syntax
success, errors = validate_template_syntax(template)
if not success:
    print(f"Syntax errors: {errors}")

# Validate hardware support
success, hardware_support = validate_hardware_support(template, "cuda")
if success:
    print(f"Template supports CUDA: {hardware_support['cuda']}")

# Comprehensive validation
success, results = validate_template(template, "test", "bert", "cuda")
if not success:
    print("Validation failed:")
    if not results["syntax"]["success"]:
        print(f"Syntax errors: {results['syntax']['errors']}")
    if not results["hardware"]["success"]:
        print("Hardware not supported")
    if not results["placeholders"]["success"]:
        print(f"Missing placeholders: {results['placeholders']['missing']}")
```

### Template Inheritance

```python
from template_utilities.template_inheritance import get_parent_for_model_type, get_inheritance_hierarchy

# Get parent template for a model type
parent, modality = get_parent_for_model_type("bert")
print(f"Parent template: {parent}")
print(f"Modality: {modality}")

# Get inheritance hierarchy for a model type
hierarchy = get_inheritance_hierarchy("bert")
print("Inheritance hierarchy:")
for i, model_type in enumerate(hierarchy):
    print(f"{'  ' * i}{model_type}")
```

### Hardware Detection

```python
from template_utilities.placeholder_helpers import detect_hardware

# Detect available hardware platforms
hardware = detect_hardware()
for platform, available in hardware.items():
    status = "Available" if available else "Not Available"
    print(f"{platform}: {status}")

# Generate context with specific hardware platform
context = get_default_context("bert-base-uncased", hardware_platform="cuda")
```

## Key Components

### Placeholder Helpers

The `placeholder_helpers` module provides comprehensive utilities for working with placeholders:

| Function | Description |
|----------|-------------|
| `get_standard_placeholders()` | Returns standard placeholders and their properties |
| `extract_placeholders(template)` | Extracts all placeholders from a template |
| `detect_missing_placeholders(template, context)` | Finds missing placeholders in a template |
| `validate_placeholders(template, context)` | Validates placeholders against a context |
| `get_default_context(model_name, hardware_platform)` | Generates context with hardware detection |
| `render_template(template, context, strict)` | Renders a template with fallbacks |
| `get_modality_for_model_type(model_type)` | Determines modality for a model type |
| `normalize_model_name(model_name)` | Normalizes model name for class names |
| `detect_hardware(use_torch)` | Detects available hardware platforms |

### Template Validation

The `template_validation` module handles all aspects of template validation:

| Function | Description |
|----------|-------------|
| `validate_template_syntax(template)` | Checks template for Python syntax errors |
| `validate_hardware_support(template, hardware_platform)` | Validates hardware compatibility |
| `validate_placeholders_in_template(template)` | Checks for required placeholders |
| `validate_template(template, template_type, model_type, hardware_platform)` | Performs all validation checks |
| `validate_template_db_schema(columns)` | Validates database schema columns |

### Template Inheritance

The `template_inheritance` module implements the modality-based inheritance system:

| Function | Description |
|----------|-------------|
| `get_parent_for_model_type(model_type)` | Gets parent template for a model type |
| `get_inheritance_hierarchy(model_type)` | Gets full inheritance chain |
| `get_default_parent_templates()` | Gets default parent templates for all modalities |
| `get_template_with_inheritance(model_type, template_type, templates_db)` | Gets template with inheritance |
| `merge_template_with_parent(child_template, parent_template)` | Merges child with parent |

### Template Database

The `template_database` module provides complete DuckDB integration:

| Function | Description |
|----------|-------------|
| `get_db_connection(db_path)` | Gets a DuckDB connection |
| `create_schema(conn)` | Creates template database schema |
| `check_database(db_path)` | Validates database existence and schema |
| `get_template(db_path, model_type, template_type, hardware_platform)` | Retrieves template |
| `store_template(db_path, model_type, template_type, template_content, ...)` | Stores template |
| `list_templates(db_path, model_type)` | Lists templates in the database |
| `add_default_parent_templates(db_path)` | Adds default parent templates |
| `update_template_inheritance(db_path)` | Updates inheritance relationships |
| `validate_all_templates(db_path, model_type)` | Validates all templates |

## Database Schema

The template database uses DuckDB with the following schema:

### Templates Table

Stores the core template content and metadata:

```sql
CREATE TABLE templates (
    id INTEGER PRIMARY KEY,
    model_type VARCHAR NOT NULL,          -- Model type (bert, vit, etc.)
    template_type VARCHAR NOT NULL,       -- Template type (test, benchmark, skill)
    template TEXT NOT NULL,               -- The actual template content
    hardware_platform VARCHAR,            -- Optional hardware platform
    validation_status VARCHAR,            -- VALID, INVALID, or UNKNOWN
    parent_template VARCHAR,              -- Parent template for inheritance
    modality VARCHAR,                     -- text, vision, audio, multimodal
    last_updated TIMESTAMP                -- When the template was last updated
);
```

### Template Validation Table

Stores detailed validation results for each template:

```sql
CREATE TABLE template_validation (
    id INTEGER PRIMARY KEY,
    template_id INTEGER,                  -- Reference to templates.id
    validation_date TIMESTAMP,            -- When validation was performed
    validation_type VARCHAR,              -- Type of validation (syntax, hardware, etc.)
    success BOOLEAN,                      -- Whether validation passed
    errors TEXT,                          -- JSON array of error messages
    hardware_support TEXT,                -- JSON object of hardware support
    FOREIGN KEY (template_id) REFERENCES templates(id)
);
```

### Template Placeholders Table

Stores documentation for standard placeholders:

```sql
CREATE TABLE template_placeholders (
    id INTEGER PRIMARY KEY,
    placeholder VARCHAR NOT NULL,         -- Placeholder name
    description TEXT,                     -- Description of the placeholder
    default_value VARCHAR,                -- Default value (if any)
    required BOOLEAN                      -- Whether the placeholder is required
);
```

## Modality-Based Inheritance

Templates are organized in a modality-based inheritance hierarchy that promotes code reuse:

```
default_text          default_vision          default_audio          default_multimodal
    │                      │                       │                        │
    ├── bert               ├── vit                 ├── whisper              ├── clip
    ├── t5                 ├── resnet              ├── wav2vec2             ├── llava
    ├── llama              └── detr                └── clap                 └── xclip
    └── gpt2
```

### Benefits of Inheritance

- **Code Reuse**: Common functionality is defined once in parent templates
- **Consistency**: Standard patterns across similar models
- **Easier Maintenance**: Update parent templates to affect all child templates
- **Specialization**: Override specific parts in child templates when needed

### Modality Features

Each modality's parent template includes specialized functionality:

| Modality | Features |
|----------|----------|
| **Text** | Tokenizer integration, sequence length handling, text preprocessing |
| **Vision** | Image processing, feature extraction, test image generation |
| **Audio** | Audio preprocessing, sampling rate handling, silence generation |
| **Multimodal** | Combined text/image/audio processing, synchronized inputs |

## Standard Placeholders

The template system includes these standard placeholders with smart defaults:

| Placeholder | Description | Default | Required |
|-------------|-------------|---------|----------|
| `model_name` | Full model name | None | Yes |
| `normalized_name` | Normalized model name for class names | Auto-generated | Yes |
| `generated_at` | Generation timestamp | Current time | Yes |
| `best_hardware` | Best available hardware | Auto-detected | No |
| `torch_device` | PyTorch device to use | Auto-detected | No |
| `has_cuda` | CUDA availability | Auto-detected | No |
| `has_rocm` | ROCm availability | Auto-detected | No |
| `has_mps` | MPS availability | Auto-detected | No |
| `has_openvino` | OpenVINO availability | Auto-detected | No |
| `has_qualcomm` | Qualcomm AI Engine availability | Auto-detected | No |
| `has_samsung` | Samsung NPU availability | Auto-detected | No |
| `has_webnn` | WebNN availability | Auto-detected | No |
| `has_webgpu` | WebGPU availability | Auto-detected | No |
| `model_type` | Model type | Auto-detected | No |
| `modality` | Model modality | Auto-detected | No |

## Integration with Existing Codebase

To use the template utilities package with existing template generators:

```python
from template_utilities import (
    get_default_context, render_template, get_template
)

# In your generator script:
def generate_test_for_model(model_name, model_type, output_path):
    # Get template from database
    template, parent, modality = get_template(
        db_path="./template_db.duckdb",
        model_type=model_type,
        template_type="test"
    )
    
    # Generate context with hardware detection
    context = get_default_context(model_name)
    
    # Render template
    rendered = render_template(template, context)
    
    # Write to output file
    with open(output_path, 'w') as f:
        f.write(rendered)
    
    return output_path
```

## Best Practices

1. **Use Inheritance**: Create parent templates for common functionality and inherit in child templates
2. **Validate Templates**: Always validate templates before using them
3. **Document Placeholders**: Add descriptions and default values for all placeholders
4. **Use Standard Placeholders**: Reuse standard placeholders for consistency
5. **Be Hardware-Aware**: Make templates work across different hardware platforms
6. **Test Templates**: Write tests for your templates to ensure they work as expected

## Troubleshooting

### Common Issues

- **Missing Placeholders**: Make sure all required placeholders are defined in your context
- **Database Connection Errors**: Check that the database file exists and has the right permissions
- **Validation Errors**: Fix syntax errors, hardware support issues, and missing placeholders
- **DuckDB Import Errors**: Install DuckDB with `pip install duckdb`

### Debugging

Enable debug logging for more detailed information:

```bash
python -m template_utilities.cli --debug db --list
```

## Contributing

To contribute to the template utilities package:

1. Add new functionality to the appropriate module
2. Update the CLI to expose new functionality
3. Add tests for new functionality in `test_utilities.py`
4. Update this README with documentation for new features
5. Ensure backward compatibility with existing code

## License

This package is part of the IPFS Accelerate Python Framework and is distributed under the same license.