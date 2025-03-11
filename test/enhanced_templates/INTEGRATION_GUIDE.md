# Template System Integration Guide

This guide explains how to integrate the enhanced template system into your existing IPFS Accelerate Python Framework workflow.

## Overview

The enhanced template system provides improved validation, inheritance, and placeholder handling for the IPFS Accelerate Python Framework's template-based test generation. This guide shows how to:

1. Migrate from using static template files to the enhanced database-driven system
2. Update existing scripts to use the new template inheritance system
3. Take advantage of the improved placeholder handling
4. Add comprehensive validation to your templates

## Migration Steps

### Step 1: Set Up the Template Database

If you haven't already created a template database, run:

```bash
# Create a new template database with base templates
python create_template_database.py --create
```

### Step 2: Apply the Template System Enhancements

Run the enhancement script to apply all improvements:

```bash
# Apply all enhancements at once
./enhanced_templates/run_template_enhancements.sh
```

Or apply specific enhancements:

```bash
# Add template inheritance 
python enhanced_templates/template_system_enhancement.py --add-inheritance

# Enhance placeholder handling
python enhanced_templates/template_system_enhancement.py --enhance-placeholders
```

### Step 3: Validate Your Templates

Check the validation status of your templates:

```bash
# List all templates with validation status
python enhanced_templates/template_system_enhancement.py --list-templates

# Validate specific model templates
python enhanced_templates/template_system_enhancement.py --validate-model-type bert
```

Fix any validation issues identified in your templates.

### Step 4: Update Your Generator Scripts

Update your existing generator scripts to use the enhanced template system:

```python
# Old way (static file-based)
def generate_test(model_name, output_path):
    # Load template from file
    with open("templates/bert_test_template.py", "r") as f:
        template = f.read()
    
    # Render template
    rendered = template.format(
        model_name=model_name,
        normalized_name=model_name.replace("-", "_")
    )
    
    # Write to file
    with open(output_path, "w") as f:
        f.write(rendered)
```

```python
# New way (database-driven with inheritance)
def generate_test(model_name, output_path):
    # Import from enhanced template system
    from enhanced_templates.example_template_generator import (
        get_model_type, 
        get_template_from_db, 
        prepare_template_context, 
        render_template
    )
    
    # Determine model type
    model_type = get_model_type(model_name)
    
    # Get template from database (with inheritance)
    template = get_template_from_db("./template_db.duckdb", model_type, "test")
    
    # Prepare context with hardware detection
    context = prepare_template_context(model_name)
    
    # Render template with enhanced placeholder handling
    rendered = render_template(template, context)
    
    # Write to file
    with open(output_path, "w") as f:
        f.write(rendered)
```

### Step 5: Use the Template Utilities Package

The enhanced system includes a utilities package for template handling:

```python
# Import template utilities
from template_utilities import (
    get_standard_placeholders,
    detect_missing_placeholders,
    get_default_context,
    render_template
)

# Get default context with hardware detection
context = get_default_context(model_name="bert-base-uncased")

# Render template with placeholder substitution and error handling
rendered = render_template(template, context)
```

## Adding New Templates

### Adding a Model-Specific Template

To add a new template for a specific model type:

1. Identify the appropriate parent template based on modality
2. Create your specialized template with inheritance in mind
3. Add it to the database

```bash
# First, list existing templates to find an appropriate parent
python enhanced_templates/template_system_enhancement.py --list-templates

# Add your new template to the database
python create_template_database.py --update
```

### Adding a Hardware-Specific Template

To add a hardware-specific template:

1. Create a template with hardware-specific optimizations
2. Add it to the database with the hardware platform specified

Example SQL for adding a hardware-specific template:
```sql
INSERT INTO templates 
(model_type, template_type, template, hardware_platform, parent_template, modality)
VALUES ('bert', 'test', '...template content...', 'cuda', 'default_text', 'text');
```

## Using Template Inheritance

The enhanced system provides modality-based inheritance:

```
default_text          default_vision          default_audio          default_multimodal
    │                      │                       │                        │
    ├── bert               ├── vit                 ├── whisper              ├── clip
    ├── t5                 ├── resnet              ├── wav2vec2             ├── llava
    ├── llama              └── detr                └── clap                 └── xclip
    └── gpt2
```

When you generate a template, the system automatically:
1. Looks for a model-specific and hardware-specific template
2. Falls back to a model-specific template if no hardware-specific one exists
3. Falls back to the parent modality template if no model-specific one exists
4. Falls back to the default template as a last resort

## Example Workflow

Here's a complete example workflow using the enhanced template system:

```bash
# 1. Initialize the template database
python create_template_database.py --create

# 2. Apply template system enhancements
./enhanced_templates/run_template_enhancements.sh

# 3. Validate templates
python enhanced_templates/template_system_enhancement.py --validate-templates

# 4. Generate a test template for bert-base-uncased
python enhanced_templates/example_template_generator.py --model bert-base-uncased --output test_bert.py

# 5. Run the generated test
python test_bert.py
```

## Hardware-Specific Template Example

Here's an example of using hardware-specific templates:

```bash
# Generate a CUDA-optimized template for bert-base-uncased
python enhanced_templates/example_template_generator.py --model bert-base-uncased --hardware cuda --output test_bert_cuda.py

# Generate a WebGPU-optimized template for vit-base
python enhanced_templates/example_template_generator.py --model vit-base --hardware webgpu --output test_vit_webgpu.py

# Generate templates based on available hardware (auto-detection)
python enhanced_templates/example_template_generator.py --model bert-base-uncased --detect-hardware --output test_bert_auto.py
```

## Troubleshooting

### Template Validation Failed

If template validation fails:

1. Check the validation errors:
   ```bash
   python enhanced_templates/template_system_enhancement.py --validate-model-type bert
   ```

2. Common issues include:
   - Missing required placeholders (`model_name`, `normalized_name`, `generated_at`)
   - Python syntax errors in the template
   - Unbalanced braces in placeholder definitions

### Template Not Found

If a template is not found:

1. Check available templates:
   ```bash
   python enhanced_templates/template_system_enhancement.py --list-templates
   ```

2. Check model type detection:
   ```python
   from enhanced_templates.example_template_generator import get_model_type
   model_type = get_model_type("your-model-name")
   print(f"Detected model type: {model_type}")
   ```

3. Add the missing template to the database

### Database Connectivity Issues

If you encounter database connectivity issues:

1. Check if DuckDB is installed:
   ```bash
   pip show duckdb
   ```

2. Verify database file exists:
   ```bash
   ls -la ./template_db.duckdb
   ```

3. Try creating a new database:
   ```bash
   python create_template_database.py --create
   ```

## Next Steps

After integrating the enhanced template system, consider:

1. Adding more specialized templates for your model types
2. Creating hardware-specific optimizations for critical models
3. Extending the template validation for your specific requirements
4. Contributing improvements back to the template system

## Reference

For complete documentation of the enhanced template system, see:
- [README.md](README.md): Overview of template system enhancements
- [TEMPLATE_SYSTEM_ENHANCEMENTS.md](TEMPLATE_SYSTEM_ENHANCEMENTS.md): Detailed documentation
- [template_system_enhancement.py](template_system_enhancement.py): Implementation details
- [example_template_generator.py](example_template_generator.py): Usage examples