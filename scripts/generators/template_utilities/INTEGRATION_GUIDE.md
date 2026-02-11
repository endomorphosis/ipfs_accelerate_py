# Template Utilities Integration Guide

This guide explains how to integrate the `template_utilities` package with existing template generators and test systems in the IPFS Accelerate Python Framework.

## Introduction

The Template Utilities package provides a comprehensive solution for template management, validation, and rendering. This guide will help you integrate it with existing code to take advantage of features like:

- Template inheritance
- Hardware-aware template generation
- Placeholder validation and smart defaults
- Database-backed template storage

## Integration Steps

### Step 1: Set Up the Template Database

First, create and initialize the template database:

```bash
# Create the database with proper schema
python -m template_utilities.cli db --create

# Add default parent templates
python -m template_utilities.cli db --add-defaults
```

### Step 2: Import Existing Templates

Convert your existing static templates to database-stored templates:

```bash
# For each template file:
python -m template_utilities.cli template --store \
  --model-type bert \
  --template-type test \
  --input /path/to/existing/bert_template.py
```

### Step 3: Update Existing Generator Code

#### Before:

```python
# Old way: reading templates from static files
def generate_test(model_name, model_type):
    template_path = f"templates/{model_type}_template.py"
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Hard-coded context
    context = {
        "model_name": model_name,
        "normalized_name": model_name.replace("-", "_").title()
    }
    
    # Simple string formatting
    rendered = template.format(**context)
    return rendered
```

#### After:

```python
from template_utilities.placeholder_helpers import get_default_context, render_template
from template_utilities.template_database import get_template

def generate_test(model_name, model_type):
    # Get template from database with inheritance support
    template, parent, modality = get_template(
        db_path="./template_db.duckdb",
        model_type=model_type,
        template_type="test"
    )
    
    # Generate context with hardware detection
    context = get_default_context(model_name)
    
    # Add any custom context values if needed
    context["custom_value"] = "something_specific"
    
    # Render with smart fallbacks
    rendered = render_template(template, context)
    return rendered
```

### Step 4: Add Validation to Existing Workflows

Add template validation to your existing CI/CD workflows:

```python
from template_utilities.template_validation import validate_template

def validate_before_commit(template_content, model_type, template_type):
    # Validate template before committing
    success, results = validate_template(
        template=template_content,
        template_type=template_type,
        model_type=model_type
    )
    
    if not success:
        print("Validation failed:")
        if not results["syntax"]["success"]:
            print(f"Syntax errors: {results['syntax']['errors']}")
        if not results["hardware"]["success"]:
            print("Hardware not supported")
        if not results["placeholders"]["success"]:
            print(f"Missing placeholders: {results['placeholders']['missing']}")
        return False
    
    return True
```

### Step 5: Use Hardware Detection in Template Generation

Add hardware awareness to your template generation:

```python
from template_utilities.placeholder_helpers import detect_hardware, get_default_context

def generate_hardware_specific_test(model_name, model_type, hardware_platform=None):
    # Detect available hardware
    hardware = detect_hardware()
    
    # Choose best hardware if not specified
    if not hardware_platform:
        if hardware["cuda"]:
            hardware_platform = "cuda"
        elif hardware["mps"]:
            hardware_platform = "mps"
        else:
            hardware_platform = "cpu"
    
    # Get hardware-specific template
    template, parent, modality = get_template(
        db_path="./template_db.duckdb",
        model_type=model_type,
        template_type="test",
        hardware_platform=hardware_platform
    )
    
    # Generate context with specific hardware
    context = get_default_context(model_name, hardware_platform)
    
    # Render template
    rendered = render_template(template, context)
    return rendered
```

## Working with Template Inheritance

### Converting to Modality-Based Templates

Organize your templates by modality:

```python
from template_utilities.template_database import update_template_inheritance

# After adding templates to the database, update inheritance
update_template_inheritance("./template_db.duckdb")
```

### Creating Templates that Inherit from Parents

When creating new templates, focus on model-specific logic and inherit common functionality:

```python
# Example: Custom BERT template that inherits from default_text
bert_template = """
#!/usr/bin/env python3
\"\"\"
BERT model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

# This template inherits from default_text template
# It only needs to specify BERT-specific functionality

# BERT-specific imports
from transformers import BertModel, BertTokenizer

class Test{normalized_name}(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup inherited from default_text template
        super().setUpClass()
        
        # BERT-specific setup
        cls.tokenizer = BertTokenizer.from_pretrained("{model_name}")
        cls.model = BertModel.from_pretrained("{model_name}")
    
    # Other BERT-specific tests...
"""

# Store with parent template
store_template(
    db_path="./template_db.duckdb",
    model_type="bert",
    template_type="test",
    template_content=bert_template,
    parent_template="default_text",
    modality="text"
)
```

## Integrating with Existing Generators

### Merged Test Generator Integration

```python
# In merged_test_generator.py

from template_utilities.placeholder_helpers import get_default_context, render_template
from template_utilities.template_database import get_template

def generate_model_test(args):
    # Get template with inheritance
    template, parent, modality = get_template(
        db_path=args.db_path,
        model_type=args.model_type,
        template_type="test",
        hardware_platform=args.hardware_platform
    )
    
    # Generate context
    context = get_default_context(args.model_name, args.hardware_platform)
    
    # Add any generator-specific context values
    context.update({
        # Your specific values here
    })
    
    # Render template
    rendered = render_template(template, context)
    
    # Write to output file
    output_path = args.output or f"test_{args.model_name}.py"
    with open(output_path, 'w') as f:
        f.write(rendered)
    
    return output_path
```

### Template-Based Generator Integration

```python
# In template_based_generator.py

from template_utilities.placeholder_helpers import get_default_context, render_template
from template_utilities.template_database import get_template, list_templates

def list_available_templates():
    templates = list_templates(db_path="./template_db.duckdb")
    print("Available templates:")
    for template in templates:
        print(f"{template['model_type']}/{template['template_type']}/{template['hardware_platform']}")

def generate_from_template(model_name, model_type, template_type, output_path):
    template, parent, modality = get_template(
        db_path="./template_db.duckdb",
        model_type=model_type,
        template_type=template_type
    )
    
    context = get_default_context(model_name)
    rendered = render_template(template, context)
    
    with open(output_path, 'w') as f:
        f.write(rendered)
    
    print(f"Generated {template_type} for {model_name} at {output_path}")
```

## Command-Line Integration

Add template utilities to your existing scripts:

```python
import argparse
from template_utilities.placeholder_helpers import get_default_context, render_template
from template_utilities.template_database import get_template

def parse_args():
    parser = argparse.ArgumentParser(description="Generate model tests")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--model-type", required=True, help="Model type")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--db-path", default="./template_db.duckdb", help="Template database path")
    parser.add_argument("--hardware", help="Hardware platform")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get template
    template, parent, modality = get_template(
        db_path=args.db_path,
        model_type=args.model_type,
        template_type="test",
        hardware_platform=args.hardware
    )
    
    # Generate context
    context = get_default_context(args.model, args.hardware)
    
    # Render template
    rendered = render_template(template, context)
    
    # Write to output file
    output_path = args.output or f"test_{args.model.replace('/', '_')}.py"
    with open(output_path, 'w') as f:
        f.write(rendered)
    
    print(f"Generated test for {args.model} at {output_path}")

if __name__ == "__main__":
    main()
```

## Testing the Integration

To verify that the integration works correctly:

```bash
# Test template database access
python -c "from template_utilities.template_database import check_database; print(check_database('./template_db.duckdb'))"

# Test template rendering
python -c "from template_utilities.placeholder_helpers import get_default_context; from template_utilities.template_database import get_template; from template_utilities.placeholder_helpers import render_template; template, _, _ = get_template('./template_db.duckdb', 'bert', 'test'); context = get_default_context('bert-base-uncased'); print(render_template(template, context)[:100] + '...')"
```

## Advanced Integration

### Using Template Inheritance in Custom Generators

```python
from template_utilities.template_inheritance import get_inheritance_hierarchy

def get_template_with_custom_inheritance(model_type, template_type):
    # Get the inheritance hierarchy for this model type
    hierarchy = get_inheritance_hierarchy(model_type)
    
    # Custom hierarchy traversal
    for model_type in hierarchy:
        template, parent, modality = get_template(
            db_path="./template_db.duckdb",
            model_type=model_type,
            template_type=template_type
        )
        
        if template:
            return template, model_type, modality
    
    return None, None, None
```

### Adding Hardware Detection to Existing Generators

```python
from template_utilities.placeholder_helpers import detect_hardware

def generate_with_hardware_awareness(model_name, model_type, template_type):
    # Detect available hardware
    hardware = detect_hardware()
    
    # For CUDA-specific models
    if model_name.endswith("-cuda") and hardware["cuda"]:
        hardware_platform = "cuda"
    elif model_name.endswith("-cpu"):
        hardware_platform = "cpu"
    else:
        # Choose best available hardware
        hardware_platform = None
    
    # Get appropriate template
    template, parent, modality = get_template(
        db_path="./template_db.duckdb",
        model_type=model_type,
        template_type=template_type,
        hardware_platform=hardware_platform
    )
    
    # Generate context with hardware awareness
    context = get_default_context(model_name, hardware_platform)
    
    # Render template
    rendered = render_template(template, context)
    return rendered
```

### Integration with CI/CD Workflows

```python
from template_utilities.template_validation import validate_all_templates

def validate_templates_in_ci():
    """Run in CI to validate all templates"""
    results = validate_all_templates("./template_db.duckdb")
    
    if results["invalid"] > 0:
        print(f"ERROR: {results['invalid']} invalid templates found")
        return 1
    
    print(f"All {results['valid']} templates are valid")
    return 0
```

## Common Error Handling

```python
from template_utilities.template_database import get_template
from template_utilities.placeholder_helpers import render_template, get_default_context

def generate_with_error_handling(model_name, model_type, template_type):
    try:
        # Get template
        template, parent, modality = get_template(
            db_path="./template_db.duckdb",
            model_type=model_type,
            template_type=template_type
        )
        
        if not template:
            # Fallback to a default template
            print(f"Template for {model_type}/{template_type} not found, using default")
            template, parent, modality = get_template(
                db_path="./template_db.duckdb",
                model_type="default",
                template_type=template_type
            )
            
            if not template:
                raise ValueError(f"No template found for {model_type}/{template_type}")
        
        # Generate context
        context = get_default_context(model_name)
        
        # Render template
        try:
            rendered = render_template(template, context)
            return rendered
        except KeyError as e:
            # Handle missing placeholders
            print(f"Missing placeholder: {e}")
            context[str(e).strip("'")] = f"MISSING_{str(e).strip('')}"
            return render_template(template, context)
    
    except Exception as e:
        print(f"Error generating template: {e}")
        # Return a minimal fallback template
        return f"""
        # Fallback template for {model_name}
        # Error: {e}
        
        def test_{model_name.replace('-', '_')}():
            print("Test for {model_name}")
        """
```

## Conclusion

By following this guide, you can smoothly integrate the template utilities package with your existing code, while taking advantage of template inheritance, hardware awareness, and validation features. This approach improves code organization, reduces duplication, and enhances the quality of your generated templates.

For more details on specific components, refer to the main [README.md](README.md) documentation.

If you encounter any issues during integration, please check the [Troubleshooting](#troubleshooting) section in the README or open an issue in the repository.