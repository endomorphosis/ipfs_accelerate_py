# Template Validator System

This directory contains the validation system for template-based code generators, which ensures that generated code meets quality standards and has proper syntax.

## Overview

The template validation system checks for:

- Python syntax correctness
- Import statements completeness
- Class structure and required methods
- Consistent indentation
- Hardware compatibility
- Resource pool integration (optional)

## Usage

### Within Generator Scripts

```python
from generators.validators.template_validator_integration import (
    validate_template_for_generator,
    validate_template_file_for_generator
)

# Validate a template string
template_content = """..."""
is_valid, errors = validate_template_for_generator(
    template_content,
    generator_type="merged_test_generator",
    validate_hardware=True,
    check_resource_pool=True
)

if not is_valid:
    for error in errors:
        logger.warning(f"Template validation error: {error}")

# Validate a template file
file_path = "path/to/template.py"
is_valid, errors = validate_template_file_for_generator(
    file_path, 
    generator_type="merged_test_generator"
)
```

### Command Line Usage

```bash
# Validate a template file
python -m generators.validators.template_validator_integration --file path/to/template.py --generator merged_test_generator

# Validate with hardware compatibility check
python -m generators.validators.template_validator_integration --file path/to/template.py --hardware

# Validate with resource pool check
python -m generators.validators.template_validator_integration --file path/to/template.py --resource-pool
```

## Integration with Generators

The validator can be integrated with various generator scripts. For example, to add validation to a test generator:

```python
# Import validator
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from generators.validators.template_validator_integration import (
        validate_template_for_generator,
        validate_template_file_for_generator
    )
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False
    # Define minimal validation function as fallback
    def validate_template_for_generator(template_content, generator_type, **kwargs):
        return True, []
```

Then, in your generator code:

```python
# Add validation command line arguments
parser.add_argument("--validate", action="store_true", 
                    help="Validate templates before generation")
parser.add_argument("--skip-validation", action="store_true",
                    help="Skip template validation")
parser.add_argument("--strict-validation", action="store_true",
                    help="Fail on validation errors")

# Use validation in your code
should_validate = HAS_VALIDATOR and (args.validate and not args.skip_validation)
if should_validate:
    is_valid, validation_errors = validate_template_for_generator(
        template_content, 
        "your_generator_name",
        validate_hardware=True
    )
    
    if not is_valid:
        for error in validation_errors:
            logger.warning(f"Template validation error: {error}")
        
        if args.strict_validation:
            raise ValueError("Template validation failed")
```

## Extending the Validator

To add new validation checks, extend the `TemplateValidator` class in `template_validator_integration.py`:

```python
def validate_new_feature(self, content: str) -> bool:
    # Implement your validation logic
    has_feature = "feature" in content.lower()
    
    if not has_feature:
        self.errors.append("Feature is missing from template")
        
    return has_feature
```

Then update the `validate_all` method to include your new check.