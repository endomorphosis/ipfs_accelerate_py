# Template Validation System

This directory contains the comprehensive template validation system for the IPFS Accelerate Python project. The system ensures that templates have proper syntax, indentation, hardware compatibility, and follow project conventions.

## Overview

The template validation system is a critical component of our template-based code generation infrastructure. It was implemented to address inconsistencies in template formatting, ensure hardware compatibility, and enforce coding standards across generated code.

### Key Components

1. **Template Validator Integration** (`template_validator_integration.py`):
   - Validates template syntax, imports, class structure, indentation, hardware compatibility, and more
   - Can be imported into any generator script for integrated validation
   - Provides command-line interface for standalone validation
   - Supports both strict and lenient validation modes for different use cases

2. **Indentation Fixer** (`fix_template_indentation.py`):
   - Analyzes and fixes indentation issues in template strings
   - Handles various template formats and structures (triple-quoted strings, dictionary values, etc.)
   - Provides dry-run option to preview changes without modifying files
   - Intelligently preserves intentional indentation structure while fixing inconsistencies

3. **Combined Validator and Fixer** (`validate_and_fix_templates.py`):
   - Combines template validation and indentation fixing in a single workflow
   - Provides command-line interface for integrated validation and fixing
   - Generates detailed reports on validation and fixing results
   - Supports batch processing of multiple files or directories

4. **Helper Tools**:
   - `apply_validation_to_generators.py`: Automatically adds validation to existing generator scripts
   - `create_fixed_template_generator.py`: Example fixed template generator using the validator
   - `fixed_template_example.py`: Contains examples of properly formatted templates

### Why Template Validation Matters

Template validation is essential for maintaining high-quality code generation:

- **Ensures Consistency**: Templates are used across multiple generators and model types, so consistency is critical
- **Prevents Errors**: Syntax or indentation errors in templates can propagate to generated code
- **Simplifies Maintenance**: Well-validated templates are easier to maintain and update
- **Enables Cross-Platform Support**: Validation ensures templates support all required hardware platforms
- **Improves Readability**: Proper indentation makes templates and generated code more readable

## Usage Guide

We've designed the template validation system to be easy to use in various contexts. Below you'll find detailed instructions for using each component.

### Template Validator Integration

The template validator is the core component for validating templates. It can be used directly in your Python code or via the command line.

#### Python API

```python
from generators.validators.template_validator_integration import (
    validate_template_for_generator,
    validate_template_file_for_generator,
    TemplateValidator
)

# Validate a template string
is_valid, errors = validate_template_for_generator(
    template_content,                       # The template string to validate
    generator_type="merged_test_generator",  # Type of generator (affects validation rules)
    validate_hardware=True,                 # Check for hardware platform support
    check_resource_pool=True,               # Check for resource pool integration
    strict_indentation=False                # Be lenient with indentation in templates
)

if not is_valid:
    for error in errors:
        print(f"Error: {error}")

# Validate a template file
is_valid, errors = validate_template_file_for_generator(
    file_path,                              # Path to the template file
    generator_type="fixed_template_generator",
    validate_hardware=True,
    check_resource_pool=True,
    strict_indentation=False
)

# For more control, use the TemplateValidator class directly
validator = TemplateValidator(generator_type="custom_generator")
is_valid, errors = validator.validate_all(
    template_content,
    validate_hardware=True,
    check_resource_pool=True,
    strict_indentation=False
)

# You can also validate specific aspects
validator = TemplateValidator()
syntax_valid = validator.validate_syntax(template_content)
imports_valid = validator.validate_imports(template_content)
classes_valid = validator.validate_classes(template_content)
indentation_valid = validator.validate_indentation(template_content)
hardware_valid = validator.validate_hardware_compatibility(template_content)
```

#### Command-Line Usage

The validator can be used directly from the command line:

```bash
# Validate a file
python scripts/generators/validators/template_validator_integration.py --file path/to/template.py --generator merged_test_generator

# Validate and check hardware compatibility
python scripts/generators/validators/template_validator_integration.py --file path/to/template.py --hardware

# Validate and check resource pool integration
python scripts/generators/validators/template_validator_integration.py --file path/to/template.py --resource-pool

# Validate with strict indentation rules
python scripts/generators/validators/template_validator_integration.py --file path/to/template.py --strict-indentation

# Validate a string directly
python scripts/generators/validators/template_validator_integration.py --content "def test(): pass" --generator test_generator
```

### Indentation Fixer

```python
from generators.validators.fix_template_indentation import (
    fix_file,
    fix_directory,
    identify_template_variables
)

# Fix indentation in a file
success, fixed_count = fix_file(file_path, dry_run=False)

# Fix indentation in all files in a directory
success, file_count, fixed_count = fix_directory(directory, pattern="*.py", dry_run=False)

# Identify template variables in a file
variables = identify_template_variables(file_path)
```

Command-line usage:

```bash
# Fix a specific file
python scripts/generators/validators/fix_template_indentation.py --file path/to/file.py

# Fix all files in a directory
python scripts/generators/validators/fix_template_indentation.py --directory path/to/dir --pattern "*.py"

# Dry run to see what would be changed
python scripts/generators/validators/fix_template_indentation.py --file path/to/file.py --dry-run

# Identify template variables in a file
python scripts/generators/validators/fix_template_indentation.py --file path/to/file.py --identify-templates
```

### Combined Validator and Fixer

```bash
# Validate and fix a file
python scripts/generators/validators/validate_and_fix_templates.py --file path/to/file.py

# Validate and fix all files in a directory
python scripts/generators/validators/validate_and_fix_templates.py --directory path/to/dir --pattern "*.py"

# Validate only (no fixing)
python scripts/generators/validators/validate_and_fix_templates.py --file path/to/file.py --no-fix

# Dry run (report only, no changes)
python scripts/generators/validators/validate_and_fix_templates.py --file path/to/file.py --dry-run

# Strict validation
python scripts/generators/validators/validate_and_fix_templates.py --file path/to/file.py --strict

# Generate JSON report
python scripts/generators/validators/validate_and_fix_templates.py --file path/to/file.py --json report.json
```

## Integrating with Generator Scripts

To add template validation to an existing generator script:

```bash
python scripts/generators/validators/apply_validation_to_generators.py --file path/to/generator.py
```

## Example: Fixed Template Generator

The fixed template generator (`create_fixed_template_generator.py`) demonstrates how to use the validator in a real generator. It generates test files with consistent indentation and validates the generated templates.

```bash
python scripts/generators/validators/create_fixed_template_generator.py --model bert-base-uncased --validate
```

## Adding Validation to Projects

The validation system can be added to any template-based code generation project. The apply_validation_to_generators.py script automates the process of adding validation to existing generators.

## Validation Criteria

The template validator checks for:

1. **Syntax Correctness**: Templates must have valid Python syntax
2. **Import Statements**: Templates must include required imports
3. **Class Structure**: Templates must have classes with required methods
4. **Indentation Consistency**: Templates must have consistent indentation
5. **Hardware Compatibility**: Templates must support all hardware platforms
6. **Resource Pool Integration**: Templates can optionally support resource pools

## Troubleshooting

If you encounter validation errors:

1. **Syntax Errors**: Check for mismatched parentheses, quotes, or indentation
2. **Import Errors**: Ensure all required imports are present
3. **Class Structure Errors**: Check that template contains required classes and methods
4. **Indentation Errors**: Run the indentation fixer with --dry-run to see proposed fixes
5. **Hardware Compatibility Errors**: Ensure the template includes support for all hardware platforms