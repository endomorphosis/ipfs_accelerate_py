# Syntax Error Fixing Guide for Test Generator

This document explains the syntax error fixing capabilities that have been integrated into the HuggingFace model test generator. These improvements address common syntax issues, particularly those involving hyphenated model names.

## Hyphenated Model Name Problem

Many HuggingFace models have hyphenated names (e.g., `xlm-roberta`, `gpt-j`, `transfo-xl`, `mlp-mixer`). When these names are used directly in Python code, they can cause syntax errors since hyphens are not valid characters in Python identifiers.

## Integrated Solutions

The test generator now includes the following fixes:

### 1. Valid Identifier Conversion

```python
def to_valid_identifier(text):
    """Convert text to a valid Python identifier."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text
```

This function converts hyphenated names like `xlm-roberta` to valid Python identifiers like `xlm_roberta`.

### 2. PascalCase Conversion for Class Names

```python
def get_pascal_case_identifier(text):
    """Convert a model name (potentially hyphenated) to PascalCase for class names."""
    # Split by hyphens and capitalize each part
    parts = text.split('-')
    return ''.join(part.capitalize() for part in parts)
```

This function converts hyphenated names like `xlm-roberta` to PascalCase for class names like `XlmRoberta`.

### 3. Progressive Syntax Error Fixing

The generator implements a multi-stage fixing process when syntax errors are detected:

1. **Hyphenated Reference Fixing**: Checks for missed conversions of hyphenated model names
2. **Common Syntax Error Fixes**: Addresses issues like unclosed string literals and docstrings
3. **Indentation Fixes**: Corrects common indentation problems, especially in class methods and try/except blocks
4. **Delimiter Balancing**: Ensures parentheses, brackets, and braces are balanced

### 4. Specialized Testing for Hyphenated Models

The generator includes a special testing mode for hyphenated models:

```bash
python test_generator_fixed.py --test-hyphenated --verify
```

This mode specifically tests all hyphenated model families to ensure they can be processed without syntax errors.

## Common Fixes Applied

The generator now automatically handles:

1. **Variable Names**: Converts `xlm-roberta` → `xlm_roberta`
2. **Class Names**: Converts `TestXlm-RobertaModels` → `TestXlmRobertaModels`
3. **Registry Constants**: Converts `XLM-ROBERTA_MODELS_REGISTRY` → `XLM_ROBERTA_MODELS_REGISTRY`
4. **Imports & Paths**: Fixes file paths and import references with hyphenated names
5. **Method Indentation**: Corrects indentation in class methods
6. **String Literals**: Fixes unclosed string literals and docstrings
7. **Unbalanced Delimiters**: Adds missing parentheses, brackets, and braces

## Usage in CI/CD

For automated testing in CI/CD environments, it's recommended to use:

```bash
python test_generator_fixed.py --test-hyphenated --verify --output-dir generated_tests
```

This will generate tests for all hyphenated model families, verify their syntax, and save them to the specified output directory.

## Handling Remaining Errors

While the generator includes comprehensive fixes, some complex syntax issues may still require manual attention. If you encounter persistent errors:

1. Run with the `--generate` flag for a specific model family
2. Check the log output for detailed error information
3. Apply manual fixes to the template file if needed

## Integration with Templates

The syntax fixing has been integrated at the template level. When new templates are created, ensure they properly handle hyphenated model names by using the `to_valid_identifier` and `get_pascal_case_identifier` functions.