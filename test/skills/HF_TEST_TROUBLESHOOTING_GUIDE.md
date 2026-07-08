# HuggingFace Test Troubleshooting Guide

This guide provides solutions for common issues encountered when implementing test files for HuggingFace models, with a special focus on hyphenated model names.

## Common Issues and Solutions

### 1. Syntax Errors with Hyphenated Model Names

**Issue**: Python syntax errors when using hyphenated model names like `gpt-j` or `xlm-roberta`.

**Solution**: Convert hyphenated model names to valid Python identifiers by replacing hyphens with underscores.

```python
# Convert "gpt-j" to "gpt_j"
def to_valid_identifier(text):
    return text.replace("-", "_")

# Usage
model_type = "gpt-j"
valid_name = to_valid_identifier(model_type)  # "gpt_j"
```

### 2. Class Name Capitalization Inconsistencies

**Issue**: Different models have different capitalization patterns for their class names.

**Solution**: Create a mapping of model types to their proper capitalization formats.

```python
CLASS_NAME_CAPITALIZATION = {
    "gpt-j": "GPTJ",
    "gpt-neo": "GPTNeo",
    "xlm-roberta": "XLMRoBERTa"
}

def get_class_name_capitalization(model_type):
    if model_type.lower() in CLASS_NAME_CAPITALIZATION:
        return CLASS_NAME_CAPITALIZATION[model_type.lower()]
    # Default handling for unknown models
    parts = model_type.split('-')
    return ''.join(part.capitalize() for part in parts)
```

### 3. Incorrect Registry Names

**Issue**: Registry variable names containing hyphens cause syntax errors.

**Solution**: Always use valid Python identifiers for registry names.

```python
# Incorrect
GPT-J_MODELS_REGISTRY = { ... }

# Correct
GPT_J_MODELS_REGISTRY = { ... }
```

### 4. Incorrect Class Names in Imports and References

**Issue**: References to transformers classes containing hyphens cause syntax errors.

**Solution**: Use the proper capitalized form without hyphens.

```python
# Incorrect
model_class = transformers.GPT-JLMHeadModel

# Correct
model_class = transformers.GPTJLMHeadModel
```

### 5. Class Declaration Syntax Errors

**Issue**: Class declarations containing hyphens cause syntax errors.

**Solution**: Use proper Python identifier format for class names.

```python
# Incorrect
class TestGPT-JModels:

# Correct
class TestGPTJModels:
```

### 6. Template Selection Errors

**Issue**: Using the wrong template for a model architecture leads to incorrect test structure.

**Solution**: Determine the architecture type for each model and select the appropriate template.

```python
def get_architecture_type(model_type):
    model_type_lower = model_type.lower()
    
    if any(model in model_type_lower for model in ["bert", "roberta", "electra"]):
        return "encoder-only"
        
    if any(model in model_type_lower for model in ["gpt", "bloom", "llama"]):
        return "decoder-only"
        
    if any(model in model_type_lower for model in ["t5", "bart", "pegasus"]):
        return "encoder-decoder"
        
    # ... other architectures
    
    return "unknown"
```

### 7. Indentation Errors

**Issue**: Inconsistent indentation in generated Python files causes syntax errors.

**Solution**: Validate file syntax after generation and fix indentation issues.

```python
def verify_syntax(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        # Try to compile the code
        compile(content, file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False
```

### 8. File Naming Consistency Issues

**Issue**: Inconsistent file naming for hyphenated models (e.g., `test_hf_gpt-j.py` vs `test_hf_gpt_j.py`).

**Solution**: Always use valid Python identifiers in filenames.

```python
def get_test_file_name(model_type):
    valid_name = model_type.replace("-", "_")
    return f"test_hf_{valid_name}.py"
```

## Validation and Consistency Checks

### Checking Test File Syntax

```python
import py_compile

def check_file_syntax(file_path):
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"✅ {file_path}: Syntax is valid")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ {file_path}: Syntax error: {e}")
        return False
```

### Checking for Hyphenated Names in Files

```python
import re

def check_for_hyphenated_names(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for class definitions with hyphens
    class_matches = re.findall(r"class\s+Test(\w+)-(\w+)Models", content)
    if class_matches:
        print(f"Found class with hyphen in {file_path}:")
        for match in class_matches:
            class_name = f"Test{match[0]}-{match[1]}Models"
            print(f"  - {class_name}")
    
    # Check for registry variables with hyphens
    var_matches = re.findall(r'(\w+)-(\w+)_MODELS_REGISTRY\s*=', content)
    if var_matches:
        print(f"Found variable with hyphen in {file_path}:")
        for match in var_matches:
            var_name = f"{match[0]}-{match[1]}_MODELS_REGISTRY"
            print(f"  - {var_name}")
```

## Error Patterns and Solutions

| Error Pattern | Example | Solution |
|---------------|---------|----------|
| Invalid syntax in class declaration | `class TestGPT-JModels:` | Replace with `class TestGPTJModels:` |
| Invalid syntax in variable name | `GPT-J_MODELS_REGISTRY = {...}` | Replace with `GPT_J_MODELS_REGISTRY = {...}` |
| Invalid syntax in method call | `model_class = transformers.GPT-JLMHeadModel` | Replace with `model_class = transformers.GPTJLMHeadModel` |
| Invalid filename | `test_hf_gpt-j.py` | Rename to `test_hf_gpt_j.py` |
| Inconsistent capitalization | `gptj` vs `GPTJ` | Use consistent capitalization pattern for each model |

## Using Automated Test Generators

For systematic test generation and validation:

1. **find_models.py**: Discover model classes and architecture types
2. **fix_indentation_and_apply_template.py**: Create test files with proper templates
3. **generate_hyphenated_tests.py**: Generate files for hyphenated models
4. **check_test_consistency.py**: Verify test file consistency

Example workflow:

```bash
# Discover models
./find_models.py --output model_data.json

# Generate tests for all discovered models
./generate_tests.py --models-data model_data.json

# Validate all generated tests
./check_test_consistency.py
```

## Running Tests

To verify that tests work correctly:

```bash
# Run a specific test with list-models option
python fixed_tests/test_hf_gpt_j.py --list-models

# Run a quick test with a specific model
python fixed_tests/test_hf_gpt_j.py --model "EleutherAI/gpt-j-6b" --cpu-only
```

## Advanced Troubleshooting

For more complex issues, you can use the `create_coverage_report.py` script to generate a comprehensive report of test coverage and identify any missing or problematic tests. This report will help you prioritize which models need attention and which specific issues need to be fixed.