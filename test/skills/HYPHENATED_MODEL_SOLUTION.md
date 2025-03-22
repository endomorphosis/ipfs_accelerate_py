# Hyphenated Model Name Solution

## Problem Analysis

The current template-based test generator has several flaws when handling hyphenated model names like "xlm-roberta" and "gpt-j":

1. **String-based replacement causes code mangling**: Using naive string replacement causes issues with code structure, especially for complex templates.

2. **Indentation issues in try/except blocks**: Despite multiple fixing attempts, the code still has problems with indentation in try/except blocks.

3. **Unterminated triple-quoted strings**: The current fix doesn't handle all cases of unterminated triple quotes.

4. **Method and class definition issues**: Method definitions sometimes get combined with docstrings.

## Implemented Solution

After analyzing various approaches, a direct solution was implemented that bypasses template processing entirely for hyphenated model names:

1. **Direct String Generation**: Instead of using templates, we generate properly formatted test files directly using string literals, ensuring valid syntax.

2. **Special Case Handling**: Dedicated handling for each hyphenated model name ensures proper identifier conversion.

3. **Integration with Main Generator**: The `test_generator_fixed.py` script now checks for hyphenated names and routes them to a specialized generator.

4. **Syntax Validation**: Each generated file is validated for syntax before writing to disk.

## Implementation Details

### 1. Specialized Generator Function in simplified_fix_hyphenated.py

This script contains the `create_hyphenated_test_file()` function that generates valid test files for models with hyphenated names.

Key features:
- Maps hyphenated model names to valid Python identifiers
- Provides proper class name capitalization
- Creates syntactically valid test files with all necessary imports and functions
- Validates syntax before writing to disk

### 2. Integration in test_generator_fixed.py

The main generator now has special handling for hyphenated models:

```python
def generate_test_file(model_family, output_dir="."):
    """Generate a test file for a model family."""
    # Special handling for hyphenated model names
    if "-" in model_family:
        # Use the specialized function for hyphenated models
        try:
            from simplified_fix_hyphenated import create_hyphenated_test_file
            logger.info(f"Using specialized hyphenated model generator for {model_family}")
            result = create_hyphenated_test_file(model_family, output_dir)
            if result[0]:  # Success
                return True
            else:
                logger.warning(f"Specialized generator failed: {result[1]}, falling back to standard approach")
        except ImportError:
            logger.warning("simplified_fix_hyphenated.py not available, using standard approach for hyphenated model")
    
    # Standard approach for non-hyphenated models...
```

### 3. Helper Functions in Both Files

Both files implement these helper functions:

- `to_valid_identifier(text)`: Converts hyphenated names to valid Python identifiers (e.g., "gpt-j" → "gpt_j")
- `get_class_name_capitalization(model_name)`: Gets proper class name (e.g., "gpt-j" → "GPTJ") 
- `get_upper_case_name(model_name)`: Gets uppercase name for constants (e.g., "gpt-j" → "GPT_J")

## Verification Results

The solution has been tested with the following hyphenated models:

1. **gpt-j**: Successfully generates `test_hf_gpt_j.py` with valid identifiers and class names.
2. **xlm-roberta**: Successfully generates `test_hf_xlm_roberta.py` with proper handling of mixed-case identifiers.

Example output:
```
2025-03-21 23:36:44,128 - INFO - Creating test file for xlm-roberta -> id: xlm_roberta, class: XLMRoBERTa, type: encoder-only
2025-03-21 23:36:44,129 - INFO - Successfully created test_output/test_hf_xlm_roberta.py
```

## Advantages of This Approach

1. **Reliability**: The direct generation approach avoids template parsing issues completely.
2. **Maintainability**: Clear, explicit code generation is easier to understand and modify.
3. **Extensibility**: New hyphenated model types can be easily added to the mapping.
4. **Syntax Correctness**: Generated files are validated for syntax before writing.

## Next Steps

1. Test the solution with all hyphenated model types in the HuggingFace ecosystem.
2. Add task-specific content for different model types (e.g., fill-mask for encoder models, text-generation for decoder models).
3. Update the model registry with more detailed information for hyphenated models.
4. Consider extending the approach to other problematic model names (e.g., those with unusual characters).