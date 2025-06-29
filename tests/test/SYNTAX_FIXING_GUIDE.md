# Syntax Error Fixing Tools and Guide

This guide covers the tools available for fixing syntax errors in Python test files, particularly focusing on common issues like hyphenated model names and indentation problems.

## Current Status

As of March 21, 2025:
- **341 files with syntax errors** out of 495 test files in the `/skills` directory (68.9% failure rate)
- **644 files with syntax errors** out of 2883 test files across the entire codebase (22.3% failure rate)

Most errors are related to:

1. Unmatched parentheses: `unmatched ')'`
2. Indentation errors: `expected an indented block after 'try' statement`
3. Hyphenated model names (causing syntax errors in variables, classes)

## Available Tools

### 1. `check_test_syntax.py` - Find Syntax Errors

Use this tool to identify files with syntax errors before and after fixes.

```bash
# Check all test files
python check_test_syntax.py --dir skills

# Check only fixed files
python check_test_syntax.py --dir skills/fixed_tests
```

### 2. `fix_hyphenated_model_names.py` - Fix Hyphenated Model Names

Use this tool to fix syntax errors caused by hyphenated model names like "transfo-xl" or "mlp-mixer".

```bash
# Fix all files in a directory
python fix_hyphenated_model_names.py --dir skills

# Fix specific files
python fix_hyphenated_model_names.py --files path/to/file1.py path/to/file2.py
```

The tool handles these conversions:
- Variable names: `transfo-xl` → `transfo_xl` 
- Class names: `TestTransfo-xlModels` → `TestTransfoXlModels`
- Registry constants: `TRANSFO-XL_MODELS_REGISTRY` → `TRANSFO_XL_MODELS_REGISTRY`

### 3. `fix_indentation_and_syntax.py` - Fix Common Syntax Errors

Use this tool to fix more general syntax errors:

```bash
# Fix all files in a directory
python fix_indentation_and_syntax.py --dir skills

# Fix specific files
python fix_indentation_and_syntax.py --files path/to/file1.py path/to/file2.py
```

This tool addresses:
- Unmatched parentheses (extra or missing)
- Indentation errors, especially after `try:` statements
- Missing code blocks
- And other common syntax problems

## Fixing Process

For best results, follow this process:

1. **Identify problematic files**:
   ```bash
   python check_test_syntax.py --dir skills
   ```

2. **Fix hyphenated model names first**:
   ```bash
   python fix_hyphenated_model_names.py --dir skills
   ```

3. **Fix remaining syntax errors**:
   ```bash
   python fix_indentation_and_syntax.py --dir skills
   ```

4. **Verify fixes**:
   ```bash
   python check_test_syntax.py --dir skills
   ```

## Success Stories

The syntax fixing tools have been successfully applied to fix several files:

- Fixed 2 files with try-statement indentation errors (`test_hf_git.py` and `test_hf_paligemma.py`)
- Fixed reference implementations of test files with hyphenated model names (`test_hf_mlp-mixer.py`)

These properly formatted files are now available in the `skills/fixed_tests/` directory and can be used as examples for future fixes.

## Recommended Next Steps

1. **Apply the fixing tools to the broader codebase**:
   ```bash
   python fix_indentation_and_syntax.py --dir .
   ```

2. **Address the remaining hyphenated model name issues**:
   ```bash
   python fix_hyphenated_model_names.py --dir .
   ```

3. **Focus on high-impact files first** - prioritize model test files that are part of active test suites

4. **Consider batch processing by error type** - fix similar errors across multiple files at once

5. **Update the test generator** to ensure it produces syntactically valid Python code in the future

## Common Error Patterns and Solutions

### 1. Hyphenated Model Names

**Problem**: Python identifiers cannot contain hyphens, causing errors when using model names directly.

**Solution**: Convert hyphens to underscores for variables, and use proper capitalization for class names.

**Example**:
```python
# INCORRECT - Syntax Error
TRANSFO-XL_MODELS_REGISTRY = {...}
class TestTransfo-xlModels:

# CORRECT - Fixed
TRANSFO_XL_MODELS_REGISTRY = {...}
class TestTransfoXlModels:
```

### 2. Try-Except Indentation

**Problem**: Missing indented block after `try:` statement.

**Solution**: Properly indent the code block inside the try statement.

**Example**:
```python
# INCORRECT - Syntax Error
try:
logger.info("Testing model...")

# CORRECT - Fixed
try:
    logger.info("Testing model...")
```

### 3. Unmatched Parentheses

**Problem**: Extra closing parentheses or missing opening parentheses.

**Solution**: Fixed by adding or removing parentheses to ensure proper matching.

## Manual Fixes for Complex Cases

Some files might require manual fixing if the automated tools can't resolve all issues. Common techniques include:

1. **Complete rewrite for complex files**: For files with severe syntax issues, a complete rewrite based on a working template is sometimes the best approach.

2. **File-specific matching**: Certain files have unique patterns that require custom replacements or edits.

3. **AST-based validation**: After making changes, validate using Python's Abstract Syntax Tree (AST) parsing to ensure syntactic correctness.

## Notes on Existing Fixed Files

The `skills/fixed_tests/` directory contains properly formatted test files that can be used as references when fixing similar models. Key examples include:

- `test_hf_git.py` - A complete example for a model with a simple name
- `test_hf_mlp-mixer.py` - An example showing how hyphenated model names are handled
- `test_hf_paligemma.py` - Another example with complex structures properly formatted