# HuggingFace Test File Indentation Fixes

This guide explains the complete process for fixing indentation issues in HuggingFace test files. We've developed multiple tools and scripts to address these problems.

## The Problem

The original test generator (`test_generator.py`) has indentation issues that lead to syntax errors in generated test files. Key issues include:

1. Inconsistent indentation in method declarations and bodies
2. Missing spacing between methods
3. Incorrect indentation in exception handling
4. Poorly formatted dependency check blocks
5. Indentation issues in mock class declarations

## Solution Components

We've created multiple components to address these issues:

### 1. Improved Generator: `test_generator_fixed.py`

Contains architecture-aware code generation with proper indentation handling:

- Added `apply_indentation()` helper function
- Fixed `generate_*` functions to produce correctly indented code
- Added `fix_method_boundaries()` utility

### 2. Individual Fixers

- `fix_file_indentation.py`: Comprehensive fixer for individual files
- `simple_fixer.py`: Pattern-based quick fixes
- `complete_cleanup.py`: Advanced fixer for multiple issues

### 3. Integration Tools

- `regenerate_tests_with_fixes.py`: Regenerate test files with fixes
- `execute_integration.py`: Integrate fixes into the main generator
- `integrate_generator_fixes.py`: Another approach for integration

### 4. Documentation

- `INTEGRATION_PLAN.md`: Strategy for integrating the fixes
- `HF_TEST_TROUBLESHOOTING_GUIDE.md`: Guide for fixing common issues
- `TESTING_FIXES_SUMMARY.md`: Summary of all fixes and their effects

## Fix Workflow

### Option 1: Fix Individual Files

If you have test files that need fixing:

```bash
# Comprehensive fix for a specific file
python fix_file_indentation.py path/to/test_file.py

# Simple pattern-based fixes for multiple files
python simple_fixer.py file1.py file2.py file3.py

# Advanced cleaning for complex issues
python complete_cleanup.py path/to/test_file.py
```

### Option 2: Regenerate Tests with Fixes

If you want to regenerate test files with proper indentation:

```bash
# Regenerate specific model families
python regenerate_tests_with_fixes.py --families bert gpt2 t5 vit

# Regenerate all known model families
python regenerate_tests_with_fixes.py --all

# Specify output directory
python regenerate_tests_with_fixes.py --output-dir fixed_files
```

### Option 3: Integrate Fixes into Generator

If you want to fix the source generator:

```bash
# Dry run to see what would happen
python execute_integration.py --dry-run

# Execute the integration
python execute_integration.py

# Alternative approach
python integrate_generator_fixes.py
```

## Minimal Test Files

We've created minimal working examples in the `minimal_tests` directory:

- `minimal_tests/test_hf_bert.py`: Encoder-only text model
- `minimal_tests/test_hf_gpt2.py`: Decoder-only text model
- `minimal_tests/test_hf_t5.py`: Encoder-decoder text model
- `minimal_tests/test_hf_vit.py`: Vision model

These files have correct indentation and are useful as templates.

## Validation

Always validate the syntax of fixed files:

```bash
# Using Python's compile function
python -m py_compile fixed_file.py

# Using the validation function
python -c "
from execute_integration import verify_python_syntax
success, error = verify_python_syntax('path/to/file.py')
print('Valid' if success else f'Error: {error}')
"
```

## Understanding Indentation Patterns

See the `HF_TEST_TROUBLESHOOTING_GUIDE.md` for detailed information about indentation patterns.

## Next Steps

After fixing the indentation issues:

1. Run the fixed test files to ensure they work correctly
2. Integrate the fixes back into the main generator
3. Update the CI/CD pipeline to validate indentation in future changes

## Help and Support

If you encounter any issues with the fixing process:

1. Check `HF_TEST_TROUBLESHOOTING_GUIDE.md` for common issues
2. Compare your file with the minimal templates in `minimal_tests/`
3. Try using the simplified versions in `fixed_tests/` directory