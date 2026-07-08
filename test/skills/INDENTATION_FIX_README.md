# HuggingFace Test Indentation Fix

This directory contains a comprehensive solution for fixing indentation issues in HuggingFace test files.

## Overview

The solution addresses indentation issues at multiple levels:

1. **Root Cause Fix**: Improves the test generator to produce correctly indented code
2. **Post-Processing Fixes**: Provides tools to fix indentation in already generated test files
3. **Comprehensive Validation**: Verifies syntax correctness of all fixed files
4. **Integrated Workflow**: Combines all tools into a unified interface

## Key Components

- `complete_indentation_fix.py`: Main indentation fixing tool for individual files
- `run_indentation_fix.py`: Batch fixer for running fixes on multiple files
- `comprehensive_test_fix.py`: Unified interface combining generation, fixing, and verification
- `test_generator_fixed.py`: Improved generator with proper indentation handling
- `integrate_generator_fixes.py`: Tool to integrate fixes into the main generator

## Indentation Rules

The fix enforces these Python indentation conventions:

- Class definitions: 0 spaces
- Class methods: 4 spaces
- Method content: 8 spaces
- Nested blocks: 12 spaces
- Docstrings: 8 spaces inside methods
- Proper spacing between methods

## Usage Examples

### Single File Fix

To fix indentation in a single file:

```bash
python complete_indentation_fix.py /path/to/test_hf_bert.py --verify
```

### Batch Fixing

To fix indentation in multiple files:

```bash
python run_indentation_fix.py --pattern "test_hf_*.py" --directory . --verify
```

### Regenerating Tests

To regenerate a test with proper indentation:

```bash
python comprehensive_test_fix.py regenerate --families bert gpt2 --output-dir fixed_tests --verify
```

### Complete Workflow

To run the entire workflow (regenerate, fix, verify, integrate):

```bash
python comprehensive_test_fix.py all --families bert gpt2 t5 vit --verify
```

## Architecture-Aware Approach

The solution recognizes that different model architectures require different handling:

- **Encoder-only** (BERT, ViT): Focused on masked inputs and encoding
- **Decoder-only** (GPT-2): Specialized for autoregressive generation
- **Encoder-decoder** (T5): Handles both encoding and generation
- **Vision models** (ViT): Specific processing for image inputs

These architectural differences are addressed in both the generator and fixing tools.

## Technical Implementation

### Key Functions

- `apply_indentation()`: Normalizes indentation in code blocks
- `fix_method_boundaries()`: Ensures proper spacing between methods
- `extract_method()`: Isolates method blocks for targeted fixing
- `fix_method_content()`: Applies proper indentation to method bodies
- `verify_python_syntax()`: Validates syntax correctness

### Indentation Strategies

The fix applies multiple strategies:

1. **Pattern-based replacements**: Fixes common indentation patterns
2. **Content extraction**: Isolates and normalizes specific methods
3. **Context-aware indentation**: Applies different indentation based on code context
4. **Boundary fixing**: Ensures proper spacing between code blocks
5. **Syntax validation**: Verifies fixes with Python's compiler

## Future Work

- Extend support to all HuggingFace model families
- Integrate indentation validation into CI/CD pipelines
- Add pre-commit hooks for indentation checking
- Develop test coverage for the fixing tools themselves

## Troubleshooting

If you encounter issues:

1. Check the log files generated during fix operations
2. Try fixing a single file with verbose output
3. For syntax errors, look at the specific line number reported
4. Manual inspection of the fixed file can help identify specific issues