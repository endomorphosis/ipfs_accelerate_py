# Fixed Hugging Face Test Generator

This directory contains tools for fixing indentation issues in Hugging Face test files and creating minimal, correctly-indented test files.

## Issues Addressed

The original test generator (`test_generator.py`) produced files with various indentation problems:

- Inconsistent indentation within class methods
- Incorrect spacing around method declarations
- Mixed indentation within blocks
- Line continuation indentation errors
- Method definition duplication (e.g., `def method(self,(self,(self,`)
- Missing indentation after conditional statements
- Mock class definition issues

## Solution Approach

Instead of trying to fix each individual issue in the existing files, we've adopted a multi-pronged approach:

1. A minimal test generator that creates clean, properly-indented files from scratch
2. Indentation fixing tools for existing files that attempt to correct common patterns
3. Clean, minimal template files that serve as reference implementations

## Key Files

### 1. Clean Generation

- **`create_minimal_test.py`**: Creates minimal test files with proper indentation
- **`fixed_tests/*.py`**: Generated minimal test files for key model families

### 2. Indentation Fixing

- **`fix_test_indentation.py`**: Tool to attempt fixing indentation in existing files
- **`fix_file_indentation.py`**: Comprehensive indentation fixing for specific patterns

### 3. Template Management

- **`regenerate_tests.py`**: Regenerates test files using the fixed generator
- **`test_generator_fixed.py`**: Corrected test generator with proper indentation handling

## Usage Examples

### Create Minimal Test Files

```bash
# Create minimal test files for all model families
python create_minimal_test.py

# Create for specific families
python create_minimal_test.py --families bert gpt2 t5 vit

# Specify output directory
python create_minimal_test.py --output-dir fixed_tests
```

### Fix Existing Test Files

```bash
# Try to fix indentation issues in existing files
python fix_test_indentation.py test_hf_bert.py test_hf_gpt2.py

# Check syntax without fixing
python fix_test_indentation.py --check-only test_hf_*.py
```

### Regenerate Test Files

```bash
# Regenerate test files for all model families
python regenerate_tests.py --all

# Regenerate for specific families
python regenerate_tests.py --families bert gpt2 t5 vit

# List available model families
python regenerate_tests.py --list
```

## Indentation Standards

To ensure consistent indentation, we follow these rules:

- Top-level code: 0 spaces
- Class definitions: 0 spaces
- Class methods: 4 spaces
- Method content: 8 spaces
- Nested blocks: 12 spaces

## Model Families

Support for the following model families is included:

1. **BERT family** (encoder-only models)
   - `bert-base-uncased`
   - Can be extended to RoBERTa, DistilBERT, etc.

2. **GPT-2 family** (decoder-only models)
   - `gpt2`
   - Can be extended to GPT-Neo, GPT-J, etc.

3. **T5 family** (encoder-decoder models)
   - `t5-small`
   - Can be extended to BART, mT5, etc.

4. **ViT family** (vision models)
   - `google/vit-base-patch16-224`
   - Can be extended to DeiT, BEiT, etc.

## Implementation Approach

We prioritized:

1. **Clarity over completeness**: Clean minimal implementations that work correctly
2. **Modularity**: Separation of test logic from test structure
3. **Validation**: Ensuring syntax correctness for all generated files
4. **Extensibility**: Easy to extend to new model families

## Next Steps

1. Extend support to more model families with specialized architectures
2. Add integration with benchmarking infrastructure
3. Enhance error handling and reporting
4. Add more automation for batch processing