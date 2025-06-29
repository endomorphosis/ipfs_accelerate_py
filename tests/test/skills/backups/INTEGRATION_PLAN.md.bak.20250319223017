# Integration Plan for Fixed Hugging Face Test Files

This document outlines the plan for integrating our fixed test files and tools into the main project.

## Current Status

We have developed:

1. A minimal test generator (`create_minimal_test.py`) that creates clean, correctly-indented test files
2. Several tools for fixing indentation issues in existing test files (`fix_test_indentation.py`, `fix_file_indentation.py`)
3. Minimal test files for key model families (`fixed_tests/test_hf_*.py`)
4. Documentation on fixing indentation issues and extending the solution

## Integration Steps

### Phase 1: Copy Fixed Files to Main Directory

1. **Deploy minimal test files for core models**
   ```bash
   cp fixed_tests/test_hf_bert.py fixed_tests/test_hf_gpt2.py fixed_tests/test_hf_t5.py fixed_tests/test_hf_vit.py ..
   ```

2. **Copy utility scripts**
   ```bash
   cp create_minimal_test.py fix_test_indentation.py ..
   ```

3. **Copy documentation**
   ```bash
   cp HF_TEST_TROUBLESHOOTING_GUIDE.md FIXED_GENERATOR_README.md ..
   ```

### Phase 2: Verify Functionality

1. **Test the minimal files**
   ```bash
   python ../test_hf_bert.py --cpu-only
   python ../test_hf_gpt2.py --cpu-only
   python ../test_hf_t5.py --cpu-only
   python ../test_hf_vit.py --cpu-only
   ```

2. **Verify syntax for all files**
   ```bash
   python -m py_compile ../test_hf_bert.py ../test_hf_gpt2.py ../test_hf_t5.py ../test_hf_vit.py
   ```

### Phase 3: Generate Additional Test Files

1. **Extend the model families in `create_minimal_test.py`**
   ```python
   MODEL_FAMILIES = {
       # ...existing families...
       "roberta": {
           "model_id": "roberta-base",
           "model_class": "RobertaModel",
           "tokenizer_class": "RobertaTokenizer",
           "task": "fill-mask",
           "test_text": "The man worked as a <mask>.",
           "architecture_type": "encoder_only"
       },
       # Add more families...
   }
   ```

2. **Generate additional test files**
   ```bash
   python ../create_minimal_test.py --families roberta bart deit gpt_neo
   ```

### Phase 4: Fix More Complex Files

For more complex model families that require specialized handling:

1. **Apply the indentation fixing tools**
   ```bash
   python ../fix_test_indentation.py ../test_hf_complex_model.py
   ```

2. **For files that can't be fixed automatically, create custom minimal templates**
   ```bash
   # Create a specialized template
   cp ../create_minimal_test.py ../create_specialized_test.py
   # Modify to add specialized handling for complex models
   ```

3. **Generate files from your specialized templates**
   ```bash
   python ../create_specialized_test.py --families complex_model
   ```

### Phase 5: Update Generator

1. **Replace the original test generator with the fixed version**
   ```bash
   cp test_generator_fixed.py ../test_generator.py
   ```

2. **Add comprehensive testing to ensure generated files have correct indentation**
   ```bash
   # Create a test script
   cat > ../test_generator_indentation.py << 'EOF'
   #!/usr/bin/env python3
   import os
   import sys
   import subprocess
   
   # Generate a test file for each family
   families = ["bert", "gpt2", "t5", "vit"]
   for family in families:
       subprocess.run(["python", "test_generator.py", "--family", family])
       # Verify syntax
       result = subprocess.run(["python", "-m", "py_compile", f"test_hf_{family}.py"])
       if result.returncode == 0:
           print(f"✅ {family} - Syntax OK")
       else:
           print(f"❌ {family} - Syntax Error")
   EOF
   
   # Make executable and run
   chmod +x ../test_generator_indentation.py
   ./test_generator_indentation.py
   ```

## Testing Strategy

To ensure integration is successful:

1. **Syntax validation**: Verify all files pass Python's syntax checker
   ```bash
   python -m py_compile ../test_hf_*.py
   ```

2. **Functionality testing**: Run each test file to verify it works
   ```bash
   for file in ../test_hf_*.py; do python $file --cpu-only; done
   ```

3. **Integration testing**: Test with the main project's testing pipeline
   ```bash
   cd .. && python run_all_tests.py --include "test_hf_*"
   ```

## Rollback Plan

In case of issues, we have backups:

1. **Restore original files from .bak versions**
   ```bash
   for file in ../test_hf_*.py.bak; do 
     cp $file "${file%.bak}"; 
   done
   ```

2. **Keep both versions temporarily**
   ```bash
   # Rename fixed files with .fixed extension
   for file in ../test_hf_*.py; do
     cp $file "$file.fixed";
   done
   
   # Use original files
   for file in ../test_hf_*.py.bak; do
     cp $file "${file%.bak}";
   done
   ```

## Documentation Updates

1. **Add a section to the main README about the indentation standards**
   ```markdown
   ## Indentation Standards
   
   All Python files in this project follow these indentation rules:
   - Top-level code: 0 spaces
   - Class definitions: 0 spaces
   - Class methods: 4 spaces
   - Method content: 8 spaces
   - Nested blocks: 12 spaces
   ```

2. **Create a pull request template with indentation checklist**
   ```markdown
   ## Checklist
   - [ ] Code follows project's indentation standards
   - [ ] All new and modified files pass `python -m py_compile`
   - [ ] Tests have been added for new functionality
   ```

## Timeline

- **Day 1**: Deploy fixed files for core model families (bert, gpt2, t5, vit)
- **Day 2**: Generate and test additional model families
- **Day 3**: Fix complex model families and update documentation
- **Day 4**: Replace test generator and perform comprehensive testing
- **Day 5**: Monitor for issues and provide support

## Success Criteria

The integration will be considered successful when:

1. All test files pass syntax validation
2. All test files run successfully with expected results
3. The test generator produces correctly indented files
4. Documentation is updated with indentation standards
5. Process is in place for maintaining indentation quality

## Maintenance Plan

To ensure ongoing indentation quality:

1. **Add a CI check for Python syntax**
   ```yaml
   # .github/workflows/syntax-check.yml
   name: Python Syntax Check
   on: [push, pull_request]
   jobs:
     syntax-check:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Set up Python
           uses: actions/setup-python@v2
           with:
             python-version: '3.8'
         - name: Check Python syntax
           run: |
             find . -name "*.py" -exec python -m py_compile {} \;
   ```

2. **Add a pre-commit hook for Python indentation**
   ```bash
   # .git/hooks/pre-commit
   #!/bin/bash
   
   # Check Python files for indentation issues
   files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
   if [ -n "$files" ]; then
     python -m py_compile $files || {
       echo "Python syntax check failed. Please fix indentation issues."
       exit 1
     }
   fi
   ```