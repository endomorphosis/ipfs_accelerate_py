# Hugging Face Test Files Troubleshooting Guide

This guide helps identify and fix common indentation issues in Hugging Face model test files.

## Common Indentation Issues

### 1. Class Method Definition Issues

**Problem:** Method definitions with multiple `self` parameters or incorrect indentation
```python
def test_pipeline(self,(self,(self, device="auto"):
```

**Solution:** Use 4 spaces for method definitions with correct parameters
```python
def test_pipeline(self, device="auto"):
```

### 2. Missing Indentation After Conditionals

**Problem:** No indentation after `if` statements
```python
if not HAS_TOKENIZERS:
class MockTokenizer:
```

**Solution:** Add proper indentation (4 spaces) after the conditional
```python
if not HAS_TOKENIZERS:
    class MockTokenizer:
```

### 3. Inline Statements

**Problem:** Multiple statements on the same line
```python
HAS_TORCH = False        logger.warning("torch not available, using mock")
```

**Solution:** Split into separate lines with proper indentation
```python
HAS_TORCH = False
logger.warning("torch not available, using mock")
```

### 4. Inconsistent Try/Except Blocks

**Problem:** Missing indentation in `try`/`except` blocks
```python
try:
import openvino
capabilities["openvino"] = True
except ImportError:
pass
```

**Solution:** Add proper indentation to block contents
```python
try:
    import openvino
    capabilities["openvino"] = True
except ImportError:
    pass
```

### 5. Nested Block Indentation

**Problem:** Incorrect or missing indentation in nested blocks
```python
if device == "cuda":
try:
_ = pipeline(pipeline_input)
except Exception:
pass
```

**Solution:** Ensure consistent indentation (4 spaces per level)
```python
if device == "cuda":
    try:
        _ = pipeline(pipeline_input)
    except Exception:
        pass
```

### 6. Mock Class Method Indentation

**Problem:** Method definitions in mock classes with incorrect indentation
```python
class MockTokenizer:
def __init__(self, *args, **kwargs):
    self.vocab_size = 32000
    
def decode(self, ids, **kwargs):
    return "Decoded text from mock"
```

**Solution:** Add proper indentation (4 spaces) for class methods
```python
class MockTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 32000
    
    def decode(self, ids, **kwargs):
        return "Decoded text from mock"
```

### 7. If/Else Block Indentation

**Problem:** Missing indentation after `if`/`else`/`elif` statements
```python
if "cuda" in error_str:
results["pipeline_error_type"] = "cuda_error"
elif "memory" in error_str:
results["pipeline_error_type"] = "out_of_memory"
```

**Solution:** Add proper indentation (4 spaces) for the contents of each block
```python
if "cuda" in error_str:
    results["pipeline_error_type"] = "cuda_error"
elif "memory" in error_str:
    results["pipeline_error_type"] = "out_of_memory"
```

## Step-by-Step Troubleshooting Process

1. **Verify the syntax errors**
   ```bash
   python -m py_compile test_file.py
   ```

2. **Check line numbers from the error message**
   - Python errors will specify the line number of syntax issues

3. **Use the automated fixing tools**
   ```bash
   python fix_test_indentation.py test_file.py
   ```

4. **For persistent issues, create a minimal file**
   ```bash
   python create_minimal_test.py --families bert
   ```

## Indentation Guidelines

For consistent Python files, follow these indentation patterns:

- Top-level code: 0 spaces
- Class definitions: 0 spaces
- Class methods: 4 spaces
- Method content: 8 spaces
- Nested blocks: 12 spaces

## Manual Fixing Techniques

If automated tools fail, you can fix files manually:

1. **Fix import mocks**: Ensure proper spacing between import mocks
   ```python
   try:
       import torch
       HAS_TORCH = True
   except ImportError:
       torch = MagicMock()
       HAS_TORCH = False
       logger.warning("torch not available, using mock")
   ```

2. **Fix method definitions**: Ensure proper indentation and parameter structure
   ```python
   def test_pipeline(self, device="auto"):
       """Method docstring."""
       if device == "auto":
           device = self.preferred_device
   ```

3. **Fix mock class definitions**: Ensure proper indentation in mock classes
   ```python
   if not HAS_TOKENIZERS:
       class MockTokenizer:
           def __init__(self, *args, **kwargs):
               self.vocab_size = 32000
           
           def decode(self, ids, **kwargs):
               return "Decoded text from mock"
   ```

4. **Fix conditional blocks**: Ensure consistent indentation in conditional statements
   ```python
   if success:
       print(f"✅ Successfully tested {model_id}")
   else:
       print(f"❌ Failed to test {model_id}")
   ```

## When to Use Minimal Test Files

If a file has severe indentation issues that are difficult to fix, it's often easier to generate a minimal test file:

```bash
python create_minimal_test.py --families bert --output-dir fixed_tests
```

The minimal test files provide a clean, correctly-indented implementation that covers the essential functionality while avoiding complex indentation patterns that can lead to errors.