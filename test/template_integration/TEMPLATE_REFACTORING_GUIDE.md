# Template Integration with the Refactored Test Suite

## Overview

This guide explains how to update the template generation system to produce tests that are compatible with the refactored test suite structure. Our goal is to ensure all newly generated tests follow our new standardized approach while maintaining all the customization capabilities needed for different model architectures.

## Key Objectives

1. Integrate proper base class inheritance in templates
2. Generate standard setup/teardown methods
3. Ensure template customization preserves refactored structure
4. Add verification to check compliance with new standards

## Base Class Integration

All templates must be updated to inherit from the appropriate base class based on the model type:

```python
# Current template class structure:
class Test<ModelName>:
    """Test class for <model_name> model."""
    
    def __init__(self):
        # initialization code...
```

```python
# Updated template class structure:
from refactored_test_suite.model_test import ModelTest

class Test<ModelName>(ModelTest):
    """Test class for <model_name> model."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # model-specific setup code...
```

## StandardTest Method Structure

The following methods must be consistently implemented across all templates:

1. `setUp()` - Replace initialization methods
2. `tearDown()` - Add proper cleanup
3. `test_model_loading()` - Standard test for model loading
4. `test_basic_inference()` - Standard inference test
5. `test_hardware_compatibility()` - Standard hardware compatibility test

## Template Customization Updates

The template customization system must be updated to:

1. Map model architectures to appropriate base classes
2. Convert initialization code to setUp/tearDown methods
3. Standardize assertion patterns
4. Preserve special handling code with proper indentation

## Implementation Details

### 1. Base Class Mapping

Create a mapping between model architectures and base classes:

```python
BASE_CLASS_MAPPING = {
    "text": "ModelTest",
    "vision": "ModelTest",
    "audio": "ModelTest",
    "multimodal": "ModelTest",
    # Add other specialized mappings if needed
}
```

### 2. Import Statements

Add appropriate import statements to all templates:

```python
from refactored_test_suite.<base_module> import <BaseClass>
```

### 3. Method Conversion

Transform existing class methods to adhere to the unittest-style structure:

- Convert `__init__` to `setUp`
- Add proper `super().setUp()` calls
- Ensure all test methods start with `test_`
- Add proper assertions using base class utilities

### 4. Special Handling Preservation

Ensure that special handling code (like for images or audio) is preserved with proper indentation and wrapped in appropriate setUp/tearDown methods.

## Example Transformation

### Before:

```python
class TestBertModel:
    """Test class for BERT model."""
    
    def __init__(self):
        """Initialize the test with model details."""
        self.model_name = "bert-base-uncased"
        self.model_type = "text"
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection."""
        # Hardware detection code...
    
    def test_inference(self):
        """Run a basic inference test."""
        # Inference test code...
```

### After:

```python
from refactored_test_suite.model_test import ModelTest

class TestBertModel(ModelTest):
    """Test class for BERT model."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model_name = "bert-base-uncased"
        self.model_type = "text"
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection."""
        # Hardware detection code...
    
    def test_basic_inference(self):
        """Test that the model can perform basic inference."""
        # Updated inference test code with standard assertions
        model = self.load_model()
        result = model.predict("Hello world")
        self.assertIsNotNone(result)
        # More assertions...
```

## Implementation Plan

1. **Update Template Files**
   - Modify all base templates in `/template_integration/templates/` directory
   - Add refactored imports and base classes
   - Update method signatures

2. **Enhance Customization Function**
   - Update the `customize_template()` function to handle refactored structure
   - Ensure proper indentation of special handling code
   - Add verification for compliance with new standards

3. **Validate Generated Files**
   - Add additional checks to verify_test_file()
   - Ensure all generated files inherit from correct base class
   - Validate that required methods are present

4. **Update Generator Script**
   - Modify main generator to handle new structure
   - Add option to generate tests in refactored format
   - Create a command-line argument to specify target directory in refactored structure

## Testing the Integration

To test the template-refactored test integration:

```bash
# Generate a test file with refactored structure
python fix_template_issues.py --generate-specific bert --refactored

# Run the generated test
python run_refactored_test_suite.py --subdirs models/text
```

## Next Steps

After updating the templates to work with the refactored test suite:

1. Regenerate all model tests to follow the new structure
2. Create a validation script to verify all generated tests pass
3. Document the new structure in the template system documentation
4. Update CI/CD pipeline to use the refactored test runner