# HuggingFace Test Development Guide

This guide provides detailed instructions for creating and maintaining HuggingFace model test files.

## Table of Contents

1. [Introduction](#introduction)
2. [Test File Structure](#test-file-structure)
3. [Creating Tests for New Models](#creating-tests-for-new-models)
4. [Architecture-Specific Considerations](#architecture-specific-considerations)
5. [Testing Best Practices](#testing-best-practices)
6. [Common Pitfalls](#common-pitfalls)
7. [Advanced Topics](#advanced-topics)
8. [Development Workflow](#development-workflow)

## Introduction

Our testing framework ensures compatibility between IPFS Accelerate Python and all HuggingFace model architectures. Well-structured tests are essential for catching regressions, verifying compatibility, and measuring performance across platforms.

## Test File Structure

Each HuggingFace model test file follows this structure:

```python
#!/usr/bin/env python3

# 1. Standard imports
import os, sys, json, time, datetime, logging, argparse, traceback
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional

# 2. Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 3. Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 4. Try to import required packages with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# 5. Hardware detection function
def check_hardware():
    """Check available hardware and return capabilities."""
    # Implementation details

# 6. Models registry
MODEL_REGISTRY = {
    "model-id": {
        "description": "model description",
        "class": "ModelClass",
    }
}

# 7. Test class
class TestModelFamily:
    """Test class for model family."""
    
    def __init__(self, model_id=None):
        """Initialize the test class."""
        # Implementation details
    
    def test_pipeline(self, device="auto"):
        """Test the model using pipeline API."""
        # Implementation details
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        # Implementation details

# 8. Command-line entry point
def main():
    """Command-line entry point."""
    # Implementation details

if __name__ == "__main__":
    main()
```

## Creating Tests for New Models

### Using the Test Generator

The easiest way to create a new test file is using our test generator:

```bash
# Generate a test for a specific model family
python test_generator.py --family MODEL_FAMILY --output ./output_dir

# Example
python test_generator.py --family llama --output ./test
```

### Manual Test Creation

If creating a test file manually, follow these steps:

1. Identify the model's architecture (encoder-only, decoder-only, encoder-decoder, or vision)
2. Copy a template from a similar model type
3. Customize the registry, class name, and model-specific configurations
4. Update the test inputs and task type
5. Validate the syntax and functionality

## Architecture-Specific Considerations

### Encoder-Only Models (BERT, RoBERTa, etc.)

Key considerations:
- Task type: typically "fill-mask"
- Test input: text with mask token (e.g., "[MASK]" or "\<mask\>")
- Example: "The man worked as a [MASK]."

Template example:
```bash
python test_generator.py --family bert --output ./output_dir
```

### Decoder-Only Models (GPT-2, LLaMA, etc.)

Key considerations:
- Task type: typically "text-generation"
- Test input: prompt for text completion
- Special handling: set `tokenizer.pad_token = tokenizer.eos_token`
- Example: "Once upon a time,"

Template example:
```bash
python test_generator.py --family gpt2 --output ./output_dir
```

### Encoder-Decoder Models (T5, BART, etc.)

Key considerations:
- Task type: typically "translation" or "summarization"
- Test input: text for translation/summarization
- Special handling: prepare decoder inputs
- Example: "Translate English to French: Hello, how are you?"

Template example:
```bash
python test_generator.py --family t5 --output ./output_dir
```

### Vision Models (ViT, Swin, etc.)

Key considerations:
- Task type: typically "image-classification"
- Test input: image tensor or path to image
- Special handling: proper tensor shape
- Example: tensor of shape (batch_size, channels, height, width)

Template example:
```bash
python test_generator.py --family vit --output ./output_dir
```

## Testing Best Practices

1. **Proper Indentation**: Maintain consistent 4-space indentation throughout the file.

2. **Error Handling**: Always use try/except blocks to gracefully handle failures.

3. **Mock Dependencies**: Use MagicMock for unavailable dependencies to make tests resilient.

4. **Hardware Detection**: Include comprehensive hardware detection to adapt to available resources.

5. **Performance Metrics**: Collect timing information for model loading and inference.

6. **Descriptive Logging**: Use clear log messages to track test progress and failures.

7. **Result Collection**: Store results in structured JSON format for analysis.

8. **Command-Line Arguments**: Provide flexible options for model selection and hardware preferences.

## Common Pitfalls

1. **Indentation Errors**: Inconsistent indentation is the most common issue. Always use 4 spaces.

2. **Missing Dependencies**: Ensure package imports are wrapped in try/except blocks.

3. **Hardcoded Paths**: Avoid hardcoded paths; use relative paths or command-line arguments.

4. **Inappropriate Task Types**: Using wrong task types for specific model architectures.

5. **Incorrect Input Formats**: Failing to format inputs according to model requirements.

6. **Ignoring Hardware Capabilities**: Not adapting to available hardware resources.

7. **Insufficient Error Handling**: Failing to catch and report specific error types.

## Advanced Topics

### Testing Multiple Model Variants

For families with multiple variants, expand the registry:

```python
MODEL_REGISTRY = {
    "bert-base-uncased": {
        "description": "BERT base model (uncased)",
        "class": "BertModel",
    },
    "bert-large-uncased": {
        "description": "BERT large model (uncased)",
        "class": "BertModel",
    }
}
```

### Custom Input Processors

For complex inputs, implement custom processors:

```python
def prepare_input(self, text):
    """Prepare input for the model."""
    # Custom preprocessing logic
    return processed_input
```

### Hardware-Specific Testing

Implement specific testing logic for different hardware:

```python
def test_openvino(self):
    """Test with OpenVINO backend."""
    # OpenVINO-specific logic
```

## Development Workflow

1. **Generate Test File**:
   ```bash
   python test_generator.py --family MODEL_FAMILY --output ./test
   ```

2. **Verify Syntax**:
   ```bash
   python -m py_compile ./test/test_hf_MODEL_FAMILY.py
   ```

3. **Test Functionality**:
   ```bash
   cd test
   python test_hf_MODEL_FAMILY.py --model MODEL_ID --cpu-only
   ```

4. **Run CI Validation**:
   ```bash
   cd test/skills
   python test_generator_test_suite.py
   ```

5. **Generate Coverage Report**:
   ```bash
   cd test/skills
   python visualize_test_coverage.py
   ```

6. **Submit for Review**:
   - Ensure the test passes all CI checks
   - Include summary of model-specific adaptations
   - Reference any relevant HuggingFace documentation

## Additional Resources

- **Test Generator**: `/test/test_generator.py`
- **Test Suite**: `/test/skills/test_generator_test_suite.py`
- **Coverage Visualization**: `/test/skills/visualize_test_coverage.py`
- **Regeneration Script**: `/test/skills/regenerate_model_tests.sh`
- **Architecture Templates**: Examples for each architecture in `/test/test_hf_*.py`

---

Last updated: March 19, 2025