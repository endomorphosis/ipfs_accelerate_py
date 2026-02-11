# Template Extension Guide

This guide explains how to extend the template system to support new model architectures and specific models.

## Adding New Model Architectures

### 1. Create a New Template File

Create a new template file in the `TEMPLATES_DIR` directory:

```python
#!/usr/bin/env python3

# Import hardware detection capabilities if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback

# ... standard imports ...

# Models registry - Maps model IDs to their specific configurations
NEW_ARCHITECTURE_MODELS_REGISTRY = {
    # Example model entry
    "example/model-base": {
        "description": "Example model description",
        "class": "ExampleModelClass",
        "default_model": "example/model-base",
        "architecture": "new-architecture",
        "task": "example-task"
    }
}

class TestNewArchitectureModels:
    """Base test class for all new architecture models."""
    
    def __init__(self, model_id=None):
        # ... initialization logic ...
    
    def test_pipeline(self, device="auto"):
        # ... pipeline test logic ...
    
    def test_from_pretrained(self, device="auto"):
        # ... from_pretrained test logic ...
    
    # ... other test methods ...

# ... main function and other helpers ...
```

### 2. Update Architecture Types

Add the new architecture to `ARCHITECTURE_TYPES` in `model_template_fixes.py`:

```python
ARCHITECTURE_TYPES = {
    # ... existing architectures ...
    "new-architecture": {
        "template": "new_architecture_template.py",
        "registry_name": "NEW_ARCHITECTURE_MODELS_REGISTRY",
        "models": ["example-model"]
    }
}
```

### 3. Create Model Registry

In the template file, create a model registry:

```python
NEW_ARCHITECTURE_MODELS_REGISTRY = {
    # Example model entry
    "example/model-base": {
        "description": "Example model description",
        "class": "ExampleModelClass",
        "default_model": "example/model-base",
        "architecture": "new-architecture",
        "task": "example-task"
    }
}
```

### 4. Test the New Architecture

Generate a test file using the new architecture:

```bash
python model_template_fixes.py --generate-model example-model --verify
```

## Adding New Models to Existing Architectures

### 1. Add Model Configuration

Add a new model to `MODEL_CONFIG` in `model_template_fixes.py`:

```python
MODEL_CONFIG = {
    # ... existing models ...
    "new-model": {
        "architecture": "existing-architecture",
        "model_id": "org/new-model-id",
        "class_name": "NewModelClass",
        "task": "model-task",
        "test_inputs": {
            "text": "Example input text",
            "image": "test.jpg"  # If applicable
        },
        "processor_class": "AutoProcessor",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_new_model.py"),
        "custom_imports": [
            "import numpy as np"
        ],
        "special_handling": """
        # Special handling code for the model
        special_input = "Example special input"
        """
    }
}
```

### 2. Update Architecture Types

Add the new model to the appropriate architecture in `ARCHITECTURE_TYPES`:

```python
ARCHITECTURE_TYPES = {
    "existing-architecture": {
        "template": "existing_template.py",
        "registry_name": "EXISTING_MODELS_REGISTRY",
        "models": ["existing-model", "new-model"]  # Add the new model here
    }
}
```

### 3. Generate the Test File

Generate a test file for the new model:

```bash
python model_template_fixes.py --generate-model new-model --verify
```

## Special Handling for Different Model Types

### Vision Models

For vision models that require image handling:

```python
"special_handling": """
# Create a dummy image for testing if needed
if not os.path.exists(test_image_path):
    dummy_image = Image.new('RGB', (224, 224), color='white')
    dummy_image.save(test_image_path)
"""
```

### Speech Models

For speech models that require audio handling:

```python
"special_handling": """
# Create a dummy audio file for testing if needed
if not os.path.exists(test_audio_path):
    sample_rate = 16000
    dummy_audio = np.random.randn(sample_rate * 2)  # 2 seconds of random noise
    # Save as WAV file using scipy
    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(test_audio_path, sample_rate, dummy_audio.astype(np.float32))
    except ImportError:
        # Alternative: save using numpy directly
        with open(test_audio_path, 'wb') as f:
            np.save(f, dummy_audio.astype(np.float32))
"""
```

### Multimodal Models

For multimodal models that require multiple input types:

```python
"special_handling": """
# Create dummy inputs for testing
# Image input
if not os.path.exists(test_image_path):
    dummy_image = Image.new('RGB', (224, 224), color='white')
    dummy_image.save(test_image_path)
    
# Text input
text_prompt = "This is a test prompt for multimodal processing."

# Combined input
combined_input = {
    "image": test_image_path,
    "text": text_prompt
}
"""
```

## Best Practices

### Template Structure

- Use a consistent template structure
- Include hardware detection
- Add mock object support
- Include comprehensive test methods
- Support result collection

### Model Configuration

- Use descriptive model names
- Include all required parameters
- Document special handling code
- Avoid duplicate imports
- Use proper indentation in special handling code

### Indentation Management

Ensure proper indentation in special handling code:

```python
def customize_template(template_content, model_name, model_config):
    # ... other customization logic ...
    
    # Find the indentation used in the 'try' block
    indentation = 0
    for j in range(try_index + 1, min(try_index + 10, len(lines))):
        if lines[j].strip():
            indentation = len(lines[j]) - len(lines[j].lstrip())
            break
    
    # Format special handling code with proper indentation
    formatted_code = []
    for line in special_handling.strip().split('\n'):
        formatted_code.append(f"{' ' * indentation}{line.strip()}")
    
    # Insert at the right position
    lines.insert(try_index + 1, "\n".join(formatted_code))
```

### Testing

Test template customization thoroughly:

1. Generate a test file:
   ```bash
   python model_template_fixes.py --generate-model new-model
   ```

2. Verify the syntax:
   ```bash
   python model_template_fixes.py --verify-model new-model
   ```

3. Run the test file:
   ```bash
   python /path/to/fixed_tests/test_hf_new_model.py --help
   ```

4. Check for runtime errors:
   ```bash
   python /path/to/fixed_tests/test_hf_new_model.py
   ```

## Example: Adding a New Model

Here's a complete example of adding a new model:

```python
# 1. Add to MODEL_CONFIG
MODEL_CONFIG["roberta-large"] = {
    "architecture": "encoder-only",
    "model_id": "roberta-large",
    "class_name": "RobertaForSequenceClassification",
    "task": "text-classification",
    "test_inputs": {
        "text": "This is a test input for RoBERTa."
    },
    "processor_class": "AutoTokenizer",
    "source_file": os.path.join(FINAL_MODELS_DIR, "test_roberta_large.py"),
    "custom_imports": [],
    "special_handling": """
    # RoBERTa-specific setup
    max_length = 512
    """
}

# 2. Update ARCHITECTURE_TYPES
# No need - "encoder-only" already includes "roberta"

# 3. Generate the test file
# python model_template_fixes.py --generate-model roberta-large --verify
```

## Troubleshooting

If you encounter issues, consult the [Troubleshooting Guide](TROUBLESHOOTING.md).