# Advanced Model Selection for HuggingFace Test Generator

This document describes the enhanced model selection system that provides task-specific,
hardware-aware, and framework-compatible model recommendations for the HuggingFace test generator.

## Overview

The advanced model selection system:

1. Selects the most appropriate HuggingFace models based on multiple criteria:
   - Task compatibility (e.g., text-classification, image-segmentation)
   - Hardware constraints (e.g., CPU/GPU, memory limits)
   - Framework compatibility (PyTorch, TensorFlow, JAX)
   - Download popularity (trending and well-maintained models)
   - Benchmark performance (when available)

2. Provides robust fallback mechanisms:
   - API queries for fresh, popularity-based selection
   - Registry-based caching for offline/CI environments
   - Size-based fallbacks (e.g., large → base → small → tiny)
   - Static defaults as final fallbacks

## Using Advanced Selection

### Command Line Options

The `test_generator_fixed.py` now supports the following advanced options:

```bash
# Basic model generation with automatic model selection
python test_generator_fixed.py --generate bert

# Task-specific model selection
python test_generator_fixed.py --generate bert --task text-classification

# Hardware-constrained model selection
python test_generator_fixed.py --generate gpt2 --hardware cpu-small

# Size-constrained model selection
python test_generator_fixed.py --generate t5 --max-size 500

# Framework-specific model selection
python test_generator_fixed.py --generate bert --framework pytorch
```

### Testing Advanced Features

Use the provided test script to explore different selection options:

```bash
# Test all selection methods for a model type
python test_model_lookup_advanced.py --model-type bert

# Test with task constraints
python test_model_lookup_advanced.py --model-type bert --task text-classification

# Detect available hardware
python test_model_lookup_advanced.py --detect-hardware

# List suitable tasks for a model type
python test_model_lookup_advanced.py --model-type bert --list-tasks
```

## Supported Tasks

The system supports the following tasks mapped to appropriate model types:

| Task | Description | Suitable Model Types |
|------|-------------|---------------------|
| text-classification | Text classification/categorization | bert, roberta, distilbert, xlm-roberta |
| token-classification | Token-level tasks (NER, POS) | bert, roberta, distilbert |
| question-answering | Question answering | bert, roberta, distilbert |
| text-generation | Open-ended text generation | gpt2, gpt-j, gpt-neo, llama |
| summarization | Text summarization | t5, bart, pegasus |
| translation | Machine translation | t5, mbart, mt5 |
| image-classification | Image classification | vit, resnet, swin |
| image-segmentation | Image segmentation | mask2former, segformer |
| automatic-speech-recognition | Speech recognition | whisper, wav2vec2, hubert |
| visual-question-answering | VQA | llava, blip |

## Hardware Profiles

The system includes predefined hardware profiles for common scenarios:

| Profile | Description | Maximum Model Size |
|---------|-------------|-------------------|
| cpu-small | Limited CPU environments | 500 MB |
| cpu-medium | Standard CPU environments | 2,000 MB |
| cpu-large | High-memory CPU environments | 10,000 MB |
| gpu-small | Entry-level GPUs | 5,000 MB |
| gpu-medium | Mid-range GPUs | 15,000 MB |
| gpu-large | High-end GPUs | 50,000 MB |

## Integration with CI/CD

The system includes a GitHub Actions workflow for automated registry updates:

1. Weekly updates of model popularity data
2. Verification of new model selections
3. Pull request creation for registry changes
4. Test sample generation for verification

Use the following environment variables to control behavior in CI/CD:

- `USE_STATIC_REGISTRY=true` - Disable API calls and use cached registry
- `FORCE_API_CALLS=true` - Force API calls even in CI environments
- `MAX_REGISTRY_AGE_DAYS=30` - Control maximum age of registry before refresh

## Extending the System

### Adding New Tasks

To add a new task, update the `TASK_TO_MODEL_TYPES` dictionary in `advanced_model_selection.py`:

```python
TASK_TO_MODEL_TYPES = {
    # Existing tasks...
    "new-task-name": ["model-type1", "model-type2"],
}
```

### Adding New Hardware Profiles

To add a new hardware profile, update the `HARDWARE_PROFILES` dictionary:

```python
HARDWARE_PROFILES = {
    # Existing profiles...
    "new-profile-name": {
        "max_size_mb": 8000,
        "description": "Description of the profile"
    },
}
```

## Troubleshooting

### API Rate Limiting

If you encounter HuggingFace API rate limiting:

1. Set a longer delay between requests
2. Use the cached registry data with `USE_STATIC_REGISTRY=true`
3. Update specific model types individually rather than all at once

### Size Estimation Errors

If model size estimation is inaccurate:

1. Provide an explicit size constraint with `--max-size`
2. Add size indicators in the model registry
3. Update the size estimation heuristics in `estimate_model_size()`

## Components

The advanced model selection system consists of:

- `advanced_model_selection.py` - Core selection logic
- `find_models.py` - HuggingFace API interaction
- Integration with `test_generator_fixed.py`
- Support files (registry, benchmarks, etc.)

## Development

When developing the system:

1. Run tests with `python test_advanced_model_selection.py`
2. Verify integration with `python test_model_lookup_advanced.py`
3. Update documentation when adding new features
