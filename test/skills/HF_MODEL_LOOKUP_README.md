# HuggingFace Model Lookup and Registry

This document explains how to use the enhanced model lookup features to find appropriate models for your tests and ensure they are correct and accessible.

## Overview

The `find_models.py` script provides several features to help you find and verify HuggingFace models:

1. **Model Discovery**: Find popular models for a specific model type
2. **API Integration**: Query the HuggingFace API to find the most downloaded models
3. **Model Verification**: Verify that a model exists and is accessible
4. **Registry Management**: Maintain a registry of default models for each model type
5. **Smart Model Selection**: Automatically select smaller, more suitable models for testing

## Usage Examples

### Find Popular Models for a Type

To find popular models for a specific model type (e.g., "bert"), use:

```bash
python find_models.py --model-type bert
```

This will query the HuggingFace API and display the most downloaded models of that type, along with a recommended default model.

### Verify a Specific Model

To verify that a specific model exists and is accessible:

```bash
python find_models.py --verify "bert-base-uncased"
```

### Update the Model Registry

To update the registry with the latest information from HuggingFace:

```bash
python find_models.py --update-registry
```

This will create or update the `huggingface_model_types.json` file with the latest model information.

### Query and Update a Specific Model Type

To query a specific model type and update its entry in the registry:

```bash
python find_models.py --model-type whisper --update-registry
```

### Find All Models

To discover all available model types in the transformers library:

```bash
python find_models.py
```

### Focus on Hyphenated Models

To find models with hyphenated names (which need special handling):

```bash
python find_models.py --hyphenated-only
```

## Smart Model Selection

The script will automatically recommend suitable default models based on several criteria:

1. **Popularity**: Models with more downloads are preferred
2. **Size**: Smaller models are preferred when available
   - Models with "base" or "small" in their name are prioritized
3. **Accessibility**: Only models that are publicly accessible are considered

## Registry File

The model registry is stored in `huggingface_model_types.json` and contains:

```json
{
  "bert": {
    "default_model": "bert-base-uncased",
    "models": ["bert-base-uncased", "bert-large-uncased", "..."],
    "downloads": {
      "bert-base-uncased": 10000000,
      "bert-large-uncased": 5000000
    },
    "updated_at": "2025-03-20T12:34:56.789Z"
  },
  "gpt2": {
    "default_model": "gpt2",
    "models": ["gpt2", "gpt2-medium", "..."],
    "downloads": {
      "gpt2": 15000000,
      "gpt2-medium": 7500000
    },
    "updated_at": "2025-03-20T12:34:56.789Z"
  },
  // ...
}
```

## Integrating with Test Generator

The test generator has been enhanced with direct integration to use the model lookup system for dynamic model selection:

```python
# Full integration with the test generator
python integrate_model_lookup.py --update-all

# Integration with a specific model update
python integrate_model_lookup.py --model gpt2

# Integration only (no registry updates)
python integrate_model_lookup.py
```

The test generator now uses a sophisticated approach to select the most appropriate models:

```python
def get_model_from_registry(model_type):
    """Get the best default model for a model type, using dynamic lookup if available."""
    if HAS_MODEL_LOOKUP:
        # Try to get a recommended model from the HuggingFace API
        try:
            default_model = get_recommended_default_model(model_type)
            logger.info(f"Using recommended model for {model_type}: {default_model}")
            return default_model
        except Exception as e:
            logger.warning(f"Error getting recommended model for {model_type}: {e}")
            # Fall back to registry lookup
    
    # Use the static registry as fallback
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type].get("default_model")
    
    # For unknown models, use a heuristic approach
    return f"{model_type}-base" if "-base" not in model_type else model_type
```

This implementation provides:

1. **Dynamic Lookup**: When available, uses the HuggingFace API to find the best model
2. **Registry Fallback**: If API lookup fails, falls back to the stored registry data
3. **Static Fallback**: If the model is not in the registry, uses a heuristic approach
4. **Graceful Degradation**: Works even if dependencies or network access are unavailable

## Handling Hyphenated Models

Special handling is included for hyphenated model names (e.g., "gpt-j", "xlm-roberta"), which are normalized to valid Python identifiers and mapped to the correct class names.

## Debugging Issues

If you encounter model loading errors:

1. Verify the model exists using `--verify`
2. Try a different model of the same type with `--model-type`
3. Check the model registry with `cat huggingface_model_types.json`
4. Update the registry with `--update-registry`

## CI/CD Considerations

In CI/CD environments, API access might be restricted or unreliable. The system accounts for this with:

1. **Fallback Mechanism**: If API access fails, falls back to the registry or default patterns
2. **Registry Caching**: Uses the previously saved registry without requiring API access
3. **Dry Run Mode**: Supports a `--dry-run` flag to preview changes without applying them
4. **Environment Variables**: Can be configured to skip API calls in specific environments:
   ```bash
   # Force use of the registry without API calls
   export HF_API_DISABLED=1
   python integrate_model_lookup.py
   ```

## Best Practices

1. **Regular Registry Updates**: Update the registry periodically (weekly) to keep model recommendations current
2. **Pre-Commit Updates**: Update the registry before major test generation sessions
3. **Version Control**: Commit the registry file to maintain a history of model changes
4. **CI/CD Integration**: Set up a recurring job to update the registry automatically
5. **Verification**: Before large test runs, verify critical models with the `--verify` flag

## Next Steps for Development

1. ✅ Implement automatic registry updates in the CI/CD pipeline - COMPLETED
2. ✅ Add smart model selection to prioritize "base" and "small" variants - COMPLETED
3. ✅ Direct integration with test generator for dynamic model selection - COMPLETED
4. Add model size information to further refine smaller model selection
5. Expand model verification to check task compatibility
6. Add specialized model suggestion features for different tasks
7. Implement automatic detection of model architecture families
8. Create a model migration detection system to track renamed or deprecated models