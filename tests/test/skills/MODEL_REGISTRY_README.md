# HuggingFace Model Registry

This document describes the model registry system that provides dynamic model lookup
for HuggingFace models. The system queries the HuggingFace API to find popular models
and selects appropriate defaults based on download counts and model size.

## Default Models

The following default models are currently recommended by the system:

- **bert**: `google-bert/bert-base-uncased`
- **gpt2**: `datificate/gpt2-small-spanish`
- **t5**: `amazon/chronos-t5-small`
- **vit**: `google/vit-base-patch16-224-in21k`
- **gpt-j**: `mmnga/aibuncho-japanese-novel-gpt-j-6b-gguf`
- **whisper**: `openai/whisper-base.en`

## Usage

To use the model registry in your code:

```python
from find_models import get_recommended_default_model

# Get the recommended default model for a model type
default_model = get_recommended_default_model("bert")
print(f"Using model: {default_model}")
```

The system will:
1. Try to query the HuggingFace API for popular models
2. Select an appropriate model based on popularity and size
3. Fall back to static defaults if the API is unavailable
4. Cache results to avoid repeated API calls

## Model Selection Criteria

Models are selected based on:
1. Download count (popularity)
2. Size preference (base/small models preferred over large ones)
3. Availability and stability

## Implementation

The model lookup system has several components:

1. **find_models.py**: Core module with the following key functions:
   - `query_huggingface_api(model_type, limit=5, min_downloads=1000)`: Queries the HuggingFace API for popular models
   - `get_recommended_default_model(model_type)`: Selects the best model based on size and popularity
   - `try_model_access(model_name)`: Verifies model availability

2. **huggingface_model_types.json**: Registry file for caching model information:
   ```json
   {
     "bert": {
       "default_model": "google-bert/bert-base-uncased",
       "models": ["list", "of", "models"],
       "downloads": {"model": download_count},
       "updated_at": "timestamp"
     }
   }
   ```

3. **test_generator_fixed.py**: Integration with test generator through:
   ```python
   def get_model_from_registry(model_type):
       '''Get the best default model for a model type, using dynamic lookup if available.'''
       if HAS_MODEL_LOOKUP:
           try:
               default_model = get_recommended_default_model(model_type)
               return default_model
           except Exception as e:
               # Fall back to registry lookup
               pass
       
       # Use static registry as fallback
       if model_type in MODEL_REGISTRY:
           return MODEL_REGISTRY[model_type].get("default_model")
       
       # For unknown models, use a heuristic approach
       return f"{model_type}-base" if "-base" not in model_type else model_type
   ```

## How it Works

1. **API Query**: The system queries the HuggingFace API to get a list of popular models of a given type, sorted by download count.

2. **Smart Selection**: From these models, it selects ones with "base" or "small" in the name, preferring smaller models that are still popular enough to be reliable.

3. **Fallback Mechanism**: If the API query fails, the system falls back to cached data in the registry file. If that fails too, it uses static defaults.

4. **Registry Management**: The system updates the registry file periodically to ensure fresh data while limiting API calls.

## Test Files

- **test_model_lookup.py**: Demonstrates basic usage of the model lookup system
- **test_generator_fixed.py**: Shows integration with the test generator

## Adding New Model Types

To add a new model type to the registry:

1. Update the **ARCHITECTURE_TYPES** dictionary in find_models.py to include the new model type
2. Run `python find_models.py --model-type NEW_TYPE --update-registry` to query and add the model

## CI/CD Configuration

For CI/CD environments where API access is not desirable:

1. Set `USE_STATIC_REGISTRY=true` environment variable to skip API calls
2. Ensure huggingface_model_types.json is committed to the repository