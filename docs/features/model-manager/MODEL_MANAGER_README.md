# Model Manager Documentation

The Model Manager is a comprehensive system for storing and managing metadata about different types of machine learning models. It provides a centralized registry for model information including input/output types, HuggingFace configurations, inference code locations, and performance metrics.

## Features

- **Comprehensive Metadata Storage**: Store detailed information about models including architecture, I/O specifications, performance metrics, and hardware requirements
- **Multiple Storage Backends**: Support for both JSON file storage and DuckDB database storage
- **HuggingFace Integration**: Native support for importing and working with HuggingFace model configurations
- **Input/Output Type Mapping**: Track and query models by their input and output data types
- **Search and Filtering**: Advanced search capabilities by name, tags, architecture, and model type
- **Performance Tracking**: Store and query performance metrics and hardware requirements
- **Integration Support**: Easy integration with existing model metadata systems

## Quick Start

### Basic Usage

```python
from ipfs_accelerate_py.model_manager import (
    ModelManager, ModelMetadata, IOSpec, ModelType, DataType
)

# Create a model manager
manager = ModelManager()

# Create model metadata
model = ModelMetadata(
    model_id="my-model/sentiment-analyzer",
    model_name="Sentiment Analyzer",
    model_type=ModelType.LANGUAGE_MODEL,
    architecture="BertForSequenceClassification",
    inputs=[
        IOSpec(name="input_ids", data_type=DataType.TOKENS),
        IOSpec(name="attention_mask", data_type=DataType.TOKENS, optional=True)
    ],
    outputs=[
        IOSpec(name="logits", data_type=DataType.LOGITS)
    ],
    tags=["sentiment-analysis", "classification"]
)

# Add to registry
manager.add_model(model)

# Retrieve model
retrieved = manager.get_model("my-model/sentiment-analyzer")
print(f"Model: {retrieved.model_name}")
```

### Integration with Existing Systems

```python
from model_manager_integration import ModelManagerIntegration

# Create integration utility
integration = ModelManagerIntegration()

# Import from existing metadata systems
imported_count = integration.import_from_existing_metadata()
print(f"Imported {imported_count} models")

# Add popular HuggingFace models
popular_models = ["bert-base-uncased", "gpt2", "t5-small"]
integration.populate_huggingface_models(popular_models)

# Export registry
integration.export_model_registry("my_models.json")
```

## Core Components

### ModelMetadata

The central data structure containing all model information:

```python
@dataclass
class ModelMetadata:
    model_id: str                              # Unique identifier
    model_name: str                            # Human-readable name
    model_type: ModelType                      # Type classification
    architecture: str                          # Model architecture
    inputs: List[IOSpec]                       # Input specifications
    outputs: List[IOSpec]                      # Output specifications
    huggingface_config: Optional[Dict]         # HF config if applicable
    inference_code_location: Optional[str]     # Path to inference code
    supported_backends: List[str]              # Supported backends
    hardware_requirements: Optional[Dict]      # Hardware requirements
    performance_metrics: Optional[Dict]        # Performance data
    tags: List[str]                           # Searchable tags
    source_url: Optional[str]                 # Source URL
    license: Optional[str]                    # License information
    description: str                          # Description
    created_at: Optional[datetime]            # Creation timestamp
    updated_at: Optional[datetime]            # Update timestamp
```

### IOSpec

Input/Output specification for models:

```python
@dataclass
class IOSpec:
    name: str                     # Input/output name
    data_type: DataType          # Data type (TEXT, IMAGE, AUDIO, etc.)
    shape: Optional[Tuple]       # Tensor shape if applicable
    dtype: str                   # Data type (float32, int64, etc.)
    description: str             # Human-readable description
    optional: bool               # Whether this I/O is optional
```

### Model Types

Supported model type classifications:

- `ModelType.LANGUAGE_MODEL` - General language models
- `ModelType.VISION_MODEL` - Computer vision models
- `ModelType.AUDIO_MODEL` - Audio processing models
- `ModelType.MULTIMODAL` - Models handling multiple modalities
- `ModelType.EMBEDDING_MODEL` - Embedding/encoder models
- `ModelType.ENCODER_DECODER` - Encoder-decoder architectures
- `ModelType.ENCODER_ONLY` - Encoder-only models (e.g., BERT)
- `ModelType.DECODER_ONLY` - Decoder-only models (e.g., GPT)

### Data Types

Supported input/output data types:

- `DataType.TEXT` - Text data
- `DataType.IMAGE` - Image data
- `DataType.AUDIO` - Audio data
- `DataType.VIDEO` - Video data
- `DataType.EMBEDDINGS` - Vector embeddings
- `DataType.TOKENS` - Tokenized text
- `DataType.LOGITS` - Model output logits
- `DataType.FEATURES` - Feature vectors

## Storage Backends

### JSON Storage (Default)

Simple file-based storage using JSON format:

```python
manager = ModelManager(storage_path="./models.json", use_database=False)
```

Features:
- Human-readable format
- Easy to version control
- No external dependencies
- Suitable for smaller registries

### DuckDB Storage

Database storage using DuckDB (requires `duckdb` package):

```python
manager = ModelManager(storage_path="./models.duckdb", use_database=True)
```

Features:
- Better performance for large registries
- SQL query capabilities
- ACID transactions
- Concurrent access support

## Search and Filtering

### Basic Filtering

```python
# Filter by model type
language_models = manager.list_models(model_type=ModelType.LANGUAGE_MODEL)

# Filter by architecture
bert_models = manager.list_models(architecture="BertForMaskedLM")

# Filter by tags
classification_models = manager.list_models(tags=["classification"])
```

### Text Search

```python
# Search by name, description, or tags
sentiment_models = manager.search_models("sentiment")
bert_models = manager.search_models("bert")
```

### Input/Output Type Queries

```python
# Find models by input type
text_models = manager.get_models_by_input_type(DataType.TEXT)

# Find models by output type
embedding_models = manager.get_models_by_output_type(DataType.EMBEDDINGS)

# Find compatible models
compatible = manager.get_compatible_models(DataType.TEXT, DataType.EMBEDDINGS)
```

## HuggingFace Integration

### Creating Models from HF Config

```python
from ipfs_accelerate_py.model_manager import create_model_from_huggingface

hf_config = {
    "architectures": ["BertForMaskedLM"],
    "model_type": "bert",
    "vocab_size": 30522,
    "hidden_size": 768
}

model = create_model_from_huggingface(
    model_id="bert-base-uncased",
    hf_config=hf_config,
    inference_code_location="/path/to/bert_inference.py"
)

manager.add_model(model)
```

### Batch Import

```python
from model_manager_integration import ModelManagerIntegration

integration = ModelManagerIntegration()

# Import multiple models
model_list = [
    "bert-base-uncased",
    "gpt2-medium", 
    "t5-small",
    "facebook/bart-large"
]

count = integration.populate_huggingface_models(model_list)
print(f"Added {count} models")
```

## Performance and Hardware Tracking

### Adding Performance Metrics

```python
model.performance_metrics = {
    "accuracy": 0.94,
    "f1_score": 0.91,
    "inference_time_ms": 23.5,
    "throughput_samples_per_sec": 450,
    "perplexity": 15.2
}

model.hardware_requirements = {
    "min_memory_gb": 4,
    "recommended_memory_gb": 8,
    "min_gpu_memory_gb": 2,
    "cpu_cores": 4,
    "supports_gpu": True,
    "supports_cpu": True
}
```

### Querying Performance Data

```python
# Get statistics
stats = manager.get_stats()
print(f"Total models: {stats['total_models']}")
print(f"Models with performance data: {len([m for m in manager.list_models() if m.performance_metrics])}")

# Find fast models
fast_models = [
    m for m in manager.list_models() 
    if m.performance_metrics and m.performance_metrics.get('inference_time_ms', float('inf')) < 50
]
```

## Export and Import

### Export Registry

```python
# Export to JSON
manager.export_metadata("models_backup.json", format="json")

# Export to YAML (requires PyYAML)
manager.export_metadata("models_backup.yaml", format="yaml")
```

### Integration Export

```python
from model_manager_integration import ModelManagerIntegration

integration = ModelManagerIntegration()

# Generate compatibility matrix
compatibility = integration.generate_compatibility_matrix()
print("Input types:", compatibility['input_types'])
print("Output types:", compatibility['output_types'])

# Export with additional analysis
integration.export_model_registry("full_export.json")
```

## Advanced Usage

### Custom Model Types

```python
# Extend existing enums for custom types
class CustomModelType(Enum):
    MULTIMODAL_LLM = "multimodal_llm"
    RETRIEVAL_MODEL = "retrieval_model"
    REASONING_MODEL = "reasoning_model"

# Use in metadata (will be stored as string)
custom_model = ModelMetadata(
    model_id="custom/reasoning-model",
    model_name="Custom Reasoning Model",
    model_type=CustomModelType.REASONING_MODEL,
    # ... other fields
)
```

### Complex I/O Specifications

```python
# Multi-modal inputs
inputs = [
    IOSpec(
        name="text_input",
        data_type=DataType.TEXT,
        shape=(None,),
        description="Input text query"
    ),
    IOSpec(
        name="image_input", 
        data_type=DataType.IMAGE,
        shape=(3, 224, 224),
        description="Input image",
        optional=True
    ),
    IOSpec(
        name="context_embeddings",
        data_type=DataType.EMBEDDINGS,
        shape=(None, 768),
        description="Context embeddings",
        optional=True
    )
]

# Structured outputs
outputs = [
    IOSpec(
        name="answer_text",
        data_type=DataType.TEXT,
        description="Generated answer"
    ),
    IOSpec(
        name="confidence_score",
        data_type=DataType.FEATURES,
        shape=(1,),
        description="Confidence score [0-1]"
    )
]
```

### Batch Operations

```python
# Add multiple models
models = [model1, model2, model3]
for model in models:
    manager.add_model(model)

# Batch search
search_terms = ["bert", "gpt", "t5"]
all_results = []
for term in search_terms:
    all_results.extend(manager.search_models(term))

# Remove outdated models
old_models = manager.list_models(tags=["deprecated"])
for model in old_models:
    manager.remove_model(model.model_id)
```

## Best Practices

### Model Naming

- Use descriptive, hierarchical IDs: `organization/model-name-variant`
- Include version information when applicable: `my-org/bert-sentiment-v2`
- Use consistent naming conventions across your organization

### Metadata Quality

- Always include meaningful descriptions
- Add relevant tags for discoverability
- Specify hardware requirements for deployment planning
- Include performance metrics when available
- Keep inference code locations up to date

### Organization

- Use consistent tagging schemes
- Group related models using prefixes
- Regular cleanup of deprecated models
- Backup registries before major changes

### Performance

- Use database storage for large registries (>1000 models)
- Index frequently queried fields
- Consider pagination for large result sets
- Cache frequently accessed models

## Error Handling

```python
try:
    manager = ModelManager(storage_path="./models.json")
    
    # Add model with validation
    if not manager.add_model(model):
        print("Failed to add model")
    
    # Safe retrieval
    model = manager.get_model("some-id")
    if model is None:
        print("Model not found")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    manager.close()  # Always close to save changes
```

## Context Manager Usage

```python
# Recommended pattern
with ModelManager(storage_path="./models.json") as manager:
    manager.add_model(my_model)
    results = manager.search_models("bert")
    # Automatically saved and closed
```

## Integration Examples

See `model_manager_example.py` for comprehensive usage examples and `model_manager_integration.py` for integration utilities with existing systems.

## API Reference

For complete API documentation, see the docstrings in the source code. Key classes:

- `ModelManager`: Main registry class
- `ModelMetadata`: Model metadata container
- `IOSpec`: Input/output specification
- `ModelManagerIntegration`: Integration utilities
- `ModelType`, `DataType`: Enumeration types

## Testing

Run the test suite:

```bash
python test_model_manager.py
```

The test suite covers:
- Basic CRUD operations
- Search and filtering
- Storage backends
- HuggingFace integration
- Export/import functionality
- Error handling