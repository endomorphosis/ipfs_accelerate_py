# Template-Based Test Generator Integration

This component of the Distributed Testing Framework enables dynamic generation of tests based on templates stored in the DuckDB database. It provides a flexible way to generate tests for different model types and hardware platforms without duplicating code.

## Overview

The Template-Based Test Generator Integration provides the following features:

- **Template Database Management**: Store and retrieve test templates in DuckDB
- **Model Family Detection**: Automatically determine model families for proper template selection
- **Variable Substitution**: Replace placeholders in templates with model-specific values
- **Test Dependency Management**: Set up proper execution order for tests
- **Resource Estimation**: Calculate memory requirements and priorities based on model type and batch size
- **Template Reuse**: Use the same template for multiple models within a family

## Setup

### Dependencies

The Template-Based Test Generator Integration requires the following dependencies:

```bash
pip install duckdb
```

Additional dependencies for the generated tests:
```bash
pip install torch transformers numpy
```

### Template Database Initialization

Before using the integration, you need to initialize the template database:

```bash
# Create a new template database
python -m duckdb_api.distributed_testing.test_generator_integration \
  --template-db ./templates.duckdb \
  --add-template \
  --template-name "text_embedding_template" \
  --model-family "text_embedding" \
  --template-file "templates/text_embedding.py"
```

## Usage

### Adding Templates

Add templates for different model families:

```bash
# Add text embedding template
python -m duckdb_api.distributed_testing.test_generator_integration \
  --template-db ./templates.duckdb \
  --add-template \
  --template-name "text_embedding_template" \
  --model-family "text_embedding" \
  --template-file "templates/text_embedding.py"

# Add vision template
python -m duckdb_api.distributed_testing.test_generator_integration \
  --template-db ./templates.duckdb \
  --add-template \
  --template-name "vision_template" \
  --model-family "vision" \
  --template-file "templates/vision.py" \
  --hardware "cuda"
```

### Adding Model Mappings

Map specific models to their model families:

```bash
# Map BERT to text_embedding
python -m duckdb_api.distributed_testing.test_generator_integration \
  --template-db ./templates.duckdb \
  --add-mapping \
  --model "bert-base-uncased" \
  --model-family "text_embedding" \
  --description "BERT base uncased model"
```

### Generating Tests

Generate tests for specific models with hardware and batch size configurations:

```bash
# Generate tests for BERT on CPU and CUDA with batch sizes 1 and 4
python -m duckdb_api.distributed_testing.test_generator_integration \
  --template-db ./templates.duckdb \
  --generate \
  --model "bert-base-uncased" \
  --hardware "cpu,cuda" \
  --batch-sizes "1,4"
```

### Listing Templates

List all available templates in the database:

```bash
python -m duckdb_api.distributed_testing.test_generator_integration \
  --template-db ./templates.duckdb \
  --list-templates
```

## Template Format

Templates should be Python files with placeholder variables for model-specific values. The placeholders use the format `${variable_name}` and will be replaced during test generation.

### Example Template for Text Embedding Models

```python
# Test for ${model_name} on ${hardware_type} with batch size ${batch_size}
import torch
from transformers import AutoModel, AutoTokenizer

def test_${model_family}_${hardware_type}():
    # Initialize model
    model_name = "${model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to appropriate device
    device = "${hardware_type}"
    if device != "cpu":
        model = model.to(device)
    
    # Prepare input
    text = "Example text for embedding test"
    inputs = tokenizer(text, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference with batch size ${batch_size}
    batch = {k: v.repeat(${batch_size}, 1) for k, v in inputs.items()}
    
    # Execute model
    with torch.no_grad():
        outputs = model(**batch)
    
    # Return results
    return {
        "model_name": "${model_name}",
        "hardware_type": "${hardware_type}",
        "batch_size": ${batch_size},
        "embedding_shape": outputs.last_hidden_state.shape,
        "success": True
    }
```

## Integration with Coordinator

The Template Generator Integration can be integrated with the Coordinator to automate test submission:

```python
from test_generator_integration import TestGeneratorIntegration
from coordinator_client import CoordinatorClient

# Initialize components
coordinator_client = CoordinatorClient("http://localhost:8080")
generator = TestGeneratorIntegration(
    template_db_path="./templates.duckdb",
    coordinator_client=coordinator_client
)

# Generate and submit tests
success, tests = generator.generate_and_submit_tests(
    model_name="bert-base-uncased",
    hardware_types=["cpu", "cuda", "rocm"],
    batch_sizes=[1, 4, 8, 16]
)

# Print results
print(f"Generated and submitted {len(tests)} tests")
```

## Testing

The implementation includes comprehensive unit tests to verify all functionality:

```bash
# Install test dependencies
pip install pytest

# Run tests
python -m pytest duckdb_api/distributed_testing/test_template_generator.py -v
```

## Key Components

- **TestGeneratorIntegration**: Main class for template-based test generation
- **Template Database**: DuckDB database for storing templates and model mappings
- **Model Family Detection**: Automatic detection of model families based on name
- **Variable Substitution**: Replacement of placeholders in templates
- **Test Dependency Management**: Setting up proper execution order for tests
- **Resource Estimation**: Calculating memory requirements and priorities