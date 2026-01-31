# ONNX Model Verification and Conversion System

## Overview

The ONNX Model Verification and Conversion System provides a robust solution for handling ONNX model files in the IPFS Accelerate Python Framework. This system verifies the existence of ONNX model files on HuggingFace before benchmarking, and automatically converts PyTorch models to ONNX format when necessary, storing them locally for future use.

## Key Features

- **ONNX File Verification**: Checks if ONNX files exist on HuggingFace before attempting to download
- **PyTorch to ONNX Conversion**: Automatic fallback to PyTorch model conversion when ONNX files aren't available
- **Local Model Caching**: Stores converted models in a local cache to avoid redundant conversions
- **Conversion Registry**: Maintains a registry of all conversions with metadata
- **Model-Specific Configurations**: Supports custom conversion parameters for different model types
- **Database Integration**: Records conversion status in benchmark results for reporting and analysis
- **Comprehensive Error Handling**: Graceful handling of verification and conversion errors
- **Command-Line Tools**: Utilities for checking ONNX files and updating database schema
- **Testing Suite**: Comprehensive tests for the verification and conversion system

## Components

The system consists of the following components:

1. **`onnx_verification.py`**: Core utility for verifying ONNX file existence and converting PyTorch models to ONNX
2. **`benchmark_onnx_integration.py`**: Example integration with the benchmark system
3. **`onnx_db_schema_update.py`**: Script for updating the DuckDB schema with ONNX tracking fields
4. **`test_onnx_verification.py`**: Test suite for the ONNX verification and conversion utility
5. **`check_onnx_files.py`**: Command-line utility for checking ONNX file availability

### Class Structure

- **`OnnxVerifier`**: Handles ONNX file verification and manages the conversion registry
- **`PyTorchToOnnxConverter`**: Converts PyTorch models to ONNX format
- **Helper Functions**:
  - `verify_and_get_onnx_model()`: Main integration function for benchmark scripts
  - Custom exception classes for error handling

## Usage

### Basic Usage

```python
from onnx_verification import verify_and_get_onnx_model

# Get model path with verification and fallback conversion
model_path, was_converted = verify_and_get_onnx_model(
    model_id="bert-base-uncased",
    onnx_path="model.onnx"
)

# Use the model path in your application
print(f"Using {'converted' if was_converted else 'original'} model at {model_path}")
```

### Advanced Usage with Custom Conversion Configuration

```python
from onnx_verification import verify_and_get_onnx_model

# Define custom conversion parameters
conversion_config = {
    "model_type": "bert",
    "opset_version": 12,
    "input_shapes": {
        "batch_size": 1,
        "sequence_length": 128
    },
    "input_names": ["input_ids", "attention_mask"],
    "output_names": ["last_hidden_state", "pooler_output"],
    "dynamic_axes": {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
    }
}

# Get model path with verification and fallback conversion
model_path, was_converted = verify_and_get_onnx_model(
    model_id="bert-base-uncased",
    onnx_path="model.onnx",
    conversion_config=conversion_config
)
```

### Benchmark Integration

The `benchmark_onnx_integration.py` script provides a complete example of integrating the ONNX verification system with the benchmark pipeline:

```bash
# Run benchmark on CPU for all configured models
python duckdb_api/core/benchmark_onnx_integration.py --hardware cpu

# Run benchmark for specific models
python duckdb_api/core/benchmark_onnx_integration.py --hardware cuda --models bert-base-uncased t5-small
```

### Command-Line ONNX File Checker

The `check_onnx_files.py` utility allows you to check ONNX file availability from the command line:

```bash
# Check a single model
python check_onnx_files.py --model bert-base-uncased --onnx-path model.onnx

# Check multiple models and convert if needed
python check_onnx_files.py --models bert-base-uncased t5-small --onnx-path model.onnx --convert

# Check models from a file
python check_onnx_files.py --model-file models.txt --convert --cache-dir ./onnx_cache

# Save results to a file
python check_onnx_files.py --models bert-base-uncased t5-small --output results.json --convert
```

### Database Schema Update

The `onnx_db_schema_update.py` script updates the DuckDB schema to track ONNX conversion:

```bash
# Update database schema
python onnx_db_schema_update.py --db-path ./benchmark_db.duckdb

# Update schema and migrate existing registry
python onnx_db_schema_update.py --db-path ./benchmark_db.duckdb --registry-path ./conversion_registry.json
```

### Testing the ONNX Verification System

The `test_onnx_verification.py` script provides comprehensive tests:

```bash
# Run all tests
python scripts/generators/models/test_onnx_verification.py

# Run a specific test
python scripts/generators/models/test_onnx_verification.py --test test_verify_onnx_file

# Run with verbose output
python scripts/generators/models/test_onnx_verification.py --verbose
```

## Supported Models

The system currently supports the following model types with optimized conversion configurations:

- **BERT** (and BERT variants)
- **T5** (and T5 variants)
- **GPT-2** (and GPT variants)
- **ViT** (Vision Transformers)
- **CLIP** (both text and vision components)
- **Whisper** (speech recognition)
- **Wav2Vec2** (speech processing)

Support for other model types can be added by extending the model detection and configuration systems.

## Configuration

### Cache Directory

By default, converted models are stored in `~/.ipfs_accelerate/model_cache`. You can customize this by providing a `cache_dir` parameter to the `OnnxVerifier` constructor:

```python
from onnx_verification import OnnxVerifier

verifier = OnnxVerifier(cache_dir="/path/to/custom/cache")
```

### Conversion Registry

The system maintains a conversion registry at `{cache_dir}/conversion_registry.json` that tracks all conversions with metadata including:

- Model ID
- ONNX path
- Local path to the converted model
- Conversion time
- Conversion configuration
- Source (always "pytorch_conversion" for converted models)

## Database Integration

### Schema Structure

The database schema includes the following components for ONNX tracking:

#### 1. New Columns in `performance_results` Table

- `onnx_source`: Source of the ONNX model ("pytorch_conversion" or "huggingface")
- `onnx_conversion_status`: Status of the verification ("converted" or "original")
- `onnx_conversion_time`: Timestamp of the conversion (NULL for original models)
- `onnx_local_path`: Path to the local ONNX file (for converted models)

#### 2. New `onnx_conversions` Table

```sql
CREATE TABLE onnx_conversions (
    id INTEGER PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    model_type VARCHAR,
    onnx_path VARCHAR NOT NULL,
    local_path VARCHAR NOT NULL,
    conversion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversion_config JSON,
    conversion_success BOOLEAN DEFAULT TRUE,
    error_message VARCHAR,
    opset_version INTEGER,
    file_size_bytes INTEGER,
    verification_status VARCHAR,
    last_used TIMESTAMP,
    use_count INTEGER DEFAULT 1
)
```

#### 3. New `model_onnx_registry` View

```sql
CREATE VIEW model_onnx_registry AS
SELECT 
    m.model_id,
    m.model_name,
    m.model_type,
    CASE 
        WHEN oc.model_id IS NOT NULL THEN 'available_converted'
        ELSE 'unknown'
    END as onnx_status,
    oc.local_path as onnx_local_path,
    oc.conversion_time,
    oc.opset_version,
    oc.file_size_bytes,
    oc.use_count
FROM models m
LEFT JOIN onnx_conversions oc ON m.model_id = oc.model_id
```

#### 4. SQL Functions for ONNX Status Checking

```sql
CREATE FUNCTION has_converted_onnx(model_id VARCHAR)
RETURNS BOOLEAN AS
$$
    SELECT EXISTS (
        SELECT 1 FROM onnx_conversions 
        WHERE model_id = model_id AND conversion_success = TRUE
    )
$$;

CREATE FUNCTION get_onnx_status(model_id VARCHAR)
RETURNS VARCHAR AS
$$
    SELECT CASE
        WHEN EXISTS (SELECT 1 FROM onnx_conversions WHERE model_id = model_id AND conversion_success = TRUE)
        THEN 'converted'
        WHEN EXISTS (SELECT 1 FROM onnx_conversions WHERE model_id = model_id AND conversion_success = FALSE)
        THEN 'conversion_failed'
        ELSE 'unknown'
    END
$$;
```

### Integration with Benchmark System

When using the system with benchmarks, it automatically:

1. Checks if the database has the ONNX tracking schema fields
2. Adds the following fields to benchmark results:
   - `onnx_source`: Source of the ONNX model ("pytorch_conversion" or "huggingface")
   - `onnx_conversion_status`: Status of the verification ("converted" or "original")
   - `onnx_conversion_time`: Timestamp of the conversion (NULL for original models)
   - `onnx_local_path`: Path to the local ONNX file (for converted models)
3. Stores the results in the database with proper ONNX tracking

Example database integration:

```python
# Benchmark with database integration
model_path, was_converted = verify_and_get_onnx_model(
    model_id="bert-base-uncased",
    onnx_path="model.onnx"
)

# Add ONNX conversion information to benchmark results
benchmark_result.update({
    "was_converted": was_converted,
    "onnx_source": "pytorch_conversion" if was_converted else "huggingface",
    "onnx_conversion_status": "converted" if was_converted else "original",
    "onnx_local_path": model_path if was_converted else None
})

# Store result in database
store_benchmark_in_database(benchmark_result)
```

## Error Handling

The system provides two custom exception classes:

- **`OnnxVerificationError`**: Raised when ONNX file verification fails
- **`OnnxConversionError`**: Raised when PyTorch to ONNX conversion fails

These exceptions include detailed error messages for debugging and logging.

## Best Practices

1. **Verify ONNX files before benchmarking**: Use this system to verify ONNX files before running benchmarks
2. **Convert when needed**: Use the automatic conversion feature when ONNX files aren't available
3. **Track conversion status**: Include conversion status in benchmark results for proper analysis
4. **Use model-specific configurations**: Different models may require specific configurations for optimal conversion
5. **Cache converted models**: Use the local cache to avoid redundant conversions
6. **Store in database**: Use the database schema update for comprehensive conversion tracking
7. **Run periodic checks**: Use the command-line utility to check ONNX availability across your model set
8. **Handle errors gracefully**: Implement appropriate error handling for cases where verification or conversion fails

## Requirements

- **Core Requirements**:
  - PyTorch (required for conversion)
  - transformers (required for loading PyTorch models)
  - onnx (required for model verification)
  - requests (required for checking HuggingFace)

- **Database Integration**:
  - duckdb (required for database schema update)

- **Testing**:
  - unittest (built-in)
  - tempfile (built-in)
  - shutil (built-in)

## Troubleshooting

### ONNX File Not Found

If an ONNX file is not found on HuggingFace, check if the model actually provides ONNX exports, or use the conversion feature:

```bash
python check_onnx_files.py --model your-model-id --onnx-path model.onnx --convert
```

### Conversion Fails

If conversion fails, try specifying the model type explicitly:

```bash
python check_onnx_files.py --model your-model-id --onnx-path model.onnx --convert --model-type bert
```

### Database Schema Issues

If you encounter database schema issues, run the schema update script:

```bash
python onnx_db_schema_update.py --db-path ./your_benchmark_db.duckdb
```

### Common Errors

- **Import Errors**: Make sure PyTorch, transformers, and onnx are installed
- **Verification Errors**: Check if the ONNX file path is correct
- **Conversion Errors**: Ensure the model is compatible with ONNX conversion
- **Database Errors**: Run the schema update script to ensure proper schema

## Future Extensions

- **Model-Specific Conversion Tuning**: Fine-tune conversion parameters for more model types
- **Improved Caching Strategy**: Implement more sophisticated caching with LRU eviction
- **Distributed Conversion**: Support distributed conversion for large models
- **Advanced Optimizations**: Apply post-conversion optimizations to ONNX models
- **CI/CD Integration**: Add automatic ONNX verification in CI/CD pipelines