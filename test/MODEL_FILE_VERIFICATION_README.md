# Model File Verification and Conversion Pipeline

This document describes the Model File Verification and Conversion Pipeline, designed to ensure that necessary model files are available before benchmarking and to handle conversion between different formats as needed.

**Date: March 9, 2025**

## Overview

The Model File Verification and Conversion Pipeline is designed to solve several common problems encountered during benchmarking:

1. **Missing ONNX Files**: Many models on Hugging Face don't have pre-built ONNX files.
2. **Connectivity Issues**: Intermittent network problems during model downloads.
3. **Format Consistency**: Different models use different formats (ONNX, PyTorch, etc.).
4. **Conversion Complexity**: Converting between formats requires model-specific knowledge.
5. **Caching Management**: Efficiently managing locally cached model files.

## Key Features

The pipeline provides a comprehensive solution with these features:

- **Pre-benchmark Model File Verification**: Verify model files exist before starting benchmarks.
- **PyTorch to ONNX Conversion Pipeline**: Automatically convert from PyTorch when ONNX files are missing.
- **Automated Retry Logic**: Handle connectivity issues with configurable retry logic.
- **Local Disk Caching**: Cache converted models for future benchmark runs.
- **Model-Specific Conversion Parameters**: Apply optimal conversion settings for each model type.
- **Comprehensive Error Handling**: Provide detailed error messages for missing files.
- **Database Integration**: Track model sources and conversion status in the benchmark database.
- **Batch Processing**: Process multiple models efficiently with batch verification.
- **Memory Management**: Automatically clean up old cached files when space is limited.

## Components

The pipeline consists of these core components:

1. **`model_file_verification.py`**: Main module implementing the verification and conversion functionality.
2. **`benchmark_model_verification.py`**: Integration with the benchmark system.
3. **`run_model_verification.sh`**: Example script showing how to use the system.

## Usage

### Basic Verification

```bash
python model_file_verification.py --model bert-base-uncased --file-path model.onnx
```

### Verification with Model Type Specification

```bash
python model_file_verification.py --model bert-base-uncased --file-path model.onnx --model-type bert
```

### Check if a Model File Exists

```bash
python model_file_verification.py --model bert-base-uncased --file-path model.onnx --check-exists
```

### Get Model Metadata

```bash
python model_file_verification.py --model bert-base-uncased --get-metadata
```

### Batch Verification

```bash
python model_file_verification.py --batch --batch-file batch_models.json --output batch_results.json
```

Where `batch_models.json` contains:

```json
[
    {
        "model_id": "bert-base-uncased",
        "file_path": "model.onnx",
        "model_type": "bert"
    },
    {
        "model_id": "t5-small",
        "file_path": "model.onnx",
        "model_type": "t5"
    }
]
```

### Benchmark Integration

```bash
python benchmark_model_verification.py --model bert-base-uncased --file-path model.onnx
```

### Multiple Models Benchmark

```bash
python benchmark_model_verification.py --models bert-base-uncased t5-small --file-path model.onnx
```

### From a Model List File

```bash
python benchmark_model_verification.py --model-file models.txt
```

Where `models.txt` contains one model ID per line:

```
bert-base-uncased
t5-small
google/vit-base-patch16-224
```

### Run the Full Example Script

```bash
./run_model_verification.sh
```

## Automatic Model Type Detection

The system can automatically detect the model type from the model ID using a simple heuristic:

- Models with "bert" in the name are detected as "bert" type.
- Models with "t5" in the name are detected as "t5" type.
- Models with "gpt" in the name are detected as "gpt" type.
- Models with "vit" or "vision" in the name are detected as "vit" type.
- Models with "clip" in the name are detected as "clip" type.
- Models with "whisper" in the name are detected as "whisper" type.
- Models with "wav2vec" in the name are detected as "wav2vec2" type.

You can override this with the `--model-type` parameter when needed.

## Advanced Configuration

### Cache Directory

Set a custom cache directory:

```bash
python model_file_verification.py --model bert-base-uncased --file-path model.onnx --cache-dir /path/to/cache
```

### HuggingFace Token

Provide a HuggingFace API token for private models:

```bash
python model_file_verification.py --model private-repo/model --file-path model.onnx --token YOUR_TOKEN
```

### Cache Cleanup Thresholds

The system automatically cleans up old cached files when the total size exceeds a threshold. These thresholds are configurable through environment variables:

```bash
# Set cache cleanup threshold to 50GB
export CACHE_CLEANUP_THRESHOLD=50
# Set minimum age for files to be considered for cleanup to 14 days
export CACHE_CLEANUP_MIN_AGE_DAYS=14

python model_file_verification.py --model bert-base-uncased --file-path model.onnx
```

## API Usage

The system can also be used programmatically:

```python
from model_file_verification import ModelFileVerifier

# Create a verifier
verifier = ModelFileVerifier(
    cache_dir="/path/to/cache",
    huggingface_token="YOUR_TOKEN"
)

# Verify a model file
model_path, was_converted = verifier.verify_model_for_benchmark(
    model_id="bert-base-uncased",
    file_path="model.onnx",
    model_type="bert"
)

# Check if a model file exists
exists = verifier.verify_model_exists("bert-base-uncased", "model.onnx")

# Get model metadata
metadata = verifier.get_model_metadata("bert-base-uncased")

# Batch verify multiple models
results = verifier.batch_verify_models([
    {"model_id": "bert-base-uncased", "file_path": "model.onnx", "model_type": "bert"},
    {"model_id": "t5-small", "file_path": "model.onnx", "model_type": "t5"}
])
```

## Integration with Benchmark System

The Model File Verification and Conversion Pipeline integrates seamlessly with the benchmark system:

1. Before running a benchmark, the system verifies that the required model file exists.
2. If the model file is not found, the system attempts to convert it from another format.
3. The system tracks whether the model was converted or used as-is.
4. All benchmark results include the source of the model file for proper analysis.

The `benchmark_model_verification.py` script demonstrates how to integrate the system with the benchmark workflow.

## Error Handling

The system provides several custom exception types for detailed error handling:

- `ModelVerificationError`: Base exception for model verification errors.
- `ModelConversionError`: Error during model conversion.
- `ModelFileNotFoundError`: Model file not found and could not be converted.
- `ModelConnectionError`: Error connecting to model repositories.

These exceptions provide detailed error messages to help diagnose and fix issues.

## Database Integration

When using the benchmark integration, all results are automatically stored in the benchmark database (if available):

- The source of the model (original or converted) is tracked.
- The conversion status is recorded for analysis.
- The local path to the model file is stored for reference.

This enables comprehensive analysis of benchmark results, including any impact of model conversion on performance.

## Requirements

- Python 3.8 or higher
- `huggingface_hub` package for HuggingFace integration
- `torch` and `transformers` packages for PyTorch model conversion
- `onnx` package for ONNX model verification
- `duckdb` package for database integration (optional)

## Future Improvements

Planned enhancements for the Model File Verification and Conversion Pipeline:

1. **Advanced Model Type Detection**: Use the HuggingFace API to get precise model architecture information.
2. **Multi-Format Support**: Add support for more model formats (TensorFlow, JAX, etc.).
3. **Parallel Processing**: Add parallel verification for large batch operations.
4. **Auto-Optimization**: Automatically optimize conversion parameters based on model characteristics.
5. **Version Tracking**: Track model and conversion tool versions for reproducibility.
6. **Integrity Checks**: Add checks to verify model file integrity after download.
7. **Distributed Cache**: Support for distributed caching across multiple machines.

## Conclusion

The Model File Verification and Conversion Pipeline provides a robust solution for ensuring model availability before benchmarking, with automatic conversion, caching, and comprehensive error handling. By integrating this system with the benchmark workflow, you can ensure consistent and reliable benchmark results.

For more details, see the source code and example scripts in the repository.

For detailed API documentation, refer to the docstrings in the source code.

## Author

IPFS Accelerate Team - March 2025