# Merged Hugging Face Test Generator

This tool provides a comprehensive framework for generating test files that cover
all Hugging Face model architectures, with enhanced functionality for managing test files
and exporting model registry data.

## Key Features

- Support for multiple hardware backends (CPU, CUDA, OpenVINO, MPS, ROCm, Qualcomm)
- Testing for both from_pretrained() and pipeline() API approaches
- Consistent performance benchmarking and result collection
- Automatic model discovery and test generation
- Batch processing of multiple model families
- Parallel test generation for efficiency
- Export of model registry to parquet format
- Enhanced test templates with standardized structure

## Installation

This tool requires the following dependencies:

```bash
pip install transformers pandas pyarrow datasets
# Optional for duckdb export
pip install duckdb
```

## Basic Usage

The merged test generator can be used with various command-line arguments:

```bash
# Show help
python merged_test_generator.py --help

# List all model families in registry
python merged_test_generator.py --list-families

# Generate a test file for a specific model family
python merged_test_generator.py --generate bert

# Generate tests for all model families
python merged_test_generator.py --all

# Generate tests for a specific set of models
python merged_test_generator.py --batch-generate bert,gpt2,t5,vit,clip
```

## Advanced Features

### 1. Generate Missing Tests

Automatically identify and generate test files for models missing test implementations:

```bash
# List all missing test implementations without generating files
python merged_test_generator.py --generate-missing --list-only

# Generate up to 10 test files for missing models
python merged_test_generator.py --generate-missing --limit 10

# Generate tests only for high priority models
python merged_test_generator.py --generate-missing --high-priority-only

# Generate tests for a specific category of models
python merged_test_generator.py --generate-missing --category vision
```

### 2. Export Model Registry

Export the model registry data to parquet format for analysis or integration with other tools:

```bash
# Export using HuggingFace Datasets (default)
python merged_test_generator.py --export-registry

# Export using DuckDB
python merged_test_generator.py --export-registry --use-duckdb
```

### 3. Other Features

```bash
# Suggest new models to add to the registry
python merged_test_generator.py --suggest-models

# Generate a registry entry for a specific model
python merged_test_generator.py --generate-registry-entry sam

# Auto-add new models (limited to 5 by default)
python merged_test_generator.py --auto-add

# Update test_all_models.py with all model families
python merged_test_generator.py --update-all-models
```

## Hardware Platforms Support

The generated test files include support for multiple hardware platforms:

- CPU: Always tested
- CUDA: Tested if available
- OpenVINO: Tested if installed
- MPS (Apple Silicon): Reserved for future testing
- ROCm (AMD): Reserved for future testing
- Qualcomm AI: Reserved for future testing

## Test File Structure

The generated test files use a standardized structure:

1. **Environment Detection**: Automatic detection of available hardware and libraries
2. **Platform-Specific Testing**: Separate tests for each hardware platform
3. **Input Specialization**: Task-specific test inputs based on model type
4. **Batch Processing**: Support for both single and batch inputs
5. **Result Collection**: Detailed benchmarking and performance data collection
6. **Standardized Reporting**: Consistent reporting format for comparison

## Working with the Parquet Output

The exported model registry parquet file can be used with various tools:

```python
# Python/Pandas
import pandas as pd
df = pd.read_parquet("model_registry.parquet")
print(df.head())

# DuckDB
import duckdb
conn = duckdb.connect()
conn.execute("SELECT * FROM 'model_registry.parquet' WHERE category = 'vision'")
results = conn.fetchall()

# PySpark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("model_registry.parquet")
df.createOrReplaceTempView("models")
spark.sql("SELECT * FROM models WHERE num_models > 1").show()
```

## Test Template Customization

The generated test templates can be customized by modifying the `generate_test_template` function in the tool. The template includes:

- Mock implementations for missing dependencies
- Hardware capability detection
- Standardized test methods for consistent results
- Detailed reporting and logging

## Category Detection

Models are automatically categorized based on their pipeline tasks:

- **Language**: text-generation, fill-mask, etc.
- **Vision**: image-classification, object-detection, etc.
- **Audio**: automatic-speech-recognition, audio-classification, etc.
- **Multimodal**: image-to-text, visual-question-answering, etc.
- **Specialized**: protein-folding, time-series-prediction, etc.

## Extending the Tool

To add support for new model types or tasks:

1. Update the `SPECIALIZED_MODELS` dictionary to include task mappings
2. Add task-specific inputs to the `get_specialized_test_inputs` function
3. Include any new categories in the `get_pipeline_category` function
4. Add appropriate model names to the `get_appropriate_model_name` function

## Best Practices

1. Generate test files in batches to avoid overwhelming the system
2. Use the `--category` flag to focus on specific model types
3. Start with high-priority models using the `--high-priority-only` flag
4. Export the model registry regularly to track test coverage
5. Add test directories to version control for tracking changes