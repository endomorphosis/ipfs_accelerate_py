# Merged Hugging Face Test Generator

This tool provides a comprehensive framework for generating test files that cover
all Hugging Face model architectures, with enhanced functionality for managing test files
and exporting model registry data.

## Key Features

- **Modality-Specific Templates**: Specialized templates for text, vision, audio, and multimodal models
- **Automatic Modality Detection**: Smart detection of model types based on name patterns
- **Support for Multiple Hardware Backends**: CPU, CUDA, OpenVINO, Apple Silicon (MPS), AMD ROCm, Qualcomm AI, WebNN, and WebGPU
- **Web Platform Support**: WebNN and WebGPU (transformers.js) for browser-based inference
- **Testing for Both API Approaches**: Support for from_pretrained() and pipeline() patterns
- **Consistent Performance Benchmarking**: Standardized performance metrics across all templates
- **Automatic Model Discovery**: Smart identification of missing test implementations
- **Batch Processing**: Generate tests for multiple model families at once
- **Parallel Test Generation**: Efficient test creation using concurrency
- **Export of Model Registry**: Structured export to parquet format
- **Enhanced Test Templates**: Standardized structure with modality-specific features

## Installation

This tool requires the following dependencies:

```bash
pip install transformers pandas pyarrow datasets
# Optional for duckdb export
pip install duckdb
# Optional for web export capabilities
pip install onnx onnxruntime
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

The generated test files now include comprehensive support for multiple hardware platforms with modality-specific optimizations:

- **CPU**: Always tested, with modality-appropriate processors and handlers
- **CUDA**: GPU acceleration with hardware-specific optimizations:
  - Half-precision for text and vision models
  - Optimized memory formats for vision models
  - Automatic batch size adjustments based on model type
- **OpenVINO**: Hardware-optimized inference with Intel acceleration
- **MPS (Apple Silicon)**: Specialized support for M1/M2/M3 chips
- **ROCm (AMD)**: Support for AMD GPU acceleration
- **Qualcomm AI**: Mobile-optimized inference support for Qualcomm chips
- **WebNN**: Browser-based hardware acceleration via Web Neural Network API:
  - ONNX conversion and export
  - Browser-optimized inference handlers
  - Fallback mechanisms for unsupported browsers
- **WebGPU (transformers.js)**: GPU-accelerated browser execution:
  - Integration with transformers.js library
  - WebGPU acceleration for compatible browsers
  - JavaScript code generation for web deployment

Each hardware platform includes:
- Automatic hardware detection
- Platform-specific optimizations
- Modality-appropriate processing
- Error handling and mock fallbacks

## Test File Structure

The enhanced test files now use a modality-specific architecture with standardized components:

### Modality-Specific Components

1. **Text Models**:
   - Specialized tokenizer initialization and text processing
   - Support for variable-length inputs and batch processing
   - Token-based performance metrics (tokens/second)

2. **Vision Models**:
   - Automatic image creation and preprocessing
   - Format conversions and resolution handling
   - Frame-based performance metrics (FPS)

3. **Audio Models**:
   - Audio file handling and format conversion
   - Sampling rate management and audio preprocessing
   - Real-time factor performance metrics

4. **Multimodal Models**:
   - Combined text and image processing
   - Different input formats (image+text, video+text)
   - End-to-end performance metrics

### Common Structure Elements

1. **Environment Detection**: Automatic detection of available hardware and libraries
2. **Platform-Specific Testing**: Separate tests for each hardware platform (including web platforms)
3. **Input Specialization**: Modality-specific test data and input processing
4. **Hardware Optimization**: Platform-specific optimizations for each modality
5. **Web Platform Integration**: Browser-specific handlers for WebNN and WebGPU
6. **Error Handling**: Robust error capture with mock implementations as fallbacks
7. **Result Collection**: Detailed benchmarking and performance data collection
8. **Standardized Reporting**: Consistent reporting format for comparison

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

## Modality and Category Detection

Models are automatically detected and categorized based on their type and pipeline tasks:

### Modality Detection

The system now automatically detects model modalities using the `detect_model_modality` function:

- **Text**: BERT, GPT2, T5, RoBERTa, LLaMA, Mistral, Phi, etc.
- **Vision**: ViT, DETR, Swin, ConvNeXT, ResNet, SAM, etc.
- **Audio**: Whisper, Wav2Vec2, HuBERT, CLAP, MusicGen, etc.
- **Multimodal**: CLIP, LLaVA, BLIP, Pix2Struct, etc.
- **Specialized**: Time series models, protein models, graph models, etc.

### Pipeline Task Categories

Models are also categorized based on their pipeline tasks:

- **Text**: text-generation, fill-mask, text-classification, etc.
- **Vision**: image-classification, object-detection, image-segmentation, etc.
- **Audio**: automatic-speech-recognition, audio-classification, etc.
- **Multimodal**: image-to-text, visual-question-answering, etc.
- **Specialized**: protein-folding, time-series-prediction, etc.

## Extending the Tool

To add support for new model types or modalities:

1. **Add New Model Types**:
   - Update the `MODALITY_TYPES` dictionary to include new model names
   - Add new patterns to the `detect_model_modality` function
   - Include new task mappings in the `SPECIALIZED_MODELS` dictionary

2. **Create New Modality Templates**:
   - Add a new section to the `generate_modality_specific_template` function
   - Create appropriate template code with proper input/output handling
   - Add hardware-specific optimizations for the new modality

3. **Update Pipeline Categories**:
   - Add new task types to the appropriate categories in `get_pipeline_category`
   - Update the task-specific inputs in the test generator

4. **Add Hardware Optimizations**:
   - Implement specialized code for each hardware platform
   - Create appropriate mock implementations for testing
   - Add benchmarking metrics specific to the new modality

5. **Extend Web Platform Support**:
   - Add specialized handlers for WebNN and WebGPU
   - Create browser-specific optimizations for each modality
   - Add JavaScript code templates for web deployment

## Best Practices

1. **Use Modality-Specific Generation**:
   - Generate tests by modality for best results
   - Use `generate_sample_tests.py --modality text` to target specific modalities
   - Verify the generated tests are appropriate for the model type

2. **Batch Processing**:
   - Generate test files in batches to avoid overwhelming the system
   - Use `--batch-generate` with comma-separated model names
   - Process one modality at a time for most consistent results

3. **Testing Strategy**:
   - Start with high-priority models using the `--high-priority-only` flag
   - Verify each modality with representative models before mass generation
   - Test each hardware platform separately if resources are limited

4. **Maintenance**:
   - Export the model registry regularly to track test coverage
   - Add test directories to version control for tracking changes
   - Update the modality detection when adding new model families
   
5. **Hardware-Specific Testing**:
   - Focus on CPU tests first to verify basic functionality
   - Add CUDA tests for performance-critical models
   - Use specialized hardware tests (MPS, ROCm, Qualcomm) only for relevant models
   - Test web platforms (WebNN, WebGPU) for models intended for browser deployment

6. **Web Platform Integration**:
   - Use the WebNN export for general browser compatibility
   - Use WebGPU/transformers.js for models requiring GPU acceleration in browsers
   - Consult the guides in `sample_tests/export/` for platform-specific optimizations
   - Test in multiple browsers to ensure cross-platform compatibility