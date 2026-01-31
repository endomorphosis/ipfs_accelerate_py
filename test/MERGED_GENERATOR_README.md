# Merged Hugging Face Test Generator (March 2025 Update)

> **NEW! March 6, 2025:** Critical fixes for model registry, test methods, OpenVINO integration, and modality-specific handling. For detailed information, see [PHASE16_GENERATOR_FIXES.md](PHASE16_GENERATOR_FIXES.md).

> **COMPLETE:** Enhanced with specialized hardware support for 13 key model classes and full web platform integration. See [KEY_MODELS_README.md](KEY_MODELS_README.md) for details.

This tool provides a comprehensive framework for generating test files that cover
all Hugging Face model architectures, with enhanced functionality for managing test files
and exporting model registry data.

## Key Features

- **Model Registry System**: Comprehensive mapping of short model names to full model IDs for consistent identification
- **Modality-Specific Templates**: Specialized templates for text, vision, audio, and multimodal models with appropriate initialization
- **Automatic Modality Detection**: Smart detection of model types based on name patterns
- **Support for Multiple Hardware Backends**: CPU, CUDA, OpenVINO, Apple Silicon (MPS), AMD ROCm, Qualcomm AI, WebNN, and WebGPU
- **Robust Test Interface**: Standardized test class structure with proper run_tests() method and hardware detection
- **Specialized Model Initialization**: Modality-specific model and processor initialization for different model types
- **Flexible Output Validation**: Robust output structure validation that handles different model output formats 
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
python scripts/generators/test_scripts/generators/merged_test_generator.py --help

# List all model families in registry
python scripts/generators/test_scripts/generators/merged_test_generator.py --list-families

# Generate a test file for a specific model family
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate bert

# Generate a test with specific hardware platforms
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate vit --platform cuda,openvino,webgpu

# Generate a test with cross-platform support for all hardware
python fixed_scripts/generators/test_scripts/generators/merged_test_generator.py --generate clip --cross-platform

# Generate tests for all model families
python scripts/generators/test_scripts/generators/merged_test_generator.py --all

# Generate tests for a specific set of models
python scripts/generators/test_scripts/generators/merged_test_generator.py --batch-generate bert,gpt2,t5,vit,clip
```

## Advanced Features

### 1. Generate Missing Tests

Automatically identify and generate test files for models missing test implementations:

```bash
# List all missing test implementations without generating files
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --list-only

# Generate up to 10 test files for missing models
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --limit 10

# Generate tests only for high priority models
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --high-priority-only

# Generate tests for a specific category of models
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --category vision

# Generate tests for key models with enhanced hardware support
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --key-models-only

# Prioritize key models (t5, clap, whisper, llava, etc.) with hardware optimizations
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --prioritize-key-models
```

### 2. Export Model Registry

Export the model registry data to parquet format for analysis or integration with other tools:

```bash
# Export using HuggingFace Datasets (default)
python scripts/generators/test_scripts/generators/merged_test_generator.py --export-registry

# Export using DuckDB
python scripts/generators/test_scripts/generators/merged_test_generator.py --export-registry --use-duckdb
```

### 3. Other Features

```bash
# Suggest new models to add to the registry
python scripts/generators/test_scripts/generators/merged_test_generator.py --suggest-models

# Generate a registry entry for a specific model
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-registry-entry sam

# Auto-add new models (limited to 5 by default)
python scripts/generators/test_scripts/generators/merged_test_generator.py --auto-add

# Update test_all_models.py with all model families
python scripts/generators/test_scripts/generators/merged_test_generator.py --update-all-models
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

## Enhanced Hardware Support for Key Models

The generator now provides specialized hardware implementations for 13 key model types:

### Key Supported Models

1. **T5** - Enhanced OpenVINO support, complete web platform (WebNN/WebGPU) integration
2. **CLAP** - Enhanced OpenVINO support
3. **Wav2Vec2** - Enhanced OpenVINO support
4. **Whisper** - Full web platform (WebNN/WebGPU) support
5. **LLaVA** - Enhanced OpenVINO, MPS, and ROCm support
6. **LLaVA-Next** - Added OpenVINO, MPS, and ROCm support
7. **Qwen2/3** - Better implementations for OpenVINO, MPS, ROCm platforms
8. **XCLIP** - Added web platform support
9. **DETR** - Added web platform support
10. **BERT** - Already has complete hardware support
11. **ViT** - Already has complete hardware support
12. **CLIP** - Already has complete hardware support
13. **LLAMA** - Already has optimal hardware support for supported platforms

### Enhanced Implementation Details

These enhanced models include:

- **OpenVINO Optimizations**: Real model conversion and specialized preprocessing for key models
- **ROCm (AMD) Support**: Enhanced implementations with AMD-specific optimizations
- **MPS (Apple) Support**: Optimized implementations for Apple Silicon
- **WebNN/WebGPU Support**: Browser-specific optimizations for web deployment
- **Multimodal Handling**: Specialized preprocessing for complex multimodal models

To generate tests with these enhancements:
```bash
# Generate tests for all key models
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --key-models-only

# Prioritize key models but include others
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --prioritize-key-models

# Focus on a specific modality of key models
python scripts/generators/test_scripts/generators/merged_test_generator.py --generate-missing --key-models-only --category multimodal
```

## Test File Structure

The enhanced test files now use a modality-specific architecture with standardized components:

### Modality-Specific Components

1. **Text Models**:
   - Specialized tokenizer initialization and text processing
   - Support for variable-length inputs and batch processing
   - Token-based performance metrics (tokens/second)
   - AutoTokenizer and AutoModel initialization pattern
   - Appropriate output validation for text model outputs

2. **Vision Models**:
   - Automatic image creation and preprocessing with PIL
   - AutoImageProcessor and AutoModelForImageClassification initialization
   - Image format conversions and resolution handling
   - Frame-based performance metrics (FPS)
   - Image-specific output validation for vision models

3. **Audio Models**:
   - Audio file handling and format conversion
   - AutoFeatureExtractor and AutoModelForAudioClassification initialization
   - Sampling rate management and audio preprocessing
   - Real-time factor performance metrics
   - Audio-specific output validation logic

4. **Multimodal Models**:
   - AutoProcessor and combined model initialization
   - Combined text and image processing with appropriate formats
   - Different input formats (image+text, video+text)
   - End-to-end performance metrics
   - Multimodal-specific output validation for embeddings

### Common Structure Elements (Updated March 2025)

1. **Model Registry Integration**: Full mapping of short model names to complete model IDs
2. **Run Tests Method**: Standard run_tests() method that properly invokes unittest.main()
3. **Environment Detection**: Comprehensive detection of available hardware and libraries
4. **Centralized Hardware Detection**: Integration with centralized hardware detection module
5. **Platform-Specific Testing**: Separate tests for each hardware platform (including web platforms)
6. **Input Specialization**: Modality-specific test data and input processing
7. **Hardware Optimization**: Platform-specific optimizations for each modality
8. **Web Platform Integration**: Browser-specific handlers for WebNN and WebGPU
9. **Enhanced OpenVINO Support**: Proper initialization with openvino_label parameter
10. **Qualcomm AI Engine Support**: Full integration with Qualcomm AI Engine
11. **Error Handling**: Robust error capture with mock implementations as fallbacks
12. **Output Validation**: Flexible validation that handles different output structures
13. **Result Collection**: Detailed benchmarking and performance data collection
14. **Standardized Reporting**: Consistent reporting format for comparison

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