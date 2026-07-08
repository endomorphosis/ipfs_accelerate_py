# Template System Enhancements

This document outlines the enhancements to the IPFS Accelerate Python Framework's template system, building on the previous database-driven template framework.

## Overview

We've enhanced the existing template database system with:

1. **Comprehensive Template Validation**: Validation for syntax, hardware platform support, and placeholder completeness
2. **Template Inheritance System**: Modality-based template hierarchy with parent-child relationships 
3. **Improved Placeholder Handling**: Enhanced placeholder management, automatic detection, and documentation

## Getting Started

### Prerequisites

- **DuckDB**: The template database system requires DuckDB
- **Python 3.9+**: The system uses f-strings and other modern Python features
- **Existing Template Database**: Run `create_template_database.py --create` first if you don't have one

### Using the Enhanced Template System

```bash
# Check if database exists and has the right schema
python enhanced_templates/template_system_enhancement.py --check-db

# Validate all templates in the database
python enhanced_templates/template_system_enhancement.py --validate-templates

# Validate templates for a specific model type
python enhanced_templates/template_system_enhancement.py --validate-model-type bert

# List all templates with their validation status
python enhanced_templates/template_system_enhancement.py --list-templates

# Add template inheritance system
python enhanced_templates/template_system_enhancement.py --add-inheritance

# Enhance placeholder handling
python enhanced_templates/template_system_enhancement.py --enhance-placeholders

# Apply all enhancements at once
python enhanced_templates/template_system_enhancement.py --apply-all-enhancements
```

For convenience, you can run all enhancements with the provided shell script:

```bash
./enhanced_templates/run_template_enhancements.sh
```

### Using the Template Generator

The example template generator demonstrates how to use the enhanced template system:

```bash
# Generate a test template for a specific model
python enhanced_templates/example_template_generator.py --model bert-base-uncased

# Generate a benchmark template for a specific model
python enhanced_templates/example_template_generator.py --model bert-base-uncased --template-type benchmark

# Generate a template for a specific hardware platform
python enhanced_templates/example_template_generator.py --model bert-base-uncased --hardware cuda

# Save the generated template to a file
python enhanced_templates/example_template_generator.py --model bert-base-uncased --output test_bert.py

# Detect available hardware on the system
python enhanced_templates/example_template_generator.py --detect-hardware
```

## Enhanced Features

### 1. Comprehensive Template Validation

The enhanced system validates templates for:

#### Syntax Validation
- Checks for valid Python syntax in templates
- Validates balanced braces and placeholder formats
- Identifies common syntax issues
- Warns about potential issues like double braces or escape sequences in triple-quoted strings
- Validates Python code by attempting compilation

#### Hardware Platform Support Validation
- Detects hardware-specific code in templates
- Validates support for CPU, CUDA, ROCm, MPS, OpenVINO, Qualcomm, Samsung, WebNN, WebGPU
- Reports hardware compatibility matrix for each template
- Analyzes hardware-specific imports, device settings, and platform-specific function calls
- Allows validating for a specific hardware platform

#### Placeholder Validation
- Identifies missing mandatory placeholders
- Reports all placeholders used in templates
- Suggests standard placeholder replacements
- Validates that required placeholders like `model_name`, `normalized_name`, and `generated_at` are present
- Provides information on optional and auto-detected placeholders

Example validation output:
```
Templates with Validation Status:
----------------------------------------------------------------------------------------------------
Model Type      Template Type   Hardware   Status     Modality     Latest Validation           Hardware Support
----------------------------------------------------------------------------------------------------
bert           test           generic    VALID      text         2025-03-10 12:34:56.789 ✅ PASS    cpu, cuda, mps
bert           test           cuda       VALID      text         2025-03-10 12:34:56.789 ✅ PASS    cpu, cuda
t5             test           generic    INVALID    text         2025-03-10 12:34:56.789 ❌ FAIL    cpu
...
```

### 2. Template Inheritance System

The enhanced template system introduces a modality-based inheritance hierarchy:

```
default_text          default_vision          default_audio          default_multimodal
    │                      │                       │                        │
    ├── bert               ├── vit                 ├── whisper              ├── clip
    ├── t5                 ├── resnet              ├── wav2vec2             ├── llava
    ├── llama              └── detr                └── clap                 └── xclip
    └── gpt2
```

Benefits of template inheritance:
- **Reduced Duplication**: Common functionality defined once in parent templates
- **Consistent Implementation**: Standardized patterns across similar models
- **Easier Maintenance**: Update parent templates to affect all child templates
- **Specialization Support**: Override specific parts in child templates

The inheritance system automatically adds appropriate parent-child relationships based on model type and modality. It creates default parent templates for each modality if they don't exist.

#### Default Parent Templates

The system provides robust default parent templates for each modality:
- **Text Models**: Specialized for text input/output with tokenizer integration
- **Vision Models**: Image processing capabilities with automatic test image creation
- **Audio Models**: Audio handling with sampling rate management and silence generation
- **Multimodal Models**: Combined text and image processing with synchronized inputs

#### Specialized Model Functionality

Child templates inherit from these parent templates but can override specific functionality:
- **BERT/T5/LLAMA**: Text-specific testing with specialized verification
- **ViT/ResNet/DETR**: Vision-specific processing and feature extraction
- **Whisper/Wav2Vec2/CLAP**: Audio processing with sampling rate handling
- **CLIP/LLaVA/XCLIP**: Multimodal processing with combined inputs

### 3. Improved Placeholder Handling

The enhanced placeholder system includes:

#### Standard Placeholder Definitions
- Core placeholders (`model_name`, `normalized_name`, `generated_at`)
- Hardware-related placeholders (`best_hardware`, `torch_device`, `has_cuda`, etc.)
- Model-related placeholders (`model_family`, `model_subfamily`)
- Each placeholder has documentation, default value (if applicable), and required status

#### Automatic Placeholder Detection
- Scans all templates to detect used placeholders
- Registers discovered placeholders in database
- Provides documentation for all placeholders
- Distinguishes between standard and custom placeholders

#### Placeholder Helper Utilities
- `placeholder_helpers.py` module provides helper functions
- `detect_missing_placeholders()` for finding undefined placeholders
- `get_default_context()` for standard context creation
- `render_template()` for safe template rendering with fallbacks
- Auto-fills missing placeholders with defaults when possible

Example of enhanced rendering:
```python
from template_utilities import get_default_context, render_template

# Get default context with hardware detection
context = get_default_context(model_name="bert-base-uncased")

# Render template with placeholder substitution
rendered = render_template(template, context)
```

## Database Schema Enhancements

The enhanced system extends the database schema with:

```sql
-- Main templates table extensions
ALTER TABLE templates ADD COLUMN validation_status VARCHAR;
ALTER TABLE templates ADD COLUMN parent_template VARCHAR;
ALTER TABLE templates ADD COLUMN modality VARCHAR;
ALTER TABLE templates ADD COLUMN last_updated TIMESTAMP;

-- New template_validation table
CREATE TABLE IF NOT EXISTS template_validation (
    id INTEGER PRIMARY KEY,
    template_id INTEGER,
    validation_date TIMESTAMP,
    validation_type VARCHAR,
    success BOOLEAN,
    errors TEXT,
    hardware_support TEXT
);

-- New template_placeholders table
CREATE TABLE IF NOT EXISTS template_placeholders (
    id INTEGER PRIMARY KEY,
    placeholder VARCHAR,
    description TEXT,
    default_value VARCHAR,
    required BOOLEAN
);
```

### Schema Design Benefits

- **Validation Status Tracking**: Records validation results for each template
- **Template Inheritance**: Tracks parent-child relationships between templates
- **Modality Classification**: Categorizes templates by modality for better organization
- **Change Tracking**: Records when templates were last updated
- **Detailed Validation History**: Stores comprehensive validation results
- **Placeholder Management**: Centralizes placeholder definitions and documentation

## Modality-Based Templates

The system now organizes templates by modality, providing specialized templates for:

### Text Models
- BERT, T5, LLAMA, GPT2
- Specialized for text input/output
- Tokenizer integration
- Text preprocessing and embedding extraction
- Sequence length handling

### Vision Models
- ViT, ResNet, DETR
- Image processing and feature extraction
- Vision-specific metrics
- Automatic test image generation
- Image transformation and normalization

### Audio Models
- Whisper, Wav2Vec2, CLAP
- Audio preprocessing
- Sampling rate handling
- Silence generation for testing
- Audio length normalization

### Multimodal Models
- CLIP, LLaVA, XCLIP
- Combined text and image processing
- Multiple input formats
- Synchronization between different input types
- Joint representation handling

Each modality gets appropriate default templates and specialized functionality.

## Hardware-Specific Template Support

The enhanced system improves hardware platform support with:

### CUDA Optimization
- Automatic device detection and tensor movement
- Mixed precision support for compatible models
- Memory optimizations for large models
- Batch processing optimizations

### MPS (Apple Silicon) Support
- Metal Performance Shaders integration
- Apple Silicon-specific optimizations
- Fallback mechanisms for unsupported operations

### OpenVINO Integration
- Intel-specific optimizations
- Model conversion utilities
- CPU and Neural Compute Stick support

### ROCm Support
- AMD GPU accelerations
- HIP-based optimizations
- ROCm-specific memory management

### WebNN/WebGPU Support
- Browser-based acceleration
- WebNN API integration
- WebGPU with transformers.js
- Cross-browser compatibility

### Qualcomm and Mobile Support
- Mobile-optimized code paths
- Qualcomm AI Engine integration
- Power-efficient operation modes
- Battery impact considerations

## Web Platform Support

The enhanced templates include improved web platform (WebNN/WebGPU) support:

- **WebNN Integration**: Templates for WebNN acceleration
- **WebGPU/transformers.js**: Templates for WebGPU with transformers.js
- **Browser Compatibility**: Firefox optimization for audio models, Edge for WebNN
- **Fallback Mechanisms**: Graceful degradation to simulation mode
- **Model Serialization**: Proper tensor conversion between Python and JavaScript
- **Error Handling**: Robust error management with graceful fallbacks
- **Performance Monitoring**: Browser-specific performance metrics collection

## Implementation Details

### Validation System
The validation system performs multiple levels of checks:
1. **Syntax Validation**: Uses Python's `compile()` function to verify code correctness
2. **Hardware Detection**: Pattern matching to identify hardware-specific code
3. **Placeholder Extraction**: Regular expression-based placeholder detection
4. **Schema Validation**: Database schema integrity checking

### Inheritance System
The inheritance system implements:
1. **Parent Template Creation**: Automatic generation of modality-specific parent templates
2. **Relationship Management**: Database tracking of parent-child template relationships
3. **Modality Detection**: Automatic assignment of templates to modality categories
4. **Fallback Logic**: Multi-level template selection with inheritance-based fallbacks

### Placeholder System
The placeholder system provides:
1. **Documentation Generation**: Automatic documentation for placeholders
2. **Default Value Management**: Smart defaults for common placeholders
3. **Context Generation**: Automatic context creation with hardware detection
4. **Rendering Utilities**: Safe template rendering with error handling

## Testing and Verification

The enhanced system includes comprehensive unit tests:
- **Syntax Validation Tests**: Verify that template syntax checking works correctly
- **Hardware Support Tests**: Confirm that hardware detection functions properly
- **Placeholder Tests**: Validate placeholder extraction and management
- **Inheritance Tests**: Verify parent-child relationships work as expected
- **Template Generation Tests**: End-to-end tests of the template generation process

## Future Enhancements

- **Template Versioning System**: Track and manage template versions over time
- **Web-Based Template Editor**: Create a user interface for template management
- **Template Analytics**: Monitor template usage and performance
- **Template Generation from Source Code**: Automatic template creation from example code
- **AI-Assisted Template Creation**: Use AI to suggest template improvements

## Conclusion

The enhanced template system improves maintainability, reduces duplication, and provides better validation for the IPFS Accelerate Python Framework. It makes generating tests, benchmarks, and skills more reliable and consistent across model families and hardware platforms.

With modality-based inheritance, comprehensive validation, and improved placeholder handling, the template system can now efficiently support hundreds of model types across diverse hardware platforms while maintaining code quality and consistency.