# Refactored Generator Suite

This is a comprehensive refactored generator suite for HuggingFace model testing. 

## Overview

The HuggingFace Model Generator is a sophisticated system that automatically creates test files for HuggingFace model architectures. It enables 100% test coverage of the 300+ HuggingFace model classes through a component-based architecture with clear separation of concerns.

## Key Features

- **Component-Based Architecture**: Clear separation of concerns with modular components.
- **Template System**: Specialized templates for each model architecture (encoder-only, decoder-only, etc.).
- **Model Registry**: Centralized registry for model metadata and configuration.
- **Hardware Detection**: Intelligent detection of available hardware (CUDA, ROCm, MPS, etc.).
- **Coverage Reporting**: Track and report on test coverage across all model architectures.
- **Batch Generation**: Generate tests in batches for efficient processing.
- **Mock Support**: Automatic support for running tests in mock mode for CI/CD.
- **Validation & Auto-fixing**: Automatic validation and fixing of common syntax issues.
- **HuggingFace API Integration**: Generate tests by searching the HuggingFace model hub.
- **Coverage Matrix Export**: Export coverage reports in multiple formats (Markdown, HTML, CSV, JSON).

## Architecture

The generator suite is built on a modular, component-based architecture:

```
refactored_generator_suite/
├── generator_core/       - Core generator components
├── templates/            - Template implementations for each architecture
├── model_selection/      - Model selection components
├── hardware/             - Hardware detection and recommendation
├── dependencies/         - Dependency management
├── syntax/               - Syntax validation and fixing
├── scripts/              - Utility scripts
├── results/              - Result collection
├── tests/                - Test suite
├── examples/             - Example implementations
```

## Template System

The template system supports all major HuggingFace model architectures:

1. **Encoder-Only Models** (BERT, RoBERTa, etc.)
2. **Decoder-Only Models** (GPT-2, LLaMA, etc.)
3. **Encoder-Decoder Models** (T5, BART, etc.)
4. **Vision Models** (ViT, ResNet, etc.)
5. **Vision-Text Models** (CLIP, BLIP, etc.)
6. **Speech Models** (Whisper, Wav2Vec2, etc.)

Each template handles architecture-specific requirements like imports, model classes, and testing approaches.

## Usage

### Generating a Coverage Report

```bash
make report
```

This command analyzes the current state of HuggingFace model test coverage and generates a report showing:

- Overall coverage statistics
- Coverage by architecture
- Missing models by priority
- Implementation plan

### Generating High Priority Models

```bash
make generate-high
```

This generates test files for all high-priority missing models from the roadmap.

### Generating Models by Architecture

```bash
make generate-arch ARCH=decoder-only
```

This generates test files for all missing models of the specified architecture.

### Generating from HuggingFace API

```bash
make huggingface MODEL_TYPE=llama QUERY="llama-3" LIMIT=3
```

This searches the HuggingFace API for LLaMA-3 models and generates test files for them.

### Using Advanced Generator with Auto-fixing

```bash
make advanced ARCH=vision-text BATCH_SIZE=5
```

This uses the advanced generator with automatic syntax fixing for 5 vision-text models.

### Exporting Coverage Matrix

```bash
make export-matrix FORMATS="markdown html csv"
```

This exports the coverage matrix in multiple formats for reporting and visualization.

### Validating Generated Tests

```bash
make validate
```

This validates the syntax and structure of generated test files.

## Advanced Features

### Customizing Templates

Templates can be customized to handle specific model types with special requirements:

```python
from templates import TemplateBase

class CustomTemplate(TemplateBase):
    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update({
            "name": "CustomTemplate",
            "description": "Custom template for specialized models",
            "supported_architectures": ["custom-arch"],
            "supported_models": ["custom-model"]
        })
        return metadata
    
    def get_template_str(self):
        return """
# Custom template content
{{ model_info.name }} implementation
"""
```

### Automatic Syntax Fixing

The system includes an automatic syntax fixer that can handle common issues:

- Missing parentheses and brackets
- Unterminated strings
- Indentation errors
- Import statement issues
- Duplicate imports
- And more

### Batch Processing with Progress Tracking

For large-scale test generation, use the batch processing capabilities with progress tracking:

```bash
# Generate first batch
make batch BATCH_SIZE=20 START_INDEX=0

# Generate next batch (automatically indicated in output)
make batch BATCH_SIZE=20 START_INDEX=20
```

## Development

To set up a development environment:

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the test suite:
   ```bash
   pytest
   ```

## Contributing

Contributions to improve the generator suite are welcome! Please follow these steps:

1. Create an issue describing the improvement
2. Fork the repository
3. Create a branch for your changes
4. Make your changes and add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.