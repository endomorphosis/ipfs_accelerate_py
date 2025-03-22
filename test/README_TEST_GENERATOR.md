# HuggingFace Model Test Generator

A comprehensive test generator system for creating test files for all HuggingFace model architectures.

## Overview

This test generator system automatically creates test files for HuggingFace models based on their architecture type:

- **Encoder-only models**: BERT, RoBERTa, DistilBERT, etc.
- **Decoder-only models**: GPT-2, LLaMA, Mistral, etc.
- **Encoder-decoder models**: T5, BART, etc.
- **Vision models**: ViT, ResNet, etc.
- **Vision-text models**: CLIP, BLIP, etc.
- **Speech models**: Whisper, Wav2Vec2, etc.

The system includes two implementations:
1. A component-based architectural implementation in `refactored_generator_suite/`
2. A simplified template-based implementation in `simple_generator.py`

## Quick Start

### Using the Simplified Generator

To generate a test for a specific model:

```bash
python simple_generator.py --model gpt2 --output-dir ./generated_tests
```

To generate tests for all models of a specific architecture:

```bash
python simple_generator.py --architecture decoder-only --output-dir ./generated_tests --limit 3
```

To generate tests for all architectures:

```bash
python generate_all_model_tests.py --output-dir ./generated_tests
```

### Using the Architectural Generator

```bash
python refactored_generator_suite/run_generator.py --model gpt2 --output-dir ./generated_tests
```

## Features

- **Architecture Mapping**: Maps model types to their architectures for appropriate template selection
- **Template Rendering**: Handles Jinja syntax for variable replacement and conditional blocks
- **Syntax Validation**: Validates and attempts to fix syntax in generated files
- **Batch Generation**: Supports generating tests for multiple models in one run
- **Reporting**: Generates summary reports of the generation process
- **Hardware Awareness**: Detects available hardware (CUDA, MPS, etc.) for inclusion in test files

## Templates

Templates are located in:
- `skills/templates/` - Original templates
- `refactored_generator_suite/templates/` - Refactored templates

Each template corresponds to a specific model architecture and includes:
- Hardware detection
- Model-specific test cases
- Mock implementations for CI/CD environments

## Documentation

For more information, see:
- [Test Generator System Summary](TEST_GENERATOR_SYSTEM_SUMMARY.md)
- [TODO List](TEST_GENERATOR_TODO.md)
- [HuggingFace Test Files Guide](HF_TEST_IMPLEMENTATION_CHECKLIST.md)

## Development

When developing the generator system:

1. **Adding Models**: Update architecture mappings in `registry.py` or `simple_generator.py`
2. **Modifying Templates**: Edit template files in the appropriate directory
3. **Testing Changes**: Use the provided scripts to generate tests and verify output

## License

This project is part of the IPFS Accelerate Python Framework and is subject to the same license terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- HuggingFace team for their incredible model ecosystem
- The team at IPFS Accelerate for supporting this work