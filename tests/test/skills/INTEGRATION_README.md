# HuggingFace Test Automation Toolkit

This toolkit provides a comprehensive solution for generating and maintaining test files for HuggingFace transformers models. It addresses indentation issues and implements an architecture-aware approach to test generation.

## Key Features

- **Architecture-Aware Test Generation**: Creates tests tailored to the model architecture (encoder-only, decoder-only, etc.)
- **Indentation Fixing**: Comprehensive tools to fix Python indentation issues
- **Automated Testing**: Run tests across different hardware targets (CPU, CUDA, OpenVINO)
- **Comprehensive Reporting**: Generate detailed test coverage reports
- **Integration Pipeline**: End-to-end test generation, fixing, verification, and execution

## Components

1. **Test Generator**: Creates architecture-appropriate test files for each model type
2. **Indentation Fixer**: Fixes common indentation issues in Python files
3. **Integration Script**: Orchestrates the entire process from generation to reporting
4. **Templates**: Architecture-specific templates for different model families
5. **Reporting Tools**: Generates comprehensive coverage reports

## Quick Start

### Generate a Test for a Specific Model

```bash
python test_integration.py --generate --models bert
```

### Fix Indentation in Existing Test Files

```bash
python test_integration.py --fix
```

### Verify Test File Syntax

```bash
python test_integration.py --verify
```

### Run Tests

```bash
python test_integration.py --run
```

### End-to-End Integration

```bash
# Generate, fix, verify, run, and report for core models
python test_integration.py --all --core

# Generate, fix, verify, run, and report for a specific architecture
python test_integration.py --all --arch encoder-only

# Generate, fix, verify, run, and report for specific models
python test_integration.py --all --models "bert,roberta,distilbert"
```

### Generate Coverage Report

```bash
python test_integration.py --report
```

## Architecture Support

The toolkit supports the following model architectures:

1. **Encoder-Only** (BERT, RoBERTa, DistilBERT, etc.)
2. **Decoder-Only** (GPT-2, LLaMA, Mistral, etc.)
3. **Encoder-Decoder** (T5, BART, Pegasus, etc.)
4. **Vision** (ViT, Swin, DeiT, etc.)
5. **Vision-Encoder-Text-Decoder** (CLIP, BLIP, etc.)
6. **Speech** (Wav2Vec2, Whisper, Bark, etc.)
7. **Multimodal** (LLaVA, CLIP, BLIP, etc.)

## Test Structure

Each generated test file follows a consistent structure:

1. **Hardware Detection**: Identifies available hardware (CPU, CUDA, MPS, OpenVINO)
2. **Model Registry**: Defines model-specific configurations and default values
3. **Test Class**: The primary test class with methods for different testing approaches
4. **Pipeline Test**: Tests using the `transformers.pipeline()` API
5. **From Pretrained Test**: Tests using direct model loading with `from_pretrained()`
6. **Hardware-Specific Tests**: Tests tailored to different hardware backends
7. **Results Collection**: Comprehensive performance and functionality metrics

## Indentation Fixing

The indentation fixer addresses common issues in Python files:

1. **Method Boundaries**: Fixes spacing and indentation around method definitions
2. **Dependency Checks**: Properly indents dependency check blocks
3. **Try/Except Blocks**: Fixes indentation in error handling sections
4. **Mock Classes**: Correctly indents mock class implementations
5. **Bracket Matching**: Fixes issues with parentheses, brackets, and braces

## Advanced Usage

### Custom Templates

You can create custom templates for specific model architectures:

```
templates/
  encoder_only_template.py
  decoder_only_template.py
  encoder_decoder_template.py
  vision_template.py
  vision_text_template.py
  speech_template.py
  multimodal_template.py
```

### Batch Processing

Generate tests for multiple models at once:

```bash
python test_integration.py --all --models "bert,gpt2,t5,vit,clip,whisper,llava"
```

### Architecture-Specific Testing

Focus on a specific model architecture:

```bash
python test_integration.py --all --arch decoder-only
```

## Integration with CI/CD

Add this to your CI/CD pipeline:

```yaml
# GitHub Actions example
name: HuggingFace Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run test integration
      run: |
        cd test/skills
        python test_integration.py --all --core
```

## Troubleshooting

See [HF_TEST_TROUBLESHOOTING_GUIDE.md](HF_TEST_TROUBLESHOOTING_GUIDE.md) for common issues and solutions.

## Roadmap

1. **Phase 1**: Complete core model tests (BERT, GPT-2, T5, ViT)
2. **Phase 2**: Add tests for 20 high-priority models
3. **Phase 3**: Implement 50 models across all architecture categories
4. **Phase 4**: Achieve 100% coverage for all 300+ model types

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to the test framework.

## License

Apache 2.0