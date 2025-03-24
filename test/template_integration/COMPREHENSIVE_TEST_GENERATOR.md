# Comprehensive Test Generator for HuggingFace Transformers

The Comprehensive Test Generator is a single-entry-point tool for generating test files that cover all HuggingFace Transformers model classes. It ensures complete coverage of the `from_pretrained` and pipeline methods across the entire library.

## Features

- **Complete Coverage**: Automatically discovers all HuggingFace model classes with `from_pretrained` support
- **Single Entry Point**: One tool to generate tests for all model types
- **Architecture Categorization**: Properly categorizes models by architecture type (vision, text, speech, multimodal)
- **Intelligent Model Selection**: Uses recommended models for each class
- **Pipeline Task Mapping**: Automatically selects appropriate pipeline tasks for each model
- **Parallel Processing**: Utilizes multithreading for efficient test generation
- **Validation**: Integrated validation of generated test files
- **Reporting**: Comprehensive coverage reports to track progress

## Architecture Categories

The generator supports all major model architectures in the HuggingFace Transformers library:

1. **Vision Models**: ViT, DeiT, BEiT, ConvNeXT, Swin, SegFormer, DETR, etc.
2. **Encoder-Only Models**: BERT, RoBERTa, ALBERT, ELECTRA, DistilBERT, etc.
3. **Decoder-Only Models**: GPT2, OPT, GPTNeo, GPTJ, LLaMA, etc.
4. **Encoder-Decoder Models**: T5, BART, FlanT5, MT5, mBART, etc.
5. **Speech/Audio Models**: Whisper, Wav2Vec2, HuBERT, SEW, EnCodec, etc.
6. **Multimodal Models**: CLIP, BLIP, LLaVA, FLAVA, etc.

## Usage

### Basic Usage

```bash
./comprehensive_test_generator.py
```

This will discover all Transformers classes and generate test files for them in the default output directory.

### Discovery Only

```bash
./comprehensive_test_generator.py --discover-only --discovery-output transformers_classes.json
```

This will just discover and save information about the classes without generating tests.

### Generate Tests for Specific Categories

```bash
./comprehensive_test_generator.py --categories vision multimodal
```

This will generate tests only for vision and multimodal models.

### Generate Tests for Specific Class Types

```bash
./comprehensive_test_generator.py --classes BERT GPT2 CLIP
```

This will generate tests for classes starting with BERT, GPT2, or CLIP.

### Dry Run

```bash
./comprehensive_test_generator.py --dry-run --categories vision
```

Shows what tests would be generated without actually creating files.

### Advanced Options

```bash
./comprehensive_test_generator.py --output-dir /custom/output/path --max-workers 8 --overwrite --report-file vision_report.md --categories vision
```

- `--output-dir`: Custom output directory for test files
- `--max-workers`: Number of parallel workers for test generation
- `--overwrite`: Overwrite existing test files
- `--report-file`: Custom file for the coverage report
- `--results-file`: Custom file for the JSON results

## Generated Tests

Each generated test file includes comprehensive test coverage:

1. **Model Loading**: Tests loading the model with `from_pretrained`
2. **Pipeline API**: Tests using the model with the appropriate pipeline
3. **Direct Inference**: Tests direct model usage for inference
4. **Hardware Support**: Tests with different hardware configurations (CPU, CUDA, MPS)
5. **OpenVINO Integration**: Tests compatibility with Intel OpenVINO

## Reports

The generator creates two types of reports:

1. **Results JSON**: Contains detailed success/failure information for each test
2. **Coverage Report**: Markdown report showing test coverage by category

## Integration with Template System

The generator builds on the existing template system but provides a more comprehensive approach:

1. Tries to use the template system first
2. Falls back to reference file-based generation if the template system fails
3. Ensures consistent test files across all model types

## Benefits

- **Reduced Code Duplication**: Single code path for all test generation
- **Improved Maintainability**: Centralized configuration and templates
- **Better Coverage Tracking**: Comprehensive reports for test coverage
- **Faster Development**: Parallel generation of test files
- **Future-Proof**: Automatically discovers new model classes in Transformers updates