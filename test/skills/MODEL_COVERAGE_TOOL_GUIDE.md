# HuggingFace Model Coverage Tool Guide

This guide explains how to use the `create_coverage_tool.py` utility to generate, manage, and track test coverage for HuggingFace models in the IPFS Accelerate Python framework.

## Overview

The Model Coverage Tool helps automate the process of:

1. Generating test files for HuggingFace model types
2. Tracking which models have test coverage
3. Generating comprehensive coverage reports
4. Handling hyphenated model names correctly

The tool uses a token-based replacement strategy that properly handles code structure and addresses issues with indentation, unterminated strings, and other syntax problems that can occur during template-based generation.

## Installation

No additional installation is required beyond the existing dependencies in the IPFS Accelerate Python framework.

## Basic Usage

### Generate a Test for a Specific Model

```bash
python create_coverage_tool.py --generate bert
```

This generates a test file for the BERT model type using the appropriate template. The output file will be saved in the default output directory (`./output_tests/`).

### Generate Tests in Batches

```bash
python create_coverage_tool.py --batch 10
```

This command:
1. Identifies models that don't have tests yet
2. Generates tests for the first 10 missing models
3. Updates the coverage report

### List Models Without Tests

```bash
python create_coverage_tool.py --list-missing
```

Displays a list of all model types that don't have tests yet.

### Update the Coverage Report

```bash
python create_coverage_tool.py --update-report
```

Scans existing test files and generates an updated coverage report in `MODEL_TEST_COVERAGE.md`.

### List All Supported Model Types

```bash
python create_coverage_tool.py --list-all
```

Displays a list of all supported model types.

## Advanced Usage

### Specify an Output Directory

```bash
python create_coverage_tool.py --generate llama --output-dir /path/to/output
```

Generates a test file in the specified output directory.

### Using with CI/CD

The tool can be easily integrated into CI/CD workflows to automatically generate tests for new models:

```yaml
# Example GitHub Actions workflow
name: Generate Missing Model Tests

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  generate-tests:
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
      - name: Generate missing tests
        run: |
          python create_coverage_tool.py --batch 20
      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          title: 'Auto-generate tests for HuggingFace models'
          body: 'Automatically generated tests for HuggingFace models.'
          branch: auto-generate-model-tests
```

## Working with Hyphenated Models

The tool automatically handles hyphenated model names like `xlm-roberta` and `gpt-j` by:

1. Converting them to valid Python identifiers (e.g., `xlm_roberta`) for variable names
2. Converting them to PascalCase (e.g., `XlmRoberta`) for class names
3. Applying specialized fixes for indentation and syntax issues

For example:

```bash
python create_coverage_tool.py --generate gpt-j
```

This generates a syntactically valid test file for the GPT-J model with proper handling of the hyphenated name.

## Architecture Support

The tool supports all major architecture types:

1. **encoder-only**: BERT, RoBERTa, DistilBERT, etc.
2. **decoder-only**: GPT-2, LLaMA, Mistral, etc.
3. **encoder-decoder**: T5, BART, PEGASUS, etc.
4. **vision**: ViT, Swin, DeiT, etc.
5. **vision-text**: CLIP, BLIP, etc.
6. **speech**: Whisper, Wav2Vec2, HuBERT, etc.
7. **multimodal**: LLaVA, FLAVA, etc.

## Troubleshooting

### Syntax Errors in Generated Files

If you encounter syntax errors in generated files, check for:

1. Issues with the template file
2. Special handling required for the specific model type
3. Problems with indentation or unterminated strings

The tool includes robust error handling and logging to help diagnose issues.

### Missing Model Types

If a model type is missing, you can add it to the `ARCHITECTURE_TYPES` dictionary in the tool. This ensures it gets detected correctly when generating test coverage reports.

## Contributing

To contribute improvements to the Model Coverage Tool:

1. Add new template types for additional architectures
2. Improve the token-based replacement system
3. Enhance syntax error handling
4. Update the model registry with new models

## Conclusion

The Model Coverage Tool makes it easy to achieve comprehensive test coverage for all HuggingFace model types in the IPFS Accelerate Python framework. By automating the generation and tracking of test files, it significantly reduces the effort required to maintain full model coverage.