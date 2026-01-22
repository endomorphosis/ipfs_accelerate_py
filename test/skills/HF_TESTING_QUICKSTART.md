# HF_TESTING_QUICKSTART.md
# HuggingFace Testing Framework Quickstart Guide

This guide provides a step-by-step introduction to using the comprehensive HuggingFace Testing Framework for testing, benchmarking, and publishing metrics for HuggingFace models.

## Overview

The HuggingFace Testing Framework includes several integrated components:

1. **Test Generator**: Creates and manages test files for 300+ HuggingFace model architectures
2. **Hardware Compatibility Matrix**: Tests models across multiple hardware platforms
3. **Distributed Testing Framework**: Distributes tests across multiple workers
4. **Test Dashboard**: Visualizes test results and performance metrics
5. **Benchmark Publisher**: Publishes performance metrics to HuggingFace Model Hub

## Installation

### Prerequisites

- Python 3.7+
- Git
- HuggingFace `transformers` and `tokenizers` libraries

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install dependencies
pip install -r requirements.txt

# Optional: Install hardware-specific dependencies
pip install torch torchvision torchaudio  # For PyTorch/CUDA support
pip install openvino                      # For OpenVINO support
```

## Quick Start

### 1. Run Tests for a Specific Model

```bash
cd test/skills/fixed_tests
python test_hf_bert.py --model bert-base-uncased --save
```

### 2. Generate Tests for a New Model

```bash
cd test/skills
python regenerate_fixed_tests.py --model roberta --verify
```

### 3. Create Hardware Compatibility Matrix

```bash
cd test/skills
python create_hardware_compatibility_matrix.py --architectures encoder-only decoder-only
```

### 4. Run Distributed Tests

```bash
cd test/skills
python update_for_distributed_testing.py --dir fixed_tests --verify --create-framework
python run_distributed_tests.py --model-family bert --workers 4
```

### 5. Generate Dashboard

```bash
cd test/skills
python create_test_dashboard.py --static --output-dir dashboard
```

### 6. Publish Benchmarks to Model Hub

```bash
cd test/skills
python publish_model_benchmarks.py --token YOUR_HF_TOKEN
```

## Key Components in Detail

### Test Generator (`test_generator_fixed.py`)

The test generator creates properly formatted Python test files for any HuggingFace model:

```bash
# Generate test for a specific model
python regenerate_fixed_tests.py --model bert --verify

# Generate tests for all models
python regenerate_fixed_tests.py --all --verify

# Generate tests for missing high-priority models
python generate_missing_model_tests.py --priority high --verify
```

### Hardware Compatibility Matrix (`create_hardware_compatibility_matrix.py`)

Tests models across different hardware platforms and generates compatibility reports:

```bash
# Test specific models on all available hardware
python create_hardware_compatibility_matrix.py --models bert-base-uncased gpt2 t5-small

# Test specific architecture categories
python create_hardware_compatibility_matrix.py --architectures vision multimodal

# Only detect hardware without running tests
python create_hardware_compatibility_matrix.py --detect-only
```

### Distributed Testing Framework

Distributes tests across multiple workers for parallel execution:

```bash
# Update test files to support distributed testing
python update_for_distributed_testing.py --dir fixed_tests --verify

# Run distributed tests for specific model family
python run_distributed_tests.py --model-family bert --workers 4

# Run all tests with distributed framework
python run_distributed_tests.py --all --workers 8

# Check available hardware
python run_distributed_tests.py --hardware-check
```

### Test Dashboard (`create_test_dashboard.py`)

Creates visualization dashboards for test results:

```bash
# Generate static HTML dashboard
python create_test_dashboard.py --static --output-dir dashboard

# Launch interactive dashboard server
python create_test_dashboard.py --interactive --port 8050

# Specify data sources
python create_test_dashboard.py --results-dir collected_results --dist-dir distributed_results --hardware-db hardware_compatibility_matrix.duckdb
```

### Benchmark Publisher (`publish_model_benchmarks.py`)

Publishes performance metrics to HuggingFace Model Hub:

```bash
# Publish benchmarks to Model Hub
python publish_model_benchmarks.py --token YOUR_HF_TOKEN

# Run in dry-run mode (no publishing)
python publish_model_benchmarks.py --dry-run

# Save reports locally
python publish_model_benchmarks.py --local --output-dir benchmark_reports
```

## Working with Test Results

Test results are stored in different formats:

1. **JSON files** in `collected_results/` directory (standard test results)
2. **JSON files** in `distributed_results/` directory (distributed test results)
3. **DuckDB database** in `hardware_compatibility_matrix.duckdb` (hardware benchmarks)

## Common Workflows

### Complete Test Suite Execution

```bash
# 1. Generate tests for all model architectures
python regenerate_fixed_tests.py --all --verify

# 2. Run hardware compatibility matrix generation
python create_hardware_compatibility_matrix.py --all

# 3. Run distributed tests for all models
python run_distributed_tests.py --all --workers 8

# 4. Generate visualization dashboard
python create_test_dashboard.py --static --output-dir dashboard

# 5. Publish benchmarks to Model Hub
python publish_model_benchmarks.py --token YOUR_HF_TOKEN
```

### CI/CD Integration

The framework includes GitHub Actions workflow files:

- `github-workflow-test-generator.yml`: For test generation and validation
- `github-actions-test-validation.yml`: For test execution and reporting

## Advanced Features

### Mock Detection System

The tests automatically detect when they're running in a CI/CD environment without real models:

- 🚀 indicates real model inference
- 🔷 indicates mock objects for CI/CD environments

### Architecture-Specific Templates

Each model architecture has a specialized template in the `templates/` directory:

- `encoder_only_template.py`: For BERT, RoBERTa, etc.
- `decoder_only_template.py`: For GPT-2, LLaMA, etc.
- `encoder_decoder_template.py`: For T5, BART, etc.
- `vision_template.py`: For ViT, Swin, etc.
- `multimodal_template.py`: For CLIP, BLIP, etc.
- `speech_template.py`: For Whisper, Wav2Vec2, etc.

## Troubleshooting

### Common Issues

- **Missing Dependencies**: Install task-specific dependencies (torch, tokenizers, etc.)
- **GPU Detection Issues**: Check CUDA installation and environment variables
- **Test Failures**: Look for mock indicators (🔷) which suggest missing dependencies

### Getting Help

For detailed documentation, refer to:

- `fixed_tests/README.md`: Core test implementation details
- `TESTING_FIXES_SUMMARY.md`: Overview of all framework components
- `HARDWARE_COMPATIBILITY_README.md`: Hardware testing details
- `DISTRIBUTED_TESTING_README.md`: Distributed testing framework
- `DASHBOARD_README.md`: Dashboard and visualization tools
- `BENCHMARK_PUBLISHING_README.md`: Benchmark publishing to Model Hub

## Next Steps

1. Explore the more detailed documentation for each component
2. Try testing your own custom HuggingFace models
3. Integrate with your existing CI/CD pipelines
4. Contribute to the framework by adding support for new architectures
