# Hardware Benchmarking Tools

This directory contains tools for benchmarking machine learning models across different hardware backends. These tools help you measure performance (latency, throughput, memory usage) and make informed decisions about hardware selection for your models.

## Overview

The benchmarking suite includes:

- **Model support**: BERT, T5, ViT, Whisper models from Hugging Face
- **Hardware support**: CPU, CUDA (NVIDIA GPUs), ROCm (AMD GPUs), MPS (Apple Silicon), OpenVINO
- **Metrics**: Latency, throughput, memory usage for different batch sizes
- **Reports**: Markdown, JSON, and CSV formats with comprehensive comparisons

## Available Tools

### 1. Interactive Benchmarking Tool

The easiest way to run benchmarks is with the interactive tool:

```bash
./interactive_hardware_benchmark.py
```

This tool will:
- Detect available hardware backends on your system
- Guide you through selecting models, hardware, and benchmark parameters
- Run the benchmarks and save reports

### 2. Command-line Benchmark Runner

For more control or automation, use the shell script:

```bash
./run_hardware_benchmark.sh [options]
```

Options:
- `--model-set MODEL_SET`: Set of models to benchmark (text_embedding, text_generation, vision, audio, all, quick)
- `--models "MODEL1 MODEL2..."`: Space-separated list of specific models to benchmark
- `--hardware-set HARDWARE_SET`: Set of hardware backends to test (local, gpu, intel, all, web, quick)
- `--hardware "HW1 HW2..."`: Space-separated list of specific hardware backends to test
- `--batch-sizes "SIZE1 SIZE2..."`: Space-separated list of batch sizes to test (default: "1 4 16")
- `--format FORMAT`: Output format: markdown, json, csv (default: markdown)
- `--debug`: Enable debug logging
- `--openvino-precision PRECISION`: Precision for OpenVINO models: FP32, FP16, INT8 (default: FP32)
- `--warmup COUNT`: Number of warmup runs (default: 2)
- `--runs COUNT`: Number of measurement runs (default: 5)
- `--output-dir DIR`: Directory to save benchmark results (default: ./benchmark_results)

### 3. Python Benchmarking API

For maximum flexibility or integration into other tools, use the Python API directly:

```python
from run_hardware_comparison import run_benchmark, generate_markdown_report, generate_csv_report

# Run benchmark for a model on specific hardware
result = run_benchmark(
    model_name="prajjwal1/bert-tiny",
    hardware="cuda",
    batch_sizes=[1, 4, 16],
    warmup=2,
    runs=5
)

# Generate report from multiple results
import os
from pathlib import Path
output_dir = Path("./benchmark_results")
os.makedirs(output_dir, exist_ok=True)

all_results = {
    "prajjwal1/bert-tiny": {
        "model_name": "prajjwal1/bert-tiny",
        "hardware_results": {
            "cpu": result_cpu,
            "cuda": result_cuda
        }
    }
}

# Generate reports in different formats
md_report_file = generate_markdown_report(all_results, output_dir)
csv_report_file = generate_csv_report(all_results, output_dir)
```

## Model Sets

The following predefined model sets are available:

- **text_embedding**: BERT models for text embeddings
  - prajjwal1/bert-tiny, bert-base-uncased
- **text_generation**: T5 models for text generation
  - google/t5-efficient-tiny, t5-small
- **vision**: Vision Transformer (ViT) models
  - google/vit-base-patch16-224
- **audio**: Whisper models for speech recognition
  - openai/whisper-tiny
- **all**: All model types listed above
- **quick**: Fast subset for quick testing
  - prajjwal1/bert-tiny, google/t5-efficient-tiny

## Hardware Sets

The following predefined hardware sets are available:

- **local**: Just CPU (universally available)
- **gpu**: CUDA (NVIDIA GPUs)
- **intel**: CPU and OpenVINO
- **all**: All available hardware (CPU, CUDA, ROCm, MPS, OpenVINO)
- **web**: CPU + WebNN + WebGPU (simulation mode)
- **quick**: CPU + CUDA (if available)

## Example Usage

### Interactive Mode

```bash
./interactive_hardware_benchmark.py
```

### Benchmark BERT Models on CPU and CUDA

```bash
./run_hardware_benchmark.sh --model-set text_embedding --hardware "cpu cuda" --batch-sizes "1 4 16 32"
```

### Benchmark Custom Models and Export to CSV for Data Analysis

```bash
./run_hardware_benchmark.sh --models "prajjwal1/bert-tiny google/t5-efficient-tiny" --hardware "cpu openvino" --format csv
```

The CSV format is especially useful for importing benchmark results into spreadsheet applications for custom analysis and visualization.

### Run Quick Benchmarks with Default Settings

```bash
./run_hardware_benchmark.sh --model-set quick --hardware-set quick
```

## Reports

Reports are saved in the specified output directory (default: ./benchmark_results) and are available in three formats:

### Markdown Reports (.md)
Comprehensive reports with formatted tables and sections:
1. **Summary table**: Overview of latency and throughput for each model-hardware combination
2. **Hardware comparison**: Throughput comparison across hardware platforms for different batch sizes
3. **Detailed results**: Per-model, per-hardware detailed metrics including memory usage

### CSV Reports (.csv)
Tabular data format ideal for importing into spreadsheet applications or data analysis tools:
1. **Structured data**: All benchmark results in a single table with columns for model, hardware, batch size, and metrics
2. **Complete metrics**: Includes all metrics (latency, throughput, memory) for all model-hardware-batch size combinations
3. **Analysis-ready**: Perfect for creating custom charts or performing statistical analysis

### JSON Reports (.json)
Machine-readable format ideal for programmatic processing:
1. **Complete data**: Contains all raw benchmark data including detailed metrics and test parameters
2. **Hierarchical structure**: Organized by model, hardware, and batch size
3. **Integration-friendly**: Easily parsed by any programming language

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- NumPy
- OpenVINO (optional, for OpenVINO benchmarks)

## Tips

- Start with the `quick` model set to verify everything works
- Use `--debug` flag when troubleshooting issues
- For accurate results, close other applications and ensure consistent system load
- Run each benchmark multiple times to account for variance
- When benchmarking on CPU, be aware that background processes can affect results
- Adjust batch sizes based on your model size and available memory