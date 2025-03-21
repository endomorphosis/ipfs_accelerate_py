# HuggingFace Model Hub Benchmark Publisher

This tool automatically publishes hardware compatibility and performance benchmarks to HuggingFace Model Hub model cards, enhancing model discoverability and providing valuable information to users.

## Features

- **Automated Benchmark Publishing**: Extract performance metrics from your `hardware_compatibility_matrix.duckdb` database and publish them directly to model cards
- **Standardized Performance Badges**: Generate eye-catching badges showing key performance metrics for quick comparisons
- **Comprehensive Metrics**: Include inference time, memory usage, and model loading time across multiple hardware platforms
- **Local Report Generation**: Option to save benchmark reports locally for review before publishing
- **Integration with CI/CD**: Can be integrated into CI/CD pipelines for automated benchmark updates

## Prerequisites

- Python 3.7+
- Required packages:
  - `duckdb`: For accessing the benchmark database
  - `pandas`: For data manipulation
  - `huggingface_hub`: For interacting with the Hugging Face Hub API

## Installation

```bash
pip install duckdb pandas huggingface_hub
```

## Usage

### Publishing Benchmarks to Model Hub

```bash
python publish_model_benchmarks.py --token YOUR_HF_TOKEN
```

### Running in Dry-Run Mode (No Publishing)

```bash
python publish_model_benchmarks.py --dry-run
```

### Saving Reports Locally

```bash
python publish_model_benchmarks.py --local --output-dir benchmark_reports
```

### Publishing Benchmarks for a Specific Model

```bash
python publish_model_benchmarks.py --token YOUR_HF_TOKEN --model bert-base-uncased
```

### Limiting the Number of Models

```bash
python publish_model_benchmarks.py --token YOUR_HF_TOKEN --limit 10
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--db` | Path to DuckDB database with benchmark results (default: `hardware_compatibility_matrix.duckdb`) |
| `--token` | HuggingFace API token (or set the `HF_TOKEN` environment variable) |
| `--dry-run` | Generate benchmark data but don't publish to Hub |
| `--local` | Save benchmark reports locally instead of publishing to Hub |
| `--output-dir` | Directory to save local benchmark reports (default: `benchmark_reports`) |
| `--limit` | Limit the number of models to process |
| `--model` | Process only the specified model ID |

## Example Output

When publishing benchmarks for a model, the tool adds:

1. **Performance Badges**: Visual indicators of performance on different hardware
2. **Benchmark Table**: Detailed metrics for each hardware platform
3. **CPU-GPU Speedup**: Comparative performance between CPU and GPU
4. **Methodology Section**: Description of the benchmarking process

Example benchmark section on a model card:

![CPU Benchmark](https://img.shields.io/badge/CPU%20Inference-42.5ms-blue) ![CUDA Benchmark](https://img.shields.io/badge/CUDA%20Inference-3.2ms-orange) 

## Performance Benchmarks

| Hardware | Inference Time (ms) | Memory Usage (MB) | Load Time (s) |
|----------|---------------------|-------------------|---------------|
| CPU | 42.5 | 553.2 | 1.24 |
| CUDA | 3.2 | 1245.8 | 1.38 |
| MPS | 8.4 | 876.3 | 1.42 |

*Benchmarks last updated: 2025-03-20 14:25:36*

*Measured with IPFS Accelerate Testing Framework*

**CPU to GPU Speedup: 13.3x**

## Benchmark Methodology

These benchmarks were collected using the IPFS Accelerate Testing Framework:

- **Inference Time**: Average time for a single forward pass
- **Memory Usage**: Peak memory consumption during inference  
- **Load Time**: Time to load model from disk and initialize

Measurements performed in a controlled environment with warm-up passes to ensure stability.

## Integration with CI/CD

For automated benchmark updates, you can integrate this tool with your CI/CD pipeline:

```yaml
# GitHub Actions workflow example
benchmark-update:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install duckdb pandas huggingface_hub
    - name: Publish benchmarks
      run: python publish_model_benchmarks.py
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

## Generating Scheduled Benchmarks

To keep benchmarks up-to-date, set up a scheduled job:

```bash
# Add to crontab for weekly benchmarks
0 0 * * 0 cd /path/to/project && python publish_model_benchmarks.py
```

## Troubleshooting

- **Authentication Issues**: Make sure your Hugging Face token has write access to the repositories
- **Missing Benchmarks**: Ensure your database contains records for the models you're trying to publish
- **Rate Limiting**: When publishing many benchmarks, add delays between requests

## Example Workflow

1. **Generate Hardware Benchmarks**:
   ```bash
   python create_hardware_compatibility_matrix.py --all
   ```

2. **Review Local Reports** (optional):
   ```bash
   python publish_model_benchmarks.py --local
   ```

3. **Publish to Model Hub**:
   ```bash
   python publish_model_benchmarks.py --token YOUR_HF_TOKEN
   ```

## Contributing

Contributions to improve the benchmark publisher are welcome! Please submit a pull request with your enhancements.

## License

This project is licensed under the Apache 2.0 License.