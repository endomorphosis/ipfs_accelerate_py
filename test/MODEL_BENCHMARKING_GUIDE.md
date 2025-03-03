# Model Benchmarking Guide

This guide explains how to use the model benchmarking tools to verify model functionality and measure performance across different hardware platforms.

## Overview

The IPFS Accelerate framework provides comprehensive tools for benchmarking and validating models:

1. **Functionality Verification**: Verify that models can be loaded and run on different hardware platforms
2. **Performance Benchmarking**: Measure detailed performance metrics like latency, throughput, and memory usage
3. **Visualization**: Generate plots to compare performance across models and hardware
4. **Hardware Compatibility Matrix**: Track which models work on different hardware platforms

## Key Components

- `verify_model_functionality.py`: Tests if models can be loaded and run on different hardware
- `hardware_benchmark_runner.py`: Measures detailed performance metrics for models
- `run_benchmark_suite.py`: A suite runner for hardware benchmarks
- `run_model_benchmarks.py`: A comprehensive tool that combines functionality verification and performance benchmarking

## Quick Start

To run a complete benchmark of key models on all available hardware:

```bash
python test/run_model_benchmarks.py --output-dir ./benchmark_results
```

For a faster run with smaller models:

```bash
python test/run_model_benchmarks.py --models-set small --output-dir ./benchmark_results
```

To benchmark only specific models:

```bash
python test/run_model_benchmarks.py --specific-models bert t5 vit --output-dir ./benchmark_results
```

## Command-Line Options

### `run_model_benchmarks.py` Options

| Option | Description |
|--------|-------------|
| `--output-dir` | Directory to save benchmark results (default: `./benchmark_results`) |
| `--models-set` | Which model set to use: `key` (default), `small`, or `custom` |
| `--custom-models` | JSON file with custom models configuration (required if `models-set=custom`) |
| `--hardware` | Hardware platforms to test (defaults to all available) |
| `--batch-sizes` | Batch sizes to test (default: `1 4 8`) |
| `--verify-only` | Only verify functionality without performance benchmarks |
| `--benchmark-only` | Only run performance benchmarks without verification |
| `--no-plots` | Disable plot generation |
| `--no-compatibility-update` | Disable compatibility matrix update |
| `--no-resource-pool` | Disable ResourcePool for model caching |
| `--specific-models` | Only benchmark specific models from the selected set |
| `--debug` | Enable debug logging |

### Key Model Sets

The benchmarking tools include predefined model sets for different purposes:

#### Key Models
The complete set of 13 critical models:
- bert
- clap
- clip
- detr
- llama
- llava
- llava_next
- qwen2
- t5
- vit
- wav2vec2
- whisper
- xclip

#### Small Models
A smaller set for faster testing:
- bert (tiny)
- t5 (tiny)
- vit (tiny)
- whisper (tiny)
- clip (base)

#### Custom Models
Define your own set of models by creating a JSON file:

```json
{
  "custom_bert": {
    "name": "prajjwal1/bert-tiny",
    "family": "embedding",
    "size": "tiny",
    "modality": "text"
  },
  "custom_t5": {
    "name": "google/t5-efficient-tiny", 
    "family": "text_generation",
    "size": "tiny",
    "modality": "text"
  }
}
```

Then use it with:
```bash
python test/run_model_benchmarks.py --models-set custom --custom-models my_models.json
```

## Output and Reports

The benchmarking tools generate comprehensive output:

1. **JSON Results**: Raw benchmark data for each model and hardware platform
2. **Markdown Reports**: Human-readable reports summarizing results
3. **Visualizations**: Plots showing performance comparisons
4. **Hardware Compatibility Matrix**: JSON file tracking model compatibility across hardware

Example report sections:
- Hardware platforms detected
- Models tested
- Functionality verification results
- Performance benchmark results
- Hardware compatibility matrix
- Recommendations based on results
- Next steps

## Hardware Support

The benchmarking tools automatically detect and test on available hardware:

- CPU (always available)
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- OpenVINO
- ROCm (AMD GPUs)

## Use Cases

### Verify Model Functionality Across Hardware

To check if models run correctly on different hardware:

```bash
python test/run_model_benchmarks.py --verify-only --specific-models bert t5 vit
```

### Measure Performance for Specific Hardware

To benchmark models only on specific hardware:

```bash
python test/run_model_benchmarks.py --hardware cuda cpu --models-set small
```

### Generate New Hardware Compatibility Matrix

To update the hardware compatibility matrix:

```bash
python test/run_model_benchmarks.py --verify-only
```

### Comprehensive Benchmarks for All Models

To run complete benchmarks for all key models:

```bash
python test/run_model_benchmarks.py --output-dir ./comprehensive_benchmarks
```

## Integration with ResourcePool

The benchmarking tools integrate with the ResourcePool system for efficient model caching and memory management. To disable ResourcePool integration:

```bash
python test/run_model_benchmarks.py --no-resource-pool
```

## Best Practices

1. Start with the `small` model set to verify everything works correctly
2. Use `--verify-only` for a quick check of model functionality
3. Use `--benchmark-only` when you've already verified functionality
4. Consider running benchmarks overnight for the full `key` model set
5. Use `--specific-models` to focus on models relevant to your use case
6. Review the hardware compatibility matrix to understand which models work on different hardware
7. Use the resource pool to improve benchmark speed for repeated runs

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes or benchmark fewer models at once
2. **Model Not Found**: Check if you're using the correct model name or try a smaller model
3. **Hardware Not Detected**: Ensure the required drivers and libraries are installed
4. **Plot Generation Fails**: Install missing matplotlib and pandas libraries

### Debug Mode

For detailed output, use the `--debug` flag:

```bash
python test/run_model_benchmarks.py --models-set small --debug
```

## Additional Resources

- Check the `CLAUDE.md` file for the latest performance benchmarks
- See `hardware_compatibility_matrix.json` for the current hardware compatibility status
- Review `MODEL_FAMILY_CLASSIFIER_GUIDE.md` for information on model family classification
- Refer to `RESOURCE_POOL_GUIDE.md` for details on resource management