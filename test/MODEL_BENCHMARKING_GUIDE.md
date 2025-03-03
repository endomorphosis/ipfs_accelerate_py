# Model Benchmarking Guide

## Introduction

This guide provides instructions for benchmarking models across different hardware platforms using the IPFS Accelerate Python Framework. It covers performance testing, hardware compatibility verification, and optimization techniques.

## Quick Start

To run basic model benchmarks:

```bash
python test/run_model_benchmarks.py --output-dir ./benchmark_results
```

For a specific hardware platform:

```bash
python test/run_model_benchmarks.py --hardware cuda --output-dir ./benchmark_results
```

## Comprehensive Key Model Benchmarking

For benchmarking all 13 high-priority model classes across all hardware platforms, use the dedicated script:

```bash
python test/benchmark_all_key_models.py --output-dir ./benchmark_results
```

This script ensures complete hardware coverage for the following model classes:
- BERT (embedding)
- CLAP (audio)
- CLIP (multimodal)
- DETR (vision)
- LLAMA (text generation)
- LLaVA (multimodal)
- LLaVA-Next (multimodal)
- Qwen2 (text generation)
- T5 (text generation)
- ViT (vision)
- Wav2Vec2 (audio)
- Whisper (audio)
- XCLIP (multimodal)

To use smaller model variants for faster testing:

```bash
python test/benchmark_all_key_models.py --small-models --output-dir ./benchmark_results
```

To test specific hardware platforms:

```bash
python test/benchmark_all_key_models.py --hardware cpu cuda openvino --output-dir ./benchmark_results
```

The script automatically identifies and attempts to fix implementation issues for complete compatibility across platforms. To disable this behavior:

```bash
python test/benchmark_all_key_models.py --no-fix --output-dir ./benchmark_results
```

## Benchmark Configuration

The benchmark script accepts several configuration options:

- `--models-set`: Which model set to use ('key', 'small', or 'custom')
- `--hardware`: Hardware platforms to test
- `--batch-sizes`: Batch sizes to test
- `--verify-only`: Only verify functionality without performance benchmarks
- `--benchmark-only`: Only run performance benchmarks without verification
- `--specific-models`: Only benchmark specific models from the selected set

Example:

```bash
python test/run_model_benchmarks.py --models-set small --hardware cpu cuda --batch-sizes 1 4 8 --specific-models bert t5
```

## Hardware Platforms

The framework supports the following hardware platforms:

- `cpu`: CPU (always available)
- `cuda`: NVIDIA CUDA GPUs
- `rocm`: AMD ROCm GPUs
- `mps`: Apple Metal Performance Shaders (Apple Silicon)
- `openvino`: Intel OpenVINO
- `webnn`: Web Neural Network API
- `webgpu`: WebGPU API

## Model Sets

The framework provides predefined model sets for benchmarking:

- `key`: Standard set of key models for each family
- `small`: Smaller variants of models for faster testing
- `custom`: Custom model set defined in a JSON file

## Benchmark Process

The benchmark process includes:

1. Hardware detection to identify available platforms
2. Model functionality verification on each platform
3. Performance benchmark measurement with varying batch sizes
4. Result visualization and reporting
5. Hardware compatibility matrix generation

## Implementation Issue Detection and Fixing

The comprehensive benchmarking script can detect and fix common implementation issues:

- Mocked implementations in test files (particularly for OpenVINO)
- Missing hardware-specific code paths
- Compatibility issues between model architectures and hardware platforms

When issues are detected, the script attempts to add the necessary implementation code to provide full hardware compatibility. The fixes are reported in the final benchmark report.

## Hardware Compatibility Matrix

The benchmark results include a hardware compatibility matrix that shows the compatibility level for each model family across different hardware platforms:

- ✅ High: Fully compatible with excellent performance
- ✅ Medium: Compatible with good performance
- ⚠️ Low: Compatible but with performance limitations
- ❌ N/A: Not compatible or not available

This matrix helps guide hardware selection for different model types.

## Results Interpretation

Benchmark results include:

- Functionality verification results for each model and hardware platform
- Performance metrics (latency, throughput, memory usage)
- Batch size scaling behavior
- Hardware compatibility matrix
- Implementation issues and fixes
- Recommendations for hardware and model selection

### Automated Hardware Selection and Performance Prediction

To leverage benchmark results for automated hardware selection and performance prediction, use the new hardware model predictor:

```bash
# Get hardware recommendation based on benchmark data
python test/hardware_model_predictor.py --model bert-base-uncased --batch-size 8

# Predict performance for a model on specific hardware
python test/hardware_model_predictor.py --model t5-small --hardware cuda --batch-size 16

# Generate matrix of predictions for multiple models and hardware
python test/hardware_model_predictor.py --generate-matrix --output-file matrix.json
```

The hardware model predictor uses machine learning trained on benchmark data to provide:
- Optimal hardware recommendations for any model
- Performance predictions (throughput, latency, memory usage)
- Batch size scaling predictions
- Hardware comparison visualizations

For more details, see the [Hardware Model Predictor Guide](HARDWARE_MODEL_PREDICTOR_GUIDE.md).

## Advanced Usage

### Custom Model Sets

To benchmark custom models, create a JSON file with model definitions:

```json
{
  "bert_tiny": {
    "name": "prajjwal1/bert-tiny",
    "family": "embedding",
    "size": "tiny",
    "modality": "text"
  },
  "t5_small": {
    "name": "t5-small",
    "family": "text_generation",
    "size": "small",
    "modality": "text"
  }
}
```

Then run:

```bash
python test/run_model_benchmarks.py --models-set custom --custom-models path/to/models.json
```

### Performance Profiling

For detailed performance profiling:

```bash
python test/run_benchmark_suite.py --profile --output-dir ./profiling_results
```

### Result Visualization

Generate visualizations from benchmark results:

```bash
python test/benchmark_visualizer.py --input-dir ./benchmark_results --output-dir ./visualizations
```

## Integrating with Database Storage

To automatically store benchmark results in the database system:

```bash
python test/benchmark_all_key_models.py --output-dir ./benchmark_results --auto-store-db
```

This integrates with the DuckDB/Parquet-based database system for efficient storage and analysis of benchmark results.

## Troubleshooting

Common issues and solutions:

- **Out of memory errors**: Reduce batch size or use a smaller model variant
- **Hardware detection failures**: Ensure hardware drivers are properly installed
- **Model loading errors**: Check model compatibility with the specified hardware platform
- **Implementation issues**: Use the `--debug` flag to get more information about implementation issues and fixes