# Hardware Benchmarking Guide

## Overview

The hardware benchmarking system provides comprehensive performance testing across different hardware platforms for various model families. This system helps you understand model performance characteristics, optimize deployment strategies, and keep the hardware compatibility matrix up to date with real-world performance data.

## Key Components

The benchmarking system consists of these primary components:

1. **Hardware Benchmark Runner** (`hardware_benchmark_runner.py`): Core component that performs benchmarks for specific models on specific hardware
2. **Benchmark Suite Runner** (`run_benchmark_suite.py`): Orchestration tool for running comprehensive benchmark suites across multiple model families
3. **Benchmark Visualization**: Automated generation of performance comparison charts and graphs
4. **Compatibility Matrix Integration**: Automatic updating of hardware compatibility information based on benchmark results

## Hardware Benchmark Runner

The `hardware_benchmark_runner.py` script provides the core benchmarking functionality:

### Features

- **Comprehensive Hardware Support**: Tests across CPU, CUDA, MPS (Apple Silicon), ROCm (AMD), OpenVINO, WebNN, and WebGPU
- **Model Family Coverage**: Supports embedding, text generation, vision, audio, and multimodal models
- **Flexible Configuration**: Configurable batch sizes, sequence lengths, and iteration counts
- **Detailed Metrics**: Measures latency, throughput, memory usage, and hardware-specific statistics
- **ResourcePool Integration**: Uses the ResourcePool for efficient model loading and caching
- **Graceful Degradation**: Works even when some framework components are missing
- **Resilient Error Handling**: Continues benchmarking even if some tests fail
- **Comprehensive Reporting**: Generates detailed JSON results and Markdown reports

### Usage

```bash
# Basic usage with default settings
python hardware_benchmark_runner.py

# Benchmark specific model families on specific hardware
python hardware_benchmark_runner.py --model-families embedding text_generation --hardware cuda cpu

# Customize batch sizes and iterations
python hardware_benchmark_runner.py --batch-sizes 1 4 16 --warmup 10 --iterations 50

# Run benchmarks in parallel across hardware types
python hardware_benchmark_runner.py --parallel

# Include web platforms in benchmarks (WebNN/WebGPU)
python hardware_benchmark_runner.py --include-web-platforms

# Set custom output directory
python hardware_benchmark_runner.py --output-dir ./my_benchmark_results
```

### Key Parameters

- `--model-families`: List of model families to benchmark (embedding, text_generation, vision, audio, multimodal)
- `--hardware`: List of hardware types to benchmark (cpu, cuda, mps, openvino, etc.)
- `--batch-sizes`: List of batch sizes to test
- `--warmup`: Number of warmup iterations before timing
- `--iterations`: Number of benchmark iterations for timing
- `--timeout`: Maximum time in seconds for a benchmark
- `--parallel`: Run benchmarks in parallel across hardware types
- `--use-resource-pool`: Use ResourcePool for model caching (default: True)
- `--include-web-platforms`: Include WebNN and WebGPU in benchmarks
- `--debug`: Enable debug logging

## Benchmark Suite Runner

The `run_benchmark_suite.py` script provides a higher-level orchestration tool for running comprehensive benchmark suites:

### Features

- **Configuration-Driven**: Uses JSON configuration files for defining benchmark suites
- **Result Aggregation**: Collects and aggregates results across multiple benchmark runs
- **Visualization**: Generates plots and charts showing performance comparisons
- **Scheduling**: Can be configured to run benchmarks on a schedule
- **Comprehensive Reporting**: Generates both detailed and summary reports
- **Compatibility Matrix Updates**: Automatically updates the hardware compatibility matrix

### Usage

```bash
# Run benchmarks with default configuration
python run_benchmark_suite.py

# Specify a custom configuration file
python run_benchmark_suite.py --config my_benchmark_config.json

# Create a default configuration file
python run_benchmark_suite.py --create-config

# Only test specific model families
python run_benchmark_suite.py --models embedding vision

# Only test specific hardware types
python run_benchmark_suite.py --hardware cuda cpu

# Check prerequisites without running benchmarks
python run_benchmark_suite.py --check

# Generate plots from existing results
python run_benchmark_suite.py --plot-only ./benchmark_results/20250302_123456

# Update compatibility matrix from existing results
python run_benchmark_suite.py --update-matrix-only ./benchmark_results/20250302_123456
```

### Configuration File

The benchmark suite is configured using a JSON file. Here's an example configuration:

```json
{
  "batch_sizes": [1, 4, 8],
  "sequence_lengths": [32, 128, 512],
  "warmup_iterations": 5,
  "benchmark_iterations": 20,
  "timeout": 600,
  "model_families": {
    "embedding": ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
    "text_generation": ["gpt2", "t5-small", "google/flan-t5-small"],
    "vision": ["google/vit-base-patch16-224", "microsoft/resnet-50"],
    "audio": ["openai/whisper-tiny", "facebook/wav2vec2-base"],
    "multimodal": ["openai/clip-vit-base-patch32"]
  },
  "hardware_types": ["cpu", "cuda", "mps", "openvino"],
  "include_web_platforms": false,
  "parallel": true,
  "use_resource_pool": true,
  "generate_plots": true,
  "update_compatibility_matrix": true,
  "schedule": {
    "enabled": false,
    "frequency": "daily",
    "time": "02:00"
  }
}
```

## Benchmark Reports

The benchmark system generates several types of reports:

### JSON Results

Detailed benchmark results are saved in JSON format with comprehensive metrics:

```json
{
  "timestamp": "20250302_123456",
  "system_info": {
    "platform": "Linux-5.15.0-x86_64-with-glibc2.31",
    "processor": "Intel(R) Core(TM) i9-11900K @ 3.50GHz",
    "python_version": "3.8.10",
    "cuda_available": true,
    "cuda_version": "11.8"
  },
  "benchmarks": {
    "embedding": {
      "bert-base-uncased": {
        "cuda": {
          "status": "completed",
          "model_load_time": 2.34,
          "benchmark_results": {
            "batch_1_seq_32": {
              "status": "completed",
              "avg_latency": 0.0023,
              "throughput": 434.78
            }
          },
          "performance_summary": {
            "latency": {
              "min": 0.0023,
              "max": 0.0087,
              "mean": 0.0045
            },
            "throughput": {
              "min": 114.94,
              "max": 434.78,
              "mean": 222.22
            }
          }
        }
      }
    }
  }
}
```

### Markdown Reports

Human-readable benchmark reports are generated in Markdown format:

```markdown
# Hardware Benchmark Report - 20250302_123456

## System Information
- **Platform**: Linux-5.15.0-x86_64-with-glibc2.31
- **Processor**: Intel(R) Core(TM) i9-11900K @ 3.50GHz
- **Python Version**: 3.8.10
- **CUDA Version**: 11.8
- **GPU Count**: 1
- **GPUs**:
  - GPU 0: NVIDIA GeForce RTX 3090, 24.00 GB, Compute Capability 8.6

## Available Hardware
- **cpu**: ✅ Available
- **cuda**: ✅ Available
- **mps**: ❌ Not available
- **openvino**: ✅ Available

## Benchmark Results

### Embedding Models

#### Latency Comparison (ms)
| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 12.45 | 2.30 | 5.67 |
| distilbert-base-uncased | 6.78 | 1.45 | 3.21 |
| roberta-base | 13.56 | 2.42 | 5.89 |

#### Throughput Comparison (items/sec)
| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 80.32 | 434.78 | 176.37 |
| distilbert-base-uncased | 147.49 | 689.66 | 311.53 |
| roberta-base | 73.75 | 413.22 | 169.78 |

#### Relative Performance (vs CPU)
| Hardware | Speedup Factor |
|---|---|
| CUDA | 5.41x |
| OPENVINO | 2.20x |
```

### Visualization Plots

The benchmark system also generates various plots for visual analysis:

1. **Hardware Comparison**: Bar charts comparing latency and throughput across hardware platforms
2. **Model Performance**: Bar charts comparing performance across models within each family
3. **Batch Size Scaling**: Line charts showing how performance scales with batch size
4. **Relative Speedup**: Bar charts showing speedup factors relative to CPU
5. **Memory Usage**: Bar charts comparing memory usage across hardware platforms

## Hardware Compatibility Matrix

The benchmark system automatically updates the hardware compatibility matrix with empirical measurements:

```json
{
  "timestamp": "2025-03-02T12:34:56.789Z",
  "hardware_types": ["cpu", "cuda", "mps", "openvino", "webnn", "webgpu"],
  "model_families": {
    "embedding": {
      "hardware_compatibility": {
        "cpu": {
          "compatible": true,
          "performance_rating": "medium",
          "benchmark_results": [
            {
              "timestamp": "2025-03-02T12:34:56.789Z",
              "model_name": "bert-base-uncased",
              "mean_latency": 0.01245,
              "mean_throughput": 80.32
            }
          ]
        },
        "cuda": {
          "compatible": true,
          "performance_rating": "high",
          "benchmark_results": [
            {
              "timestamp": "2025-03-02T12:34:56.789Z",
              "model_name": "bert-base-uncased",
              "mean_latency": 0.0023,
              "mean_throughput": 434.78
            }
          ]
        }
      }
    }
  }
}
```

## Integration with Resource Pool

The benchmark system integrates with the ResourcePool to efficiently share models across benchmarks:

```python
# Using ResourcePool for efficient model loading
pool = get_global_resource_pool()

# Define hardware preferences
hardware_preferences = {"device": hardware_type}

# Load model using ResourcePool
model = pool.get_model(
    model_type=family,
    model_name=model_name,
    constructor=create_model,
    hardware_preferences=hardware_preferences
)
```

## Integration with Hardware Detection

The benchmark system uses hardware detection for accurate hardware capability identification:

```python
# Use comprehensive hardware detection
hardware_info = detect_hardware_with_comprehensive_checks()

# Extract available hardware platforms
for hw_type in [CPU, CUDA, MPS, ROCM, OPENVINO, WEBNN, WEBGPU]:
    self.available_hardware[hw_type] = hardware_info.get(hw_type, False)
```

## Integration with Model Family Classification

The benchmark system uses model family classification for optimized benchmarking:

```python
# Classify model to understand its characteristics
model_info = classify_model(model_name=model_name)
model_family = model_info.get("family")

# Use model family information for benchmark configuration
if model_family == "embedding":
    # Configure benchmark for embedding models
    ...
elif model_family == "text_generation":
    # Configure benchmark for text generation models
    ...
```

## Best Practices

### Performance Testing

1. **Warm-up Adequately**: Always use sufficient warm-up iterations (at least 5-10) before timing
2. **Consistent Environment**: Run benchmarks in a controlled environment with minimal background processes
3. **Multiple Iterations**: Use at least 20 benchmark iterations for reliable statistics
4. **Various Batch Sizes**: Test with multiple batch sizes to understand scaling behavior
5. **Test All Hardware**: Include all available hardware platforms for comprehensive comparison
6. **Monitor Temperature**: Be aware of thermal throttling during extended benchmarks
7. **Clear Cache Between Tests**: Use ResourcePool's cleanup functionality between tests

### Benchmark Suite Configuration

1. **Start Small**: Begin with a small subset of models for initial testing
2. **Incremental Addition**: Add more models incrementally to identify performance outliers
3. **Regular Scheduling**: Set up regular benchmark runs to track performance over time
4. **Mix Model Sizes**: Include both small and large models to test different resource requirements
5. **Timeout Settings**: Use appropriate timeouts for different model sizes

### Result Analysis

1. **Compare Similar Models**: Group models by size and type for fair comparisons
2. **Consider Throughput**: For batch processing, throughput may be more important than latency
3. **Memory Analysis**: Pay attention to memory usage for resource-constrained environments
4. **Speedup Ratios**: Focus on relative performance rather than absolute numbers
5. **Performance Consistency**: Look at standard deviation of latency, not just averages

## Continuous Benchmarking

For ongoing performance tracking, you can set up continuous benchmarking:

```bash
# Create a cron job for daily benchmarking
echo "0 2 * * * $(which python) $(pwd)/run_benchmark_suite.py" | crontab -

# Or use a systemd timer
cat << EOF > /etc/systemd/system/benchmark.service
[Unit]
Description=Run Model Benchmarks

[Service]
Type=oneshot
ExecStart=$(which python) $(pwd)/run_benchmark_suite.py
User=$(whoami)
WorkingDirectory=$(pwd)

[Install]
WantedBy=multi-user.target
EOF

cat << EOF > /etc/systemd/system/benchmark.timer
[Unit]
Description=Run model benchmarks daily

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl enable benchmark.timer
sudo systemctl start benchmark.timer
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or test smaller models
2. **Slow Loading**: Enable ResourcePool caching to speed up model loading
3. **Missing Results**: Check for timeouts or failures in the benchmark logs
4. **Inconsistent Results**: Increase iteration count for more reliable statistics
5. **Plot Generation Errors**: Ensure matplotlib and pandas are installed

### Debugging

```bash
# Enable debug logging
python hardware_benchmark_runner.py --debug

# Run only a single model family for testing
python hardware_benchmark_runner.py --model-families embedding

# Check system prerequisites
python run_benchmark_suite.py --check
```

## Advanced Usage

### Custom Model Benchmarking

For benchmarking custom models not included in the standard lists:

```bash
# Create a custom configuration file
cat << EOF > custom_benchmark_config.json
{
  "batch_sizes": [1, 2, 4],
  "sequence_lengths": [32, 64],
  "warmup_iterations": 5,
  "benchmark_iterations": 20,
  "model_families": {
    "embedding": ["path/to/my/custom/model"]
  },
  "hardware_types": ["cpu", "cuda"]
}
EOF

# Run benchmark with custom configuration
python run_benchmark_suite.py --config custom_benchmark_config.json
```

### Benchmark Result Aggregation

To aggregate results from multiple benchmark runs:

```bash
# Aggregate results from multiple directories
python run_benchmark_suite.py --aggregate-results ./benchmark_results/run1 ./benchmark_results/run2

# Generate comparative plots
python run_benchmark_suite.py --plot-comparison ./benchmark_results/before ./benchmark_results/after
```

## Further Resources

- [ResourcePool Guide](RESOURCE_POOL_GUIDE.md): Details on the ResourcePool for model caching
- [Hardware Detection Guide](HARDWARE_DETECTION_GUIDE.md): Information on the hardware detection system
- [Model Family Classifier Guide](MODEL_FAMILY_CLASSIFIER_GUIDE.md): Details on model family classification
- [Model Compression Guide](MODEL_COMPRESSION_GUIDE.md): Guide to model compression techniques
- [Hardware Platform Test Guide](HARDWARE_PLATFORM_TEST_GUIDE.md): Guide to hardware platform testing
- [Hardware Model Validation Guide](HARDWARE_MODEL_VALIDATION_GUIDE.md): Guide to model validation
- [Hardware Model Integration Guide](HARDWARE_MODEL_INTEGRATION_GUIDE.md): Guide to hardware-model integration

---

*This guide is part of the IPFS Accelerate Python Framework documentation*