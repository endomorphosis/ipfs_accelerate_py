# Training Benchmarking Guide

## Overview

The Training Benchmarking System is a comprehensive framework for measuring, analyzing, and optimizing the training performance of models across different hardware platforms. Part of the IPFS Accelerate Python Framework's Phase 16 implementation, this system extends the benchmarking capabilities from inference to training, providing insights into model training efficiency and hardware utilization.

## Key Features

- **Comprehensive Training Metrics**: Measure training time, memory usage, throughput, and loss convergence
- **Multi-Hardware Support**: Test training performance across CPU, CUDA, ROCm, MPS, and other hardware platforms
- **Training Parameter Testing**: Benchmark different batch sizes, learning rates, and optimizers
- **Gradient Accumulation**: Test the impact of gradient accumulation on training performance
- **Mixed Precision Support**: Compare mixed precision vs. full precision training performance
- **Distributed Training**: Benchmarking for distributed training configurations
- **Memory Analysis**: Track memory usage patterns during model training
- **Database Integration**: Store and retrieve training benchmark results for analysis
- **Visualization Tools**: Generate visual reports of training performance metrics

## Prerequisites

Before running training benchmarks, ensure you have:

1. **PyTorch**: Version 1.8+ with appropriate hardware support (CUDA, ROCm, MPS)
2. **Python Dependencies**:
   ```
   pip install numpy pandas matplotlib scikit-learn
   ```
3. **Hardware Access**: Access to the hardware platforms you wish to benchmark
4. **Models**: Hugging Face Transformers models for testing (will be downloaded automatically)
5. **Benchmark Database**: Set up the benchmark database directory structure

## Getting Started

### Basic Usage

To run basic training benchmarks:

```bash
python training_benchmark_runner.py
```

This will:
1. Detect available hardware platforms
2. Run training benchmarks for the default model set
3. Store results in the benchmark database
4. Generate a performance summary

### Customizing Benchmarks

To customize your training benchmarks:

```bash
python training_benchmark_runner.py --model-families embedding text_generation --hardware cpu cuda --batch-sizes 1 4 16 --learning-rates 1e-5 5e-5
```

Key parameters:
- `--model-families`: Model families to benchmark (embedding, text_generation, vision, audio, multimodal)
- `--hardware`: Hardware types to test on (cpu, cuda, rocm, mps)
- `--batch-sizes`: Batch sizes to test
- `--learning-rates`: Learning rates to test
- `--mixed-precision`: Enable mixed precision training
- `--profiling`: Enable detailed profiling

### Configuration File

You can customize benchmarks using a configuration file:

```bash
python training_benchmark_runner.py --config training_config.json
```

Example configuration file:

```json
{
  "batch_sizes": [1, 4, 16, 32, 64],
  "sequence_lengths": [32, 128, 256],
  "warmup_iterations": 5,
  "training_iterations": 50,
  "learning_rates": [1e-5, 5e-5, 1e-4],
  "optimizers": ["adam", "adamw"],
  "model_families": {
    "embedding": ["bert-base-uncased", "distilbert-base-uncased"],
    "text_generation": ["gpt2", "facebook/opt-125m"]
  },
  "hardware_types": ["cpu", "cuda", "mps"],
  "mixed_precision": true,
  "gradient_accumulation_steps": [1, 2, 4, 8],
  "distributed_configs": [
    {"world_size": 1, "backend": "nccl"}
  ]
}
```

## Training Benchmark Components

### Benchmark Structure

Each training benchmark measures:

1. **Model Loading Time**: Time to load and initialize the model
2. **Training Time**: Per-iteration training time
3. **Memory Usage**: Memory consumption during training
4. **Throughput**: Samples processed per second
5. **Loss Convergence**: Rate of loss reduction over iterations

### Training Configuration Parameters

The benchmarks test various training configurations:

- **Batch Sizes**: Impact of different batch sizes on training efficiency
- **Learning Rates**: Effect of learning rates on convergence and performance
- **Optimizers**: Comparison of different optimizers (Adam, AdamW, SGD)
- **Gradient Accumulation**: Performance with different gradient accumulation steps
- **Mixed Precision**: Comparison of mixed precision vs. full precision training
- **Sequence Lengths**: Impact of different sequence lengths on training performance

### Hardware-Specific Optimizations

The benchmark runner automatically applies hardware-specific optimizations:

- **CUDA**: Utilizes CUDA-specific features and mixed precision
- **ROCm**: Applies ROCm-specific configurations 
- **MPS**: Optimizes for Apple Silicon
- **CPU**: Applies CPU-specific optimizations

## Benchmark Results Format

Training benchmark results are stored in JSON format:

```json
{
  "timestamp": "2025-03-02T12:34:56.789Z",
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
          "training_results": {
            "batch_1_seq_32_lr_1e-05_opt_adam_accum_1": {
              "status": "completed",
              "batch_size": 1,
              "gradient_accumulation_steps": 1,
              "effective_batch_size": 1,
              "training_time": {
                "min": 0.0234,
                "max": 0.0287,
                "mean": 0.0256,
                "std": 0.0015
              },
              "loss": {
                "min": 2.3456,
                "max": 1.2345,
                "mean": 1.7890,
                "std": 0.3456
              },
              "memory_usage": {
                "mean": 1234.5,
                "max": 1345.6
              },
              "throughput": 39.06,
              "loss_convergence_rate": -0.0234,
              "mixed_precision": true,
              "distributed": false
            }
          },
          "performance_summary": {
            "training_time": {
              "min": 0.0234,
              "max": 0.1234,
              "mean": 0.0567,
              "by_batch_size": {
                "1": 0.0256,
                "4": 0.0345,
                "16": 0.0567
              },
              "by_optimizer": {
                "adam": 0.0456,
                "adamw": 0.0678
              },
              "by_mixed_precision": {
                "mixed_precision": 0.0456,
                "full_precision": 0.0789
              }
            },
            "throughput": {
              "min": 39.06,
              "max": 234.56,
              "mean": 123.45
            },
            "loss_convergence": {
              "min": -0.0345,
              "max": -0.0123,
              "mean": -0.0234,
              "by_learning_rate": {
                "1e-05": -0.0123,
                "5e-05": -0.0234,
                "0.0001": -0.0345
              }
            }
          }
        }
      }
    }
  }
}
```

## Performance Summary Analysis

The training benchmark system provides detailed performance analysis:

### By Batch Size

Analyzes how training performance scales with batch size:
- Training time per sample vs. batch size
- Memory usage vs. batch size
- Throughput scaling with batch size

### By Learning Rate

Examines the impact of learning rate on:
- Loss convergence rate
- Training stability
- Optimal learning rate for each model and hardware combination

### By Optimizer

Compares different optimizers:
- Training performance with different optimizers
- Memory usage patterns
- Convergence characteristics

### Mixed Precision Analysis

Analyzes the benefits of mixed precision training:
- Speedup from mixed precision vs. full precision
- Memory savings with mixed precision
- Impact on convergence and accuracy

## Integration with Hardware Selection

Training benchmark results are integrated with the Hardware Selection System to provide recommendations for training hardware:

1. **Training Performance Data**: Collected in the benchmark database
2. **Performance Analysis**: Analyzed to determine optimal hardware for training
3. **Hardware Recommendations**: Generated for training different model types
4. **Training Parameter Recommendations**: Provides optimal batch size, learning rate, and optimizer suggestions

## Best Practices

### Test Preparation

1. **Start Small**: Begin with a smaller subset of models and configurations
2. **Control Environment**: Minimize background processes and system load
3. **Warm Up Hardware**: Run a few short benchmarks to warm up the hardware
4. **Ensure Sufficient Memory**: Verify you have enough memory for the largest models and batch sizes

### Test Execution

1. **Prioritize Combinations**: Focus on the most important model-hardware-parameter combinations
2. **Monitor Resource Usage**: Watch for memory leaks and thermal throttling
3. **Use Timeout Limits**: Set reasonable timeouts for each benchmark
4. **Batch Similar Tests**: Group similar tests to minimize model loading overhead

### Result Analysis

1. **Compare Relative Performance**: Focus on relative performance across hardware rather than absolute numbers
2. **Look for Bottlenecks**: Identify the limiting factors (compute, memory, I/O)
3. **Analyze Scaling Efficiency**: Examine how performance scales with batch size and model size
4. **Check Loss Convergence**: Ensure that faster training doesn't compromise convergence

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Problem: Training with large batch sizes causes OOM errors
   - Solution: Reduce batch size, enable mixed precision, or use gradient accumulation

2. **Slow Convergence**:
   - Problem: Model training shows poor convergence rate
   - Solution: Adjust learning rate or try a different optimizer

3. **Large Variance in Results**:
   - Problem: High variability in timing measurements
   - Solution: Increase warmup iterations and benchmark iterations, minimize background activity

4. **Hardware Utilization Issues**:
   - Problem: Hardware not fully utilized during training
   - Solution: Adjust batch size, enable profiling to identify bottlenecks

### Debugging Tools

1. **Profiling**: Enable profiling to identify bottlenecks
   ```bash
   python training_benchmark_runner.py --profiling
   ```

2. **Verbose Logging**: Enable debug logging for detailed information
   ```bash
   python training_benchmark_runner.py --debug
   ```

3. **Single Configuration Testing**: Test a specific configuration for debugging
   ```bash
   python training_benchmark_runner.py --model-families embedding --hardware cuda --batch-sizes 16 --learning-rates 5e-5 --optimizers adam
   ```

## Advanced Usage

### Distributed Training Benchmarks

For benchmarking distributed training:

```python
# Configure distributed benchmark
distributed_config = {
    "world_size": 2,
    "backend": "nccl",
    "rank": 0
}

# Run benchmark
result = runner.run_training_benchmark(
    model_family="text_generation",
    model_name="gpt2",
    hardware_type="cuda",
    batch_size=16,
    distributed_config=distributed_config
)
```

### Custom Model Support

To benchmark custom models:

1. Implement a custom model loader function
2. Register your model with the benchmark runner
3. Run benchmarks with your custom model

Example:

```python
def load_custom_model(model_path, device):
    # Custom model loading logic
    return model, tokenizer, optimizer_fn, loss_fn

# Register custom model
runner.register_custom_model("my_custom_model", load_custom_model)

# Run benchmark with custom model
runner.run_training_benchmark(
    model_family="custom",
    model_name="my_custom_model",
    hardware_type="cuda"
)
```

### Combining with Inference Benchmarks

For comprehensive model evaluation, combine training and inference benchmarks:

```bash
# Run training benchmarks
python training_benchmark_runner.py --output-dir ./benchmark_results

# Run inference benchmarks on the same models
python hardware_benchmark_runner.py --output-dir ./benchmark_results

# Generate combined report
python generate_comprehensive_report.py --input-dir ./benchmark_results
```

## Visualization and Reporting

The training benchmark system provides tools for visualizing results:

1. **Training Time Comparison**: Compare training time across hardware platforms
2. **Throughput Scaling**: Visualize throughput scaling with batch size
3. **Memory Usage Analysis**: Analyze memory usage patterns
4. **Loss Convergence Visualization**: Compare loss convergence rates
5. **Hardware Efficiency**: Compare training efficiency across hardware

Example command to generate visualizations:

```bash
python visualize_training_benchmarks.py --results-file benchmark_results/training_benchmark_results_20250302_123456.json --output-dir ./visualizations
```

## Conclusion

The Training Benchmarking System provides a comprehensive framework for evaluating and optimizing model training performance across hardware platforms. By leveraging these benchmarks, you can make informed decisions about hardware selection, training parameters, and optimization strategies for your machine learning workloads.

---

For more information, see related documentation:
- [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md)
- [Hardware Selection Guide](HARDWARE_SELECTION_GUIDE.md)
- [Model Compression Guide](MODEL_COMPRESSION_GUIDE.md)