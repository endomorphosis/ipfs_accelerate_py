# Hardware-Aware Metrics Guide

This guide explains how to use the hardware-aware metrics in our benchmark suite to analyze model performance across different hardware platforms.

## Overview

The hardware-aware benchmarking suite extends standard performance metrics (latency, throughput) with additional metrics that measure hardware efficiency:

1. **Power Efficiency Metrics**: Measure power consumption and calculate efficiency metrics like GFLOPs/watt.
2. **Memory Bandwidth Metrics**: Measure memory bandwidth utilization and provide roofline model analysis.

These metrics help you understand not just how fast a model runs, but how efficiently it uses the available hardware resources.

## Available Hardware Metrics

### Power Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `power_avg_watts` | Average power consumption | Watts |
| `power_max_watts` | Maximum power consumption | Watts |
| `energy_joules` | Total energy consumed | Joules |
| `ops_per_watt` | Operations per watt (raw) | OPs/W |
| `gflops_per_watt` | Billions of operations per watt | GFLOPs/W |
| `throughput_per_watt` | Items processed per watt | items/W |

### Bandwidth Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `avg_bandwidth_gbps` | Average memory bandwidth | GB/s |
| `peak_bandwidth_gbps` | Peak memory bandwidth during inference | GB/s |
| `peak_theoretical_bandwidth_gbps` | Theoretical peak bandwidth of the device | GB/s |
| `bandwidth_utilization_percent` | Percentage of theoretical bandwidth used | % |
| `memory_transfers_gb` | Total memory transfers | GB |
| `arithmetic_intensity_flops_per_byte` | Compute operations per byte accessed | FLOPs/byte |
| `compute_bound` | Whether the workload is compute-bound | Boolean |
| `memory_bound` | Whether the workload is memory-bound | Boolean |

## Using Hardware-Aware Metrics

### Basic Usage

To include hardware-aware metrics in your benchmark, simply add them to the metrics list:

```python
from benchmark import ModelBenchmark

benchmark = ModelBenchmark(
    model_id="google/vit-base-patch16-224",
    hardware=["cpu", "cuda"],  # Test on both CPU and GPU (if available)
    batch_sizes=[1, 2, 4, 8],
    metrics=["latency", "throughput", "memory", "flops", "power", "bandwidth"],
    output_dir="benchmark_results"
)

# Run the benchmark
results = benchmark.run()

# Export results
results.export_to_json()
results.export_to_markdown()

# Visualize hardware-aware metrics
results.plot_power_efficiency()
results.plot_bandwidth_utilization()
```

### Hardware-Aware Optimizations

Our suite supports model-specific hardware optimizations:

```python
benchmark = ModelBenchmark(
    model_id="google/vit-base-patch16-224",
    hardware=["cuda"],
    metrics=["latency", "throughput", "power", "bandwidth"],
    # Hardware optimization options
    flash_attention=True,  # Use Flash Attention (transformer models)
    torch_compile=True     # Use torch.compile (PyTorch 2.0+)
)
```

### Model-Specific Optimizations

Our adapters automatically detect model types and apply appropriate optimizations:

```python
# For large language models
llm_benchmark = ModelBenchmark(
    model_id="meta-llama/Llama-2-7b-hf",
    metrics=["latency", "throughput", "power", "bandwidth"],
    use_4bit=True  # Use 4-bit quantization for large models
)

# For vision models like DETR or SAM
vision_benchmark = ModelBenchmark(
    model_id="facebook/detr-resnet-50",
    metrics=["latency", "throughput", "power", "bandwidth"],
    flash_attention=True  # Use Flash Attention for transformer components
)

# For speech models like Whisper or Wav2Vec2
speech_benchmark = ModelBenchmark(
    model_id="openai/whisper-small",
    metrics=["latency", "throughput", "power", "bandwidth"],
    use_torch_compile=True  # Use torch.compile for optimized audio processing
)

# For multimodal models like LLaVA or BLIP2
multimodal_benchmark = ModelBenchmark(
    model_id="llava-hf/llava-1.5-7b-hf",
    metrics=["latency", "throughput", "power", "bandwidth"],
    flash_attention=True,  # Use Flash Attention for transformer components
    use_torch_compile=True  # Use torch.compile for optimized processing
)
```

## Interpreting Hardware-Aware Metrics

### Power Efficiency

- **Higher GFLOPs/watt** indicates better computational efficiency.
- Compare power efficiency across different:
  - Hardware platforms (CPU vs. GPU)
  - Batch sizes (larger batches often improve efficiency)
  - Model architectures (modern architectures often have better efficiency)

### Memory Bandwidth

- **Bandwidth utilization percentage** tells you how effectively you're using the available memory bandwidth.
- **Arithmetic intensity** indicates the computational work performed per byte of memory accessed.
- The **roofline model** helps identify whether your workload is compute-bound or memory-bound:
  - If **compute_bound** is True: Focus on computational optimizations.
  - If **memory_bound** is True: Focus on reducing memory accesses and improving memory access patterns.

## Special Considerations for Model Types

### Text Models (LLMs)
- Large language models often benefit from:
  - KV cache optimizations for generation tasks
  - Attention pattern optimizations for long sequences
  - Quantization (4-bit, 8-bit) for memory efficiency
  - Tensor core utilization for matrix multiplications

### Vision Models
- Vision models may benefit from:
  - Feature map optimizations for CNN architectures
  - Vision-specific memory layouts for improved cache efficiency
  - Specialized convolution implementations (e.g., FFT, Winograd)
  - Quantization for post-convolution activations

### Speech Models
- Speech models may benefit from:
  - Audio processing thread optimizations for real-time performance
  - Audio feature extraction acceleration
  - Sample rate optimizations based on hardware
  - CNN-based architectures for spectrogram processing

### Multimodal Models
- Multimodal models face unique challenges that affect hardware metrics:
  - **Cross-modal fusion**: The fusion of different modality features (text, vision, audio) can create memory bandwidth bottlenecks
  - **Mixed architecture**: Different encoder components may have different optimal hardware configurations
  - **CUDA stream prioritization**: Text and vision components can benefit from different stream priorities
  - **Memory distribution**: Balancing memory usage between modalities is critical for large models
  - **Component-specific optimizations**: Apply Flash Attention to the text component while optimizing CNN operations for the vision component

When benchmarking multimodal models, consider these strategies:
1. Use `apply_multimodal_hardware_optimizations` for architecture-aware optimization
2. Test with different input modality combinations to identify bottlenecks
3. Analyze memory bandwidth for cross-modal operations specifically
4. Use the roofline model to identify which modality component is creating bottlenecks
5. Adjust thread allocation based on modality workload distribution

## Example: Comparing Models Across Hardware

```python
# Choose models to compare
models = [
    "google/vit-base-patch16-224",  # ViT
    "facebook/convnext-tiny-224",   # ConvNeXt
    "facebook/dino-vitb16",         # DINOv2
    "facebook/detr-resnet-50"       # DETR
]

# Hardware platforms to test
hardware = ["cpu", "cuda"]  # Include MPS for Apple Silicon

# Run benchmarks
for model_id in models:
    benchmark = ModelBenchmark(
        model_id=model_id,
        hardware=hardware,
        batch_sizes=[1, 2, 4],
        metrics=["latency", "throughput", "power", "bandwidth"],
        output_dir=f"benchmark_results/{model_id.split('/')[-1]}"
    )
    benchmark.run()
```

## Example: Comparing Multimodal Model Architectures

```python
# Import the specialized example script
from refactored_benchmark_suite.examples import multimodal_hardware_aware_benchmark

# Compare different multimodal architectures with hardware metrics
multimodal_hardware_aware_benchmark.benchmark_modern_multimodal_models(
    use_power_metrics=True,
    use_bandwidth_metrics=True,
    model_size="small",  # Use smaller models for quicker comparison
    output_dir="multimodal_comparison"
)

# Analyze a specific multimodal model in depth
results = multimodal_hardware_aware_benchmark.benchmark_specific_multimodal_model(
    model_id="llava-hf/llava-1.5-7b-hf",
    use_hardware_metrics=True,
    output_dir="llava_hardware_analysis"
)
```

## Platform-Specific Considerations

### NVIDIA GPUs

- Power monitoring uses NVIDIA SMI.
- CUDA-specific optimizations include cuDNN benchmark mode and CUDA graphs.
- Tensor cores are leveraged automatically for supported operations.

### AMD GPUs

- Power and bandwidth monitoring use ROCm SMI.
- ROCm-specific optimizations are applied where available.

### Apple Silicon (MPS)

- Power monitoring uses powermetrics (requires sudo access).
- MPS-specific optimizations focus on Metal Performance Shaders.

### CPUs

- Power monitoring uses Intel RAPL on supported platforms.
- CPU-specific optimizations include:
  - Thread pinning
  - oneDNN/MKL optimizations
  - Cache-friendly memory access patterns

## Advanced Usage: Custom Analysis

You can access the raw metrics for custom analysis:

```python
results = benchmark.run()

# Get all results for further analysis
for result in results.results:
    hardware = result.hardware
    batch_size = result.batch_size
    
    if "power_avg_watts" in result.metrics:
        power = result.metrics["power_avg_watts"]
        gflops_per_watt = result.metrics["gflops_per_watt"]
        print(f"{hardware}, Batch {batch_size}: {gflops_per_watt:.2f} GFLOPs/watt")
    
    if "arithmetic_intensity_flops_per_byte" in result.metrics:
        ai = result.metrics["arithmetic_intensity_flops_per_byte"]
        is_compute_bound = result.metrics["compute_bound"]
        bound_type = "compute-bound" if is_compute_bound else "memory-bound"
        print(f"{hardware}, Batch {batch_size}: {ai:.2f} FLOPs/byte ({bound_type})")
```

## Example Hardware-Specific Reporting

Here's an example of how to interpret the results across different hardware:

```
=== HARDWARE EFFICIENCY INSIGHTS ===

Hardware Platform Comparison:
  - CPU to GPU Speedup: 15.3x

Power Efficiency Insights:
  CPU:
    - Average Power: 85.42 watts
    - Computational Efficiency: 12.78 GFLOPs/watt
    - Best Batch Size for Efficiency: 8
  CUDA:
    - Average Power: 165.23 watts
    - Computational Efficiency: 98.45 GFLOPs/watt
    - Best Batch Size for Efficiency: 4

Memory Bandwidth Insights:
  CPU:
    - Average Bandwidth: 18.75 GB/s
    - Peak Theoretical Bandwidth: 51.20 GB/s
    - Utilization: 36.62%
    - Performance Characteristic: memory-bound
    - Arithmetic Intensity: 1.25 FLOPs/byte
  CUDA:
    - Average Bandwidth: 352.64 GB/s
    - Peak Theoretical Bandwidth: 936.00 GB/s
    - Utilization: 37.67%
    - Performance Characteristic: compute-bound
    - Arithmetic Intensity: 8.92 FLOPs/byte
```

## Common Issues and Solutions

### Power Metrics Not Available

If power metrics show as not supported:

1. For NVIDIA GPUs: Ensure nvidia-smi is installed and accessible
2. For Intel CPUs: Verify RAPL is accessible via /sys/class/powercap/
3. For macOS: powermetrics requires sudo access

### Inaccurate Bandwidth Measurements

If bandwidth metrics seem incorrect:

1. Check that the `memory_transfers_bytes` is being properly estimated
2. Use the `estimate_memory_transfers()` method to provide a better estimate
3. Verify the theoretical peak bandwidth is correctly detected

## Further Reading

1. [Roofline Model for Performance Analysis](https://en.wikipedia.org/wiki/Roofline_model)
2. [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
3. [Optimizing Deep Learning for Apple Silicon](https://developer.apple.com/metal/pytorch/)
4. [Intel Neural Network Optimization Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-training-on-xeon.html)