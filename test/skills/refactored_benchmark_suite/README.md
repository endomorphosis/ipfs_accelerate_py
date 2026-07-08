# Refactored HuggingFace Model Benchmark Suite

A comprehensive benchmarking framework for HuggingFace models with hardware-aware metrics, enhanced features, extensibility, and reporting capabilities.

## Key Improvements in the Refactored Suite

- **Modular Architecture**: Clean separation of concerns for better maintainability
- **Extended Hardware Support**: Support for CPU, CUDA, MPS, ROCm, OpenVINO, WebNN, and WebGPU 
- **Hardware-Aware Metrics**: Power efficiency and memory bandwidth metrics across platforms
- **Comprehensive Model Support**: Enhanced adapters for text (LLMs), vision (DETR, SAM, DINOv2), speech (Whisper, Wav2Vec2), and multimodal (LLaVA, BLIP2, ImageBind) architectures
- **Hardware Optimizations**: Support for Flash Attention, torch.compile, 4-bit and 8-bit quantization
- **Enhanced Metrics**: Detailed performance metrics including memory tracking, FLOPs estimation, and more
- **Roofline Performance Model**: Visualizations to identify memory vs. compute bottlenecks
- **Flexible Input Configurations**: Support for variable sequence lengths, batch sizes, and data types
- **Visualization Tools**: Interactive performance dashboards and comparison charts
- **Export Capabilities**: Export to JSON, CSV, Markdown, and Hugging Face Model Cards
- **Parallel Benchmarking**: Run benchmarks in parallel across devices and models
- **Test Suites**: Pre-defined benchmark test suites for common use cases
- **CI/CD Integration**: Streamlined integration with GitHub Actions and other CI/CD systems

## Directory Structure

```
refactored_benchmark_suite/
├── __init__.py
├── benchmark.py              # Main benchmark orchestration module
├── metrics/                  # Performance metrics collection modules
│   ├── __init__.py
│   ├── memory.py             # Memory usage tracking
│   ├── timing.py             # Detailed timing metrics
│   ├── flops.py              # FLOPs calculation
│   ├── power.py              # Power efficiency metrics
│   ├── bandwidth.py          # Memory bandwidth metrics
│   └── base.py               # Base metrics classes
├── exporters/                # Result export modules
│   ├── __init__.py
│   ├── json_exporter.py      # JSON export functionality
│   ├── csv_exporter.py       # CSV export functionality
│   ├── markdown_exporter.py  # Markdown report generation
│   └── hf_hub_exporter.py    # HuggingFace Hub card publishing
├── visualizers/              # Visualization tools
│   ├── __init__.py
│   ├── dashboard.py          # Interactive dashboard
│   └── plots.py              # Performance charts and roofline model generation
├── models/                   # Model-specific adapters
│   ├── __init__.py
│   ├── text_models.py        # Text model benchmarking adapters
│   ├── vision_models.py      # Vision model benchmarking adapters
│   ├── speech_models.py      # Speech model benchmarking adapters
│   └── multimodal_models.py  # Multimodal model benchmarking adapters
├── hardware/                 # Hardware-specific implementations
│   ├── __init__.py
│   ├── cpu.py                # CPU-specific benchmarking
│   ├── cuda.py               # CUDA-specific benchmarking
│   ├── mps.py                # MPS-specific benchmarking
│   ├── rocm.py               # ROCm-specific benchmarking
│   ├── openvino.py           # OpenVINO-specific benchmarking
│   ├── webnn.py              # WebNN-specific benchmarking
│   └── webgpu.py             # WebGPU-specific benchmarking
├── data/                     # Sample inputs for benchmarking
│   ├── __init__.py
│   ├── text_samples.py       # Sample text inputs
│   ├── image_samples.py      # Sample image inputs
│   └── audio_samples.py      # Sample audio inputs
├── config/                   # Configuration management
│   ├── __init__.py
│   ├── benchmark_config.py   # Benchmark configuration system
│   └── default_configs/      # Default configurations for models
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── logging.py            # Logging setup
│   └── profiling.py          # Profiling utilities
└── ci/                       # CI/CD integration
    ├── github_actions.py     # GitHub Actions utilities
    └── benchmarks_workflow.yml  # Sample GitHub workflow
```

## Installation

```bash
# Install from the repository
cd /path/to/ipfs_accelerate_py
pip install -e .

# Install optional dependencies
pip install -e ".[visualization]"  # For visualization tools
pip install -e ".[export]"         # For all exporters
pip install -e ".[all]"            # Install all dependencies
```

## Basic Usage

Run a benchmark on a specific model:

```python
from refactored_benchmark_suite import ModelBenchmark

# Simple benchmark
benchmark = ModelBenchmark(model_id="bert-base-uncased")
results = benchmark.run()
results.export_to_json("benchmark_results.json")

# More advanced configuration
benchmark = ModelBenchmark(
    model_id="bert-base-uncased",
    batch_sizes=[1, 2, 4, 8],
    sequence_lengths=[16, 32, 64, 128],
    hardware=["cpu", "cuda"],
    metrics=["latency", "throughput", "memory", "flops", "power", "bandwidth"],
    warmup_iterations=5,
    test_iterations=20
)

results = benchmark.run()

# Export in different formats
results.export_to_json("benchmark_results.json")
results.export_to_csv("benchmark_results.csv")
results.export_to_markdown("benchmark_results.md")

# Visualize results
results.plot_latency_comparison()
results.plot_memory_usage()
results.plot_throughput_scaling()
results.plot_power_efficiency()     # Hardware-aware power metrics
results.plot_bandwidth_utilization() # Memory bandwidth and roofline model
```

## Command-line Usage

```bash
# Run benchmarks on specific models
python -m refactored_benchmark_suite --model bert-base-uncased gpt2 t5-small

# Run benchmarks with specific configuration
python -m refactored_benchmark_suite --model bert-base-uncased --batch-sizes 1 2 4 8 --hardware cpu cuda --export-formats json markdown

# Run a pre-defined benchmark suite
python -m refactored_benchmark_suite --suite text-classification --hardware cuda

# Generate interactive dashboard from results
python -m refactored_benchmark_suite.visualizers.dashboard --results-dir ./benchmark_results
```

## Benchmark Configuration

You can customize benchmarks through a configuration file:

```yaml
# benchmark_config.yaml
models:
  - id: bert-base-uncased
    task: fill-mask
    batch_sizes: [1, 2, 4, 8]
    sequence_lengths: [16, 32, 64, 128]
  
  - id: gpt2
    task: text-generation
    batch_sizes: [1, 2, 4]
    sequence_lengths: [32, 64, 128]

hardware:
  - cpu
  - cuda
  - mps

metrics:
  - latency
  - throughput
  - memory
  - flops
  - power       # Hardware power efficiency metrics
  - bandwidth   # Memory bandwidth metrics

export:
  formats:
    - json
    - csv
    - markdown
  publish_to_hub: true
  hub_token: ${HF_TOKEN}

visualization:
  generate_dashboard: true
  dashboard_title: "Model Performance Dashboard"
  plot_formats: ["latency", "throughput", "scaling", "power", "bandwidth", "roofline"]
  hardware_metrics: true  # Enable hardware-aware metrics visualization
```

## Integration with GitHub Actions

```yaml
# .github/workflows/benchmark.yml
name: Model Benchmarks

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e ".[all]"
      
      - name: Run benchmarks
        run: |
          python -m refactored_benchmark_suite --config benchmark_config.yaml
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
      
      - name: Archive results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: ./benchmark_results/
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## Hardware-Aware Metrics Documentation

For detailed documentation on using and interpreting the hardware-aware metrics, see:

- [Hardware Metrics Guide](docs/hardware_metrics_guide.md) - Complete guide to power efficiency and bandwidth metrics
- [Hardware Abstraction Layer](docs/hardware_abstraction.md) - Documentation on hardware platform support

## Hardware-Aware Benchmarking Examples

### Text Models with Quantization

```python
# Run a comprehensive hardware-aware benchmark for large language models
from refactored_benchmark_suite.examples import hardware_aware_benchmark

# Auto-detect hardware and run LLM benchmark with quantization
results = hardware_aware_benchmark.run_hardware_aware_benchmark(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="llm_benchmark_results",
    use_4bit=True,  # Enable 4-bit quantization
    use_flash_attention=True  # Enable Flash Attention
)

# Print hardware efficiency insights
hardware_aware_benchmark.print_hardware_efficiency_insights(results)
```

### Vision Models with Modern Architectures

```python
# Run hardware-aware benchmark for vision models
from refactored_benchmark_suite.examples import vision_hardware_aware_benchmark

# Benchmark a specific vision model with hardware optimizations
vision_hardware_aware_benchmark.run_vision_hardware_aware_benchmark(
    model_id="facebook/detr-resnet-50",  # DETR model
    output_dir="vision_benchmark_results",
    use_torch_compile=True  # Enable torch.compile optimization
)

# Compare multiple vision model families
vision_hardware_aware_benchmark.run_vision_model_family_comparison(
    output_dir="vision_comparison_results",
    hardware=["cpu", "cuda"]
)
```

### Speech Models with Hardware Optimizations

```python
# Run hardware-aware benchmark for speech models
from refactored_benchmark_suite.examples import speech_hardware_aware_benchmark

# Benchmark a specific speech model with hardware metrics
results = speech_hardware_aware_benchmark.benchmark_speech_model(
    model_id="openai/whisper-small",
    use_power_metrics=True,
    use_bandwidth_metrics=True
)

# Compare different speech model architectures
speech_hardware_aware_benchmark.compare_speech_architectures(
    models=["openai/whisper-base", "facebook/wav2vec2-base", "facebook/hubert-base"],
    output_dir="speech_comparison"
)
```

### Multimodal Models with Hardware-Aware Metrics

```python
# Run hardware-aware benchmark for multimodal models
from refactored_benchmark_suite.examples import multimodal_hardware_aware_benchmark

# Benchmark multiple modern multimodal models with hardware optimizations
results = multimodal_hardware_aware_benchmark.benchmark_modern_multimodal_models(
    use_power_metrics=True,
    use_bandwidth_metrics=True,
    model_size="base",  # Choose from tiny, small, base, large
    output_dir="multimodal_benchmark_results"
)

# Or benchmark a specific multimodal model with custom configuration
single_results = multimodal_hardware_aware_benchmark.benchmark_specific_multimodal_model(
    model_id="llava-hf/llava-1.5-7b-hf",
    use_hardware_metrics=True
)

# Run benchmarks for a specific multimodal task
task_results = multimodal_hardware_aware_benchmark.benchmark_specific_multimodal_task(
    task="visual-question-answering",
    model_id="dandelin/vilt-b32-finetuned-vqa"
)
```

Or use the command-line interface:

```bash
# Run hardware-aware benchmark with all metrics for text models
python -m refactored_benchmark_suite.examples.hardware_aware_benchmark --model gpt2 --output-dir results

# Run hardware-aware benchmark for vision models
python -m refactored_benchmark_suite.examples.vision_hardware_aware_benchmark --model facebook/detr-resnet-50 --flash-attention --torch-compile

# Run hardware-aware benchmark for speech models
python -m refactored_benchmark_suite.examples.speech_hardware_aware_benchmark --model openai/whisper-small --power --bandwidth

# Run hardware-aware benchmark for multimodal models
python -m refactored_benchmark_suite.examples.multimodal_hardware_aware_benchmark --mode multi --size base --output-dir multimodal_results

# Run benchmark for a specific multimodal model
python -m refactored_benchmark_suite.examples.multimodal_hardware_aware_benchmark --mode single --model llava-hf/llava-1.5-7b-hf
```

## License

This project is licensed under the Apache 2.0 License.