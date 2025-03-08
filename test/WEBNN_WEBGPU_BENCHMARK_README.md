# Real WebNN/WebGPU Benchmark System

## Overview

This system provides a comprehensive benchmarking framework for WebNN and WebGPU implementations using real browser-based hardware acceleration. It ensures a clear distinction between real hardware acceleration and simulation mode, allowing for accurate performance measurements.

Key features:
- Robust WebSocket bridge for reliable browser communication
- Support for Chrome, Firefox, and Edge browsers
- Browser-specific optimizations (e.g., Firefox for audio models with compute shaders)
- Comprehensive benchmarking across multiple models, batch sizes, and precision levels
- Database integration for result storage
- Detailed reporting in multiple formats (JSON, Markdown, HTML)

## Components

The benchmark system consists of several key components:

1. **WebSocket Bridge** (`fixed_web_platform/websocket_bridge.py`): Provides robust communication between Python and browsers with error handling, automatic reconnection, and message queuing.

2. **Benchmark Runner** (`run_real_webnn_webgpu_benchmarks.py`): The core benchmarking system that interfaces with browsers through the WebSocket bridge.

3. **User-Friendly Wrapper** (`benchmark_real_webnn_webgpu.py`): A convenient command-line interface for running benchmarks with sensible defaults.

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  ```
  pip install selenium websockets duckdb pandas
  ```

- WebDrivers for browsers:
  - Chrome/Chromium: [ChromeDriver](https://sites.google.com/chromium.org/driver/)
  - Firefox: [GeckoDriver](https://github.com/mozilla/geckodriver/releases)
  - Edge: [EdgeDriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

## Usage

The easiest way to run benchmarks is through the `benchmark_real_webnn_webgpu.py` script:

```bash
# Run WebGPU benchmarks with Chrome
python benchmark_real_webnn_webgpu.py --webgpu --chrome

# Run WebNN benchmarks with Edge (best for WebNN)
python benchmark_real_webnn_webgpu.py --webnn --edge

# Run audio model benchmarks with Firefox (best for audio with compute shaders)
python benchmark_real_webnn_webgpu.py --audio --firefox

# Run quantized model benchmarks
python benchmark_real_webnn_webgpu.py --bits 8 --mixed-precision

# Run comprehensive benchmarks across multiple models and configurations
python benchmark_real_webnn_webgpu.py --comprehensive
```

### Command-Line Options

#### Platform Options
- `--webgpu`: Use WebGPU platform (default)
- `--webnn`: Use WebNN platform

#### Browser Options
- `--chrome`: Use Chrome browser (default for most models)
- `--firefox`: Use Firefox browser (better for audio models)
- `--edge`: Use Edge browser (better for WebNN)

#### Model Type Options
- `--text`: Benchmark text models (default: bert-base-uncased)
- `--vision`: Benchmark vision models (default: vit-base-patch16-224)
- `--audio`: Benchmark audio models (default: whisper-tiny)
- `--multimodal`: Benchmark multimodal models (default: clip-vit-base-patch16)

#### Model Options
- `--model MODEL`: Specify a specific model to benchmark

#### Quantization Options
- `--bits {2,4,8,16}`: Bit precision for quantization
- `--mixed-precision`: Use mixed precision quantization

#### Benchmark Options
- `--batch-sizes SIZES`: Comma-separated list of batch sizes (default: "1,2,4,8")
- `--repeats N`: Number of repeats for each configuration (default: 3)
- `--warmup N`: Number of warmup runs (default: 1)
- `--comprehensive`: Run comprehensive benchmarks across multiple models

#### Output Options
- `--output-dir DIR`: Directory for output files (default: "./benchmark_results")
- `--output-format {json,markdown,html}`: Output format (default: markdown)
- `--db-path PATH`: Path to DuckDB database
- `--no-db`: Disable database storage

#### Execution Options
- `--headless`: Run browsers in headless mode
- `--verbose`: Enable verbose output

## Browser-Specific Optimizations

This benchmark system leverages browser-specific optimizations to get the most accurate performance measurements:

- **Firefox**: Best for audio models due to superior compute shader performance. Firefox shows ~20% better performance for audio models like Whisper and Wav2Vec2 compared to Chrome and Edge.

- **Edge**: Best for WebNN. Microsoft Edge has the most mature WebNN implementation and typically shows better performance with WebNN than Chrome or Firefox.

- **Chrome**: Good all-around performance, especially for vision models with WebGPU.

## Simulation vs. Real Hardware

The benchmark system clearly distinguishes between real hardware acceleration and simulation mode:

- **Real Hardware Acceleration**: When the browser successfully uses WebNN or WebGPU acceleration, the benchmark runs in real mode and provides accurate measurements.

- **Simulation Mode**: If WebNN or WebGPU is not available (or fails to initialize), the benchmark automatically falls back to simulation mode, which emulates the performance characteristics but does not provide real hardware acceleration.

The benchmark results clearly indicate whether real hardware or simulation was used, both in the console output and in the generated reports.

## Database Integration

Results can be stored in a DuckDB database for easy querying and analysis. To enable database storage:

```bash
# Specify database path directly
python benchmark_real_webnn_webgpu.py --db-path ./benchmark_db.duckdb

# Or set the environment variable
export BENCHMARK_DB_PATH=./benchmark_db.duckdb
python benchmark_real_webnn_webgpu.py
```

## March 2025 Optimizations

The March 2025 version includes three major optimizations for web platform models:

1. **WebGPU Compute Shader Optimization for Audio Models**:
   - Firefox shows ~20% better performance than Chrome for audio models
   - Computation is optimized with specialized workgroup configurations
   - Particularly effective for Whisper, Wav2Vec2, and CLAP models

2. **Parallel Model Loading for Multimodal Models**:
   - 30-45% loading time reduction for CLIP, LLaVA, and other multimodal models
   - Components are loaded simultaneously instead of sequentially

3. **Shader Precompilation for Faster Startup**:
   - 30-45% faster first inference for all model types
   - Especially beneficial for vision models with complex shader pipelines

## Performance Metrics

The benchmark system collects and reports the following performance metrics:

- **Inference Time**: Time taken to perform inference (ms)
- **Throughput**: Number of items processed per second
- **Memory Usage**: Estimated memory usage (where available)
- **Quantization Impact**: Performance difference with various bit precisions

## Examples

### Basic Benchmarking

```bash
# Benchmark BERT model with WebGPU in Chrome
python benchmark_real_webnn_webgpu.py --text --chrome

# Benchmark ViT model with WebGPU in Chrome
python benchmark_real_webnn_webgpu.py --vision --chrome

# Benchmark Whisper model with WebGPU in Firefox
python benchmark_real_webnn_webgpu.py --audio --firefox
```

### Advanced Benchmarking

```bash
# Benchmark with 8-bit quantization
python benchmark_real_webnn_webgpu.py --text --bits 8

# Benchmark with 4-bit mixed precision
python benchmark_real_webnn_webgpu.py --text --bits 4 --mixed-precision

# Comprehensive benchmark with HTML report
python benchmark_real_webnn_webgpu.py --comprehensive --output-format html
```

## Troubleshooting

1. **Browser crashes or WebSocket connection fails**:
   - Check that you have the latest WebDriver for your browser
   - Ensure your browser supports WebGPU/WebNN (check chrome://gpu or edge://gpu)
   - Try running without headless mode to see browser output

2. **Model initialization fails**:
   - Check if the model is too large for browser memory
   - Try a smaller model or reduce precision with quantization

3. **Results show simulation mode when you expected real hardware**:
   - Check browser GPU acceleration settings
   - Verify WebGPU/WebNN is enabled in your browser
   - Some browsers may require additional flags to enable experimental features

## Contributing

Contributions to improve the benchmark system are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## License

This benchmark system is provided under the same license as the main project.