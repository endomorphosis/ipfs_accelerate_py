# Comprehensive Benchmark Execution Guide

This guide provides detailed information about running comprehensive benchmarks for all model types across all hardware platforms using the updated benchmarking system.

**Date: April 2025**  
**Last Updated: March 16, 2025**  
**Status: IN PROGRESS (85% Complete)**

## Overview

The comprehensive benchmarking system executes benchmarks for 13 key model types across 8 hardware platforms, collecting detailed performance metrics and storing them in a centralized DuckDB database for analysis and reporting.

The system has been enhanced with a new orchestration script that simplifies the process of running benchmarks and provides better integration with the database.

## NEW: Enhanced Hardware Benchmarking Tools (March 2025)

We've developed new hardware benchmarking tools that provide an even simpler interface for running hardware comparisons:

### Interactive Benchmarking Tool

The easiest way to run benchmarks is with our new interactive tool:

```bash
./interactive_hardware_benchmark.py
```

This tool:
- Automatically detects available hardware backends on your system
- Guides you through selecting models, hardware, and benchmark parameters
- Runs the benchmarks and generates comprehensive reports
- Provides real-time feedback during benchmark execution

### Command-line Hardware Benchmark Runner

For more control or automation, use the hardware benchmark script:

```bash
./run_hardware_benchmark.sh --model-set text_embedding --hardware "cpu cuda openvino" --batch-sizes "1 4 16"
```

Key features:
- Supports predefined model sets or custom model lists
- Detects available hardware backends automatically
- Provides detailed performance metrics with memory profiling
- Generates markdown and JSON reports with hardware comparisons

### Direct Python API with OpenVINO Integration

For maximum flexibility, use the direct Python API:

```bash
python run_hardware_comparison.py --models prajjwal1/bert-tiny google/t5-efficient-tiny --hardware cpu cuda openvino --batch-sizes 1 4 16
```

Enhanced features:
- Uses Optimum library for optimal OpenVINO model conversion
- Supports different precision levels (FP32, FP16, INT8)
- Graceful fallback to CPU when OpenVINO fails
- Proper batch input handling for different model types

For complete documentation of these new tools, see [Hardware Benchmarking README](HARDWARE_BENCHMARKING_README.md).

## Components

The system consists of the following core components:

1. `run_comprehensive_benchmarks.py` - New orchestration script for running benchmarks (April 2025)
2. `execute_comprehensive_benchmarks.py` - Core benchmark execution module
3. `benchmark_hardware_models.py` - Hardware-specific benchmark implementation
4. `benchmark_timing_report.py` - Report generation from benchmark results
5. `duckdb_api/core/benchmark_db_api.py` - Database interface for storing and retrieving results

## New Feature: Simplified Benchmark Execution

The new `run_comprehensive_benchmarks.py` script provides a simplified interface for running benchmarks:

```bash
# Run benchmarks for default models on available hardware
python test/run_comprehensive_benchmarks.py

# Specify models to benchmark
python test/run_comprehensive_benchmarks.py --models bert,t5,vit

# Specify hardware platforms
python test/run_comprehensive_benchmarks.py --hardware cpu,cuda

# Specify batch sizes to test
python test/run_comprehensive_benchmarks.py --batch-sizes 1,4,16

# Force hardware platforms even if not detected
python test/run_comprehensive_benchmarks.py --force-hardware rocm,webgpu

# List available hardware platforms
python test/run_comprehensive_benchmarks.py --list-available-hardware

# Use full-sized models instead of smaller variants
python test/run_comprehensive_benchmarks.py --no-small-models

# Generate report in different formats
python test/run_comprehensive_benchmarks.py --report-format markdown

# Specify database path and output directory
python test/run_comprehensive_benchmarks.py --db-path ./benchmark_db.duckdb --output-dir ./benchmark_results

# Run benchmarks on all hardware platforms (may use simulation)
python test/run_comprehensive_benchmarks.py --all-hardware

# Specify timeout for benchmarks
python test/run_comprehensive_benchmarks.py --timeout 1200
```

### Key Features

- **Hardware Auto-detection**: Automatically detects available hardware platforms with advanced detection capabilities
- **Central Hardware Detection**: Uses the `centralized_hardware_detection` module when available for more accurate detection
- **Model Selection**: Run benchmarks for specific models or model subsets
- **Batch Size Customization**: Specify custom batch sizes for more targeted testing
- **Force Hardware Mode**: Run benchmarks on specific hardware even if not detected as available
- **Output Control**: Specify where results and reports are saved 
- **Small Model Support**: Use smaller model variants for faster testing
- **Real-time Logging**: View benchmark progress in real-time
- **Status Tracking**: Stores benchmark status in JSON files for monitoring and resume capabilities
- **Report Format Options**: Generate reports in HTML, Markdown, JSON, or CSV formats
- **Performance Metrics**: Captures detailed metrics on latency, throughput, and memory usage 
- **Timeout Control**: Set custom timeouts for benchmarks to prevent hanging
- **Comprehensive Status Reporting**: Detailed status reporting with timing and error information
- **Latest Report Symlinks**: Automatically creates symlinks to the latest reports for easy access

## Usage Guide

### Basic Usage

The simplest way to run benchmarks is:

```bash
python test/run_comprehensive_benchmarks.py
```

This will:
1. Auto-detect available hardware platforms (CPU, CUDA, etc.)
2. Run benchmarks for a default set of models (bert, t5, vit, whisper)
3. Use smaller model variants for faster testing
4. Store results in the default database (./benchmark_db.duckdb)
5. Generate a comprehensive timing report

### Advanced Usage

#### Selecting Models

To benchmark specific models:

```bash
# Single model
python test/run_comprehensive_benchmarks.py --models bert

# Multiple models
python test/run_comprehensive_benchmarks.py --models bert,t5,vit,clip
```

#### Selecting Hardware

To benchmark on specific hardware platforms:

```bash
# Single hardware platform
python test/run_comprehensive_benchmarks.py --hardware cpu

# Multiple hardware platforms
python test/run_comprehensive_benchmarks.py --hardware cpu,cuda

# Force hardware platforms even if not available (may use simulation)
python test/run_comprehensive_benchmarks.py --force-hardware rocm,webgpu

# List available hardware platforms and exit
python test/run_comprehensive_benchmarks.py --list-available-hardware

# Run on all supported hardware platforms (may use simulation)
python test/run_comprehensive_benchmarks.py --all-hardware
```

#### Customizing Batch Sizes

To test specific batch sizes:

```bash
python test/run_comprehensive_benchmarks.py --batch-sizes 1,4,16,32
```

#### Using Full-sized Models

By default, smaller model variants are used for faster testing. To use full-sized models:

```bash
python test/run_comprehensive_benchmarks.py --no-small-models
```

#### Custom Database and Output Directory

To specify a custom database path and output directory:

```bash
python test/run_comprehensive_benchmarks.py --db-path /path/to/db.duckdb --output-dir /path/to/results
```

#### Custom Report Format

To generate reports in different formats:

```bash
python test/run_comprehensive_benchmarks.py --report-format html
python test/run_comprehensive_benchmarks.py --report-format markdown
python test/run_comprehensive_benchmarks.py --report-format json
python test/run_comprehensive_benchmarks.py --report-format csv
```

#### Setting Timeouts

To set a custom timeout for each benchmark (in seconds):

```bash
python test/run_comprehensive_benchmarks.py --timeout 1200  # 20 minutes
```

### Benchmark Status Tracking

The new system includes enhanced status tracking capabilities:

- Status files are created in the output directory with timestamps
- A `benchmark_status_latest.json` file is always maintained with the latest status
- Status includes detailed timing information and error messages if applicable
- Status tracks all parameters used for the benchmark run
- Output is captured for debugging purposes

To check the status of the latest benchmark run:

```bash
cat benchmark_results/benchmark_status_latest.json
```

### Available Models

The system supports benchmarking the following model types:

| Model Type | Default Small Variant | Full-sized Variant |
|------------|----------------------|-------------------|
| bert       | prajjwal1/bert-tiny  | bert-base-uncased |
| t5         | google/t5-efficient-tiny | t5-small      |
| vit        | facebook/deit-tiny-patch16-224 | google/vit-base-patch16-224 |
| whisper    | openai/whisper-tiny | openai/whisper-tiny |
| clip       | openai/clip-vit-base-patch16-224 | openai/clip-vit-base-patch32 |
| llama      | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| clap       | laion/clap-htsat-unfused | laion/clap-htsat-unfused |
| wav2vec2   | facebook/wav2vec2-base | facebook/wav2vec2-base |
| llava      | llava-hf/llava-1.5-7b-hf | llava-hf/llava-1.5-7b-hf |
| llava-next | llava-hf/llava-v1.6-mistral-7b | llava-hf/llava-v1.6-mistral-7b |
| xclip      | microsoft/xclip-base-patch32 | microsoft/xclip-base-patch32 |
| qwen2      | Qwen/Qwen2-0.5B-Instruct | Qwen/Qwen2-0.5B-Instruct |
| detr       | facebook/detr-resnet-50 | facebook/detr-resnet-50 |

### Available Hardware Platforms

The system supports benchmarking on the following hardware platforms:

| Hardware Platform | Description | Detection Method |
|-------------------|-------------|------------------|
| cpu               | Standard CPU processing | Always available |
| cuda              | NVIDIA GPU acceleration | PyTorch detection or centralized hardware detection |
| rocm              | AMD GPU acceleration | Environment variable or centralized hardware detection |
| mps               | Apple Silicon GPU acceleration | PyTorch MPS backend detection |
| openvino          | Intel acceleration | Import test for OpenVINO package |
| qnn               | Qualcomm AI Engine | Centralized hardware detection |
| webnn             | Browser neural network API | Centralized hardware detection |
| webgpu            | Browser graphics API for ML | Centralized hardware detection |

Note: Actual availability depends on your hardware configuration. The script automatically detects available platforms and can be forced to run on platforms that aren't detected.

## Hardware Detection System

The enhanced script now includes a sophisticated hardware detection system:

- **Two-Tier Detection**: First tries the centralized hardware detection system, then falls back to basic detection
- **Complete Hardware Mapping**: Maps all 8 supported hardware platforms with availability status
- **Detailed Logging**: Provides detailed logs of hardware detection process
- **Force Mode**: Allows forcing benchmarks on hardware even if not detected
- **Simulation Flag Tracking**: Properly marks simulated hardware in benchmark results
- **Status Reporting**: Lists available hardware platforms with visual indicators
- **Hardware Validation**: Tracks and reports hardware validation status

To list available hardware platforms:

```bash
python test/run_comprehensive_benchmarks.py --list-available-hardware
```

## Database Integration

All benchmark results are stored in a DuckDB database, which provides:

- Efficient storage of benchmark results
- Fast querying capabilities for analysis
- Historical tracking of performance metrics
- Integration with reporting tools
- Simulation status tracking

### Database Schema

The database uses the following schema for benchmark results:

```sql
-- Performance results table
CREATE TABLE performance_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,
    model_id INTEGER,
    hardware_id INTEGER,
    batch_size INTEGER,
    average_latency_ms FLOAT,
    throughput_items_per_second FLOAT,
    memory_peak_mb FLOAT,
    is_simulated BOOLEAN,
    simulation_reason VARCHAR,
    test_timestamp TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
)
```

## Report Generation

After running benchmarks, a comprehensive timing report is automatically generated. The report includes:

- Comparative performance analysis across hardware platforms
- Visualization of latency, throughput, and memory usage
- Optimization recommendations based on benchmark results
- Detailed metrics for each model-hardware combination
- Clear indication of simulated vs. real hardware results

To generate a report from existing benchmark data:

```bash
python test/benchmark_timing_report.py --generate --format html --output report.html
python test/benchmark_timing_report.py --generate --format markdown --output report.md
python test/benchmark_timing_report.py --generate --format json --output report.json
python test/benchmark_timing_report.py --generate --format csv --output report.csv
```

## Implementation Progress (April 2025)

The comprehensive benchmarking system implementation is currently at **65%** completion:

- ✅ Core benchmark execution framework (COMPLETED)
- ✅ Database schema for storing results (COMPLETED)
- ✅ Report generation infrastructure (COMPLETED)
- ✅ Orchestration script for running benchmarks (COMPLETED)
- ✅ Hardware detection and compatibility (COMPLETED)
- ✅ Critical benchmark system fixes (COMPLETED)
- ✅ Enhanced hardware detection with centralized system (COMPLETED)
- ✅ Batch size customization (COMPLETED)
- ✅ Multiple report format support (COMPLETED)
- ✅ Benchmark status tracking and reporting (COMPLETED)
- ✅ Timeout control for benchmarks (COMPLETED)
- ✅ Support for forcing hardware platforms (COMPLETED)
- 🔄 Running benchmarks on available hardware (IN PROGRESS)
- 🔄 Collecting timing metrics for all model-hardware combinations (IN PROGRESS)
- 🔄 Hardware procurement for missing platforms (IN PROGRESS)
- ❌ WebNN and WebGPU testing environment (NOT STARTED)
- ❌ Performance analysis of all 13 models on all 8 platforms (NOT STARTED)
- ❌ Final optimization recommendations (NOT STARTED)

## Recent Improvements (April 9, 2025)

1. **Enhanced Hardware Detection**
   - Added integration with centralized hardware detection system
   - Improved fallback mechanisms for hardware detection
   - Added detailed hardware status reporting

2. **Customizable Benchmarking**
   - Added support for custom batch sizes
   - Added support for forcing hardware platforms
   - Added support for custom timeouts

3. **Status Tracking and Reporting**
   - Added comprehensive status tracking in JSON format
   - Added detailed timing information
   - Added error capturing and reporting
   - Added symlinks to latest reports and status files

4. **Multiple Report Formats**
   - Added support for HTML, Markdown, JSON, and CSV reports
   - Added more detailed performance metrics
   - Improved visualization capabilities
   - CSV format for data analysis in spreadsheet applications

## Conclusion

The enhanced comprehensive benchmarking system provides a robust framework for evaluating model performance across different hardware platforms. With the new orchestration script, running benchmarks is now simpler and more reliable.

The latest improvements enable more flexible testing with custom batch sizes, timeouts, and hardware forcing capabilities. The enhanced status tracking provides better visibility into benchmark progress and results.

As the project progresses toward completion, additional hardware platforms will be tested, and a complete set of benchmark results will be published to guide hardware selection decisions.

For more information, see the project roadmap in NEXT_STEPS.md or contact the development team.

---

**Related Documentation:**
- [Benchmark Timing Guide](BENCHMARK_TIMING_GUIDE.md)
- [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md)
- [Database Integration Guide](BENCHMARK_DATABASE_GUIDE.md)
- [Next Steps](NEXT_STEPS.md)