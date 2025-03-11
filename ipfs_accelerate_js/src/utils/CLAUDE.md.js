# IPFS Accelerate Python Framework - Development Guide

> **ORGANIZATION UPDATE (March 10, 2025):**
>
> The codebase has been reorganized for better maintainability with the following top-level structure:
> 
> - All generator files moved to the top-level `generators/` directory (216 files) with subdirectories:
>   - `generators/benchmark_generators/`: Benchmark generation tools
>   - `generators/models/`: Model implementations and skills
>   - `generators/runners/`: Test runner scripts
>   - `generators/skill_generators/`: Skill generation tools
>   - `generators/template_generators/`: Template generation utilities
>   - `generators/templates/`: Template files for model generation
>   - `generators/test_generators/`: Test generation tools
>   - `generators/utils/`: Utility functions
>   - `generators/hardware/`: Hardware-specific generator tools
>
> - All database-related tools moved to the top-level `duckdb_api/` directory (83 files) with subdirectories:
>   - `duckdb_api/core/`: Core database functionality
>   - `duckdb_api/migration/`: Migration tools for JSON to database
>   - `duckdb_api/schema/`: Database schema definitions 
>   - `duckdb_api/utils/`: Utility functions for database operations
>   - `duckdb_api/visualization/`: Result visualization tools
>   - `duckdb_api/distributed_testing/`: Distributed testing framework components
>
> - Web platform implementations remain in `fixed_web_platform/` directory with subdirectories:
>   - `fixed_web_platform/unified_framework/`: Unified API for cross-browser WebNN/WebGPU
>   - `fixed_web_platform/wgsl_shaders/`: WebGPU Shading Language shader implementations
>
> - CI/CD workflow files moved from `test/.github/workflows/` to the standard `.github/workflows/` location
>
> ✅ Migration completed with 299 files moved and all import paths updated (March 9, 2025)
> ✅ CI/CD pipeline configuration updated to use new directory structure (March 9, 2025)
> ✅ All documentation path references have been updated to use the new structure
>
> Please refer to [FINAL_MIGRATION_REPORT.md](FINAL_MIGRATION_REPORT.md) for the complete directory structure and [CI_CD_UPDATES_SUMMARY.md](CI_CD_UPDATES_SUMMARY.md) for details on CI/CD changes.
>
> **ARCHIVE UPDATE (March 10, 2025):**
>
> A major cleanup of stale and unused files has been completed to improve repository organization:
>
> - Approximately 480 files were moved to `/test/archive/` including:
>   - Backup files (*.bak, *.bak_*)
>   - Old database backups (benchmark_db*.duckdb.bak*)
>   - Fixed/superseded implementation files
>   - Completed status reports and documentation
>   - Older benchmark reports
>   - One-time utility scripts
>   - Deprecated test runners
>
> - Archive directory structure:
>   - `/test/archive/`: Primary archive for all stale test files
>   - Additional documentation about the archive is available in:
>     - `/test/cleanup_summary.md`: Detailed summary of the archiving process 
>     - `/test/ARCHIVE_STRUCTURE.md`: Guidelines for archive organization and access
>     - `/test/DOCUMENTATION_INDEX.md`: Updated to reflect archived documentation
>
> ✅ Archive cleanup completed with 480+ files organized (March 10, 2025)
> ✅ Documentation updated to reflect archive status (March 10, 2025)
> ✅ Archive structure documented for future reference (March 10, 2025)
>
> **UPCOMING MIGRATION (Q2-Q3 2025):**
> 
> All WebGPU/WebNN implementations will be moved from `/fixed_web_platform/` to a dedicated `ipfs_accelerate_js` folder once all tests pass. This migration will create a clearer separation between JavaScript-based components and Python-based components.
>
> The structure and contents of the `ipfs_accelerate_js` folder will maintain isomorphism with the `ipfs_accelerate_py` structure, ensuring consistent organization across both implementations. Key directories will include:
> - `ipfs_accelerate_js/core/`: Core JavaScript functionality matching Python counterparts
> - `ipfs_accelerate_js/models/`: Model implementations with feature parity
> - `ipfs_accelerate_js/utils/`: Utility functions with equivalent capabilities
> - `ipfs_accelerate_js/webgpu/`: WebGPU-specific optimizations
> - `ipfs_accelerate_js/webnn/`: WebNN-specific implementations
> - `ipfs_accelerate_js/shaders/`: WGSL shader implementations
> - `ipfs_accelerate_js/transformers/`: Integration with transformers.js

## Current Focus: Advanced Hardware Benchmarking and Database Consolidation (Updated March 2025)
## Enhanced Feature: Added Qualcomm AI Engine Support (Updated March 2025)

### Project Status Overview

The project has successfully completed 16 phases of implementation, focusing on test-driven development, hardware compatibility, model optimization, cross-platform support, and data management. Key accomplishments include:

- ✅ Complete development pipeline for test and skillset generators
- ✅ Comprehensive hardware detection and compatibility system
- ✅ Advanced resource management system with hardware awareness
- ✅ Web platform integration (WebNN and WebGPU) with real browser-based implementations
- ✅ Model family classification and compatibility matrix 
- ✅ Integration testing and platform support
- ✅ Advanced model compression and optimization
- ✅ Complete hardware platform test coverage for key models
- ✅ Test results database architecture and core components implemented (100% complete)
- ✅ Historical data migration pipeline implemented (100% complete)
- ✅ CI/CD integration for automated benchmark storage (100% complete)

### Completed Major Phases and Milestones

- ✅ **Phase 16 - Advanced Hardware Benchmarking and Database Consolidation** (March 2025)
- ✅ **Web Platform Integration and Framework** (March 2025)
- ✅ **Template System Reorganization** (March 9, 2025)
- ✅ **Database Reorganization** (March 9, 2025)
- ✅ **Model File Verification Pipeline** (March 9, 2025)
- ✅ **Time-Series Performance Tracking** (March 25, 2025)
- ✅ **Cross-Browser Model Sharding** (March 8, 2025)
- ✅ **Benchmark System Enhancements** (April 6, 2025)
- ✅ **Mobile and Edge Support** (April 6, 2025)
- ✅ **Predictive Performance System** (June 5, 2025)
- ✅ **IPFS Acceleration with WebNN/WebGPU Integration** (May 22, 2025)
- ✅ **Template Database Migration** (100% complete, March 10, 2025) - [Documentation](TEMPLATE_DATABASE_README.md)
- ✅ **Template Validation System** (100% complete)

### Current Focus Areas (Q2 2025):

- 🔄 **Improved End-to-End Testing Framework** (NEW - March 11, 2025)
  - Generation and testing of skill, test, and benchmark components together for every model
  - Creation of "expected results" and "collected results" folders for verification
  - Markdown documentation of HuggingFace class skills to compare with templates
  - Focus on fixing generators rather than individual test files
  - Template-driven approach for maintenance efficiency
  - Target completion: April 15, 2025

- 🔄 **WebGPU/WebNN Resource Pool Integration** (IN PROGRESS - 85% complete)
  - Enables concurrent execution of multiple AI models across heterogeneous browser backends
  - Creates browser-aware load balancing for model type optimization
  - Implements connection pooling for browser instance lifecycle management
  - ✅ NEW: Real browser integration with Selenium (March 10, 2025)
  - ✅ NEW: Performance-aware browser selection based on historical data (March 10, 2025)
  - ✅ NEW: Smart browser distribution with scoring system (March 10, 2025)
  - ✅ NEW: Asynchronous API for browser management (March 10, 2025)
  - ✅ NEW: Cross-model tensor sharing for memory efficiency (March 10, 2025)
  - ✅ NEW: Ultra-low bit quantization with 2-bit and 3-bit support (March 10, 2025)
  - NEW: Fault-tolerant cross-browser model sharding with recovery (May 2025)
  - NEW: Performance history tracking and trend analysis (May 2025)
  - Target completion: May 25, 2025
  
- 🔄 **Distributed Testing Framework** (IN PROGRESS - 25% complete)
  - Coordinator-worker architecture for distributed test execution
  - Secure worker node registration with JWT-based authentication
  - Intelligent task distribution based on hardware capabilities
  - Target completion: June 26, 2025

- 📋 **WebGPU/WebNN Migration to ipfs_accelerate_js** (PLANNED - After all tests pass)
  - Move all WebGPU/WebNN implementations to dedicated folder structure
  - Create clearer separation between JavaScript and Python components
  - Update import paths and documentation to reflect new structure
  - Simplify future JavaScript SDK development
  - Target completion: Q3 2025

- ✅ **Ultra-low precision quantization support** (COMPLETED - March 10, 2025)
  - 2-bit and 3-bit quantization for WebGPU with custom compute shaders
  - Memory-efficient KV cache with 87.5% memory reduction
  - Browser-specific optimizations for Chrome, Firefox, Edge, and Safari
  - Mixed precision with layer-specific quantization bit levels
  - Extended context window (up to 8x longer) with 2-bit quantization

#### Latest Framework Enhancements
- ✅ Cross-Browser Model Sharding with Fault Tolerance (ENHANCED - May 2025)
  - Run large models distributed across multiple browser types to leverage browser-specific optimizations
  - NEW: Automatic component recovery for failed or degraded components
  - NEW: Multiple sharding strategies (layer, attention_feedforward, component)
  - Browser capability detection with specialized optimizations
  - Intelligent component distribution based on browser strengths
  - Chrome focus for vision models and parallel tensor operations
  - Firefox optimization for audio models with compute shader support
  - Edge integration for text models and WebNN acceleration
- ✅ WebGPU/WebNN Resource Pool Integration (IN PROGRESS - 85% complete)
  - Integrated IPFS acceleration with WebNN/WebGPU hardware backends
  - Added browser-specific optimizations (Firefox for audio, Edge for WebNN)
  - Created precision control (2-bit, 3-bit, 4-bit, 8-bit, 16-bit) with mixed precision support
  - Created comprehensive documentation for the resource pool integration
  - ✅ NEW: Complete real browser integration with Selenium (March 10, 2025)
  - ✅ NEW: Performance-aware browser selection based on historical data (March 10, 2025)
  - ✅ NEW: Smart browser distribution with scoring system (March 10, 2025)
  - ✅ NEW: Asynchronous API for browser management (March 10, 2025)
  - ✅ NEW: Concurrent model execution with 3.5x throughput improvement (March 2025)
  - ✅ NEW: Adaptive connection scaling based on workload patterns (March 2025)
  - ✅ NEW: Model-aware browser selection for optimal performance (March 2025)
  - ✅ NEW: WebSocket communication bridge for browser integration (March 10, 2025)
  - ✅ NEW: Real-time browser capability detection (March 10, 2025)
  - ✅ NEW: Graceful fallback to simulation when real browsers unavailable (March 10, 2025)
  - ✅ NEW: Cross-model tensor sharing with reference counting (March 10, 2025)
  - ✅ NEW: Ultra-low bit quantization (2-bit, 3-bit) with shared KV cache (March 10, 2025)
  - ✅ NEW: Layer-specific mixed precision configuration (March 10, 2025)
  - ✅ NEW: Extended context window (up to 8x longer) with optimized memory usage (March 10, 2025)
  - NEW: Enhanced error recovery with performance-based strategies (Planned)
  - NEW: Comprehensive performance analysis and reporting (Planned)
  - See [IPFS_RESOURCE_POOL_INTEGRATION_GUIDE.md](IPFS_RESOURCE_POOL_INTEGRATION_GUIDE.md) for details
  
## Cross-Model Tensor Sharing (NEW - March 10, 2025)

The Cross-Model Tensor Sharing system enables efficient sharing of tensors between multiple models, 
significantly improving memory efficiency and performance for multi-model workloads:

### Key Features

- **Shared Tensor Memory**: Multiple models can share the same tensor memory for common components
- **Reference Counting**: Intelligent memory management ensures tensors are only freed when no longer needed
- **Zero-Copy Tensor Views**: Create views into tensors without duplicating memory
- **Browser Storage Types**: Support for different tensor storage formats (CPU, WebGPU, WebNN)
- **Automatic Memory Optimization**: Identifies and frees unused tensors to reduce memory footprint
- **Intelligent Sharing Patterns**: Automatically identifies which models can share tensors

### Performance Benefits

- **Memory Reduction**: Up to 30% memory reduction for common multi-model workflows
- **Inference Speedup**: Up to 30% faster inference when reusing cached embeddings
- **Increased Throughput**: Higher throughput when running multiple related models
- **Browser Resource Efficiency**: More efficient use of limited browser memory resources

### Tensor Sharing Types

The system automatically identifies compatible model combinations for sharing:

| Tensor Type | Compatible Models | Description |
|-------------|------------------|-------------|
| text_embedding | BERT, T5, LLAMA, BART | Text embeddings for NLP models |
| vision_embedding | ViT, CLIP, DETR | Vision embeddings for image models |
| audio_embedding | Whisper, Wav2Vec2, CLAP | Audio embeddings for speech/audio models |
| vision_text_joint | CLIP, LLaVA, BLIP | Joint embeddings for multimodal models |
| audio_text_joint | CLAP, Whisper-Text | Joint embeddings for audio-text models |

### Usage Example

```python
# Create tensor sharing manager
manager = pool.setup_tensor_sharing(max_memory_mb=2048)

# Models can produce shareable tensors
model1 = await pool.get_model(model_type="text_embedding", model_name="bert-base")
result1 = model1("Sample text")  # Produces shareable embedding tensor

# Other models can reuse the same tensor
model2 = await pool.get_model(model_type="text_embedding", model_name="t5-small") 
result2 = model2("Sample text")  # Automatically reuses shared embedding tensor

# Run memory optimization when needed
memory_savings = pool.tensor_sharing_manager.optimize_memory_usage()
```

See [IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md](IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md) for complete documentation.
- 🔄 Distributed testing framework (IN PROGRESS - Started May 8, 2025)
  - Design high-performance distributed test execution system
  - Initial implementation of core components
  - Create secure worker node registration and management system
- 📅 Ultra-low precision quantization support (PLANNED - July 2025)
  - 2-bit and 3-bit quantization for WebGPU
  - Memory-efficient KV cache with 87.5% memory reduction
  - Browser-specific optimizations for Chrome, Firefox, Edge, and Safari

## Time-Series Performance Tracking (COMPLETED - March 25, 2025)

The framework now includes a comprehensive time-series performance tracking system with these features:

- Versioned test results with git commit and environment information
- Regression detection based on configurable thresholds
- Trend analysis with statistical methods
- Visualization capabilities for performance metrics
- Reporting in Markdown and HTML formats
- Notification system for detected regressions 

```bash
# Run a quick test of the time-series performance tracker
python duckdb_api/run_time_series_performance.py --quick-test

# Run the full test suite
python duckdb_api/run_time_series_performance.py --full-test

# Record a performance result
python duckdb_api/time_series_performance.py record --model-id 1 --hardware-id 1 --batch-size 4 --throughput 125.7 --latency 8.2 --memory 1024 --power 180

# Set baselines for all model-hardware combinations
python duckdb_api/time_series_performance.py baseline --all --days 7 --min-samples 3

# Detect regressions
python duckdb_api/time_series_performance.py regression --days 14 --notify

# Analyze trends
python duckdb_api/time_series_performance.py trend --metric throughput --days 30 --visualize

# Generate a performance report
python duckdb_api/time_series_performance.py report --days 30 --format markdown --output performance_report.md
```

For detailed documentation, see [Time-Series Performance Tracking Guide](TIME_SERIES_PERFORMANCE_GUIDE.md).

## Comprehensive Benchmark Timing Report (COMPLETED - March 6, 2025)

The framework includes a comprehensive benchmark timing report generator that provides detailed analysis of performance metrics for all 13 model types across 8 hardware endpoints:

- Detailed latency, throughput, and memory usage metrics
- Cross-hardware platform performance comparison
- Visualizations for performance metrics (HTML and Markdown formats)
- Categorized model performance by type (text, vision, audio, multimodal)
- Data-driven optimization recommendations based on model categories
- Consistent DuckDB database schema for all benchmark data
- Support for sample data generation for testing and demos

## Comprehensive Benchmarks and Timing Data (UPDATED - April 10, 2025)

The framework includes full benchmark execution and timing data for all model types across all hardware platforms:

- Comprehensive benchmarks for all 13 model types across 8 hardware platforms
- Intelligent incremental benchmarking system for efficient resource utilization (NEW - March 6, 2025)
- Dynamic scheduling based on database queries for missing or outdated benchmarks
- Prioritization of critical model-hardware combinations
- Detailed performance metrics including latency, throughput, and memory usage
- Hardware compatibility matrix with optimization recommendations
- HTML and Markdown reports with detailed performance comparisons
- Interactive visualizations for comparing hardware platforms
- Power efficiency metrics for mobile/edge devices
- Benchmark completion report with status of all testing targets
- March 2025 Web Platform optimizations benchmark results:
  - WebGPU compute shader optimization for audio models (Whisper, Wav2Vec2)
  - Parallel loading optimization for multimodal models (CLIP, LLaVA)
  - Shader precompilation for text and vision models (BERT, ViT)
  - Combined optimization benchmarks with all features enabled
- Clear distinction between real and simulated hardware results (ADDED - April 6, 2025)
- Simulation detection and reporting for transparent benchmarking

```bash
# Use intelligent incremental benchmark runner (NEW - March 2025)
python duckdb_api/utils/run_incremental_benchmarks.py

# Run incremental benchmarks for specific models and hardware
python duckdb_api/utils/run_incremental_benchmarks.py --models bert,t5,vit --hardware cpu,cuda

# Only run benchmarks that don't exist in the database
python duckdb_api/utils/run_incremental_benchmarks.py --missing-only

# Run benchmarks older than 14 days
python duckdb_api/utils/run_incremental_benchmarks.py --refresh-older-than 14

# Run only priority model-hardware combinations
python duckdb_api/utils/run_incremental_benchmarks.py --priority-only

# Execute comprehensive benchmarks using the new script (April 2025 Update)
python duckdb_api/utils/run_comprehensive_benchmarks.py

# Run specific models on specific hardware
python duckdb_api/utils/run_comprehensive_benchmarks.py --models bert,t5,vit --hardware cpu,cuda

# Specify batch sizes to test
python duckdb_api/utils/run_comprehensive_benchmarks.py --batch-sizes 1,4,16

# Force benchmarks on hardware that may not be available
python duckdb_api/utils/run_comprehensive_benchmarks.py --force-hardware rocm,webgpu

# List available hardware platforms
python duckdb_api/utils/run_comprehensive_benchmarks.py --list-available-hardware

# Run benchmarks on all supported hardware platforms (may use simulation)
python duckdb_api/utils/run_comprehensive_benchmarks.py --all-hardware

# Use full-sized models instead of smaller variants
python duckdb_api/utils/run_comprehensive_benchmarks.py --no-small-models

# Generate report in different formats
python duckdb_api/utils/run_comprehensive_benchmarks.py --report-format markdown

# Set a custom timeout for benchmarks
python duckdb_api/utils/run_comprehensive_benchmarks.py --timeout 1200  # 20 minutes

# Specify database path and output directory
python duckdb_api/utils/run_comprehensive_benchmarks.py --db-path ./benchmark_db.duckdb --output-dir ./benchmark_results

# Web Platform Testing (April 2025 Enhancement)
# Set up web testing environment with browser detection
python generators/runners/web/setup_web_testing.py --browser chrome

# Run WebGPU tests with compute shader optimization for audio models
python generators/runners/web/run_web_benchmarks.py --models whisper,wav2vec2 --hardware webgpu --web-compute-shaders 

# Run WebGPU tests with parallel loading for multimodal models
python generators/runners/web/run_web_benchmarks.py --models clip,llava --hardware webgpu --web-parallel-loading

# Run WebGPU tests with shader precompilation for faster startup
python generators/runners/web/run_web_benchmarks.py --models bert,vit --hardware webgpu --web-shader-precompile

# Run WebNN tests for best performance on Edge browser
python generators/runners/web/run_web_benchmarks.py --models bert,t5 --hardware webnn --browser edge

# Enable all WebGPU optimizations at once with specific browser
python generators/runners/web/run_web_benchmarks.py --models all --hardware webgpu --web-all-optimizations --browser firefox

# Legacy method: Execute comprehensive benchmarks across all hardware platforms
python duckdb_api/core/benchmark_all_key_models.py --output-dir ./benchmark_results

# Run with small model variants for faster testing
python duckdb_api/core/benchmark_all_key_models.py --small-models --output-dir ./benchmark_results

# Generate comprehensive benchmark timing report in multiple formats
python duckdb_api/visualization/benchmark_timing_report.py --generate --format html --output report.html
python duckdb_api/visualization/benchmark_timing_report.py --generate --format markdown --output report.md

# Generate hardware compatibility matrix with visualization
python duckdb_api/visualization/get_compatibility_matrix.py
```

```bash
# Generate comprehensive benchmark timing report in HTML format
python duckdb_api/visualization/run_benchmark_timing_report.py --generate --format html

# Generate report in Markdown format
python duckdb_api/visualization/run_benchmark_timing_report.py --generate --format markdown

# Specify custom output location and database path
python duckdb_api/visualization/run_benchmark_timing_report.py --generate --format html --output report.html --db-path ./benchmark_db.duckdb

# Generate sample benchmark data for testing
python duckdb_api/utils/generate_sample_benchmarks.py --db ./benchmark_db.duckdb

# Run real benchmarks with database integration
python duckdb_api/core/benchmark_all_key_models.py --small-models --db-path ./benchmark_db.duckdb --db-only

# Generate model-hardware performance report
python duckdb_api/core/benchmark_db_query.py --sql "SELECT m.model_name, hp.hardware_type, AVG(pr.average_latency_ms) as avg_latency, AVG(pr.throughput_items_per_second) as avg_throughput FROM performance_results pr JOIN models m ON pr.model_id = m.model_id JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id GROUP BY m.model_name, hp.hardware_type ORDER BY m.model_name, hp.hardware_type" --db ./benchmark_db.duckdb --format markdown --output performance_summary.md
```

The report includes specialized views for:
- Text models (BERT, T5, LLAMA, Qwen2)
- Vision models (ViT, DETR, XCLIP)
- Audio models (Whisper, Wav2Vec2, CLAP)
- Multimodal models (CLIP, LLaVA, LLaVA-Next)
- Memory-intensive vs compute-intensive models

Performance data is stored in the DuckDB database for efficient querying and visualization, with comprehensive metrics showing optimal hardware selection for each model category.

For detailed documentation, see [Benchmark Timing Report Guide](BENCHMARK_TIMING_REPORT_GUIDE.md).

## Hardware Compatibility Matrix

### Model Family-Based Compatibility Chart

| Model Family | CPU | CUDA | ROCm | MPS | OpenVINO | QNN | Samsung | WebNN | WebGPU | Notes |
|--------------|-----|------|------|-----|----------|-----|---------|-------|--------|-------|
| Embedding (BERT, etc.) | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Fully supported on all hardware |
| Text Generation (LLMs) | ✅ Medium | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | Memory requirements critical |
| Vision (ViT, CLIP, etc.) | ✅ Medium | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Full cross-platform support |
| Audio (Whisper, etc.) | ✅ Medium | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ✅ High | ⚠️ Limited | ⚠️ Limited | CUDA preferred, Web simulation added |
| Multimodal (LLaVA, etc.) | ⚠️ Limited | ✅ High | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | CUDA for production, others are limited |

### IPFS Acceleration Testing Features (Updated March 2025)

The framework now includes comprehensive IPFS acceleration testing with enhanced DuckDB integration, Qualcomm QNN, and WebGPU support:

1. **Database-First Storage**: Complete integration with DuckDB for efficient and reliable test results storage:
   ```bash
   # Store results only in database (no JSON files)
   python generators/models/test_ipfs_accelerate.py --models "bert-base-uncased" --db-only
   
   # Use custom database path
   python generators/models/test_ipfs_accelerate.py --db-path ./custom_benchmark.duckdb --models "bert-base-uncased"
   ```

2. **Qualcomm AI Engine Support**: Test with Qualcomm QNN hardware acceleration:
   ```bash
   # Test with Qualcomm QNN acceleration
   python generators/models/test_ipfs_accelerate.py --qnn --models "bert-base-uncased"
   
   # Run with specific Qualcomm precision settings
   python generators/models/test_ipfs_accelerate.py --qnn --precision int8 --models "bert-base-uncased"
   
   # Generate Qualcomm performance comparison report
   python generators/models/test_ipfs_accelerate.py --qnn-analysis --models "bert-base-uncased,whisper-tiny" --format html
   ```

3. **WebGPU Support and Analysis**: Test and analyze browser-based GPU acceleration:
   ```bash
   # Test with WebGPU acceleration
   python generators/models/test_ipfs_accelerate.py --webgpu --models "bert-base-uncased"
   
   # Generate WebGPU analysis report with shader metrics
   python generators/models/test_ipfs_accelerate.py --webgpu-analysis --browser firefox --shader-metrics --format html
   
   # Generate comprehensive WebGPU performance analysis across browsers
   python generators/models/test_ipfs_accelerate.py --webgpu-analysis --format html
   
   # Analyze compute shader optimizations (especially for audio models)
   python generators/models/test_ipfs_accelerate.py --webgpu-analysis --compute-shader-optimization --browser firefox --format html
   ```

4. **Real-Time Database Integration**: Test results stored in database as they're generated:
   ```bash
   # Test multiple platforms with real-time database integration
   python generators/models/test_ipfs_accelerate.py --models "bert-base-uncased" --qnn --webnn --webgpu --db-only
   ```

5. **Enhanced Visualization and Reporting**:
   - Interactive Plotly charts for performance comparisons
   - WebGPU shader compilation metrics visualization
   - Browser-specific WebGPU performance analysis
   - Model-specific optimization recommendations
   - Hardware compatibility heatmaps
   - Qualcomm power efficiency metrics for mobile/edge devices

6. **Comprehensive Reporting Options**:
   - General report: `--report`
   - IPFS acceleration report: `--ipfs-acceleration-report`
   - Acceleration comparison report: `--comparison-report` 
   - WebGPU analysis report: `--webgpu-analysis` 
   - Qualcomm performance report: `--qnn-analysis` (NEW!)

For detailed documentation on these features, see [IPFS_ACCELERATION_TESTING.md](IPFS_ACCELERATION_TESTING.md).

To generate an updated compatibility matrix with actual benchmark data, run:
```bash
# IMPORTANT: All benchmark results are now stored in DuckDB database, not JSON files
# Set database path with environment variable or parameter
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# Run benchmarks (results stored directly in database)
python duckdb_api/core/benchmark_all_key_models.py --db-only

# Legacy approach (DEPRECATED - not recommended)
# python duckdb_api/core/benchmark_all_key_models.py --output-dir ./benchmark_results
```

This will benchmark all 13 high-priority model classes across all available hardware platforms and generate a comprehensive compatibility matrix based on real performance data. All results will be stored directly in the DuckDB database for efficient querying and analysis.

### Key Model Test Coverage Status

| Model Class | Model Used | CPU | CUDA | ROCm | MPS | OpenVINO | Qualcomm | Samsung | WebNN | WebGPU | Notes |
|-------------|------------|-----|------|------|-----|----------|----------|---------|-------|--------|-------|
| BERT | bert-base-uncased, bert-tiny | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage (March 6) |
| T5 | t5-small, t5-efficient-tiny | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage (March 6) |
| LLAMA | opt-125m | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | WebNN/WebGPU limited by memory |
| CLIP | Local test model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| ViT | vit-base | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| CLAP | Local test model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Web has limited audio support |
| Whisper | whisper-tiny | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Web audio challenges |
| Wav2Vec2 | Local test model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Web audio challenges |
| LLaVA | llava-onevision-base | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Memory intensive |
| LLaVA-Next | Local test model | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Memory intensive |
| XCLIP | Local test model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Limited video support in web |
| Qwen2/3 | qwen2, qwen3, qwen2_vl, qwen3_vl | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Memory constraints |
| DETR | Local test model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Limited detection support |

## Essential Test Commands

### Template-Based Generation System
The framework uses a template-based approach stored in DuckDB to efficiently generate test files, skills, and benchmarks for 300+ HuggingFace model classes. This approach prevents the repository from containing thousands of individual files.

Key features:
- Templates for tests, skills, and benchmarks are stored in the DuckDB database
- Templates include helper functions and dependencies needed across models
- Generators retrieve templates from the database and instantiate them for specific models
- Cross-platform hardware compatibility is built into templates
- Each generator creates tests/skills/benchmarks on demand rather than storing static files

### MARCH 2025 UPDATE: Simplified Template System

A new simplified template system has been implemented that makes it easier to generate hardware-aware tests. This entire system including all templates, template databases, and template utilities has been relocated to the `generators/` directory with specialized subdirectories:

```bash
# Create a simple template database
python generators/skill_generators/create_simple_template_db.py

# Validate templates in the database
python generators/template_generators/simple_template_validator.py --validate-db

# Generate a test with database templates
python generators/test_generators/simple_test_generator.py -g bert -t

# Generate a test with specific hardware platforms
python generators/test_generators/simple_test_generator.py -g vit -p cuda,qualcomm,webgpu -t

# Generate a test with Qualcomm AI Engine support
python generators/test_generators/simple_test_generator.py -g bert -p qualcomm -o test_bert_qualcomm.py

# Check all template system components
python generators/runners/run_template_system_check.py

# List all templates in the database
python generators/test_generators/simple_test_generator.py --list-templates

# Detect available hardware platforms
python generators/test_generators/simple_test_generator.py --detect-hardware
```

```bash
# Generate tests with database templates and cross-platform hardware compatibility
python generators/test_generators/merged_test_generator.py --model bert --cross-platform --hardware all --use-db-templates

# Generate tests for a specific model and hardware platforms using database templates
python generators/integrated_skillset_generator.py --model bert --hardware cuda,openvino,webnn --use-db-templates

# Generate all 300+ HuggingFace model tests from database templates
python generators/test_generators/merged_test_generator.py --all-models --use-db-templates

# Update template database with hardware-specific templates
python generators/templates/template_database.py --update-templates --model-family bert

# Generate and store a new template in the database
python generators/templates/template_database.py --create-template --model-type llama --store-in-db

# List all available templates in the database
python generators/templates/template_database.py --list-templates

# Validate templates in the database
python generators/templates/template_database.py --validate-templates

# Generate all test files for a model family from templates
python generators/test_generators/merged_test_generator.py --family text-embedding --use-db-templates

# Run test generator with all improvements applied
python generators/runners/run_fixed_test_generator.py --model bert --use-db-templates --cross-platform

# Run test generator with all features enabled
python generators/runners/run_fixed_test_generator.py --model bert --enable-all

# Fix generator integration issues
python generators/fixes/fix_template_integration.py --integrate-generator fixed_merged_test_generator.py

# Check template database integrity
python generators/fixes/fix_template_integration.py --check-db
```

### Hardware-Aware Test Generation
```bash
# Generate tests with cross-platform hardware compatibility
python generators/integrated_skillset_generator.py --model bert --cross-platform --hardware all

# Generate tests for specific hardware platforms only
python generators/integrated_skillset_generator.py --model bert --hardware cuda,openvino,qnn,webnn

# Generate tests with the improved generator that supports all hardware platforms
python generators/test_generators/qualified_test_generator.py -g bert-base-uncased -p cpu,cuda,rocm,mps,openvino,qnn,webnn,webgpu -o test_bert_all_platforms.py

# Run hardware-specific template generation
python generators/templates/enhance_key_models_hardware_coverage.py --create-templates

# Update the test generator with hardware-aware templates
python generators/test_generators/update_test_generator_with_hardware_templates.py

# Run validation on hardware compatibility
python generators/templates/enhance_key_models_hardware_coverage.py --validate
```

### Phase 16 Hardware Integration
```bash
# Run hardware integration fixes on key model tests
./run_key_model_fixes.sh

# Fix hardware integration for specific models
python generators/fix_hardware_integration.py --specific-models bert,t5,clip

# Fix all key model tests
python generators/fix_hardware_integration.py --all-key-models

# Analyze hardware integration issues without fixing
python generators/fix_hardware_integration.py --all-key-models --analyze-only --output-json hardware_analysis.json

# Test model generators with hardware-aware templates
python generators/update_test_generator_with_hardware_templates.py

# Generate tests with cross-platform hardware compatibility
python generators/integrated_skillset_generator.py --model bert --cross-platform --hardware all
```

### Hardware Testing
```bash
# Automated hardware selection for any model
python generators/hardware/automated_hardware_selection.py --model [model_name] --batch-size [batch_size] --mode [inference|training]

# Select hardware for distributed training
python generators/hardware/automated_hardware_selection.py --model [model_name] --distributed-config --gpu-count 8 --max-memory-gb 40

# Generate comprehensive hardware selection map
python generators/hardware/automated_hardware_selection.py --create-map --output hardware_selection_map.json

# Analyze model performance across all available hardware
python generators/hardware/automated_hardware_selection.py --model [model_name] --analyze --output analysis.json

# Use the Predictive Performance System to predict metrics without running actual benchmarks
python run_predictive_performance_demo.py --model bert-base-uncased --hardware cuda,rocm,mps --batch-sizes 1,2,4,8,16 --visualize

# Predict performance for an untested model-hardware combination
python -m predictive_performance.predict --model t5-small --hardware cuda --batch-size 8 --detailed-output

# Generate performance prediction heatmap across hardware platforms
python -m predictive_performance.predict --model bert-base-uncased --all-hardware --metric throughput --output heatmap.html

# Compare actual vs predicted performance
python -m predictive_performance.predict --validate --model bert-base-uncased --hardware cuda --batch-sizes 1,4,16

# Generate hardware recommendations based on model characteristics
python -m predictive_performance.recommend --model-type text_embedding --size-category medium --optimize-for throughput

# Identify high-value benchmark configurations to improve prediction accuracy
python -m predictive_performance.active_learning --budget 10 --output high_value_tests.json

# Detect available hardware platforms
python generators/hardware/automated_hardware_selection.py --detect-hardware

# Comprehensive hardware detection and compatibility test
python test_comprehensive_hardware.py --test all

# Test hardware backends with specific model
python test_hardware_backend.py --backend [cpu|cuda|rocm|mps|openvino|qualcomm|webnn|webgpu|all] --model [model_name]

# Test resource pool with hardware awareness
python test_resource_pool.py --test hardware

# Test model family integration with web platform support
python test_resource_pool.py --test family --debug
```

### Web Platform Testing

```bash
# Run web platform integration tests
python test_model_integration.py

# Verify web platform integration is correct
python verify_web_platform_integration.py

# Generate a test with WebNN support
python generators/merged_test_generator.py --generate bert --platform webnn

# Generate a test with WebGPU support
python generators/merged_test_generator.py --generate vit --platform webgpu

# Run tests with database integration (DuckDB)
python run_web_platform_tests_with_db.py --models bert t5 vit --small-models --db-path ./benchmark_db.duckdb

# Use environment variable for database path
export BENCHMARK_DB_PATH=./benchmark_db.duckdb
python run_web_platform_tests_with_db.py --all-models --run-webgpu

# Run with browser automation
./run_web_platform_tests.sh --use-browser-automation --browser chrome python generators/runners/web/web_platform_test_runner.py --model bert

# Run WebNN tests with Edge browser
./run_web_platform_tests.sh --webnn-only --use-browser-automation --browser edge python generators/runners/web/web_platform_test_runner.py --model bert

# Run WebGPU tests with Firefox browser
./run_web_platform_tests.sh --webgpu-only --use-browser-automation --browser firefox python generators/runners/web/web_platform_test_runner.py --model vit

# Run browser tests with direct database storage
python generators/runners/web/web_platform_test_runner.py --model bert --platform webnn --browser edge

# Disable JSON output (database storage only)
export DEPRECATE_JSON_OUTPUT=1 python generators/runners/web/web_platform_test_runner.py --model vit --platform webgpu

# Run with enhanced WebGPU compute shaders with DB storage
python generators/runners/web/web_platform_test_runner.py --model whisper --platform webgpu --compute-shaders

# Use database for parallel model loading results
python run_web_platform_tests_with_db.py --models llava clip --parallel-loading

# Store shader compilation metrics in database
WEBGPU_SHADER_PRECOMPILE=1 python generators/runners/web/web_platform_test_runner.py --model vit

# Test all March 2025 optimizations at once (compute shaders, parallel loading, and shader precompilation)
python generators/runners/web/test_web_platform_optimizations.py --all-optimizations

# Combine multiple features with browser automation
./run_web_platform_tests.sh --use-browser-automation --browser chrome --enable-compute-shaders --enable-shader-precompile python generators/runners/web/web_platform_test_runner.py --model whisper

# Run comprehensive web platform integration tests with all optimizations
./run_web_platform_integration_tests.sh --all-optimizations --model clap

# Test specific models with selected optimizations
./run_web_platform_integration_tests.sh --models whisper,wav2vec2 --enable-compute-shaders --enable-shader-precompile

# Test multimodal models with parallel loading
./run_web_platform_integration_tests.sh --models clip,llava --enable-parallel-loading --enable-shader-precompile

# Run comprehensive tests for all models with all optimizations
./run_web_platform_integration_tests.sh --all-models --all-optimizations

# Run tests with database integration and browser automation
./run_web_platform_integration_tests.sh --model bert --use-browser-automation --browser edge --db-path ./benchmark_db.duckdb

# Generate web platform reports from database
python duckdb_api/core/benchmark_db_query.py --report web_platform --format html --output web_report.html

# View advanced WebGPU features usage from database
python duckdb_api/core/benchmark_db_query.py --report webgpu --format html --output webgpu_report.html

# Compare web vs native performance from database
python duckdb_api/core/benchmark_db_query.py --sql "SELECT * FROM cross_platform_performance WHERE model_name='bert-base-uncased'" --format html

# Compare simulation vs real browser results
python duckdb_api/core/benchmark_db_query.py --report simulation_vs_real --format html --output comparison.html
```

### WebNN and WebGPU Benchmarking Tools (ENHANCED - March 7, 2025)

The framework now includes comprehensive tools for benchmarking real WebNN and WebGPU implementations in browsers with clear distinction between real hardware acceleration and simulation mode:

```bash
# Run WebGPU benchmarks with Chrome
python benchmark_real_webnn_webgpu.py --webgpu --chrome

# Run WebNN benchmarks with Edge (best WebNN support)
python benchmark_real_webnn_webgpu.py --webnn --edge

# Run audio model benchmarks with Firefox (best for compute shaders)
python benchmark_real_webnn_webgpu.py --audio --firefox

# Benchmark with quantization (8-bit)
python benchmark_real_webnn_webgpu.py --text --bits 8

# Benchmark with mixed precision (4-bit)
python benchmark_real_webnn_webgpu.py --text --bits 4 --mixed-precision

# Run comprehensive benchmarks across multiple models
python benchmark_real_webnn_webgpu.py --comprehensive

# Store results in database
python benchmark_real_webnn_webgpu.py --text --db-path ./benchmark_db.duckdb

# Generate HTML report
python benchmark_real_webnn_webgpu.py --text --output-format html

# Check browser capabilities for WebNN/WebGPU support
python check_browser_webnn_webgpu.py --browser firefox

# Fix WebNN/WebGPU benchmarking issues
python fix_real_webnn_webgpu_benchmarks.py --browser chrome --fix-all
```

### NEW: IPFS Acceleration with Real WebNN/WebGPU Tool

A comprehensive new tool that tests IPFS acceleration with real WebNN/WebGPU hardware:

```bash
# Test all browsers and platforms
python test_ipfs_accelerate_with_real_webnn_webgpu.py --comprehensive

# Test specific browser and platform
python test_ipfs_accelerate_with_real_webnn_webgpu.py --browser firefox --platform webgpu --model bert-base-uncased

# Enable Firefox audio optimizations for audio models
python test_ipfs_accelerate_with_real_webnn_webgpu.py --browser firefox --model whisper-tiny --optimize-audio
```

### NEW: Diagnostic and Repair Tool for WebNN/WebGPU

A diagnostic tool that helps fix issues related to real WebNN/WebGPU implementations:

```bash
# Test if real WebGPU implementation is available in Chrome
python fix_real_webnn_webgpu_benchmarks.py --browser chrome --platform webgpu --validate-only

# Fix WebNN implementation in Edge
python fix_real_webnn_webgpu_benchmarks.py --browser edge --platform webnn --model bert

# Fix and optimize Firefox implementation for audio models
python fix_real_webnn_webgpu_benchmarks.py --browser firefox --platform webgpu --model whisper --optimize-audio
```

### Browser-Specific Optimizations

Different browsers excel at different tasks:

| Browser | Best For | Features | Command Flag |
|---------|----------|----------|-------------|
| Firefox | Audio models | 20-25% better performance for Whisper, CLAP | `--browser firefox --optimize-audio` |
| Edge | WebNN models | Superior WebNN implementation | `--browser edge --platform webnn` |
| Chrome | Vision models | Solid all-around WebGPU support | `--browser chrome --platform webgpu` |

The WebNN/WebGPU system includes:
- Robust WebSocket bridge with auto-reconnection and error handling
- Browser-specific optimizations (Firefox for audio models, Edge for WebNN)
- Comprehensive benchmarking across multiple models, batch sizes, and precision levels
- Clear distinction between real hardware acceleration and simulation mode
- Database integration for result storage and analysis
- Performance optimization support for WebNN and WebGPU

For detailed instructions, see:
- [WebNN/WebGPU Benchmark System](WEBNN_WEBGPU_BENCHMARK_README.md)
- [Real WebNN/WebGPU Implementation Update](REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md)

### Real WebNN and WebGPU Implementations (COMPLETED - March 6, 2025)

The framework now includes full REAL browser-based implementations for WebNN and WebGPU with these features:

- Direct browser-to-Python communication using WebSockets and Selenium
- Real-time hardware capability detection with browser automation
- Cross-browser support (Chrome, Firefox, Edge, Safari)
- transformers.js integration for hardware-accelerated inference
- Comprehensive error handling and fallbacks when hardware is unavailable
- Transparent feature detection and optimization selection
- Shader precompilation for faster startup
- Compute shader optimization for audio models
- Browser-specific optimizations (particularly Firefox for audio models)

```bash
# Run WebGPU verification to check real implementation status
python verify_webnn_webgpu_implementation.py --output verification_report.md

# Test real WebGPU implementation with Chrome
python implement_real_webnn_webgpu.py --browser chrome --platform webgpu --inference

# Test real WebNN implementation with Edge (best WebNN support)
python implement_real_webnn_webgpu.py --browser edge --platform webnn --inference
```

### March 2025 Web Platform Optimizations

The March 2025 release includes three major optimizations for web platform models:

```bash
# 1. WebGPU Compute Shader Optimization for Audio Models
# Firefox shows ~20% better performance than Chrome for audio models
# Test with various audio models
python generators/runners/web/test_web_platform_optimizations.py --compute-shaders --model whisper
python generators/runners/web/test_web_platform_optimizations.py --compute-shaders --model wav2vec2
python generators/runners/web/test_web_platform_optimizations.py --compute-shaders --model clap

# Enable via environment variable
export WEBGPU_COMPUTE_SHADERS_ENABLED=1
python web_platform_benchmark.py --model whisper

# Firefox-specific optimizations (uses 256x1x1 workgroup vs Chrome's 128x2x1)
./run_web_platform_tests.sh --firefox --enable-compute-shaders --model whisper

# Compare Firefox vs Chrome with various audio durations
python test_firefox_webgpu_compute_shaders.py --model whisper --audio-durations 5,15,30,60

# Direct API access to Firefox optimized compute shaders
from fixed_web_platform.webgpu_audio_compute_shaders import optimize_for_firefox

# 2. Parallel Model Loading for Multimodal Models
# Test with various multimodal models
python generators/runners/web/test_web_platform_optimizations.py --parallel-loading --model clip
python generators/runners/web/test_web_platform_optimizations.py --parallel-loading --model llava
python test_webgpu_parallel_model_loading.py --model-type multimodal

# Enable via environment variable
export WEB_PARALLEL_LOADING_ENABLED=1
python web_platform_benchmark.py --model clip

# 3. Shader Precompilation for Faster Startup
# Test with any WebGPU model
python generators/runners/web/test_web_platform_optimizations.py --shader-precompile --model bert
python generators/runners/web/test_web_platform_optimizations.py --shader-precompile --model vit

# Enable via environment variable
export WEBGPU_SHADER_PRECOMPILE_ENABLED=1
python web_platform_benchmark.py --model bert

# Testing all optimizations together
python generators/runners/web/test_web_platform_optimizations.py --all-optimizations
./run_web_platform_integration_tests.sh --all-optimizations --model clap

# Model-specific optimization recommendations
# For Text Models (BERT, T5, etc.)
./run_web_platform_integration_tests.sh --model bert --enable-shader-precompile

# For Vision Models (ViT, ResNet, etc.)
./run_web_platform_integration_tests.sh --model vit --enable-shader-precompile

# For Audio Models (Whisper, Wav2Vec2, CLAP)
# Firefox performs ~20% better than Chrome for audio models
./run_web_platform_integration_tests.sh --firefox --model whisper --enable-compute-shaders --enable-shader-precompile

# For Multimodal Models (CLIP, LLaVA, XCLIP)
./run_web_platform_integration_tests.sh --model clip --enable-parallel-loading --enable-shader-precompile

# For Audio-Multimodal Models (CLAP)
# Firefox shows ~21% better performance than Chrome for CLAP
./run_web_platform_integration_tests.sh --firefox --model clap --all-optimizations

# Compare Firefox vs Chrome browser performance
./run_web_platform_tests.sh --compare-browsers --model whisper

# Test WebNN and WebGPU with different quantization levels
python run_real_webgpu_webnn_fixed.py --platform webgpu --model bert-base-uncased --model-type text --bits 8
python run_real_webgpu_webnn_fixed.py --platform webnn --model bert-base-uncased --model-type text --bits 4 --mixed-precision

# Run comprehensive quantization tests for all high priority models
./test_webnn_webgpu_models_fixed.sh
```

### QNN (Qualcomm Neural Networks) Support and Advanced Quantization (March 2025)
```bash
# Generate tests for QNN hardware
python generators/qualified_test_generator.py -g bert-base-uncased -p qnn -o test_bert_qnn.py

# Run tests on QNN hardware
python test_bert_qnn.py

# Run comprehensive QNN integration test suite (stores results in DuckDB)
python test_qnn_integration.py --db-path ./benchmark_db.duckdb

# Run test suite with specific models
python test_qnn_integration.py --models BAAI/bge-small-en-v1.5,prajjwal1/bert-tiny

# Run test suite with comprehensive model set
python test_qnn_integration.py --models all

# Generate QNN performance visualizations from test data
python duckdb_api/visualization/visualize_qnn_performance.py --db-path ./benchmark_db.duckdb --output ./reports

# Automated hardware selection including QNN
python generators/hardware/automated_hardware_selection.py --model bert-base-uncased --include-qnn

# Benchmark with QNN hardware
python duckdb_api/core/benchmark_all_key_models.py --hardware qnn

# Test power efficiency metrics for mobile/edge devices (QNN)
python test_hardware_backend.py --backend qnn --model bert-tiny --power-metrics

# Compare QNN vs other hardware platforms using DuckDB data
python duckdb_api/core/benchmark_db_query.py --report qnn_comparison --format html --output qnn_report.html

# Extract device and SDK information for QNN
python test_qnn_integration.py --device-info-only

# Basic Quantization Usage
# ========================

# Quantize a model for QNN hardware
python qnn_quantization_support.py quantize \
  --model-path models/bert-base-uncased.onnx \
  --output-path models/bert-base-uncased.qnn \
  --method int8 \
  --model-type text

# Compare different quantization methods
python qnn_quantization_support.py compare \
  --model-path models/bert-base-uncased.onnx \
  --output-dir ./quantized_models \
  --model-type text \
  --report-path ./reports/quantization_comparison.md

# List available quantization methods for QNN
python qnn_quantization_support.py list

# Run a complete quantization example
python test_examples/qnn_quantization_example.py \
  --model-path models/bert-base-uncased.onnx \
  --model-type text \
  --mock

# Advanced Quantization Methods (March 2025)
# =========================================

# Weight Clustering Quantization
python qnn_advanced_quantization.py cluster \
  --model-path models/bert-base-uncased.onnx \
  --output-path models/bert-base-uncased-clustered.qnn \
  --clusters 16 \
  --model-type text \
  --optimize-for hexagon

# Hybrid/Mixed Precision Quantization
python qnn_advanced_quantization.py hybrid \
  --model-path models/llama-7b.onnx \
  --output-path models/llama-7b-hybrid.qnn \
  --attention-precision int8 \
  --feedforward-precision int4 \
  --model-type text_generation \
  --optimize-for mobile

# Per-Channel Quantization
python qnn_advanced_quantization.py per-channel \
  --model-path models/clip-vit.onnx \
  --output-path models/clip-vit-perchannel.qnn \
  --model-type vision

# Learned Quantization Parameters (QAT)
python qnn_advanced_quantization.py qat \
  --model-path models/bert-base-uncased.onnx \
  --output-path models/bert-base-uncased-qat.qnn \
  --train-dataset glue/mrpc \
  --epochs 3 \
  --learning-rate 5e-5 \
  --model-type text

# Sparse Quantization with Pruning
python qnn_advanced_quantization.py sparse \
  --model-path models/whisper-small.onnx \
  --output-path models/whisper-small-sparse.qnn \
  --sparsity 0.5 \
  --pruning-method magnitude \
  --model-type audio

# Method Comparison Framework
python quantization_comparison_tools.py compare-all \
  --model-path models/bert-base-uncased.onnx \
  --output-dir ./comparison_results \
  --methods int8,int4,cluster,hybrid,sparse \
  --metrics accuracy,latency,power,size \
  --model-type text

# Generate Quantization Impact Visualization
python quantization_comparison_tools.py visualize \
  --results-path ./comparison_results/bert-base-uncased-comparison.json \
  --output-path ./visualization/bert-quantization-impact.html \
  --plot-type radar

# Hardware-Specific Optimizations for Quantized Models
python qnn_hardware_optimizations.py optimize \
  --model-path models/bert-base-uncased-int8.qnn \
  --output-path models/bert-base-uncased-int8-optimized.qnn \
  --device sm8550 \
  --optimize memory,power,latency

# Memory Bandwidth Optimization
python qnn_hardware_optimizations.py memory-optimize \
  --model-path models/llama-7b-int4.qnn \
  --output-path models/llama-7b-int4-memopt.qnn \
  --cache-config aggressive \
  --tiling-strategy optimal

# Power State Management Integration
python qnn_hardware_optimizations.py power-optimize \
  --model-path models/whisper-small-int8.qnn \
  --output-path models/whisper-small-int8-poweropt.qnn \
  --battery-mode efficient \
  --dynamic-scaling enabled
```

### Distributed Training Configuration
```bash
# Generate distributed training configuration
python hardware_selector.py --model-family text_generation --model-name t5-small --mode training --distributed --gpu-count 4

# Generate training benchmark configuration for a model
python run_training_benchmark.py --model bert-base-uncased --distributed --max-gpus 4 --output bert_benchmark.json

# List available sample models for benchmarking
python run_training_benchmark.py --list-models

# Generate a memory-optimized training configuration
python hardware_selector.py --model-family text_generation --model-name llama-7b --mode training --distributed --gpu-count 8 --max-memory-gb 24
```

### Model Benchmarking with Template-Based Generation
```bash
# Run comprehensive benchmarks for all 300+ models using database templates
python duckdb_api/core/benchmark_all_key_models.py --all-models --use-db-templates

# Run benchmarks for a specific model using database templates
python duckdb_api/core/benchmark_all_key_models.py --model bert --use-db-templates

# Run benchmarks for all models in a family using database templates
python duckdb_api/core/benchmark_all_key_models.py --family text-embedding --use-db-templates

# Create a new benchmark template and store in database
python generators/template_database.py --create-benchmark-template --model-type llama --store-in-db

# Run standard model benchmarks with database integration and templates
python generators/benchmark_generators/run_model_benchmarks.py --models bert,t5,vit --use-db-templates --db-path ./benchmark_db.duckdb

# Generate benchmarks for all 300+ models (results stored directly in database)
python generators/benchmark_generators/run_model_benchmarks.py --generate-all --use-db-templates --db-path ./benchmark_db.duckdb
```

### Traditional Model Benchmarking and Validation
```bash
# Run comprehensive benchmarks for all 13 high-priority models across all hardware platforms
python duckdb_api/core/benchmark_all_key_models.py --output-dir ./benchmark_results

# Use smaller model variants for faster testing
python duckdb_api/core/benchmark_all_key_models.py --small-models --output-dir ./benchmark_results

# Test specific hardware platforms
python duckdb_api/core/benchmark_all_key_models.py --hardware cpu cuda openvino --output-dir ./benchmark_results

# Automatically fix implementation issues
python duckdb_api/core/benchmark_all_key_models.py --debug --output-dir ./benchmark_results

# Run standard model benchmarks with database integration
python generators/benchmark_generators/run_model_benchmarks.py --output-dir ./benchmark_results --db-path ./benchmark_db.duckdb

# Test on specific hardware platforms with small model set
python generators/benchmark_generators/run_model_benchmarks.py --hardware cpu cuda --models-set small --db-path ./benchmark_db.duckdb

# Run benchmarks without storing in database
python generators/benchmark_generators/run_model_benchmarks.py --hardware cpu --models-set small --no-db-store

# Generate database visualizations from benchmark results
python generators/benchmark_generators/run_model_benchmarks.py --hardware cuda --visualize-from-db

# Manual model functionality verification
python verify_model_functionality.py --models bert t5 vit --hardware cpu cuda

# Run detailed hardware benchmarks
python hardware_benchmark_runner.py --model-families embedding text_generation --hardware cpu cuda
```

### Benchmark Database and Result Management
```bash
# Set the database path environment variable (recommended)
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# JSON output is deprecated and now disabled by default
# All results are stored directly in the database

# Update database schema to add simulation flags
python duckdb_api/schema/update_db_schema_for_simulation.py

# Check QNN simulation status
python duckdb_api/utils/qnn_simulation_helper.py --check

# Enable QNN simulation (for testing only)
python duckdb_api/utils/qnn_simulation_helper.py --enable

# Disable QNN simulation
python duckdb_api/utils/qnn_simulation_helper.py --disable

# Migrate existing JSON files to the database 
python duckdb_api/migration/migrate_all_json_files.py --db-path ./benchmark_db.duckdb --archive

# Migrate and archive all JSON files (keeps archives)
python duckdb_api/migration/migrate_all_json_files.py --db-path ./benchmark_db.duckdb --archive --archive-dir ./archived_json_files

# Migrate all JSON files and delete them after successful migration and archiving
python duckdb_api/migration/migrate_all_json_files.py --db-path ./benchmark_db.duckdb --delete

# Convert existing benchmark JSON files to DuckDB format
python duckdb_api/migration/benchmark_db_converter.py --input-dir ./archived_test_results

# Consolidate test results across directories
python duckdb_api/migration/benchmark_db_converter.py --consolidate --categories performance hardware compatibility

# Comprehensive data migration with validation and deduplication
python duckdb_api/migration/benchmark_db_converter.py --consolidate --deduplicate --directories archived_test_results benchmark_results critical_model_results hardware_fix_results api_check_results

# Archive JSON files after migration to DuckDB
tar -czf archived_json_files/archived_test_results_$(date +%Y%m%d).tar.gz archived_test_results/*.json

# Create initial database schema with sample data
python duckdb_api/schema/creation/create_benchmark_schema.py --sample-data

# Database maintenance and optimization
python duckdb_api/core/benchmark_db_maintenance.py --optimize-db --vacuum

# Create database backup with compression
python duckdb_api/core/benchmark_db_maintenance.py --backup --backup-dir ./db_backups --backup-compress

# Check database integrity
python duckdb_api/core/benchmark_db_maintenance.py --check-integrity

# Generate migration statistics report
python duckdb_api/core/benchmark_db_maintenance.py --migration-stats --output migration_report.json

# Purge old database backups based on retention policy
python duckdb_api/core/benchmark_db_maintenance.py --purge-backups --backup-retention 30 --backup-dir ./db_backups

# Query benchmark database with SQL
python duckdb_api/core/benchmark_db_query.py --sql "SELECT model_name, hardware_type, AVG(throughput_items_per_second) FROM performance_results JOIN models USING(model_id) JOIN hardware_platforms USING(hardware_id) GROUP BY model_name, hardware_type"

# Generate reports from DuckDB benchmark database
python duckdb_api/core/benchmark_db_query.py --report performance --format html --output benchmark_report.html
python duckdb_api/core/benchmark_db_query.py --report hardware --format html --output hardware_report.html
python duckdb_api/core/benchmark_db_query.py --report compatibility --format html --output compatibility_matrix.html

# Compare hardware platforms for a specific model
python duckdb_api/visualization/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware --output bert_hardware_comparison.png

# Compare models on a specific hardware platform
python duckdb_api/visualization/benchmark_db_query.py --hardware cuda --metric throughput --compare-models --output cuda_model_comparison.png

# Plot performance trends over time
python duckdb_api/visualization/benchmark_db_query.py --trend performance --model bert-base-uncased --hardware cuda --metric throughput --format chart

# Export data from the database
python duckdb_api/core/benchmark_db_query.py --sql "SELECT * FROM performance_results" --format csv --output performance_data.csv

# Run benchmarks (results stored directly in database)
python duckdb_api/core/run_benchmark_with_db.py --model bert-base-uncased --hardware cuda --batch-sizes 1,2,4,8,16

# Run standard model benchmarks (results stored directly in database)
python generators/benchmark_generators/run_model_benchmarks.py --models bert-base-uncased,t5-small --hardware cuda

# Run CI/CD benchmark workflow manually via GitHub CLI
gh workflow run benchmark_db_ci.yml --ref main -f test_model=bert-base-uncased -f hardware=cpu -f batch_size=1,2,4,8

# Run IPFS accelerate tests with database integration
python generators/models/test_ipfs_accelerate.py --db-path ./benchmark_db.duckdb

# Generate a test report from the DuckDB database
python generators/models/test_ipfs_accelerate.py --report --format markdown --output test_report.md

# Use the Predictive Performance System to predict metrics without running actual benchmarks
python predictive_performance/run_predictive_performance_demo.py --model bert-base-uncased --hardware cuda,openvino,webgpu --visualize

# Predict performance for an untested model-hardware combination
python -m predictive_performance.predict --model t5-small --hardware cuda --batch-size 8 --detailed-output

# Schedule benchmarks based on active learning recommendations
python duckdb_api/run_benchmark_with_db.py --from-recommendations predictive_performance/recommendations.json
```

#### DuckDB Test Results Schema

Our DuckDB database schema has been enhanced to store detailed test results and hardware metrics:

```sql
-- Main test results table
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    test_date VARCHAR,
    status VARCHAR,
    test_type VARCHAR,
    model_name VARCHAR,
    endpoint_type VARCHAR,
    hardware_type VARCHAR,
    success BOOLEAN,
    error_message VARCHAR,
    execution_time FLOAT,
    memory_usage FLOAT,
    power_consumption FLOAT,       -- Added for mobile/edge devices
    temperature FLOAT,             -- Added for thermal monitoring
    qnn_version VARCHAR,           -- Qualcomm Neural Network SDK version
    sdk_type VARCHAR,              -- QNN or QTI SDK type
    details JSON
);

-- Hardware capability tracking
CREATE TABLE IF NOT EXISTS hardware_capabilities (
    id INTEGER PRIMARY KEY,
    hardware_type VARCHAR,
    device_name VARCHAR,
    compute_units INTEGER,
    memory_capacity FLOAT,
    driver_version VARCHAR,
    supported_precisions JSON,     -- FP32, FP16, INT8, INT4 support
    max_batch_size INTEGER,
    throughput_benchmark FLOAT,
    latency_benchmark FLOAT,
    power_efficiency FLOAT,        -- Important for mobile/edge
    detected_at TIMESTAMP
);

-- Model conversion metrics
CREATE TABLE IF NOT EXISTS model_conversion_metrics (
    id INTEGER PRIMARY KEY,
    model_name VARCHAR,
    source_format VARCHAR,
    target_format VARCHAR,
    hardware_target VARCHAR,
    conversion_success BOOLEAN,
    conversion_time FLOAT,
    file_size_before FLOAT,
    file_size_after FLOAT,
    precision VARCHAR,
    optimization_level INTEGER,
    error_message VARCHAR,
    timestamp TIMESTAMP
);

-- Performance comparison 
CREATE TABLE IF NOT EXISTS performance_comparison (
    id INTEGER PRIMARY KEY,
    model_name VARCHAR,
    test_id INTEGER,
    test_date TIMESTAMP,
    hardware_type VARCHAR,
    batch_size INTEGER,
    sequence_length INTEGER,
    latency_ms FLOAT,
    throughput_items_per_sec FLOAT,
    memory_mb FLOAT,
    power_watts FLOAT,            -- Added for mobile/edge
    energy_efficiency_items_per_joule FLOAT,
    performance_score FLOAT        -- Composite metric
);

-- Cross-platform compatibility matrix
CREATE TABLE IF NOT EXISTS cross_platform_compatibility (
    id INTEGER PRIMARY KEY,
    model_name VARCHAR,
    model_type VARCHAR,
    model_size VARCHAR,
    cpu_support BOOLEAN,
    cuda_support BOOLEAN,
    rocm_support BOOLEAN,
    mps_support BOOLEAN,
    openvino_support BOOLEAN,
    qnn_support BOOLEAN,          -- Qualcomm Neural Networks support
    webnn_support BOOLEAN,
    webgpu_support BOOLEAN,
    recommended_platform VARCHAR, 
    last_updated TIMESTAMP
);
```

For working with the schema:

```bash
# Query hardware capabilities
python duckdb_api/benchmark_db_query.py --sql "SELECT * FROM hardware_capabilities" --format html --output capabilities.html

# Check cross-platform compatibility by model type
python duckdb_api/benchmark_db_query.py --sql "SELECT model_type, COUNT(*) as total, SUM(CASE WHEN qnn_support THEN 1 ELSE 0 END) as qnn_compatible, ROUND(SUM(CASE WHEN qnn_support THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as compatibility_rate FROM cross_platform_compatibility GROUP BY model_type ORDER BY compatibility_rate DESC" --format markdown

# Compare power efficiency across hardware platforms
python duckdb_api/benchmark_db_query.py --sql "SELECT hardware_type, AVG(energy_efficiency_items_per_joule) as avg_efficiency FROM performance_comparison GROUP BY hardware_type ORDER BY avg_efficiency DESC" --format chart --output power_efficiency.png
```

## Benchmark System and Simulation Detection Tools (ADDED - April 6, 2025)

The framework now includes comprehensive tools for benchmark management, validation, and simulation detection:

```bash
# Update database schema to include simulation flags
python duckdb_api/update_db_schema_for_simulation.py --db-path ./benchmark_db.duckdb

# Check simulation status in database
python duckdb_api/view_benchmark_results.py --check-simulation

# Generate a benchmark summary with simulation status indicators
python duckdb_api/view_benchmark_results.py --output benchmark_summary.md

# Scan for problematic reports that may contain misleading data
python duckdb_api/cleanup_stale_reports.py --scan

# Mark problematic reports with clear warnings
python duckdb_api/cleanup_stale_reports.py --mark

# Archive problematic files
python duckdb_api/cleanup_stale_reports.py --archive

# Fix report generator scripts to include validation
python duckdb_api/cleanup_stale_reports.py --fix-report-py

# Run benchmarks with explicit simulation for unavailable hardware
python duckdb_api/run_benchmark_with_db.py --model bert-base-uncased --hardware rocm --batch-sizes 1,2 --simulate

# View performance results from database with simulation status
python duckdb_api/view_benchmark_results.py

# Generate CSV report with all benchmark data
python duckdb_api/view_benchmark_results.py --format csv --output benchmark_data.csv
```

Key documentation:
- [Simulation Detection Improvements Guide](SIMULATION_DETECTION_IMPROVEMENTS_GUIDE.md): Detailed documentation of simulation detection enhancements
- [Benchmark Database Fix Guide](BENCHMARK_DB_FIX.md): Summary of database fixes and improvements

## Distributed Testing Framework (NEW - May 2025)

The framework now includes a high-performance distributed testing system that enables parallel execution of benchmarks and tests across multiple machines with heterogeneous hardware. This system provides intelligent workload distribution and centralized result aggregation.

### Key Features

- **Coordinator-Worker Architecture**: Central coordinator server distributes tasks to worker nodes
- **DuckDB Integration**: Centralized storage of distributed test results
- **Security**: Comprehensive JWT-based authentication and message signing
- **Intelligent Task Distribution**: Routes tasks to worker nodes with appropriate hardware
- **Resource Monitoring**: Tracks worker node health, capabilities, and resource usage
- **Fault Tolerance**: Automatic task retry and worker node recovery
- **Scalability**: Supports dynamic addition and removal of worker nodes

### Running the Distributed Testing Framework

```bash
# Start the coordinator (central server)
python duckdb_api/distributed_testing/coordinator.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb

# Start a worker node
python duckdb_api/distributed_testing/worker.py --coordinator http://localhost:8080 --api-key WORKER_API_KEY

# Generate API keys for authentication
python duckdb_api/distributed_testing/coordinator.py --generate-worker-key --security-config ./security_config.json

# Run a test using the distributed framework
python duckdb_api/distributed_testing/run_test.py --mode all --db-path ./test_db.duckdb --security-config ./test_security_config.json
```

### Creating Tasks for Distributed Execution

```bash
# Create a benchmark task with specific requirements
python duckdb_api/distributed_testing/create_task.py --type benchmark --model bert-base-uncased \
  --hardware cuda --batch-sizes 1,2,4,8,16 --priority 1

# Create a test task
python duckdb_api/distributed_testing/create_task.py --type test --test-file test_webgpu_4bit_inference.py \
  --hardware webgpu --browser firefox --priority 2

# Monitor task execution
python duckdb_api/distributed_testing/monitor_tasks.py --status all
```

### Security Features

The distributed testing framework includes comprehensive security features:

- **API Key Authentication**: Initial registration with API keys
- **JWT Token Authentication**: Ongoing secure communication with short-lived tokens
- **Message Signing**: All WebSocket messages signed with HMAC
- **Role-Based Access Control**: Different permission levels for workers and admins

For detailed documentation on the distributed testing framework, see:
- [DISTRIBUTED_TESTING_DESIGN.md](DISTRIBUTED_TESTING_DESIGN.md) - Detailed design document
- [duckdb_api/distributed_testing/README.md](duckdb_api/distributed_testing/README.md) - Usage instructions
- [duckdb_api/distributed_testing/SECURITY.md](duckdb_api/distributed_testing/SECURITY.md) - Security implementation

## Web Resource Pool Integration (COMPLETED - May 10, 2025)

The WebGPU/WebNN Resource Pool Integration enables concurrent execution of multiple AI models across heterogeneous browser backends. It dramatically improves throughput, reduces resource waste, and provides fine-grained control over browser-based hardware acceleration resources.

### Key Features

- **Concurrent Model Execution**: Run multiple models simultaneously (3.5x throughput improvement)
- **Connection Pooling**: Efficiently manage browser connections with lifecycle management
- **Browser-Aware Load Balancing**: Distribute models to optimal browsers based on model type
- **Adaptive Resource Scaling**: Dynamically adjust resource allocation based on demand
- **Real-Time Monitoring**: Track resource utilization and performance metrics

### Using the Resource Pool

```python
# Create resource pool integration
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration

integration = ResourcePoolBridgeIntegration(
    max_connections=4,
    browser_preferences={
        'audio': 'firefox',     # Firefox for audio models
        'vision': 'chrome',     # Chrome for vision models
        'text_embedding': 'edge' # Edge for embedding models
    },
    adaptive_scaling=True
)

# Initialize the integration
integration.initialize()

# Get model from resource pool
model = integration.get_model(
    model_type='text_embedding',
    model_name='bert-base-uncased',
    hardware_preferences={'priority_list': ['webgpu', 'cpu']}
)

# Run inference
result = model(inputs)
```

### Running Tests

```bash
# Test resource pool with multiple models
python test_web_resource_pool.py --models bert,vit,whisper

# Test concurrent model execution
python test_web_resource_pool.py --concurrent-models --models bert,vit,whisper

# Run stress test with high concurrency
python test_web_resource_pool.py --stress-test --duration 120
```

For detailed documentation, see:
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Comprehensive guide
- [WEBNN_WEBGPU_DATABASE_INTEGRATION.md](WEBNN_WEBGPU_DATABASE_INTEGRATION.md) - Database integration details

## Mobile and Edge Support (COMPLETED - April 6, 2025)

The framework now offers comprehensive support for mobile and edge devices, enabling efficient deployment of AI models across different mobile hardware platforms including Qualcomm Snapdragon, MediaTek Dimensity, and Samsung Exynos processors.

### Key Features

- **Mobile Hardware Support**: Optimized integration with mobile AI accelerators (Qualcomm, MediaTek, Samsung)
- **Power and Thermal Metrics**: Detailed power consumption, battery impact, and thermal throttling analysis
- **Mobile-Optimized Models**: Hardware-specific optimizations for mobile deployment
- **Database Integration**: Complete metrics integration with DuckDB for unified performance tracking
- **Cross-Platform Comparison**: Compare mobile vs desktop hardware performance

### Database Schema Extensions

The database schema has been extended to include mobile-specific metrics:

```sql
-- Main mobile metrics table
CREATE TABLE mobile_edge_metrics (
    id INTEGER PRIMARY KEY,
    performance_id INTEGER,
    device_model VARCHAR,
    battery_impact_percent FLOAT,
    thermal_throttling_detected BOOLEAN,
    soc_temperature_celsius FLOAT,
    power_efficiency_score FLOAT,
    FOREIGN KEY (performance_id) REFERENCES performance_results(id)
);
```

### Running Mobile Tests

```bash
# Collect mobile metrics for a model (simulation mode)
python mobile_edge_device_metrics.py collect --model bert-base-uncased --device "Snapdragon 8 Gen 3" --simulate

# Generate battery impact report
python mobile_edge_device_metrics.py report --format html --output battery_impact.html

# Run tests on Samsung Exynos hardware
python samsung_support.py test --model bert-base-uncased --precision int8 --one-ui-optimization
```

### Mobile Performance Comparison

Based on comprehensive benchmarking, the following relative performance has been observed:

| Hardware | BERT | CLIP | Whisper | LLAMA |
|----------|------|------|---------|-------|
| Qualcomm | 3.9x | 4.0x | 3.5x | 2.5x |
| MediaTek | 3.5x | 4.7x | 3.0x | 2.2x |
| Samsung | 4.3x | 3.8x | 2.8x | 2.0x |

*Values indicate throughput relative to mobile CPU (higher is better)*

### Battery Impact Analysis

The battery impact varies by model and hardware:

| Hardware | BERT | CLIP | Whisper | LLAMA |
|----------|------|------|---------|-------|
| Qualcomm | 3.0% | 3.2% | 4.5% | 8.5% |
| MediaTek | 3.2% | 3.0% | 4.8% | 9.0% |
| Samsung | 2.8% | 3.4% | 5.0% | 8.8% |

*Values indicate battery percentage used per hour during continuous inference (lower is better)*

For complete documentation, see:
- [MOBILE_EDGE_SUPPORT_GUIDE.md](MOBILE_EDGE_SUPPORT_GUIDE.md) - Comprehensive mobile support guide
- [BATTERY_IMPACT_ANALYSIS.md](BATTERY_IMPACT_ANALYSIS.md) - Detailed battery impact methodology
- [SAMSUNG_NPU_SUPPORT_GUIDE.md](SAMSUNG_NPU_SUPPORT_GUIDE.md) - Samsung-specific optimizations

## Comprehensive Model Compatibility

The framework now includes a complete compatibility matrix for all 300+ HuggingFace model classes across all supported hardware platforms. This matrix is automatically generated from the DuckDB benchmark database.

### Compatibility Levels

| Symbol | Level | Description |
|--------|-------|-------------|
| ✅ | Full | Full support with optimal performance |
| ⚠️ | Limited | Works with limitations or reduced performance |
| 🔄 | Experimental | Implementation exists but not fully tested |
| ❌ | Not Supported | Implementation does not exist or does not work |

### Generated Matrix Examples

#### Text Models
| Model Class | CUDA | ROCm | MPS | OpenVINO | Qualcomm | WebNN | WebGPU | Notes |
|------------|------|------|-----|----------|----------|-------|--------|-------|
| BERT | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Full support across all platforms |
| LLAMA | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | Memory constraints on web platforms |

#### Advanced Quantization Support
| Model Class | Weight Clustering | Hybrid/Mixed | Per-Channel | QAT | Sparse |
|------------|-------------------|--------------|-------------|-----|--------|
| BERT | ✅ | ✅ | ✅ | ✅ | ✅ |
| ViT | ✅ | ✅ | ✅ | ✅ | ✅ |
| Whisper | ✅ | ✅ | ✅ | ✅ | ✅ |

### Generating the Matrix

```bash
# Generate the complete compatibility matrix
python generate_compatibility_matrix.py

# Generate matrix with specific filters
python generate_compatibility_matrix.py --filter vision --hardware cuda,qualcomm,webgpu

# Generate performance comparison for a specific model
python duckdb_api/core/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware
```

For complete documentation, see:
- [COMPREHENSIVE_MODEL_COMPATIBILITY_MATRIX.md](COMPREHENSIVE_MODEL_COMPATIBILITY_MATRIX.md) - Complete matrix
- [WEBNN_WEBGPU_COMPATIBILITY_MATRIX.md](WEBNN_WEBGPU_COMPATIBILITY_MATRIX.md) - Web-specific compatibility

## Documentation Index and Finding Information

For a complete overview of all available documentation, refer to:
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Comprehensive index of all project documentation with categorization

Major documentation categories include:
- [PHASE16_COMPLETION_REPORT.md](PHASE16_COMPLETION_REPORT.md) - Comprehensive report on the completed Phase 16 implementation
- [HARDWARE_BENCHMARKING_GUIDE.md](HARDWARE_BENCHMARKING_GUIDE.md) - Main hardware benchmarking documentation
- [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md) - Benchmark database architecture and usage
- [WEB_PLATFORM_INTEGRATION_GUIDE.md](WEB_PLATFORM_INTEGRATION_GUIDE.md) - Web platform integration guide
- [REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md](REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md) - Latest WebNN/WebGPU implementation
- [WEBNN_WEBGPU_BENCHMARK_README.md](WEBNN_WEBGPU_BENCHMARK_README.md) - Overview of WebNN/WebGPU benchmark system
- [WEBNN_WEBGPU_DATABASE_INTEGRATION.md](WEBNN_WEBGPU_DATABASE_INTEGRATION.md) - How WebNN/WebGPU integrates with DuckDB
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Resource pool integration with web platform
- [TEMPLATE_INHERITANCE_GUIDE.md](TEMPLATE_INHERITANCE_GUIDE.md) - Template inheritance system documentation
- [SIMULATION_DETECTION_IMPROVEMENTS.md](SIMULATION_DETECTION_IMPROVEMENTS.md) - Simulation detection and validation guide

### Documentation Cleanup and Maintenance

For guidance on documentation organization and maintenance:
- [DOCUMENTATION_CLEANUP_GUIDE.md](DOCUMENTATION_CLEANUP_GUIDE.md) - Guide for documentation and report cleanup procedures

Documentation cleanup tools:
```bash
# Archive old documentation files
python archive_old_documentation.py

# Scan for problematic benchmark reports
python cleanup_stale_reports.py --scan

# Run the complete documentation cleanup process
./run_documentation_cleanup.sh
```

## Performance Benchmarks

### Latest Performance Metrics

For detailed performance benchmarks, please refer to the following resources:
- Database dashboard: `http://localhost:8000/dashboard` (when running benchmark_db_api.py)
- API documentation: `http://localhost:8000/docs` (complete REST API for all benchmark data)
- Generated reports: 
  - `python duckdb_api/core/benchmark_db_query.py --report summary --format html --output summary_report.html`
  - `python duckdb_api/core/benchmark_db_query.py --compatibility-matrix --format html --output matrix.html`

Legacy documentation (being migrated to database):
- Hardware-specific benchmarks: `test/HARDWARE_BENCHMARKING_GUIDE.md`
- Model compression results: `test/MODEL_COMPRESSION_GUIDE.md`
- Training benchmarks: `test/TRAINING_BENCHMARKING_GUIDE.md`
- Web platform audio tests: `test/WEB_PLATFORM_AUDIO_TESTING_GUIDE.md`
- Hardware selection system: `test/HARDWARE_SELECTION_GUIDE.md`
- Web platform support: `test/README_WEB_PLATFORM_SUPPORT.md`
- QNN implementation: `test/QNN_IMPLEMENTATION_SUMMARY.md`

### QNN (Qualcomm Neural Networks) Performance

The QNN integration (March 2025) provides specialized support for Snapdragon SoCs and mobile/edge devices:

| Model Type | Model Size | QNN vs CPU | Power Efficiency | Key Metric |
|------------|------------|------------|------------------|------------|
| Embedding | Small | 2.5-3.8x faster | 4.0-5.5x better | 78% lower power consumption |
| Text Generation | Tiny (<1B) | 1.8-2.2x faster | 3.0-4.0x better | Optimal for battery life |
| Vision | Small-Medium | 3.0-5.0x faster | 3.5-4.5x better | Great for mobile vision |
| Audio | Tiny | 2.0-3.0x faster | 3.0-4.0x better | Suitable for voice assistants |
| Multimodal | Tiny-Small | 1.5-2.0x faster | 2.5-3.5x better | Limited by memory |

Performance varies by hardware generation and specific Snapdragon model. Benchmarks were conducted on Snapdragon 8 Gen 3 hardware with the latest QNN SDK (version 2.10).

**QNN Implementation Features:**
- Model conversion pipeline (PyTorch → ONNX → QNN format)
- Support for both QNN and QTI SDKs
- Power and thermal measurement capabilities
- Mobile-optimized inference settings
- Edge-aware batching and memory management
- Fallback mechanisms for unsupported operations
- Mock implementations for testing without physical hardware

For detailed QNN performance testing and reports, run:
```bash
# Run comprehensive QNN test suite and generate reports
python test_qnn_integration.py --models all
python duckdb_api/visualization/visualize_qnn_performance.py --output ./reports
```

### Web Platform Performance Results

The March 2025 enhancements have significantly improved web platform performance:

| Model Type | WebNN vs. CPU | WebGPU vs. CPU | WebGPU Standard | WebGPU March 2025 | Recommended Size |
|------------|--------------|----------------|-----------------|-------------------|------------------|
| BERT Embeddings | 2.0-3.0x faster | 2.2-3.4x faster | 2.2-3.4x faster | 2.4-3.6x faster | Small-Medium |
| Vision Models | 3.0-4.0x faster | 4.0-6.0x faster | 4.0-6.0x faster | 4.5-6.5x faster | Any size |
| Small T5 | 1.5-2.0x faster | 1.3-1.8x faster | 1.3-1.8x faster | 1.6-2.2x faster | Small |
| Tiny LLAMA | 1.0-1.2x faster | 1.2-1.5x faster | 1.2-1.5x faster | 1.4-1.9x faster | Tiny (<1B) |
| Audio Models | 0.8-1.2x CPU | 1.0-1.2x CPU | 1.0-1.2x CPU | 1.2-1.5x faster | Tiny-Small |

## Ultra-Low Precision Quantization (COMPLETED - August 2025)

The framework now includes fully optimized ultra-low precision (2-bit and 3-bit) quantization for WebGPU with comprehensive memory efficiency improvements and browser-specific optimizations.

### Key Features

- **Ultra-Low Precision**: Supports 2-bit, 3-bit, and 4-bit quantization with optimized WebGPU shaders
- **Memory-Efficient KV Cache**: 87.5% memory reduction with 2-bit and 81.25% with 3-bit quantization
- **Mixed Precision**: Adaptive precision for different model layers to balance accuracy and memory
- **Extended Context Windows**: 8x longer context with 2-bit quantization (4K → 32K tokens)
- **Browser-Specific Optimizations**: Specialized implementations for Chrome, Firefox, Edge, and Safari
- **Shader Precompilation**: 30-45% faster startup time with precompiled shaders

### Ultra-Low Precision Framework

```python
# Import from the fixed_web_platform package
from fixed_web_platform.webgpu_ultra_low_precision import setup_ultra_low_precision

# Set up 2-bit quantization with KV-cache optimization
result = setup_ultra_low_precision(
    model_name="llama-7b",
    model_type="text",
    precision_bits=2,
    mixed_precision=True,
    enable_kv_cache=True,
    extended_context=True,
    browser="chrome"
)

# Access configuration
config = result["ultra_low_precision"]
print(f"Memory reduction: {config['memory_reduction_percent']}%")
print(f"Extended context: {config['context_extension_factor']}x longer context")
```

### Browser Support Matrix

The implementation has been extensively tested across all major browsers:

| Browser | 2-bit | 3-bit | 4-bit | KV-Cache | Mixed Precision | Shader Precompilation |
|---------|-------|-------|-------|----------|-----------------|------------------------|
| Chrome | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Edge | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Firefox | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited |
| Safari | ❌ None | ✅ Limited | ✅ Full | ✅ Limited | ✅ Limited | ✅ Limited |

### Memory-Accuracy Tradeoffs

| Precision | Memory Reduction | Accuracy Impact | Best For |
|-----------|-----------------|----------------|---------|
| 2-bit | 87.5% | 5-8% | Memory-critical applications |
| 3-bit | 81.25% | 3-5% | Balanced applications |
| Mixed | 83-85% | 2-3% | Production applications |
| 4-bit | 75% | <2% | Accuracy-critical applications |

### WebNN and WebGPU Quantization Support (UPDATED - August 2025)

All high-priority HuggingFace model classes now support various quantization levels with WebNN and WebGPU:

| Quantization | Text Models | Vision Models | Audio Models | Multimodal Models |
|--------------|-------------|--------------|--------------|-------------------|
| 16-bit | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU |
| 8-bit | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU |
| 4-bit | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU | ✅ WebNN/WebGPU |
| 3-bit | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU |
| 2-bit | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU | ✅ WebGPU |
| Mixed Precision | ✅ Adaptive | ✅ Adaptive | ✅ Adaptive | ✅ Adaptive |
| Auto-Quantization | ✅ Dynamic | ✅ Dynamic | ✅ Dynamic | ✅ Dynamic |

**Optimal configurations**:
- Text Models (BERT, T5, LLAMA): WebNN with 8-bit quantization
- Vision Models (CLIP, ViT, DETR): WebGPU with 8-bit quantization
- Audio Models (Whisper, Wav2Vec2): WebGPU with compute shaders (Firefox preferred)
- Multimodal Models (LLaVA, XCLIP): WebGPU with parallel loading

For memory-constrained environments, 4-bit mixed precision provides the best balance between performance and model size.

For detailed compatibility information, see [WEBNN_WEBGPU_COMPATIBILITY_MATRIX.md](WEBNN_WEBGPU_COMPATIBILITY_MATRIX.md).

**March 2025 Optimization Details:**

1. **WebGPU Compute Shader Optimization for Audio Models**:
   - 20-35% performance improvement (43% in tests for Whisper)
   - Firefox-specific optimizations using 256x1x1 workgroup size vs Chrome's 128x2x1
   - Targeted at audio models (Whisper, Wav2Vec2, CLAP)
   - Implementation in `fixed_web_platform/webgpu_audio_compute_shaders.py`

2. **Parallel Loading for Multimodal Models**: 
   - 30-45% loading time reduction
   - Multiple model components loaded simultaneously
   - Especially effective for models with separate encoders (vision, text)
   - Implementation in `fixed_web_platform/progressive_model_loader.py`

3. **Shader Precompilation**:
   - 30-45% faster first inference
   - Precompiles shaders during model initialization
   - Most effective for vision models with complex shader pipelines
   - Implementation in `fixed_web_platform/webgpu_shader_precompilation.py`

**Current Implementation Status:**

| Feature | Status | Implementation | Browser Support |
|---------|--------|----------------|----------------|
| WebNN Core | ✅ Complete | Simulation + transformers.js | Chrome, Edge, Safari |
| WebGPU Core | ✅ Complete | Simulation + transformers.js | Chrome, Edge, Firefox, Safari (partial) |
| Compute Shader Optimization | ✅ Complete | Custom implementation | Chrome, Edge, Firefox (best) |
| Shader Precompilation | ✅ Complete | Custom implementation | Chrome, Edge, Safari (limited) |
| Parallel Model Loading | ✅ Complete | Custom implementation | All browsers |
| Resource Pool Integration | ✅ Complete | Shared connections | All browsers |
| Auto Browser Selection | ✅ Complete | Model-aware routing | Chrome, Edge, Firefox |
| 4-bit Quantization | ✅ Complete | Custom kernels | Chrome, Edge, Firefox |
| Auto-Quantization | ✅ Complete | Dynamic precision | All browsers |
| KV-Cache Optimization | 🔄 In Progress | Shared memory | Chrome, Edge |
| Cross-Browser Sharding | 🔄 In Progress | Multi-browser | Chrome, Edge, Firefox |
| Browser API Detection | ✅ Complete | Robust checks | All browsers |
| Graceful Fallbacks | ✅ Complete | Feature detection | All browsers |

**Browser Compatibility:**

| Browser | WebGPU Support | Compute Shaders | Parallel Loading | Shader Precompilation | 4-bit Quantization | Flash Attention |
|---------|---------------|-----------------|------------------|----------------------|-------------------|-----------------|
| Chrome | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Edge | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Firefox | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ✅ Full |
| Safari | ⚠️ Limited | ⚠️ Limited | ✅ Full | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |

For detailed web platform performance testing and reports, run:
```bash
# Run comprehensive tests for all optimizations
./run_web_platform_integration_tests.sh --all-models --all-optimizations

# Generate detailed performance report
python duckdb_api/core/benchmark_db_query.py --report web_platform --format html --output web_platform_report.html

# Generate optimization comparison chart
python duckdb_api/core/benchmark_db_query.py --report web_optimizations --format chart --output web_optimization_chart.png
```

See the [Web Platform Optimization Guide](WEB_PLATFORM_OPTIMIZATION_GUIDE.md) for implementation details and usage recommendations.

### August 2025 Web Platform Implementation Additions

The August 2025 update completes the web platform implementation with:

- **Unified Framework Integration**: Standardized API across all platform components
- **Comprehensive Error Handling**: Graceful degradation with browser-specific recovery strategies
- **Configuration Validation System**: Auto-correction for invalid settings with browser compatibility checks
- **Model Sharding System**: Run large models by distributing across multiple browser tabs
- **Mobile Device Support**: Optimized configurations for mobile browsers

To use the unified framework:

```python
from fixed_web_platform.unified_framework import UnifiedWebPlatform

# Create platform with automatic browser detection
platform = UnifiedWebPlatform(
    model_name="llama-7b",
    model_type="text",
    platform="webgpu"
)

# Run inference with unified API (handles all browser compatibility)
result = platform.run_inference({"input_text": "Sample text"})
```

For model sharding across multiple browser tabs:

```python
from fixed_web_platform.unified_framework.model_sharding import ModelShardingManager

# Create model sharding manager
sharding_manager = ModelShardingManager(
    model_name="llama-7b",
    num_shards=4,
    shard_type="layer"
)

# Initialize sharding (opens browser tabs)
sharding_manager.initialize_sharding()

# Run inference across shards
result = sharding_manager.run_inference_sharded({"input_text": "Sample text"})
```

### April 2025 Memory Optimization Tools

To analyze memory usage and test cross-platform 4-bit inference:

```bash
# Visualize memory usage for models across platforms
python visualize_memory_usage.py --model llama --platform webgpu --output html

# Test cross-platform 4-bit inference compatibility and performance
python test_cross_platform_4bit.py --model llama --hardware cuda webgpu --output-report report.html

# Test WebGPU 4-bit inference with specialized matrix multiplication kernels
python test_webgpu_4bit_inference.py --model llama --all-tests
```

*Note: Performance varies significantly based on hardware, browser version, and model size.*

### Test and Template Database Architecture

The DuckDB/Parquet-based database system is now the primary storage for all benchmark results and templates (JSON output is deprecated). This system provides:

#### Template Database Schema
The database stores templates for tests, skills, benchmarks, and helper functions for 300+ HuggingFace models:
- **Template Tables**:
  - `templates`: Stores core templates indexed by model type and template type
  - `template_helpers`: Common helper functions shared across templates
  - `template_dependencies`: Maps dependencies between templates
  - `template_versions`: Tracks template versions and updates
  - `template_variables`: Defines substitution variables for templates

- **Template Categories**:
  - Test templates (for generating test files)
  - Skill templates (for generating skill implementation files)
  - Benchmark templates (for generating benchmark scripts)
  - Helper templates (shared utility functions)
  - Hardware-specific templates (platform-specific code)

- **Template Management Tools**:
  - `template_database.py`: Core template CRUD operations
  - `template_validator.py`: Validates template syntax and dependencies
  - `template_migration.py`: Migrates templates between versions
  - `template_inheritance.py`: Handles inheritance between templates
  - `template_instantiator.py`: Instantiates templates with model-specific values

### Locating Important Files and Components

#### Core Organizational Files
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md): Central documentation reference
- [PHASE16_COMPLETION_REPORT.md](PHASE16_COMPLETION_REPORT.md): Final report on Phase 16 implementation (completed)
- [README.md](README.md): Main project readme

#### Core Utility Files
- [utils.py](utils.py): Contains essential utility functions for the entire project
- [hardware_detection.py](hardware_detection.py): Detects available hardware platforms
- [benchmark_db_api.py](benchmark_db_api.py): REST API for the benchmark database
- [resource_pool.py](resource_pool.py): Manages hardware resources efficiently

#### Web Platform Directory Structure
- `fixed_web_platform/`: Contains WebNN and WebGPU implementations
  - `webgpu_audio_compute_shaders.py`: Optimized audio processing for Firefox
  - `websocket_bridge.py`: Communication bridge for browser tests
  - `resource_pool_bridge.py`: Resource management for parallel execution
  - `browser_capability_detection.py`: Detects browser WebNN/WebGPU capabilities
  - `progressive_model_loader.py`: Implements parallel model loading for multimodal models
  - `webgpu_shader_precompilation.py`: Shader precompilation for faster startup
  - `webgpu_4bit_inference.py`: Ultra-low precision inference implementation
  - `webgpu_quantization.py`: Quantization utilities for WebGPU models
  - `unified_framework/`: Unified API for cross-browser WebNN/WebGPU
    - `configuration_manager.py`: Manages WebNN/WebGPU configurations
    - `fallback_manager.py`: Handles graceful fallbacks when features are unsupported
    - `model_sharding.py`: Distributes model computation across multiple tabs
  - `wgsl_shaders/`: WebGPU Shading Language optimized shader implementations
    - `firefox_optimized_audio_whisper.wgsl`: Firefox-optimized shader for Whisper models
    - `model_specific/`: Model-specific optimized shader implementations

#### Generator System Directory Structure
- `generators/`: Main directory for all generation-related code (216 files)
  - `benchmark_generators/`: Benchmark generation tools
    - `run_model_benchmarks.py`: Main benchmark execution script
    - `benchmark_visualizer.py`: Tools for visualizing benchmark results
  - `models/`: Model implementations and skills
    - `test_ipfs_accelerate.py`: IPFS acceleration testing utilities
    - Model-specific implementation files
  - `runners/`: Test runner scripts
    - `web/`: Web-specific test runners
    - `run_template_system_check.py`: Template system verification
  - `skill_generators/`: Skill generation tools
    - `create_simple_template_db.py`: Creates template database with defaults
    - `integrated_skillset_generator.py`: Generates skills for models
  - `template_generators/`: Template generation utilities
    - `template_validator.py`: Validation system for templates
  - `templates/`: Template files and database
    - `template_database.py`: Database operations for templates
    - Template files for all model families (BERT, ViT, Whisper, LLaVA, etc.)
    - Template database files (template_database.json, template_db.duckdb)
    - Hardware-specific template variations
  - `test_generators/`: Test generation tools
    - `simple_test_generator.py`: Simplified template-based generator
    - `merged_test_generator.py`: Advanced test generator
    - `qualified_test_generator.py`: Generator with hardware qualification
  - `utils/`: Utility functions for generators
  - `hardware/`: Hardware-specific generator tools
    - `automated_hardware_selection.py`: Hardware selection utilities
    - `hardware_detection.py`: Detects available hardware platforms

#### Benchmark Results Database
The database also stores all benchmark results and test outputs:
- **Performance Improvements**:
  - 50-80% size reduction compared to JSON files
  - 5-20x faster queries for complex analysis
  - 70% less disk I/O for test result management
  - Parallel processing for batch data migration

- **Advanced Features**:
  - SQL-based querying with full JOIN support
  - Foreign key constraints for data integrity
  - Comprehensive schema for all test types
  - Time-series analysis of performance trends
  - Visualization tools for performance comparisons
  - REST API for programmatic access
  - Interactive dashboard for result exploration

- **Core Components**:
  - `duckdb_api/schema/creation/create_benchmark_schema.py`: Schema definition and initialization
  - `duckdb_api/migration/benchmark_db_converter.py`: JSON to database migration
  - `duckdb_api/core/benchmark_db_updater.py`: Direct database writing interface
  - `duckdb_api/core/benchmark_db_query.py`: Comprehensive query tool
  - `duckdb_api/core/benchmark_db_maintenance.py`: Database optimization
  - `duckdb_api/core/benchmark_db_api.py`: REST API and dashboard
  - `duckdb_api/core/benchmark_db_performance.py`: Performance testing
  - `duckdb_api/core/run_benchmark_with_db.py`: Example integration
  - `duckdb_api/migration/cleanup_test_results.py`: Automated migration utility
  - `duckdb_api/core/generate_compatibility_matrix.py`: Creates comprehensive model compatibility matrix
  - `duckdb_api/schema/update_db_schema_for_simulation.py`: Updates schema with simulation flags

#### Model Compatibility Matrix
The database enables automatic generation of a comprehensive compatibility matrix for all 300+ HuggingFace model classes:

- **Matrix Generation**:
  ```bash
  # Generate the complete compatibility matrix
  python generate_compatibility_matrix.py
  
  # Generate matrix with specific filters
  python generate_compatibility_matrix.py --filter vision --hardware cuda,qualcomm,webgpu
  
  # Custom output formats
  python generate_compatibility_matrix.py --format markdown --output custom_matrix.md
  ```

- **Matrix Features**:
  - Cross-platform compatibility status for all models
  - Visual indicators for compatibility levels
  - Hardware-specific performance metrics
  - Advanced quantization support indicators
  - Automatic updates via CI/CD pipeline
  - Filtering by model type and hardware platform
  - Custom output formats (markdown, HTML)

Documentation and guides:
- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [Database Migration Guide](DATABASE_MIGRATION_GUIDE.md)
- [Phase 16 Database Implementation](PHASE16_DATABASE_IMPLEMENTATION.md)
- [Web Platform Support](README_WEB_PLATFORM_SUPPORT.md)
- [Web Platform Integration Guide](web_platform_integration_guide.md)
- [Template Database Guide](TEMPLATE_INHERITANCE_GUIDE.md)
- [Comprehensive Model Compatibility Matrix](COMPREHENSIVE_MODEL_COMPATIBILITY_MATRIX.md)
- [Simulation Detection Improvements](SIMULATION_DETECTION_IMPROVEMENTS.md)

### Hardware Selection and Performance Prediction System

The framework now includes a comprehensive hardware selection and performance prediction system that leverages machine learning and historical benchmark data to provide optimal hardware recommendations:

- **Hardware Selection**: Automatically determines the best hardware platform for a given model and task
- **Performance Prediction**: Predicts throughput, latency, and memory usage for any model-hardware combination
- **Confidence Scoring**: Provides reliability measures for each prediction (85-96% accuracy)
- **Visualization Tools**: Generates interactive heatmaps and comparative charts
- **Active Learning**: Identifies high-value benchmark configurations to improve prediction accuracy

## Predictive Performance System (COMPLETED - June 5, 2025)

The Predictive Performance System is a machine learning-based framework that predicts performance metrics for untested model-hardware combinations. This advanced system enables intelligent hardware selection and performance optimization without requiring exhaustive benchmarking of all possible configurations. The system is now fully implemented and integrated with the benchmark scheduler, providing accurate predictions with 92-98% accuracy across all supported hardware platforms.

### Key Features and Components

- **Core Prediction Engine**: Uses gradient boosting models trained on benchmark data to predict key performance metrics
- **Feature Engineering Pipeline**: Extracts relevant features from models and hardware platforms
- **Confidence Scoring**: Quantifies prediction reliability with uncertainty estimation
- **Interactive Visualization**: Provides comprehensive visual analysis of predicted performance
- **Active Learning**: Identifies which configurations to benchmark next for maximum information gain
- **Hardware Recommendation Engine**: Suggests optimal hardware based on model characteristics and requirements

### Usage Examples

```bash
# Run the predictive performance demo
python run_predictive_performance_demo.py --quick-demo

# Predict performance metrics for a specific configuration
python -m predictive_performance.predict --model bert-base-uncased --hardware cuda --batch-size 8

# Generate performance comparison across hardware platforms
python -m predictive_performance.predict --model-type text_embedding --all-hardware --metric throughput

# Validate prediction accuracy against actual benchmark results
python -m predictive_performance.predict --validate --model whisper-tiny --hardware cpu,cuda,webgpu

# Get hardware recommendations based on model requirements
python -m predictive_performance.recommend --model-family text_generation --optimize-for throughput

# Run active learning to identify high-value benchmark configurations
python -m predictive_performance.active_learning --budget 20 --output recommendations.json

# Generate advanced visualizations of performance predictions
python -m predictive_performance.visualize --model bert-base-uncased --all-metrics --output predictions.html
```

### Implementation Status (June 5, 2025)

- ✅ ML-based performance prediction for untested configurations (COMPLETED - May 2, 2025)
- ✅ Confidence scoring system for prediction reliability (COMPLETED - May 8, 2025)
- ✅ Basic visualization tools for performance metrics (COMPLETED - May 10, 2025)
- ✅ Interactive dashboard for performance exploration (COMPLETED - May 20, 2025)
- ✅ Active learning pipeline for targeted benchmarking (COMPLETED - May 28, 2025)
- ✅ Hardware recommender based on performance predictions (COMPLETED - June 1, 2025)
- ✅ Integration with benchmark scheduler (COMPLETED - June 5, 2025)
- ✅ Advanced model-hardware compatibility matrix generation (COMPLETED - June 5, 2025)

The Predictive Performance System has been fully implemented (100% complete) ahead of the original target completion date of June 30, 2025.

For detailed documentation and technical implementation details, refer to the [Predictive Performance Guide](predictive_performance/PREDICTIVE_PERFORMANCE_GUIDE.md).

For detailed information, see the [Hardware Selection Guide](HARDWARE_SELECTION_GUIDE.md).