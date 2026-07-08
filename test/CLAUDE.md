# IPFS Accelerate Python Framework - Development Guide

## Current Focus: Distributed Testing Framework (Updated July 2025)

### Project Status Overview

The project has successfully completed 16 phases of implementation and multiple major feature additions. Current status:

### Ongoing Projects (July 2025)

- ✅ **Distributed Testing Framework** (COMPLETED - 100% complete)
  - ✅ COMPLETED:
    - Core Infrastructure for task distribution and worker management
    - Security with API key authentication and role-based access
    - Intelligent Task Distribution with hardware-aware routing
    - Cross-Platform Worker Support for Linux, Windows, macOS, and containers
    - CI/CD Pipeline Integration with GitHub Actions, GitLab CI, and Jenkins
    - High Availability Clustering with automatic failover (July 20, 2025)
    - Dynamic Resource Management with adaptive scaling and cloud integration (July 21, 2025)
    - Real-Time Performance Metrics Dashboard with statistical regression detection (July 22, 2025)
    - Integration with external monitoring systems (Grafana, Prometheus) (July 24, 2025)
    - Performance Trend Analysis with machine learning-based anomaly detection (July 24, 2025)
    - Advanced scheduling algorithms for optimal task allocation (July 24, 2025)
  - COMPLETED: July 24, 2025

- ✅ **Ultra-Low Precision Quantization Support** (COMPLETED - 100% complete)
  - ✅ WebGPU 2-bit and 3-bit quantization implementations
  - ✅ Memory-efficient KV cache with 87.5% memory reduction
  - ✅ Browser-specific optimizations for Chrome, Firefox, Edge, and Safari
  - ✅ Mixed precision configurations for optimal performance/quality tradeoff
  - COMPLETED: July 15, 2025

### Completed Features

- ✅ **High Availability Clustering** (COMPLETED - July 20, 2025)
  - ✅ Raft-inspired consensus algorithm for leader election
  - ✅ State replication across coordinator nodes
  - ✅ Automatic failover with zero downtime
  - ✅ Health monitoring with CPU, memory, disk, and network metrics
  - ✅ WebNN/WebGPU detection for hardware-aware coordination
  - ✅ Visualization generation for cluster state and health metrics
  - ✅ Message integrity verification with hash-based verification
  - ✅ Self-healing capabilities for resource constraint recovery

- ✅ **Cross-Browser Model Sharding with Fault Tolerance** (COMPLETED - May 2025)
  - Distributes large models across multiple browser types to leverage browser-specific optimizations
  - Multiple sharding strategies (layer, attention_feedforward, component, hybrid, pipeline)
  - Comprehensive fault tolerance with recovery mechanisms (simple, progressive, parallel, coordinated)
  - Browser capability detection with specialized optimizations
  - Intelligent component distribution based on browser strengths
  - End-to-end testing across all sharding strategies with 100% test coverage
  - Circuit breaker pattern implementation for preventing cascading failures

- ✅ **Cross-Model Tensor Sharing** (COMPLETED - March 2025)
  - Memory reduction: Up to 30% memory reduction for multi-model workflows
  - Inference speedup: Up to 30% faster inference when reusing cached embeddings
  - Supports sharing between compatible models (text, vision, audio embeddings)
  - See [IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md](IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md) for details

## Implementation Priorities (July 2025)

Based on the current status, the following are the prioritized tasks for completion:

1. **Priority 1: Complete Distributed Testing Framework** (✅ COMPLETED - 100%)
   - ✅ Enhance dynamic scaling and resource allocation
   - ✅ Add real-time performance metrics visualization
   - ✅ Implement advanced scheduling algorithms
   - ✅ Complete integration with external monitoring systems (Prometheus/Grafana)
   - ✅ Implement ML-based anomaly detection and performance trend analysis
   - COMPLETED: July 24, 2025

2. **Priority 2: Comprehensive HuggingFace Model Testing (300+ classes)** (✅ COMPLETED - 100%)
   - ✅ Developed robust test generator with token-based replacement system (March 21, 2025)
   - ✅ Implemented special handling for hyphenated model names like xlm-roberta (March 21, 2025)
   - ✅ Created coverage tracking system with automated reporting (March 21, 2025)
   - ✅ Achieved test coverage for all 309 model types spanning all architecture categories (March 22, 2025)
   - ✅ Completed implementation of all Phase 2 high-priority models (March 21, 2025)
   - ✅ Created automated script for systematic test generation (March 21, 2025)
   - ✅ Completed Phase 4 medium-priority models implementation (March 22, 2025)
   - ✅ Added support for all model architectures including encoder, decoder, and audio models (March 22, 2025)
   - ✅ Implemented validation for from_pretrained() method coverage (March 22, 2025)
   - ✅ Created comprehensive model class inventory system (March 22, 2025)
   - ✅ Integrated with compatibility matrix in DuckDB (March 22, 2025)
   - COMPLETED: March 22, 2025, ahead of August 1, 2025 target

3. **Priority 3: Enhance API Integration with Distributed Testing** (✅ COMPLETED - 100%)
   - ✅ Developed comprehensive integration between API backends and distributed testing framework (July 29, 2025)
   - ✅ Created unified testing interface for all API types (July 29, 2025)
   - ✅ Implemented performance metrics collection for API benchmark comparison (July 29, 2025)
   - ✅ Added support for multiple API providers (OpenAI, Claude, Groq) (July 29, 2025)
   - ✅ Implemented anomaly detection and predictive analytics for API performance metrics (July 29, 2025)
   - ✅ Created comprehensive end-to-end example with simulation capabilities (July 29, 2025)
   - COMPLETED: July 29, 2025, ahead of August 10, 2025 target

4. **Priority 4: Advance UI for Visualization Dashboard** (✅ COMPLETED - 100%)
   - ✅ Create interactive visualization dashboard for performance metrics
   - ✅ Implement real-time monitoring of distributed testing
   - ✅ Develop comparative visualization tools for API performance
   - ✅ Enhance UI for regression detection visualization
   - ✅ Add visualization options panel with controls for confidence intervals, trend lines, and annotations
   - ✅ Implement enhanced export functionality with multiple format support (HTML, PNG, SVG, JSON, PDF)
   - ✅ Add comprehensive test suite for UI features
   - ✅ Create end-to-end test runner for visualization features
   - COMPLETED: July 20, 2025

5. **Priority 5: Test Codebase Refactoring Analysis** (✅ COMPLETED - 100%)
   - ✅ Generated AST (Abstract Syntax Tree) report for all test files
   - ✅ Analyzed class and method structures across test suite
   - ✅ Identified redundant test patterns and duplication
   - ✅ Developed comprehensive refactoring plan to standardize and unify similar tests
   - ✅ Created base test class hierarchy with ModelTest, HardwareTest, APITest, and BrowserTest
   - ✅ Implemented migration path with setup script and migration guide
   - ✅ Created sample implementations of refactored tests
   - ✅ Established directory structure for refactored tests
   - COMPLETED: July 27, 2025, ahead of August 15, 2025 target

6. **Priority 6: Comprehensive Benchmark System with FastAPI Integration** (✅ COMPLETED - 100% complete)
   - ✅ Implemented DuckDB integration for benchmark results storage (August 1, 2025)
   - ✅ Created refactored benchmark suite with standardized architecture (August 3, 2025)
   - ✅ Implemented HuggingFace integration for 300+ model benchmarking (August 5, 2025)
   - ✅ Developed complete benchmark pipeline for end-to-end execution (August 7, 2025)
   - ✅ Enhanced with resource-aware scheduling and incremental benchmarking (August 8, 2025)
   - ✅ Implemented FastAPI endpoints for Electron container integration (August 10, 2025)
   - ✅ Added WebSocket support for real-time benchmark progress tracking (August 10, 2025)
   - ✅ Created interactive dashboard for benchmark results visualization (August 11, 2025)
   - COMPLETED: August 11, 2025, ahead of August 17, 2025 target

7. **Priority 7: Mobile Edge Expansion** (✅ COMPLETED - 100% complete)
   - ✅ Developed Android Test Harness with real model execution support (April 15, 2025)
   - ✅ Implemented battery impact analysis methodology (April 17, 2025)
   - ✅ Created thermal monitoring system for mobile devices (April 18, 2025)
   - ✅ Developed iOS Test Harness with Core ML support (April 22, 2025)
   - ✅ Implemented database integration for mobile benchmark results (April 24, 2025)
   - ✅ Created Cross-Platform Analysis Tool for iOS/Android comparison (April 27, 2025)
   - ✅ Implemented Android CI Benchmark Runner (April 29, 2025)
   - ✅ Implemented iOS CI Benchmark Runner (April 29, 2025)
   - ✅ Created Benchmark Database Merger Utility (April 30, 2025)
   - ✅ Developed Mobile Performance Regression Detection Tool (April 30, 2025)
   - ✅ Implemented Mobile Performance Dashboard Generator (April 30, 2025)
   - ✅ Created Android CI Workflow (May 1, 2025)
   - ✅ Created iOS CI Workflow (May 1, 2025)
   - ✅ Implemented Cross-Platform Analysis Workflow (May 1, 2025)
   - ✅ Developed Test Model Downloaders for Android and iOS (May 1, 2025)
   - ✅ Created CI Runner Setup Utility for environment configuration (May 1, 2025)
   - ✅ Implemented CI Workflow Installation Tool (May 1, 2025)
   - ✅ Created comprehensive Mobile CI Runner Setup Guide (May 2, 2025)
   - ✅ Developed Android CI Runner Setup Script (May 2, 2025)
   - ✅ Developed iOS CI Runner Setup Script (May 2, 2025)
   - ✅ Created CI Installation Script for GitHub Actions integration (May 2, 2025)
   - COMPLETED: May 2, 2025, ahead of May 5, 2025 target

8. **Priority 8: Simulation Data Calibration and DuckDB Integration** (✅ COMPLETED - 100% complete)
   - ✅ Implemented DuckDB repository for calibration data storage (August 13, 2025) 
   - ✅ Developed adapter classes for connecting calibration components with DuckDB (August 13, 2025)
   - ✅ Created command-line tool for running calibration with DuckDB integration (August 13, 2025)
   - ✅ Implemented comprehensive database schema for calibration data (August 13, 2025)
   - ✅ Added support for tracking parameter stability and uncertainty (August 13, 2025)
   - ✅ Developed parameter drift detection and monitoring (August 13, 2025)
   - ✅ Created comprehensive documentation for calibration DuckDB integration (August 13, 2025)
   - ✅ Implemented sample data generation for testing and demonstration (August 13, 2025)
   - COMPLETED: August 13, 2025

9. **Priority 9: Predictive Performance Modeling System with DuckDB and API Integration** (✅ COMPLETED - 100% complete)
   - ✅ Implemented DuckDB repository for predictive performance data storage (August 14, 2025)
   - ✅ Developed adapter classes for connecting hardware model predictor with DuckDB (August 14, 2025)
   - ✅ Created adapter for ML-based performance prediction models with DuckDB (August 14, 2025)
   - ✅ Implemented comprehensive database schema for predictions, measurements, and ML models (August 14, 2025)
   - ✅ Added support for tracking prediction accuracy and feature importance (August 14, 2025)
   - ✅ Developed recommendation history and feedback system (August 14, 2025)
   - ✅ Created command-line tool for predictive performance with DuckDB integration (August 14, 2025)
   - ✅ Created comprehensive documentation for predictive performance DuckDB integration (August 14, 2025)
   - ✅ Implemented sample data generation for testing and demonstration (August 14, 2025)
   - ✅ Created FastAPI server with RESTful endpoints for all predictive performance functionality (August 15, 2025)
   - ✅ Implemented WebSocket support for real-time task progress tracking (August 15, 2025)
   - ✅ Developed both synchronous and asynchronous client libraries (August 15, 2025)
   - ✅ Created integration module to connect with Unified API Server (August 15, 2025)
   - ✅ Updated Unified API Server to include Predictive Performance API endpoints (August 15, 2025)
   - ✅ Created comprehensive documentation and example scripts (August 15, 2025)
   - COMPLETED: August 15, 2025

10. **Priority 10: Hardware Optimization Exporter with ZIP Archive Support** (✅ COMPLETED - 100% complete)
   - ✅ Implemented `OptimizationExporter` class for exporting recommendations as deployable files (August 28, 2025)
   - ✅ Created framework-specific templates for PyTorch, TensorFlow, OpenVINO, WebGPU, and WebNN (August 28, 2025)
   - ✅ Developed export formats including Python, JSON, YAML, and markdown documentation (August 29, 2025)
   - ✅ Implemented batch export functionality for multiple recommendations (August 29, 2025)
   - ✅ Created comprehensive web interface for exporting and downloading recommendations (August 30, 2025)
   - ✅ Added ZIP archive creation capabilities for easy download of all exported files (August 30, 2025)
   - ✅ Integrated with the FastAPI server for RESTful API access (August 30, 2025)
   - ✅ Added streaming download endpoint for ZIP archives (August 30, 2025)
   - ✅ Created comprehensive documentation and example scripts (August 30, 2025)
   - ✅ Developed command-line interface for export operations (August 30, 2025)
   - ✅ Implemented test script for verification of export and ZIP functionality (August 30, 2025)
   - COMPLETED: August 30, 2025

7. **Priority 7: Mobile Edge Expansion** (✅ COMPLETED - 100% complete)
   - ✅ Developed Android Test Harness with real model execution support (April 15, 2025)
   - ✅ Implemented battery impact analysis methodology (April 17, 2025)
   - ✅ Created thermal monitoring system for mobile devices (April 18, 2025)
   - ✅ Developed iOS Test Harness with Core ML support (April 22, 2025)
   - ✅ Implemented database integration for mobile benchmark results (April 24, 2025)
   - ✅ Created Cross-Platform Analysis Tool for iOS/Android comparison (April 27, 2025)
   - ✅ Implemented Android CI Benchmark Runner (April 29, 2025)
   - ✅ Implemented iOS CI Benchmark Runner (April 29, 2025)
   - ✅ Created Benchmark Database Merger Utility (April 30, 2025)
   - ✅ Developed Mobile Performance Regression Detection Tool (April 30, 2025)
   - ✅ Implemented Mobile Performance Dashboard Generator (April 30, 2025)
   - ✅ Created Android CI Workflow (May 1, 2025)
   - ✅ Created iOS CI Workflow (May 1, 2025)
   - ✅ Implemented Cross-Platform Analysis Workflow (May 1, 2025)
   - ✅ Developed Test Model Downloaders for Android and iOS (May 1, 2025)
   - ✅ Created CI Runner Setup Utility for environment configuration (May 1, 2025)
   - ✅ Implemented CI Workflow Installation Tool (May 1, 2025)
   - ✅ Created comprehensive Mobile CI Runner Setup Guide (May 2, 2025)
   - ✅ Developed Android CI Runner Setup Script (May 2, 2025)
   - ✅ Developed iOS CI Runner Setup Script (May 2, 2025)
   - ✅ Created CI Installation Script for GitHub Actions integration (May 2, 2025)
   - COMPLETED: May 2, 2025, ahead of May 5, 2025 target

## High Availability Clustering

The High Availability Clustering feature provides coordinator redundancy through a Raft-inspired consensus algorithm, enabling automatic failover and improved fault tolerance for the Distributed Testing Framework.

### Architecture Highlights

- **Raft-Inspired Consensus**: Modified algorithm for leader election among coordinator nodes
- **State Machine Architecture**: Coordinator states (leader, follower, candidate, offline) with transition rules
- **Health Monitoring**: Real-time tracking of CPU, memory, disk, and network metrics
- **Self-Healing**: Automatic recovery from resource constraints
- **WebNN/WebGPU Detection**: Browser and hardware capability awareness
- **Visualization Tools**: Both graphical and text-based state visualization

### Documentation

For Distributed Testing Framework documentation, see these resources:
- [DISTRIBUTED_TESTING_DESIGN.md](DISTRIBUTED_TESTING_DESIGN.md): Comprehensive design documentation with architecture details
- [DISTRIBUTED_TESTING_GUIDE.md](DISTRIBUTED_TESTING_GUIDE.md): Overview of the Distributed Testing Framework
- [DISTRIBUTED_TESTING_COMPLETION.md](DISTRIBUTED_TESTING_COMPLETION.md): Summary of the completed Distributed Testing Framework

For High Availability Clustering documentation:
- [HARDWARE_FAULT_TOLERANCE_GUIDE.md](HARDWARE_FAULT_TOLERANCE_GUIDE.md): Detailed guide on fault tolerance mechanisms
- [README_AUTO_RECOVERY.md](README_AUTO_RECOVERY.md): User guide for the Auto Recovery System

For Real-Time Performance Metrics and Monitoring:
- [REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md](REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md): Comprehensive documentation for the performance dashboard
- [DYNAMIC_RESOURCE_MANAGEMENT.md](DYNAMIC_RESOURCE_MANAGEMENT.md): Documentation for the Dynamic Resource Management system

For ML-based Anomaly Detection and Prometheus/Grafana Integration:
- The `distributed_testing/ml_anomaly_detection.py` module provides comprehensive machine learning capabilities for detecting anomalies in metrics
- The `distributed_testing/prometheus_grafana_integration.py` module connects the framework to external monitoring systems
- The `distributed_testing/advanced_scheduling.py` module implements intelligent task scheduling algorithms

For Test Codebase Refactoring Documentation:
- [COMPREHENSIVE_TEST_REFACTORING_PLAN.md](COMPREHENSIVE_TEST_REFACTORING_PLAN.md): Complete refactoring strategy with timeline and implementation plan
- [README_TEST_REFACTORING_IMPLEMENTATION.md](README_TEST_REFACTORING_IMPLEMENTATION.md): Detailed implementation plan for Phase 1 of the refactoring
- [REFACTORED_TEST_MIGRATION_GUIDE.md](REFACTORED_TEST_MIGRATION_GUIDE.md): Guide for migrating tests to the refactored structure
- [TEST_REFACTORING_SUMMARY.md](TEST_REFACTORING_SUMMARY.md): Summary of the completed test refactoring analysis and implementation

## Model Skillset Generation

The framework includes a comprehensive skillset generation system for creating hardware-aware implementations for 300+ HuggingFace model types. These implementations include support for multiple hardware backends (CPU, CUDA, ROCm, OpenVINO, Apple, Qualcomm) with proper fallback mechanisms.

### Skillset Generator Overview

The skillset generator creates Python modules for each model type with the following features:
- Hardware detection and optimization for multiple backends
- Model-specific initialization and inference code
- Task-specific processing based on model architecture
- Graceful fallback when hardware is unavailable
- Mock implementations for testing and development

### Generated Skillset Directory

All skillset files are generated in the `test/ipfs_accelerate_py/worker/skillset/` directory (relative to the repo root). This serves as a staging area for testing before deployment to production.

### Regenerating Skillset Files

To regenerate the skillset files, use the following commands:

#### Generate All 300+ Model Files at Once

To generate all 300+ model skillset files in a single command:

```bash
cd /path/to/ipfs_accelerate_py/test/refactored_generator_suite
python generate_all_skillsets.py --priority all
```

This command will generate implementations for every supported model type in the HuggingFace ecosystem, including all architectures:
- Encoder-only models (BERT, RoBERTa, etc.)
- Decoder-only models (GPT-2, LLaMA, etc.)
- Encoder-decoder models (T5, BART, etc.)
- Vision models (ViT, BEiT, etc.)
- Vision-text models (CLIP, BLIP, etc.)
- Speech models (Whisper, Wav2Vec2, etc.)
- Diffusion models, MoE models, state-space models, and more

The process may take a few minutes to complete as it generates over 300 model implementations.

#### Generate by Priority

```bash
# Generate only critical priority models (bert, gpt2, llama, etc.)
python generate_all_skillsets.py --priority critical

# Generate high priority models (includes critical + more common models)
python generate_all_skillsets.py --priority high

# Generate medium priority models (includes high + critical + less common models)
python generate_all_skillsets.py --priority medium
```

#### Generate Specific Models

```bash
# Generate a single model implementation
python generate_simple_model.py bert

# Generate specific model types
python generate_simple_model.py llama
python generate_simple_model.py mixtral
python generate_simple_model.py stable-diffusion
```

#### Specialized Model Types

Special model architectures are also supported:
- State-space models: `python generate_simple_model.py mamba`
- Mixture-of-experts models: `python generate_simple_model.py mixtral`
- Diffusion models: `python generate_simple_model.py stable-diffusion`
- Vision-text models: `python generate_simple_model.py clip`
- Speech models: `python generate_simple_model.py whisper`

### Verifying Generated Files

After generation, verify the generated files:

```bash
# Count total generated files
find /path/to/ipfs_accelerate_py/test/ipfs_accelerate_py/worker/skillset -type f -name "hf_*.py" | wc -l

# List all generated files
find /path/to/ipfs_accelerate_py/test/ipfs_accelerate_py/worker/skillset -type f -name "hf_*.py" | sort
```

### Customizing the Generator

To customize the generator behavior:

1. Modify templates in `refactored_generator_suite/templates/`
2. Update hardware detection in `refactored_generator_suite/hardware/hardware_detection.py`
3. Add new model types in `refactored_generator_suite/scripts/generators/model_generator.py`

### Generator Architecture

The generator is implemented as a modular system with the following components:

1. **Model Metadata**: Architecture type detection and model class selection
2. **Hardware Templates**: Hardware-specific code for different backends
3. **Task Templates**: Task-specific code for different model use cases
4. **Template Composition**: Combines hardware and task templates into complete modules

## Skillset Benchmarking System (NEW)

The framework now includes a comprehensive benchmarking system for measuring the performance of skillset implementations across various hardware backends. The system is specifically designed to benchmark the **test implementation** of models in `test/ipfs_accelerate_py/worker/skillset/` rather than the production implementations.

### Benchmark Types

The benchmarking system supports two primary benchmark types:

1. **Inference Benchmarks**: Measure initialization performance including module import time, class instantiation time, and model initialization time across different batch sizes.

2. **Throughput Benchmarks**: Measure concurrent execution performance including throughput in models per second and speedup compared to sequential execution.

### Hardware Support

The benchmarking system supports multiple hardware backends:
- CPU: Standard CPU processing
- CUDA: NVIDIA GPU acceleration
- ROCm: AMD GPU acceleration
- OpenVINO: Intel's neural network optimization framework
- MPS: Apple's Metal Performance Shaders for Apple Silicon
- QNN: Qualcomm Neural Network processing

### Generating Benchmark Files

To generate benchmark files for skillset implementations:

```bash
cd /path/to/ipfs_accelerate_py/test/refactored_benchmark_suite
python generate_skillset_benchmarks.py --skillset-dir ../ipfs_accelerate_py/worker/skillset/
```

This will generate benchmark files for all 300+ skillset implementations in the `data/benchmarks/skillset/` directory.

To generate a benchmark for a specific model:

```bash
python generate_skillset_benchmarks.py --model bert --skillset-dir ../ipfs_accelerate_py/worker/skillset/
```

### Running Benchmarks

To run benchmarks for all skillset implementations:

```bash
# Run inference benchmarks on CPU
python run_all_skillset_benchmarks.py --type inference --hardware cpu --report --skillset-dir ../ipfs_accelerate_py/worker/skillset/

# Run throughput benchmarks on CPU
python run_all_skillset_benchmarks.py --type throughput --hardware cpu --report --skillset-dir ../ipfs_accelerate_py/worker/skillset/

# Run both benchmark types
python run_all_skillset_benchmarks.py --type both --hardware cpu --report --skillset-dir ../ipfs_accelerate_py/worker/skillset/
```

To benchmark specific models:

```bash
python run_all_skillset_benchmarks.py --type both --hardware cpu --model bert --model roberta --model t5 --report --skillset-dir ../ipfs_accelerate_py/worker/skillset/
```

### Complete Benchmark Pipeline

The system now includes a unified end-to-end benchmark pipeline that handles the entire process:

```bash
python run_complete_benchmark_pipeline.py --priority high --hardware cpu,cuda --progressive-mode --incremental --db-path ./benchmark_db.duckdb --skillset-dir ../ipfs_accelerate_py/worker/skillset/
```

This will:
1. Generate model skillsets in the test directory
2. Create benchmark files for these implementations
3. Run benchmarks with resource-aware scheduling
4. Store results in the DuckDB database
5. Generate comprehensive reports and visualizations

### FastAPI Integration for Electron

The benchmark system is now integrated with a FastAPI backend, allowing the Electron container to interact with it:

```bash
# Start the benchmark API server
python benchmark_api_server.py --port 8000
```

API Endpoints:

The FastAPI server provides the following RESTful endpoints:

- **POST /api/benchmark/run** - Start a benchmark run with specified parameters
  ```json
  {
    "priority": "high",
    "hardware": ["cpu", "cuda"],
    "models": ["bert", "gpt2"],
    "batch_sizes": [1, 8],
    "precision": "fp32",
    "progressive_mode": true,
    "incremental": true
  }
  ```

- **GET /api/benchmark/status/{run_id}** - Get status of a running benchmark
- **GET /api/benchmark/results/{run_id}** - Get results of a completed benchmark
- **GET /api/benchmark/models** - List available models for benchmarking
- **GET /api/benchmark/hardware** - List available hardware platforms
- **GET /api/benchmark/reports** - List available benchmark reports
- **GET /api/benchmark/query** - Query benchmark results with optional filters
- **WebSocket /api/benchmark/ws/{run_id}** - Real-time benchmark progress tracking

### FastAPI Integration and WebSocket Support

The benchmark system now includes a comprehensive FastAPI server with WebSocket support for real-time progress tracking:

```bash
# Start the benchmark API server
cd /path/to/ipfs_accelerate_py/test/refactored_benchmark_suite
python benchmark_api_server.py --port 8000
```

#### WebSocket Support

For real-time updates on benchmark progress, connect to the run-specific WebSocket endpoint:

```javascript
// Example JavaScript code for Electron
const ws = new WebSocket('ws://localhost:8000/api/benchmark/ws/YOUR_RUN_ID');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress);
  console.log('Current step:', data.current_step);
  console.log('Completed models:', data.completed_models);
  console.log('Total models:', data.total_models);
  
  // Update UI with progress
  updateProgressBar(data.progress);
  updateStatusText(data.current_step);
};

// To start a benchmark run
fetch('http://localhost:8000/api/benchmark/run', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    priority: 'high',
    hardware: ['cpu'],
    models: ['bert', 'gpt2'],
    batch_sizes: [1, 8],
    precision: 'fp32',
    progressive_mode: true,
    incremental: true
  })
}).then(response => response.json())
  .then(data => {
    console.log('Run ID:', data.run_id);
    // Connect to WebSocket for this specific run
    const ws = new WebSocket(`ws://localhost:8000/api/benchmark/ws/${data.run_id}`);
  });
```

#### Key Features

- **Test Implementation Focus**: Specifically benchmarks the test implementations in `test/ipfs_accelerate_py/worker/skillset/`
- **Asynchronous Execution**: Non-blocking benchmark execution with real-time status updates
- **Database Integration**: Stores results in DuckDB for efficient querying and analysis
- **Progress Tracking**: Real-time progress updates via WebSockets
- **Resource-Aware Scheduling**: Optimizes hardware utilization during benchmarking
- **Incremental Benchmarking**: Only runs benchmarks for missing or outdated results

### Command-line Options

The benchmark runner supports the following options:
- `--skillset-dir`: Directory containing skillset implementations
- `--output-dir`: Directory to write benchmark results
- `--benchmark-dir`: Directory to write benchmark files
- `--hardware`: Hardware backend to use (cpu, cuda, rocm, openvino, mps, qnn)
- `--type`: Type of benchmark to run (inference, throughput, both)
- `--concurrent-workers`: Number of concurrent workers for throughput benchmarks
- `--batch-sizes`: Comma-separated list of batch sizes to test
- `--runs`: Number of measurement runs
- `--model`: Specific model to benchmark (can be specified multiple times)
- `--generate-only`: Only generate benchmark files, don't run benchmarks
- `--report`: Generate HTML reports with visualizations for benchmark results
- `--db-path`: Path to DuckDB database for storing results

### Benchmark Results

Benchmark results are saved to the DuckDB database specified with `--db-path`. If the `--report` option is used, HTML reports with visualizations are also generated. The reports include:

- Module import time
- Class instantiation time
- Initialization times for each batch size
- Statistical metrics (mean, standard deviation)
- Throughput in models per second
- Speedup compared to sequential execution
- Charts and visualizations for easy analysis

### Interactive Dashboard

The system now includes an interactive web-based dashboard for comprehensive visualization and analysis of benchmark results:

```bash
# Start the benchmark dashboard
cd /path/to/ipfs_accelerate_py/test/refactored_benchmark_suite
./run_benchmark_dashboard.sh --port 8050 --api-url http://localhost:8000
```

The dashboard provides:

1. **Overview Tab**: High-level metrics and performance comparisons
   - Hardware comparison charts for specific models
   - Top-performing models by hardware platform
   - Batch size scaling visualization

2. **Comparison Tab**: Detailed performance comparisons
   - Performance heatmap for model families and hardware
   - Detailed results table with filtering and sorting
   - Export capabilities for further analysis

3. **Live Runs Tab**: Monitor and control benchmarks
   - Real-time progress tracking for active benchmarks
   - Start new benchmark runs with custom configurations
   - WebSocket integration for instant updates

4. **Reports Tab**: Access benchmark reports
   - List of all available benchmark reports
   - Custom SQL query support for advanced analysis
   - Direct access to benchmark result files

The dashboard connects to the FastAPI server for data retrieval and provides advanced visualization features including:

- Interactive filtering by model family, hardware type, and batch size
- Multiple performance metrics (throughput, latency, memory usage)
- Real-time updates for active benchmark runs
- Comparative analysis across hardware platforms and model families

## API Refactoring Initiative

To reduce code debt and provide a more cohesive user experience, we are implementing a comprehensive API integration approach using FastAPI. This initiative will standardize interfaces across all refactored components.

### Current Status (July 2025)

- ✅ **Benchmark API Server** (COMPLETED - 100%)
  - ✅ RESTful endpoints for benchmark operations
  - ✅ WebSocket support for real-time progress tracking
  - ✅ Background task processing for long-running operations
  - ✅ DuckDB integration for result storage
  - ✅ Comprehensive endpoint documentation
  
- ✅ **Test Suite API Integration** (COMPLETED - 100%)
  - ✅ API models and interface definitions
  - ✅ Core API implementation with RESTful endpoints
  - ✅ WebSocket support for real-time test progress
  - ✅ Synchronous and asynchronous client implementations
  - ✅ Integration with existing test components via TestRunner
  - ✅ Test result storage and retrieval endpoints
  - ✅ Support for test cancellation and history tracking
  - ✅ DuckDB integration for advanced querying (August 15, 2025)
  
- 🔄 **Generator API Integration** (IN PROGRESS - 100%)
  - ✅ API models and interface definitions
  - ✅ Core endpoint structure and request handling
  - ✅ Integration with generator components
  - ✅ WebSocket progress tracking implementation
  - ✅ Background task management
  - ✅ DuckDB integration for task tracking and history
  
- ✅ **Unified API Server** (COMPLETED - 100%)
  - ✅ API architecture design and documentation
  - ✅ Common patterns and models defined
  - ✅ Component API integration planning
  - ✅ Authentication framework implementation
  - ✅ API gateway implementation
  - ✅ Database integration across components
  - ✅ Cross-component database operations
  - ✅ Unified database views and statistics

### Documentation

The following documentation is available for the API integration:

- [FASTAPI_INTEGRATION_GUIDE.md](FASTAPI_INTEGRATION_GUIDE.md): Comprehensive guide to FastAPI integration
- [API_INTEGRATION_PLAN.md](refactored_test_suite/integration/API_INTEGRATION_PLAN.md): Detailed plan for API refactoring
- [BENCHMARK_FASTAPI_DASHBOARD.md](refactored_benchmark_suite/BENCHMARK_FASTAPI_DASHBOARD.md): Benchmark API dashboard documentation

### Implementation

Key files related to API implementation:

- `/test/refactored_benchmark_suite/benchmark_api_server.py`: Complete FastAPI server for benchmarks
- `/test/refactored_test_suite/api/api_client.py`: Client implementation for API interaction
- `/test/refactored_test_suite/integration/test_api_integration.py`: Integration tests for API functionality

### Integration Example

The API integration enables seamless interaction between components:

```python
# Example workflow: Generate model -> Test model -> Benchmark model
from refactored_test_suite.api.api_client import ApiClient

# Generate a model implementation
generator_client = ApiClient(base_url="http://localhost:8001")
gen_response = generator_client.run_operation("generate_model", {
    "model_name": "bert-base-uncased",
    "hardware": ["cpu", "cuda"]
})

# Wait for generation to complete
model_info = generator_client.monitor_operation(gen_response["operation_id"])

# Run tests on the generated model
test_client = ApiClient(base_url="http://localhost:8002")
test_response = test_client.run_operation("run_test", {
    "model_path": model_info["file_path"],
    "test_type": "comprehensive"
})

# Wait for tests to complete
test_results = test_client.monitor_operation(test_response["operation_id"])

# If tests pass, run benchmarks
if test_results["status"] == "passed":
    benchmark_client = ApiClient(base_url="http://localhost:8003")
    bench_response = benchmark_client.run_operation("run_benchmark", {
        "model_path": model_info["file_path"],
        "hardware": ["cpu", "cuda"],
        "batch_sizes": [1, 8, 32]
    })
    
    # Wait for benchmarks to complete
    bench_results = benchmark_client.monitor_operation(bench_response["operation_id"])
    
    # Print summary
    print(f"Model: {model_info['model_name']}")
    print(f"Tests: {test_results['status']}")
    print(f"Benchmark latency: {bench_results['latency_ms']} ms")
    print(f"Benchmark throughput: {bench_results['throughput']} items/sec")
```

## Command Reference

For detailed documentation on all commands and capabilities, see the full documentation in 
[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md).