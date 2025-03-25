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
  - Multiple sharding strategies (layer, attention_feedforward, component)
  - Browser capability detection with specialized optimizations
  - Intelligent component distribution based on browser strengths

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

3. **Priority 3: Enhance API Integration with Distributed Testing**
   - Develop comprehensive integration between API backends and distributed testing framework
   - Create unified testing interface for all API types
   - Implement performance metrics collection for API benchmark comparison
   - Target completion: August 10, 2025

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

5. **Priority 5: Test Codebase Refactoring Analysis**
   - Generate AST (Abstract Syntax Tree) report for all test files
   - Analyze class and method structures across test suite
   - Identify redundant test patterns and duplication
   - Develop refactoring plan to unify similar tests
   - Identify deprecated tests for removal
   - Create migration path for test standardization
   - Target completion: August 15, 2025

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
3. Add new model types in `refactored_generator_suite/generators/model_generator.py`

### Generator Architecture

The generator is implemented as a modular system with the following components:

1. **Model Metadata**: Architecture type detection and model class selection
2. **Hardware Templates**: Hardware-specific code for different backends
3. **Task Templates**: Task-specific code for different model use cases
4. **Template Composition**: Combines hardware and task templates into complete modules

## Skillset Benchmarking System (NEW)

The framework now includes a comprehensive benchmarking system for measuring the performance of skillset implementations across various hardware backends.

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
python generate_skillset_benchmarks.py
```

This will generate benchmark files for all 211+ skillset implementations in the `benchmarks/skillset/` directory.

To generate a benchmark for a specific model:

```bash
python generate_skillset_benchmarks.py --model bert
```

### Running Benchmarks

To run benchmarks for all skillset implementations:

```bash
# Run inference benchmarks on CPU
python run_all_skillset_benchmarks.py --type inference --hardware cpu --report

# Run throughput benchmarks on CPU
python run_all_skillset_benchmarks.py --type throughput --hardware cpu --report

# Run both benchmark types
python run_all_skillset_benchmarks.py --type both --hardware cpu --report
```

To benchmark specific models:

```bash
python run_all_skillset_benchmarks.py --type both --hardware cpu --model bert --model roberta --model t5 --report
```

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

### Benchmark Results

Benchmark results are saved as JSON files in the specified output directory. If the `--report` option is used, HTML reports with visualizations are also generated. The reports include:

- Module import time
- Class instantiation time
- Initialization times for each batch size
- Statistical metrics (mean, standard deviation)
- Throughput in models per second
- Speedup compared to sequential execution
- Charts and visualizations for easy analysis

## Command Reference

For detailed documentation on all commands and capabilities, see the full documentation in 
[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md).
