# IPFS Accelerate Python Framework - Development Guide

## Current Focus: Advanced Hardware Benchmarking and Database Consolidation (Updated March 2025)

### Project Status Overview

The project has successfully completed 15 phases of implementation, focusing on test-driven development, hardware compatibility, model optimization, and cross-platform support. Key accomplishments include:

- ✅ Complete development pipeline for test and skillset generators
- ✅ Comprehensive hardware detection and compatibility system
- ✅ Advanced resource management system with hardware awareness
- ✅ Web platform integration (WebNN and WebGPU)
- ✅ Model family classification and compatibility matrix 
- ✅ Integration testing and platform support
- ✅ Advanced model compression and optimization
- ✅ Complete hardware platform test coverage for key models
- ✅ Test results database architecture and core components implemented
- ⏱️ Historical data migration and CI/CD integration (20% complete)

### Current Focus: Phase 16 - Advanced Hardware Benchmarking and Database Consolidation

#### Hardware Performance Work
- ✅ Create comprehensive benchmark database for all model-hardware combinations (100% complete)
- ✅ Implement comparative analysis reporting system for hardware performance (100% complete)
- ⏱️ Create automated hardware selection based on benchmarking data (80% complete)
- ⏱️ Implement training mode test coverage in addition to inference (40% complete)
- ✅ Complete cross-platform test coverage for 13 key model classes (100% complete)
- ✅ Develop specialized web platform tests for audio models (100% complete)
- ⏱️ Implement distributed training test suite (15% complete)
- ⏱️ Add performance prediction for model-hardware combinations (45% complete)

#### Database Restructuring Effort
- ✅ Consolidate benchmark and test output JSON files into DuckDB/Parquet for efficient storage and querying (100% complete)
- ✅ Design unified schema for all test result types (100% complete)
- ⏱️ Develop data migration pipeline for historical test data (15% complete)
- ✅ Create programmatic database interface for test runners (100% complete)
- ✅ Build analysis and visualization tools on the new database (100% complete)
- ⏱️ Integrate database with CI/CD pipeline for automatic result storage (5% complete)

## Hardware Compatibility Matrix

### Model Family-Based Compatibility Chart

| Model Family | CUDA | ROCm (AMD) | MPS (Apple) | OpenVINO | WebNN | WebGPU | Notes |
|--------------|------|------------|-------------|----------|-------|--------|-------|
| Embedding (BERT, etc.) | ✅ High | ✅ High | ✅ High | ✅ Medium | ✅ High | ✅ Medium | Efficient on all hardware |
| Text Generation (LLMs) | ✅ High | ✅ Medium | ✅ Medium | ✅ Low | ❌ N/A | ✅ Low | Memory requirements critical |
| Vision (ViT, CLIP, etc.) | ✅ High | ✅ Medium | ✅ High | ✅ High | ✅ Medium | ✅ Medium | OpenVINO optimized |
| Audio (Whisper, etc.) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Low | ⚠️ Low | CUDA preferred, Web support added |
| Multimodal (LLaVA, etc.) | ✅ High | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | Primarily CUDA only |

### Key Model Test Coverage Status

| Model Class | Model Used | CUDA | AMD | MPS | OpenVINO | WebNN | WebGPU | Notes |
|-------------|------------|------|-----|-----|----------|-------|--------|-------|
| BERT | bert-base-uncased, bert-tiny | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| T5 | t5-small, t5-efficient-tiny | ✅ | ✅ | ✅ | ⚠️* | ✅ | ✅ | *OpenVINO implementation mocked |
| LLAMA | opt-125m | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Web platform N/A |
| CLIP | Local test model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| ViT | vit-base | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| CLAP | Local test model | ✅ | ✅ | ✅ | ⚠️* | ✅ | ✅ | Complete tests with WebNN/WebGPU |
| Whisper | whisper-tiny | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete tests with WebNN/WebGPU |
| Wav2Vec2 | Local test model | ✅ | ✅ | ✅ | ⚠️* | ✅ | ✅ | Complete tests with WebNN/WebGPU |
| LLaVA | llava-onevision-base | ✅ | ❌ | ❌ | ⚠️* | ❌ | ❌ | *OpenVINO implementation mocked |
| LLaVA-Next | Local test model | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | CUDA-only support |
| XCLIP | Local test model | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | No web platform tests |
| Qwen2/3 | qwen2, qwen3, qwen2_vl, qwen3_vl | ✅ | ⚠️* | ⚠️* | ⚠️* | ❌ | ❌ | *Limited testing implementation |
| DETR | Local test model | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | No web platform tests |

## Essential Test Commands

### Hardware Testing
```bash
# Comprehensive hardware detection and compatibility test
python test/test_comprehensive_hardware.py --test all

# Test hardware backends with specific model
python test/test_hardware_backend.py --backend [cpu|cuda|openvino|mps|rocm|webnn|webgpu|all] --model [model_name]

# Test resource pool with hardware awareness
python test/test_resource_pool.py --test hardware

# Test model family integration with web platform support
python test/test_resource_pool.py --test family --debug

# Run web audio platform tests
python test/web_audio_platform_tests.py --run-all --browser chrome

# Run specific audio model web tests
python test/web_audio_platform_tests.py --test-whisper --test-wav2vec2 --test-clap
```

### Model Benchmarking and Validation
```bash
# Run comprehensive model benchmarks with key models
python test/run_model_benchmarks.py --output-dir ./benchmark_results

# Test on specific hardware platforms with small model set
python test/run_model_benchmarks.py --hardware cpu cuda --models-set small

# Manual model functionality verification
python test/verify_model_functionality.py --models bert t5 vit --hardware cpu cuda

# Run detailed hardware benchmarks
python test/hardware_benchmark_runner.py --model-families embedding text_generation --hardware cpu cuda
```

### Benchmark Database and Result Management
```bash
# Convert benchmark JSON files to Parquet/DuckDB format
python test/benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb

# Consolidate test results across directories
python test/benchmark_db_converter.py --consolidate --categories performance hardware compatibility

# Query benchmark database with SQL
python test/benchmark_db_query.py --sql "SELECT model, hardware, AVG(throughput) FROM benchmark_performance GROUP BY model, hardware"

# Generate reports from DuckDB benchmark database
python test/benchmark_db_query.py --report performance --format html --output benchmark_report.html
python test/benchmark_db_query.py --report hardware --format html --output hardware_report.html
python test/benchmark_db_query.py --report compatibility --format html --output compatibility_report.html

# Compare hardware platforms for a specific model using the new database
python test/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware --output bert_hardware_comparison.png

# Compare models on a specific hardware platform
python test/benchmark_db_query.py --hardware cuda --metric throughput --compare-models --output cuda_model_comparison.png

# Plot performance trends over time
python test/benchmark_db_query.py --model bert-base-uncased --hardware cuda --metric throughput --plot-trend

# Export data from the database
python test/benchmark_db_query.py --export performance --format csv --output performance_data.csv

# Generate compatibility matrix
python test/benchmark_db_query.py --matrix --format html --output compatibility_matrix.html

# API for programmatic access to benchmark data
python test/benchmark_db_api.py --serve  # Starts a FastAPI server on port 8000

# Automatically convert newly generated test results
python test/run_model_benchmarks.py --hardware all --auto-store-db

# Run scheduled cleanup of JSON files already migrated to database
python test/benchmark_db_maintenance.py --clean-json --older-than 30
```

### Developing Database Scripts
```bash
# Create initial database schema
python test/scripts/create_benchmark_schema.py

# Validate and fix any schema inconsistencies in existing data
python test/scripts/validate_benchmark_data.py --fix

# Create Python ORM models for the benchmark database
python test/scripts/generate_db_models.py

# Test database performance with synthetic benchmark data
python test/scripts/benchmark_db_performance.py --rows 1000000

# Develop modular analysis functions
python test/scripts/develop_analysis_modules.py --interactive
```

### Integration Testing
```bash
# Run all integration tests
python test/integration_test_suite.py

# Run tests for specific categories
python test/integration_test_suite.py --categories hardware_detection resource_pool

# Use the CI test runner script
./test/run_integration_ci_tests.sh --all
```

### Hardware Compatibility Reporting
```bash
# Collect and report compatibility issues from all components
python test/hardware_compatibility_reporter.py --collect-all

# Generate hardware compatibility matrix
python test/hardware_compatibility_reporter.py --matrix

# Check compatibility for a specific model
python test/hardware_compatibility_reporter.py --check-model bert-base-uncased
```

### Web Platform Audio Testing
```bash
# Run all audio model web platform tests
python test/web_audio_platform_tests.py --run-all --browser chrome

# Run tests for specific audio models
python test/web_audio_platform_tests.py --test-whisper
python test/web_audio_platform_tests.py --test-wav2vec2
python test/web_audio_platform_tests.py --test-clap

# Generate a test report from previous runs
python test/web_audio_platform_tests.py --generate-report

# Use headless mode for CI environments
python test/web_audio_platform_tests.py --run-all --headless --browser edge
```

## Performance Benchmarks

### Latest Performance Metrics

For detailed performance benchmarks, please refer to the following files:
- Comprehensive results: `test/performance_results/consolidated_performance_summary.md`
- Hardware-specific benchmarks: `test/HARDWARE_BENCHMARKING_GUIDE.md`
- Model compression results: `test/MODEL_COMPRESSION_GUIDE.md`
- Training benchmarks: `test/TRAINING_BENCHMARKING_GUIDE.md`
- Web platform audio tests: `test/WEB_PLATFORM_AUDIO_TESTING_GUIDE.md`
- Hardware selection system: `test/HARDWARE_SELECTION_GUIDE.md`
- Phase 16 implementation: `test/PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md`
- Database-backed benchmark dashboard: `http://localhost:8000/dashboard` (when running benchmark_db_api.py)

### Test Results Database Architecture

The new DuckDB/Parquet-based database system consolidates all test results and provides:
- Efficient storage (50-80% size reduction over JSON files)
- Fast SQL-based querying for analysis and reporting
- Programmatic access through Python API and REST endpoints
- Version-controlled schema for tracking benchmark metrics over time
- Support for time-series analysis of performance trends
- Integration with existing testing infrastructure
- Data validation and integrity checking
- Automated database maintenance and optimization

For detailed documentation, see:
- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [Database Migration Guide](DATABASE_MIGRATION_GUIDE.md)
- [Phase 16 Database Implementation](PHASE16_DATABASE_IMPLEMENTATION.md)

#### Database Migration Plan (To Be Implemented by April 2025)

1. **Schema Definition:**
   - Define standard schemas for different test types (hardware, performance, compatibility)
   - Create common timestamp, hardware, and model dimensions for all test types
   - Implement schema versioning mechanism for future evolution

2. **Data Migration Process:**
   - Develop conversion utilities for each JSON result format
   - Batch migration of historical data in archived_test_results, performance_results
   - Implement cleanup to remove redundant JSON files after confirmation

3. **Script Modularization:**
   - Refactor existing test runners to use the database interface
   - Create adapter layers for existing tools to write to database
   - Develop shared query library for common analysis patterns
   - Build visualization components for benchmark comparison

4. **Integration with CI/CD:**
   - Update CI workflows to store results directly in the database
   - Implement versioned snapshots for historical comparisons
   - Create regression detection system using historical data

### Key Performance Metrics (Updated March 2, 2025)

| Model Category | Best Hardware | Throughput Improvement | Memory Optimization | Batch Scaling Efficiency |
|----------------|---------------|------------------------|---------------------|--------------------------|
| Embedding Models | CUDA/WebNN | 5-15x vs CPU | 25-40% reduction | Excellent (near-linear) |
| Text Generation | CUDA | 3-8x vs CPU | 15-30% with quantization | Good (sub-linear) |
| Vision Models | CUDA/MPS | 5-12x vs CPU | 20-45% with optimizations | Very good (near-linear) |
| Audio Models | CUDA | 4-8x vs CPU | 10-25% reduction | Moderate (plateaus at 8-16) |
| Multimodal | CUDA only | 5-10x vs CPU | 15-35% with pruning | Limited (memory-bound) |

For web platform specific performance data, see `test/WEB_PLATFORM_INTEGRATION_GUIDE.md`.