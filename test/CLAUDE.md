# IPFS Accelerate Python Framework - Development Guide

## Current Focus: Advanced Hardware Benchmarking and Database Consolidation (Updated March 2025)

### Project Status Overview

The project has successfully completed 16 phases of implementation, focusing on test-driven development, hardware compatibility, model optimization, cross-platform support, and data management. Key accomplishments include:

- ✅ Complete development pipeline for test and skillset generators
- ✅ Comprehensive hardware detection and compatibility system
- ✅ Advanced resource management system with hardware awareness
- ✅ Web platform integration (WebNN and WebGPU)
- ✅ Model family classification and compatibility matrix 
- ✅ Integration testing and platform support
- ✅ Advanced model compression and optimization
- ✅ Complete hardware platform test coverage for key models
- ✅ Test results database architecture and core components implemented (100% complete)
- ✅ Historical data migration pipeline implemented (100% complete)
- ✅ CI/CD integration for automated benchmark storage (100% complete)

### Current Focus: Phase 16 - Advanced Hardware Benchmarking and Database Consolidation (100% Complete)

#### Hardware Performance Work
- ✅ Create comprehensive benchmark database for all model-hardware combinations (100% complete)
- ✅ Implement comparative analysis reporting system for hardware performance (100% complete)
- ✅ Create automated hardware selection based on benchmarking data (100% complete)
- ✅ Implement training mode test coverage in addition to inference (100% complete)
- ✅ Complete cross-platform test coverage for 13 key model classes (100% complete)
- ✅ Develop specialized web platform tests for audio models (100% complete)
- ✅ Implement distributed training test suite (100% complete)
- ✅ Add performance prediction for model-hardware combinations (100% complete)

#### Database Restructuring Effort
- ✅ Consolidate benchmark and test output JSON files into DuckDB/Parquet for efficient storage and querying (100% complete)
- ✅ Design unified schema for all test result types (100% complete)
- ✅ Develop data migration pipeline for historical test data (100% complete)
- ✅ Create programmatic database interface for test runners (100% complete)
- ✅ Build analysis and visualization tools on the new database (100% complete)
- ✅ Integrate database with CI/CD pipeline for automatic result storage (100% complete)
- ✅ Implement comprehensive data migration system with validation and tracking (100% complete)
- ✅ Migrate all benchmark scripts to use DuckDB for storage and querying (100% complete)
- ✅ Complete tool integration with all test runners (100% complete)
- ✅ Develop advanced analytics dashboard with interactive visualizations (100% complete)
- ✅ Complete CI/CD integration with GitHub Actions workflow (100% complete)

## Hardware Compatibility Matrix

### Model Family-Based Compatibility Chart

| Model Family | CUDA | ROCm (AMD) | MPS (Apple) | OpenVINO | WebNN | WebGPU | Notes |
|--------------|------|------------|-------------|----------|-------|--------|-------|
| Embedding (BERT, etc.) | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Fully supported on all hardware |
| Text Generation (LLMs) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | Memory requirements critical |
| Vision (ViT, CLIP, etc.) | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Full cross-platform support |
| Audio (Whisper, etc.) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | CUDA preferred, Web simulation added |
| Multimodal (LLaVA, etc.) | ✅ High | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | CUDA for production, others are limited |

To generate an updated compatibility matrix with actual benchmark data, run:
```bash
python test/benchmark_all_key_models.py --output-dir ./benchmark_results
```

This will benchmark all 13 high-priority model classes across all available hardware platforms and generate a comprehensive compatibility matrix based on real performance data.

### Key Model Test Coverage Status

| Model Class | Model Used | CUDA | AMD | MPS | OpenVINO | WebNN | WebGPU | Notes |
|-------------|------------|------|-----|-----|----------|-------|--------|-------|
| BERT | bert-base-uncased, bert-tiny | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| T5 | t5-small, t5-efficient-tiny | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| LLAMA | opt-125m | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | WebNN/WebGPU limited by memory |
| CLIP | Local test model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| ViT | vit-base | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Complete coverage |
| CLAP | Local test model | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Web has limited audio support |
| Whisper | whisper-tiny | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Web audio challenges |
| Wav2Vec2 | Local test model | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Web audio challenges |
| LLaVA | llava-onevision-base | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Memory intensive for web |
| LLaVA-Next | Local test model | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Memory intensive for web |
| XCLIP | Local test model | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Limited video support in web |
| Qwen2/3 | qwen2, qwen3, qwen2_vl, qwen3_vl | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Memory constraints on web |
| DETR | Local test model | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Limited detection support |

## Essential Test Commands

### Hardware-Aware Test Generation
```bash
# Generate tests with cross-platform hardware compatibility
python test/integrated_skillset_generator.py --model bert --cross-platform --hardware all

# Generate tests for specific hardware platforms only
python test/integrated_skillset_generator.py --model bert --hardware cuda,openvino,webnn

# Run hardware-specific template generation
python test/enhance_key_models_hardware_coverage.py --create-templates

# Update the test generator with hardware-aware templates
python test/update_test_generator_with_hardware_templates.py

# Run validation on hardware compatibility
python test/enhance_key_models_hardware_coverage.py --validate
```

### Phase 16 Hardware Integration
```bash
# Run hardware integration fixes on key model tests
./test/run_key_model_fixes.sh

# Fix hardware integration for specific models
python test/fix_hardware_integration.py --specific-models bert,t5,clip

# Fix all key model tests
python test/fix_hardware_integration.py --all-key-models

# Analyze hardware integration issues without fixing
python test/fix_hardware_integration.py --all-key-models --analyze-only --output-json hardware_analysis.json

# Test model generators with hardware-aware templates
python test/update_test_generator_with_hardware_templates.py

# Generate tests with cross-platform hardware compatibility
python test/integrated_skillset_generator.py --model bert --cross-platform --hardware all
```

### Hardware Testing
```bash
# Automated hardware selection for any model
python test/automated_hardware_selection.py --model [model_name] --batch-size [batch_size] --mode [inference|training]

# Select hardware for distributed training
python test/automated_hardware_selection.py --model [model_name] --distributed-config --gpu-count 8 --max-memory-gb 40

# Generate comprehensive hardware selection map
python test/automated_hardware_selection.py --create-map --output hardware_selection_map.json

# Analyze model performance across all available hardware
python test/automated_hardware_selection.py --model [model_name] --analyze --output analysis.json

# Detect available hardware platforms
python test/automated_hardware_selection.py --detect-hardware

# Comprehensive hardware detection and compatibility test
python test/test_comprehensive_hardware.py --test all

# Test hardware backends with specific model
python test/test_hardware_backend.py --backend [cpu|cuda|openvino|mps|rocm|webnn|webgpu|all] --model [model_name]

# Test resource pool with hardware awareness
python test/test_resource_pool.py --test hardware

# Test model family integration with web platform support
python test/test_resource_pool.py --test family --debug
```

### Web Platform Testing

```bash
# Run web platform integration tests
python test/test_model_integration.py

# Verify web platform integration is correct
python test/verify_web_platform_integration.py

# Generate a test with WebNN support
python test/merged_test_generator.py --generate bert --platform webnn

# Generate a test with WebGPU support
python test/merged_test_generator.py --generate vit --platform webgpu

# Run tests with database integration (DuckDB)
python test/run_web_platform_tests_with_db.py --models bert t5 vit --small-models --db-path ./benchmark_db.duckdb

# Use environment variable for database path
export BENCHMARK_DB_PATH=./benchmark_db.duckdb
python test/run_web_platform_tests_with_db.py --all-models --run-webgpu

# Run browser tests with direct database storage
python test/web_platform_test_runner.py --model bert --platform webnn --browser edge

# Disable JSON output (database storage only)
export DEPRECATE_JSON_OUTPUT=1 python test/web_platform_test_runner.py --model vit --platform webgpu

# Run with enhanced WebGPU compute shaders with DB storage
python test/web_platform_test_runner.py --model whisper --platform webgpu --compute-shaders

# Use database for parallel model loading results
python test/run_web_platform_tests_with_db.py --models llava clip --parallel-loading

# Store shader compilation metrics in database
WEBGPU_SHADER_PRECOMPILE=1 python test/web_platform_test_runner.py --model vit

# Generate web platform reports from database
python test/scripts/benchmark_db_query.py --report web_platform --format html --output web_report.html

# View advanced WebGPU features usage from database
python test/scripts/benchmark_db_query.py --report webgpu --format html --output webgpu_report.html

# Compare web vs native performance from database
python test/scripts/benchmark_db_query.py --sql "SELECT * FROM cross_platform_performance WHERE model_name='bert-base-uncased'" --format html
```

### Distributed Training Configuration
```bash
# Generate distributed training configuration
python test/hardware_selector.py --model-family text_generation --model-name t5-small --mode training --distributed --gpu-count 4

# Generate training benchmark configuration for a model
python test/run_training_benchmark.py --model bert-base-uncased --distributed --max-gpus 4 --output bert_benchmark.json

# List available sample models for benchmarking
python test/run_training_benchmark.py --list-models

# Generate a memory-optimized training configuration
python test/hardware_selector.py --model-family text_generation --model-name llama-7b --mode training --distributed --gpu-count 8 --max-memory-gb 24
```

### Model Benchmarking and Validation
```bash
# Run comprehensive benchmarks for all 13 high-priority models across all hardware platforms
python test/benchmark_all_key_models.py --output-dir ./benchmark_results

# Use smaller model variants for faster testing
python test/benchmark_all_key_models.py --small-models --output-dir ./benchmark_results

# Test specific hardware platforms
python test/benchmark_all_key_models.py --hardware cpu cuda openvino --output-dir ./benchmark_results

# Automatically fix implementation issues
python test/benchmark_all_key_models.py --debug --output-dir ./benchmark_results

# Run standard model benchmarks with database integration
python test/run_model_benchmarks.py --output-dir ./benchmark_results --db-path ./benchmark_db.duckdb

# Test on specific hardware platforms with small model set
python test/run_model_benchmarks.py --hardware cpu cuda --models-set small --db-path ./benchmark_db.duckdb

# Run benchmarks without storing in database
python test/run_model_benchmarks.py --hardware cpu --models-set small --no-db-store

# Generate database visualizations from benchmark results
python test/run_model_benchmarks.py --hardware cuda --visualize-from-db

# Manual model functionality verification
python test/verify_model_functionality.py --models bert t5 vit --hardware cpu cuda

# Run detailed hardware benchmarks
python test/hardware_benchmark_runner.py --model-families embedding text_generation --hardware cpu cuda
```

### Benchmark Database and Result Management
```bash
# Convert benchmark JSON files to Parquet/DuckDB format
python test/scripts/benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb

# Consolidate test results across directories
python test/scripts/benchmark_db_converter.py --consolidate --categories performance hardware compatibility --output-db ./benchmark_db.duckdb

# Comprehensive data migration with validation
python test/scripts/benchmark_db_migration.py --migrate-all --db ./benchmark_db.duckdb --validate

# Create initial database schema with sample data
python test/scripts/create_benchmark_schema.py --output ./benchmark_db.duckdb --sample-data

# Database maintenance and optimization
python test/scripts/benchmark_db_maintenance.py --optimize-db --vacuum --db ./benchmark_db.duckdb

# Create database backup with compression
python test/scripts/benchmark_db_maintenance.py --backup --backup-dir ./db_backups --backup-compress

# Check database integrity
python test/scripts/benchmark_db_maintenance.py --check-integrity --db ./benchmark_db.duckdb

# Generate migration statistics report
python test/scripts/benchmark_db_maintenance.py --migration-stats --output migration_report.json

# Purge old database backups based on retention policy
python test/scripts/benchmark_db_maintenance.py --purge-backups --backup-retention 30 --backup-dir ./db_backups

# Query benchmark database with SQL
python test/scripts/benchmark_db_query.py --sql "SELECT model_name, hardware_type, AVG(throughput_items_per_second) FROM performance_results JOIN models USING(model_id) JOIN hardware_platforms USING(hardware_id) GROUP BY model_name, hardware_type"

# Generate reports from DuckDB benchmark database
python test/scripts/benchmark_db_query.py --report performance --format html --output benchmark_report.html
python test/scripts/benchmark_db_query.py --report hardware --format html --output hardware_report.html
python test/scripts/benchmark_db_query.py --report compatibility --format html --output compatibility_matrix.html

# Compare hardware platforms for a specific model
python test/scripts/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware --output bert_hardware_comparison.png

# Compare models on a specific hardware platform
python test/scripts/benchmark_db_query.py --hardware cuda --metric throughput --compare-models --output cuda_model_comparison.png

# Plot performance trends over time
python test/scripts/benchmark_db_query.py --trend performance --model bert-base-uncased --hardware cuda --metric throughput --format chart

# Export data from the database
python test/scripts/benchmark_db_query.py --sql "SELECT * FROM performance_results" --format csv --output performance_data.csv

# Run benchmarks with direct database storage
python test/run_benchmark_with_db.py --model bert-base-uncased --hardware cuda --batch-sizes 1,2,4,8,16

# Run local benchmark with CI integration (simulates CI workflow locally)
./test/run_local_benchmark_with_ci.sh --model bert-base-uncased --hardware cpu --simulate

# Run CI/CD benchmark workflow manually via GitHub CLI
gh workflow run benchmark_db_ci.yml --ref main -f test_model=bert-base-uncased -f hardware=cpu -f batch_size=1,2,4,8
```

## Performance Benchmarks

### Latest Performance Metrics

For detailed performance benchmarks, please refer to the following resources:
- Database dashboard: `http://localhost:8000/dashboard` (when running benchmark_db_api.py)
- API documentation: `http://localhost:8000/docs` (complete REST API for all benchmark data)
- Generated reports: 
  - `python test/scripts/benchmark_db_query.py --report summary --format html --output summary_report.html`
  - `python test/scripts/benchmark_db_query.py --compatibility-matrix --format html --output matrix.html`

Legacy documentation (being migrated to database):
- Hardware-specific benchmarks: `test/HARDWARE_BENCHMARKING_GUIDE.md`
- Model compression results: `test/MODEL_COMPRESSION_GUIDE.md`
- Training benchmarks: `test/TRAINING_BENCHMARKING_GUIDE.md`
- Web platform audio tests: `test/WEB_PLATFORM_AUDIO_TESTING_GUIDE.md`
- Hardware selection system: `test/HARDWARE_SELECTION_GUIDE.md`
- Web platform support: `test/README_WEB_PLATFORM_SUPPORT.md`

### Web Platform Performance Results

The March 2025 enhancements have significantly improved web platform performance:

| Model Type | WebNN vs. CPU | WebGPU vs. CPU | WebGPU March 2025 | Recommended Size |
|------------|--------------|----------------|-------------------|------------------|
| BERT Embeddings | 2.5-3.5x faster | 2-3x faster | 2.2-3.4x faster | Small-Medium |
| Vision Models | 3-4x faster | 3.5-5x faster | 4-6x faster | Any size |
| Small T5 | 1.5-2x faster | 1.3-1.8x faster | 1.5-2x faster | Small |
| Tiny LLAMA | 1.2-1.5x faster | 1.3-1.7x faster | 1.4-1.9x faster | Tiny (<1B) |
| Audio Models | Limited speedup | Limited speedup | 1.2-1.35x faster | Tiny-Small |

**March 2025 Improvement Highlights:**
- **WebGPU Compute Shaders**: 20-35% performance improvement for audio models
- **Shader Precompilation**: 30-45% faster initial load time
- **Parallel Model Loading**: 30-45% loading time reduction for multimodal models
- **Memory Optimizations**: 15-25% reduced memory footprint

For detailed web platform performance, run:
```bash
python test/scripts/benchmark_db_query.py --report web_platform --format html --output web_platform_report.html
```

*Note: Performance varies significantly based on hardware, browser version, and model size.*

### Test Results Database Architecture

The new DuckDB/Parquet-based database system consolidates all test results and provides:
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
  - `create_benchmark_schema.py`: Schema definition and initialization
  - `benchmark_db_converter.py`: JSON to database migration
  - `benchmark_db_updater.py`: Direct database writing interface
  - `benchmark_db_query.py`: Comprehensive query tool
  - `benchmark_db_maintenance.py`: Database optimization
  - `benchmark_db_api.py`: REST API and dashboard
  - `benchmark_db_performance.py`: Performance testing
  - `run_benchmark_with_db.py`: Example integration
  - `cleanup_test_results.py`: Automated migration utility

Documentation and guides:
- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [Database Migration Guide](DATABASE_MIGRATION_GUIDE.md)
- [Phase 16 Database Implementation](PHASE16_DATABASE_IMPLEMENTATION.md)
- [Web Platform Support](README_WEB_PLATFORM_SUPPORT.md)
- [Web Platform Integration Guide](web_platform_integration_guide.md)

### Hardware Selection and Performance Prediction System

The framework now includes a comprehensive hardware selection and performance prediction system that leverages machine learning and historical benchmark data to provide optimal hardware recommendations:

- **Hardware Selection**: Automatically determines the best hardware platform for a given model and task
- **Performance Prediction**: Predicts throughput, latency, and memory usage for any model-hardware combination
- **Cross-Platform Support**: Covers all supported hardware platforms including CUDA, ROCm, MPS, OpenVINO, WebNN, and WebGPU
- **Precision-Aware**: Considers different precision formats (fp32, fp16, int8) in recommendations
- **Visualization Tools**: Generates comparative visualizations for hardware performance analysis

For detailed information, see the [Hardware Selection Guide](HARDWARE_SELECTION_GUIDE.md).