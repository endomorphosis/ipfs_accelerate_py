# IPFS Accelerate Python Framework Documentation Index

Last Updated: May 12, 2025

This document provides a comprehensive index of all project documentation, organized by category and implementation phase.

## Recently Added Documentation

### WebNN/WebGPU Resource Pool May 2025 Enhancements (May 12, 2025)

The WebNN/WebGPU Resource Pool Integration has been enhanced with fault tolerance, performance optimization, and browser-aware resource management. New documentation includes:

- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md) - Comprehensive overview of May 2025 enhancements
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Updated main integration guide (May 2025)
- [WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md](WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md) - Guide to cross-browser model sharding with fault tolerance
- [WEB_RESOURCE_POOL_RECOVERY_GUIDE.md](WEB_RESOURCE_POOL_RECOVERY_GUIDE.md) - Enhanced error recovery mechanisms
- [WEB_PLATFORM_PERFORMANCE_HISTORY.md](WEB_PLATFORM_PERFORMANCE_HISTORY.md) - Guide to performance history tracking and trend analysis

### Distributed Testing Framework Advanced Fault Tolerance (May 12, 2025)

The Distributed Testing Framework has been enhanced with advanced fault tolerance mechanisms and comprehensive integration capabilities:

- [DISTRIBUTED_TESTING_INTEGRATION_PR.md](DISTRIBUTED_TESTING_INTEGRATION_PR.md) - Latest status update on advanced fault tolerance implementation
- [DISTRIBUTED_TESTING_GUIDE.md](DISTRIBUTED_TESTING_GUIDE.md) - Updated comprehensive user guide
- [FAULT_TOLERANCE_UPDATE.md](FAULT_TOLERANCE_UPDATE.md) - Previous update on fault tolerance implementation
- [distributed_testing/docs/ADVANCED_RECOVERY_STRATEGIES.md](distributed_testing/docs/ADVANCED_RECOVERY_STRATEGIES.md) - Advanced failure recovery mechanisms
- [distributed_testing/docs/PERFORMANCE_TREND_ANALYSIS.md](distributed_testing/docs/PERFORMANCE_TREND_ANALYSIS.md) - Documentation of the performance trend analysis system

## Recently Archived Documentation

### April 2025 Archive

The following documentation has been archived as part of the April 2025 cleanup:

- Performance reports older than 30 days have been moved to `archived_reports_april2025/`
- Outdated documentation files have been moved to `archived_documentation_april2025/`
- Each archived file has been marked with an archive notice

To access archived documentation, please check the appropriate archive directory.

## Current Hardware and Model Coverage Status (April 7, 2025)

### Hardware Backend Support

Based on the current implementation, the hardware compatibility matrix shows:

| Model Family | CUDA | ROCm (AMD) | MPS (Apple) | OpenVINO | WebNN | WebGPU | Notes |
|--------------|------|------------|-------------|----------|-------|--------|-------|
| Embedding (BERT, etc.) | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Fully supported on all hardware |
| Text Generation (LLMs) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | Memory requirements critical |
| Vision (ViT, CLIP, etc.) | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Full cross-platform support |
| Audio (Whisper, etc.) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | CUDA preferred, Web simulation added |
| Multimodal (LLaVA, etc.) | ✅ High | ⚠️ Limited | ✅ High | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | CUDA/MPS for production, others are limited |

### HuggingFace Model Coverage Summary

Overall coverage for 213+ HuggingFace model architectures:

| Hardware Platform | Coverage Percentage |
|-------------------|---------------------|
| CPU               | 100%                |
| CUDA              | 100%                |
| ROCm (AMD)        | 93%                 |
| MPS (Apple)       | 90%                 |
| OpenVINO          | 85%                 |
| Qualcomm          | 75%                 |
| WebNN             | 40%                 |
| WebGPU            | 40%                 |

For detailed model coverage information, see [CROSS_PLATFORM_TEST_COVERAGE.md](CROSS_PLATFORM_TEST_COVERAGE.md).

### Benchmark Database Status

The migration of benchmark and test scripts to use the DuckDB database system is now complete:

- **Migration Progress**: 100% Complete (17 total scripts)
- **Database Implementation**: Full schema with performance, hardware, and compatibility components
- **Benefits**: 50-80% size reduction, 5-20x faster queries, consolidated analysis

For full details on the database implementation, see [DATABASE_MIGRATION_STATUS.md](DATABASE_MIGRATION_STATUS.md).

## Current Focus Areas (May 2025)

The project is currently focused on two key areas:

### 1. WebGPU/WebNN Resource Pool Integration (85% Complete)

The WebNN/WebGPU Resource Pool Integration enables concurrent execution of multiple AI models across heterogeneous browser backends with these key features:

- **Fault-Tolerant Cross-Browser Model Sharding**: Ability to run large models distributed across multiple browser tabs
- **Performance-Aware Browser Selection**: Intelligent browser selection based on historical performance data
- **Performance History Tracking**: Time-series analysis of performance metrics
- **Enhanced Error Recovery**: Comprehensive recovery mechanisms with progressive strategies

Documentation:
- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md) - Latest enhancements
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Main integration guide
- [IPFS_RESOURCE_POOL_INTEGRATION_GUIDE.md](IPFS_RESOURCE_POOL_INTEGRATION_GUIDE.md) - IPFS integration guide

### 2. Distributed Testing Framework (90% Complete)

The Distributed Testing Framework enables parallel execution of tests across multiple machines with these key features:

- **Advanced Fault Tolerance**: Sophisticated recovery mechanisms with progressive strategies
- **Coordinator Redundancy**: High-availability clustering with automatic failover
- **Performance Trend Analysis**: Comprehensive tracking and analysis of performance metrics
- **CI/CD Integration**: Integration with GitHub Actions, GitLab CI, and Jenkins

Documentation:
- [DISTRIBUTED_TESTING_INTEGRATION_PR.md](DISTRIBUTED_TESTING_INTEGRATION_PR.md) - Latest status update
- [DISTRIBUTED_TESTING_GUIDE.md](DISTRIBUTED_TESTING_GUIDE.md) - Comprehensive user guide
- [DISTRIBUTED_TESTING_DESIGN.md](DISTRIBUTED_TESTING_DESIGN.md) - Detailed design document

## Phase 16 Documentation (Completed March 2025)

### Core Documentation

- [PHASE16_COMPLETION_REPORT.md](PHASE16_COMPLETION_REPORT.md) - Comprehensive report on the completed Phase 16 implementation (archived: [PHASE16_README.md](archived_phase16_docs/PHASE16_README.md))
- [PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md](PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md) - Detailed implementation status (100% complete)
- [PHASE16_COMPLETION_REPORT.md](PHASE16_COMPLETION_REPORT.md) - Final completion report (March 2025)
- [PHASE16_VERIFICATION_REPORT.md](PHASE16_VERIFICATION_REPORT.md) - Comprehensive verification testing results
- [PHASE16_ARCHIVED_DOCS.md](PHASE16_ARCHIVED_DOCS.md) - Reference for archived Phase 16 documentation

### Hardware Benchmarking and Performance Analysis

- [HARDWARE_BENCHMARKING_GUIDE.md](HARDWARE_BENCHMARKING_GUIDE.md) - Main hardware benchmarking documentation
- [HARDWARE_BENCHMARKING_GUIDE_PHASE16.md](HARDWARE_BENCHMARKING_GUIDE_PHASE16.md) - Phase 16 benchmarking enhancements
- [HARDWARE_SELECTION_GUIDE.md](HARDWARE_SELECTION_GUIDE.md) - Guide to hardware selection and optimization
- [HARDWARE_DETECTION_GUIDE.md](HARDWARE_DETECTION_GUIDE.md) - Hardware detection system documentation
- [HARDWARE_MODEL_INTEGRATION_GUIDE.md](HARDWARE_MODEL_INTEGRATION_GUIDE.md) - Hardware-model integration guide
- [HARDWARE_MODEL_VALIDATION_GUIDE.md](HARDWARE_MODEL_VALIDATION_GUIDE.md) - Model validation across hardware platforms
- [HARDWARE_PLATFORM_TEST_GUIDE.md](HARDWARE_PLATFORM_TEST_GUIDE.md) - Hardware platform testing guide
- [HARDWARE_MODEL_PREDICTOR_GUIDE.md](HARDWARE_MODEL_PREDICTOR_GUIDE.md) - Hardware performance prediction system
- [HARDWARE_IMPLEMENTATION_SUMMARY.md](HARDWARE_IMPLEMENTATION_SUMMARY.md) - Summary of hardware implementation status
- [APPLE_SILICON_GUIDE.md](APPLE_SILICON_GUIDE.md) - Apple Silicon (MPS) acceleration guide
- [QUALCOMM_INTEGRATION_GUIDE.md](QUALCOMM_INTEGRATION_GUIDE.md) - Qualcomm AI Engine integration guide
- [final_hardware_coverage_report.md](final_hardware_coverage_report.md) - Current hardware coverage status

### Comprehensive Benchmarking Documentation (March 2025)

- [benchmark_results/DOCUMENTATION_README.md](benchmark_results/DOCUMENTATION_README.md) - **NEW** Central index for all benchmark documentation
- [benchmark_results/NEXT_STEPS_BENCHMARKING_PLAN.md](benchmark_results/NEXT_STEPS_BENCHMARKING_PLAN.md) - **NEW** Detailed week-by-week execution plan for benchmarking
- [benchmark_results/MARCH_2025_BENCHMARK_PROGRESS.md](benchmark_results/MARCH_2025_BENCHMARK_PROGRESS.md) - **NEW** Current progress report with detailed Week 1 plan
- [benchmark_results/BENCHMARK_SUMMARY.md](benchmark_results/BENCHMARK_SUMMARY.md) - **NEW** Comprehensive summary of benchmark results
- [benchmark_results/BENCHMARK_COMMAND_CHEATSHEET.md](benchmark_results/BENCHMARK_COMMAND_CHEATSHEET.md) - **NEW** Quick reference for benchmark commands
- [run_comprehensive_benchmarks.py](run_comprehensive_benchmarks.py) - **NEW** Main script for running comprehensive benchmarks

### Database Implementation

- [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md) - Benchmark database architecture and usage
- [DATABASE_MIGRATION_GUIDE.md](DATABASE_MIGRATION_GUIDE.md) - Guide to migrating data to the database
- [DATABASE_MIGRATION_STATUS.md](DATABASE_MIGRATION_STATUS.md) - Status of database migration (100% complete)
- [PHASE16_DATABASE_IMPLEMENTATION.md](PHASE16_DATABASE_IMPLEMENTATION.md) - Database implementation details
- [run_incremental_benchmarks.py](run_incremental_benchmarks.py) - **NEW** Intelligent benchmark runner for identifying and running missing or outdated benchmarks

### Web Platform Integration

- [WEB_PLATFORM_INTEGRATION_GUIDE.md](WEB_PLATFORM_INTEGRATION_GUIDE.md) - Web platform integration guide
- [WEB_PLATFORM_INTEGRATION_SUMMARY.md](WEB_PLATFORM_INTEGRATION_SUMMARY.md) - Summary of web platform implementation
- [WEB_PLATFORM_OPTIMIZATION_GUIDE.md](WEB_PLATFORM_OPTIMIZATION_GUIDE.md) - Web platform optimization guide
- [WEB_PLATFORM_TESTING_GUIDE.md](WEB_PLATFORM_TESTING_GUIDE.md) - Guide to testing web platform implementations
- [WEB_PLATFORM_AUDIO_TESTING_GUIDE.md](WEB_PLATFORM_AUDIO_TESTING_GUIDE.md) - Web platform audio testing guide
- [WEB_PLATFORM_AUDIO_TESTING_SUMMARY.md](WEB_PLATFORM_AUDIO_TESTING_SUMMARY.md) - Summary of audio testing implementation
- [web_platform_integration_quick_reference.md](web_platform_integration_quick_reference.md) - Quick reference for web integration

### WebNN/WebGPU Documentation

- [REAL_WEBNN_WEBGPU_IMPLEMENTATION.md](REAL_WEBNN_WEBGPU_IMPLEMENTATION.md) - Current implementation details for real WebNN/WebGPU
- [REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md](REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md) - Latest implementation updates (March 2025)
- [REAL_WEBNN_WEBGPU_BENCHMARKING_GUIDE.md](REAL_WEBNN_WEBGPU_BENCHMARKING_GUIDE.md) - Comprehensive benchmarking guide
- [WEBNN_WEBGPU_BENCHMARK_README.md](WEBNN_WEBGPU_BENCHMARK_README.md) - Overview of the benchmark system
- [WEBNN_WEBGPU_DATABASE_INTEGRATION.md](WEBNN_WEBGPU_DATABASE_INTEGRATION.md) - Database integration guide
- [WEBNN_WEBGPU_ARCHIVED_DOCS.md](WEBNN_WEBGPU_ARCHIVED_DOCS.md) - Reference for archived WebNN/WebGPU documentation

### Cross-Platform Testing

- [CROSS_PLATFORM_TEST_COVERAGE.md](CROSS_PLATFORM_TEST_COVERAGE.md) - Cross-platform test coverage status
- [PHASE16_CROSS_PLATFORM_TESTING.md](PHASE16_CROSS_PLATFORM_TESTING.md) - Cross-platform testing implementation details
- [HF_COMPREHENSIVE_TESTING_GUIDE.md](HF_COMPREHENSIVE_TESTING_GUIDE.md) - Guide to testing all 300+ HuggingFace models across hardware platforms

### Training Mode Benchmarking

- [TRAINING_BENCHMARKING_GUIDE.md](TRAINING_BENCHMARKING_GUIDE.md) - Guide to training mode benchmarking
- [DISTRIBUTED_TRAINING_GUIDE.md](DISTRIBUTED_TRAINING_GUIDE.md) - Guide to distributed training capabilities

## Implementation Notes and Plans

- [PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md](PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md) - Final implementation summary for Phase 16
- [WEB_PLATFORM_IMPLEMENTATION_PLAN.md](WEB_PLATFORM_IMPLEMENTATION_PLAN.md) - Web platform implementation plan
- [WEB_PLATFORM_IMPLEMENTATION_SUMMARY.md](WEB_PLATFORM_IMPLEMENTATION_SUMMARY.md) - Web implementation summary
- [WEB_PLATFORM_IMPLEMENTATION_NEXT_STEPS.md](WEB_PLATFORM_IMPLEMENTATION_NEXT_STEPS.md) - Next implementation steps

## Model-Specific Documentation

- [model_specific_optimizations/text_models.md](docs/model_specific_optimizations/text_models.md) - Text model optimization guide
- [model_specific_optimizations/audio_models.md](docs/model_specific_optimizations/audio_models.md) - Audio model optimization guide
- [model_specific_optimizations/multimodal_models.md](docs/model_specific_optimizations/multimodal_models.md) - Multimodal model optimization guide
- [MODEL_COMPRESSION_GUIDE.md](MODEL_COMPRESSION_GUIDE.md) - Guide to model compression techniques
- [MODEL_FAMILY_GUIDE.md](MODEL_FAMILY_GUIDE.md) - Guide to model family classification
- [MODEL_FAMILY_CLASSIFIER_GUIDE.md](MODEL_FAMILY_CLASSIFIER_GUIDE.md) - Model family classifier documentation

## Advanced Features Documentation (May 2025)

- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md) - Latest enhancements to WebNN/WebGPU Resource Pool (NEW - May 12, 2025)
- [DISTRIBUTED_TESTING_INTEGRATION_PR.md](DISTRIBUTED_TESTING_INTEGRATION_PR.md) - Status of Distributed Testing Framework advanced fault tolerance (NEW - May 12, 2025)
- [MODEL_FILE_VERIFICATION_README.md](MODEL_FILE_VERIFICATION_README.md) - Comprehensive guide to the Model File Verification and Conversion Pipeline
- [ARCHIVE_STRUCTURE.md](ARCHIVE_STRUCTURE.md) - Documentation of the archive directory structure and management
- [DOCUMENTATION_CLEANUP_GUIDE.md](DOCUMENTATION_CLEANUP_GUIDE.md) - Guide for documentation and report cleanup
- [TIME_SERIES_PERFORMANCE_GUIDE.md](TIME_SERIES_PERFORMANCE_GUIDE.md) - Time-series performance tracking system
- [IPFS_ACCELERATION_TESTING.md](IPFS_ACCELERATION_TESTING.md) - IPFS acceleration testing with DuckDB integration
- [MODEL_REGISTRY_INTEGRATION.md](MODEL_REGISTRY_INTEGRATION.md) - Model registry integration system
- [MOBILE_EDGE_EXPANSION_PLAN.md](MOBILE_EDGE_EXPANSION_PLAN.md) - Mobile/edge support expansion plan
- [BATTERY_IMPACT_ANALYSIS.md](BATTERY_IMPACT_ANALYSIS.md) - Battery impact analysis methodology
- [SIMULATION_DETECTION_IMPROVEMENTS.md](SIMULATION_DETECTION_IMPROVEMENTS.md) - Simulation detection and flagging improvements
- [DOCUMENTATION_CLEANUP_SUMMARY.md](DOCUMENTATION_CLEANUP_SUMMARY.md) - Summary of documentation and report cleanup
- [NEXT_STEPS_IMPLEMENTATION.md](NEXT_STEPS_IMPLEMENTATION.md) - Implementation guide for next steps
- [NEXT_STEPS.md](NEXT_STEPS.md) - Next steps and roadmap for the framework

### Implementation Files

- [WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md](WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md) - Guide to cross-browser model sharding with fault tolerance (NEW - May 12, 2025)
- [distributed_testing/docs/ADVANCED_RECOVERY_STRATEGIES.md](distributed_testing/docs/ADVANCED_RECOVERY_STRATEGIES.md) - Advanced failure recovery mechanisms
- [model_file_verification.py](model_file_verification.py) - Core implementation of the Model File Verification and Conversion Pipeline
- [benchmark_model_verification.py](benchmark_model_verification.py) - Integration of the verification system with benchmarks
- [run_model_verification.sh](run_model_verification.sh) - Script to demonstrate model verification usage
- [archive/archive_backups.sh](archive/archive_backups.sh) - Script for archiving backup files and old reports
- [archive/archive_stale.sh](archive/archive_stale.sh) - Script for archiving stale scripts and documentation
- [archive_old_documentation.py](archive_old_documentation.py) - Utility for archiving outdated documentation
- [cleanup_stale_reports.py](cleanup_stale_reports.py) - Tool for cleaning up stale benchmark reports
- [run_documentation_cleanup.sh](run_documentation_cleanup.sh) - Script to run all documentation cleanup tools
- [time_series_performance.py](time_series_performance.py) - Time-series performance tracking implementation
- [test_ipfs_accelerate.py](test_ipfs_accelerate.py) - IPFS acceleration testing implementation with DuckDB integration
- [model_registry_integration.py](model_registry_integration.py) - Model registry integration implementation
- [mobile_edge_expansion_plan.py](mobile_edge_expansion_plan.py) - Mobile/edge support expansion implementation
- [test_model_registry_integration.py](test_model_registry_integration.py) - Test script for model registry integration
- [test_mobile_edge_expansion.py](test_mobile_edge_expansion.py) - Test script for mobile/edge support expansion
- [test_simulation_detection.py](test_simulation_detection.py) - Test script for simulation detection and flagging
- [test_simulation_awareness.py](test_simulation_awareness.py) - Test script for report simulation awareness
- [run_cleanup_stale_reports.py](run_cleanup_stale_reports.py) - Script to automate the cleanup of stale benchmark reports
- [update_db_schema_for_simulation.py](update_db_schema_for_simulation.py) - Script to update database schema with simulation flags
- [qnn_simulation_helper.py](qnn_simulation_helper.py) - Utility for controlling QNN simulation

## API and Integration Documentation

- [unified_framework_api.md](docs/unified_framework_api.md) - Comprehensive API reference for the unified framework
- [websocket_protocol_spec.md](docs/websocket_protocol_spec.md) - WebSocket protocol specification for streaming inference
- [api_reference/webgpu_streaming_inference.md](docs/api_reference/webgpu_streaming_inference.md) - WebGPU streaming inference API reference
- [RESOURCE_POOL_GUIDE.md](RESOURCE_POOL_GUIDE.md) - Guide to the ResourcePool system
- [TEMPLATE_INHERITANCE_GUIDE.md](TEMPLATE_INHERITANCE_GUIDE.md) - Template inheritance system documentation
- [MODALITY_TEMPLATE_GUIDE.md](MODALITY_TEMPLATE_GUIDE.md) - Modality-specific template documentation
- [TEMPLATE_GENERATOR_README.md](TEMPLATE_GENERATOR_README.md) - Template generator documentation
- [INTEGRATED_SKILLSET_GENERATOR_GUIDE.md](INTEGRATED_SKILLSET_GENERATOR_GUIDE.md) - Skillset generator documentation

## Archived Documentation

Older documentation files have been archived in the following directories:
- `archive/old_documentation/` - Older documentation files (March 10, 2025)
- `archive/old_reports/` - Old benchmark reports and results files (March 10, 2025)
- `archive/stale_scripts/` - Deprecated Python scripts that are no longer in active use (March 10, 2025)
- `archive/backup_files/` - Backup files with original directory structure preserved (March 10, 2025)
- `archived_md_files/` - Legacy documentation from previous phases
- `archived_documentation_april2025/` - Recently archived documentation (April 2025)
- `archived_reports_april2025/` - Recently archived performance reports (April 2025)
- `archived_stale_reports/` - Problematic benchmark reports identified during cleanup

See [ARCHIVE_STRUCTURE.md](ARCHIVE_STRUCTURE.md) for a complete guide to the archive directory structure and management procedures.

## How to Use This Index

1. **For New Users**: Start with the PHASE16_COMPLETION_REPORT.md and the main guide for your area of interest
2. **For Implementation Status**: See PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md and PHASE16_COMPLETION_REPORT.md
3. **For Verification Results**: Review PHASE16_VERIFICATION_REPORT.md for testing outcomes
4. **For Technical Details**: See the specialized guides for each component
5. **For Documentation Maintenance**: Refer to DOCUMENTATION_CLEANUP_GUIDE.md for cleanup procedures

## Documentation Maintenance

This documentation index is regularly updated as the project evolves. When adding new documentation:

1. Add a reference to this index in the appropriate category
2. Archive outdated documents using the `archive_old_documentation.py` script
3. Run `./run_documentation_cleanup.sh` periodically to maintain a clean documentation structure
4. Update status indicators in active documentation files

For any documentation questions or issues, please refer to CLAUDE.md for additional guidance or the DOCUMENTATION_CLEANUP_GUIDE.md for maintenance procedures.