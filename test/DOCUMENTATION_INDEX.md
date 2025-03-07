# IPFS Accelerate Python Framework Documentation Index

Last Updated: April 7, 2025

This document provides a comprehensive index of all project documentation, organized by category and implementation phase.

## Recently Archived Documentation (April 2025)

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

## Phase 16 Documentation (Current Phase)

### Core Documentation

- [PHASE16_README.md](PHASE16_README.md) - Central reference for all Phase 16 documentation
- [PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md](PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md) - Detailed implementation status (100% complete)
- [PHASE16_PROGRESS_UPDATE.md](PHASE16_PROGRESS_UPDATE.md) - Current progress status and next steps
- [PHASE16_COMPLETION_TASKS.md](PHASE16_COMPLETION_TASKS.md) - Tasks needed to complete Phase 16
- [PHASE_16_IMPLEMENTATION_PLAN.md](PHASE_16_IMPLEMENTATION_PLAN.md) - Original implementation plan (reference only)

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

### Cross-Platform Testing

- [CROSS_PLATFORM_TEST_COVERAGE.md](CROSS_PLATFORM_TEST_COVERAGE.md) - Cross-platform test coverage status
- [PHASE16_CROSS_PLATFORM_TESTING.md](PHASE16_CROSS_PLATFORM_TESTING.md) - Cross-platform testing implementation details
- [HF_COMPREHENSIVE_TESTING_GUIDE.md](HF_COMPREHENSIVE_TESTING_GUIDE.md) - Guide to testing all 300+ HuggingFace models across hardware platforms

### Training Mode Benchmarking

- [TRAINING_BENCHMARKING_GUIDE.md](TRAINING_BENCHMARKING_GUIDE.md) - Guide to training mode benchmarking
- [DISTRIBUTED_TRAINING_GUIDE.md](DISTRIBUTED_TRAINING_GUIDE.md) - Guide to distributed training capabilities

## Implementation Notes and Plans

- [PHASE16_IMPROVEMENTS.md](PHASE16_IMPROVEMENTS.md) - Improvements made in Phase 16
- [PHASE16_IMPLEMENTATION_UPDATE.md](PHASE16_IMPLEMENTATION_UPDATE.md) - Implementation updates
- [WEB_PLATFORM_IMPLEMENTATION_PLAN.md](WEB_PLATFORM_IMPLEMENTATION_PLAN.md) - Web platform implementation plan
- [WEB_PLATFORM_IMPLEMENTATION_PROGRESS.md](WEB_PLATFORM_IMPLEMENTATION_PROGRESS.md) - Web implementation progress
- [WEB_PLATFORM_IMPLEMENTATION_PRIORITIES.md](WEB_PLATFORM_IMPLEMENTATION_PRIORITIES.md) - Implementation priorities
- [WEB_PLATFORM_IMPLEMENTATION_NEXT_STEPS.md](WEB_PLATFORM_IMPLEMENTATION_NEXT_STEPS.md) - Next implementation steps

## Model-Specific Documentation

- [model_specific_optimizations/text_models.md](docs/model_specific_optimizations/text_models.md) - Text model optimization guide
- [model_specific_optimizations/audio_models.md](docs/model_specific_optimizations/audio_models.md) - Audio model optimization guide
- [model_specific_optimizations/multimodal_models.md](docs/model_specific_optimizations/multimodal_models.md) - Multimodal model optimization guide
- [MODEL_COMPRESSION_GUIDE.md](MODEL_COMPRESSION_GUIDE.md) - Guide to model compression techniques
- [MODEL_FAMILY_GUIDE.md](MODEL_FAMILY_GUIDE.md) - Guide to model family classification
- [MODEL_FAMILY_CLASSIFIER_GUIDE.md](MODEL_FAMILY_CLASSIFIER_GUIDE.md) - Model family classifier documentation

## Advanced Features Documentation (April 2025)

- [DOCUMENTATION_CLEANUP_GUIDE.md](DOCUMENTATION_CLEANUP_GUIDE.md) - Guide for documentation and report cleanup (NEW - April 7, 2025)
- [TIME_SERIES_PERFORMANCE_GUIDE.md](TIME_SERIES_PERFORMANCE_GUIDE.md) - Time-series performance tracking system
- [IPFS_ACCELERATION_TESTING.md](IPFS_ACCELERATION_TESTING.md) - IPFS acceleration testing with DuckDB integration (Updated March 2025)
- [MODEL_REGISTRY_INTEGRATION.md](MODEL_REGISTRY_INTEGRATION.md) - Model registry integration system
- [MOBILE_EDGE_EXPANSION_PLAN.md](MOBILE_EDGE_EXPANSION_PLAN.md) - Mobile/edge support expansion plan
- [BATTERY_IMPACT_ANALYSIS.md](BATTERY_IMPACT_ANALYSIS.md) - Battery impact analysis methodology
- [SIMULATION_DETECTION_IMPROVEMENTS.md](SIMULATION_DETECTION_IMPROVEMENTS.md) - Simulation detection and flagging improvements (Updated April 7, 2025)
- [STALE_REPORTS_CLEANUP_COMPLETED.md](STALE_REPORTS_CLEANUP_COMPLETED.md) - Completion report for stale reports cleanup (March 6, 2025)
- [STALE_BENCHMARK_REPORTS_FIXED.md](STALE_BENCHMARK_REPORTS_FIXED.md) - Detailed documentation of the stale reports cleanup task (March 6, 2025)
- [PHASE16_CLEANUP_SUMMARY.md](PHASE16_CLEANUP_SUMMARY.md) - Summary of Phase 16 cleanup activities (March 6, 2025)
- [DOCUMENTATION_CLEANUP_SUMMARY.md](DOCUMENTATION_CLEANUP_SUMMARY.md) - Summary of documentation and report cleanup (April 7, 2025)
- [NEXT_STEPS_IMPLEMENTATION.md](NEXT_STEPS_IMPLEMENTATION.md) - Implementation guide for next steps (March 2025)
- [NEXT_STEPS.md](NEXT_STEPS.md) - Next steps and roadmap for the framework

### Implementation Files

- [archive_old_documentation.py](archive_old_documentation.py) - Utility for archiving outdated documentation (NEW - April 7, 2025)
- [cleanup_stale_reports.py](cleanup_stale_reports.py) - Tool for cleaning up stale benchmark reports (UPDATED - April 7, 2025)
- [run_documentation_cleanup.sh](run_documentation_cleanup.sh) - Script to run all documentation cleanup tools (NEW - April 7, 2025)
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
- `archived_md_files/` - Legacy documentation from previous phases
- `archived_documentation_april2025/` - Recently archived documentation (April 2025)
- `archived_reports_april2025/` - Recently archived performance reports (April 2025)
- `archived_stale_reports/` - Problematic benchmark reports identified during cleanup

Each archived file has been marked with an archive notice for clarity.

## How to Use This Index

1. **For New Users**: Start with the PHASE16_README.md and the main guide for your area of interest
2. **For Implementation Status**: See PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md and PHASE16_PROGRESS_UPDATE.md
3. **For Task Planning**: Refer to PHASE16_COMPLETION_TASKS.md for outstanding implementation tasks
4. **For Technical Details**: See the specialized guides for each component
5. **For Documentation Maintenance**: Refer to DOCUMENTATION_CLEANUP_GUIDE.md for cleanup procedures

## Documentation Maintenance

This documentation index is regularly updated as the project evolves. When adding new documentation:

1. Add a reference to this index in the appropriate category
2. Archive outdated documents using the `archive_old_documentation.py` script
3. Run `./run_documentation_cleanup.sh` periodically to maintain a clean documentation structure
4. Update status indicators in active documentation files

For any documentation questions or issues, please refer to CLAUDE.md for additional guidance or the DOCUMENTATION_CLEANUP_GUIDE.md for maintenance procedures.
