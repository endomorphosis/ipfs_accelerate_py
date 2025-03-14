# IPFS Accelerate Framework Documentation Index

Last Updated: July 8, 2025

This document provides a comprehensive index of all project documentation, organized by category and implementation phase.

## Recently Added Documentation

### Samsung Exynos NPU Support (NEW - March 14, 2025)

The IPFS Accelerate Python Framework now includes comprehensive support for Samsung Exynos Neural Processing Units (NPUs), enabling detection, benchmarking, model conversion, and optimization for Samsung NPU hardware:

- [SAMSUNG_NPU_SUPPORT_GUIDE.md](SAMSUNG_NPU_SUPPORT_GUIDE.md) - **NEW** Comprehensive guide to Samsung NPU support
- [SAMSUNG_NPU_TEST_GUIDE.md](SAMSUNG_NPU_TEST_GUIDE.md) - **NEW** Guide for testing Samsung NPU support
- [SAMSUNG_NPU_DOCUMENTATION_UPDATES.md](SAMSUNG_NPU_DOCUMENTATION_UPDATES.md) - **NEW** Summary of documentation improvements
- [samsung_support.py](samsung_support.py) - **NEW** Core implementation of Samsung NPU detection and capabilities
- [test_samsung_npu_comparison.py](test_samsung_npu_comparison.py) - **NEW** Hardware comparison tool for Samsung NPU
- [test_samsung_support.py](test_samsung_support.py) - **NEW** Test suite for Samsung NPU support
- [test_samsung_npu_basic.py](test_samsung_npu_basic.py) - **NEW** Basic test for Samsung NPU support
- [test_minimal_samsung.py](test_minimal_samsung.py) - **NEW** Ultra-minimal test for Samsung NPU core functionality
- [test_mobile_npu_comparison.py](test_mobile_npu_comparison.py) - **UPDATED** Mobile NPU comparison with Samsung NPU support
- [requirements_samsung.txt](requirements_samsung.txt) - **NEW** Dependency requirements for Samsung NPU support

This implementation provides comprehensive support for all Samsung Exynos NPU-equipped devices, including model compatibility assessment, performance benchmarking, power efficiency analysis, and integration with the centralized hardware detection system. The system supports six Samsung Exynos chipsets from entry-level to flagship devices and provides detailed optimization recommendations specific to each chipset. The implementation includes a comprehensive testing suite, from ultra-minimal tests to detailed hardware comparisons.

The documentation has been significantly enhanced with detailed sections on thermal management, model compatibility, framework ecosystem integration, and troubleshooting. Advanced usage examples are provided for optimal performance on Samsung devices. A dedicated requirements file (`requirements_samsung.txt`) has been created to easily install all dependencies needed for Samsung NPU support, with clear documentation on which dependencies are needed for different levels of functionality.

### Simulation Accuracy and Validation Framework (NEW - July 8, 2025)

The Simulation Accuracy and Validation Framework is a comprehensive system for validating, calibrating, and monitoring hardware simulation accuracy, ensuring that simulation results closely match real hardware performance:

- [SIMULATION_ACCURACY_VALIDATION_DESIGN.md](SIMULATION_ACCURACY_VALIDATION_DESIGN.md) - **NEW** Design document for the Simulation Accuracy and Validation Framework
- [duckdb_api/simulation_validation/README.md](duckdb_api/simulation_validation/README.md) - **NEW** Overview of the simulation validation directory structure and components
- [duckdb_api/simulation_validation/core/base.py](duckdb_api/simulation_validation/core/base.py) - **NEW** Core interfaces and abstract classes for the framework
- [duckdb_api/simulation_validation/core/schema.py](duckdb_api/simulation_validation/core/schema.py) - **NEW** Database schema for storing simulation and validation data
- [test_simulation_validation_foundation.py](test_simulation_validation_foundation.py) - **NEW** Test script for the foundation of the framework

The framework provides tools for comparing simulation results with real hardware measurements, statistical validation of simulation accuracy, calibration of simulation parameters, detection of simulation drift, and comprehensive reporting. Implementation is planned to begin in July 2025 with a target completion date of October 15, 2025.

### Monitoring Dashboard Integration for Advanced Visualization (NEW - July 5, 2025)

The Advanced Visualization System has been fully integrated with the Monitoring Dashboard, enabling centralized visualization management, real-time updates, and improved collaboration:

- [MONITORING_DASHBOARD_INTEGRATION_GUIDE.md](MONITORING_DASHBOARD_INTEGRATION_GUIDE.md) - **NEW** Guide for integrating visualizations with the monitoring dashboard
- [dashboard_enhanced_visualization.py](duckdb_api/visualization/dashboard_enhanced_visualization.py) - **NEW** Implementation of the enhanced visualization system with dashboard integration
- [monitor_dashboard_integration.py](duckdb_api/visualization/advanced_visualization/monitor_dashboard_integration.py) - **NEW** Core implementation of the dashboard integration
- [test_dashboard_enhanced_visualization.py](test_dashboard_enhanced_visualization.py) - **NEW** Comprehensive test suite for the dashboard integration
- [run_monitoring_dashboard_integration.py](run_monitoring_dashboard_integration.py) - **NEW** Command-line tool for dashboard integration

The integration allows visualizations to be automatically synchronized with a central dashboard, enabling real-time updates via WebSocket, dashboard panel creation, and snapshot export/import capabilities. This marks the completion of the Advanced Visualization System, a major milestone in the project roadmap.

### BERT Model Implementation with Hardware Acceleration (NEW - April 5, 2025)

The BERT (Bidirectional Encoder Representations from Transformers) model has been implemented in the TypeScript SDK with full hardware acceleration support on WebGPU and WebNN backends:

- [BERT_MODEL_DOCUMENTATION.md](ipfs_accelerate_js/docs/models/BERT_MODEL_DOCUMENTATION.md) - **NEW** Comprehensive documentation for the BERT model implementation
- [bert.ts](ipfs_accelerate_js/src/model/transformers/bert.ts) - **NEW** Core implementation of the BERT model with hardware acceleration
- [bert_example.html](ipfs_accelerate_js/examples/browser/models/bert_example.html) - **NEW** Interactive browser demo for BERT model inference
- [bert_example.ts](ipfs_accelerate_js/examples/browser/models/bert_example.ts) - **NEW** TypeScript implementation of the BERT model example
- [bert_test.ts](ipfs_accelerate_js/test/models/transformers/bert_test.ts) - **NEW** Comprehensive test suite for the BERT model

The implementation provides a memory-efficient BERT model with cross-model tensor sharing, optimized matrix operations, and support for different hardware backends. This marks significant progress in the TypeScript SDK implementation, advancing the project to 97% completion.

### Browser-Specific WebGPU Optimization (NEW - March 13, 2025)

The browser-specific optimization system automatically tunes WebGPU compute shader parameters based on the detected browser type and hardware vendor, ensuring optimal performance across different environments:

- [WebGPU_BROWSER_OPTIMIZATIONS.md](WebGPU_BROWSER_OPTIMIZATIONS.md) - **NEW** Comprehensive guide to browser-specific optimizations for WebGPU
- [browser_optimized_matmul_example.html](ipfs_accelerate_js/examples/browser/basic/browser_optimized_matmul_example.html) - **NEW** Interactive example of browser-optimized matrix multiplication
- [browser_optimized_matmul_example.ts](ipfs_accelerate_js/examples/browser/basic/browser_optimized_matmul_example.ts) - **NEW** TypeScript implementation of the browser-optimized example
- [browser_optimized_operations.ts](ipfs_accelerate_js/src/hardware/webgpu/browser_optimized_operations.ts) - **NEW** Core implementation of the browser optimization system

The system automatically detects browser type (Chrome, Firefox, Safari, Edge) and hardware vendor (NVIDIA, AMD, Intel, Apple, Qualcomm, ARM) to select optimal parameters for WebGPU compute shaders. Performance improvements of up to 3x have been observed for specific operations like audio processing in Firefox and matrix multiplication in Chrome.

### Cross-Model Tensor Sharing System (NEW - March 28, 2025)

The Cross-Model Tensor Sharing system enables efficient sharing of tensors between multiple models, significantly improving memory efficiency and performance for multi-model workloads:

- [CROSS_MODEL_TENSOR_SHARING_GUIDE.md](CROSS_MODEL_TENSOR_SHARING_GUIDE.md) - **NEW** Comprehensive guide to the Cross-Model Tensor Sharing system
- [ipfs_accelerate_js_tensor_sharing_integration.ts](ipfs_accelerate_js_tensor_sharing_integration.ts) - **NEW** Implementation of the TensorSharingIntegration with Storage Manager integration
- [ipfs_accelerate_js_tensor_sharing_example.ts](ipfs_accelerate_js_tensor_sharing_example.ts) - **NEW** Example usage of the Cross-Model Tensor Sharing system
- [TensorSharingDemo.html](TensorSharingDemo.html) - **NEW** Interactive browser demo for tensor sharing visualization

This implementation provides shared tensor memory across models, reference counting for efficient memory management, zero-copy tensor views, persistent storage through IndexedDB, and seamless WebNN integration. Benchmarks show up to 30% memory reduction and 30% faster inference when sharing tensors between models.

### Monitoring Dashboard (NEW - March 17, 2025)

A comprehensive monitoring dashboard for the Distributed Testing Framework has been implemented, providing real-time monitoring and visualization:

- [MONITORING_DASHBOARD_GUIDE.md](duckdb_api/distributed_testing/MONITORING_DASHBOARD_GUIDE.md) - **NEW** Comprehensive guide to the monitoring dashboard
- [monitoring_dashboard.py](duckdb_api/distributed_testing/dashboard/monitoring_dashboard.py) - **NEW** Implementation of the comprehensive monitoring dashboard
- [run_monitoring_dashboard.py](duckdb_api/distributed_testing/run_monitoring_dashboard.py) - **NEW** Runner script for easily starting the monitoring dashboard
- [tests/test_monitoring_dashboard.py](duckdb_api/distributed_testing/dashboard/tests/test_monitoring_dashboard.py) - **NEW** Test suite for the monitoring dashboard

This implementation provides a comprehensive web-based monitoring interface with real-time metrics visualization, WebSocket-based live updates, system topology visualization, integrated alert system, task tracking, fault tolerance integration, and a comprehensive API. This completes the Distributed Testing Framework ahead of schedule.

### Hardware Fault Tolerance Enhancements (NEW - March 13, 2025)

The Distributed Testing Framework has been enhanced with a comprehensive hardware-aware fault tolerance system that includes advanced features ahead of schedule:

- [HARDWARE_FAULT_TOLERANCE_OVERVIEW.md](HARDWARE_FAULT_TOLERANCE_OVERVIEW.md) - **NEW** Executive summary and comprehensive overview
- [HARDWARE_FAULT_TOLERANCE_GUIDE.md](duckdb_api/distributed_testing/HARDWARE_FAULT_TOLERANCE_GUIDE.md) - Complete guide to the hardware-aware fault tolerance system
- [HARDWARE_FAULT_TOLERANCE_ENHANCEMENTS.md](HARDWARE_FAULT_TOLERANCE_ENHANCEMENTS.md) - **NEW** Documentation of ML pattern detection and visualization enhancements
- [hardware_aware_fault_tolerance.py](duckdb_api/distributed_testing/hardware_aware_fault_tolerance.py) - Implementation of hardware-specific recovery strategies
- [ml_pattern_detection.py](duckdb_api/distributed_testing/ml_pattern_detection.py) - **NEW** Machine learning-based pattern detection system
- [fault_tolerance_visualization.py](duckdb_api/distributed_testing/fault_tolerance_visualization.py) - **NEW** Comprehensive visualization system
- [run_fault_tolerance_visualization.py](duckdb_api/distributed_testing/run_fault_tolerance_visualization.py) - **NEW** Visualization tool with simulation capabilities
- [tests/test_hardware_fault_tolerance.py](duckdb_api/distributed_testing/tests/test_hardware_fault_tolerance.py) - Comprehensive test suite
- [tests/test_fault_tolerance_visualization.py](duckdb_api/distributed_testing/tests/test_fault_tolerance_visualization.py) - **NEW** Visualization test suite
- [run_fault_tolerance_tests.sh](duckdb_api/distributed_testing/run_fault_tolerance_tests.sh) - Enhanced test script supporting hardware fault tolerance tests

This implementation provides specialized recovery strategies for different hardware types (CPUs, GPUs, TPUs, browsers with WebGPU/WebNN), intelligent retry policies with ML-based pattern detection, comprehensive visualization and reporting, failure pattern detection, task state persistence, and checkpoint/resume capabilities. The system is deeply integrated with the heterogeneous hardware support and scheduler.

### Heterogeneous Hardware Support (NEW - March 15, 2025)

The Distributed Testing Framework has been enhanced with comprehensive support for heterogeneous hardware environments:

- [HETEROGENEOUS_HARDWARE_GUIDE.md](duckdb_api/distributed_testing/HETEROGENEOUS_HARDWARE_GUIDE.md) - Comprehensive guide to heterogeneous hardware support
- [hardware_taxonomy.py](duckdb_api/distributed_testing/hardware_taxonomy.py) - Advanced hardware taxonomy system for device classification
- [enhanced_hardware_detector.py](duckdb_api/distributed_testing/enhanced_hardware_detector.py) - Enhanced hardware detection for CPUs, GPUs, TPUs, NPUs, browsers
- [heterogeneous_scheduler.py](duckdb_api/distributed_testing/heterogeneous_scheduler.py) - Hardware-aware scheduler with multiple scheduling strategies
- [test_heterogeneous_scheduler.py](duckdb_api/distributed_testing/test_heterogeneous_scheduler.py) - Comprehensive testing and simulation infrastructure

This implementation enhances the Distributed Testing Framework with sophisticated hardware detection, classification, and scheduling capabilities optimized for heterogeneous environments. The system includes workload profiling, thermal management, performance learning, and support for specialized hardware including mobile NPUs and browser WebGPU/WebNN.

### WebGPU/WebNN TypeScript Migration (NEW - March 13, 2025)

The TypeScript migration of the IPFS Accelerate JavaScript SDK is now complete, with full integration of WebGPU and WebNN hardware acceleration:

- [TYPESCRIPT_IMPLEMENTATION_SUMMARY.md](TYPESCRIPT_IMPLEMENTATION_SUMMARY.md) - Comprehensive implementation summary of the TypeScript SDK
- [TYPESCRIPT_MIGRATION_FINAL_REPORT.md](TYPESCRIPT_MIGRATION_FINAL_REPORT.md) - Detailed report on the migration process and outcomes
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Updated API documentation with TypeScript interfaces and examples
- [SDK_DOCUMENTATION.md](SDK_DOCUMENTATION.md) - Updated SDK documentation with both Python and TypeScript SDK information
- [setup_typescript_test.py](setup_typescript_test.py) - Helper script for TypeScript validation
- [validate_import_paths.py](validate_import_paths.py) - Tool for fixing TypeScript import paths
- [src/types/webgpu.d.ts](src/types/webgpu.d.ts) - TypeScript definitions for WebGPU
- [src/types/webnn.d.ts](src/types/webnn.d.ts) - TypeScript definitions for WebNN
- [webgpu.d.ts](webgpu.d.ts) - Main WebGPU type definitions
- [webnn.d.ts](webnn.d.ts) - Main WebNN type definitions

This migration completes the WebGPU/WebNN JavaScript SDK implementation with proper TypeScript type safety and React integration.

### Adaptive Load Balancer Implementation (NEW - March 15, 2025)

The Adaptive Load Balancer component of the Distributed Testing Framework has been completed with comprehensive stress testing and monitoring capabilities:

- [LOAD_BALANCER_IMPLEMENTATION_STATUS.md](duckdb_api/distributed_testing/LOAD_BALANCER_IMPLEMENTATION_STATUS.md) - Implementation details and completion status (100% complete)
- [LOAD_BALANCER_STRESS_TESTING_GUIDE.md](duckdb_api/distributed_testing/LOAD_BALANCER_STRESS_TESTING_GUIDE.md) - Comprehensive guide for stress testing the load balancer
- [LOAD_BALANCER_MONITORING_GUIDE.md](duckdb_api/distributed_testing/LOAD_BALANCER_MONITORING_GUIDE.md) - Guide to monitoring and visualization tools (NEW - March 15, 2025)
- [LOAD_BALANCER_COMMAND_REFERENCE.md](duckdb_api/distributed_testing/LOAD_BALANCER_COMMAND_REFERENCE.md) - Comprehensive command-line reference (NEW - March 15, 2025)
- [test_load_balancer_stress.py](duckdb_api/distributed_testing/test_load_balancer_stress.py) - Comprehensive stress testing framework
- [visualize_load_balancer_performance.py](duckdb_api/distributed_testing/visualize_load_balancer_performance.py) - Performance visualization tools
- [load_balancer_live_dashboard.py](duckdb_api/distributed_testing/load_balancer_live_dashboard.py) - Real-time monitoring dashboard (NEW - March 15, 2025)
- [load_balancer_stress_config.json](duckdb_api/distributed_testing/load_balancer_stress_config.json) - Scenario-based configuration for stress testing

This implementation completes the Adaptive Load Balancer component ahead of schedule (originally planned for May 29-June 5, 2025), advancing the Distributed Testing Framework to 92% completion.

### Basic Resource Pool Fault Tolerance Test (NEW - March 13, 2025)

A new simplified test implementation for the WebGPU/WebNN Resource Pool Fault Tolerance system has been added:

- [test_basic_resource_pool_fault_tolerance.py](test_basic_resource_pool_fault_tolerance.py) - Simple, standalone test case for the WebGPU/WebNN Resource Pool fault tolerance system
- [BASIC_FAULT_TOLERANCE_TEST_README.md](BASIC_FAULT_TOLERANCE_TEST_README.md) - Comprehensive guide to using the basic fault tolerance test

### WebGPU/WebNN Resource Pool Advanced Fault Tolerance (COMPLETED - May 22, 2025)

The WebGPU/WebNN Resource Pool Integration has been completed with the addition of Advanced Fault Tolerance Visualization and Validation capabilities:

- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md) - May 2025 enhancements documentation (NEW - May 22, 2025)
- [WEB_RESOURCE_POOL_FAULT_TOLERANCE_TESTING.md](WEB_RESOURCE_POOL_FAULT_TOLERANCE_TESTING.md) - Comprehensive fault tolerance testing guide (NEW - May 22, 2025)
- [RESOURCE_POOL_FAULT_TOLERANCE_README.md](RESOURCE_POOL_FAULT_TOLERANCE_README.md) - Quick start guide for fault tolerance testing (NEW - May 22, 2025)
- [BASIC_FAULT_TOLERANCE_TEST_README.md](BASIC_FAULT_TOLERANCE_TEST_README.md) - Basic fault tolerance test guide (NEW - March 13, 2025)
- [run_web_resource_pool_fault_tolerance_test.py](run_web_resource_pool_fault_tolerance_test.py) - CLI tool for fault tolerance testing (NEW - May 22, 2025)
- [run_advanced_fault_tolerance_visualization.py](run_advanced_fault_tolerance_visualization.py) - CLI tool for fault tolerance visualization (NEW - May 22, 2025)
- [simple_fault_tolerance_test.py](simple_fault_tolerance_test.py) - Simplified fault tolerance test for CI/CD environments (NEW - May 22, 2025)
- [test_basic_resource_pool_fault_tolerance.py](test_basic_resource_pool_fault_tolerance.py) - Basic resource pool fault tolerance test (NEW - March 13, 2025)
- [test_web_resource_pool_fault_tolerance_integration.py](test_web_resource_pool_fault_tolerance_integration.py) - Integration test framework (NEW - May 22, 2025)
- [fixed_web_platform/visualization/fault_tolerance_visualizer.py](fixed_web_platform/visualization/fault_tolerance_visualizer.py) - Visualization component (NEW - May 22, 2025)
- [fixed_web_platform/fault_tolerance_visualization_integration.py](fixed_web_platform/fault_tolerance_visualization_integration.py) - Validation and visualization integration (NEW - May 22, 2025)
- [fixed_mock_cross_browser_sharding.py](fixed_mock_cross_browser_sharding.py) - Fixed mock implementation for CI/CD testing (NEW - May 22, 2025)
- [mock_cross_browser_sharding.py](mock_cross_browser_sharding.py) - Original mock implementation for testing (NEW - May 22, 2025)

This completes the WebGPU/WebNN Resource Pool Integration project ahead of schedule, with enterprise-grade fault tolerance, visualization, and testing capabilities.

### Predictive Performance System Completion (May 11, 2025)

The Predictive Performance System has been completed with the successful implementation of the Multi-Model Resource Pool Integration and Multi-Model Web Integration components:

- [PREDICTIVE_PERFORMANCE_COMPLETION.md](PREDICTIVE_PERFORMANCE_COMPLETION.md) - Comprehensive completion report (NEW - May 11, 2025)
- [predictive_performance/multi_model_web_integration.py](predictive_performance/multi_model_web_integration.py) - Multi-Model Web Integration implementation (NEW - May 11, 2025)
- [predictive_performance/test_multi_model_web_integration.py](predictive_performance/test_multi_model_web_integration.py) - Comprehensive test suite (NEW - May 11, 2025)
- [run_multi_model_web_integration.py](run_multi_model_web_integration.py) - Command-line demo with browser detection and strategy comparison (NEW - May 11, 2025)
- [verify_multi_model_integration.py](verify_multi_model_integration.py) - Verification script (NEW - May 11, 2025)
- [predictive_performance/README.md](predictive_performance/README.md) - Updated documentation (UPDATED - May 11, 2025)

This implementation completes the Predictive Performance System, providing comprehensive integration between prediction, execution, and validation components with browser-specific optimizations, tensor sharing, and empirical validation.

### Advanced Visualization System Implementation (May 15, 2025)

The Advanced Visualization System has been implemented, providing comprehensive visualization capabilities for the Predictive Performance System:

- [ADVANCED_VISUALIZATION_GUIDE.md](ADVANCED_VISUALIZATION_GUIDE.md) - Comprehensive guide to the advanced visualization capabilities (NEW - May 15, 2025)
- [run_visualization_demo.py](run_visualization_demo.py) - Updated demo script with advanced visualization features (NEW - May 15, 2025)
- [predictive_performance/visualization.py](predictive_performance/visualization.py) - Advanced visualization implementation (NEW - May 15, 2025)
- [predictive_performance/test_visualization.py](predictive_performance/test_visualization.py) - Unit tests for visualization capabilities (NEW - May 15, 2025)

This implementation provides interactive and static visualizations, including 3D visualizations, interactive dashboards, time-series tracking, power efficiency analysis, dimension reduction, and confidence visualization.

### Distributed Testing Framework Advanced Fault Tolerance (May 22, 2025)

The Distributed Testing Framework has been enhanced with advanced fault tolerance mechanisms and comprehensive integration capabilities:

- [DISTRIBUTED_TESTING_INTEGRATION_PR.md](DISTRIBUTED_TESTING_INTEGRATION_PR.md) - Latest status update on advanced fault tolerance implementation
- [DISTRIBUTED_TESTING_GUIDE.md](DISTRIBUTED_TESTING_GUIDE.md) - Updated comprehensive user guide
- [FAULT_TOLERANCE_UPDATE.md](FAULT_TOLERANCE_UPDATE.md) - Previous update on fault tolerance implementation
- [distributed_testing/docs/ADVANCED_RECOVERY_STRATEGIES.md](distributed_testing/docs/ADVANCED_RECOVERY_STRATEGIES.md) - Advanced failure recovery mechanisms
- [distributed_testing/docs/PERFORMANCE_TREND_ANALYSIS.md](distributed_testing/docs/PERFORMANCE_TREND_ANALYSIS.md) - Documentation of the performance trend analysis system
- [distributed_testing/README_PLUGIN_ARCHITECTURE.md](distributed_testing/README_PLUGIN_ARCHITECTURE.md) - Plugin architecture documentation (NEW - May 22, 2025)
- [distributed_testing/docs/RESOURCE_POOL_INTEGRATION.md](distributed_testing/docs/RESOURCE_POOL_INTEGRATION.md) - Resource Pool integration documentation (NEW - May 22, 2025)
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Updated main integration guide
- [WEB_RESOURCE_POOL_RECOVERY_GUIDE.md](WEB_RESOURCE_POOL_RECOVERY_GUIDE.md) - Recovery system documentation
- [WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md](WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md) - Guide to cross-browser model sharding
- [IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md](IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md) - Tensor sharing documentation

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

The project is currently focused on completing in-progress components and implementing planned enhancements:

### 0. WebGPU/WebNN JavaScript SDK Migration (✅ COMPLETED - March 14, 2025)

The WebGPU/WebNN migration to TypeScript has been completed with full type safety and React integration:

- ✅ **Hardware Abstraction Layer**: Unified interface for accessing hardware backends with proper TypeScript generics
  - [HARDWARE_ABSTRACTION_LAYER_GUIDE.md](HARDWARE_ABSTRACTION_LAYER_GUIDE.md) - **NEW** Comprehensive guide to the HAL
  - [ipfs_accelerate_js_hardware_abstraction.ts](ipfs_accelerate_js_hardware_abstraction.ts) - **NEW** HAL implementation
- ✅ **Hardware Abstracted Models**: Model implementations using the HAL for optimal performance
  - [HARDWARE_ABSTRACTION_VIT_GUIDE.md](HARDWARE_ABSTRACTION_VIT_GUIDE.md) - **NEW** Guide to HAL-accelerated ViT model
  - [ipfs_accelerate_js_vit_hardware_abstraction.ts](ipfs_accelerate_js_vit_hardware_abstraction.ts) - **NEW** HAL-accelerated ViT
  - [HardwareAbstractionDemo.html](HardwareAbstractionDemo.html) - **NEW** Interactive demo for HAL capabilities
- ✅ **WebGPU Backend**: Complete implementation for GPU acceleration via WebGPU API
- ✅ **WebNN Backend**: Implementation for neural network acceleration via WebNN API
- ✅ **React Integration**: Custom hooks for easy integration with React applications
- ✅ **Type Definitions**: Comprehensive types for WebGPU, WebNN, and hardware abstractions
- ✅ **Documentation**: Complete API documentation with TypeScript interfaces and examples

Documentation:
- [TYPESCRIPT_IMPLEMENTATION_SUMMARY.md](TYPESCRIPT_IMPLEMENTATION_SUMMARY.md) - Implementation summary
- [TYPESCRIPT_MIGRATION_FINAL_REPORT.md](TYPESCRIPT_MIGRATION_FINAL_REPORT.md) - Detailed migration report
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Updated API documentation with TypeScript interfaces
- [SDK_DOCUMENTATION.md](SDK_DOCUMENTATION.md) - Updated SDK documentation with both Python and TypeScript
- [BERT_MODEL_DOCUMENTATION.md](ipfs_accelerate_js/docs/models/BERT_MODEL_DOCUMENTATION.md) - **NEW** Comprehensive BERT model documentation

### 1. Predictive Performance System (✅ COMPLETED - May 11, 2025)

The Predictive Performance System has been completed with the successful implementation of the Multi-Model Resource Pool Integration and Multi-Model Web Integration components:

- ✅ **Multi-Model Execution**: Performance prediction for multiple models executing concurrently
- ✅ **Resource Pool Integration**: Connection with WebNN/WebGPU Resource Pool for empirical validation
- ✅ **Browser-Specific Optimizations**: Automatic selection of optimal browser for each model type
- ✅ **Cross-Model Tensor Sharing**: Efficient memory sharing between models (30% reduction)
- ✅ **Adaptive Strategy Selection**: Intelligent selection of execution strategies
- ✅ **Empirical Validation**: Continuous refinement based on actual measurements
- ✅ **Web Integration**: Unified interface for all components with browser acceleration

Documentation:
- [PREDICTIVE_PERFORMANCE_COMPLETION.md](PREDICTIVE_PERFORMANCE_COMPLETION.md) - Comprehensive completion report
- [predictive_performance/README.md](predictive_performance/README.md) - Updated main documentation
- [run_multi_model_web_integration.py](run_multi_model_web_integration.py) - Demo script with browser detection

### 2. Advanced Visualization System (✅ COMPLETED - July 5, 2025)

The Advanced Visualization System for the Predictive Performance System has been completed, providing comprehensive visualization capabilities with full monitoring dashboard integration:

- ✅ **3D Visualizations**: Multi-dimensional performance exploration with interactive rotation and filtering
- ✅ **Interactive Dashboards**: Performance metrics with filtering and comparison capabilities
- ✅ **Time-Series Visualization**: Performance tracking over time with trend detection and anomaly highlighting
- ✅ **Power Efficiency Analysis**: Visualizations showing performance relative to power consumption with efficiency contours
- ✅ **Dimension Reduction**: Feature importance analysis through PCA and t-SNE visualizations
- ✅ **Confidence Visualization**: Visual presentation of prediction uncertainties and confidence intervals
- ✅ **Visualization Reports**: Comprehensive HTML reports combining multiple visualization types
- ✅ **Monitoring Dashboard Integration**: Centralized visualization management with real-time updates via WebSocket
- ✅ **Dashboard Panel Creation**: Automatic creation of dashboard panels from visualizations
- ✅ **Snapshot Export/Import**: Exporting and importing dashboard snapshots for sharing

Documentation:
- [ADVANCED_VISUALIZATION_GUIDE.md](ADVANCED_VISUALIZATION_GUIDE.md) - Comprehensive visualization guide
- [MONITORING_DASHBOARD_INTEGRATION_GUIDE.md](MONITORING_DASHBOARD_INTEGRATION_GUIDE.md) - Guide for dashboard integration
- [predictive_performance/PREDICTIVE_PERFORMANCE_GUIDE.md](predictive_performance/PREDICTIVE_PERFORMANCE_GUIDE.md) - Updated main guide with visualization features
- [run_visualization_demo.py](run_visualization_demo.py) - Demo script with advanced visualization features

### 3. WebGPU/WebNN Resource Pool Integration (🔄 IN PROGRESS - 97% complete)

The WebNN/WebGPU Resource Pool Integration is nearing completion with the following status:

- 🔄 **Fault-Tolerant Cross-Browser Model Sharding**: Advanced enterprise-grade fault tolerance (90% complete)
  - ✅ Multiple sharding strategies implementation (layer-based, attention-feedforward, component-based)
  - ✅ Transaction-based state management with consistent recovery
  - ✅ Dependency-aware execution and recovery planning
  - ✅ Distributed consensus for reliable state management
  - 🔄 Advanced fault tolerance validation (95% complete)
  - 🔄 End-to-end testing across all sharding strategies (85% complete)
- ✅ **Browser-Specific Optimizations**: Intelligent optimization based on performance history (COMPLETED - May 14, 2025)
- ✅ **Performance History Tracking**: Comprehensive time-series analysis of browser performance (COMPLETED - May 14, 2025)
- ✅ **Enhanced Error Recovery**: Production-grade recovery mechanisms with progressive strategies (COMPLETED)
- ✅ **Database Integration**: Comprehensive storage and analysis of performance metrics (COMPLETED)

Documentation:
- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md) - Latest enhancements (UPDATED - May 14, 2025)
- [WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md](WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md) - Cross-browser model sharding guide (UPDATED - May 14, 2025)
- [CROSS_BROWSER_MODEL_SHARDING_TESTING_GUIDE.md](CROSS_BROWSER_MODEL_SHARDING_TESTING_GUIDE.md) - End-to-end testing guide (NEW - May 14, 2025)
- [WEB_BROWSER_PERFORMANCE_HISTORY.md](WEB_BROWSER_PERFORMANCE_HISTORY.md) - Browser performance history system (NEW - May 14, 2025)
- [WEB_RESOURCE_POOL_COMPLETION_REPORT.md](WEB_RESOURCE_POOL_COMPLETION_REPORT.md) - Initial completion report
- [WEB_RESOURCE_POOL_DATABASE_INTEGRATION.md](WEB_RESOURCE_POOL_DATABASE_INTEGRATION.md) - Database integration documentation
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Main integration guide
- [IPFS_RESOURCE_POOL_INTEGRATION_GUIDE.md](IPFS_RESOURCE_POOL_INTEGRATION_GUIDE.md) - IPFS integration guide

### 4. Simulation Accuracy and Validation Framework (🔄 PLANNED - July 2025)

The Simulation Accuracy and Validation Framework is being designed to provide comprehensive tools for validating, calibrating, and monitoring hardware simulation accuracy:

- 📋 **Simulation Validation Methodology**: Statistical metrics and validation protocols for simulation accuracy
- 📋 **Comparison Pipeline**: Tools for comparing simulation with real hardware measurements
- 📋 **Statistical Validation Tools**: Statistical methods for quantifying simulation accuracy 
- 📋 **Calibration System**: Parameter optimization for improving simulation models
- 📋 **Drift Detection**: Monitoring system for detecting simulation accuracy drift
- 📋 **Reporting and Visualization**: Comprehensive reports and visualizations for accuracy analysis

Documentation:
- [SIMULATION_ACCURACY_VALIDATION_DESIGN.md](SIMULATION_ACCURACY_VALIDATION_DESIGN.md) - Design document (NEW - July 8, 2025)
- [duckdb_api/simulation_validation/README.md](duckdb_api/simulation_validation/README.md) - Implementation overview (NEW - July 8, 2025)
- [duckdb_api/simulation_validation/core/base.py](duckdb_api/simulation_validation/core/base.py) - Core interfaces (NEW - July 8, 2025)
- [duckdb_api/simulation_validation/core/schema.py](duckdb_api/simulation_validation/core/schema.py) - Database schema (NEW - July 8, 2025)
- [test_simulation_validation_foundation.py](test_simulation_validation_foundation.py) - Foundation test script (NEW - July 8, 2025)

### 3. Distributed Testing Framework (98% Complete)

The Distributed Testing Framework enables parallel execution of tests across multiple machines with these key features:

- **Hardware-Aware Fault Tolerance**: Sophisticated recovery mechanisms tailored to hardware types (COMPLETED March 13, 2025)
  - Hardware-specific recovery strategies for different platforms
  - ML-based pattern detection for intelligent recovery
  - Comprehensive visualization and reporting system
  - Checkpoint and resume capabilities for long-running tasks
- **Heterogeneous Hardware Support**: Advanced hardware taxonomy and scheduling (COMPLETED March 15, 2025)
  - Hardware detection across CPUs, GPUs, TPUs, NPUs, and browsers
  - Workload profiling with hardware-specific requirements
  - Thermal state simulation and management
- **Adaptive Load Balancing**: Intelligent test distribution (COMPLETED March 15, 2025)
  - Thermal management for optimal worker utilization
  - Advanced scheduling algorithms with customizable weighting
  - Work stealing algorithms for load redistribution
- **Live Monitoring and Visualization**: Comprehensive monitoring and reporting (COMPLETED March 13, 2025)
  - Fault tolerance visualization and reporting system
  - Recovery strategy effectiveness analysis
  - Failure pattern detection and visualization

Documentation:
- [HARDWARE_FAULT_TOLERANCE_OVERVIEW.md](HARDWARE_FAULT_TOLERANCE_OVERVIEW.md) - Executive summary of fault tolerance system (NEW - March 13, 2025)
- [HARDWARE_FAULT_TOLERANCE_GUIDE.md](duckdb_api/distributed_testing/HARDWARE_FAULT_TOLERANCE_GUIDE.md) - Detailed guide to fault tolerance system (COMPLETED March 13, 2025)
- [HARDWARE_FAULT_TOLERANCE_ENHANCEMENTS.md](HARDWARE_FAULT_TOLERANCE_ENHANCEMENTS.md) - ML detection and visualization enhancements (NEW - March 13, 2025)
- [FAULT_TOLERANCE_VISUALIZATION_README.md](duckdb_api/distributed_testing/FAULT_TOLERANCE_VISUALIZATION_README.md) - Visualization system guide (NEW - March 13, 2025)
- [HETEROGENEOUS_HARDWARE_GUIDE.md](duckdb_api/distributed_testing/HETEROGENEOUS_HARDWARE_GUIDE.md) - Guide to heterogeneous hardware support (COMPLETED March 15, 2025)
- [DISTRIBUTED_TESTING_INTEGRATION_PR.md](DISTRIBUTED_TESTING_INTEGRATION_PR.md) - Latest status update
- [DISTRIBUTED_TESTING_GUIDE.md](DISTRIBUTED_TESTING_GUIDE.md) - Comprehensive user guide
- [DISTRIBUTED_TESTING_DESIGN.md](DISTRIBUTED_TESTING_DESIGN.md) - Detailed design document
- [LOAD_BALANCER_IMPLEMENTATION_STATUS.md](duckdb_api/distributed_testing/LOAD_BALANCER_IMPLEMENTATION_STATUS.md) - Load balancer implementation status (COMPLETED March 15, 2025)
- [LOAD_BALANCER_MONITORING_GUIDE.md](duckdb_api/distributed_testing/LOAD_BALANCER_MONITORING_GUIDE.md) - Monitoring and visualization guide (COMPLETED March 15, 2025)
- [LOAD_BALANCER_COMMAND_REFERENCE.md](duckdb_api/distributed_testing/LOAD_BALANCER_COMMAND_REFERENCE.md) - Comprehensive command reference (COMPLETED March 15, 2025)
- [LOAD_BALANCER_STRESS_TESTING_GUIDE.md](duckdb_api/distributed_testing/LOAD_BALANCER_STRESS_TESTING_GUIDE.md) - Stress testing guide (COMPLETED March 14, 2025)

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

#### Implementation and Benchmarking

- [REAL_WEBNN_WEBGPU_IMPLEMENTATION.md](REAL_WEBNN_WEBGPU_IMPLEMENTATION.md) - Current implementation details for real WebNN/WebGPU
- [REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md](REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md) - Latest implementation updates (March 2025)
- [REAL_WEBNN_WEBGPU_BENCHMARKING_GUIDE.md](REAL_WEBNN_WEBGPU_BENCHMARKING_GUIDE.md) - Comprehensive benchmarking guide
- [WEBNN_WEBGPU_BENCHMARK_README.md](WEBNN_WEBGPU_BENCHMARK_README.md) - Overview of the benchmark system
- [WEBNN_WEBGPU_DATABASE_INTEGRATION.md](WEBNN_WEBGPU_DATABASE_INTEGRATION.md) - Database integration guide
- [WEBNN_WEBGPU_ARCHIVED_DOCS.md](WEBNN_WEBGPU_ARCHIVED_DOCS.md) - Reference for archived WebNN/WebGPU documentation
- [WebGPU_BROWSER_OPTIMIZATIONS.md](WebGPU_BROWSER_OPTIMIZATIONS.md) - Browser-specific optimizations for WebGPU compute shaders (NEW - March 13, 2025)

#### TypeScript Implementation (NEW - March 13, 2025)

- [TYPESCRIPT_IMPLEMENTATION_SUMMARY.md](TYPESCRIPT_IMPLEMENTATION_SUMMARY.md) - Comprehensive implementation summary
- [TYPESCRIPT_MIGRATION_FINAL_REPORT.md](TYPESCRIPT_MIGRATION_FINAL_REPORT.md) - Detailed migration report
- [WEBGPU_WEBNN_MIGRATION_PROGRESS_UPDATED.md](WEBGPU_WEBNN_MIGRATION_PROGRESS_UPDATED.md) - Final migration progress update
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Updated API documentation with TypeScript interfaces
- [SDK_DOCUMENTATION.md](SDK_DOCUMENTATION.md) - Updated SDK documentation with TypeScript section
- [webgpu.d.ts](webgpu.d.ts) - Main WebGPU type definitions
- [webnn.d.ts](webnn.d.ts) - Main WebNN type definitions 
- [src/types/webgpu.d.ts](src/types/webgpu.d.ts) - Structure-specific WebGPU type definitions
- [src/types/webnn.d.ts](src/types/webnn.d.ts) - Structure-specific WebNN type definitions
- [WEBNN_GRAPH_BUILDING_GUIDE.md](WEBNN_GRAPH_BUILDING_GUIDE.md) - Comprehensive guide to WebNN graph building (NEW - March 14, 2025)
- [ipfs_accelerate_js_webnn_graph_builder.ts](ipfs_accelerate_js_webnn_graph_builder.ts) - Core implementation of WebNN graph building for neural networks (NEW - March 14, 2025)
- [ipfs_accelerate_js_webnn_graph_builder.test.ts](ipfs_accelerate_js_webnn_graph_builder.test.ts) - Comprehensive test suite for WebNN graph building (NEW - March 14, 2025)
- [ipfs_accelerate_js_webnn_graph_example.ts](ipfs_accelerate_js_webnn_graph_example.ts) - Example code for building neural networks with WebNN (NEW - March 14, 2025)

#### Cross-Model Tensor Sharing (NEW - March 28, 2025)

- [CROSS_MODEL_TENSOR_SHARING_GUIDE.md](CROSS_MODEL_TENSOR_SHARING_GUIDE.md) - Comprehensive guide to the Cross-Model Tensor Sharing system
- [ipfs_accelerate_js_tensor_sharing_integration.ts](ipfs_accelerate_js_tensor_sharing_integration.ts) - TensorSharingIntegration implementation
- [ipfs_accelerate_js_tensor_sharing_example.ts](ipfs_accelerate_js_tensor_sharing_example.ts) - Example usage of tensor sharing
- [TensorSharingDemo.html](TensorSharingDemo.html) - Interactive browser demo
- [ipfs_accelerate_js/src/tensor/shared_tensor.ts](ipfs_accelerate_js/src/tensor/shared_tensor.ts) - Core shared tensor implementation

#### Resource Pool and Cross-Browser Features

- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md) - Latest enhancements to WebNN/WebGPU Resource Pool (UPDATED - May 14, 2025)
- [WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md](WEB_CROSS_BROWSER_MODEL_SHARDING_GUIDE.md) - Guide to cross-browser model sharding (UPDATED - May 14, 2025)
- [CROSS_BROWSER_MODEL_SHARDING_TESTING_GUIDE.md](CROSS_BROWSER_MODEL_SHARDING_TESTING_GUIDE.md) - End-to-end testing guide for fault-tolerant model sharding (NEW - May 14, 2025)
- [WEB_BROWSER_PERFORMANCE_HISTORY.md](WEB_BROWSER_PERFORMANCE_HISTORY.md) - Browser performance history system for optimization (NEW - May 14, 2025)

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

## Advanced Features Documentation (July 2025)

### Simulation Accuracy and Validation Framework (July 2025)

- [SIMULATION_ACCURACY_VALIDATION_DESIGN.md](SIMULATION_ACCURACY_VALIDATION_DESIGN.md) - Design document for the framework (NEW - July 8, 2025)
- [duckdb_api/simulation_validation/README.md](duckdb_api/simulation_validation/README.md) - Implementation overview (NEW - July 8, 2025)
- [duckdb_api/simulation_validation/core/base.py](duckdb_api/simulation_validation/core/base.py) - Core interfaces and abstract classes (NEW - July 8, 2025)
- [duckdb_api/simulation_validation/core/schema.py](duckdb_api/simulation_validation/core/schema.py) - Database schema for simulation and validation data (NEW - July 8, 2025)
- [test_simulation_validation_foundation.py](test_simulation_validation_foundation.py) - Foundation test script (NEW - July 8, 2025)

### Predictive Performance System (May 2025)

- [PREDICTIVE_PERFORMANCE_COMPLETION.md](PREDICTIVE_PERFORMANCE_COMPLETION.md) - Complete system implementation report (NEW - May 11, 2025)
- [ADVANCED_VISUALIZATION_GUIDE.md](ADVANCED_VISUALIZATION_GUIDE.md) - Comprehensive guide to advanced visualization capabilities (NEW - May 15, 2025)
- [PREDICTIVE_PERFORMANCE_GUIDE.md](predictive_performance/PREDICTIVE_PERFORMANCE_GUIDE.md) - Main guide for the Predictive Performance System (UPDATED - May 15, 2025)
- [ACTIVE_LEARNING_DESIGN.md](predictive_performance/ACTIVE_LEARNING_DESIGN.md) - Design document for the Active Learning System
- [INTEGRATED_ACTIVE_LEARNING_GUIDE.md](predictive_performance/INTEGRATED_ACTIVE_LEARNING_GUIDE.md) - Integration guide for Active Learning
- [MODEL_UPDATE_PIPELINE_GUIDE.md](predictive_performance/MODEL_UPDATE_PIPELINE_GUIDE.md) - Documentation for the Model Update Pipeline
- [MULTI_MODEL_EXECUTION_GUIDE.md](predictive_performance/MULTI_MODEL_EXECUTION_GUIDE.md) - Guide to multi-model execution performance prediction
- [TEST_BATCH_GENERATOR_GUIDE.md](predictive_performance/TEST_BATCH_GENERATOR_GUIDE.md) - Guide to the Test Batch Generator
- [MULTI_MODEL_RESOURCE_POOL_INTEGRATION_GUIDE.md](predictive_performance/MULTI_MODEL_RESOURCE_POOL_INTEGRATION_GUIDE.md) - Guide to resource pool integration (NEW - May 11, 2025)
- [MULTI_MODEL_WEB_INTEGRATION_GUIDE.md](predictive_performance/MULTI_MODEL_WEB_INTEGRATION_GUIDE.md) - Guide to browser-based integration (NEW - May 11, 2025)
- [EMPIRICAL_VALIDATION_GUIDE.md](predictive_performance/EMPIRICAL_VALIDATION_GUIDE.md) - Guide to empirical validation (NEW - May 11, 2025)

### Resource Pool and Framework Integration

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