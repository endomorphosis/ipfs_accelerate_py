# IPFS Accelerate Python - Test Framework

This directory contains the comprehensive testing framework for the IPFS Accelerate Python library, with a focus on validating model functionality, API integrations, and hardware acceleration capabilities.

## Recent Documentation Updates

The repository now includes new comprehensive guides on resource management, model classification, and hardware integration:

- **[Resource Pool Guide](RESOURCE_POOL_GUIDE.md)** - Enhanced guide to the resource management system with device-specific features and integrated architecture diagram
- **[Model Family Classifier Guide](MODEL_FAMILY_CLASSIFIER_GUIDE.md)** - Comprehensive documentation for the model classification system
- **[Hardware Detection Guide](HARDWARE_DETECTION_GUIDE.md)** - Guide to the hardware detection system with compatibility patterns and error handling
- **[Hardware Model Integration Guide](HARDWARE_MODEL_INTEGRATION_GUIDE.md)** - Documentation for hardware and model integration
- **[Summary of Improvements](SUMMARY_OF_IMPROVEMENTS.md)** - Detailed overview of recent system enhancements
- **[API Enhancement README](API_ENHANCEMENT_README.md)** - Guide to advanced API features
- **[Web Platform Testing Guide](WEB_PLATFORM_TESTING_GUIDE.md)** - Guide to testing on WebNN and WebGPU platforms

## Overview

The test framework includes:

1. **Model Tests** - Validation for 127+ HuggingFace model types across different hardware platforms
2. **API Tests** - Integration tests for various AI API providers
3. **Hardware Tests** - Validation of CPU, CUDA, and OpenVINO acceleration
4. **Endpoint Tests** - Tests for local inference endpoints
5. **Performance Tests** - Benchmarking across hardware configurations

## Directory Structure

The repository has been organized for better readability and maintainability:

- **Root directory**: Contains the main scripts and documentation
- **test_generators/**: Tools for generating test files
- **model_test_runners/**: Model-specific test runner scripts
- **implementation_files/**: Implementation scripts for various components
- **archived_reports/**: Historical implementation reports
- **archived_test_results/**: Historical test result files
- **archived_md_files/**: Additional documentation
- **archived_cuda_fixes/**: CUDA detection fix scripts
- **old_scripts/**: Older versions of implementation scripts
- **generated_skillsets/**: Output directory for skillset generator

### New Generators Added

- **`integrated_skillset_generator.py`**: Test-driven skillset implementation generator
- **`enhanced_template_generator.py`**: Template generator with WebNN and WebGPU support
- **`merged_test_generator.py`**: Comprehensive test generator for all model types

## Core Documentation

### API Documentation

- [API_IMPLEMENTATION_STATUS.md](API_IMPLEMENTATION_STATUS.md) - API implementation status
- [API_IMPLEMENTATION_SUMMARY.md](API_IMPLEMENTATION_SUMMARY.md) - Summary of API implementations
- [API_IMPLEMENTATION_SUMMARY_UPDATED.md](API_IMPLEMENTATION_SUMMARY_UPDATED.md) - Latest implementation summary
- [API_IMPLEMENTATION_PLAN_UPDATED.md](API_IMPLEMENTATION_PLAN_UPDATED.md) - Updated implementation plan
- [QUEUE_BACKOFF_GUIDE.md](QUEUE_BACKOFF_GUIDE.md) - Queue and backoff implementation guide
- [MONITORING_AND_REPORTING_GUIDE.md](MONITORING_AND_REPORTING_GUIDE.md) - Monitoring features guide
- [ADVANCED_API_FEATURES_GUIDE.md](ADVANCED_API_FEATURES_GUIDE.md) - Advanced API features documentation
- [S3_KIT_MULTIPLEXING_GUIDE.md](S3_KIT_MULTIPLEXING_GUIDE.md) - S3 Kit connection multiplexing guide
- [API_QUICKSTART.md](API_QUICKSTART.md) - Quick start guide for API usage
- [API_TESTING_README.md](API_TESTING_README.md) - API testing documentation

### Test Generation Documentation

- [MERGED_GENERATOR_README.md](MERGED_GENERATOR_README.md) - Documentation for the merged test generator
- [MODALITY_TEMPLATE_GUIDE.md](MODALITY_TEMPLATE_GUIDE.md) - Guide to modality-specific templates
- [MERGED_GENERATOR_QUICK_REFERENCE.md](MERGED_GENERATOR_QUICK_REFERENCE.md) - Quick reference for test generation
- [INTEGRATED_SKILLSET_GENERATOR_GUIDE.md](INTEGRATED_SKILLSET_GENERATOR_GUIDE.md) - Guide for the skillset generator
- [INTEGRATED_GENERATOR_README.md](INTEGRATED_GENERATOR_README.md) - Overview of generator capabilities
- [WEB_DEPLOYMENT_EXAMPLE.md](WEB_DEPLOYMENT_EXAMPLE.md) - Complete example of web deployment

### Model and Resource Management

- [MODEL_FAMILY_CLASSIFIER_GUIDE.md](MODEL_FAMILY_CLASSIFIER_GUIDE.md) - Guide to model family classification
- [RESOURCE_POOL_GUIDE.md](RESOURCE_POOL_GUIDE.md) - Documentation for resource management system
- [SUMMARY_OF_IMPROVEMENTS.md](SUMMARY_OF_IMPROVEMENTS.md) - Summary of recent improvements
- [AMD_PRECISION_README.md](AMD_PRECISION_README.md) - Guide to AMD precision optimizations
- [ENHANCED_MODEL_REGISTRY_GUIDE.md](ENHANCED_MODEL_REGISTRY_GUIDE.md) - Enhanced model registry documentation

## Primary Implementation Tools

The following tools are used for implementing and testing the API infrastructure:

- **complete_api_improvement_plan.py** - Comprehensive implementation plan
- **run_api_improvement_plan.py** - Main orchestration script
- **standardize_api_queue.py** - Queue implementation standardization
- **enhance_api_backoff.py** - Enhanced backoff and circuit breaker implementation
- **check_api_implementation.py** - Implementation status verification

## Running Tests

### Model Tests

```bash
# Run a specific model test
python skills/test_hf_bert.py

# Run multiple tests in parallel
python run_skills_tests.py --models bert,roberta,gpt2

# Run tests for a specific category
python run_skills_tests.py --category language
```

### API Tests

```bash
# Run all API tests
python check_api_implementation.py 

# Test a specific API
python test_single_api.py [api_name]

# Test queue and backoff features
python test_api_backoff_queue.py --api [api_name]

# Run comprehensive queue and backoff tests
python run_queue_backoff_tests.py

# Run detailed Ollama backoff tests
python test_ollama_backoff_comprehensive.py
```

### Hardware Tests

```bash
# Test hardware backends
python test_hardware_backend.py

# Test specific hardware platform
python test_hardware_backend.py --backend [cpu|cuda|openvino|mps|amd|webnn|webgpu]

# Test all hardware platforms for a specific model
python test_hardware_backend.py --model bert --all-backends

# Run comprehensive hardware detection tests
python test_comprehensive_hardware.py

# Test hardware-aware model classification
python test_comprehensive_hardware.py --test integration

# Test hardware and model integration
python test_comprehensive_hardware.py --test comparison

# Test resource pool with enhanced functionality
python test_resource_pool.py --test all

# Test device-specific caching in resource pool
python test_resource_pool.py --test device

# Test memory tracking in resource pool
python test_resource_pool.py --test memory

# Test hardware-aware model selection
python test_resource_pool.py --test hardware
```

### Skillset Generation

```bash
# Generate a skillset implementation
python integrated_skillset_generator.py --model bert --run-tests

# Generate implementations for all models in a family
python integrated_skillset_generator.py --family bert

# Generate implementations for all models with web backend support
python integrated_skillset_generator.py --all --max-workers 20
```

## Current Test Coverage (March 2025)

- **HuggingFace Models**: 300 of 300 models (100%)
- **API Backends**: 11 API types with comprehensive testing
- **Hardware Support**: CPU (100%), CUDA (93.8%), OpenVINO (89.6%), MPS (87.5%), AMD (91.7%), WebNN (85.4%), WebGPU (81.3%)
- **Test Quality**: Dual-method testing, dependency tracking, remote code support, web deployment validation

## Generating New Tests

### Hardware-Aware Test Generation (NEW!)

The test generator system now includes hardware-aware test generation that automatically selects the optimal hardware for each model and creates tests that efficiently share resources:

```bash
# Generate hardware-aware tests for specific models
python test_generator_with_resource_pool.py --model bert-base-uncased
python test_generator_with_resource_pool.py --model t5-small
python test_generator_with_resource_pool.py --model vit-base-patch16-224

# Generate with specific settings
python test_generator_with_resource_pool.py --model gpt2 --output-dir ./skills
python test_generator_with_resource_pool.py --model distilbert-base-uncased --debug
python test_generator_with_resource_pool.py --model roberta-base --clear-cache
python test_generator_with_resource_pool.py --model facebook/bart-large --timeout 60

# Generate tests for multiple models with shared resources
python test_generator_with_resource_pool.py --models bert,roberta,gpt2 --output-dir ./skills
```

### Modality-Specific Test Generation

The test generator system has been enhanced with modality-specific templates that create specialized tests based on model type:

```bash
# Generate tests for specific modalities (text, vision, audio, multimodal)
python generate_modality_tests.py --modality text
python generate_modality_tests.py --modality vision
python generate_modality_tests.py --modality audio
python generate_modality_tests.py --modality multimodal

# Generate tests for all modalities
python generate_modality_tests.py --modality all

# Generate tests without verification
python generate_modality_tests.py --modality vision --no-verify

# Generate tests into a custom directory
python generate_modality_tests.py --modality text --output-dir custom_tests
```

### Legacy Test Generation

The previous test generators are still available:

```bash
# Using the primary generator
python generate_model_tests.py --list-only
python generate_model_tests.py --models layoutlmv2 nougat swinv2 vit_mae
python generate_model_tests.py --category vision --limit 5

# Using the enhanced generator with dependency tracking
python simple_model_test_generator.py --batch  # Generate batch of tests with dependency tracking
python simple_model_test_generator.py --model llama-3-70b-instruct --task text-generation  # Specific model
```

## Recent Improvements

### March 2025 Enhancements

1. **Hardware-Aware Resource Management Integration** - Comprehensive integration of resource management with hardware systems:
   - Complete integration of ResourcePool with hardware detection and model classification
   - Automatic hardware capability detection and optimal device selection
   - Model family-based hardware compatibility matrix for intelligent decisions
   - Enhanced device-specific resource management for CPU, CUDA, MPS, OpenVINO, ROCm
   - Memory monitoring and management across all hardware platforms
   - Specialized handling for memory-intensive models like LLMs
   - Automatic fallback when preferred hardware is unavailable
   - Low-memory mode with automatic detection and memory-efficient operation
   - Cross-device resource management with proper cleanup

2. **Enhanced Model Family Classification** - Significantly improved model classification system:
   - Sophisticated pattern matching with weighted scoring
   - Partial keyword matching for better subfamily detection
   - Enhanced task analysis with normalized scoring
   - Hardware compatibility analysis for family identification
   - Weighted analysis combination for accurate classifications
   - Memory requirement analysis for better family detection
   - Improved subfamily detection with specialized patterns
   - Confidence calculation with better normalization

3. **Resource Pool Enhancements** - Major improvements to the resource management system:
   - Device-specific model caching for CPU, CUDA, and MPS
   - Enhanced memory tracking capabilities with detailed device stats
   - Model family classification integration for intelligent device selection
   - Improved resource cleanup with configurable timeouts
   - Comprehensive testing for different hardware configurations
   - Low-memory mode for resource-constrained environments
   - Proper cache separation across devices
   - Enhanced API with device-specific functionality
   - Thread-safe resource sharing with robust locking

4. **Hardware Detection System** - Comprehensive hardware detection capabilities:
   - Robust detection across multiple platforms (Linux, macOS, Windows)
   - Detailed hardware capability analysis with comprehensive checks
   - Fine-grained CUDA capability detection (device count, memory, compute capability)
   - Apple Silicon (MPS) support with specialized capability detection
   - AMD ROCm support with proper identification of AMD GPUs
   - OpenVINO device detection with detailed capability reporting
   - WebNN and WebGPU capability detection for browser environments
   - Automatic selection of optimal hardware based on model characteristics
   - Enhanced support for multiple hardware backends:
     - CPU: Standard for all model types with instruction set detection
     - CUDA: With specialized optimizations by modality and memory requirements
     - OpenVINO: Hardware-optimized inference for CPU, GPU and VPU
     - Apple Silicon (MPS): M1/M2/M3 support with optimizations
     - AMD ROCm: GPU acceleration on AMD hardware with HIP detection
     - WebNN: Browser-based acceleration with feature detection
     - WebGPU: Enhanced graphics processing in browser environments

5. **Testing Infrastructure Improvements** - Enhanced testing capabilities:
   - Device-specific test verification
   - Hardware-aware test generation
   - Model family-based test optimization
   - Memory-efficient test execution
   - Command-line options for focused testing
   - Intelligent test skipping for unavailable hardware
   - Cross-device verification
   - Enhanced error reporting

5. **Documentation Enhancements** - Comprehensive documentation updates:
   - Detailed MODEL_FAMILY_CLASSIFIER_GUIDE.md with examples
   - Enhanced RESOURCE_POOL_GUIDE.md with device-specific features
   - New hardware integration documentation
   - Updated progress tracking in CLAUDE.md
   - Comprehensive SUMMARY_OF_IMPROVEMENTS.md

### Previous Improvements

1. **Dual-Method Testing** - Tests now cover both pipeline() and from_pretrained() methods
2. **Dependency Tracking** - Automatic detection and documentation of model dependencies
3. **Remote Code Support** - Proper handling of models requiring custom code execution
4. **Enhanced Mock Objects** - Sophisticated mock implementations for various dependencies
5. **Batch Testing Support** - All generated tests now include batch processing validation
6. **Advanced Error Handling** - Improved degradation mechanisms for handling missing dependencies
7. **Performance Monitoring** - Added inference time and memory usage tracking
8. **Documentation Updates** - Expanded documentation with dependency matrix and progress tracking

### API Implementations (March 2025)

All 11 API backends are now fully implemented with complete functionality:

| API | Status | Primary Use |
|-----|--------|------------|
| OpenAI | ✅ COMPLETE | GPT models, embeddings, assistants |
| Claude | ✅ COMPLETE | Claude models, streaming |
| Groq | ✅ COMPLETE | High-speed inference, Llama models |
| Ollama | ✅ COMPLETE | Local deployment, open-source models |
| HF TGI | ✅ COMPLETE | Text generation with Hugging Face models |
| HF TEI | ✅ COMPLETE | Embeddings with Hugging Face models |
| Gemini | ✅ COMPLETE | Google's models, multimodal capabilities |
| VLLM | ✅ COMPLETE | Optimized local inference |
| OVMS | ✅ COMPLETE | OpenVINO Model Server integration |
| OPEA | ✅ COMPLETE | Open Platform for Enterprise AI |
| S3 Kit | ✅ COMPLETE | Model storage and retrieval, connection multiplexing |

## Advanced API Features

We have implemented several major enhancements:

1. **Queue and Backoff System** - Thread-safe request queueing with configurable concurrency limits, exponential backoff for error handling, proper cleanup of completed requests, and request tracking with unique IDs across all 11 API backends.

2. **Priority Queue** - Three-tier priority levels (HIGH, NORMAL, LOW) with dynamic queue size configuration and priority-based scheduling.

3. **Circuit Breaker Pattern** - Three-state machine (CLOSED, OPEN, HALF-OPEN) with automatic service outage detection and self-healing capabilities.

4. **API Key Multiplexing** - Support for multiple API keys per provider with automatic round-robin rotation and least-loaded selection strategies.

5. **Monitoring and Reporting** - Comprehensive request statistics tracking with error classification, performance metrics by model and endpoint, and detailed reporting capabilities.

6. **Request Batching** - Automatic request combining for compatible models with configurable batch sizes and timeouts.

7. **S3 Connection Multiplexing** - Support for working with multiple S3-compatible storage endpoints simultaneously, with various routing strategies (round-robin, least-loaded) and per-endpoint configuration.

## Testing API Implementations

To test the API implementations:

```bash
# Run the complete API improvement plan (core implementation script)
python complete_api_improvement_plan.py

# Test a specific API's backoff and queue functionality
python test_api_backoff_queue.py --api claude

# Run all queue and backoff tests across APIs
python run_queue_backoff_tests.py

# Run comprehensive Ollama tests
python test_ollama_backoff_comprehensive.py

# Check current implementation status of all APIs
python check_api_implementation.py
```

## Next Steps

The following areas have been identified for future development:

1. **Model Registry Enhancement and Integration**
   - Implement automatic registry updates from test results
   - Create registry query tools for implementation status
   - Build visualization and reporting for implementation coverage
   - Add performance statistics to registry entries
   - Integrate registry with CI/CD pipeline

2. **Advanced Template System**
   - Create extended template system with multi-template inheritance
   - Add template verification and validation
   - Create specialized templates for edge cases
   - Develop template compatibility testing
   - Add dynamic template selection based on model version and size

3. **Resource Management Enhancements**
   - Extend resource pool to handle distributed resources across machines
   - Implement cross-process resource sharing capabilities
   - Add fine-grained memory management based on hardware characteristics
   - Add multi-device model splitting for large models
   - Create visualization tools for resource usage and allocation

4. **Advanced Hardware Integration**
   - Add specialized optimizations for different hardware platforms
   - Implement automatic quantization based on hardware capabilities
   - Create dynamic hardware selection based on workload
   - Add fallback mechanisms for hardware-specific failures
   - Develop specialized memory management for different devices

5. **Learning-Based Classification**
   - Implement machine learning-based model classification
   - Create adaptive classification that improves with usage
   - Add feedback mechanism from test results
   - Develop specialized classifiers for different model domains
   - Create confidence-based fallback mechanisms

6. **Performance Optimization**
   - Implement model-specific performance tuning
   - Optimize memory usage across different hardware
   - Create benchmarking tools for different model families
   - Add performance comparison across hardware platforms
   - Develop performance prediction based on model characteristics

7. **Web Backend Enhancements**
   - Enhance WebNN and WebGPU support
   - Improve browser-based model execution
   - Create specialized templates for web deployment
   - Add progressive enhancement for different browser capabilities
   - Develop web-specific memory management strategies

## Contributing

When adding new scripts or tests:

1. Place test generators in the `test_generators/` directory
2. Place model test runners in the `model_test_runners/` directory
3. Place implementation scripts in the `implementation_files/` directory
4. Update documentation to reflect changes

## License

This test framework follows the same license as the main IPFS Accelerate Python library.