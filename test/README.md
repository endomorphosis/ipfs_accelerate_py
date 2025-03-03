# IPFS Accelerate Python - Test Framework

This directory contains the comprehensive testing framework for the IPFS Accelerate Python library, with a focus on validating model functionality, API integrations, and hardware acceleration capabilities.

## Recent Documentation Updates

The repository now includes new comprehensive guides on API enhancements and optimizations:

- **[API Error Documentation](API_ERROR_DOCUMENTATION.md)** - Complete guide to error handling across all API backends
- **[Performance Optimization Plan](PERFORMANCE_OPTIMIZATION_PLAN.md)** - Detailed plan for optimizing API backends with focus on VLLM
- **[API Implementation Summary](API_IMPLEMENTATION_SUMMARY.md)** - Current status of all API implementations
- **[API Enhancement README](API_ENHANCEMENT_README.md)** - Guide to advanced API features

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

### Modality-Specific Test Generation (NEW!)

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

1. **Modality-Specific Test Templates** - Specialized test templates for text, vision, audio, and multimodal models with:
   - Modality-appropriate imports and dependencies
   - Automatic test data generation (text, images, audio files)
   - Hardware-specific optimizations tailored to each modality
   - Appropriate processing logic for each model type

2. **Enhanced Hardware Support** - Comprehensive hardware platform support:
   - CPU: Standard for all model types
   - CUDA: With specialized optimizations by modality
   - OpenVINO: Hardware-optimized inference
   - Apple Silicon (MPS): M1/M2/M3 support
   - AMD ROCm: GPU acceleration on AMD hardware
   - Qualcomm AI: Mobile chip optimizations

3. **Automatic Modality Detection** - Smart detection of model type based on:
   - Model name patterns
   - Model task categories
   - Pipeline compatibility

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

1. **Template Generation Improvements**
   - Integrate modality-specific templates into the merged test generator pipeline
   - Add more specialized optimizations per modality
   - Expand model registry with more detailed modality information
   - Add performance benchmarks that leverage modality-specific metrics

2. **Web Backend Enhancements**
   - Enhance WebNN export pipeline with optimizations for mobile devices
   - Improve transformers.js integration with advanced caching strategies
   - Add progressive enhancement for browsers with limited capabilities
   - Create comprehensive deployment patterns for web applications

3. **Qualcomm AI Support**
   - Complete implementation for Qualcomm AI backend
   - Add specific optimizations for mobile deployments
   - Implement test infrastructure for Qualcomm devices
   - Create conversion utilities for efficient deployment

4. **Semantic Caching Implementation**
   - Add caching layer for frequently used requests
   - Implement embedding-based similarity search
   - Add cache invalidation strategies

5. **Advanced Rate Limiting**
   - Implement token-bucket rate limiters
   - Add adaptive rate limiting based on response codes
   - Implement sliding-window rate limiters

6. **Performance Optimization**
   - Benchmark throughput and latency across all backends
   - Optimize queue processing for higher throughput
   - Fine-tune backoff parameters per provider
   - Create performance comparison dashboard

## Contributing

When adding new scripts or tests:

1. Place test generators in the `test_generators/` directory
2. Place model test runners in the `model_test_runners/` directory
3. Place implementation scripts in the `implementation_files/` directory
4. Update documentation to reflect changes

## License

This test framework follows the same license as the main IPFS Accelerate Python library.