# IPFS Accelerate Python - Test Framework

This directory contains the comprehensive testing framework for the IPFS Accelerate Python library, with a focus on validating model functionality, API integrations, and hardware acceleration capabilities.

## Overview

The test framework includes:

1. **Model Tests** - Validation for 127+ HuggingFace model types across different hardware platforms
2. **API Tests** - Integration tests for various AI API providers
3. **Hardware Tests** - Validation of CPU, CUDA, and OpenVINO acceleration
4. **Endpoint Tests** - Tests for local inference endpoints
5. **Performance Tests** - Benchmarking across hardware configurations

## Test Documentation

- [MODEL_TESTING_GUIDE.md](MODEL_TESTING_GUIDE.md) - Complete guide to model testing
- [MODEL_TESTING_README.md](MODEL_TESTING_README.md) - Detailed model test documentation
- [MODEL_TESTING_PROGRESS.md](MODEL_TESTING_PROGRESS.md) - Implementation status and progress
- [MODEL_DEPENDENCIES.md](MODEL_DEPENDENCIES.md) - Model dependency tracking matrix
- [huggingface_test_implementation_plan.md](huggingface_test_implementation_plan.md) - Implementation priorities
- [API_TESTING_README.md](API_TESTING_README.md) - API testing documentation
- [API_IMPLEMENTATION_STATUS.md](API_IMPLEMENTATION_STATUS.md) - API implementation status
- [API_IMPLEMENTATION_SUMMARY.md](API_IMPLEMENTATION_SUMMARY.md) - Summary of API implementations
- [QUEUE_BACKOFF_GUIDE.md](QUEUE_BACKOFF_GUIDE.md) - Queue and backoff implementation guide
- [API_QUICKSTART.md](API_QUICKSTART.md) - Quick start guide for API usage

## Test Generation Tools

The framework includes several tools for generating new tests:

- [generate_model_tests.py](generate_model_tests.py) - Primary test generator for HuggingFace models
- [simple_model_test_generator.py](simple_model_test_generator.py) - Enhanced generator with dependency tracking
- [generate_missing_test_files.py](generate_missing_test_files.py) - Older test generator
- [generate_comprehensive_tests.py](generate_comprehensive_tests.py) - Advanced test generator (WIP)

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
```

## Current Test Coverage (March 2025)

- **HuggingFace Models**: 137 of 299 models (45.8%)
- **API Backends**: 11 API types with comprehensive testing
- **Hardware Support**: CPU (100%), CUDA (93.8%), OpenVINO (89.6%)
- **Test Quality**: Dual-method testing, dependency tracking, remote code support

## Generating New Tests

To generate tests for missing models:

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

Recent improvements to the testing framework include:

1. **Dual-Method Testing** - Tests now cover both pipeline() and from_pretrained() methods
2. **Dependency Tracking** - Automatic detection and documentation of model dependencies
3. **Remote Code Support** - Proper handling of models requiring custom code execution
4. **Enhanced Mock Objects** - Sophisticated mock implementations for various dependencies
5. **Batch Testing Support** - All generated tests now include batch processing validation
6. **Advanced Error Handling** - Improved degradation mechanisms for handling missing dependencies
7. **Performance Monitoring** - Added inference time and memory usage tracking
8. **Documentation Updates** - Expanded documentation with dependency matrix and progress tracking

### API Improvements (March 2025)

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
| LLVM | ✅ COMPLETE | Optimized local inference |
| OVMS | ✅ COMPLETE | OpenVINO Model Server integration |
| OPEA | ✅ COMPLETE | Open Platform for Enterprise AI |
| S3 Kit | ✅ COMPLETE | Model storage and retrieval |

We have implemented several major enhancements:

1. **Queue and Backoff System (COMPLETED)** - Implemented thread-safe request queueing with configurable concurrency limits, exponential backoff for error handling, proper cleanup of completed requests, and request tracking with unique IDs across all 11 API backends.

2. **Priority Queue** - Added three-tier priority levels (HIGH, NORMAL, LOW) with dynamic queue size configuration and priority-based scheduling.

3. **Circuit Breaker Pattern** - Implemented three-state machine (CLOSED, OPEN, HALF-OPEN) with automatic service outage detection and self-healing capabilities.

4. **API Key Multiplexing** - Added support for multiple API keys per provider with automatic round-robin rotation and least-loaded selection strategies.

5. **Semantic Caching** - Implemented caching based on semantic similarity with configurable thresholds to reduce API costs.

6. **Request Batching** - Added automatic request combining for compatible models with configurable batch sizes and timeouts.

To test the API improvements:

```bash
# Test a specific API
python test_api_backoff_queue.py --api claude

# Run all API tests
python run_queue_backoff_tests.py

# Run comprehensive Ollama tests
python test_ollama_backoff_comprehensive.py

# Check implementation status
python check_api_implementation.py

# Fix any remaining issues
python fix_all_api_implementations.py
```

## Contributing

When adding new tests:

1. Use the `generate_model_tests.py` script when possible
2. Follow the established test structure
3. Include both single and batch testing
4. Test on all available hardware platforms
5. Update the documentation to reflect new tests

## License

This test framework follows the same license as the main IPFS Accelerate Python library.