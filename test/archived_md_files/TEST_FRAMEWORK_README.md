# IPFS Accelerate Python Testing Framework

*Last updated: March 1, 2025*

## Current Test Coverage

### Model Tests
- **Language Models**: bert, t5, llama, gpt2, roberta, albert, bart, distilbert, gemma, mistral, mixtral, falcon, mamba, phi, qwen3, opt (30+ models)
- **Vision Models**: clip, vit, deit, detr, sam, segformer, visual_bert (15+ models) 
- **Audio Models**: whisper, wav2vec2, hubert, encodec (10+ models)
- **Multimodal Models**: layoutlmv2, blip, llava, llava_next (10+ models)
- **Specialized Models**: patchtsmixer, zoedepth, time_series_transformer (5+ models)

Total: 70+ model test implementations

### API Tests
- **Complete implementations**: OpenAI, Claude, Groq, Ollama
- **Partial implementations**: HF TGI, HF TEI, Gemini
- **Planned implementations**: LLVM, OVMS, S3 Kit, OPEA, VLLM

### Hardware Compatibility
- **CPU**: 100% compatibility with all model types
- **CUDA**: 94% compatibility (45/48 models)
- **OpenVINO**: 90% compatibility (43/48 models)

### Test Performance
- **Parallel Execution**: Up to 8x faster test execution with multiprocessing
- **Full Test Suite**: ~15 minutes for complete test suite (48 models × 3 platforms)
- **API Test Suite**: ~5 minutes for all API backends

| Hardware Platform | Test Status | Passing Models | Notes |
|-------------------|-------------|----------------|-------|
| CPU | ✅ | All models | Default fallback for all implementations |
| CUDA | ✅ | 45/48 models | Vision-T5, MobileViT, UPerNet not fully compatible |
| OpenVINO | ✅ | 43/48 models | StableDiffusion, LLaVA-Next, BLIP not compatible |

This document provides an overview of the updated testing framework for the IPFS Accelerate Python project. The framework includes tools for generating and running tests for various components including Hugging Face models, API backends, and hardware implementations.

## Testing Components

The testing framework is divided into three main components:

1. **Model Testing**: Tests for Hugging Face and other model implementations
2. **API Backend Testing**: Tests for API clients (OpenAI, Claude, Groq, etc.)
3. **Hardware Testing**: Tests for different hardware targets (CPU, CUDA, OpenVINO)

## Test Generators

The testing framework provides two test generators:

1. **`generate_unified_test.py`**: Advanced, comprehensive test generator for both models and APIs
   - Highly configurable with many options
   - Generates sophisticated tests with performance metrics
   - Supports all model categories and tasks
   - Takes longer to generate due to complexity

2. **`generate_basic_tests.py`**: Simple, fast test generator for quick model testing
   - Limited options but much faster
   - Good for generating basic smoke tests
   - Simple command-line interface
   - Smart model selection based on model type
   - Robust error handling

### Recent Improvements (March 2025)

The test generation system has been significantly improved:

1. **Smart Model Selection**: The generator now automatically selects the appropriate test model and task for each model type
2. **Enhanced Error Handling**: Better handling of tensor outputs and real vs. mock detection
3. **Improved JSON Serialization**: Safe handling of complex objects in test results
4. **Dependency Status Reporting**: Tests now report available dependencies
5. **Support for 70+ Model Types**: Comprehensive coverage across language, vision, audio, and multimodal models
6. **Hardware-Specific Testing**: Automatic testing on CPU, CUDA, and OpenVINO when available
7. **Model Compatibility Detection**: Smart detection of which model types work with which hardware

For test implementation status details, see: `/test/test_implementation_status.json`

### Key Features

#### Unified Test Generator Features
- Comprehensive test generation for both models and APIs
- Support for different model categories (language, vision, audio, multimodal, specialized)
- Task-specific test inputs based on model functionality
- Hardware-specific test paths for CPU, CUDA, and OpenVINO
- Performance metrics collection and memory usage tracking
- Standardized test patterns for consistency across components

#### Basic Test Generator Features
- Fast and simple test generation
- Support for basic model testing
- Simple command-line interface
- Minimal dependencies
- Good for quick smoke tests and simple test coverage

### Usage

```bash
# Generate model tests (advanced - supports all features)
python generate_unified_test.py --type model --category language --limit 5

# Generate API tests (advanced - with comprehensive testing)
python generate_unified_test.py --type api --apis ollama groq

# Generate both model and API tests (advanced)
python generate_unified_test.py --type all --limit 3

# Generate basic tests quickly (simplified version)
python generate_basic_tests.py bert t5 llama --task text-generation

# List missing model tests without generating
python generate_unified_test.py --type model --list-missing

# Generate tests for specific models
python generate_unified_test.py --type model --models bert t5 llama
```

### Common Options

- `--type`: Type of tests to generate (`model`, `api`, or `all`)
- `--verbose`: Enable verbose output for debugging
- `--output-dir`: Custom output directory for generated files

#### Model Test Options

- `--models`: List of specific models to generate tests for
- `--category`: Category of models to process (language, vision, audio, multimodal, specialized, all)
- `--limit`: Maximum number of test files to generate
- `--list-missing`: Only list missing tests, don't generate files

#### API Test Options

- `--apis`: List of APIs to generate tests for
- `--force`: Force generation of test files even if they already exist

## Unified Test Runner

The `run_unified_tests.py` script efficiently runs the generated tests in parallel and provides comprehensive reporting.

### Key Features

- Parallel test execution using process pools
- Filtering by model, API, implementation type, or category
- Detailed reports with success/failure status
- Performance metrics collection
- JSON results and human-readable Markdown reports

### Usage

```bash
# Run all tests
python run_unified_tests.py

# Run only model tests
python run_unified_tests.py --type model

# Run only API tests
python run_unified_tests.py --type api

# Run tests for specific models
python run_unified_tests.py --type model --models bert t5 llama

# Run tests for specific APIs
python run_unified_tests.py --type api --apis ollama openai_api

# Run tests with more parallel workers
python run_unified_tests.py --workers 8

# Run only real implementations
python run_unified_tests.py --type model --impl-type real
```

### Common Options

- `--type`: Type of tests to run (`model`, `api`, or `all`)
- `--workers`: Maximum number of parallel workers (default: 4)
- `--verbose`: Enable verbose output
- `--output-dir`: Custom output directory for test results
- `--report`: Custom path to save the test report

#### Model Test Options

- `--models`: List of specific models to test
- `--category`: Category of models to test
- `--impl-type`: Implementation type to test (real, mock, all)
- `--limit`: Maximum number of model tests to run

#### API Test Options

- `--apis`: List of specific APIs to test

## Test Structure

### Model Tests

Generated model tests follow a standardized structure:

1. **Setup**: Import dependencies, handle missing packages, initialize resources
2. **Test Inputs**: Generate appropriate test inputs based on model category and task
3. **Init Tests**: Test model initialization on different platforms (CPU, CUDA, OpenVINO)
4. **Inference Tests**: Test model inference with both single and batch inputs
5. **Performance Metrics**: Collect memory usage and execution time metrics
6. **Results Storage**: Save results in both `collected_results` and `expected_results` directories

### API Tests

Generated API tests are structured as unittest test cases:

1. **Connection Tests**: Test API client initialization and connection
2. **Request Tests**: Test request handling and response processing
3. **Queue Tests**: Test the request queue system and concurrency handling
4. **Retry Tests**: Test exponential backoff and retry mechanisms
5. **API-Specific Tests**: Test functionality specific to each API (e.g., chat completion, embeddings)

## Test Results

Test results are stored in JSON format in the `test_results` directory. Each test run creates:

1. **JSON Results**: Detailed test results in structured JSON format
2. **Markdown Report**: Human-readable summary report with success/failure status and metrics

## Extending the Framework

### Adding New Model Categories

To add support for new model categories:

1. Add the category to the `TASK_CATEGORIES` dictionary in `generate_unified_test.py`
2. Define example models and test inputs for the new category
3. Update the `get_model_config` method to handle the new category's model types

### Adding New API Backends

To add support for new API backends:

1. Create a template for the API in the `API_TEMPLATES` dictionary
2. Define API-specific test methods
3. Update the environment variable mappings in the `generate_api_test_content` method

## Best Practices

1. **Regular Testing**: Run tests regularly to catch regressions early
2. **Test Coverage**: Aim for comprehensive test coverage across all components
3. **Real vs. Mock**: Test both real and mock implementations
4. **Performance Testing**: Monitor performance metrics over time
5. **Test Report Review**: Regularly review test reports for failures and performance issues

## Troubleshooting

Common issues and solutions:

1. **Missing Dependencies**: Install required packages with `pip install -r requirements_test.txt`
2. **API Key Issues**: Set up API keys in environment variables or use mock implementations
3. **CUDA/OpenVINO Issues**: Check hardware availability and skip tests if not available
4. **Memory Issues**: Reduce parallelism with `--workers` option if running out of memory

## Command Reference

Here's a quick reference of common commands:

```bash
# Generate advanced model and API tests
python generate_unified_test.py --type all

# Generate basic model tests (faster, simpler)
python generate_basic_tests.py bert t5 llama --task text-generation
python generate_basic_tests.py clip whisper --task automatic-speech-recognition

# List missing model tests
python generate_unified_test.py --type model --list-missing

# Generate tests for specific models
python generate_unified_test.py --type model --models bert t5 llama

# Generate tests for specific APIs
python generate_unified_test.py --type api --apis ollama openai_api

# Run all tests
python run_unified_tests.py

# Run tests with detailed output
python run_unified_tests.py --verbose

# Run tests for real implementations only
python run_unified_tests.py --type model --impl-type real

# Run tests with more parallelism
python run_unified_tests.py --workers 8

# Run tests with custom report path
python run_unified_tests.py --report custom_report.md

# Run test for one specific model/API
python run_unified_tests.py --type model --models bert
python run_unified_tests.py --type api --apis ollama

# Run all tests with increased parallelism (faster)
python run_unified_tests.py --workers 8 --type all

# Run benchmark test for performance evaluation
python run_unified_tests.py --type model --models bert t5 gpt2 --benchmark
```

## Test Coverage Implementation Plan

| Phase | Goal | Status | Target Date |
|-------|------|--------|------------|
| 1 | Core language models (BERT, T5, GPT2, etc.) | ✅ Complete | Feb 2025 |
| 2 | Vision models (CLIP, ViT, DETR, etc.) | ✅ Complete | Feb 2025 |
| 3 | Audio models (Whisper, Wav2Vec2, etc.) | ✅ Complete | Feb 2025 |
| 4 | Multimodal models (LLaVA, BLIP, etc.) | ✅ Complete | Mar 2025 |
| 5 | Specialized models (TimeSeries, Depth, etc.) | ✅ Complete | Mar 2025 |
| 6 | API tests (OpenAI, Claude, Groq, etc.) | ✅ 7/11 Complete | Mar 2025 |
| 7 | Automated test discovery & execution | 🟡 In Progress | Mar 2025 |
| 8 | Real implementations for top 20 models | 🟡 13/20 Complete | Apr 2025 |
| 9 | 100% test coverage for all models | 🟡 70/300 Complete | May 2025 |

## Conclusion

This testing framework provides comprehensive test coverage for the IPFS Accelerate Python project. By combining intelligent test generation with parallel execution, we can efficiently test a wide range of models, APIs, and hardware platforms.

The framework is continuously evolving to support new models and APIs as they are added to the project. Our implementation plan focuses on achieving complete test coverage for all 300+ supported models and 11 API backends.

Key benefits:
- Automatic detection of appropriate test configurations
- Smart handling of both real and mock implementations
- Parallel test execution for faster testing
- Comprehensive reporting with detailed metrics
- Support for multiple hardware targets

Remember to run the tests regularly, especially after making changes to the codebase, to ensure that everything is working as expected.