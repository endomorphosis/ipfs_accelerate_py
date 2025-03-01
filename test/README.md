# IPFS Accelerate Python Testing Framework

This directory contains the comprehensive testing framework for the IPFS Accelerate Python library. The testing suite is designed to validate the functionality of all supported models, APIs, and hardware integrations.

## Directory Structure

- `test/` - Main testing directory
  - `apis/` - Tests for API integrations (OpenAI, Claude, Groq, etc.)
  - `skills/` - Tests for model implementations (BERT, LLaMA, CLIP, etc.)
  - `performance_results/` - Performance test results and reports
  - `api_check_results/` - API implementation validation results
  - `collected_results/` - Hardware and model test results

## Documentation

- [MODEL_TESTING_README.md](MODEL_TESTING_README.md) - Comprehensive model testing documentation
- [NEW_MODEL_TESTS_README.md](NEW_MODEL_TESTS_README.md) - Documentation for latest model implementations (March 2025)
- [API_TESTING_README.md](API_TESTING_README.md) - API testing documentation
- [ADVANCED_TESTS_README.md](ADVANCED_TESTS_README.md) - Advanced testing scenarios
- [CLAUDE.md](CLAUDE.md) - Development guidelines and implementation status

## Running Tests

### Model Tests

```bash
# Run a specific model test
python3 skills/test_hf_bert.py

# Run multiple tests in parallel
python3 run_skills_tests.py --models bert,roberta,gpt2

# Run all model tests
python3 run_skills_tests.py --all

# Run tests for a specific type
python3 run_skills_tests.py --type language  # Options: language, vision, audio, multimodal
```

### API Tests

```bash
# Check API implementation status
python3 check_api_implementation.py

# Test specific API
python3 test_single_api.py [api_name]

# Test all APIs
python3 test_api_backend.py
```

### Hardware Tests

```bash
# Test hardware backends
python3 test_hardware_backend.py --backend [cpu|cuda|openvino] --model [model_name]

# Run performance tests
python3 run_performance_tests.py --batch_size 8 --models all
```

## Recent Updates (March 2025)

- Added new model test implementations:
  - Qwen3 (`test_hf_qwen3.py`)
  - Video-LLaVA (`test_hf_video_llava.py`)
  - Time Series Transformer (`test_hf_time_series_transformer.py`)

- Improved API backend tests:
  - Added request queue backoff testing
  - Enhanced credential management
  - Implemented comprehensive error handling validation

- Enhanced local endpoint testing:
  - Fixed endpoint handler to return callable functions
  - Added support for 47 local model endpoints
  - Implemented dictionary structure validation

## Contributing

When adding new tests, please follow these guidelines:

1. Use the existing test structure and patterns
2. Ensure tests have appropriate fallback mechanisms
3. Include both REAL and MOCK implementation support
4. Test across all supported hardware platforms
5. Document your tests in the appropriate README file

## Test Reports

Tests generate detailed reports that are saved in the appropriate results directories:

- Model test results: `skills/collected_results/`
- API test results: `apis/collected_results/`
- Hardware test results: `collected_results/`
- Performance test results: `performance_results/`

These reports include detailed information about test performance, implementation status, and compatibility across platforms.

---

For more information about the IPFS Accelerate Python framework, please refer to the main documentation.