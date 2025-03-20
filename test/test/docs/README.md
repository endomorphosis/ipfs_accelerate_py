# IPFS Accelerate Test Framework

## Overview

This is the test framework for the IPFS Accelerate Python library. The framework provides a unified approach to testing different components of the IPFS Accelerate ecosystem, including:

- Model tests (BERT, T5, ViT, Whisper, GPT, etc.)
- Hardware-specific tests (WebGPU, WebNN, CUDA, ROCm, CPU)
- API tests (OpenAI, HuggingFace, Ollama, internal APIs)
- Integration tests (cross-browser, database, distributed)

## Directory Structure

The test framework uses a logical directory structure organized by test category:

```
test/
├── api/
│   ├── huggingface/
│   ├── internal/
│   ├── llm_providers/
│   └── local_servers/
├── common/
│   ├── fixtures.py
│   ├── hardware_detection.py
│   └── model_helpers.py
├── docs/
│   ├── README.md
│   ├── MIGRATION_GUIDE.md
│   ├── TEMPLATE_SYSTEM_GUIDE.md
│   └── github-actions-example.yml
├── hardware/
│   ├── cpu/
│   ├── cuda/
│   ├── rocm/
│   ├── webgpu/
│   │   └── compute_shaders/
│   └── webnn/
├── integration/
│   ├── database/
│   └── distributed/
├── models/
│   ├── audio/
│   ├── multimodal/
│   ├── text/
│   │   └── bert/
│   └── vision/
├── template_system/
│   └── templates/
├── conftest.py
├── migrate_tests.py
├── pytest.ini
├── run.py
├── setup_test_env.sh
└── verify_test_environment.py
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pytest 7.0 or higher
- Required libraries in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/ipfs_accelerate_py.git
   cd ipfs_accelerate_py
   ```

2. Set up the test environment:
   ```bash
   cd test
   ./setup_test_env.sh
   ```

3. Verify the environment is correctly set up:
   ```bash
   python verify_test_environment.py
   ```

### Running Tests

The framework provides a unified entry point through `run.py`:

```bash
# Run all tests
python run.py

# Run all model tests
python run.py --test-type model

# Run specific model tests
python run.py --test-type model --model bert

# Run hardware-specific tests
python run.py --test-type hardware --platform webgpu

# Run API tests
python run.py --test-type api --api openai

# Run tests with specific markers
python run.py --markers "slow or webgpu"

# Run distributed tests
python run.py --distributed --worker-count 4
```

## Test Templates

The framework provides templates for creating new tests:

1. Model tests: `template_system/templates/model_test_template.py`
2. Hardware tests: `template_system/templates/hardware_test_template.py`
3. API tests: `template_system/templates/api_test_template.py`

To generate example tests from templates:

```bash
python generate_example_tests.py --all
```

## Common Utilities

Reusable test components are available in the `common` directory:

- `hardware_detection.py`: Utilities for detecting available hardware and skipping tests 
- `model_helpers.py`: Helper functions for loading models and preparing inference inputs
- `fixtures.py`: Common pytest fixtures

## Migrating Tests

For migrating existing tests to the new structure, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

## CI/CD Integration

For GitHub Actions integration, see the example workflow in [github-actions-example.yml](github-actions-example.yml).

## Contributing

1. Follow the directory structure when adding new tests
2. Use the templates to create standardized tests
3. Add appropriate markers for test categorization
4. Create fixtures in `conftest.py` for reusable test components

## Documentation

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md): Guide for migrating tests
- [TEMPLATE_SYSTEM_GUIDE.md](TEMPLATE_SYSTEM_GUIDE.md): Guide for using templates