# IPFS Accelerate Test Migration Guide

This guide provides instructions for migrating existing test files to the new test framework structure.

## Overview

The IPFS Accelerate test framework has been refactored to provide:

1. **Organized Structure**: Tests are organized by type (model, hardware, API) with a logical directory structure
2. **Standardized Patterns**: Common patterns for test setup, execution, and teardown
3. **Hardware Detection**: Automatic detection of available hardware platforms
4. **Fixtures and Utilities**: Shared fixtures and utilities for test components
5. **Template System**: Template-based test generation for consistent test files

This guide will help you migrate existing test files to this new structure.

## Migration Process

The migration process involves the following steps:

1. **Analysis**: Analyze existing test files to determine their type and parameters
2. **Generation**: Generate new test files using the template system
3. **Customization**: Customize the generated tests with specific test logic
4. **Verification**: Verify that the migrated tests work as expected

You can perform these steps manually or use the provided migration tools.

## Using the Migration Tools

### Automated Migration

The framework includes a migration script that can automate much of the migration process:

```bash
# Analyze existing tests without migration
python -m test.migrate_tests --source-dir /path/to/tests --analyze-only

# Migrate tests to the new structure
python -m test.migrate_tests --source-dir /path/to/tests --output-dir test

# Perform a dry run without creating files
python -m test.migrate_tests --source-dir /path/to/tests --output-dir test --dry-run

# Filter tests to migrate using a regex pattern
python -m test.migrate_tests --source-dir /path/to/tests --output-dir test --pattern "bert|vit"
```

### Tracking Progress

You can track migration progress using the tracking script:

```bash
# Generate a progress report
python -m test.track_migration_progress \
  --analysis-report migration_analysis.json \
  --migrated-dir test

# Generate an ASCII progress report
python -m test.track_migration_progress \
  --analysis-report migration_analysis.json \
  --migrated-dir test \
  --ascii
```

## Manual Migration

If you prefer to migrate tests manually, follow these steps:

### 1. Determine Test Type

First, determine the type of the test:

- **Model Test**: Tests specific ML models (BERT, T5, ViT, etc.)
- **Hardware Test**: Tests hardware platforms (WebGPU, WebNN, CUDA, ROCm)
- **API Test**: Tests API endpoints, clients, and integrations

### 2. Generate Template

Next, generate a template for the test using the template system:

```bash
# For a model test
python -m test.template_system.generate_test model \
  --model-name bert-base-uncased \
  --model-type text

# For a hardware test
python -m test.template_system.generate_test hardware \
  --hardware-platform webgpu \
  --test-name webgpu_matmul

# For an API test
python -m test.template_system.generate_test api \
  --api-name openai \
  --test-name openai_client
```

### 3. Customize the Test

Edit the generated test file to add the specific test logic from the original test. Pay attention to:

- **Test Setup**: Initialize any required resources
- **Test Logic**: Implement the actual test logic
- **Assertions**: Ensure the test verifies the expected behavior
- **Teardown**: Clean up any resources

### 4. Verify the Test

Run the test to verify that it works as expected:

```bash
# Run the test directly
python -m test.models.text.bert.test_bert_base_uncased

# Run with pytest
pytest test/models/text/bert/test_bert_base_uncased.py -v
```

## Directory Structure

The new test framework uses the following directory structure:

```
test/
├── models/              # Model tests
│   ├── text/            # Text model tests
│   │   ├── bert/        # BERT model tests
│   │   ├── t5/          # T5 model tests
│   │   └── gpt/         # GPT model tests
│   ├── vision/          # Vision model tests
│   │   └── vit/         # ViT model tests
│   ├── audio/           # Audio model tests
│   │   └── whisper/     # Whisper model tests
│   └── multimodal/      # Multimodal model tests
│       └── clip/        # CLIP model tests
├── hardware/            # Hardware tests
│   ├── webgpu/          # WebGPU tests
│   ├── webnn/           # WebNN tests
│   ├── cuda/            # CUDA tests
│   └── rocm/            # ROCm tests
├── api/                 # API tests
│   ├── llm_providers/   # LLM API provider tests
│   ├── huggingface/     # HuggingFace API tests
│   ├── local_servers/   # Local server API tests
│   └── internal/        # Internal API tests
├── common/              # Common utilities
│   ├── hardware_detection.py  # Hardware detection utilities
│   ├── model_helpers.py       # Model loading utilities
│   └── fixtures.py            # Shared fixtures
├── template_system/     # Template system
│   ├── templates/       # Template classes
│   └── generate_test.py # Test generator script
├── migrate_tests.py     # Migration script
├── track_migration_progress.py # Progress tracking script
└── docs/                # Documentation
    ├── README.md        # Main documentation
    ├── MIGRATION_GUIDE.md  # This migration guide
    └── TEMPLATE_SYSTEM_GUIDE.md # Template system guide
```

## Best Practices

When migrating tests, follow these best practices:

1. **Use Fixtures**: Use shared fixtures instead of duplicating setup code
2. **Add Markers**: Add appropriate pytest markers to categorize tests
3. **Skip if Hardware Unavailable**: Use skip decorators for hardware-dependent tests
4. **Use Helper Functions**: Use helper functions for common operations
5. **Follow Naming Conventions**: Use consistent names for tests and test files
6. **Include Documentation**: Add docstrings to test functions and classes
7. **Run Tests from Project Root**: Run tests relative to the project root directory

## Common Migration Patterns

Here are some common migration patterns you may encounter:

### Model Test Migration

**Original Test**:
```python
def test_bert_inference():
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model(**inputs)
    
    assert outputs.last_hidden_state.shape[0] == 1
```

**Migrated Test**:
```python
@pytest.mark.model
@pytest.mark.text
def test_basic_inference(self):
    """Run a basic inference test with the model."""
    model, tokenizer = self.get_model()
    
    if model is None or tokenizer is None:
        pytest.skip("Failed to load model or tokenizer")
    
    try:
        # Prepare input
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Check outputs
        assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
        assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
        
        logger.info("Basic inference test passed")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        pytest.fail(f"Inference failed: {e}")
```

### Hardware Test Migration

**Original Test**:
```python
def test_webgpu_available():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    
    options = Options()
    options.add_argument("--enable-webgpu")
    driver = webdriver.Chrome(options=options)
    
    driver.get("file://path/to/test.html")
    result = driver.execute_script("return navigator.gpu !== undefined")
    
    assert result
    driver.quit()
```

**Migrated Test**:
```python
@pytest.mark.webgpu
def test_webgpu_available(self):
    """Test WebGPU availability."""
    hardware_info = detect_hardware()
    assert hardware_info['platforms']['webgpu']['available']

@pytest.mark.webgpu
def test_webgpu_browser_launch(self, webgpu_browser):
    """Test WebGPU browser launch."""
    assert webgpu_browser is not None
```

### API Test Migration

**Original Test**:
```python
def test_openai_api():
    import openai
    
    openai.api_key = "test_key"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Hello, world!",
        max_tokens=5
    )
    
    assert response.choices[0].text
```

**Migrated Test**:
```python
@pytest.mark.api
def test_chat_completion(self, openai_client):
    """Test chat completion with OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        )
        
        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content
    except Exception as e:
        pytest.skip(f"API test failed: {e}")
```

## Troubleshooting

Common migration issues and solutions:

1. **Import Errors**: Use the project root in the Python path with `sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))`
2. **Hardware Detection**: Ensure the hardware detection utilities are properly initialized
3. **Fixture Not Found**: Ensure pytest can find the fixtures by importing them in `conftest.py`
4. **Dependency Missing**: Install required dependencies for the test
5. **Path Issues**: Use absolute paths or paths relative to the project root

## Getting Help

If you encounter issues during migration, you can:

1. Check the documentation in the `docs` directory
2. Look at example tests in the migrated test directories
3. Use the `--analyze-only` flag with the migration script to get more information about the test
4. Run the test with the `-v` flag to get more verbose output

## Conclusion

By following this guide, you can migrate your existing tests to the new framework structure, leading to more maintainable and consistent tests. The migration process may take some time, but the benefits in terms of test organization, standardization, and reusability are worth the effort.

Remember to run the migrated tests to ensure they work as expected, and to update the migration progress report regularly to track your progress.