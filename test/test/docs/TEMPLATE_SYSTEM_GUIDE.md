# Template System Guide

The template system provides a structured way to create standardized tests for models, hardware platforms, and APIs.

## Overview

Templates are used to generate consistent test files with standard structure, imports, and test patterns. The system includes:

1. Base template class
2. Specialized templates for models, hardware, and APIs
3. Template parameters for customization
4. Output path management

## Available Templates

The framework provides three main template types:

1. **Model Test Template**: For testing machine learning models
2. **Hardware Test Template**: For testing hardware-specific features
3. **API Test Template**: For testing external and internal APIs

## Template Directory Structure

Templates are located in the `template_system/templates/` directory:

```
template_system/
└── templates/
    ├── __init__.py
    ├── base_template.py
    ├── model_test_template.py
    ├── hardware_test_template.py
    └── api_test_template.py
```

## Using Templates

### Generating Example Tests

The easiest way to get started is to generate example tests:

```bash
# Generate all example tests
python generate_example_tests.py --all

# Generate only model tests
python generate_example_tests.py --model-tests

# Generate only hardware tests
python generate_example_tests.py --hardware-tests

# Generate only API tests
python generate_example_tests.py --api-tests

# Specify output directory
python generate_example_tests.py --all --output-dir ./new_tests
```

### Interactive Test Generation

Use the interactive test generator for custom tests:

```bash
python generate_test.py
```

This will guide you through the process of creating a custom test.

### Programmatic Usage

You can also use the templates programmatically in your scripts:

```python
from template_system.templates.model_test_template import ModelTestTemplate

# Create a model test
template = ModelTestTemplate(
    template_name="my_custom_bert_test",
    output_dir="./test/models/text/bert",
    parameters={
        'model_name': 'bert-base-uncased',
        'model_type': 'text',
        'test_name': 'bert_base_custom'
    }
)

# Write the test file
file_path = template.write()
print(f"Generated test file: {file_path}")
```

## Template Parameters

### Model Test Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| model_name | Name of the model | "bert-base-uncased" |
| model_type | Type of model (text, vision, audio) | "text" |
| test_name | Base name for the test file | "bert_base_uncased" |
| additional_imports | Extra imports to include | ["numpy as np"] |
| custom_fixtures | Custom pytest fixtures to include | ["bert_tokenizer"] |
| test_batch_sizes | Batch sizes to test | [1, 2, 4, 8] |
| test_sequence_lengths | Sequence lengths to test | [8, 16, 32, 64, 128] |

### Hardware Test Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| hardware_platform | Target platform | "webgpu" |
| test_name | Base name for the test file | "webgpu_matmul" |
| test_category | Category of test | "compute_shaders" |
| test_operation | Operation being tested | "matmul" |
| matrix_sizes | Sizes of matrices for tests | [[32, 32], [64, 64], [128, 128]] |
| custom_imports | Custom imports for the test | ["webgpu.compute"] |

### API Test Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| api_name | Name of the API | "openai" |
| api_type | Type of API | "llm_provider" |
| test_name | Base name for the test file | "openai_api" |
| endpoints | API endpoints to test | ["/v1/chat/completions", "/v1/embeddings"] |
| test_timeout | Timeout for tests in seconds | 30 |
| mock_responses | Use mock responses | True |

## Customizing Templates

### Extending Templates

You can extend the base templates to create custom templates:

```python
from template_system.templates.model_test_template import ModelTestTemplate

class CustomBertTemplate(ModelTestTemplate):
    """Custom template for BERT models with specialized tests."""
    
    def get_template_content(self):
        """Override to add custom content."""
        content = super().get_template_content()
        
        # Add custom test methods
        custom_tests = """
    def test_bert_attention_mask(self):
        \"\"\"Test BERT model with attention mask.\"\"\"
        # Custom test implementation here
        pass
        """
        
        # Insert before the last line
        lines = content.split('\n')
        lines.insert(-1, custom_tests)
        return '\n'.join(lines)
```

### Template Hooks

Templates provide hooks for customization:

- `pre_process()`: Called before template processing
- `post_process(content)`: Called after template processing
- `validate_parameters()`: Called to validate template parameters

## Example Template Files

### Model Test Template Example

Generated model test for BERT:

```python
"""
Test for bert-base-uncased model.

This test verifies the basic functionality of the bert-base-uncased model
including loading, inference, and basic performance metrics.
"""

import pytest
import torch
import time
import os
from test.common.model_helpers import load_model, prepare_input_for_model
from test.common.hardware_detection import skip_if_no_gpu

# Model parameters
MODEL_NAME = "bert-base-uncased"
MODEL_TYPE = "text"


@pytest.fixture
def model():
    """Load the BERT model for testing."""
    return load_model(MODEL_NAME)


@pytest.fixture
def tokenizer():
    """Load the tokenizer for the BERT model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.mark.model
@pytest.mark.bert
class TestBertBaseUncased:
    """Test suite for bert-base-uncased model."""

    def test_model_loading(self, model):
        """Test that the model loads correctly."""
        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("sequence_length", [8, 16, 32, 64, 128])
    def test_model_inference(self, model, tokenizer, batch_size, sequence_length):
        """Test model inference with different batch sizes and sequence lengths."""
        # Prepare input
        inputs = prepare_input_for_model(
            model_type=MODEL_TYPE,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokenizer=tokenizer
        )
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check output shape
        expected_shape = (batch_size, sequence_length, model.config.hidden_size)
        assert outputs.last_hidden_state.shape == expected_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_gpu_inference(self, model, tokenizer):
        """Test model inference on GPU."""
        model = model.to("cuda")
        
        # Prepare input
        inputs = prepare_input_for_model(
            model_type=MODEL_TYPE,
            batch_size=1,
            sequence_length=128,
            tokenizer=tokenizer,
            device="cuda"
        )
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.last_hidden_state.device.type == "cuda"

    def test_performance(self, model, tokenizer):
        """Measure inference performance."""
        # Prepare input
        inputs = prepare_input_for_model(
            model_type=MODEL_TYPE,
            batch_size=1,
            sequence_length=128,
            tokenizer=tokenizer
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(**inputs)
        
        # Measure performance
        iterations = 10
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(**inputs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        print(f"Average inference time: {avg_time:.4f} seconds")
        
        # No specific assertion, just logging performance
```

### Hardware Test Template Example

Generated hardware test for WebGPU:

```python
"""
Test for WebGPU matmul operations.

This test verifies matrix multiplication operations on WebGPU.
"""

import pytest
import numpy as np
import time
import torch
from test.common.hardware_detection import (
    skip_if_no_webgpu,
    is_webgpu_available,
    get_webgpu_device
)


@pytest.fixture
def webgpu_device():
    """Get WebGPU device for testing."""
    if not is_webgpu_available():
        pytest.skip("WebGPU not available")
    return get_webgpu_device()


@pytest.mark.hardware
@pytest.mark.webgpu
@pytest.mark.compute_shaders
class TestWebGPUMatmul:
    """Test suite for WebGPU matmul operations."""

    @skip_if_no_webgpu
    def test_device_available(self, webgpu_device):
        """Test that WebGPU device is available."""
        assert webgpu_device is not None

    @skip_if_no_webgpu
    @pytest.mark.parametrize("matrix_size", [(32, 32), (64, 64), (128, 128), (256, 256)])
    def test_matmul_correctness(self, webgpu_device, matrix_size):
        """Test matrix multiplication correctness with different matrix sizes."""
        m, n = matrix_size
        k = m  # For simplicity, use square matrices
        
        # Create random matrices
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        
        # CPU reference result
        expected = np.matmul(a, b)
        
        # WebGPU computation
        a_tensor = torch.tensor(a, device=webgpu_device)
        b_tensor = torch.tensor(b, device=webgpu_device)
        result_tensor = torch.matmul(a_tensor, b_tensor)
        result = result_tensor.cpu().numpy()
        
        # Check results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    @skip_if_no_webgpu
    @pytest.mark.benchmark
    def test_matmul_performance(self, webgpu_device):
        """Benchmark matrix multiplication performance."""
        matrix_size = 1024
        
        # Create random matrices
        a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        
        # Create tensors
        a_tensor = torch.tensor(a, device=webgpu_device)
        b_tensor = torch.tensor(b, device=webgpu_device)
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(a_tensor, b_tensor)
        
        # Benchmark
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a_tensor, b_tensor)
            webgpu_device.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        print(f"Average matmul time for {matrix_size}x{matrix_size}: {avg_time:.4f} seconds")
        
        # Calculate FLOPS
        flops = 2 * matrix_size**3  # For matrix multiplication
        gflops = flops / (avg_time * 1e9)
        print(f"Performance: {gflops:.2f} GFLOPS")

    @skip_if_no_webgpu
    def test_memory_usage(self, webgpu_device):
        """Test memory usage on WebGPU."""
        # Test with increasing matrix sizes to observe memory usage
        for size in [1024, 2048, 4096]:
            # Skip larger sizes if GPU memory is limited
            if size > 2048 and torch.cuda.get_device_properties(0).total_memory < 8e9:
                continue
                
            # Create random matrices
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            
            # Move to device
            try:
                a_tensor = torch.tensor(a, device=webgpu_device)
                b_tensor = torch.tensor(b, device=webgpu_device)
                result = torch.matmul(a_tensor, b_tensor)
                
                # Check that result is correct shape
                assert result.shape == (size, size)
                
                # Clean up to free memory
                del a_tensor, b_tensor, result
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Out of memory for size {size}x{size}")
                    # This is not a test failure, just a limitation
                    continue
                else:
                    raise
```

### API Test Template Example

Generated API test for OpenAI:

```python
"""
Test for OpenAI API integration.

This test verifies connectivity and functionality of the OpenAI API
including chat completions, embeddings, and error handling.
"""

import pytest
import os
import time
import json
import requests
from unittest import mock

# Conditionally import OpenAI client
try:
    import openai
    has_openai = True
except ImportError:
    has_openai = False

from test.common.fixtures import mock_api_response, api_key


@pytest.fixture
def openai_client():
    """Create an OpenAI client for testing."""
    if not has_openai:
        pytest.skip("OpenAI package not installed")
        
    api_key_env = os.environ.get("OPENAI_API_KEY")
    if not api_key_env:
        pytest.skip("OPENAI_API_KEY environment variable not set")
        
    return openai.OpenAI(api_key=api_key_env)


@pytest.mark.api
@pytest.mark.openai
class TestOpenAIAPI:
    """Test suite for OpenAI API integration."""
    
    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    def test_client_initialization(self, openai_client):
        """Test that the OpenAI client initializes properly."""
        assert openai_client is not None
        assert hasattr(openai_client, "chat")
        assert hasattr(openai_client, "embeddings")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    @pytest.mark.integration
    def test_chat_completion(self, openai_client):
        """Test chat completion API."""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, who are you?"}
                ],
                max_tokens=50
            )
            
            assert response is not None
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], "message")
            assert response.choices[0].message.content != ""
            
        except openai.APIError as e:
            pytest.skip(f"OpenAI API error: {str(e)}")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    @pytest.mark.integration
    def test_embeddings(self, openai_client):
        """Test embeddings API."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input="Hello world"
            )
            
            assert response is not None
            assert hasattr(response, "data")
            assert len(response.data) > 0
            assert hasattr(response.data[0], "embedding")
            assert len(response.data[0].embedding) > 0
            
        except openai.APIError as e:
            pytest.skip(f"OpenAI API error: {str(e)}")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    def test_api_error_handling(self):
        """Test API error handling."""
        # Invalid API key should raise an error
        client = openai.OpenAI(api_key="invalid_key")
        
        with pytest.raises(openai.AuthenticationError):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    @pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4"])
    def test_different_models(self, openai_client, model):
        """Test different models if available."""
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            assert response is not None
            assert hasattr(response, "model")
            assert model in response.model
            
        except openai.APIError as e:
            if "model not found" in str(e).lower():
                pytest.skip(f"Model {model} not available")
            else:
                pytest.skip(f"OpenAI API error: {str(e)}")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    def test_mock_response(self, mock_api_response):
        """Test with mocked API response."""
        mock_data = {
            "choices": [
                {
                    "message": {
                        "content": "This is a mock response",
                        "role": "assistant"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "created": int(time.time()),
            "id": "mock-id",
            "model": "gpt-3.5-turbo",
            "object": "chat.completion"
        }
        
        with mock.patch('openai.resources.chat.Completions.create', return_value=mock_data):
            client = openai.OpenAI(api_key="mock_key")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            assert response is not None
            assert response["choices"][0]["message"]["content"] == "This is a mock response"

    def test_rate_limiting(self, openai_client):
        """Test rate limiting behavior."""
        # Make multiple rapid requests to potentially trigger rate limiting
        with pytest.raises((openai.RateLimitError, openai.APIError)):
            for _ in range(20):  # Excessive number of requests in short period
                openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test message"}],
                    max_tokens=5
                )
```

## Best Practices

1. **Use Standard Templates**: Start with the provided templates for consistency
2. **Customize for Specific Needs**: Extend templates for specialized tests
3. **Include Markers**: Add appropriate markers for test categorization and filtering
4. **Handle Platform Dependencies**: Use skip decorators for platform-specific tests
5. **Document Parameters**: Add docstrings and comments for template parameters
6. **Verify Generated Tests**: Always review and test the generated files

## Troubleshooting

### Common Issues

1. **ImportError in generated tests**: 
   - Check that the package paths are correct
   - Ensure the test environment has all required dependencies

2. **Template parameters missing**:
   - Review the template parameter requirements in the template class
   - Check for typos in parameter names

3. **Output directory issues**:
   - Ensure the output directory exists or can be created
   - Check file permissions if you receive access errors

### Getting Help

If you encounter issues with the template system:

1. Check template class docstrings for parameter requirements
2. Review example generated tests for correct structure
3. Contact the test framework team for assistance