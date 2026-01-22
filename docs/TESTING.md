# IPFS Accelerate Python - Testing Guide

This guide covers the comprehensive testing framework and methodologies used in the IPFS Accelerate Python project.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Benchmark Testing](#benchmark-testing)
- [Hardware Testing](#hardware-testing)
- [API Testing](#api-testing)
- [Browser Testing](#browser-testing)
- [Integration Testing](#integration-testing)
- [Performance Testing](#performance-testing)
- [CI/CD Testing](#cicd-testing)
- [Writing Tests](#writing-tests)

## Testing Overview

The IPFS Accelerate Python project uses a comprehensive multi-layered testing strategy:

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions and workflows
- **Performance Tests**: Benchmark and performance regression testing
- **Hardware Tests**: Hardware-specific acceleration testing
- **Browser Tests**: WebNN/WebGPU browser integration testing
- **API Tests**: External API integration testing
- **End-to-End Tests**: Complete workflow validation

## Test Structure

```
test/
├── README.md                          # This document
├── DOCUMENTATION_INDEX.md            # Index of all test documentation
├── unit/                             # Unit tests
│   ├── test_core_framework.py
│   ├── test_hardware_detection.py
│   └── test_ipfs_integration.py
├── integration/                      # Integration tests
│   ├── test_end_to_end_workflows.py
│   ├── test_database_integration.py
│   └── test_model_pipelines.py
├── performance/                      # Performance and benchmark tests
│   ├── benchmark_core/
│   ├── hardware_benchmarks/
│   └── regression_tests/
├── hardware/                         # Hardware-specific tests
│   ├── test_cuda_acceleration.py
│   ├── test_openvino_integration.py
│   ├── test_webnn_webgpu.py
│   └── test_apple_mps.py
├── api/                             # API integration tests
│   ├── test_huggingface_api.py
│   ├── test_openai_api.py
│   └── test_custom_apis.py
├── browser/                         # Browser integration tests
│   ├── test_webnn_integration.py
│   ├── test_webgpu_integration.py
│   └── test_cross_browser.py
├── mobile/                          # Mobile and edge device tests
│   ├── android_test_harness/
│   ├── ios_test_harness/
│   └── edge_device_tests/
├── data/                           # Test data and fixtures
│   ├── sample_models/
│   ├── test_inputs/
│   └── expected_outputs/
└── utils/                          # Test utilities and helpers
    ├── test_helpers.py
    ├── mock_servers.py
    └── test_fixtures.py
```

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install -e ".[dev]"

# Or install specific test requirements
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ipfs_accelerate_py --cov-report=html

# Run specific test category
pytest test/unit/
pytest test/integration/
pytest test/performance/

# Run specific test file
pytest test/unit/test_core_framework.py

# Run specific test function
pytest test/unit/test_core_framework.py::test_initialization

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "gpu"       # Run only GPU tests
pytest -m "browser"   # Run only browser tests
```

### Advanced Test Options

```bash
# Run tests in parallel (with pytest-xdist)
pytest -n auto

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x

# Run tests with specific log level
pytest --log-level=DEBUG

# Run tests with specific hardware
pytest -m "cuda" --hardware=cuda
pytest -m "openvino" --hardware=openvino

# Run performance tests with specific duration
pytest test/performance/ --benchmark-min-time=5
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
# test/unit/test_core_framework.py
import pytest
from ipfs_accelerate_py import ipfs_accelerate_py

class TestCoreFramework:
    def test_initialization(self):
        """Test basic framework initialization."""
        accelerator = ipfs_accelerate_py({}, {})
        assert accelerator is not None
        
    def test_hardware_detection(self):
        """Test hardware detection functionality."""
        accelerator = ipfs_accelerate_py({}, {})
        if hasattr(accelerator, 'hardware_detection'):
            hardware_info = accelerator.hardware_detection.detect_all_hardware()
            assert isinstance(hardware_info, dict)
            assert 'cpu' in hardware_info
            
    @pytest.mark.asyncio
    async def test_async_processing(self):
        """Test asynchronous processing."""
        accelerator = ipfs_accelerate_py({}, {})
        result = await accelerator.process_async(
            model="bert-base-uncased",
            input_data={"input_ids": [101, 102, 103]},
            endpoint_type="text_embedding"
        )
        assert result is not None
```

### Integration Tests

Test component interactions:

```python
# test/integration/test_end_to_end_workflows.py
import pytest
import asyncio
from ipfs_accelerate_py import ipfs_accelerate_py

class TestEndToEndWorkflows:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_inference_pipeline(self):
        """Test complete inference pipeline."""
        # Initialize with IPFS configuration
        config = {
            "ipfs": {"local_node": "http://localhost:5001"},
            "hardware": {"prefer_cuda": True}
        }
        accelerator = ipfs_accelerate_py(config, {})
        
        # Test local inference
        local_result = accelerator.process(
            model="bert-base-uncased",
            input_data={"input_ids": [101, 2054, 2003, 102]},
            endpoint_type="text_embedding"
        )
        assert local_result is not None
        
        # Test IPFS-accelerated inference
        ipfs_result = await accelerator.accelerate_inference(
            model="bert-base-uncased",
            input_data={"input_ids": [101, 2054, 2003, 102]},
            use_ipfs=True
        )
        assert ipfs_result is not None
```

## Benchmark Testing

### Performance Benchmarks

```python
# test/performance/test_model_benchmarks.py
import pytest
import time
from ipfs_accelerate_py import ipfs_accelerate_py

class TestModelBenchmarks:
    @pytest.mark.benchmark
    @pytest.mark.parametrize("model", [
        "bert-base-uncased",
        "gpt2",
        "vit-base-patch16-224"
    ])
    def test_model_inference_speed(self, benchmark, model):
        """Benchmark model inference speed."""
        accelerator = ipfs_accelerate_py({}, {})
        
        # Prepare test data based on model type
        if "bert" in model:
            input_data = {"input_ids": [101] + list(range(100)) + [102]}
        elif "gpt" in model:
            input_data = {"input_ids": [101] + list(range(50))}
        elif "vit" in model:
            import torch
            input_data = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        # Benchmark the inference
        result = benchmark(
            accelerator.process,
            model=model,
            input_data=input_data,
            endpoint_type="auto"
        )
        assert result is not None
        
    @pytest.mark.benchmark
    @pytest.mark.parametrize("hardware", ["cpu", "cuda", "openvino"])
    def test_hardware_performance(self, benchmark, hardware):
        """Benchmark performance across hardware types."""
        config = {"hardware": {hardware: True}}
        accelerator = ipfs_accelerate_py(config, {})
        
        result = benchmark(
            accelerator.process,
            model="bert-base-uncased",
            input_data={"input_ids": [101, 2054, 2003, 102]},
            endpoint_type="text_embedding"
        )
        assert result is not None
```

### Running Benchmarks

```bash
# Run all benchmarks
pytest test/performance/ --benchmark-only

# Run benchmarks with comparison
pytest test/performance/ --benchmark-compare=previous_results.json

# Save benchmark results
pytest test/performance/ --benchmark-json=benchmark_results.json

# Run benchmarks with specific hardware
pytest test/performance/ -m "cuda" --hardware=cuda

# Generate benchmark report
pytest test/performance/ --benchmark-histogram=histogram.svg
```

## Hardware Testing

### Hardware-Specific Tests

```python
# test/hardware/test_cuda_acceleration.py
import pytest
import torch
from ipfs_accelerate_py import ipfs_accelerate_py

class TestCudaAcceleration:
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_initialization(self):
        """Test CUDA acceleration initialization."""
        config = {"hardware": {"prefer_cuda": True}}
        accelerator = ipfs_accelerate_py(config, {})
        
        # Check if CUDA is detected
        if hasattr(accelerator, 'hardware_detection'):
            hardware_info = accelerator.hardware_detection.detect_all_hardware()
            assert hardware_info["cuda"]["available"] is True
            
    @pytest.mark.cuda
    @pytest.mark.slow
    def test_cuda_inference_performance(self):
        """Test CUDA inference performance."""
        config = {"hardware": {"prefer_cuda": True}}
        accelerator = ipfs_accelerate_py(config, {})
        
        # Test with larger input for meaningful GPU utilization
        large_input = {"input_ids": [101] + list(range(1000)) + [102]}
        
        start_time = time.time()
        result = accelerator.process(
            model="bert-base-uncased",
            input_data=large_input,
            endpoint_type="text_embedding"
        )
        inference_time = time.time() - start_time
        
        assert result is not None
        # GPU should be faster than 1 second for this size input
        assert inference_time < 1.0
```

### Hardware Test Configuration

```bash
# Run CUDA tests only
pytest -m "cuda" test/hardware/

# Run all hardware tests except slow ones
pytest -m "gpu and not slow" test/hardware/

# Run specific hardware tests
pytest test/hardware/test_openvino_integration.py
pytest test/hardware/test_webnn_webgpu.py
```

## API Testing

### External API Integration Tests

```python
# test/api/test_huggingface_api.py
import pytest
from ipfs_accelerate_py import ipfs_accelerate_py

class TestHuggingFaceIntegration:
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_hf_model_loading(self):
        """Test HuggingFace model loading and inference."""
        accelerator = ipfs_accelerate_py({}, {})
        
        # Test with a small, fast model
        result = accelerator.process(
            model="distilbert-base-uncased",
            input_data={"input_ids": [101, 2054, 2003, 102]},
            endpoint_type="text_embedding"
        )
        
        assert result is not None
        # Check if result has expected structure
        if isinstance(result, dict):
            assert "embedding" in result or "output" in result
            
    @pytest.mark.api
    @pytest.mark.integration
    def test_model_family_detection(self):
        """Test automatic model family detection."""
        accelerator = ipfs_accelerate_py({}, {})
        
        if hasattr(accelerator, 'model_classifier'):
            # Test different model types
            test_models = [
                ("bert-base-uncased", "text_embedding"),
                ("gpt2", "text_generation"),
                ("vit-base-patch16-224", "vision"),
            ]
            
            for model_name, expected_type in test_models:
                # This would test model family classification
                # Implementation depends on the actual classifier
                pass
```

## Browser Testing

### WebNN/WebGPU Tests

```python
# test/browser/test_webnn_integration.py
import pytest
from ipfs_accelerate_py.webnn_webgpu_integration import accelerate_with_browser

class TestWebNNIntegration:
    @pytest.mark.browser
    @pytest.mark.webnn
    @pytest.mark.asyncio
    async def test_webnn_inference(self):
        """Test WebNN browser inference."""
        try:
            result = await accelerate_with_browser(
                model_name="bert-base-uncased",
                inputs={"input_ids": [101, 2054, 2003, 102]},
                platform="webnn",
                browser="edge",
                precision=16
            )
            assert result is not None
            assert "inference_time" in result
        except Exception as e:
            pytest.skip(f"WebNN not available: {e}")
            
    @pytest.mark.browser
    @pytest.mark.webgpu
    @pytest.mark.parametrize("browser", ["chrome", "firefox", "edge"])
    async def test_cross_browser_webgpu(self, browser):
        """Test WebGPU across different browsers."""
        try:
            result = await accelerate_with_browser(
                model_name="bert-base-uncased",
                inputs={"input_ids": [101, 2054, 2003, 102]},
                platform="webgpu",
                browser=browser,
                precision=16
            )
            assert result is not None
        except Exception as e:
            pytest.skip(f"WebGPU not available in {browser}: {e}")
```

### Browser Test Setup

```bash
# Install browser testing dependencies
pip install playwright selenium

# Setup browser drivers
playwright install

# Run browser tests
pytest -m "browser" test/browser/

# Run WebNN tests only
pytest -m "webnn" test/browser/

# Run with specific browser
pytest test/browser/ --browser=chrome
```

## Performance Testing

### Memory and CPU Profiling

```python
# test/performance/test_memory_profiling.py
import pytest
import psutil
import time
from ipfs_accelerate_py import ipfs_accelerate_py

class TestMemoryProfiling:
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during inference."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        accelerator = ipfs_accelerate_py({}, {})
        
        # Run multiple inferences to check for memory leaks
        for i in range(10):
            result = accelerator.process(
                model="bert-base-uncased",
                input_data={"input_ids": [101, 2054, 2003, 102]},
                endpoint_type="text_embedding"
            )
            assert result is not None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for this test)
        assert memory_increase < 500
        
    @pytest.mark.performance
    @pytest.mark.slow
    def test_sustained_performance(self):
        """Test performance under sustained load."""
        accelerator = ipfs_accelerate_py({}, {})
        
        times = []
        for i in range(100):
            start_time = time.time()
            result = accelerator.process(
                model="bert-base-uncased",
                input_data={"input_ids": [101, 2054, 2003, 102]},
                endpoint_type="text_embedding"
            )
            times.append(time.time() - start_time)
            
        # Performance should be consistent
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Max time shouldn't be more than 3x average time
        assert max_time < avg_time * 3
```

## CI/CD Testing

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run unit tests
      run: |
        pytest test/unit/ -v --cov=ipfs_accelerate_py
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      
  hardware-tests:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        
    - name: Run hardware detection tests
      run: |
        pytest test/hardware/test_hardware_detection.py -v
        
  browser-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        playwright install
        
    - name: Run browser tests
      run: |
        pytest -m "browser" test/browser/ -v
```

### Test Configuration

```ini
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra 
    --strict-markers 
    --strict-config
    --cov=ipfs_accelerate_py
    --cov-branch
    --cov-report=term-missing
    --cov-fail-under=80
testpaths = test
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    benchmark: Benchmark tests
    slow: Slow tests (> 5 seconds)
    gpu: GPU-accelerated tests
    cuda: CUDA-specific tests
    openvino: OpenVINO-specific tests
    mps: Apple MPS tests
    browser: Browser integration tests
    webnn: WebNN tests
    webgpu: WebGPU tests
    api: API integration tests
    mobile: Mobile device tests
    edge: Edge device tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

## Writing Tests

### Test Writing Guidelines

1. **Naming Convention**: Use descriptive test names starting with `test_`
2. **Structure**: Follow Arrange-Act-Assert pattern
3. **Independence**: Tests should be independent and not rely on each other
4. **Cleanup**: Use fixtures for setup/teardown
5. **Parametrization**: Use `@pytest.mark.parametrize` for similar tests with different inputs
6. **Markers**: Use appropriate markers to categorize tests
7. **Async Support**: Use `@pytest.mark.asyncio` for async tests

### Test Fixtures

```python
# test/conftest.py
import pytest
from ipfs_accelerate_py import ipfs_accelerate_py

@pytest.fixture
def accelerator():
    """Fixture providing a basic accelerator instance."""
    return ipfs_accelerate_py({}, {})

@pytest.fixture
def cuda_accelerator():
    """Fixture providing a CUDA-enabled accelerator instance."""
    config = {"hardware": {"prefer_cuda": True}}
    return ipfs_accelerate_py(config, {})

@pytest.fixture
def sample_text_input():
    """Fixture providing sample text input data."""
    return {"input_ids": [101, 2054, 2003, 2026, 2171, 102]}

@pytest.fixture
def sample_vision_input():
    """Fixture providing sample vision input data."""
    import torch
    return {"pixel_values": torch.randn(1, 3, 224, 224)}

@pytest.fixture(scope="session")
def ipfs_node():
    """Fixture for IPFS node setup/teardown."""
    # Setup IPFS node for testing
    # This would start a test IPFS node
    yield "http://localhost:5001"
    # Cleanup after all tests
```

### Example Test File

```python
# test/unit/test_example.py
import pytest
from ipfs_accelerate_py import ipfs_accelerate_py

class TestExampleFramework:
    """Example test class demonstrating best practices."""
    
    def test_basic_initialization(self, accelerator):
        """Test basic framework initialization."""
        assert accelerator is not None
        assert hasattr(accelerator, 'process')
        
    @pytest.mark.asyncio
    async def test_async_processing(self, accelerator, sample_text_input):
        """Test asynchronous processing functionality."""
        result = await accelerator.process_async(
            model="bert-base-uncased",
            input_data=sample_text_input,
            endpoint_type="text_embedding"
        )
        assert result is not None
        
    @pytest.mark.parametrize("model,input_type", [
        ("bert-base-uncased", "text"),
        ("gpt2", "text"),
        ("vit-base-patch16-224", "vision"),
    ])
    def test_multiple_models(self, accelerator, model, input_type):
        """Test processing with different model types."""
        if input_type == "text":
            input_data = {"input_ids": [101, 2054, 2003, 102]}
        elif input_type == "vision":
            import torch
            input_data = {"pixel_values": torch.randn(1, 3, 224, 224)}
            
        result = accelerator.process(
            model=model,
            input_data=input_data,
            endpoint_type=input_type
        )
        assert result is not None
        
    @pytest.mark.gpu
    @pytest.mark.skipif("not torch.cuda.is_available()")
    def test_gpu_acceleration(self, cuda_accelerator):
        """Test GPU acceleration when available."""
        result = cuda_accelerator.process(
            model="bert-base-uncased",
            input_data={"input_ids": [101, 2054, 2003, 102]},
            endpoint_type="text_embedding"
        )
        assert result is not None
```

## Running Specific Test Suites

```bash
# Run unit tests only
pytest test/unit/ -v

# Run integration tests with coverage
pytest test/integration/ --cov=ipfs_accelerate_py

# Run performance tests (may take longer)
pytest test/performance/ -m "not slow"

# Run hardware tests for specific hardware
pytest test/hardware/ -m "cuda"

# Run browser tests (requires browser setup)
pytest test/browser/ -m "browser"

# Run all tests except slow ones
pytest -m "not slow"

# Run tests in parallel
pytest -n auto

# Run tests with specific timeout
pytest --timeout=300  # 5 minute timeout per test
```

This comprehensive testing framework ensures the reliability, performance, and compatibility of the IPFS Accelerate Python framework across different hardware platforms, browsers, and use cases.

## Related Documentation

- [Usage Guide](USAGE.md) - How to use the framework
- [API Reference](API.md) - Complete API documentation
- [Hardware Optimization](HARDWARE.md) - Hardware-specific features
- [Installation Guide](INSTALLATION.md) - Setup and installation
- [Architecture Guide](ARCHITECTURE.md) - System architecture overview