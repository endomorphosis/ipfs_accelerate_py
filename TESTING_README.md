# 🧪 IPFS Accelerate Python - Testing Infrastructure

This document describes the comprehensive testing infrastructure for IPFS Accelerate Python, designed to enable full functionality testing without requiring GPU hardware or complex dependencies.

## 🎯 Overview

The testing infrastructure provides:

- **CPU-Only Execution**: All tests run without requiring GPUs, special hardware, or heavy dependencies
- **Hardware Simulation**: Mock backends for CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU, and Qualcomm
- **Comprehensive Coverage**: 26+ test cases covering all major functionality
- **CI/CD Ready**: Pytest-compatible tests that run in GitHub Actions
- **Performance Validation**: Tests ensure operations complete within reasonable time limits

## 📊 Test Statistics

| Test Suite | Test Count | Coverage | Run Time |
|------------|------------|----------|----------|
| Smoke Tests | 6 tests | Basic functionality | ~1s |
| Comprehensive Tests | 16 tests | Core features | ~8s |
| Integration Tests | 10 tests | End-to-end workflows | ~6s |
| **Total** | **32 tests** | **Full system** | **~15s** |

## 🚀 Quick Start

### Run All Tests
```bash
# Run all test suites
python test_smoke_basic.py
python test_comprehensive.py
python test_integration.py

# Or use pytest
pytest test_smoke_basic.py test_comprehensive.py test_integration.py -v
```

### Run Specific Test Categories
```bash
# Basic functionality only
python test_smoke_basic.py

# Core features with mocking
python test_comprehensive.py

# Integration and performance tests
python test_integration.py
```

## 📋 Test Suites

### 1. Smoke Tests (`test_smoke_basic.py`)
Basic functionality verification that runs quickly and tests essential features.

**Test Cases:**
- ✅ `test_hardware_detection_import` - Module import verification
- ✅ `test_hardware_detection_basic` - Basic hardware detection
- ✅ `test_detect_available_hardware_function` - Main detection function
- ✅ `test_model_hardware_compatibility` - Model compatibility checking
- ✅ `test_get_hardware_detection_code` - Code generation
- ✅ `test_basic_ipfs_accelerate_import` - Main package import

**Usage:**
```bash
python test_smoke_basic.py
# Output: 6/6 tests passed ✅
```

### 2. Comprehensive Tests (`test_comprehensive.py`)
Detailed testing of all core functionality with mocking and error handling.

**Test Categories:**
- **Hardware Detection Core** (5 tests): CPU detection, hardware details, detector creation
- **Hardware Mocking** (3 tests): Environment variables, priority selection, OpenVINO mocking
- **Code Generation** (2 tests): Hardware detection code generation and validation
- **Browser Features** (1 test): Browser environment detection
- **Performance & Integration** (3 tests): Performance validation, caching, multiple instances
- **Error Handling** (2 tests): Invalid inputs, cache errors

**Usage:**
```bash
python test_comprehensive.py
# Output: 16/16 tests passed ✅
```

### 3. Integration Tests (`test_integration.py`)
End-to-end integration testing with realistic workflows and performance validation.

**Test Categories:**
- **IPFS Accelerate Integration** (3 tests): Main module imports, web compatibility
- **Hardware Integration** (3 tests): Full detection workflow, model compatibility, priorities
- **Performance Integration** (2 tests): Concurrent execution, caching performance
- **Configuration Integration** (2 tests): Environment variables, cache configurations

**Usage:**
```bash
python test_integration.py  
# Output: 10/10 tests passed ✅
```

## 🎭 Hardware Mocking System

The hardware mocking system (`test_hardware_mocking.py`) provides realistic simulation of all supported hardware types.

### Supported Hardware Backends
- **CPU**: Always available (baseline)
- **CUDA**: NVIDIA GPU simulation
- **ROCm**: AMD GPU simulation
- **MPS**: Apple Silicon simulation
- **OpenVINO**: Intel hardware simulation
- **WebNN**: Browser neural network API simulation
- **WebGPU**: Browser GPU API simulation
- **Qualcomm**: Mobile/Edge AI simulation

### Environment Variable Controls
```bash
# Enable specific mock hardware
export MOCK_CUDA=true
export MOCK_OPENVINO=true
export MOCK_WEBNN=true
export MOCK_WEBGPU=true

# Run tests with mock hardware
python test_comprehensive.py
```

### Programmatic Mock Usage
```python
from test_hardware_mocking import create_cuda_environment

# Create CUDA mock environment
mocker = create_cuda_environment()
with mocker.mock_hardware_environment():
    # Your code here will see CUDA as available
    import hardware_detection
    detector = hardware_detection.HardwareDetector()
    available = detector.get_available_hardware()
    print(available)  # {'cpu': True, 'cuda': True, ...}
```

## 🔧 CI/CD Integration

### GitHub Actions Workflow
The repository includes a comprehensive GitHub Actions workflow (`.github/workflows/test-cpu-only.yml`) that:

- Tests on Python 3.8-3.12
- Tests on Ubuntu, Windows, macOS
- Tests multiple installation modes (editable, wheel, minimal)
- Runs all test suites
- Tests with mock hardware environments
- Generates detailed test reports

### Running in CI
```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install pytest
        pip install -e .
    - name: Run tests
      run: |
        pytest test_smoke_basic.py test_comprehensive.py test_integration.py -v
```

## ⚡ Performance Requirements

All tests are designed to complete quickly without external dependencies:

| Requirement | Target | Actual |
|-------------|--------|--------|
| Smoke tests | < 2s | ~1s ✅ |
| Comprehensive tests | < 10s | ~8s ✅ |
| Integration tests | < 10s | ~6s ✅ |
| Hardware detection | < 1s per call | ~0.3s ✅ |
| Concurrent detection | < 2s per thread | ~0.7s ✅ |

## 🛠️ Development Workflow

### Adding New Tests
1. **Smoke Tests**: Add basic functionality verification
2. **Comprehensive Tests**: Add detailed feature testing with mocking
3. **Integration Tests**: Add end-to-end workflow testing

### Test Development Guidelines
- **CPU-Only**: Tests must run without GPU or special hardware
- **Fast Execution**: Individual tests should complete in < 5s
- **Isolated**: Tests should not depend on external services
- **Deterministic**: Tests should produce consistent results
- **Informative**: Tests should provide clear error messages

### Mock Hardware Development
```python
# Adding a new mock hardware type
class MockHardwareConfig:
    def __init__(self):
        self.enabled_hardware = {
            # ... existing hardware ...
            'new_hardware': os.environ.get('MOCK_NEW_HARDWARE', 'false').lower() in ('true', '1', 'yes'),
        }
        
        self.hardware_capabilities = {
            # ... existing capabilities ...
            'new_hardware': {
                'version': '1.0',
                'features': ['feature1', 'feature2'],
            }
        }
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Issue: Module not found errors
# Solution: Ensure you're running from the correct directory
cd /path/to/ipfs_accelerate_py
python test_smoke_basic.py
```

**Slow Tests**
```bash
# Issue: Tests taking too long
# Solution: Check if external dependencies are being loaded
export MINIMAL_DEPENDENCIES=1
python test_comprehensive.py
```

**Mock Hardware Not Working**
```bash
# Issue: Mock hardware not detected
# Solution: Set environment variables correctly
export MOCK_CUDA=true
export MOCK_WEBGPU=true
python -c "
import hardware_detection
detector = hardware_detection.HardwareDetector()
print(detector.get_available_hardware())
"
```

### Debug Mode
```bash
# Enable debug logging for all tests
export TEST_DEBUG=1
python test_comprehensive.py

# Enable hardware detection debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import hardware_detection
detector = hardware_detection.HardwareDetector()
result = detector.get_available_hardware()
"
```

## 📈 Coverage Report

Generate test coverage reports:

```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage
pytest test_comprehensive.py --cov=hardware_detection --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🎯 Next Steps

### Planned Enhancements
1. **Model-Specific Tests**: Add tests for specific ML model architectures
2. **Performance Benchmarking**: Add realistic performance simulation
3. **Web Platform Testing**: Enhanced browser environment testing  
4. **Distributed Testing**: Multi-node testing capabilities
5. **Regression Testing**: Automated detection of performance regressions

### Contributing Tests
1. Fork the repository
2. Add tests following the existing patterns
3. Ensure all tests pass: `pytest test_*.py -v`
4. Submit a pull request with test descriptions

## 📚 References

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**📧 Support**: For testing-related questions, please open an issue with the `testing` label.

**🔄 Last Updated**: September 2025