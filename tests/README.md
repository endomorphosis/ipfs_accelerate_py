# 🧪 IPFS Accelerate Python Test Suite

This directory contains comprehensive tests for the IPFS Accelerate Python framework.

## 📂 Test Structure

### **Core Component Tests**
- `test_accelerate.py` - Main framework functionality tests
- `test_integration.py` - Integration testing suite
- `test_comprehensive.py` - Comprehensive system tests
- `test_smoke_basic.py` - Basic smoke tests for quick validation

### **Advanced Feature Tests**
- `test_advanced_features.py` - Advanced functionality testing
- `test_hardware_mocking.py` - Hardware detection and simulation tests
- `test_real_world_models.py` - Real model integration tests

### **MCP and Communication Tests**
- `test_mcp_client.py` - MCP client functionality tests
- `test_mcp_installation.py` - MCP installation and setup tests

### **AI and Model Tests**
- `test_ai_model_discovery.py` - AI model discovery tests
- `test_model_manager.py` - Model management system tests

### **Repository and Data Tests**
- `test_repo_structure.py` - Repository structure validation
- `test_repo_structure_offline.py` - Offline repository tests
- `test_single_import.py` - Import system tests

### **Analysis and Verification**
- `test_analysis/` - Directory containing test analysis results
- `verification_models.db` - Test verification database
- `test_models.db` - Test model database

### **Testing Tools and Utilities**
- `comprehensive_ai_tester.py` - Comprehensive AI testing framework
- `comprehensive_inference_verifier.py` - Inference verification tools
- `comprehensive_kitchen_sink_app.py` - Kitchen sink testing application
- `comprehensive_pipeline_documenter.py` - Pipeline documentation tools
- `comprehensive_system_tester.py` - System-wide testing tools
- `comprehensive_system_verifier.py` - System verification tools
- `kitchen_sink_pipeline_tester.py` - Pipeline testing utilities
- `playwright_pipeline_screenshots.py` - Visual testing with Playwright
- `run_all_tests.py` - Test runner for all test suites

### **Visual and UI Testing**
- `alternative_browser_tester.py` - Alternative browser testing
- `enhanced_visual_tester.py` - Enhanced visual testing framework
- `ui_test_script.py` - UI testing automation

## 🚀 Running Tests

### **Quick Test Suite**
```bash
# Run basic smoke tests (fastest)
python tests/test_smoke_basic.py

# Run comprehensive tests
python tests/test_comprehensive.py

# Run all tests
python tests/run_all_tests.py
```

### **Specific Test Categories**
```bash
# Core functionality tests
python tests/test_accelerate.py
python tests/test_integration.py

# AI and model tests
python tests/test_ai_model_discovery.py
python tests/test_model_manager.py

# MCP communication tests
python tests/test_mcp_client.py
python tests/test_mcp_installation.py

# Hardware and performance tests
python tests/test_hardware_mocking.py
python tests/test_real_world_models.py
```

### **Advanced Testing Tools**
```bash
# Comprehensive AI inference testing
python tests/comprehensive_ai_tester.py

# System verification
python tests/comprehensive_system_verifier.py

# Kitchen sink testing
python tests/comprehensive_kitchen_sink_app.py

# Visual testing with screenshots
python tests/playwright_pipeline_screenshots.py
```

## 🎯 Test Categories

### **📋 Unit Tests**
Tests for individual components and functions:
- Core framework functionality
- Model management operations
- Hardware detection and simulation
- IPFS integration components

### **🔗 Integration Tests**
Tests for component interactions:
- MCP server integration
- AI model pipeline integration
- Hardware acceleration integration
- Web interface integration

### **🌐 System Tests**
End-to-end system validation:
- Complete inference pipelines
- Multi-modal processing workflows
- Performance and optimization validation
- Enterprise feature validation

### **🎭 Visual Tests**
Browser-based and UI testing:
- Kitchen sink interface testing
- Playwright automation testing
- Screenshot comparison testing
- Cross-browser compatibility testing

## 📊 Test Results and Reporting

### **Test Databases**
- `test_models.db` - Test model metadata and results
- `verification_models.db` - Verification test results

### **Analysis Directory**
- `test_analysis/` - Contains detailed test analysis reports and metrics

### **Running with Coverage**
```bash
# Install coverage tools
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/ --cov=ipfs_accelerate_py --cov-report=html
```

## 🔧 CLI Testing

Test the CLI tool functionality:

```bash
# Test basic CLI commands
ipfs_accelerate system list-models
ipfs_accelerate system available-types

# Test text processing
ipfs_accelerate text generate --prompt "Test text generation"
ipfs_accelerate text classify --text "This is a test"

# Test with different output formats
ipfs_accelerate text generate --prompt "Hello" --output-format json
ipfs_accelerate text generate --prompt "Hello" --output-format pretty
```

## 📋 Prerequisites

Before running tests, ensure you have:

1. **Installed the framework:**
   ```bash
   pip install ipfs_accelerate_py[testing]
   ```

2. **Required testing dependencies:**
   ```bash
   pip install pytest pytest-cov pytest-timeout
   ```

3. **Optional dependencies for specific tests:**
   ```bash
   pip install ipfs_accelerate_py[webnn]  # For browser tests
   pip install ipfs_accelerate_py[full]   # For complete testing
   ```

## 🐛 Troubleshooting Tests

If tests fail:

1. **Check dependencies:**
   ```bash
   python tests/test_single_import.py  # Test import issues
   ```

2. **Run smoke tests first:**
   ```bash
   python tests/test_smoke_basic.py  # Basic functionality
   ```

3. **Check hardware detection:**
   ```bash
   python tests/test_hardware_mocking.py  # Hardware simulation
   ```

4. **Verify MCP setup:**
   ```bash
   python tests/test_mcp_installation.py  # MCP installation
   ```

## 📚 Additional Resources

- [Documentation](../docs/) - Complete documentation
- [Examples](../examples/) - Example implementations
- [Tools](../tools/) - Utility tools and scripts
- [Main README](../README.md) - Project overview
- [Testing README](../docs/TESTING_README.md) - Detailed testing documentation