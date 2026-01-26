# Browser Environment Validation Guide

This guide explains how to use the browser environment validation system to detect and validate WebNN and WebGPU capabilities across different browsers in the IPFS Accelerate project.

## Overview

The browser environment validation system automatically detects and tests WebNN and WebGPU capabilities across Chrome, Firefox, Edge, and Safari. It can determine whether real hardware acceleration is available or if browsers are using simulation mode. The system also validates the enhanced features from the March 2025 release (compute shaders, shader precompilation, and parallel model loading).

## Why Browser Environment Validation Matters

Browser environment validation is crucial for several reasons:

1. **Performance Optimization**: Different browsers have varying levels of WebNN and WebGPU support. Knowing which browsers provide real hardware acceleration allows you to optimize performance.

2. **Compatibility Assurance**: Before deploying web-based AI models, you need to know which browsers can run them efficiently with hardware acceleration.

3. **Model-Specific Recommendations**: Different model types (text, vision, audio) perform best on different browser-platform combinations.

4. **Enhanced Feature Validation**: The March 2025 features (compute shaders, shader precompilation, parallel loading) need validation across browsers to ensure they deliver the promised performance gains.

5. **Simulation vs. Real Hardware**: Distinguishing between simulation mode and real hardware acceleration is essential for accurate performance expectations.

## Features

- **Cross-Browser Testing**: Test Chrome, Firefox, Edge, and Safari with a single command
- **Platform Detection**: Check WebNN and WebGPU hardware acceleration availability and capabilities
- **Real Hardware vs. Simulation**: Distinguish between real hardware acceleration and software simulation
- **March 2025 Features**: Test enhanced features like compute shader optimization, shader precompilation, and parallel model loading
- **Model-specific Recommendations**: Get recommended browser and platform combinations for different model types (text, vision, audio)
- **CI/CD Integration**: Automatically run validation tests in CI/CD pipelines with GitHub Actions
- **Interactive Dashboard**: Visualize test results with an interactive web dashboard
- **Selenium Integration**: Use Selenium for automated browser testing with hardware acceleration detection
- **Advanced Hardware Detection**: Detailed reporting on GPU capabilities and WebGPU feature support
- **Comprehensive Reporting**: Generate HTML, Markdown, JSON, and CSV reports with detailed test results
- **Browser-Specific Optimizations**: Test specialized optimizations like Firefox's compute shader performance for audio models

## Command-line Usage

### Basic Commands

The simplest way to check browser capabilities is using the `check_browser_webnn_webgpu.py` script:

```bash
# Check all browsers for WebNN and WebGPU support
python check_browser_webnn_webgpu.py --check-all

# Check a specific browser for WebNN and WebGPU support
python check_browser_webnn_webgpu.py --browser chrome

# Check a specific browser for a specific platform
python check_browser_webnn_webgpu.py --browser firefox --platform webgpu

# Check if Edge supports WebNN with hardware acceleration
python check_browser_webnn_webgpu.py --browser edge --platform webnn
```

### Enhanced Feature Testing

To test the March 2025 features, you can use environment variables:

```bash
# Test with compute shader optimization (especially useful for audio models in Firefox)
WEBGPU_COMPUTE_SHADERS_ENABLED=1 python check_browser_webnn_webgpu.py --browser firefox --platform webgpu

# Test with shader precompilation (for faster startup)
WEBGPU_SHADER_PRECOMPILE_ENABLED=1 python check_browser_webnn_webgpu.py --browser chrome --platform webgpu

# Test with parallel model loading (for multimodal models)
WEB_PARALLEL_LOADING_ENABLED=1 python check_browser_webnn_webgpu.py --browser chrome --platform webgpu

# Enable all optimizations
WEBGPU_COMPUTE_SHADERS_ENABLED=1 WEBGPU_SHADER_PRECOMPILE_ENABLED=1 WEB_PARALLEL_LOADING_ENABLED=1 python check_browser_webnn_webgpu.py --browser chrome --platform webgpu
```

### Running in CI/CD

The browser environment validation system includes a GitHub Actions workflow for automated testing. You can run it manually or on push/PR events:

1. Navigate to your repository on GitHub
2. Go to the "Actions" tab
3. Select the "Browser Environment Validation" workflow
4. Click "Run workflow"
5. Select options as needed:
   - **Browser**: chrome, firefox, edge, or all
   - **Platform**: webnn, webgpu, or all
   - **Feature Flags**: comma-separated list of features (compute_shaders, shader_precompile, parallel_loading)

The workflow will:
1. Detect and run tests on selected browsers
2. Generate a detailed validation report
3. Create an interactive dashboard (for push events to main)
4. Display a summary in the workflow run logs

## Programmatic Usage

### Basic WebNN/WebGPU Detection

```python
import anyio
from check_browser_webnn_webgpu import check_browser_capabilities

async def test_browser():
    # Check if Chrome supports WebGPU
    capabilities = await check_browser_capabilities(
        browser="chrome",
        platform="webgpu",
        headless=True
    )
    
    # Print results
    print(f"WebGPU supported: {capabilities.get('webgpu', {}).get('supported', False)}")
    print(f"Real hardware: {capabilities.get('webgpu', {}).get('real', False)}")
    
    # Get device details
    details = capabilities.get('webgpu', {}).get('details', {})
    if details:
        print(f"Vendor: {details.get('vendor')}")
        print(f"Device: {details.get('device')}")
        print(f"Architecture: {details.get('architecture')}")

# Run the test
anyio.run(test_browser)
```

### Using the BrowserAutomation Class

For more advanced scenarios, you can use the `BrowserAutomation` class directly:

```python
import anyio
from fixed_web_platform.browser_automation import BrowserAutomation

async def run_browser_test():
    # Create automation instance
    automation = BrowserAutomation(
        platform="webgpu",
        browser_name="firefox",
        headless=False,
        compute_shaders=True,  # Enable compute shader optimization
        precompile_shaders=True,  # Enable shader precompilation
        parallel_loading=False,  # Disable parallel loading
        model_type="audio"  # Testing for audio models
    )
    
    try:
        # Launch browser
        success = await automation.launch(allow_simulation=True)
        if not success:
            print("Failed to launch browser")
            return
        
        # Check if using real hardware or simulation
        is_simulation = getattr(automation, 'simulation_mode', True)
        print(f"Using simulation mode: {is_simulation}")
        
        # Run a test
        result = await automation.run_test("whisper-tiny", "Test input")
        print(f"Test result: {result}")
        
    finally:
        # Close browser
        await automation.close()

# Run the test
anyio.run(run_browser_test)
```

## Selenium Integration

The system integrates with Selenium WebDriver for automated browser testing with hardware acceleration detection and recovery capabilities. This integration includes:

1. **Selenium Browser Bridge**: A bridge between Python and Selenium WebDriver with recovery capabilities
2. **Model-Aware Browser Configuration**: Automatically configures browsers for different model types
3. **Recovery Strategies**: Sophisticated recovery strategies for different failure scenarios
4. **Circuit Breaker Pattern**: Fault tolerance with circuit breaker protection
5. **Comprehensive Test Suite**: Systematic testing of browser and model combinations

### Running the Comprehensive Selenium Test Suite

```bash
# Run a quick test with Chrome only
./distributed_testing/run_selenium_integration_tests.sh --quick

# Run the full test suite with all browsers and models
./distributed_testing/run_selenium_integration_tests.sh --full

# Test Firefox with audio models (optimal for audio processing)
./distributed_testing/run_selenium_integration_tests.sh --firefox-only --audio-only

# Test Edge with text models and WebNN platform (optimal for text models)
./distributed_testing/run_selenium_integration_tests.sh --edge-only --text-only --webnn-only

# Run tests without failure injection
./distributed_testing/run_selenium_integration_tests.sh --no-failures

# Run in simulation mode and save test report
./distributed_testing/run_selenium_integration_tests.sh --simulate --save-report
```

For more information, see the [Selenium Integration README](distributed_testing/SELENIUM_INTEGRATION_README.md).

## Hardware Acceleration Detection

The system detects hardware acceleration by:

1. Checking if WebNN/WebGPU APIs are available in the browser
2. For WebGPU, examining the adapter vendor and device information to distinguish between real hardware and simulation
3. For WebNN, checking if the context type is not "CPU" (which indicates simulation)

### Detection Results

The validation system will categorize a browser into one of three states:

- **HARDWARE**: Real hardware acceleration is available (✅)
- **SIMULATION**: The browser supports the API but is using software simulation (⚠️)
- **NOT SUPPORTED**: The browser doesn't support the API (❌)

## Browser-Specific Recommendations

Based on testing and the implementation in the IPFS Accelerate project, we recommend:

### For TEXT Models (BERT, T5, etc.)
- **First Choice**: Edge with WebNN - Best WebNN implementation
- **Second Choice**: Chrome with WebNN or WebGPU
- **Third Choice**: Any browser with WebGPU support

### For VISION Models (ViT, CLIP, etc.)
- **First Choice**: Chrome with WebGPU - Best general WebGPU support
- **Second Choice**: Edge with WebGPU - Good performance for vision models
- **Third Choice**: Firefox with WebGPU - Acceptable for vision models

### For AUDIO Models (Whisper, Wav2Vec2, CLAP)
- **First Choice**: Firefox with WebGPU - Best compute shader support (20-25% faster for audio models)
- **Second Choice**: Chrome with WebGPU - Good general performance
- **Third Choice**: Edge with WebGPU - Acceptable for audio models

## Dashboard

The browser environment validation system includes an interactive dashboard that visualizes test results. The dashboard is automatically generated when running the GitHub Actions workflow and is deployed to GitHub Pages.

The dashboard includes:
- Browser capabilities overview
- Browser support status
- Platform support status
- Recommendations for different model types
- Detailed test results

## Implementation Details

The browser environment validation system consists of several integrated components:

### Core Files

1. **fixed_web_platform/browser_automation.py**
   - Core functionality for browser automation and capability detection
   - `BrowserAutomation` class for controlling browsers via Selenium or subprocess
   - Functions for detecting browser executables and setting appropriate flags
   - Support for testing enhanced features (compute shaders, shader precompilation, parallel loading)

2. **check_browser_capabilities.py**
   - Basic browser capability detection for WebNN and WebGPU
   - Simple command-line tool for checking a single browser's capabilities
   - Focus on basic availability checking without enhanced feature testing

3. **check_browser_webnn_webgpu.py**
   - Enhanced WebNN/WebGPU capability detection with comprehensive reporting
   - Support for testing all installed browsers (Chrome, Firefox, Edge, Safari)
   - Detailed WebGPU adapter and WebNN backend information
   - Hardware vs. simulation detection with comprehensive reporting
   - Optimized for CI/CD integration

4. **distributed_testing/selenium_browser_bridge.py**
   - Bridge between Python and Selenium WebDriver
   - WebGPU/WebNN feature detection
   - Model-aware browser configuration
   - Recovery strategy integration
   - Circuit breaker pattern integration
   - Simulation fallback mode

5. **distributed_testing/test_selenium_browser_integration.py**
   - Comprehensive test suite for Selenium integration
   - Test case generation for browser and model combinations
   - Failure injection for testing recovery strategies
   - Metrics collection and reporting
   - Browser-specific optimization testing
   - Model-specific optimization testing

6. **distributed_testing/run_selenium_integration_tests.sh**
   - Bash script for running the comprehensive test suite
   - Support for specific browser and model selections
   - Test configuration with different platforms and features
   - Report generation and test statistics

7. **.github/workflows/browser_environment_validation.yml**
   - GitHub Actions workflow for automated testing
   - Support for testing specific browsers or all available browsers
   - Feature flag configuration for enhanced feature testing
   - Dashboard generation for visualization and reporting
   - Artifact storage for test results
   - GitHub Pages deployment for interactive dashboard

8. **test_browser_environment_validation.py**
   - Comprehensive unit tests for the validation system
   - Tests for browser detection, feature flag parsing, and capability reporting
   - Mocked browser testing for CI/CD environments
   - Validation of browser argument generation and HTML test file creation

### Architectural Design

The system follows a layered architecture:

1. **Browser Management Layer**
   - Browser detection and executable location
   - Browser process management
   - Browser arguments and flags

2. **Testing Layer**
   - HTML test file generation
   - WebNN/WebGPU capability detection
   - Feature testing for enhanced capabilities
   - Hardware vs. simulation detection

3. **Reporting Layer**
   - Comprehensive result formatting
   - HTML, Markdown, and JSON report generation
   - Model-specific recommendations
   - Browser-specific optimization suggestions

4. **Integration Layer**
   - CI/CD workflow integration
   - Dashboard generation
   - GitHub Pages deployment
   - Artifact storage and retrieval

### Recent Improvements (March 2025)

1. **Selenium Integration with Recovery**
   - Comprehensive integration of Selenium WebDriver with browser recovery strategies
   - Model-aware browser configuration for different model types
   - Circuit breaker pattern integration for fault tolerance
   - Simulation fallback mode for environments without Selenium
   - Progressive recovery strategies with increasing levels of intervention
   - Comprehensive test suite for browser and model combinations
   - Failure injection for testing recovery strategies
   - Detailed metrics collection and reporting

2. **Compute Shader Optimization**
   - Firefox now has 20-25% better performance for audio models with compute shader optimization
   - Custom workgroup sizes (256x1x1 for Firefox vs. Chrome's 128x2x1)
   - Specialized audio processing optimizations
   - Integrated with WebGPU backend for Whisper, Wav2Vec2, and CLAP models

3. **Shader Precompilation**
   - Reduces model startup time by 30-45% for all WebGPU models
   - Precompiles shaders during initialization phase
   - Browser-specific shader caching strategies
   - Particularly effective for large vision models

4. **Parallel Model Loading**
   - Reduces loading time by 30-45% for multimodal models
   - Loads multiple model components concurrently
   - Optimized for models with separate encoders (vision, text)
   - Resource-aware loading to prevent browser crashes

5. **Browser-Specific Optimizations**
   - Firefox optimizations for audio models
   - Chrome optimizations for vision models
   - Edge optimizations for WebNN models
   - Safari fallback strategies
   - Browser detection with specialized configuration

6. **Feature Detection**
   - Detailed WebGPU adapter capabilities reporting
   - WebGPU features and limits inspection
   - WebNN backend detection
   - Support for checking compute shader availability
   - WebGPU limits validation for specific model requirements

7. **CI/CD Integration**
   - Automated browser testing in GitHub Actions
   - Interactive dashboard generation
   - Historical tracking of browser capabilities
   - Feature flag testing
   - Comprehensive reporting

## Troubleshooting

### Browser Not Found

If a browser is not detected automatically:

1. Make sure the browser is installed
2. Check if it's in a standard installation location
3. Try specifying the full path to the browser executable:
   ```bash
   BROWSER_PATH=/path/to/chrome python check_browser_webnn_webgpu.py --browser chrome
   ```

### Hardware Acceleration Not Detected

If hardware acceleration is not detected:

1. Make sure your GPU drivers are up to date
2. Check if your browser is configured to use hardware acceleration
3. In Chrome, go to `chrome://gpu` to check WebGPU status
4. In Firefox, go to `about:config` and make sure `dom.webgpu.enabled` is set to `true`
5. In Edge, check `edge://gpu` for WebGPU status

### Selenium Issues

If you encounter issues with Selenium:

1. Make sure Selenium is installed: `pip install selenium`
2. Install the appropriate webdriver for your browser
3. Consider using `webdriver-manager` for automatic webdriver installation

## Future Improvements

The browser environment validation system is continuously evolving. Our roadmap for future improvements includes:

### Short-term Improvements (Q3 2025)

1. **Enhanced Safari Support**
   - Better hardware acceleration detection for Safari on macOS
   - WebGPU feature detection for Safari 18+
   - Safari-specific optimizations for various model types
   - Metal-specific performance optimizations

2. **Mobile Browser Testing**
   - Support for Android Chrome, Firefox, and Samsung Internet
   - Support for iOS Safari
   - Mobile-specific hardware acceleration detection
   - Battery impact and thermal monitoring for mobile devices

3. **Comprehensive Performance Benchmarking**
   - Standardized benchmarks for WebNN and WebGPU across browsers
   - Performance comparison with native implementations
   - Model-specific benchmarks for text, vision, and audio models
   - Detailed metrics for latency, throughput, and memory usage

### Medium-term Improvements (Q4 2025)

4. **WebGPU Feature Testing**
   - Testing for specific WebGPU features and extensions
   - Detailed reporting on WebGPU limits and capabilities
   - Testing for browser-specific WebGPU implementations
   - Compute shader capability analysis
   - Feature compatibility matrix for advanced WebGPU features

5. **Enhanced WebNN Backend Detection**
   - Detailed reporting on WebNN backends (CPU, GPU, NPU)
   - WebNN operator support testing
   - Browser-specific WebNN optimizations
   - WebNN hardware acceleration verification

6. **CI/CD Results Database**
   - Historical tracking of browser capabilities over time
   - Regression testing for WebNN/WebGPU support
   - Trend analysis for browser capabilities
   - Automated alerts for browser support changes

### Long-term Vision (2026+)

7. **Real-time Performance Monitoring**
   - Real-time monitoring of browser performance for WebNN/WebGPU
   - Integration with application monitoring systems
   - Performance degradation detection
   - Automatic fallback strategies

8. **Cross-Browser Model Sharding**
   - Distributed model execution across multiple browsers
   - Browser-specific component optimization
   - Load balancing based on browser capabilities
   - Fault tolerance and recovery

9. **Browser Simulation Testing**
   - Testing with simulated browser environments
   - Compatibility testing for future browser versions
   - Regression testing for browser updates
   - Automated compatibility checking for new models

10. **Automated Optimization Recommendations**
    - AI-driven recommendations for browser selection
    - Automated configuration for optimal performance
    - Model-specific optimization suggestions
    - Environment-aware performance tuning

## Integration with Distributed Testing Framework

The browser environment validation system is designed to integrate with the Distributed Testing Framework (currently at 40% completion). Once both systems are complete, you'll be able to:

1. **Distributed Browser Testing**: Run browser validation tests across multiple machines
2. **Hardware-Aware Testing**: Automatically select appropriate hardware for browser tests
3. **Cross-Platform Validation**: Test browser capabilities across different operating systems
4. **Scalable Testing**: Scale browser testing to hundreds of browser instances
5. **Centralized Reporting**: Unified reporting for all browser validation results

## Conclusion

The browser environment validation system is a comprehensive solution for detecting and validating WebNN and WebGPU capabilities across different browsers. It helps you understand which browsers and platforms provide the best support for your AI models and ensures that your applications can make informed decisions about hardware acceleration.

By using this system, you can:
- Choose the optimal browser for each model type
- Detect real hardware acceleration vs. simulation
- Test enhanced features like compute shaders and shader precompilation
- Automate browser validation in CI/CD pipelines
- Visualize browser capabilities with interactive dashboards

This validation system is a critical component of the IPFS Accelerate project, enabling efficient deployment of AI models across web browsers with hardware acceleration.

---

**Documentation Version**: 1.0.0 (June 2025)  
**Last Updated**: June 14, 2025  
**Contributors**: IPFS Accelerate Team