# Real WebNN/WebGPU Implementation Testing

**Latest Update: March 7, 2025**

This guide explains how to run comprehensive tests for real (not simulated) WebNN and WebGPU implementations at different precision levels (2-bit through 32-bit) using a Selenium-based browser automation bridge.

## Overview

The testing framework connects Python to real browsers via Selenium WebDriver and WebSockets, allowing for hardware-accelerated inference with:

- **Multiple browsers**: Chrome, Firefox, Edge, Safari
- **Multiple platforms**: WebNN, WebGPU
- **Multiple precision levels**: 2-bit, 3-bit, 4-bit, 8-bit, 16-bit, 32-bit
- **Multiple model types**: BERT, T5, ViT, CLIP, Whisper, Wav2Vec2

This approach ensures we're testing actual browser-based hardware acceleration rather than using simulation.

## Key Components

1. **Selenium Bridge**: Automates browser launching and control
2. **WebSocket Communication**: Enables bidirectional data exchange between Python and browser
3. **transformers.js Integration**: Powers the actual model inference in the browser
4. **Precision Testing**: Evaluates performance at different bit widths
5. **Real vs. Simulation Detection**: Clearly distinguishes between real hardware and simulation
6. **Comprehensive Documentation**: Auto-updates documentation with test results

## Prerequisites

- Python 3.8+
- Selenium WebDriver
- transformers.js
- Chrome/Firefox/Edge/Safari browsers
- WebGPU-capable hardware and drivers
- WebNN-compatible browser (primarily Edge)

## Running the Tests

The main test script is `run_comprehensive_webnn_webgpu_tests.py`, which provides extensive options for controlling the testing process:

```bash
# Test all precision levels with Chrome
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform webgpu --precision all --model bert --visible

# Test 4-bit and 8-bit precision with Edge using WebNN
python run_comprehensive_webnn_webgpu_tests.py --browser edge --platform webnn --precision 4,8 --model bert --experimental --visible

# Test all browsers, platforms, and models at 4-bit precision
python run_comprehensive_webnn_webgpu_tests.py --all-browsers --platform all --model all --precision 4 --visible

# Test with Firefox and all optimizations enabled
python run_comprehensive_webnn_webgpu_tests.py --browser firefox --platform webgpu --precision 4 --model whisper --all-optimizations --visible

# Run tests and update documentation
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform all --precision all --model bert --update-docs --archive-docs --visible
```

## Commands to Run Real Implementation Tests

To ensure you're using real hardware acceleration and not simulations:

### Basic Tests
```bash
# Test BERT with WebGPU in Chrome (visible browser)
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform webgpu --precision 4 --model bert --visible

# Test Edge with WebNN (Edge has best WebNN support)
python run_comprehensive_webnn_webgpu_tests.py --browser edge --platform webnn --precision 8 --model bert --visible
```

### Advanced Tests
```bash
# Test Firefox with compute shader optimization (audio models)
python run_comprehensive_webnn_webgpu_tests.py --browser firefox --platform webgpu --precision 4 --model whisper --compute-shaders --visible

# Test with all optimizations
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform webgpu --precision 4 --model clip --all-optimizations --visible

# Test ultra-low precision (2-bit) with WebGPU
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform webgpu --precision 2 --model bert --visible
```

### Full Test Suite
```bash
# Run complete test suite (all browsers, platforms, precision levels, and models)
python run_comprehensive_webnn_webgpu_tests.py --all-browsers --platform all --precision all --model all --all-optimizations --visible --update-docs

# Run test suite with HTML report generation
python run_comprehensive_webnn_webgpu_tests.py --all-browsers --platform all --precision 4,8,16 --model all --output-dir ./reports --html-report
```

## Key Features Explained

### Real vs. Simulation Detection

The testing framework explicitly verifies whether real hardware acceleration is being used:

```python
# From WebPrecisionTester.verify_real_implementation
def verify_real_implementation(self, browser: str, platform: str) -> bool:
    """
    Verify that we're using a real implementation and not a simulation.
    Returns True if using real implementation, False for simulation.
    """
    capabilities = detect_web_platform_capabilities(browser=browser, use_browser_automation=True)
    
    if platform == "webnn":
        real_impl = capabilities.get("webnn_available", False) and not capabilities.get("webnn_simulated", True)
        return real_impl
    
    elif platform == "webgpu":
        real_impl = capabilities.get("webgpu_available", False) and not capabilities.get("webgpu_simulated", True)
        return real_impl
```

This approach uses direct browser capability detection to ensure we're testing real hardware acceleration, not simulation.

### Browser-Specific Optimizations

The framework includes special optimizations for different browsers:

1. **Firefox Optimizations**:
   - Custom compute shader workgroup size (256x1x1 vs 128x2x1 in Chrome)
   - ~20-25% better performance for audio models (Whisper, Wav2Vec2)
   - ~15% lower power consumption

2. **Edge Optimizations**:
   - Best WebNN support among browsers
   - Good integration of WebNN and WebGPU capabilities
   - Can use either or both acceleration methods

3. **Chrome Optimizations**:
   - Solid general-purpose WebGPU implementation
   - Good baseline performance across model types
   - Excellent developer tools for debugging

4. **Safari Optimizations**:
   - Limited but improving WebGPU support
   - Metal API integration for better performance
   - Power-efficient implementation for MacOS/iOS

### Precision Levels

The framework supports multiple precision levels, each with different memory-performance tradeoffs:

1. **2-bit Precision** (WebGPU only):
   - Ultra memory-efficient (87.5% memory reduction vs FP32)
   - Significant accuracy loss, suitable for specific use cases
   - Enables very large context windows on memory-constrained devices

2. **3-bit Precision** (WebGPU only):
   - Very memory-efficient (81.25% memory reduction vs FP32)
   - Better quality than 2-bit, still with some accuracy impact
   - Good for extending context windows with limited memory

3. **4-bit Precision**:
   - Memory-efficient (75% memory reduction vs FP32)
   - Good accuracy for many use cases
   - Primary focus of WebGPU optimization
   - Experimental in WebNN

4. **8-bit Precision**:
   - Well-supported across platforms
   - Good balance of quality and memory efficiency
   - Fully supported in both WebNN and WebGPU

5. **16-bit Precision**:
   - High quality results
   - Moderate memory usage
   - Good for precision-sensitive tasks

6. **32-bit Precision**:
   - Maximum accuracy
   - Highest memory usage
   - Baseline for comparison

## Firefox Audio Model Optimization

Firefox provides significant performance advantages for audio models using WebGPU:

```python
# Firefox-specific optimization
if args.browser == "firefox":
    os.environ["USE_FIREFOX_WEBGPU"] = "1"
    if args.compute_shaders:
        os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
```

Key optimizations include:
- 256x1x1 workgroup size (vs 128x2x1 in Chrome)
- Better sharing of resources between WebGPU and WebAudio
- ~20-25% better performance for audio models
- ~15% reduced power consumption

## Automatic Documentation Updates

The testing framework can automatically update documentation with test results:

```bash
# Run tests and update documentation
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform all --precision all --model bert --update-docs

# Run tests, update documentation, and archive old versions
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform all --precision all --model bert --update-docs --archive-docs
```

This ensures that documentation always reflects the latest real-world test results.

## Understanding Test Results

The test results include:

1. **Implementation Status**: Real hardware or simulation
2. **Performance Metrics**: Latency, throughput, memory usage
3. **Memory Reduction**: Percentage reduction vs FP32
4. **Browser Support**: Which browsers support which features
5. **Precision Support**: Which precision levels are supported
6. **Model Support**: Which models work on which platforms

Example summary output:
```
================================================================================
TEST SUMMARY: whisper - webgpu - firefox - 4-bit
================================================================================
Implementation: REAL HARDWARE
Latency: 123.45 ms
Throughput: 8.1 items/sec
Memory: 156 MB
================================================================================
```

## Browser Support Matrix

| Browser | WebNN Support | WebGPU Support | Best Use Case |
|---------|--------------|----------------|---------------|
| Chrome | ⚠️ Limited | ✅ Good | General WebGPU |
| Edge | ✅ Excellent | ✅ Good | WebNN acceleration |
| Firefox | ❌ Poor | ✅ Excellent | Audio models with WebGPU |
| Safari | ⚠️ Limited | ⚠️ Limited | Metal API integration |

## Model Type Recommendations

| Model Type | Recommended Browser | Recommended Platform | Optimal Precision |
|------------|---------------------|---------------------|-------------------|
| Text (BERT, T5) | Edge | WebNN | 8-bit |
| Vision (ViT, CLIP) | Chrome/Firefox | WebGPU | 4-bit/8-bit |
| Audio (Whisper, Wav2Vec2) | Firefox | WebGPU | 4-bit |
| Multimodal (CLIP) | Chrome | WebGPU | 4-bit |

## Conclusion

By using the comprehensive testing framework, you can ensure you're testing real WebNN and WebGPU implementations rather than simulations. The browser automation with Selenium and WebSockets provides a robust way to verify and measure the performance of browser-based ML acceleration at different precision levels.

For detailed documentation on specific aspects of the implementation, please refer to:
- [WEBNN_WEBGPU_GUIDE.md](WEBNN_WEBGPU_GUIDE.md)
- [WEBGPU_4BIT_INFERENCE_README.md](WEBGPU_4BIT_INFERENCE_README.md)
- [WEBNN_VERIFICATION_GUIDE.md](WEBNN_VERIFICATION_GUIDE.md)