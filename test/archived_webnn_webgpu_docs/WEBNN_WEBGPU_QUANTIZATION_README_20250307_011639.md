# WebNN and WebGPU Quantization Testing

This document provides a quick reference for using the WebNN and WebGPU quantization testing tools.

## Overview

These tools allow you to test and benchmark quantization capabilities in WebNN and WebGPU implementations:

- **WebGPU** supports 2-bit, 4-bit, 8-bit and 16-bit precision
- **WebNN** supports 8-bit and 16-bit precision (automatically upgrades from lower bits)
- All tools support mixed precision configurations for optimal accuracy/performance balance

## Quick Start

### Simplified Test (Quickest Option)

```bash
# Test WebGPU with 4-bit quantization on Chrome
python test_webnn_webgpu_simplified.py --platform webgpu --bits 4 --browser chrome

# Test WebNN with 8-bit quantization on Edge
python test_webnn_webgpu_simplified.py --platform webnn --bits 8 --browser edge

# Test WebGPU with ultra-low precision (2-bit)
python test_webnn_webgpu_simplified.py --platform webgpu --bits 2 --browser firefox

# Test WebNN with 4-bit request (will automatically upgrade to 8-bit)
python test_webnn_webgpu_simplified.py --platform webnn --bits 4 --browser chrome

# Test both platforms with default settings
python test_webnn_webgpu_simplified.py --platform both

# Check result status with emojis (✅/❌)
```

### Shell Script for Comprehensive Testing

```bash
# Run the complete test suite
./run_webnn_webgpu_quantization.sh --all --chrome --firefox

# Test WebGPU with Chrome and Firefox
./run_webnn_webgpu_quantization.sh --webgpu-only --chrome --firefox

# Test WebNN with Edge
./run_webnn_webgpu_quantization.sh --webnn-only --edge

# Test with mixed precision
./run_webnn_webgpu_quantization.sh --all --mixed-precision

# Test with ultra-low precision (2-bit)
./run_webnn_webgpu_quantization.sh --webgpu-only --chrome --ultra-low-prec

# Test WebNN with 4-bit request (will demonstrate 8-bit fallback)
./run_webnn_webgpu_quantization.sh --webnn-only --chrome --ultra-low-prec

# Run in headless mode
./run_webnn_webgpu_quantization.sh --all --headless

# Test with specific model
./run_webnn_webgpu_quantization.sh --model whisper-tiny --firefox
```

### Simplified Test Configuration

```bash
# Test WebGPU with Chrome and 4-bit quantization
python test_webnn_webgpu_simplified.py --platform webgpu --browser chrome --model bert-base-uncased --bits 4

# Test WebNN with Edge and 8-bit quantization
python test_webnn_webgpu_simplified.py --platform webnn --browser edge --model bert-base-uncased --bits 8

# Test with 2-bit ultra-low precision
python test_webnn_webgpu_simplified.py --platform webgpu --browser chrome --model bert-base-uncased --bits 2

# Enable mixed precision
python test_webnn_webgpu_simplified.py --platform webgpu --browser chrome --mixed-precision

# Run browser in headless mode
python test_webnn_webgpu_simplified.py --platform webgpu --browser chrome --model bert-base-uncased --bits 4 --headless
```

## Required Environment

- Python 3.8 or higher
- Modern browser with WebGPU/WebNN support (Chrome, Firefox, Edge, Safari)
- Python dependencies: `websockets`, `selenium`, `webdriver-manager`

## Install Required Drivers

```bash
# Install WebDrivers for browsers
python test_webnn_webgpu_integration.py --install-drivers
```

## Example Output

```
=== Testing WebGPU with Chrome ===
Testing 4-bit quantization...
2025-03-07 10:15:23 - INFO - Initializing WebGPU implementation
2025-03-07 10:15:25 - INFO - WebGPU implementation initialized in 1574.29 ms
2025-03-07 10:15:25 - INFO - Initializing model bert-base-uncased with 4-bit quantization
2025-03-07 10:15:29 - INFO - Model initialized in 3721.45 ms
2025-03-07 10:15:29 - INFO - Running inference with 4-bit quantization
2025-03-07 10:15:30 - INFO - Inference completed in 974.82 ms (avg: 974.82 ms)
2025-03-07 10:15:30 - INFO - Memory usage: 48.3MB
2025-03-07 10:15:30 - INFO - Memory reduction: 75.2%

Testing 8-bit quantization...
...

✅ WebGPU 4-bit quantization test passed
✅ WebGPU 8-bit quantization test passed
```

## Further Resources

- [WebNN WebGPU Usage Guide](WEBNN_WEBGPU_USAGE_GUIDE.md): Comprehensive implementation guide
- [WebNN WebGPU Guide](WEBNN_WEBGPU_GUIDE.md): General WebNN and WebGPU documentation 
- [WebGPU WebNN Quantization Summary](WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md): Detailed information on quantization techniques and performance results

## Troubleshooting

- **Browser crashes**: Try running with `--headless` flag
- **WebNN unavailable**: Make sure you're using Chrome 122+ or Edge for WebNN support 
- **Out of memory errors**: Try using lower bit precision (4-bit or 2-bit)
- **WebDriver errors**: Run `python test_webnn_webgpu_integration.py --install-drivers` to update
- **WebNN with 4-bit/2-bit**: Now attempts experimental support for low precision without fallback
- **Firefox with WebNN**: Firefox doesn't support WebNN, use WebGPU instead
- **Safari with 2-bit**: Safari has limited support for 2-bit precision, use 4-bit or 8-bit instead

### [MARCH 2025] Experimental Low Precision Support for WebNN

WebNN officially supports 8-bit quantization. However, we've enhanced our implementation with two modes:

#### 1. Standard Mode (Default)
- Automatically upgrades 4-bit/2-bit requests to 8-bit
- Provides silent fallback behavior for maximum compatibility
- No precision-related errors, ensuring robust operation

#### 2. Experimental Mode (New)
- Attempts to use the requested precision (4-bit, 2-bit) without automatic fallback
- Reports actual errors from the browser for debugging purposes
- Provides detailed error information instead of silently upgrading to 8-bit

```bash
# Standard mode - auto-upgrades to 8-bit with no errors
python test_webnn_webgpu_simplified.py --platform webnn --bits 4 --browser chrome

# Experimental mode - attempts true 4-bit with error reporting
python test_webnn_webgpu_simplified.py --platform webnn --bits 4 --browser chrome --experimental-precision

# Experimental mode - attempts true 2-bit with error reporting
python test_webnn_webgpu_simplified.py --platform webnn --bits 2 --browser edge --experimental-precision
```

This enhancement gives developers a choice between robust fallback behavior (standard mode) and detailed error reporting (experimental mode) for research and debugging purposes.

```python
# Programmatic usage of experimental mode:
os.environ["WEBNN_EXPERIMENTAL_PRECISION"] = "1"  # Enable experimental mode
webnn_impl = RealWebNNImplementation(browser_name="edge")
await webnn_impl.initialize()
await webnn_impl.run_inference(
    "bert-base-uncased", 
    "This is a test", 
    {"use_quantization": True, "bits": 4, "mixed_precision": False}
)
```

Experimental mode is especially useful for:
- Debugging WebNN precision limitations
- Understanding browser-specific error messages
- Testing advanced quantization approaches
- Developing browser-specific optimizations

## Complete Parameter List

### test_webnn_webgpu_simplified.py

```
--platform [webgpu|webnn|both] : Which platform(s) to test
--browser [chrome|firefox|edge|safari] : Which browser to use
--model [model_name] : Which model to test (default: bert-base-uncased)
--bits [2|4|8|16] : Bit precision for quantization (default: 4 for WebGPU, 8 for WebNN)
--mixed-precision : Enable mixed precision testing
--experimental-precision : Try using lower precision with WebNN (may fail with errors)
```

Note: With the new `--experimental-precision` flag, WebNN will attempt to use the requested precision (even 2-bit or 4-bit) instead of automatically upgrading to 8-bit. This will likely fail but provides useful error information for development.

### run_webnn_webgpu_quantization.sh

```
--all : Test both WebGPU and WebNN
--webgpu-only : Test only WebGPU
--webnn-only : Test only WebNN
--chrome : Test with Chrome browser
--firefox : Test with Firefox browser
--edge : Test with Edge browser (Windows only)
--safari : Test with Safari browser (macOS only)
--model [model_name] : Which model to test (default: bert-base-uncased)
--mixed-precision : Enable mixed precision testing
--ultra-low-prec : Enable ultra-low precision (2-bit) testing for WebGPU; lower precision (4-bit/2-bit) for WebNN
--experimental : Try experimental precision with WebNN (may fail with errors, reports detailed error messages)
--headless : Run browser in headless mode
--help : Display help message
```

The script automatically uses appropriate bit-widths for each platform:
- WebGPU: Tests 4-bit and 8-bit by default, adds 2-bit when --ultra-low-prec is used
- WebNN: Tests 8-bit by default
  - With --ultra-low-prec: Tests 4-bit in addition to 8-bit
  - With --experimental: Attempts to use requested precision without fallback, reporting actual errors
  - With both flags: Attempts true 4-bit and 2-bit quantization with WebNN, showing detailed error messages