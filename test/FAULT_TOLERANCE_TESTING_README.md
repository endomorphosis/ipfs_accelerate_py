# Fault Tolerance Testing Guide

This guide explains how to run fault tolerance tests with real browsers to validate the cross-browser model sharding recovery mechanisms.

## Prerequisites

1. Ensure you have the required dependencies installed by running:
   ```
   ./install_fault_tolerance_test_deps.sh
   ```

2. Make sure you have at least one of the following browsers installed:
   - Google Chrome
   - Mozilla Firefox
   - Microsoft Edge

3. For comprehensive testing, selenium and webdriver-manager are required:
   ```
   pip install selenium webdriver-manager
   ```

## Running Tests

### Basic Test (Single Browser Type)

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome
```

### Testing with Multiple Browser Types

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome,firefox,edge
```

### Testing Fault Tolerance with Forced Failures

```bash
# Force a random browser failure during testing
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure

# Specify a failure type
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --failure-type crash

# Available failure types:
# - crash: Force browser to crash
# - hang: Force browser to hang (infinite loop)
# - memory: Force memory pressure by allocating large arrays
# - disconnect: Simulate network disconnection
```

### Testing with Different Recovery Strategies

```bash
# Test progressive recovery (tries simple strategies first, then more complex ones)
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --recovery-strategy progressive

# Other recovery strategies:
# - restart: Restart the failed browser
# - reconnect: Attempt to reconnect to the browser
# - failover: Switch to another browser
# - parallel: Try multiple strategies in parallel
```

### Testing Different Model Types

```bash
# Test text models
python test_real_browser_fault_tolerance.py --model bert-base-uncased --model-type text

# Test audio models (Firefox preferred for compute shader support)
python test_real_browser_fault_tolerance.py --model whisper-tiny --model-type audio

# Test vision models
python test_real_browser_fault_tolerance.py --model vit-base-patch16-224 --model-type vision

# Test multimodal models
python test_real_browser_fault_tolerance.py --model clip-vit-base-patch32 --model-type multimodal
```

### Performance Benchmarking After Recovery

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --benchmark --iterations 20
```

### Debugging

To see the browser windows during testing (disable headless mode):

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --show-browsers
```

For more detailed logs:

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --verbose
```

### Comprehensive Tests

For complete testing including browser failure, recovery, and performance benchmarking:

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome,firefox,edge --force-failure --benchmark --iterations 10 --output test_results.json
```

## Test Results

If an output file is specified with `--output`, test results will be saved in JSON format. Results include:

- Basic test information (model, browsers, configuration)
- Phases (each phase with status, duration, and results)
- For failure tests, detailed recovery information
- For benchmark tests, performance statistics
- Detailed execution timing and browser capabilities

## Troubleshooting

- If browser drivers are not found, you may need to install them manually or ensure webdriver-manager is working correctly
- On Linux, you may need to install additional dependencies for headless browser testing
- If WebGPU/WebNN are not detected, ensure your browsers are up-to-date
- For Firefox compute shader support issues, set MOZ_WEBGPU_ADVANCED_COMPUTE=1 environment variable