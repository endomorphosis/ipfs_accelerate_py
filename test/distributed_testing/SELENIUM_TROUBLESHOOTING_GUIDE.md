# Selenium Integration Troubleshooting Guide

This guide provides solutions for common issues encountered when using the Selenium integration with browser recovery strategies. It covers setup issues, runtime problems, and recommendations for optimal testing configurations.

## Table of Contents

1. [Setup Issues](#setup-issues)
   - [Selenium Installation Problems](#selenium-installation-problems)
   - [WebDriver Installation Problems](#webdriver-installation-problems)
   - [Browser Detection Issues](#browser-detection-issues)
   
2. [Runtime Issues](#runtime-issues)
   - [Browser Launch Failures](#browser-launch-failures)
   - [WebGPU/WebNN Detection Problems](#webgpuwebnn-detection-problems)
   - [Test Failure Patterns](#test-failure-patterns)
   - [Recovery Strategy Failures](#recovery-strategy-failures)
   
3. [Browser-Specific Issues](#browser-specific-issues)
   - [Chrome Issues](#chrome-issues)
   - [Firefox Issues](#firefox-issues)
   - [Edge Issues](#edge-issues)
   
4. [Model-Specific Issues](#model-specific-issues)
   - [Text Model Issues](#text-model-issues)
   - [Vision Model Issues](#vision-model-issues)
   - [Audio Model Issues](#audio-model-issues)
   - [Multimodal Model Issues](#multimodal-model-issues)
   
5. [Performance Optimization](#performance-optimization)
   - [Reducing Test Time](#reducing-test-time)
   - [Improving Recovery Success Rate](#improving-recovery-success-rate)
   - [Memory Usage Optimization](#memory-usage-optimization)
   
6. [Advanced Diagnostics](#advanced-diagnostics)
   - [Enabling Verbose Logging](#enabling-verbose-logging)
   - [Manual Browser Testing](#manual-browser-testing)
   - [Circuit Breaker Analysis](#circuit-breaker-analysis)
   - [Recovery Strategy Analysis](#recovery-strategy-analysis)

## Setup Issues

### Selenium Installation Problems

**Problem**: Unable to install Selenium with pip.

**Solutions**:
1. Ensure you have Python 3.7+ installed: `python --version`
2. Update pip: `pip install --upgrade pip`
3. Install with specific version: `pip install selenium==4.9.0`
4. Check for permission issues: `pip install selenium --user`
5. If behind a proxy or firewall, configure pip accordingly.

**Problem**: ImportError when trying to import Selenium.

**Solutions**:
1. Ensure you've installed Selenium in the correct Python environment.
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Reinstall Selenium: `pip uninstall selenium && pip install selenium`

### WebDriver Installation Problems

**Problem**: WebDriver not found or couldn't be installed.

**Solutions**:
1. Install webdriver-manager: `pip install webdriver-manager`
2. Manually download the appropriate WebDriver:
   - ChromeDriver: [Chrome WebDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)
   - GeckoDriver: [Firefox WebDriver](https://github.com/mozilla/geckodriver/releases)
   - EdgeDriver: [Edge WebDriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)
3. Add WebDriver to your PATH.
4. Verify WebDriver version matches browser version.

**Problem**: WebDriver version compatibility issues.

**Solutions**:
1. Check browser version: 
   - Chrome: Navigate to `chrome://version`
   - Firefox: Navigate to `about:support`
   - Edge: Navigate to `edge://version`
2. Download the matching WebDriver version.
3. Use the `--browser-version` flag when installing with webdriver-manager.

### Browser Detection Issues

**Problem**: Script doesn't detect installed browsers.

**Solutions**:
1. Ensure browsers are installed in standard locations.
2. Add browser locations to your PATH.
3. Specify browser path directly: `BROWSER_PATH=/path/to/chrome ./setup_and_run_selenium_tests.sh`
4. Run with the `--simulate` flag if no browsers are installed.

## Runtime Issues

### Browser Launch Failures

**Problem**: Browser fails to launch with Selenium.

**Solutions**:
1. Check WebDriver compatibility with browser version.
2. Increase timeout: Set the `--test-timeout` argument to a higher value.
3. Check for existing browser processes that might interfere.
4. Run in headless mode by setting the `headless` option to `True`.
5. Check system resources (memory, disk space).
6. Check for browser crash logs in:
   - Chrome: `~/.config/google-chrome/Crash Reports/`
   - Firefox: `~/.mozilla/firefox/*.default/minidumps/`
   - Edge: `~/.config/microsoft-edge/Crash Reports/`

**Problem**: Permission or sandbox issues when launching browsers.

**Solutions**:
1. Run with appropriate permissions.
2. For Chrome/Edge, add `--no-sandbox` to browser arguments:
   ```python
   bridge.add_browser_arg("--no-sandbox")
   ```
3. Check system security policies (SELinux, AppArmor).

### WebGPU/WebNN Detection Problems

**Problem**: WebGPU/WebNN not detected despite browser support.

**Solutions**:
1. Ensure browser version supports the technology:
   - Chrome/Edge 113+ for WebGPU 
   - Edge 113+ for WebNN
   - Firefox 113+ for WebGPU
2. Enable experimental features:
   - Chrome/Edge: Navigate to `chrome://flags` or `edge://flags` and enable WebGPU
   - Firefox: Set `dom.webgpu.enabled` to `true` in `about:config`
3. Check for hardware acceleration support:
   - Chrome/Edge: Navigate to `chrome://gpu` or `edge://gpu`
   - Firefox: Navigate to `about:support` and check "Graphics" section
4. For WebGPU, ensure you have updated graphics drivers.

**Problem**: System reports WebGPU/WebNN is available, but tests show simulation mode.

**Solutions**:
1. The browser may support the API but is using software simulation rather than hardware acceleration.
2. Check GPU status in browser's internal pages.
3. Update graphics drivers.
4. Enable hardware acceleration in browser settings.
5. Disable other programs using the GPU.

### Test Failure Patterns

**Problem**: Tests always fail with the same error.

**Solutions**:
1. Check the error pattern and look for commonalities.
2. Enable verbose logging to get more details (see [Enabling Verbose Logging](#enabling-verbose-logging)).
3. Run with the `--no-failures` flag to avoid injecting artificial failures.
4. Check if recovery strategies are being triggered correctly.
5. If the error is browser-specific, try with a different browser.

**Problem**: Tests occasionally fail with random errors.

**Solutions**:
1. Increase retry count: Use the `--retry-count` argument.
2. Check for resource contention or system load.
3. Run tests individually rather than in batch.
4. Check for network connectivity issues if tests involve remote resources.
5. Monitor system resources during test execution.

### Recovery Strategy Failures

**Problem**: Recovery strategies failing to recover from failures.

**Solutions**:
1. Check the failure type to ensure appropriate recovery strategies are being used.
2. Ensure browser settings are appropriate for the model type.
3. Adjust timeouts for recovery operations.
4. Check for resource constraints (memory, CPU, GPU).
5. Enable verbose logging to diagnose specific recovery issues.
6. Try with a different browser that might better support the model type.

**Problem**: Circuit breaker keeps opening and not recovering.

**Solutions**:
1. Increase circuit breaker recovery timeout.
2. Adjust circuit breaker failure threshold.
3. Check for underlying systemic issues causing repeated failures.
4. Consider debugging with the recovery strategy disabled to identify the root cause.

## Browser-Specific Issues

### Chrome Issues

**Problem**: Chrome crashes during WebGPU operations.

**Solutions**:
1. Update Chrome to the latest version.
2. Update graphics drivers.
3. Add the following browser arguments:
   ```python
   bridge.add_browser_arg("--disable-gpu-watchdog")
   bridge.add_browser_arg("--disable-gpu-process-crash-limit")
   ```
4. Reduce WebGPU workload by adjusting batch size.
5. Check system memory usage and GPU memory usage.

**Problem**: Chrome shows poor performance with audio models.

**Solutions**:
1. For audio models, Firefox often performs better with compute shaders.
2. If using Chrome, adjust workgroup size:
   ```python
   bridge.add_browser_arg("--enable-dawn-features=compute_shaders")
   ```
3. Reduce audio model complexity or duration.

### Firefox Issues

**Problem**: Firefox WebGPU fails with "WebGPU not supported" error.

**Solutions**:
1. Ensure Firefox version is 113+ for WebGPU support.
2. Enable WebGPU in Firefox:
   - Navigate to `about:config`
   - Set `dom.webgpu.enabled` to `true`
3. For compute shader support, also set:
   - `dom.webgpu.advanced-compute` to `true`
4. Restart Firefox after changing settings.

**Problem**: Firefox shows excessive memory usage.

**Solutions**:
1. Limit memory usage by adjusting browser preferences:
   ```python
   bridge.add_browser_pref("browser.cache.memory.capacity", 32768)  # 32MB cache
   bridge.add_browser_pref("browser.sessionhistory.max_entries", 10)
   ```
2. Close Firefox between test runs.
3. Monitor memory usage and add cleanup steps if needed.

### Edge Issues

**Problem**: Edge cannot initialize WebNN context.

**Solutions**:
1. Ensure Edge version supports WebNN (113+).
2. Enable WebNN features:
   ```python
   bridge.add_browser_arg("--enable-features=WebNN,WebNNCompileOptions")
   bridge.add_browser_arg("--enable-dawn-features=enable_webnn_extension")
   ```
3. Check that Edge is using hardware acceleration:
   - Navigate to `edge://gpu`
   - Look for WebNN status
4. Update graphics drivers.

**Problem**: Edge launches but fails to execute scripts.

**Solutions**:
1. Check for conflicting Edge processes or extensions.
2. Run Edge with clean profile:
   ```python
   bridge.add_browser_arg("--user-data-dir=/tmp/edge-test-profile")
   ```
3. Disable Edge features that might interfere:
   ```python
   bridge.add_browser_arg("--disable-features=EdgeCollections")
   ```

## Model-Specific Issues

### Text Model Issues

**Problem**: Poor performance with text models across browsers.

**Solutions**:
1. For text models, Edge with WebNN often performs best:
   ```python
   # In configuration:
   bridge.set_platform("webnn")
   bridge.set_resource_settings(max_batch_size=1, optimize_for="latency")
   ```
2. Enable shader precompilation for faster startup:
   ```python
   bridge.set_shader_precompilation(True)
   ```
3. Ensure model sizes are appropriate for browser limitations.
4. Monitor memory usage and adjust batch size accordingly.

### Vision Model Issues

**Problem**: Vision models show inconsistent performance.

**Solutions**:
1. For vision models, Chrome with WebGPU often performs best:
   ```python
   # In configuration:
   bridge.set_platform("webgpu")
   bridge.set_resource_settings(max_batch_size=4, optimize_for="throughput")
   bridge.set_shader_precompilation(True)
   ```
2. For Chrome, add zero-copy optimization:
   ```python
   bridge.add_browser_arg("--enable-zero-copy")
   bridge.add_browser_arg("--enable-gpu-memory-buffer-video-frames")
   ```
3. Ensure input image dimensions are appropriate.
4. Monitor GPU memory usage and adjust batch size if needed.

### Audio Model Issues

**Problem**: Audio models perform poorly or crash.

**Solutions**:
1. For audio models, Firefox with compute shaders performs best:
   ```python
   # In configuration:
   bridge.set_platform("webgpu")
   bridge.set_compute_shaders(True)
   bridge.add_browser_pref("dom.webgpu.workgroup_size", "256,1,1")
   ```
2. Limit audio duration and complexity.
3. If using Firefox, ensure compute shaders are enabled.
4. For Chrome, adjust workgroup size:
   ```python
   bridge.add_browser_arg("--enable-dawn-features=compute_shaders")
   ```
5. Check system audio configuration if processing real audio.

### Multimodal Model Issues

**Problem**: Multimodal models exceed browser memory or crash.

**Solutions**:
1. Enable parallel loading for multimodal models:
   ```python
   bridge.set_parallel_loading(True)
   ```
2. Optimize for memory usage:
   ```python
   bridge.set_resource_settings(optimize_for="memory")
   ```
3. Reduce model complexity or use smaller model variants.
4. Implement progressive loading of model components.
5. Use Chrome for best multimodal performance:
   ```python
   bridge.add_browser_arg("--enable-features=ParallelDownloading")
   ```

## Performance Optimization

### Reducing Test Time

1. Use `--quick` option for rapid testing with limited scope.
2. Test only specific browser-model combinations:
   ```bash
   ./run_selenium_integration_tests.sh --firefox-only --audio-only
   ```
3. Avoid simulation mode when real browsers are available.
4. Enable parallel test execution when supported:
   ```bash
   ./run_selenium_integration_tests.sh --parallel 2
   ```
5. Use smaller model variants for development testing.
6. Enable shader precompilation for faster model loading.

### Improving Recovery Success Rate

1. Analyze recovery performance metrics to identify weak spots.
2. Adjust circuit breaker settings based on environment stability:
   ```python
   circuit = CircuitBreaker(
       name="browser_firefox_whisper",
       failure_threshold=5,  # Higher for less stable environments
       recovery_timeout=30.0,  # Longer for slower systems
       half_open_max_calls=2,
       success_threshold=3
   )
   ```
3. Match browsers to model types (Edge for text, Chrome for vision, Firefox for audio).
4. Adjust recovery strategy starting level based on observed failures.
5. Enable model-specific optimizations for each browser.
6. Implement custom recovery strategies for specific failure patterns.

### Memory Usage Optimization

1. Close browsers between test runs.
2. Limit browser cache and history size.
3. Run in headless mode for lower memory usage.
4. Monitor system memory during tests and adjust batch sizes.
5. Implement cleanup steps between model runs.
6. Consider using smaller model variants for testing.
7. For multimodal models, optimize for memory:
   ```python
   bridge.set_resource_settings(
       max_batch_size=1,
       optimize_for="memory",
       progressive_loading=True
   )
   ```

## Advanced Diagnostics

### Enabling Verbose Logging

For detailed debugging information, enable verbose logging:

```bash
# Set environment variables before running tests
export SELENIUM_BRIDGE_LOG_LEVEL=DEBUG
export RECOVERY_STRATEGY_DEBUG=1
export CIRCUIT_BREAKER_VERBOSE=1

# Run the tests with verbose output
./setup_and_run_selenium_tests.sh
```

These settings will provide detailed logs of:
- Browser launch and configuration
- WebGPU/WebNN feature detection
- Recovery strategy selection and execution
- Circuit breaker state transitions
- Failure categorization and analysis

### Manual Browser Testing

To verify browser capabilities outside of the test framework:

1. Check WebGPU support:
   - Chrome/Edge: Navigate to `chrome://gpu` or `edge://gpu`
   - Firefox: Navigate to `about:support` and check "Graphics" section
   
2. Test WebGPU with sample applications:
   - Visit [WebGPU Samples](https://webgpu.github.io/webgpu-samples/)
   - If samples work, the browser supports WebGPU

3. Check WebNN support (Edge):
   - Navigate to `edge://gpu`
   - Look for WebNN status under "Feature Status"

4. Check compute shader support:
   - Chrome/Edge: Visit `chrome://gpu` and look for "accelerated" status
   - Firefox: Set `dom.webgpu.advanced-compute` to `true` in `about:config`

### Circuit Breaker Analysis

To analyze circuit breaker performance:

1. Extract circuit breaker metrics:
   ```python
   metrics = bridge.circuit_breaker.get_metrics()
   print(json.dumps(metrics, indent=2))
   ```

2. Key metrics to monitor:
   - `current_state`: Should return to CLOSED after recovery
   - `circuit_open_count`: Number of times circuit has opened
   - `success_rate`: Should improve over time
   - `avg_downtime_seconds`: Lower is better
   - `recovery_rate`: Higher is better

3. Adjust circuit breaker parameters based on metrics:
   - If `circuit_open_count` is high, increase `failure_threshold`
   - If `avg_downtime_seconds` is high, decrease `recovery_timeout`
   - If `success_rate` is low after recovery, increase `success_threshold`

### Recovery Strategy Analysis

To analyze recovery strategy performance:

1. Create a recovery manager and analyze performance:
   ```python
   recovery_manager = ProgressiveRecoveryManager()
   analysis = recovery_manager.analyze_performance()
   
   # Find best strategy for each browser/model combination
   best_strategies = analysis["best_strategies"]
   print(f"Best strategy for Firefox/Audio: {best_strategies['firefox']['audio']['strategy']}")
   
   # Get time-series performance data
   time_series = analysis["time_series"]
   ```

2. Key metrics to monitor:
   - Strategy success rates by browser and model type
   - Average execution time for different strategies
   - Recovery paths that lead to success
   - Failure types and their frequency

3. Optimize based on analysis:
   - Use the most successful strategies as starting points
   - Adjust strategy parameters based on execution time
   - Customize strategies for specific failure patterns
   - Create specialized strategies for problematic browser/model combinations

## Further Assistance

If you encounter issues not covered in this guide, or if the suggested solutions don't resolve your problem, please:

1. Check the detailed logs by running with `export SELENIUM_BRIDGE_LOG_LEVEL=DEBUG`
2. Run the standalone test cases to isolate the issue
3. Consult the browser-specific documentation for WebGPU/WebNN requirements
4. Review the [SELENIUM_INTEGRATION_README.md](SELENIUM_INTEGRATION_README.md) for additional information
5. File an issue with detailed reproduction steps and logs

For advanced monitoring and visualization of test results, integration with the Distributed Testing Framework dashboard is recommended once it becomes available.