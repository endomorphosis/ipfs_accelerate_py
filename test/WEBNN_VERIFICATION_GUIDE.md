# WebNN and WebGPU Verification Guide

This guide provides instructions for verifying that WebNN and WebGPU are properly enabled and functional in browsers. These web APIs provide hardware acceleration for neural network inference and general compute tasks in web browsers.

## Prerequisites

1. A browser that supports WebNN or WebGPU:
   - **Chrome**: Version 113+ (WebGPU), Version 120+ (WebNN experimental)
   - **Edge**: Version 113+ (WebGPU), Version 120+ (WebNN experimental, better support than Chrome)
   - **Firefox**: Version 113+ (WebGPU only, no WebNN support yet)
   - **Safari**: Version 17+ (WebGPU only, limited support, no WebNN)

2. Required Python packages:
   ```bash
   pip install selenium websockets transformers torch webdriver-manager
   ```

3. WebDriver for browser automation is installed automatically via webdriver-manager

## Step 1: Check Browser Configuration

### Chrome/Edge WebNN Configuration

1. Launch Chrome/Edge with the following flags:
   ```bash
   # For Chrome
   google-chrome --enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist
   
   # For Edge
   msedge --enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist
   ```

2. Check if WebNN is enabled:
   - Navigate to `chrome://flags` or `edge://flags`
   - Search for "WebNN"
   - Ensure "WebNN API" is enabled
   - Restart the browser

### Firefox WebGPU Configuration

Firefox doesn't support WebNN yet, but has good WebGPU support:

1. Navigate to `about:config`
2. Search for "webgpu"
3. Set `dom.webgpu.enabled` to `true`
4. For better audio model support, set `dom.webgpu.compute.enabled` to `true`

## Step 2: Run Automatic Capability Detection

Use our automated tools to check browser capabilities:

```bash
# Run the browser capability check script
./run_browser_capability_check.sh --browser chrome
```

This will:
1. Launch the browser with appropriate flags
2. Check for WebNN and WebGPU support
3. Generate a JSON report of all capabilities
4. Print a summary of the support status

Example output:
```
=== Browser Capabilities ===
Browser: chrome
User Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.85 Safari/537.36

WebNN:
  Supported: Yes
  Backends: cpu, gpu

WebGPU:
  Supported: Yes
  Adapter: Google - SwiftShader

WebGL:
  Supported: Yes
  Vendor: Google Inc.
  Renderer: ANGLE (Google, SwiftShader Device)

WebAssembly:
  Supported: Yes
  SIMD: Yes
============================
```

## Step 3: Run WebNN Performance Benchmark

To verify WebNN is not just enabled but actually working and providing performance benefits:

```bash
# Run the WebNN benchmark script
./run_webnn_benchmark.sh --browser edge --model bert-base-uncased

# For smaller model (faster testing and better compatibility)
./run_webnn_benchmark.sh --browser edge --model prajjwal1/bert-tiny
```

This will:
1. Launch the browser with appropriate WebNN flags
2. Run a benchmark comparing WebNN performance with CPU performance
3. Generate a JSON report with performance metrics
4. Print a summary of the results

Example output:
```
=== WebNN Benchmark Results ===
Model: bert-base-uncased
Browser: edge
WebNN Status: real_hardware

WebNN Performance:
  Average Latency: 85.45 ms
  Throughput: 11.70 items/sec
  Simulation Mode: No

CPU Performance:
  Average Latency: 195.23 ms
  Throughput: 5.12 items/sec

Speedup: 2.28x
================================
```

You can test with different models and browsers to compare performance:

```bash
# Test with Chrome (latest version)
./run_webnn_benchmark.sh --browser chrome --model prajjwal1/bert-tiny

# Test with Edge (typically best WebNN support)
./run_webnn_benchmark.sh --browser edge --model prajjwal1/bert-tiny

# Test with different batch sizes (increases computational load)
./run_webnn_benchmark.sh --browser edge --model prajjwal1/bert-tiny --batch-size 8
```

### Understanding the Results

- **WebNN Status**: 
  - `real_hardware`: WebNN is enabled and using real hardware acceleration
  - `simulation`: WebNN API is present but not using real hardware acceleration
  - `supported`: WebNN is supported but not accelerating this model
  - `not_supported`: WebNN is not supported by this browser

- **Speedup**: The performance ratio between CPU and WebNN. A value > 1 indicates WebNN is faster.

## Step 4: Run a Full Web Platform Benchmark

For a more comprehensive benchmark across multiple models:

```bash
# Using the DuckDB-integrated benchmark runner
python run_web_platform_tests_with_db.py --run-webnn --models bert t5 vit --db-path ./benchmark_db.duckdb

# Compare WebNN and WebGPU performance
python run_web_platform_tests_with_db.py --comparative --browser edge --db-path ./benchmark_db.duckdb
```

This will:
1. Run benchmarks for multiple models on WebNN and/or WebGPU
2. Store results in the benchmark database
3. Generate visualizations comparing performance

## Common Issues and Solutions

### WebNN Not Detected

If WebNN is not detected despite enabling flags:

1. Make sure you're using Chrome 120+ or Edge 120+
2. Try launching with all flags combined:
   ```bash
   google-chrome --enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist
   ```
3. Check Chrome/Edge version:
   ```bash
   google-chrome --version
   msedge --version
   ```

### Using Simulation Mode

If WebNN is in "simulation mode":

1. Your hardware might not support WebNN accelerated operations
2. Try using Edge instead of Chrome (better WebNN support)
3. Check GPU drivers are up to date
4. Try a simpler model like "prajjwal1/bert-tiny"

### WebGPU as Alternative

If WebNN isn't working, WebGPU is a good alternative:

```bash
# Run WebGPU benchmark
python run_web_platform_tests_with_db.py --run-webgpu --models bert --browser firefox --db-path ./benchmark_db.duckdb
```

## Firefox-Specific Optimizations

For audio model testing, Firefox often outperforms Chrome/Edge with WebGPU compute shaders:

```bash
# Test audio models with Firefox's optimized compute shaders
python run_web_platform_tests_with_db.py --run-webgpu --models whisper,wav2vec2,clap --browser firefox --compute-shaders
```

## Further Verification Methods

### Check JavaScript API Availability

In the browser console (F12), you can verify APIs are available:

```javascript
// Check WebNN
console.log("WebNN available:", 'ml' in navigator);

// Check WebNN backends
if ('ml' in navigator) {
  navigator.ml.createContext({devicePreference: 'gpu'})
    .then(ctx => console.log("WebNN GPU context created:", ctx))
    .catch(err => console.error("WebNN GPU context failed:", err));
}

// Check WebGPU  
console.log("WebGPU available:", 'gpu' in navigator);

// Check WebGPU adapter
if ('gpu' in navigator) {
  navigator.gpu.requestAdapter()
    .then(adapter => {
      console.log("WebGPU adapter:", adapter);
      return adapter?.requestAdapterInfo();
    })
    .then(info => console.log("Adapter info:", info))
    .catch(err => console.error("WebGPU adapter error:", err));
}
```

### Test with Sample Models Online

Several websites can be used to verify WebNN/WebGPU:

1. **TensorFlow.js WebNN Demo**: https://webnn-tfjs.netlify.app/
2. **WebNN Demos**: https://webnn-samples.github.io/webnn-samples/
3. **WebGPU Samples**: https://austin-eng.com/webgpu-samples/

## Performance Measurements

For detailed performance analysis, the system's test tools provide:

- Hardware acceleration verification
- Real-time performance metrics
- Cross-browser comparisons
- Memory usage tracking
- Hardware-software compatibility matrix

Results are stored in DuckDB for analysis, with automatic generation of performance comparison reports and charts.

## Conclusion

By following these steps, you can confirm that WebNN is properly enabled and providing hardware acceleration benefits. Remember that WebNN is still an experimental technology, and support varies across browsers and hardware.

For the latest updates on WebNN and WebGPU support, check:
- [WebNN API Status](https://webnn.dev)
- [WebGPU Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)
- [Chrome Platform Status](https://chromestatus.com/feature/5650147831078912) (WebNN)

For troubleshooting issues or updating this guide, please open an issue in the repository.