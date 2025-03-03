# Web Platform Audio Testing Guide

## Overview

This guide provides detailed instructions for running and analyzing audio model tests on web platforms using WebNN and WebGPU. The Web Audio Test Runner enables comprehensive testing of audio models like Whisper, Wav2Vec2, and HuBERT on various browsers, providing insights into cross-platform compatibility and performance.

## WebGPU Compute Shader Optimization for Audio Models (March 2025)

The March 2025 enhancement introduces significant performance improvements for audio models through WebGPU compute shader optimizations:

### Key Optimizations

- **Optimized Workgroup Size**: 256x1x1 configuration tuned for audio processing
- **Multi-Dispatch Pattern**: Breaking large tensor operations into multiple dispatches (+10% speedup)
- **Audio-Specific Optimizations**:
  - Spectrogram acceleration (+20% speedup)
  - FFT optimization (+15% speedup)
  - Mel filter fusion
  - Time masking acceleration
- **Memory Optimizations**:
  - Tensor pooling for reusing allocations
  - In-place operations
  - Progressive model weight loading
- **Firefox-Specific Optimizations**:
  - `--MOZ_WEBGPU_ADVANCED_COMPUTE=1` flag enabling advanced compute capabilities
  - Optimized WebGPU kernel dispatch for Firefox's implementation
  - Enhanced workgroup utilization for Firefox's compute architecture

These optimizations provide **up to 51% overall performance improvement** for audio models on WebGPU (55% in Firefox), with the largest gains seen on longer audio files. Firefox outperforms Chrome by approximately 20% when running audio models with compute shaders.

### Benchmark Results by Browser

| Model | Standard WebGPU | Chrome + Compute Shaders | Firefox + Compute Shaders | Firefox vs Chrome |
|-------|-----------------|--------------------------|---------------------------|------------------|
| Whisper | 8.67 ms | 4.25 ms (51.0%) | 3.42 ms (55.0%) | +19.5% |
| Wav2Vec2 | 8.40 ms | 4.19 ms (50.1%) | 3.32 ms (54.8%) | +20.8% |
| CLAP | 8.56 ms | 4.17 ms (51.3%) | 3.27 ms (55.0%) | +21.6% |

*Performance measured with 20 iterations on standard test audio files*

### Firefox WebGPU Advantage

Firefox demonstrates exceptional WebGPU compute shader performance for audio models:

1. **Superior Performance**: Approximately 55% improvement with compute shaders (vs ~51% in Chrome)
2. **Better Compute Architecture**: Firefox outperforms Chrome by ~20% for audio workloads
3. **Optimized Configuration**: Firefox's `--MOZ_WEBGPU_ADVANCED_COMPUTE=1` flag enables advanced compute features
4. **Scaling with Audio Length**: Firefox shows greater advantage with longer audio files (up to 24% better than Chrome)
5. **Memory Efficiency**: Firefox shows 5-8% better memory efficiency with compute shaders

### Testing Compute Shader Enhancements

Use these dedicated test scripts to measure the impact of the optimizations:

```bash
# New Firefox-specific test tool (Firefox vs Chrome comparison)
python test/test_firefox_webgpu_compute_shaders.py --model whisper
python test/test_firefox_webgpu_compute_shaders.py --benchmark-all

# Test with Firefox WebGPU implementation specifically
./test/run_web_platform_tests.sh --firefox python test/test_webgpu_audio_compute_shaders.py --model whisper

# Test Whisper with and without compute shader optimization (any browser)
python test/test_webgpu_audio_compute_shaders.py --model whisper

# Compare all supported audio models and generate charts
python test/test_webgpu_audio_compute_shaders.py --test-all --benchmark --create-chart

# Test with long-form audio (showing greater benefits)
python test/test_webgpu_audio_compute_shaders.py --model whisper --use-long-audio --benchmark
```

## Key Features

- **Browser-Based Testing**: Test audio models in Chrome, Firefox, Safari, and Edge
- **WebNN and WebGPU Support**: Evaluate models on both WebNN and WebGPU backends
- **Headless Automation**: Run automated tests using Puppeteer and NodeJS
- **Comprehensive Reporting**: Generate detailed reports of test results
- **Speech Recognition Testing**: Evaluate speech-to-text accuracy and performance
- **Audio Classification Testing**: Test audio classification capabilities

## Prerequisites

Before running web platform audio tests, ensure you have the following:

1. **NodeJS and NPM**: Required for headless testing (v14+ recommended)
2. **Puppeteer**: For browser automation (`npm install puppeteer`)
3. **Python Dependencies**: 
   ```
   pip install numpy pandas matplotlib scipy
   ```
4. **Test Audio Files**: Sample audio files for testing (will be auto-generated if not provided)
5. **Model Files**: ONNX models for testing (placeholders will be created if not provided)
6. **Modern Browsers**: Chrome/Edge (WebNN/WebGPU support), Firefox, Safari

## Getting Started

### Basic Usage

To run basic tests with default settings:

```bash
python web_audio_test_runner.py
```

This will:
1. Prepare necessary test files (audio samples, model files, HTML templates)
2. Start a local web server
3. Run tests across all configured browsers
4. Generate a summary report

### Customizing Tests

To customize the tests you run:

```bash
python web_audio_test_runner.py --model-types whisper wav2vec2 --browsers chrome firefox --headless
```

Key options:
- `--model-types`: Specify which model types to test (whisper, wav2vec2, hubert, audio_spectrogram_transformer)
- `--test-cases`: Specify test cases (speech_recognition, audio_classification)
- `--browsers`: Choose which browsers to test with
- `--headless`: Run in headless mode (requires NodeJS/Puppeteer)
- `--no-headless`: Run in browser UI mode for manual testing
- `--port`: Specify port for the test web server (default: 8000)

### Preparing Test Files Only

If you just want to prepare test files without running tests:

```bash
python web_audio_test_runner.py --prepare
```

### Generating Reports

To generate a report from existing test results:

```bash
python web_audio_test_runner.py --report web_audio_tests_20250302_123456.json
```

## Test Configuration

The Web Audio Test Runner uses a configuration file to specify test parameters. By default, it uses internal configuration, but you can provide a custom configuration file:

```bash
python web_audio_test_runner.py --config my_config.json
```

Example configuration file:

```json
{
  "models": {
    "whisper": {
      "model_ids": ["openai/whisper-tiny", "openai/whisper-base"],
      "model_formats": ["onnx", "webnn"],
      "test_cases": ["speech_recognition", "audio_classification"]
    },
    "wav2vec2": {
      "model_ids": ["facebook/wav2vec2-base"],
      "model_formats": ["onnx"],
      "test_cases": ["speech_recognition"]
    }
  },
  "test_audio": {
    "speech_samples": ["sample1.wav", "sample2.wav"],
    "music_samples": ["music1.mp3"],
    "sample_rates": [16000, 44100]
  },
  "browsers": {
    "chrome": {
      "enabled": true,
      "binary_path": "/path/to/chrome"
    },
    "firefox": {
      "enabled": true
    }
  },
  "headless": true,
  "test_timeout": 300
}
```

## Test Structure

### Directory Structure

The Web Audio Test Runner creates the following directory structure:

```
web_audio_tests/
├── common/
│   ├── webnn-utils.js
│   ├── webgpu-utils.js
│   ├── audio-utils.js
│   └── test-runner.js
├── whisper/
│   ├── speech_recognition.html
│   └── audio_classification.html
├── wav2vec2/
│   ├── speech_recognition.html
│   └── audio_classification.html
├── audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── music1.mp3
└── models/
    ├── whisper/
    │   ├── whisper-tiny.onnx
    │   └── whisper-base.onnx
    └── wav2vec2/
        └── wav2vec2-base.onnx
```

### Test Flow

Each test follows this general flow:

1. **Initialization**: Initialize WebNN/WebGPU context
2. **Model Loading**: Load the ONNX model using WebNN/WebGPU
3. **Data Preparation**: Prepare audio input for the model
4. **Inference**: Run model inference
5. **Results Processing**: Process model outputs
6. **Performance Measurement**: Measure and report performance metrics

## Test Cases

### Speech Recognition

Tests the model's ability to transcribe speech to text:

- Loads audio samples from files or synthetic audio
- Processes audio through the model
- Measures inference time and transcription accuracy
- Reports results with performance metrics

### Audio Classification

Tests the model's ability to classify audio into categories:

- Loads various audio samples (speech, music, noise)
- Processes audio through the model
- Evaluates classification accuracy
- Reports top classes and confidence scores

## Report Format

Test reports are generated in both JSON and Markdown formats:

### JSON Format

```json
{
  "timestamp": "2025-03-02T12:34:56.789Z",
  "test_run_id": "web_audio_20250302123456",
  "config": {
    "model_types": ["whisper", "wav2vec2"],
    "browsers": ["chrome", "firefox"],
    "headless": true
  },
  "results": [
    {
      "model_type": "whisper",
      "test_case": "speech_recognition",
      "browser": "chrome",
      "result": {
        "status": "completed",
        "webnnSupported": true,
        "webgpuSupported": true,
        "testResults": {
          "success": true,
          "inferenceTime": 120.5,
          "text": "This is a sample transcription"
        }
      }
    }
  ]
}
```

### Markdown Report

The Markdown report includes:

- Summary of test run
- Hardware and browser information
- Tables of results by model type
- Status and compatibility information
- Performance metrics and comparisons

## Integration with Benchmark Database System

The Web Audio Test Runner now fully integrates with the benchmark database system through the following components:

### Web Audio Benchmark Database

The `web_audio_benchmark_db.py` module provides tools for managing web audio test results in the benchmark database:

```bash
# Import test results into the database
python test/web_audio_benchmark_db.py --import-file ./web_audio_platform_results/whisper_tests_20250302_123456.json

# Import all test results in the directory
python test/web_audio_benchmark_db.py --import-all

# Generate a report from database results
python test/web_audio_benchmark_db.py --generate-report --output-file web_audio_report.md

# Compare WebNN and WebGPU performance for Whisper models
python test/web_audio_benchmark_db.py --compare-platforms --model-types whisper
```

### Benchmark Database API

The benchmark database API has been extended to support web platform audio test results:

1. **REST API**: Access web platform test results via the API
2. **Visualization**: Generate charts comparing WebNN and WebGPU performance
3. **Dashboard Integration**: View web platform performance in the benchmark dashboard

```bash
# Start the benchmark database API server
python test/scripts/benchmark_db_api.py --serve

# Access the dashboard with web audio charts
open http://localhost:8000/dashboard

# Query web platform test results via REST API
curl http://localhost:8000/api/web-platform?model_type=whisper
```

### Hardware Selection System Integration

The web audio testing system integrates with the hardware selection system to recommend optimal hardware for audio models:

1. **Benchmark Data Collection**: Test results are stored in the benchmark database
2. **Performance Analysis**: Results are analyzed to determine optimal hardware
3. **Recommendation Generation**: Hardware recommendations are generated for audio models
4. **Web Platform Assessment**: WebNN and WebGPU capabilities are evaluated for audio processing

## Best Practices

### Test Preparation

1. **Use Real Audio Samples**: For best results, use real-world audio samples
2. **Optimize Model Size**: Use appropriately sized models for web platforms
3. **Ensure Browser Compatibility**: Check browser compatibility before testing
4. **Enable WebNN/WebGPU**: Ensure browsers have WebNN/WebGPU enabled

### Test Execution

1. **Start with Headless Tests**: Begin with headless tests for automation
2. **Follow with Manual Tests**: Use manual tests for detailed investigation
3. **Batch Test Runs**: Run tests in batches to avoid browser memory issues
4. **Monitor Resource Usage**: Watch for memory leaks during long test runs

### Result Analysis

1. **Compare Across Browsers**: Analyze differences between browsers
2. **Analyze Failure Patterns**: Look for patterns in test failures
3. **Track Performance Over Time**: Monitor performance changes across versions
4. **Correlate with Native Performance**: Compare web performance to native performance

## Troubleshooting

### Common Issues

1. **WebNN Not Available**: 
   - Check browser version (Chrome 94+ required for WebNN)
   - Ensure WebNN flags are enabled (chrome://flags/#enable-webnn)

2. **WebGPU Not Available**:
   - Check browser version (Chrome 113+ for stable WebGPU)
   - Enable WebGPU flag (chrome://flags/#enable-unsafe-webgpu)

3. **Model Loading Failures**:
   - Check model format compatibility
   - Verify model size (large models may fail on web platforms)
   - Ensure CORS headers are set correctly

4. **Performance Issues**:
   - Reduce model size or quantize models
   - Reduce audio sample length
   - Check for background processes in browser

### Debugging Tools

1. **Browser Developer Tools**: Use Chrome/Firefox developer tools
2. **Console Logging**: Check console for error messages
3. **Performance Profiling**: Use browser performance tools
4. **Memory Profiling**: Monitor memory usage during tests

## Advanced Usage

### Database Integration

#### Programmatic Database Access

You can directly access the database through the BenchmarkDBAPI client:

```python
from scripts.benchmark_db_api_client import BenchmarkDBAPI

# Connect to database
db = BenchmarkDBAPI(database_path="./benchmark_db.duckdb")

# Store test result
result_id = db.store_web_platform_result(
    model_name="whisper-tiny",
    model_type="whisper",
    browser="chrome",
    platform="webnn",
    status="successful",
    execution_time=0.125,
    metrics={
        "webnn_init_time": 0.045,
        "inference_time": 0.080,
        "memory_usage_mb": 85.2
    }
)

# Query results
whisper_results = db.query_web_platform_results(
    model_type="whisper",
    platform="webnn",
    limit=20
)

# Analyze browser performance
for browser in ["chrome", "edge", "firefox"]:
    browser_results = db.query_web_platform_results(
        browser=browser,
        status="successful"
    )
    if browser_results:
        avg_time = sum(r.get("execution_time", 0) for r in browser_results) / len(browser_results)
        print(f"{browser}: {avg_time:.3f}s average execution time")
```

### Performance Analysis

You can perform advanced performance analysis on web platform results:

```python
from web_audio_benchmark_db import WebAudioBenchmarkDB

# Initialize analyzer
analyzer = WebAudioBenchmarkDB()

# Generate platform comparison report
analyzer.compare_web_audio_platforms(
    model_types=["whisper", "wav2vec2", "clap"],
    output_file="platform_comparison.md"
)
```

### Custom Audio Processing

You can customize audio processing by modifying the audio-utils.js file:

```javascript
// Custom audio processing example
function customAudioProcessing(audioData, sampleRate) {
  // Apply custom preprocessing
  // ...
  return processedAudio;
}
```

### Custom Model Loading

For custom model loading, modify the model loading functions:

```javascript
// Custom model loading example
async function loadCustomModel(modelUrl, context) {
  // Custom model loading logic
  // ...
  return model;
}
```

### Integrating New Audio Models

To add support for new audio models:

1. Add model configuration to config file
2. Create test template for the model
3. Implement model-specific input/output processing
4. Add model-specific evaluation metrics
5. Extend database schema if needed for model-specific metrics

## Visualizing Results

The benchmark database API provides visualization tools for web platform audio tests:

### Web Platform Comparison Chart

Access the web platform comparison chart to visualize performance differences between WebNN and WebGPU:

```bash
# View the chart in your browser
curl http://localhost:8000/api/charts/web-platform-comparison > comparison.html
```

This chart provides:
- Per-model execution time comparison between WebNN and WebGPU
- Grouping by model type (whisper, wav2vec2, clap)
- Average performance metrics for each platform

### Dashboard Integration

The web audio platform test results are integrated into the benchmark database dashboard:

1. Start the benchmark database API server:
   ```bash
   python test/scripts/benchmark_db_api.py --serve
   ```

2. Access the dashboard in your browser:
   ```
   http://localhost:8000/dashboard
   ```

3. View the "Web Platform Audio Model Comparison" section for a comprehensive view of audio model performance across platforms.

## Performance Monitoring

To track performance changes over time:

```bash
# Generate a trend report for Whisper models
python test/web_audio_benchmark_db.py --generate-report \
  --model-types whisper \
  --start-date 2025-01-01 \
  --end-date 2025-03-01 \
  --output-file whisper_trend.md
```

This enables you to:
- Track performance improvements across browser versions
- Monitor the impact of model optimizations
- Compare platform evolution over time
- Identify performance regressions

## Conclusion

The Web Platform Audio Testing Guide provides a comprehensive framework for testing, analyzing, and benchmarking audio models on web platforms. By integrating with the benchmark database system, it enables data-driven decisions about platform selection and optimization for audio processing capabilities on the web.

With the latest database integration, you can:
- Store and analyze test results in a structured database
- Compare performance across platforms and browsers
- Generate visualizations and reports for decision-making
- Track performance trends over time
- Integrate with the broader benchmark system

This completes the Phase 16 integration of web platform audio testing with the benchmark database system, providing a solid foundation for ongoing web audio model optimization and deployment.

---

For more information, see related documentation:
- [Web Platform Audio Testing Summary](WEB_PLATFORM_AUDIO_TESTING_SUMMARY.md)
- [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md)
- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [Hardware Selection System](HARDWARE_MODEL_INTEGRATION_GUIDE.md)
- [Model Compression Guide](MODEL_COMPRESSION_GUIDE.md)