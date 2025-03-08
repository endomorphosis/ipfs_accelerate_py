# WebNN and WebGPU Benchmarking Guide for 13 High-Priority Models

This document provides comprehensive instructions for running WebNN and WebGPU benchmarks for the 13 high-priority HuggingFace model classes across different quantization levels.

## Existing Selenium Bridge Implementation

The codebase already has a robust implementation for Selenium bridge testing with WebNN and WebGPU. The key files are:

- `/home/barberb/ipfs_accelerate_py/test/implement_real_webnn_webgpu.py`: Main implementation for real browser integration
- `/home/barberb/ipfs_accelerate_py/test/fixed_web_platform/browser_automation.py`: Browser automation utilities
- `/home/barberb/ipfs_accelerate_py/test/run_web_platform_tests.sh`: Script for running web platform tests with browser automation

## Running WebNN/WebGPU Benchmarks

### Prerequisites

1. Ensure you have the required packages installed:
   ```bash
   pip install selenium websockets webdriver-manager
   ```

2. Make sure browsers are installed (Chrome, Firefox, Edge as needed)

### Important: Browser Flag Configuration

The existing Selenium bridge must be properly configured with browser-specific flags to enable experimental WebNN and WebGPU features:

- **Chrome/Edge**:
  ```
  --enable-features=WebML,WebNN,WebNNDMLCompute 
  --disable-web-security 
  --enable-dawn-features=allow_unsafe_apis 
  --enable-webgpu-developer-features 
  --ignore-gpu-blocklist
  ```

- **Firefox**:
  ```
  --MOZ_WEBGPU_FEATURES=dawn 
  --MOZ_ENABLE_WEBGPU=1 
  --MOZ_WEBGPU_ADVANCED_COMPUTE=1
  ```

The existing implementation already sets these flags in `implement_real_webnn_webgpu.py` and `browser_automation.py`.

### Running Benchmarks for All 13 High-Priority Models

To run benchmarks for all 13 high-priority models with WebGPU and different quantization levels:

```bash
# Set database path for storing results
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# Run Web Platform Integration Tests with all optimizations
./run_web_platform_integration_tests.sh --all-models --all-optimizations --use-browser-automation --browser chrome --db-path ./benchmark_db.duckdb
```

For Firefox (which has better audio model performance):
```bash
./run_web_platform_integration_tests.sh --all-models --all-optimizations --use-browser-automation --browser firefox --db-path ./benchmark_db.duckdb
```

For Edge (best WebNN support):
```bash
./run_web_platform_integration_tests.sh --all-models --all-optimizations --use-browser-automation --browser edge --db-path ./benchmark_db.duckdb
```

### Testing Individual Models

To run tests for specific models with browser automation:

```bash
# Test BERT with WebGPU
./run_web_platform_tests.sh --model bert --webgpu-only --use-browser-automation --browser chrome

# Test whisper with WebGPU and compute shader optimization (better with Firefox)
./run_web_platform_tests.sh --model whisper --webgpu-only --enable-compute-shaders --use-browser-automation --browser firefox

# Test BERT with WebNN (best with Edge)
./run_web_platform_tests.sh --model bert --webnn-only --use-browser-automation --browser edge
```

### Testing Specific Optimizations

The framework supports three main optimization types from the March 2025 enhancements:

1. **Compute Shaders** - Especially beneficial for audio models (Whisper, Wav2Vec2, CLAP) with Firefox:
   ```bash
   ./run_web_platform_tests.sh --model whisper --enable-compute-shaders --use-browser-automation --browser firefox
   ```

2. **Shader Precompilation** - Reduces first inference time for all models:
   ```bash
   ./run_web_platform_tests.sh --model bert --enable-shader-precompile --use-browser-automation
   ```

3. **Parallel Loading** - Beneficial for multimodal models (CLIP, LLaVA, XCLIP):
   ```bash
   ./run_web_platform_tests.sh --model clip --enable-parallel-loading --use-browser-automation
   ```

### Testing Quantization Levels

To test different quantization levels with WebGPU:

```bash
# 4-bit quantization for LLMs
python /home/barberb/ipfs_accelerate_py/test/test_webgpu_4bit_inference.py --model llama --compare-precision --compare-hardware --specialized-kernels --browser-specific --output-report webgpu_quantization_report.html
```

## Real Implementation Details

The `implement_real_webnn_webgpu.py` file provides a comprehensive implementation for real browser-based WebNN and WebGPU acceleration:

1. **Browser Automation**: Uses Selenium to launch and control browsers (Chrome, Firefox, Edge, Safari)
2. **Bridge Architecture**: Establishes a WebSocket server for communication between Python and browser
3. **Feature Detection**: Automatically detects browser capabilities for WebNN and WebGPU
4. **transformers.js Integration**: Uses transformers.js for real model inference when available
5. **HTML Interface**: Creates a temporary HTML file with JavaScript code for WebNN/WebGPU execution
6. **Fallback Mechanism**: Gracefully falls back to simulation mode if real implementations fail
7. **Performance Metrics**: Collects detailed metrics on inference time, memory usage, etc.
8. **Resource Pool Integration**: Leverages resource pools to run models concurrently on both GPU and CPU backends

### Resource Pool Integration for Parallel Execution

The implementation now supports running different models concurrently using both GPU (WebGPU) and CPU backends through the resource pool system:

```python
from resource_pool import get_global_resource_pool

# Get the resource pool
pool = get_global_resource_pool()

# Create hardware-aware preferences for WebGPU and CPU
webgpu_preferences = {
    "priority_list": ["webgpu", "cpu"],
    "preferred_memory_mode": "balanced",
    "browser_optimized": True,
    "model_family": "vision"
}

cpu_preferences = {
    "priority_list": ["cpu"],
    "preferred_memory_mode": "low",
    "browser_optimized": True,
    "model_family": "text"
}

# Load a vision model with WebGPU preferences
vision_model = pool.get_model(
    "vision",
    "vit-base-patch16-224",
    constructor=lambda: create_vision_model(),
    hardware_preferences=webgpu_preferences
)

# Simultaneously load a text model with CPU preferences
text_model = pool.get_model(
    "text_embedding",
    "bert-base-uncased",
    constructor=lambda: create_text_model(),
    hardware_preferences=cpu_preferences
)

# Run both models simultaneously - one on WebGPU, one on CPU
vision_result = run_vision_inference(vision_model, image_input)
text_result = run_text_inference(text_model, text_input)
```

This resource pool integration enables efficient utilization of all available hardware resources (both GPU and CPU) in the browser environment, significantly improving overall system throughput.

## The 13 High-Priority Models

These are the 13 high-priority model classes benchmarked:

1. **BERT** (bert-base-uncased) - Text embedding model
2. **T5** (t5-small) - Text-to-text model
3. **LLAMA** (opt-125m) - Large language model
4. **CLIP** (openai/clip-vit-base-patch32) - Multimodal vision-text model
5. **ViT** (google/vit-base-patch16-224) - Vision transformer
6. **CLAP** (laion/clap-htsat-unfused) - Audio-text multimodal model
7. **Whisper** (openai/whisper-tiny) - Audio transcription model
8. **Wav2Vec2** (facebook/wav2vec2-base) - Audio processing model
9. **LLaVA** (llava-hf/llava-1.5-7b-hf) - Large multimodal model
10. **LLaVA-Next** - Next generation multimodal model
11. **XCLIP** (microsoft/xclip-base-patch32) - Video-text multimodal model
12. **Qwen2** (qwen2) - Language model
13. **DETR** (facebook/detr-resnet-50) - Object detection model

## Recommendations for Different Model Types

Based on testing, here are recommendations for optimal browser and optimization combinations:

| Model Type | Best Browser | Recommended Optimizations |
|------------|-------------|---------------------------|
| Text Models (BERT, T5) | Edge for WebNN, Chrome for WebGPU | Shader Precompilation |
| Vision Models (ViT, CLIP) | Chrome | Shader Precompilation |
| Audio Models (Whisper, Wav2Vec2) | Firefox | Compute Shaders + Shader Precompilation |
| Multimodal Models (CLIP, LLaVA) | Chrome | Parallel Loading + Shader Precompilation |
| Large Language Models (LLAMA, Qwen2) | Chrome | 4-bit Quantization + Shader Precompilation |

## Analyzing Benchmark Results

After running benchmarks, query results from the DuckDB database:

```bash
# Generate a comprehensive report of WebGPU benchmark results
python /home/barberb/ipfs_accelerate_py/test/scripts/benchmark_db_query.py --db ./benchmark_db.duckdb --report web_platform --format html --output webgpu_report.html

# Compare WebGPU performance across browsers
python /home/barberb/ipfs_accelerate_py/test/scripts/benchmark_db_query.py --db ./benchmark_db.duckdb --report webgpu --format html --output browser_comparison.html

# Query specific model performance data
python /home/barberb/ipfs_accelerate_py/test/scripts/benchmark_db_query.py --db ./benchmark_db.duckdb --sql "SELECT m.model_name, hp.hardware_type, w.browser, AVG(w.inference_time_ms) as avg_time FROM web_platform_results w JOIN models m ON w.model_id = m.model_id JOIN hardware_platforms hp ON w.hardware_id = hp.hardware_id WHERE w.platform = 'webgpu' GROUP BY m.model_name, hp.hardware_type, w.browser ORDER BY m.model_name" --format markdown --output webgpu_perf.md
```

## Troubleshooting

1. **Browser not found**: Make sure browsers are installed and in your PATH
2. **WebDriver issues**: Try reinstalling the WebDriver with webdriver-manager
3. **WebGPU/WebNN not available**: Check if your browser version supports these features
4. **Simulation mode**: If you see "Simulation" in results, it means real hardware was not available
5. **Database errors**: Ensure DuckDB is properly set up and the path is correct

## Conclusion

Using this guide, you can effectively benchmark all 13 high-priority HuggingFace model classes with WebNN and WebGPU across different browsers and optimization settings. The framework leverages the existing Selenium bridge implementation with proper browser flags to enable experimental features.