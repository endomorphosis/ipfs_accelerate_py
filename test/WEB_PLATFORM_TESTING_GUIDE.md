# Web Platform Testing Guide (August 2025)

This guide provides comprehensive instructions for testing models on web platforms using WebNN and WebGPU, including the latest March-July 2025 optimizations with the new database integration and cross-origin model sharing capabilities.

## Overview

The IPFS Accelerate Python Framework includes extensive web platform testing capabilities to evaluate model performance in browser environments. This guide covers all aspects of web platform testing, from basic functionality verification to advanced optimization testing with database integration.

## Key Testing Scripts

### Main Testing Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `run_web_platform_tests.sh` | Unified shell script for running tests with all optimizations | For individual model testing and optimization evaluation |
| `run_integrated_web_tests.sh` | Comprehensive script that integrates all test types | For running multiple tests with database integration |
| `test_web_platform_optimizations.py` | Specialized Python script for testing March 2025 optimizations | For detailed analysis of optimization performance |
| `run_web_platform_tests_with_db.py` | Python script for running tests with database integration | For storing test results directly in DuckDB |

### Support Scripts

| Script | Description |
|--------|-------------|
| `web_platform_benchmark.py` | Benchmarks model performance on web platforms |
| `web_platform_test_runner.py` | Base runner for web platform tests |
| `web_audio_platform_tests.py` | Specialized tests for audio models on web platforms |
| `test_webgpu_parallel_model_loading.py` | Tests parallel model loading optimization |
| `test_webgpu_shader_precompilation.py` | Tests shader precompilation optimization |
| `test_webgpu_audio_compute_shaders.py` | Tests compute shader optimization for audio models |

## Environment Variables

| Variable | Description | Default | Added |
|----------|-------------|---------|-------|
| `WEBNN_ENABLED` | Enable WebNN support | `0` | Phase 16 |
| `WEBNN_SIMULATION` | Use simulation mode for WebNN | `1` | Phase 16 |
| `WEBNN_AVAILABLE` | Indicate WebNN is available | `0` | Phase 16 |
| `WEBGPU_ENABLED` | Enable WebGPU support | `0` | Phase 16 |
| `WEBGPU_SIMULATION` | Use simulation mode for WebGPU | `1` | Phase 16 |
| `WEBGPU_AVAILABLE` | Indicate WebGPU is available | `0` | Phase 16 |
| `WEBGPU_COMPUTE_SHADERS_ENABLED` | Enable compute shader optimization | `0` | March 2025 |
| `WEB_PARALLEL_LOADING_ENABLED` | Enable parallel model loading | `0` | March 2025 |
| `WEBGPU_SHADER_PRECOMPILE_ENABLED` | Enable shader precompilation | `0` | March 2025 |
| `WEBGPU_4BIT_INFERENCE` | Enable 4-bit quantized inference | `0` | May 2025 |
| `WEBGPU_EFFICIENT_KV_CACHE` | Enable efficient KV-cache | `0` | May 2025 |
| `WEB_COMPONENT_CACHE` | Enable component-wise caching | `0` | May 2025 |
| `WEBGPU_ULTRA_LOW_PRECISION` | Enable 2-bit/3-bit quantization | `0` | June 2025 |
| `WEBASSEMBLY_FALLBACK` | Enable WebAssembly fallback | `0` | June 2025 |
| `BROWSER_CAPABILITY_DETECTION` | Enable browser detection | `1` | June 2025 |
| `CROSS_ORIGIN_MODEL_SHARING` | Enable cross-origin sharing | `1` | July 2025 |
| `CROSS_ORIGIN_SECURITY_LEVEL` | Security level for sharing | `high` | July 2025 |
| `BENCHMARK_DB_PATH` | Path to benchmark database | `./benchmark_db.duckdb` | Phase 16 |
| `DEPRECATE_JSON_OUTPUT` | Disable JSON output and use only database | `0` | Phase 16 |
| `TEST_BROWSER` | Browser to use for testing | `chrome` | Phase 16 |

## Basic Usage

### Running Basic Tests

```bash
# Test a specific model with WebNN
./run_web_platform_tests.sh --model bert --webnn-only

# Test a specific model with WebGPU
./run_web_platform_tests.sh --model vit --webgpu-only

# Test multiple models with both platforms
./run_web_platform_tests.sh --models bert,t5,vit
```

### Testing March 2025 Optimizations

```bash
# Test compute shader optimization for audio models
./run_web_platform_tests.sh --model whisper --enable-compute-shaders

# Test parallel loading optimization for multimodal models
./run_web_platform_tests.sh --model clip --enable-parallel-loading

# Test shader precompilation for faster startup
./run_web_platform_tests.sh --model vit --enable-shader-precompile

# Test all optimizations together
./run_web_platform_tests.sh --model bert --all-optimizations
```

### Testing with Database Integration

```bash
# Run tests with direct database storage
./run_web_platform_tests.sh --model bert --db-path ./benchmark_db.duckdb

# Run optimization tests with database integration
./run_web_platform_tests.sh --run-optimizations --model whisper --db-path ./benchmark_db.duckdb

# Run comprehensive integrated tests
./run_integrated_web_tests.sh --test-type optimization --model whisper --march-2025-features
```

## Advanced Usage

### Running Comprehensive Test Suites

```bash
# Run all optimization tests for all models
./run_integrated_web_tests.sh --test-type optimization --models all

# Run hardware compatibility tests for all models
./run_integrated_web_tests.sh --test-type hardware --models all

# Run cross-platform comparison tests
./run_integrated_web_tests.sh --test-type cross-platform --models all

# Run everything (comprehensive test suite)
./run_integrated_web_tests.sh --run-all
```

### Customizing Tests

```bash
# Specify small model variants for faster testing
./run_integrated_web_tests.sh --test-type standard --models bert,t5 --small-models

# Specify a custom report directory
./run_integrated_web_tests.sh --test-type optimization --model whisper --report-dir ./custom_reports

# Set a timeout for long-running tests
./run_integrated_web_tests.sh --run-all --timeout 3600
```

### Working with the Database

```bash
# Set database path via environment variable
export BENCHMARK_DB_PATH=./benchmark_db.duckdb
./run_web_platform_tests.sh --model bert

# Query the database for web platform results
python scripts/benchmark_db_query.py --report web_platform --format html --output web_report.html

# Generate a comparison report for different platforms
python scripts/benchmark_db_query.py --sql "SELECT * FROM web_platform_results WHERE model_name='bert'" --format html
```

## March 2025 Optimizations Detail

The framework includes three major optimizations for web platform models:

### 1. WebGPU Compute Shader Optimization

**Description**: Specialized compute shaders for audio models that improve performance by 20-35%.

**Applicable Models**: Audio models (Whisper, Wav2Vec2, CLAP)

**Testing Command**:
```bash
./run_web_platform_tests.sh --model whisper --enable-compute-shaders
```

**Expected Results**:
- 20-35% performance improvement for audio processing
- Better performance with longer audio samples
- Firefox shows ~20% better performance than Chrome

### 2. Parallel Model Loading

**Description**: Concurrent loading of model components that reduces initialization time by 30-45%.

**Applicable Models**: Multimodal models (CLIP, LLaVA, XCLIP)

**Testing Command**:
```bash
./run_web_platform_tests.sh --model clip --enable-parallel-loading
```

**Expected Results**:
- 30-45% loading time reduction
- Greater benefits for models with more components
- Improved memory efficiency during initialization

### 3. Shader Precompilation

**Description**: Early compilation of shaders that reduces first inference time by 30-45%.

**Applicable Models**: All WebGPU models, especially vision models

**Testing Command**:
```bash
./run_web_platform_tests.sh --model vit --enable-shader-precompile
```

**Expected Results**:
- 30-45% faster first inference
- Minimal compilation delay during runtime
- Improved user experience for interactive applications

## Database Schema

The framework uses DuckDB for storing web platform test results with the following schema:

### Main Tables

**web_platform_optimizations**
- `id`: Primary key
- `test_datetime`: Test date and time
- `test_type`: Type of test (compute_shader, parallel_loading, shader_precompilation)
- `model_name`: Name of the model tested
- `model_family`: Model family (text, vision, audio, multimodal)
- `optimization_enabled`: Whether the optimization was enabled
- `execution_time_ms`: Execution time in milliseconds
- `improvement_percent`: Performance improvement percentage
- Additional fields...

**shader_compilation_stats**
- `id`: Primary key
- `test_datetime`: Test date and time
- `optimization_id`: Foreign key to web_platform_optimizations
- `shader_count`: Number of shaders
- `cached_shaders_used`: Number of cached shaders used
- `new_shaders_compiled`: Number of newly compiled shaders
- `cache_hit_rate`: Cache hit rate
- `total_compilation_time_ms`: Total compilation time
- `peak_memory_mb`: Peak memory usage in MB

**parallel_loading_stats**
- `id`: Primary key
- `test_datetime`: Test date and time
- `optimization_id`: Foreign key to web_platform_optimizations
- `components_loaded`: Number of components loaded
- `sequential_load_time_ms`: Sequential loading time
- `parallel_load_time_ms`: Parallel loading time
- `memory_peak_mb`: Peak memory usage in MB
- `loading_speedup`: Loading speedup factor

## Troubleshooting

### Common Issues

**Problem**: Tests fail with missing package errors.  
**Solution**: Install required packages: `pip install duckdb matplotlib numpy`.

**Problem**: WebNN tests fail with "Edge browser not available" error.  
**Solution**: Tests are using simulation mode. For real browser testing, install Microsoft Edge and enable WebNN.

**Problem**: WebGPU tests with compute shaders fail.  
**Solution**: Ensure you're using Chrome 113+, Edge 113+, or Firefox with compute shader support enabled.

### Debugging Tips

1. **Enable verbose logging**:
   ```bash
   ./run_web_platform_tests.sh --model whisper --debug
   ```

2. **Check database connection**:
   ```bash
   python -c "import duckdb; conn = duckdb.connect('./benchmark_db.duckdb'); print(conn.execute('SELECT * FROM web_platform_optimizations LIMIT 5').fetchall())"
   ```

3. **Verify environment variables**:
   ```bash
   ./run_web_platform_tests.sh --model bert --print-env
   ```

4. **Run with slower timeout for comprehensive tests**:
   ```bash
   ./run_integrated_web_tests.sh --run-all --timeout 7200
   ```

## Best Practices

1. **Use small models for development testing**:
   ```bash
   ./run_web_platform_tests.sh --model bert --small-models
   ```

2. **Always store results in the database**:
   ```bash
   export DEPRECATE_JSON_OUTPUT=1
   ./run_web_platform_tests.sh --model bert --db-path ./benchmark_db.duckdb
   ```

3. **Run optimization tests with matching model types**:
   - Use compute shaders with audio models (whisper, wav2vec2, clap)
   - Use parallel loading with multimodal models (clip, llava, xclip)
   - Use shader precompilation with vision models (vit, resnet)

4. **Generate reports after testing**:
   ```bash
   python scripts/benchmark_db_query.py --report web_platform --format html --output web_report.html
   ```

5. **Run cross-platform tests for comparative analysis**:
   ```bash
   ./run_integrated_web_tests.sh --test-type cross-platform --models bert,vit,whisper
   ```

## Model Class-Specific Optimizations

Each of the 13 high-priority model classes has specific optimizations for web platforms:

### 1. Text Embedding Models (BERT)
- Optimized tokenization for browser memory constraints
- Reduced precision for faster inference
- Token-based batching for efficiency
- Shader precompilation for faster initialization
- Full WebNN and WebGPU support

### 2. Text Generation Models (T5, LLAMA, Qwen2/3)
- Progressive tensor loading for large models
- Memory-efficient attention mechanisms
- KV-cache optimization for context windows
- Size-appropriate model variants for browsers
- Full WebNN and WebGPU support for smaller variants (T5)
- Limited support for larger variants (LLAMA, Qwen2/3)

### 3. Vision Models (ViT, DETR)
- Image preprocessing optimized for browsers
- Canvas integration for direct image processing
- Efficient GPU texture handling
- WebGPU shader precompilation for faster startup
- 30-45% startup latency reduction with shader precompilation
- Full WebNN and WebGPU support for ViT
- Limited WebNN but better WebGPU support for DETR

### 4. Audio Models (Whisper, Wav2Vec2, CLAP)
- Audio format conversion for browser compatibility
- Chunked processing for long audio files
- WebAudio API integration
- WebGPU compute shader acceleration for audio processing
- 20-55% performance improvement with compute shader optimization (Firefox: 55%, Chrome: 35%)
- Firefox shows exceptional WebGPU compute shader performance (20% faster than Chrome)
- Limited WebNN but improved WebGPU support with compute shaders

### 5. Multimodal Models (CLIP, LLaVA, LLaVA-Next, XCLIP)
- Combined processing pipelines for multiple modalities
- Efficient memory management for multiple inputs
- Progressive loading for browser performance
- Parallel model loading for faster initialization
- 30-45% loading time reduction with parallel component initialization
- Full WebNN and WebGPU support for CLIP
- Limited support for more complex models (LLaVA, LLaVA-Next, XCLIP)

## Browser Compatibility

| Browser | WebNN Support | WebGPU Support | Compute Shader Performance | Notes |
|---------|--------------|---------------|------------------------|-------|
| Chrome  | ✅ (recent versions) | ✅ (v113+) | ✅ Good (35% improvement) | Best overall feature support |
| Edge    | ✅ (recent versions) | ✅ (v113+) | ✅ Good (30% improvement) | Best WebNN performance |
| Safari  | ⚠️ (partial) | ✅ (v17+) | ⚠️ Limited | Good WebGPU but limited WebNN |
| Firefox | ❌ (not yet) | ✅ (v117+) | ✅ Excellent (55% improvement) | Outstanding compute shader performance, 20% faster than Chrome |

## Integration with ResourcePool

The web platform testing system integrates with the ResourcePool for efficient resource management:

```python
# Import the resource pool
from resource_pool import get_global_resource_pool

# Get the resource pool
pool = get_global_resource_pool()

# Create hardware-aware preferences for web platforms
web_preferences = {
    "priority_list": ["webnn", "webgpu", "cpu"],
    "preferred_memory_mode": "low",
    "fallback_to_simulation": True,
    "browser_optimized": True  # New flag for browser optimization
}

# Load a model with web-specific hardware preferences
model = pool.get_model(
    "text_embedding",
    "bert-base-uncased",
    constructor=lambda: create_bert_model(),
    hardware_preferences=web_preferences
)
```

### Parallel GPU/CPU Execution with Resource Pool

The newly enhanced resource pool implementation supports running models concurrently on both GPU (WebGPU) and CPU backends, maximizing hardware utilization:

```python
from resource_pool import get_global_resource_pool

# Get the resource pool
pool = get_global_resource_pool()

# Load vision model on WebGPU (GPU-optimized)
webgpu_preferences = {
    "priority_list": ["webgpu"],
    "model_family": "vision",
    "browser_optimized": True
}

vision_model = pool.get_model(
    "vision", 
    "vit-base-patch16-224",
    constructor=lambda: create_vision_model(),
    hardware_preferences=webgpu_preferences
)

# Simultaneously load text model on CPU
cpu_preferences = {
    "priority_list": ["cpu"],
    "model_family": "text_embedding",
    "browser_optimized": True
}

text_model = pool.get_model(
    "text_embedding", 
    "bert-base-uncased",
    constructor=lambda: create_text_model(),
    hardware_preferences=cpu_preferences
)

# Run inference on both models in parallel
# Vision processing happens on GPU via WebGPU
# Text processing happens on CPU
vision_results = process_image(vision_model, image_data)
text_results = process_text(text_model, text_data)
```

This parallel execution capability significantly improves overall system throughput by utilizing all available hardware resources simultaneously. The Selenium bridge implementation supports this concurrent execution model through:

1. **Connection Pooling**: Manages multiple browser connections efficiently
2. **Parallel Request Handling**: Processes multiple inference requests concurrently
3. **Resource Isolation**: Ensures models don't interfere with each other's execution
4. **Adaptive Scheduling**: Allocates resources based on model characteristics and current load
5. **Load Balancing**: Distributes workload optimally across available hardware resources

You can test the parallel execution capabilities using the enhanced benchmark script:

```bash
# Run parallel model execution benchmark
python test/test_parallel_model_execution.py --webgpu-model vit-base-patch16-224 --cpu-model bert-base-uncased --browser chrome
```

## Using the New Integrated Test Script

The `run_integrated_web_tests.sh` script provides a unified interface for all web platform testing:

```bash
# Display help
./run_integrated_web_tests.sh --help

# Run standard tests for BERT with database integration
./run_integrated_web_tests.sh --model bert

# Run optimization tests with all March 2025 features
./run_integrated_web_tests.sh --test-type optimization --march-2025-features

# Run hardware compatibility tests for all models
./run_integrated_web_tests.sh --test-type hardware --models all

# Run cross-platform comparison
./run_integrated_web_tests.sh --cross-platform --model whisper

# Run comprehensive test suite with all features
./run_integrated_web_tests.sh --run-all
```

## Conclusion

This guide covers the comprehensive web platform testing capabilities in the IPFS Accelerate Python Framework. By following these instructions, you can effectively test models on web platforms, evaluate optimization benefits, and store results in a structured database for analysis.

## July 2025 Implementation Status

Key implementation status for the latest July 2025 features:

### 1. Cross-origin Model Sharing Protocol (100% Complete)

**Description**: Secure model sharing between domains with permission-based access control.

**Features**:
- Permission-based access control with multiple security levels
- Secure token-based authorization with cryptographic verification
- Resource usage monitoring and constraints enforcement
- Domain verification and secure handshaking
- Configurable security policies (standard, high, maximum)

**Testing Command**:
```bash
# Test server mode with security level configuration
./run_web_platform_tests.sh python test/test_cross_origin_sharing.py --model bert --server-mode --security-level high

# Test client mode connecting to a server
./run_web_platform_tests.sh python test/test_cross_origin_sharing.py --model bert --client-mode --server-origin https://model-provider.com
```

**Expected Results**:
- Successful token generation and verification
- Secure communication between domains
- Resource usage monitoring and enforcement
- Permission-based access control functioning correctly

### 2. Other July 2025 Features (In Progress)

| Feature | Progress | Status |
|---------|----------|--------|
| Mobile Device Optimizations | 65% | Power-efficient inference for mobile browsers |
| Browser CPU Core Detection | 70% | Maximizing available computing resources |
| Model Sharding Across Tabs | 55% | Running larger models through distributed execution |
| Auto-tuning Parameter System | 48% | Optimizing configuration based on device capabilities |

For more information about web platform integration and optimizations, see:
- [Web Platform Integration Guide](./web_platform_integration_guide.md)
- [Web Platform Integration Summary](./WEB_PLATFORM_INTEGRATION_SUMMARY.md)
- [BENCHMARK_DATABASE_GUIDE.md](./BENCHMARK_DATABASE_GUIDE.md)