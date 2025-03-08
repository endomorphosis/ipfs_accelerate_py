# WebNN and WebGPU Benchmarking Guide - May 2025 Update

## Overview

This guide provides detailed instructions for benchmarking models using WebNN and WebGPU in real browser environments. Our enhanced unified entry point script now supports all 13 high-priority HuggingFace model classes with comprehensive quantization formats across various browsers.

## Key Entry Point: run_real_webnn_webgpu.py

`run_real_webnn_webgpu.py` serves as the primary entry point for all WebNN and WebGPU benchmarking tasks. This enhanced script provides a unified interface for testing models in real browser environments with various configurations, batch sizes, and precision formats.

### Basic Usage

```bash
# Test a model with WebGPU in Chrome
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased

# Test a model with WebNN in Edge
python test/run_real_webnn_webgpu.py --platform webnn --browser edge --model bert-base-uncased

# Test with specific batch size
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --batch-size 4

# Test with specific precision (quantization)
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --precision int8
```

### Advanced Usage

```bash
# Test with mixed precision
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --mixed-precision

# Ultra-low precision (2-bit)
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --precision int2

# Run in headless mode (for CI environments)
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --headless

# Store results directly in database
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --db-path ./benchmark_db.duckdb

# Generate a report after benchmarking
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --report --format markdown --output webgpu_report.md
```

### Testing All High-Priority Model Classes

```bash
# Test all 13 high-priority model classes with default settings
python test/run_real_webnn_webgpu.py --test-all-models --platform webgpu --browser chrome

# Test specific model categories
python test/run_real_webnn_webgpu.py --test-text-models --platform webgpu --browser chrome
python test/run_real_webnn_webgpu.py --test-vision-models --platform webgpu --browser chrome
python test/run_real_webnn_webgpu.py --test-audio-models --platform webgpu --browser firefox
python test/run_real_webnn_webgpu.py --test-multimodal-models --platform webgpu --browser chrome
python test/run_real_webnn_webgpu.py --test-llm-models --platform webgpu --browser chrome

# Test all model classes with a specific precision format
python test/run_real_webnn_webgpu.py --test-all-models --platform webgpu --precision int4

# Quick test with a tiny model
python test/run_real_webnn_webgpu.py --quick-test --platform webgpu --browser chrome
```

### Browser-Specific Optimizations

```bash
# Enable compute shader optimization for audio models (best with Firefox)
python test/run_real_webnn_webgpu.py --test-audio-models --platform webgpu --browser firefox --enable-compute-shaders

# Enable shader precompilation for faster startup
python test/run_real_webnn_webgpu.py --test-text-models --platform webgpu --browser chrome --enable-shader-precompile

# Enable parallel loading for multimodal models
python test/run_real_webnn_webgpu.py --test-multimodal-models --platform webgpu --browser chrome --enable-parallel-loading

# Enable all optimizations at once
python test/run_real_webnn_webgpu.py --test-all-models --platform webgpu --browser chrome --all-optimizations
```

## Supported Precision Formats

The framework supports multiple precision formats for WebNN and WebGPU inference:

| Precision | WebGPU Support | WebNN Support | Command Line Flag | Memory Reduction | Notes |
|-----------|---------------|---------------|-------------------|------------------|-------|
| FP32 | ✅ Full | ✅ Full | `--precision fp32` or `--bits 32` | None | Highest precision, baseline for accuracy |
| FP16 | ✅ Full | ✅ Full | `--precision fp16` or `--bits 16` | ~50% vs FP32 | Native format for most WebGPU implementations |
| INT8 | ✅ Full | ✅ Full | `--precision int8` or `--bits 8` | ~75% vs FP32 | Good balance of accuracy and performance |
| INT4 | ✅ Full | ⚠️ Experimental | `--precision int4` or `--bits 4` | ~87.5% vs FP32 | Some accuracy loss, but excellent performance |
| INT2 | ✅ Full | ⚠️ Experimental | `--precision int2` or `--bits 2` | ~93.75% vs FP32 | Significant accuracy loss, but maximum memory savings |
| Mixed | ✅ Full | ⚠️ Partial | `--mixed-precision` | Varies | Uses higher precision for critical layers |

Example commands for testing different precision formats:

```bash
# Test FP16 precision (default for WebGPU)
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --precision fp16

# Test INT8 precision (default for WebNN)
python test/run_real_webnn_webgpu.py --platform webnn --browser edge --model bert-base-uncased --precision int8

# Test INT4 precision with WebGPU
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --precision int4

# Test INT2 precision with WebGPU
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --precision int2

# Test experimental INT4 precision with WebNN
python test/run_real_webnn_webgpu.py --platform webnn --browser edge --model bert-base-uncased --precision int4 --experimental-precision

# Test with mixed precision
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --mixed-precision
```

## Browser-Specific Configurations

Different browsers have varying levels of support for WebNN and WebGPU:

### Chrome

```bash
# WebGPU in Chrome (best overall support)
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased

# WebNN in Chrome (good support)
python test/run_real_webnn_webgpu.py --platform webnn --browser chrome --model bert-base-uncased

# Chrome with shader precompilation optimization
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --enable-shader-precompile
```

Chrome browser flags set automatically:
- `--enable-features=WebML,WebNN,WebNNDMLCompute`
- `--disable-web-security`
- `--enable-dawn-features=allow_unsafe_apis`
- `--enable-webgpu-developer-features`
- `--ignore-gpu-blocklist`

### Firefox

```bash
# WebGPU in Firefox (best for audio models)
python test/run_real_webnn_webgpu.py --platform webgpu --browser firefox --model whisper-tiny

# Firefox with audio optimization (20% faster than Chrome for audio)
python test/run_real_webnn_webgpu.py --platform webgpu --browser firefox --model whisper-tiny --enable-compute-shaders

# Firefox with all optimizations
python test/run_real_webnn_webgpu.py --platform webgpu --browser firefox --model whisper-tiny --all-optimizations
```

Firefox browser flags set automatically:
- `--MOZ_WEBGPU_FEATURES=dawn`
- `--MOZ_ENABLE_WEBGPU=1`
- `--MOZ_WEBGPU_ADVANCED_COMPUTE=1`

### Edge

```bash
# WebNN in Edge (best WebNN support)
python test/run_real_webnn_webgpu.py --platform webnn --browser edge --model bert-base-uncased

# WebNN in Edge with experimental 4-bit precision
python test/run_real_webnn_webgpu.py --platform webnn --browser edge --model bert-base-uncased --precision int4 --experimental-precision
```

Edge browser flags set automatically:
- `--enable-features=WebML,WebNN,WebNNDMLCompute`
- `--disable-web-security`
- `--enable-dawn-features=allow_unsafe_apis`
- `--enable-webgpu-developer-features`
- `--ignore-gpu-blocklist`

### Safari

```bash
# WebGPU in Safari (limited support)
python test/run_real_webnn_webgpu.py --platform webgpu --browser safari --model bert-base-uncased

# WebNN in Safari (limited support)
python test/run_real_webnn_webgpu.py --platform webnn --browser safari --model bert-base-uncased
```

## High-Priority Model Classes

The framework now provides comprehensive testing for all 13 high-priority model classes:

| Category | Model Classes | Command Line Option | Example Models | 
|----------|--------------|---------------------|----------------|
| Text | BERT, T5 | `--test-text-models` | bert-base-uncased, t5-small |
| Text Generation | LLAMA, Qwen2/3 | `--test-llm-models` | facebook/opt-125m, Qwen/Qwen1.5-0.5B |
| Vision | ViT, CLIP, DETR | `--test-vision-models` | google/vit-base-patch16-224, facebook/detr-resnet-50 |
| Audio | Whisper, Wav2Vec2, CLAP | `--test-audio-models` | openai/whisper-tiny, facebook/wav2vec2-base |
| Multimodal | LLaVA, LLaVA-Next, XCLIP | `--test-multimodal-models` | llava-hf/llava-1.5-7b-hf, hit-cvlab/xclip-base-patch32 |
| All Model Classes | All 13 classes | `--test-all-models` | All of the above |

Example commands:

```bash
# Test all model classes with default settings
python test/run_real_webnn_webgpu.py --test-all-models --platform webgpu --browser chrome

# Test text models with WebNN and 8-bit precision
python test/run_real_webnn_webgpu.py --test-text-models --platform webnn --browser edge --precision int8

# Test audio models with Firefox's optimized compute shaders
python test/run_real_webnn_webgpu.py --test-audio-models --platform webgpu --browser firefox --enable-compute-shaders

# Test vision models with mixed precision
python test/run_real_webnn_webgpu.py --test-vision-models --platform webgpu --browser chrome --mixed-precision

# Test a specific model with detailed options
python test/run_real_webnn_webgpu.py --model openai/whisper-tiny --model-type audio --platform webgpu --browser firefox --enable-compute-shaders
```

## Batch Sizes and Advanced Options

The framework supports advanced configuration options for benchmarking:

```bash
# Test with different batch sizes
python test/run_real_webnn_webgpu.py --model bert-base-uncased --batch-size 8 --platform webgpu

# Check browser capabilities only
python test/run_real_webnn_webgpu.py --check-capabilities --browser chrome

# Force simulation mode (useful for testing without hardware)
python test/run_real_webnn_webgpu.py --model bert-base-uncased --simulation-only

# Require real hardware (fail if simulation would be used)
python test/run_real_webnn_webgpu.py --model bert-base-uncased --no-simulation
```

## Model URL Verification and Fallback System

The framework includes a robust model verification and fallback system:

```bash
# Test with URL verification and fallback conversion
python test/run_real_webnn_webgpu.py --model bert-base-uncased --verify-url --fallback-convert
```

## Database Integration

All benchmark results can be stored in a DuckDB database for comprehensive analysis:

```bash
# Run benchmark with database integration
python test/run_real_webnn_webgpu.py --test-all-models --db-path ./benchmark_db.duckdb

# Store only in database (no JSON output)
python test/run_real_webnn_webgpu.py --test-all-models --db-only
```

## Report Generation

The framework includes comprehensive reporting capabilities:

```bash
# Generate Markdown report for all model classes
python test/run_real_webnn_webgpu.py --test-all-models --platform webgpu --report --format markdown

# Generate HTML report with specific output path
python test/run_real_webnn_webgpu.py --test-all-models --platform webgpu --report --format html --output comprehensive_report.html

# Generate JSON data for custom analysis
python test/run_real_webnn_webgpu.py --test-all-models --platform webgpu --report --format json --output results.json
```

## Comprehensive Testing Script

For comprehensive testing across multiple browsers, models, and precision formats:

```bash
# Run the comprehensive benchmark suite
./test/run_comprehensive_webnn_webgpu_benchmarks.sh

# Run specific subset of tests
./test/run_comprehensive_webnn_webgpu_benchmarks.sh --browsers chrome,firefox --models bert,whisper --precisions int8,int4
```

## Troubleshooting

### Common Issues and Solutions

1. **Browser Launch Failures**
   - Check if the browser is installed and accessible in PATH
   - Verify WebDriver is properly installed and compatible
   - Try running without `--headless` to see browser errors

2. **WebNN/WebGPU Not Available**
   - Ensure you're using a recent browser version with WebNN/WebGPU support
   - Chrome 122+ or Edge recommended for WebNN
   - Chrome, Firefox, or Edge recommended for WebGPU
   - Verify the browser flags are properly set (check logs with `--verbose`)

3. **Model Loading Failures**
   - Check internet connectivity for model downloads
   - Use `--verify-url` and `--fallback-convert` for model verification
   - Check disk space for model caching

4. **Performance Issues**
   - Reduce batch size for memory-constrained environments
   - Try lower precision (int8, int4) for better performance
   - Use browser-specific optimizations (e.g., Firefox for audio models)
   - Enable appropriate optimizations for your model type

### Diagnostic Commands

```bash
# Check browser capabilities
python test/run_real_webnn_webgpu.py --check-capabilities --browser chrome

# Run with verbose logging
python test/run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --verbose

# Test with minimal model for quick verification
python test/run_real_webnn_webgpu.py --quick-test
```

## Best Practices

### Precision Recommendations by Model Type

| Model Type | Recommended Precision | Fallback Precision | Suggested Browser | Browser Optimizations |
|------------|----------------------|-------------------|-------------------|----------------------|
| Text (BERT, T5) | INT8 or INT4 | INT8 | Chrome/Edge | Shader Precompilation |
| Text Generation (LLAMA, Qwen) | INT8 with Mixed | INT8 | Chrome | Shader Precompilation |
| Vision (ViT, CLIP, DETR) | INT8 or INT4 | INT8 | Chrome | Shader Precompilation |
| Audio (Whisper, Wav2Vec2, CLAP) | INT8 | INT8 | Firefox | Compute Shaders |
| Multimodal (LLaVA, XCLIP) | INT8 with Mixed | INT8 | Chrome | Parallel Loading |

### Testing Methodology

1. **Start with Quick Tests**
   - Use `--quick-test` with a tiny model to verify your setup
   - Test a representative model from each category before running all models

2. **Hardware-Specific Testing**
   - Test WebNN on Edge for best native acceleration
   - Test WebGPU on Chrome for general performance
   - Test WebGPU on Firefox for audio models

3. **Precision Scaling**
   - Start with INT8 precision for reliable performance
   - Test INT4 for memory-constrained environments
   - Use mixed precision for complex models like LLaVA and LLAMA

4. **Browser Optimizations**
   - Enable compute shaders for audio models (`--enable-compute-shaders`)
   - Enable shader precompilation for faster startup (`--enable-shader-precompile`)
   - Enable parallel loading for multimodal models (`--enable-parallel-loading`)
   - For best results, use `--all-optimizations` with appropriate browsers

## Conclusion

The enhanced `run_real_webnn_webgpu.py` script now provides a comprehensive, unified entry point for testing all 13 high-priority HuggingFace model classes with various quantization formats on both WebNN and WebGPU. This tool enables accurate performance measurement and comparison across browsers, models, and precision formats, supporting informed decisions for browser-based machine learning deployments.

For more detailed information, see:
- [WebGPU Optimization Guide](./WEB_PLATFORM_OPTIMIZATION_GUIDE.md)
- [WebNN WebGPU Quantization Guide](./WEBNN_WEBGPU_QUANTIZATION_GUIDE.md)
- [Web Platform Documentation](./WEB_PLATFORM_DOCUMENTATION.md)