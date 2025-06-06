# IPFS Accelerate Python Examples

This directory contains example applications demonstrating the usage of IPFS Accelerate Python with various hardware backends, model types, and configurations.

## Available Examples

### WebNN/WebGPU Integration

- **[demo_webnn_webgpu.py](demo_webnn_webgpu.py)**: Demonstrates the WebNN/WebGPU integration with IPFS acceleration with support for various model types, browsers, and configurations.

Usage:
```bash
# Run BERT with WebGPU in Chrome
python demo_webnn_webgpu.py --model bert-base-uncased --platform webgpu --browser chrome

# Run ViT with WebGPU in Firefox
python demo_webnn_webgpu.py --model vit-base-patch16-224 --platform webgpu --browser firefox

# Run Whisper with WebGPU in Firefox (optimized for audio)
python demo_webnn_webgpu.py --model openai/whisper-small --platform webgpu --browser firefox

# Run BERT with WebNN in Edge
python demo_webnn_webgpu.py --model bert-base-uncased --platform webnn --browser edge --precision 16 --mixed-precision

# Run benchmark with multiple runs
python demo_webnn_webgpu.py --model bert-base-uncased --platform webgpu --runs 5 --save benchmark_results.json

# Run with real browser (non-headless)
python demo_webnn_webgpu.py --model bert-base-uncased --platform webgpu --browser chrome --real-browser
```

### Example Applications (Coming Soon)

The following examples are planned for future development:

- **[multi_model_pipeline.py](multi_model_pipeline.py)**: Pipeline multiple models with shared tensors for efficient inference
- **[ipfs_content_addressing.py](ipfs_content_addressing.py)**: Advanced IPFS content addressing for model storage and retrieval
- **[hardware_benchmark.py](hardware_benchmark.py)**: Benchmark models across multiple hardware platforms
- **[cross_browser_model_sharding.py](cross_browser_model_sharding.py)**: Distribute model components across multiple browsers

## Getting Started

Before running the examples, ensure you have the required dependencies installed:

```bash
# For WebNN/WebGPU examples
pip install -e ".[webnn]"

# For visualization examples
pip install -e ".[viz]"
```

## Additional Resources

For more information on IPFS Accelerate Python, see the following resources:

- [Main README](../README.md)
- [WebNN/WebGPU README](../WEBNN_WEBGPU_README.md)
- [API Documentation](../docs/API.md)
- [Hardware Optimization Guide](../docs/HARDWARE.md)
- [IPFS Integration Guide](../docs/IPFS.md)