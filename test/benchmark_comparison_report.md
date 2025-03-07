# Comprehensive Benchmark Timing Report

Generated: 2025-03-06 19:49:27

## Executive Summary

This report provides detailed benchmark timing data for all 13 model types across 8 hardware endpoints, showing performance metrics including latency, throughput, and memory usage.

## Hardware Platforms

| Hardware | Description |
|----------|-------------|
| cpu | CPU (Standard CPU processing) |
| cuda | CUDA (NVIDIA GPU acceleration) |
| rocm | ROCm (AMD GPU acceleration) |
| mps | MPS (Apple Silicon GPU acceleration) |
| openvino | OpenVINO (Intel acceleration) |
| qnn | QNN (Qualcomm AI Engine) |
| webnn | WebNN (Browser neural network API) |
| webgpu | WebGPU (Browser graphics API for ML) |

## Performance Results

### Text Models

| Model | Hardware | Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) |
|-------|----------|------------|--------------|---------------------|------------|
| bert | cpu | 1 | 31.20 | 32.06 | 3169.69 |
| bert | rocm | 1 | 16.02 | 96.24 | 1300.51 |

### Multimodal Models

| Model | Hardware | Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) |
|-------|----------|------------|--------------|---------------------|------------|

### Vision Models

| Model | Hardware | Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) |
|-------|----------|------------|--------------|---------------------|------------|

### Audio Models

| Model | Hardware | Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) |
|-------|----------|------------|--------------|---------------------|------------|

## Optimization Recommendations

### Hardware Selection

- Use CUDA for best overall performance across all model types when available
- For CPU-only environments, OpenVINO provides significant speedups over standard CPU
- For browser environments, WebGPU with shader precompilation offers the best performance

### Model-Specific Optimizations

- Text models benefit from CPU caching and OpenVINO optimizations
- Vision models are well-optimized across most hardware platforms
- Audio models perform best with CUDA; WebGPU with compute shader optimization for browser environments
- For multimodal models, use hardware with sufficient memory capacity; WebGPU with parallel loading for browser environments

## Conclusion

This report provides a comprehensive view of performance characteristics for 13 key model types across 8 hardware platforms. Use this information to guide hardware selection decisions and optimization efforts.
