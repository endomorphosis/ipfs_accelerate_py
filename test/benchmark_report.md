# IPFS Accelerate Python Framework - Benchmark Report

## Report Generated: 2025-03-06 09:46:13

## Summary

This report provides an analysis of benchmark results from the IPFS Accelerate Python Framework database.

### Hardware Platforms

The following hardware platforms are included in the benchmark results:

| Hardware Type | Device Name | Driver Version |
|---------------|-------------|----------------|
| cpu | CPU | N/A |
| cuda | NVIDIA GPU | N/A |
| rocm | AMD GPU | N/A |
| mps | Apple Silicon | N/A |
| openvino | Intel CPU/GPU | N/A |
| webnn | WebNN | N/A |
| webgpu | WebGPU | N/A |
| qualcomm | Qualcomm QNN | N/A |

### Models

The following models are included in the benchmark results:

| Model | Model Family |
|-------|-------------|
| bert-base-uncased | bert |
| t5-small | t5 |
| vit-base-patch16-224 | vit |
| whisper-tiny | whisper |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | llama |

### Performance Results

The following table shows the performance results for each model on different hardware platforms:

| Model | Hardware | Batch Size | Precision | Latency (ms) | Throughput (items/sec) | Memory (MB) |
|-------|----------|------------|-----------|--------------|------------------------|------------|
| bert-base-uncased | cpu | 1 | fp32 | 55.23 | 133.27 | 2874.40 |
| bert-base-uncased | cuda | 1 | fp32 | 93.38 | 121.16 | 3944.93 |
| bert-base-uncased | cuda | 2 | fp32 | 6.08 | 293.61 | 4172.81 |
| bert-base-uncased | cuda | 4 | fp32 | 14.94 | 228.85 | 3712.58 |

## Conclusion

This report provides a snapshot of the current benchmark results. For more detailed analysis, please use the benchmark_db_query.py tool.
