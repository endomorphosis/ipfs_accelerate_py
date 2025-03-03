# Hardware Benchmark Report - 20250302_202226

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 39.88 GB
- **CUDA Version**: 12.4
- **GPU Count**: 1
- **GPUs**:
  - GPU 0: Quadro P4000, 7.92 GB, Compute Capability 6.1
- **Apple Silicon MPS**: Not available
- **OpenVINO Version**: 2025.0.0-17942-1f68be9f594-releases/2025/0

---
## Available Hardware

- **cpu**: ✅ Available
- **cuda**: ✅ Available
- **mps**: ❌ Not available
- **rocm**: ❌ Not available
- **openvino**: ✅ Available

## Benchmark Results

### Text_Generation Models

#### Latency Comparison (ms)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| gpt2 | - | - | - |
| t5-small | - | - | - |
| google/flan-t5-small | - | - | - |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| gpt2 | - | - | - |
| t5-small | - | - | - |
| google/flan-t5-small | - | - | - |

#### gpt2 - Detailed Results

##### CPU

- **Model Load Time**: 12.17 seconds
- **Latency (seconds)**:
  - Min: 0.0000
  - Max: 0.0000
  - Mean: 0.0000
  - Median: 0.0000
- **Throughput (items/sec)**:
  - Min: 0.00
  - Max: 0.00
  - Mean: 0.00
  - Median: 0.00
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:

##### OPENVINO

- **Model Load Time**: 0.00 seconds
- **Latency (seconds)**:
  - Min: 0.0000
  - Max: 0.0000
  - Mean: 0.0000
  - Median: 0.0000
- **Throughput (items/sec)**:
  - Min: 0.00
  - Max: 0.00
  - Mean: 0.00
  - Median: 0.00
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:


#### t5-small - Detailed Results

##### CPU

- **Model Load Time**: 20.90 seconds
- **Latency (seconds)**:
  - Min: 0.0000
  - Max: 0.0000
  - Mean: 0.0000
  - Median: 0.0000
- **Throughput (items/sec)**:
  - Min: 0.00
  - Max: 0.00
  - Mean: 0.00
  - Median: 0.00
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:

##### OPENVINO

- **Model Load Time**: 0.00 seconds
- **Latency (seconds)**:
  - Min: 0.0000
  - Max: 0.0000
  - Mean: 0.0000
  - Median: 0.0000
- **Throughput (items/sec)**:
  - Min: 0.00
  - Max: 0.00
  - Mean: 0.00
  - Median: 0.00
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:


#### google/flan-t5-small - Detailed Results

##### CPU

- **Model Load Time**: 15.25 seconds
- **Latency (seconds)**:
  - Min: 0.0000
  - Max: 0.0000
  - Mean: 0.0000
  - Median: 0.0000
- **Throughput (items/sec)**:
  - Min: 0.00
  - Max: 0.00
  - Mean: 0.00
  - Median: 0.00
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:

##### OPENVINO

- **Model Load Time**: 0.00 seconds
- **Latency (seconds)**:
  - Min: 0.0000
  - Max: 0.0000
  - Mean: 0.0000
  - Median: 0.0000
- **Throughput (items/sec)**:
  - Min: 0.00
  - Max: 0.00
  - Mean: 0.00
  - Median: 0.00
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:

---
