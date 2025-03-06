# Hardware Benchmark Report - 20250305_195321

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 31.43 GB
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

### Embedding Models

#### Latency Comparison (ms)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 1590.78 | - | - |
| distilbert-base-uncased | 1027.16 | - | - |
| roberta-base | 1367.76 | - | - |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 2.44 | - | - |
| distilbert-base-uncased | 3.62 | - | - |
| roberta-base | 2.36 | - | - |

#### bert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 5.71 seconds
- **Latency (seconds)**:
  - Min: 0.4093
  - Max: 15.6714
  - Mean: 3.6025
  - Median: 1.5908
- **Throughput (items/sec)**:
  - Min: 0.35
  - Max: 7.59
  - Mean: 2.55
  - Median: 2.44
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.4093 seconds
    - Throughput: 2.44 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.9499 seconds
    - Throughput: 1.05 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 2.8758 seconds
    - Throughput: 0.35 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.7633 seconds
    - Throughput: 5.24 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.5908 seconds
    - Throughput: 2.51 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 6.0008 seconds
    - Throughput: 0.67 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.0534 seconds
    - Throughput: 7.59 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 3.1079 seconds
    - Throughput: 2.57 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 15.6714 seconds
    - Throughput: 0.51 items/second


#### distilbert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 0.21 seconds
- **Latency (seconds)**:
  - Min: 0.1040
  - Max: 7.7083
  - Mean: 1.8518
  - Median: 1.0272
- **Throughput (items/sec)**:
  - Min: 0.83
  - Max: 17.33
  - Mean: 5.62
  - Median: 3.62
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1040 seconds
    - Throughput: 9.62 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.5138 seconds
    - Throughput: 1.95 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.2106 seconds
    - Throughput: 0.83 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.3650 seconds
    - Throughput: 10.96 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.0272 seconds
    - Throughput: 3.89 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 3.0666 seconds
    - Throughput: 1.30 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.4615 seconds
    - Throughput: 17.33 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.2091 seconds
    - Throughput: 3.62 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 7.7083 seconds
    - Throughput: 1.04 items/second


#### roberta-base - Detailed Results

##### CPU

- **Model Load Time**: 0.24 seconds
- **Latency (seconds)**:
  - Min: 0.1740
  - Max: 16.9418
  - Mean: 3.5924
  - Median: 1.3678
- **Throughput (items/sec)**:
  - Min: 0.47
  - Max: 8.13
  - Mean: 3.46
  - Median: 2.36
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1740 seconds
    - Throughput: 5.75 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.4243 seconds
    - Throughput: 2.36 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.4037 seconds
    - Throughput: 0.71 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.4967 seconds
    - Throughput: 8.05 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.3678 seconds
    - Throughput: 2.92 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 6.9119 seconds
    - Throughput: 0.58 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.9842 seconds
    - Throughput: 8.13 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 3.6272 seconds
    - Throughput: 2.21 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 16.9418 seconds
    - Throughput: 0.47 items/second

---
