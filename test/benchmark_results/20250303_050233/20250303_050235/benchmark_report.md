# Hardware Benchmark Report - 20250303_050235

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 37.02 GB
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
| bert-base-uncased | 523.79 | - | - |
| distilbert-base-uncased | 299.17 | - | - |
| roberta-base | 601.60 | - | - |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 7.24 | - | - |
| distilbert-base-uncased | 12.53 | - | - |
| roberta-base | 4.89 | - | - |

#### bert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 2.84 seconds
- **Latency (seconds)**:
  - Min: 0.0537
  - Max: 4.9288
  - Mean: 1.1021
  - Median: 0.5238
- **Throughput (items/sec)**:
  - Min: 1.62
  - Max: 33.54
  - Mean: 12.18
  - Median: 7.24
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.0537 seconds
    - Throughput: 18.64 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.1429 seconds
    - Throughput: 7.00 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 0.5612 seconds
    - Throughput: 1.78 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.1318 seconds
    - Throughput: 30.36 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 0.5238 seconds
    - Throughput: 7.64 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 2.2340 seconds
    - Throughput: 1.79 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.2386 seconds
    - Throughput: 33.54 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 1.1045 seconds
    - Throughput: 7.24 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 4.9288 seconds
    - Throughput: 1.62 items/second


#### distilbert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 0.21 seconds
- **Latency (seconds)**:
  - Min: 0.0263
  - Max: 2.4724
  - Mean: 0.6000
  - Median: 0.2992
- **Throughput (items/sec)**:
  - Min: 2.92
  - Max: 50.78
  - Mean: 20.33
  - Median: 12.53
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.0263 seconds
    - Throughput: 38.04 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.0833 seconds
    - Throughput: 12.00 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 0.2992 seconds
    - Throughput: 3.34 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.0861 seconds
    - Throughput: 46.43 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 0.3192 seconds
    - Throughput: 12.53 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 1.3719 seconds
    - Throughput: 2.92 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.1575 seconds
    - Throughput: 50.78 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 0.5840 seconds
    - Throughput: 13.70 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 2.4724 seconds
    - Throughput: 3.24 items/second


#### roberta-base - Detailed Results

##### CPU

- **Model Load Time**: 0.54 seconds
- **Latency (seconds)**:
  - Min: 0.0728
  - Max: 8.5011
  - Mean: 1.8774
  - Median: 0.6016
- **Throughput (items/sec)**:
  - Min: 0.94
  - Max: 20.14
  - Mean: 7.42
  - Median: 4.89
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.0728 seconds
    - Throughput: 13.73 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.2045 seconds
    - Throughput: 4.89 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 0.6795 seconds
    - Throughput: 1.47 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.1986 seconds
    - Throughput: 20.14 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 0.6016 seconds
    - Throughput: 6.65 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 4.0136 seconds
    - Throughput: 1.00 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.5702 seconds
    - Throughput: 14.03 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.0549 seconds
    - Throughput: 3.89 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 8.5011 seconds
    - Throughput: 0.94 items/second

---
