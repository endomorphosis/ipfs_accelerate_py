# Hardware Benchmark Report - 20250303_051018

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 37.09 GB
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
| bert-base-uncased | 1200.88 | - | - |
| distilbert-base-uncased | 245.93 | - | - |
| roberta-base | 448.11 | - | - |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 3.33 | - | - |
| distilbert-base-uncased | 14.57 | - | - |
| roberta-base | 8.37 | - | - |

#### bert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 4.79 seconds
- **Latency (seconds)**:
  - Min: 0.1191
  - Max: 5.1030
  - Mean: 1.7802
  - Median: 1.2009
- **Throughput (items/sec)**:
  - Min: 0.74
  - Max: 13.63
  - Mean: 4.95
  - Median: 3.33
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1191 seconds
    - Throughput: 8.40 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.3759 seconds
    - Throughput: 2.66 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.3516 seconds
    - Throughput: 0.74 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.4251 seconds
    - Throughput: 9.41 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.2009 seconds
    - Throughput: 3.33 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 4.8379 seconds
    - Throughput: 0.83 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.5871 seconds
    - Throughput: 13.63 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.0212 seconds
    - Throughput: 3.96 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 5.1030 seconds
    - Throughput: 1.57 items/second


#### distilbert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 0.22 seconds
- **Latency (seconds)**:
  - Min: 0.0300
  - Max: 2.0190
  - Mean: 0.4869
  - Median: 0.2459
- **Throughput (items/sec)**:
  - Min: 3.55
  - Max: 76.32
  - Mean: 25.14
  - Median: 14.57
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.0300 seconds
    - Throughput: 33.38 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.0686 seconds
    - Throughput: 14.57 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 0.2459 seconds
    - Throughput: 4.07 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.0691 seconds
    - Throughput: 57.91 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 0.2917 seconds
    - Throughput: 13.71 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 1.1274 seconds
    - Throughput: 3.55 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.1048 seconds
    - Throughput: 76.32 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 0.4260 seconds
    - Throughput: 18.78 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 2.0190 seconds
    - Throughput: 3.96 items/second


#### roberta-base - Detailed Results

##### CPU

- **Model Load Time**: 0.23 seconds
- **Latency (seconds)**:
  - Min: 0.0412
  - Max: 5.1529
  - Mean: 1.0574
  - Median: 0.4481
- **Throughput (items/sec)**:
  - Min: 1.55
  - Max: 38.44
  - Mean: 14.24
  - Median: 8.37
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.0412 seconds
    - Throughput: 24.27 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.1194 seconds
    - Throughput: 8.37 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 0.5550 seconds
    - Throughput: 1.80 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.1165 seconds
    - Throughput: 34.35 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 0.4481 seconds
    - Throughput: 8.93 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 1.9141 seconds
    - Throughput: 2.09 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.2081 seconds
    - Throughput: 38.44 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 0.9612 seconds
    - Throughput: 8.32 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 5.1529 seconds
    - Throughput: 1.55 items/second

---
