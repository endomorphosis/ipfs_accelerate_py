# Hardware Benchmark Report - 20250305_195054

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 33.59 GB
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
| bert-base-uncased | 1391.15 | 1399.99 | 1405.58 |
| distilbert-base-uncased | 1162.34 | 1207.06 | 1193.26 |
| roberta-base | 2234.90 | 2232.37 | 2261.85 |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 2.05 | 2.03 | 2.01 |
| distilbert-base-uncased | 3.44 | 3.31 | 3.35 |
| roberta-base | 1.79 | 1.79 | 1.77 |

#### bert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 3.60 seconds
- **Latency (seconds)**:
  - Min: 0.1661
  - Max: 18.0551
  - Mean: 4.0161
  - Median: 1.3911
- **Throughput (items/sec)**:
  - Min: 0.44
  - Max: 10.28
  - Mean: 3.44
  - Median: 2.05
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1661 seconds
    - Throughput: 6.02 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.4879 seconds
    - Throughput: 2.05 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.5581 seconds
    - Throughput: 0.64 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.3892 seconds
    - Throughput: 10.28 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.3911 seconds
    - Throughput: 2.88 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 8.6881 seconds
    - Throughput: 0.46 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.2858 seconds
    - Throughput: 6.22 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.1236 seconds
    - Throughput: 1.94 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 18.0551 seconds
    - Throughput: 0.44 items/second

##### OPENVINO

- **Model Load Time**: 3.60 seconds
- **Latency (seconds)**:
  - Min: 0.1678
  - Max: 17.8241
  - Mean: 4.0307
  - Median: 1.4056
- **Throughput (items/sec)**:
  - Min: 0.44
  - Max: 9.93
  - Mean: 3.39
  - Median: 2.01
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1678 seconds
    - Throughput: 5.96 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.4977 seconds
    - Throughput: 2.01 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.5511 seconds
    - Throughput: 0.64 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.4028 seconds
    - Throughput: 9.93 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.4056 seconds
    - Throughput: 2.85 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 9.0491 seconds
    - Throughput: 0.44 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.2718 seconds
    - Throughput: 6.29 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.1064 seconds
    - Throughput: 1.95 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 17.8241 seconds
    - Throughput: 0.45 items/second

##### CUDA

- **Model Load Time**: 3.60 seconds
- **Latency (seconds)**:
  - Min: 0.1664
  - Max: 17.8744
  - Mean: 4.0574
  - Median: 1.4000
- **Throughput (items/sec)**:
  - Min: 0.44
  - Max: 10.13
  - Mean: 3.39
  - Median: 2.03
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1664 seconds
    - Throughput: 6.01 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.4922 seconds
    - Throughput: 2.03 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.5448 seconds
    - Throughput: 0.65 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.3950 seconds
    - Throughput: 10.13 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.4000 seconds
    - Throughput: 2.86 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 9.1673 seconds
    - Throughput: 0.44 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.3209 seconds
    - Throughput: 6.06 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.1554 seconds
    - Throughput: 1.93 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 17.8744 seconds
    - Throughput: 0.45 items/second


#### distilbert-base-uncased - Detailed Results

##### CUDA

- **Model Load Time**: 0.35 seconds
- **Latency (seconds)**:
  - Min: 0.1589
  - Max: 8.9471
  - Mean: 2.1487
  - Median: 1.2071
- **Throughput (items/sec)**:
  - Min: 0.66
  - Max: 12.71
  - Mean: 4.44
  - Median: 3.31
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1589 seconds
    - Throughput: 6.29 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.4307 seconds
    - Throughput: 2.32 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.5116 seconds
    - Throughput: 0.66 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.4496 seconds
    - Throughput: 8.90 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.2071 seconds
    - Throughput: 3.31 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 3.9079 seconds
    - Throughput: 1.02 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.6293 seconds
    - Throughput: 12.71 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.0960 seconds
    - Throughput: 3.82 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 8.9471 seconds
    - Throughput: 0.89 items/second

##### CPU

- **Model Load Time**: 0.35 seconds
- **Latency (seconds)**:
  - Min: 0.1481
  - Max: 8.9969
  - Mean: 2.1605
  - Median: 1.1623
- **Throughput (items/sec)**:
  - Min: 0.70
  - Max: 12.70
  - Mean: 4.49
  - Median: 3.44
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1481 seconds
    - Throughput: 6.75 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.4396 seconds
    - Throughput: 2.27 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.4214 seconds
    - Throughput: 0.70 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.4533 seconds
    - Throughput: 8.82 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.1623 seconds
    - Throughput: 3.44 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 4.0922 seconds
    - Throughput: 0.98 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.6298 seconds
    - Throughput: 12.70 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.1010 seconds
    - Throughput: 3.81 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 8.9969 seconds
    - Throughput: 0.89 items/second

##### OPENVINO

- **Model Load Time**: 0.35 seconds
- **Latency (seconds)**:
  - Min: 0.1525
  - Max: 8.9171
  - Mean: 2.1751
  - Median: 1.1933
- **Throughput (items/sec)**:
  - Min: 0.65
  - Max: 12.59
  - Mean: 4.43
  - Median: 3.35
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.1525 seconds
    - Throughput: 6.56 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.4647 seconds
    - Throughput: 2.15 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 1.5485 seconds
    - Throughput: 0.65 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.4463 seconds
    - Throughput: 8.96 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 1.1933 seconds
    - Throughput: 3.35 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 4.0457 seconds
    - Throughput: 0.99 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 0.6354 seconds
    - Throughput: 12.59 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.1720 seconds
    - Throughput: 3.68 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 8.9171 seconds
    - Throughput: 0.90 items/second


#### roberta-base - Detailed Results

##### CPU

- **Model Load Time**: 0.45 seconds
- **Latency (seconds)**:
  - Min: 0.2970
  - Max: 19.6022
  - Mean: 4.4706
  - Median: 2.2349
- **Throughput (items/sec)**:
  - Min: 0.39
  - Max: 6.69
  - Mean: 2.37
  - Median: 1.79
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.2970 seconds
    - Throughput: 3.37 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.8165 seconds
    - Throughput: 1.22 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 2.5565 seconds
    - Throughput: 0.39 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.7858 seconds
    - Throughput: 5.09 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 2.2349 seconds
    - Throughput: 1.79 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 8.5488 seconds
    - Throughput: 0.47 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.1958 seconds
    - Throughput: 6.69 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.1977 seconds
    - Throughput: 1.91 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 19.6022 seconds
    - Throughput: 0.41 items/second

##### CUDA

- **Model Load Time**: 0.45 seconds
- **Latency (seconds)**:
  - Min: 0.2952
  - Max: 19.5862
  - Mean: 4.4941
  - Median: 2.2324
- **Throughput (items/sec)**:
  - Min: 0.38
  - Max: 6.73
  - Mean: 2.38
  - Median: 1.79
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.2952 seconds
    - Throughput: 3.39 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.7988 seconds
    - Throughput: 1.25 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 2.6266 seconds
    - Throughput: 0.38 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.7771 seconds
    - Throughput: 5.15 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 2.2324 seconds
    - Throughput: 1.79 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 8.5997 seconds
    - Throughput: 0.47 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.1895 seconds
    - Throughput: 6.73 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.3418 seconds
    - Throughput: 1.84 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 19.5862 seconds
    - Throughput: 0.41 items/second

##### OPENVINO

- **Model Load Time**: 0.45 seconds
- **Latency (seconds)**:
  - Min: 0.3017
  - Max: 19.7448
  - Mean: 4.4993
  - Median: 2.2618
- **Throughput (items/sec)**:
  - Min: 0.38
  - Max: 6.68
  - Mean: 2.36
  - Median: 1.77
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.3017 seconds
    - Throughput: 3.31 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.7680 seconds
    - Throughput: 1.30 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 2.6328 seconds
    - Throughput: 0.38 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.7991 seconds
    - Throughput: 5.01 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 2.2618 seconds
    - Throughput: 1.77 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 8.5551 seconds
    - Throughput: 0.47 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.1983 seconds
    - Throughput: 6.68 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.2318 seconds
    - Throughput: 1.89 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 19.7448 seconds
    - Throughput: 0.41 items/second

#### Relative Performance (vs CPU)

| Hardware | Speedup Factor |
|---|---|
| CUDA | 1.00x |
| OPENVINO | 0.99x |

---
