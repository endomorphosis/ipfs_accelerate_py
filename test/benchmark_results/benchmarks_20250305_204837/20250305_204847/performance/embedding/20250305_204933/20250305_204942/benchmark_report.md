# Hardware Benchmark Report - 20250305_204942

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 23.53 GB
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
| bert-base-uncased | 5037.57 | 5406.79 | 5313.63 |
| distilbert-base-uncased | 2602.82 | 2807.03 | 2833.82 |
| roberta-base | 2215.98 | 2186.38 | 2174.62 |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| bert-base-uncased | 0.79 | 0.74 | 0.75 |
| distilbert-base-uncased | 1.36 | 1.32 | 1.26 |
| roberta-base | 1.79 | 1.83 | 1.79 |

#### bert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 10.19 seconds
- **Latency (seconds)**:
  - Min: 0.7530
  - Max: 45.3353
  - Mean: 10.7748
  - Median: 5.0376
- **Throughput (items/sec)**:
  - Min: 0.14
  - Max: 2.38
  - Mean: 0.87
  - Median: 0.79
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.7530 seconds
    - Throughput: 1.33 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 2.3117 seconds
    - Throughput: 0.43 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 6.9156 seconds
    - Throughput: 0.14 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 2.5577 seconds
    - Throughput: 1.56 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 5.0376 seconds
    - Throughput: 0.79 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 20.7163 seconds
    - Throughput: 0.19 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 3.3633 seconds
    - Throughput: 2.38 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 9.9824 seconds
    - Throughput: 0.80 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 45.3353 seconds
    - Throughput: 0.18 items/second

##### OPENVINO

- **Model Load Time**: 10.20 seconds
- **Latency (seconds)**:
  - Min: 0.7393
  - Max: 45.1682
  - Mean: 10.8890
  - Median: 5.3136
- **Throughput (items/sec)**:
  - Min: 0.16
  - Max: 2.19
  - Mean: 0.85
  - Median: 0.75
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.7393 seconds
    - Throughput: 1.35 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 2.3718 seconds
    - Throughput: 0.42 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 6.4249 seconds
    - Throughput: 0.16 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 2.5227 seconds
    - Throughput: 1.59 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 5.3136 seconds
    - Throughput: 0.75 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 21.9197 seconds
    - Throughput: 0.18 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 3.6547 seconds
    - Throughput: 2.19 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 9.8861 seconds
    - Throughput: 0.81 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 45.1682 seconds
    - Throughput: 0.18 items/second

##### CUDA

- **Model Load Time**: 10.19 seconds
- **Latency (seconds)**:
  - Min: 0.8076
  - Max: 44.9849
  - Mean: 10.8906
  - Median: 5.4068
- **Throughput (items/sec)**:
  - Min: 0.15
  - Max: 2.27
  - Mean: 0.86
  - Median: 0.74
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.8076 seconds
    - Throughput: 1.24 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 2.4603 seconds
    - Throughput: 0.41 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 6.5229 seconds
    - Throughput: 0.15 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 2.1928 seconds
    - Throughput: 1.82 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 5.4068 seconds
    - Throughput: 0.74 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 21.5852 seconds
    - Throughput: 0.19 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 3.5291 seconds
    - Throughput: 2.27 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 10.5261 seconds
    - Throughput: 0.76 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 44.9849 seconds
    - Throughput: 0.18 items/second


#### distilbert-base-uncased - Detailed Results

##### CPU

- **Model Load Time**: 0.33 seconds
- **Latency (seconds)**:
  - Min: 0.4556
  - Max: 10.0029
  - Mean: 3.4902
  - Median: 2.6028
- **Throughput (items/sec)**:
  - Min: 0.32
  - Max: 4.30
  - Mean: 1.94
  - Median: 1.36
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.4556 seconds
    - Throughput: 2.19 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.9912 seconds
    - Throughput: 1.01 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 3.1541 seconds
    - Throughput: 0.32 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 1.0502 seconds
    - Throughput: 3.81 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 2.9344 seconds
    - Throughput: 1.36 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 10.0029 seconds
    - Throughput: 0.40 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.8598 seconds
    - Throughput: 4.30 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.6028 seconds
    - Throughput: 3.07 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 8.3609 seconds
    - Throughput: 0.96 items/second

##### CUDA

- **Model Load Time**: 0.33 seconds
- **Latency (seconds)**:
  - Min: 0.4076
  - Max: 10.6842
  - Mean: 3.5225
  - Median: 2.8070
- **Throughput (items/sec)**:
  - Min: 0.31
  - Max: 6.75
  - Mean: 2.15
  - Median: 1.32
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.4076 seconds
    - Throughput: 2.45 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.9767 seconds
    - Throughput: 1.02 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 3.2019 seconds
    - Throughput: 0.31 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 1.2112 seconds
    - Throughput: 3.30 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 3.0190 seconds
    - Throughput: 1.32 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 10.6842 seconds
    - Throughput: 0.37 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.1854 seconds
    - Throughput: 6.75 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.8070 seconds
    - Throughput: 2.85 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 8.2098 seconds
    - Throughput: 0.97 items/second

##### OPENVINO

- **Model Load Time**: 0.33 seconds
- **Latency (seconds)**:
  - Min: 0.4462
  - Max: 10.2378
  - Mean: 3.4658
  - Median: 2.8338
- **Throughput (items/sec)**:
  - Min: 0.32
  - Max: 6.90
  - Mean: 2.14
  - Median: 1.26
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.4462 seconds
    - Throughput: 2.24 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.9926 seconds
    - Throughput: 1.01 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 3.1441 seconds
    - Throughput: 0.32 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 1.2094 seconds
    - Throughput: 3.31 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 3.1800 seconds
    - Throughput: 1.26 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 10.2378 seconds
    - Throughput: 0.39 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.1592 seconds
    - Throughput: 6.90 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 2.8338 seconds
    - Throughput: 2.82 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 7.9892 seconds
    - Throughput: 1.00 items/second


#### roberta-base - Detailed Results

##### CPU

- **Model Load Time**: 0.25 seconds
- **Latency (seconds)**:
  - Min: 0.2245
  - Max: 16.5922
  - Mean: 4.1903
  - Median: 2.2160
- **Throughput (items/sec)**:
  - Min: 0.44
  - Max: 6.49
  - Mean: 2.51
  - Median: 1.79
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.2245 seconds
    - Throughput: 4.45 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.6789 seconds
    - Throughput: 1.47 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 2.2377 seconds
    - Throughput: 0.45 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.6162 seconds
    - Throughput: 6.49 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 2.2160 seconds
    - Throughput: 1.81 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 9.1656 seconds
    - Throughput: 0.44 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.5241 seconds
    - Throughput: 5.25 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.4572 seconds
    - Throughput: 1.79 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 16.5922 seconds
    - Throughput: 0.48 items/second

##### CUDA

- **Model Load Time**: 0.25 seconds
- **Latency (seconds)**:
  - Min: 0.2126
  - Max: 17.0346
  - Mean: 4.2364
  - Median: 2.1864
- **Throughput (items/sec)**:
  - Min: 0.43
  - Max: 6.35
  - Mean: 2.53
  - Median: 1.83
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.2126 seconds
    - Throughput: 4.70 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.6809 seconds
    - Throughput: 1.47 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 2.2137 seconds
    - Throughput: 0.45 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.6299 seconds
    - Throughput: 6.35 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 2.1864 seconds
    - Throughput: 1.83 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 9.2646 seconds
    - Throughput: 0.43 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.5216 seconds
    - Throughput: 5.26 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.3831 seconds
    - Throughput: 1.83 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 17.0346 seconds
    - Throughput: 0.47 items/second

##### OPENVINO

- **Model Load Time**: 0.25 seconds
- **Latency (seconds)**:
  - Min: 0.2138
  - Max: 17.3194
  - Mean: 4.2837
  - Median: 2.1746
- **Throughput (items/sec)**:
  - Min: 0.43
  - Max: 6.37
  - Mean: 2.52
  - Median: 1.79
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_seq_32**:
    - Avg Latency: 0.2138 seconds
    - Throughput: 4.68 items/second
  - **batch_1_seq_128**:
    - Avg Latency: 0.6815 seconds
    - Throughput: 1.47 items/second
  - **batch_1_seq_512**:
    - Avg Latency: 2.2109 seconds
    - Throughput: 0.45 items/second
  - **batch_4_seq_32**:
    - Avg Latency: 0.6277 seconds
    - Throughput: 6.37 items/second
  - **batch_4_seq_128**:
    - Avg Latency: 2.1746 seconds
    - Throughput: 1.84 items/second
  - **batch_4_seq_512**:
    - Avg Latency: 9.3218 seconds
    - Throughput: 0.43 items/second
  - **batch_8_seq_32**:
    - Avg Latency: 1.5397 seconds
    - Throughput: 5.20 items/second
  - **batch_8_seq_128**:
    - Avg Latency: 4.4643 seconds
    - Throughput: 1.79 items/second
  - **batch_8_seq_512**:
    - Avg Latency: 17.3194 seconds
    - Throughput: 0.46 items/second

#### Relative Performance (vs CPU)

| Hardware | Speedup Factor |
|---|---|
| CUDA | 0.99x |
| OPENVINO | 0.99x |

---
