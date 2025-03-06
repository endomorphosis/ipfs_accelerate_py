# Hardware Benchmark Report - 20250305_215053

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 31.40 GB
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
- **openvino**: ✅ Available

## Benchmark Results

### Vision Models

#### Latency Comparison (ms)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| google/vit-base-patch16-224 | 1555.60 | - | - |
| microsoft/resnet-50 | 352.23 | - | - |
| facebook/convnext-tiny-224 | 388.45 | - | - |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| google/vit-base-patch16-224 | 3.66 | - | - |
| microsoft/resnet-50 | 11.36 | - | - |
| facebook/convnext-tiny-224 | 8.70 | - | - |

#### google/vit-base-patch16-224 - Detailed Results

##### CPU

- **Model Load Time**: 9.98 seconds
- **Latency (seconds)**:
  - Min: 1.0941
  - Max: 3.3357
  - Mean: 1.8114
  - Median: 1.5556
- **Throughput (items/sec)**:
  - Min: 0.43
  - Max: 6.70
  - Mean: 3.10
  - Median: 3.66
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_img_224x224**:
    - Avg Latency: 2.3521 seconds
    - Throughput: 0.43 items/second
  - **batch_1_img_384x384**:
    - Avg Latency: 1.5556 seconds
    - Throughput: 0.64 items/second
  - **batch_4_img_224x224**:
    - Avg Latency: 3.3357 seconds
    - Throughput: 1.20 items/second
  - **batch_4_img_384x384**:
    - Avg Latency: 1.0941 seconds
    - Throughput: 3.66 items/second
  - **batch_8_img_224x224**:
    - Avg Latency: 1.3368 seconds
    - Throughput: 5.98 items/second
  - **batch_8_img_384x384**:
    - Avg Latency: 1.1938 seconds
    - Throughput: 6.70 items/second


#### microsoft/resnet-50 - Detailed Results

##### CPU

- **Model Load Time**: 5.82 seconds
- **Latency (seconds)**:
  - Min: 0.1084
  - Max: 0.7921
  - Mean: 0.3852
  - Median: 0.3522
- **Throughput (items/sec)**:
  - Min: 8.67
  - Max: 13.50
  - Mean: 10.87
  - Median: 11.36
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_img_224x224**:
    - Avg Latency: 0.1084 seconds
    - Throughput: 9.22 items/second
  - **batch_1_img_384x384**:
    - Avg Latency: 0.1154 seconds
    - Throughput: 8.67 items/second
  - **batch_4_img_224x224**:
    - Avg Latency: 0.2963 seconds
    - Throughput: 13.50 items/second
  - **batch_4_img_384x384**:
    - Avg Latency: 0.3522 seconds
    - Throughput: 11.36 items/second
  - **batch_8_img_224x224**:
    - Avg Latency: 0.7921 seconds
    - Throughput: 10.10 items/second
  - **batch_8_img_384x384**:
    - Avg Latency: 0.6468 seconds
    - Throughput: 12.37 items/second


#### facebook/convnext-tiny-224 - Detailed Results

##### CPU

- **Model Load Time**: 6.63 seconds
- **Latency (seconds)**:
  - Min: 0.1149
  - Max: 1.0177
  - Mean: 0.4989
  - Median: 0.3884
- **Throughput (items/sec)**:
  - Min: 7.86
  - Max: 10.91
  - Mean: 9.03
  - Median: 8.70
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_img_224x224**:
    - Avg Latency: 0.1149 seconds
    - Throughput: 8.70 items/second
  - **batch_1_img_384x384**:
    - Avg Latency: 0.1204 seconds
    - Throughput: 8.30 items/second
  - **batch_4_img_224x224**:
    - Avg Latency: 0.3884 seconds
    - Throughput: 10.30 items/second
  - **batch_4_img_384x384**:
    - Avg Latency: 0.3665 seconds
    - Throughput: 10.91 items/second
  - **batch_8_img_224x224**:
    - Avg Latency: 1.0177 seconds
    - Throughput: 7.86 items/second
  - **batch_8_img_384x384**:
    - Avg Latency: 0.9853 seconds
    - Throughput: 8.12 items/second

---
