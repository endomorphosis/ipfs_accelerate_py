# Hardware Benchmark Report - 20250305_202703

## System Information

- **Platform**: Linux-6.8.0-11-generic-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.3
- **CPU Cores**: 4 physical cores, 8 logical cores
- **System Memory**: 62.21 GB total
- **Available Memory**: 28.87 GB
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

### Audio Models

#### Latency Comparison (ms)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| openai/whisper-tiny | - | - | - |
| facebook/wav2vec2-base | 10564.67 | - | - |

#### Throughput Comparison (items/sec)

| Model | CPU | CUDA | OPENVINO |
|---|---|---|---|
| openai/whisper-tiny | - | - | - |
| facebook/wav2vec2-base | 0.19 | - | - |

#### openai/whisper-tiny - Detailed Results

##### CPU

- **Model Load Time**: 9.29 seconds
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


#### facebook/wav2vec2-base - Detailed Results

##### CPU

- **Model Load Time**: 1.24 seconds
- **Latency (seconds)**:
  - Min: 2.9734
  - Max: 76.0406
  - Mean: 32.6499
  - Median: 10.5647
- **Throughput (items/sec)**:
  - Min: 0.06
  - Max: 0.43
  - Mean: 0.22
  - Median: 0.19
- **Memory Usage (MB)**:
  - Min Allocated: 0.00
  - Max Allocated: 0.00
  - Mean Allocated: 0.00
- **Benchmark Configurations**:
  - **batch_1_audio_5s**:
    - Avg Latency: 2.9734 seconds
    - Throughput: 0.34 items/second
    - Real-time Factor: 1.68x
  - **batch_1_audio_10s**:
    - Avg Latency: 4.3101 seconds
    - Throughput: 0.23 items/second
    - Real-time Factor: 2.32x
  - **batch_1_audio_30s**:
    - Avg Latency: 10.1823 seconds
    - Throughput: 0.10 items/second
    - Real-time Factor: 2.95x
  - **batch_4_audio_5s**:
    - Avg Latency: 9.3369 seconds
    - Throughput: 0.43 items/second
    - Real-time Factor: 2.14x
  - **batch_4_audio_10s**:
    - Avg Latency: 10.5647 seconds
    - Throughput: 0.38 items/second
    - Real-time Factor: 3.79x
  - **batch_4_audio_30s**:
    - Avg Latency: 72.3652 seconds
    - Throughput: 0.06 items/second
    - Real-time Factor: 1.66x
  - **batch_8_audio_5s**:
    - Avg Latency: 41.9638 seconds
    - Throughput: 0.19 items/second
    - Real-time Factor: 0.95x
  - **batch_8_audio_10s**:
    - Avg Latency: 76.0406 seconds
    - Throughput: 0.11 items/second
    - Real-time Factor: 1.05x
  - **batch_8_audio_30s**:
    - Avg Latency: 66.1126 seconds
    - Throughput: 0.12 items/second
    - Real-time Factor: 3.63x

---
