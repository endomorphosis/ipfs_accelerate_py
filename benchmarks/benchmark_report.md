# Comprehensive Benchmark Timing Report

Generated: 2025-03-06 18:40:01

## Overview

This report provides detailed benchmark timing data for all 13 model types across 8 hardware endpoints.

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

## Model Performance

### Text Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| BERT (Text embedding model) | 25.62ms / 45.10it/s | 8.46ms / 223.33it/s | 8.77ms / 198.83it/s | 15.71ms / 114.66it/s | 12.24ms / 158.85it/s | 17.13ms / 102.82it/s | 18.49ms / 87.06it/s | 15.31ms / 130.58it/s |
| T5 (Text-to-text generation model) | 34.87ms / 34.06it/s | 9.62ms / 168.71it/s | 11.05ms / 152.89it/s | 20.61ms / 93.98it/s | 16.38ms / 123.48it/s | 22.20ms / 83.97it/s | 22.83ms / 79.10it/s | 19.38ms / 102.84it/s |
| LLAMA (Large language model) | 88.48ms / 13.16it/s | 17.61ms / 86.08it/s | 21.58ms / 77.71it/s | 40.90ms / 33.41it/s | 38.78ms / 41.38it/s | 70.58ms / 18.80it/s | 98.79ms / 16.26it/s | 77.94ms / 21.13it/s |
| Qwen2 (Advanced text generation model) | 94.49ms / 10.07it/s | 20.41ms / 73.96it/s | 23.84ms / 60.95it/s | 50.42ms / 28.10it/s | 40.84ms / 37.05it/s | 82.77ms / 15.87it/s | 111.44ms / 12.23it/s | 79.60ms / 18.19it/s |

### Multimodal Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| CLIP (Vision-text multimodal model) | 28.93ms / 38.05it/s | 8.15ms / 179.22it/s | 9.40ms / 167.43it/s | 18.90ms / 106.82it/s | 16.21ms / 142.22it/s | 25.02ms / 72.61it/s | 27.18ms / 67.68it/s | 17.70ms / 107.04it/s |
| LLaVA (Vision-language model) | 106.02ms / 8.24it/s | 26.70ms / 65.18it/s | 28.53ms / 56.54it/s | 55.03ms / 23.37it/s | 44.24ms / 31.18it/s | 86.57ms / 13.04it/s | 111.34ms / 11.43it/s | 96.09ms / 16.14it/s |
| LLaVA-Next (Advanced vision-language model) | 107.82ms / 7.61it/s | 26.15ms / 57.57it/s | 30.81ms / 51.23it/s | 56.69ms / 20.18it/s | 50.33ms / 28.81it/s | 89.39ms / 12.90it/s | 116.39ms / 10.22it/s | 106.76ms / 14.95it/s |

### Vision Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| ViT (Vision transformer model) | 26.63ms / 42.85it/s | 7.50ms / 196.74it/s | 8.40ms / 178.90it/s | 16.35ms / 102.37it/s | 13.38ms / 148.00it/s | 22.88ms / 93.25it/s | 24.45ms / 86.63it/s | 16.30ms / 110.24it/s |
| XCLIP (Video-text multimodal model) | 40.03ms / 28.79it/s | 9.11ms / 145.46it/s | 10.64ms / 145.89it/s | 22.06ms / 82.20it/s | 20.15ms / 109.50it/s | 32.82ms / 52.06it/s | 38.12ms / 51.52it/s | 23.55ms / 81.09it/s |
| DETR (DEtection TRansformer for object detection) | 39.93ms / 24.81it/s | 11.30ms / 133.65it/s | 13.16ms / 116.35it/s | 23.85ms / 74.23it/s | 23.06ms / 93.52it/s | 37.20ms / 48.74it/s | 42.91ms / 41.31it/s | 28.07ms / 69.62it/s |

### Audio Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| CLAP (Audio-text multimodal model) | 31.06ms / 37.41it/s | 8.48ms / 191.07it/s | 9.73ms / 168.91it/s | 15.73ms / 108.17it/s | 14.56ms / 150.15it/s | 26.65ms / 71.89it/s | 30.67ms / 59.93it/s | 16.95ms / 137.89it/s |
| Wav2Vec2 (Speech recognition model) | 34.39ms / 33.88it/s | 8.73ms / 168.07it/s | 10.38ms / 168.00it/s | 17.15ms / 99.80it/s | 15.26ms / 140.21it/s | 26.21ms / 70.95it/s | 32.15ms / 54.71it/s | 18.06ms / 125.81it/s |
| Whisper (Speech recognition model) | 35.39ms / 33.80it/s | 8.57ms / 172.35it/s | 9.67ms / 142.73it/s | 20.82ms / 91.18it/s | 17.50ms / 133.73it/s | 26.35ms / 62.76it/s | 33.94ms / 50.05it/s | 18.99ms / 125.33it/s |

## Optimization Recommendations

- Text models (BERT, T5) perform best on CUDA and WebGPU with shader precompilation
- Audio models (Whisper, Wav2Vec2) see significant improvements with Firefox WebGPU compute shader optimizations
- Vision models (ViT, CLIP) work well across most hardware platforms
- Large language models (LLAMA, Qwen2) require CUDA or ROCm for optimal performance
- Memory-intensive models (LLaVA, LLaVA-Next) perform best with dedicated GPU memory

## Conclusion

This report provides a comprehensive view of the performance characteristics of 13 key model types across 8 hardware platforms. Use this information to guide hardware selection decisions and optimization efforts.
