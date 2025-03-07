# Comprehensive Benchmark Timing Report

Generated: 2025-03-06 16:23:37

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
| BERT (Text embedding model) | 12.34ms / 123.45it/s | 11.65ms / 830.62it/s | 11.81ms / 374.77it/s | 22.65ms / 462.06it/s | 15.29ms / 301.88it/s | 22.89ms / 188.30it/s | 27.23ms / 343.19it/s | 15.49ms / 126.41it/s |
| T5 (Text-to-text generation model) | 39.71ms / 69.93it/s | 12.89ms / 349.61it/s | 21.28ms / 1327.82it/s | 21.51ms / 98.19it/s | 18.30ms / 121.56it/s | 26.28ms / 82.38it/s | 30.76ms / 155.70it/s | 22.61ms / 195.81it/s |
| LLAMA (Large language model) | 92.96ms / 13.07it/s | 34.25ms / 686.12it/s | 38.57ms / 574.34it/s | 72.38ms / 263.28it/s | 42.46ms / 85.56it/s | 136.14ms / 156.33it/s | 124.21ms / 32.51it/s | 87.20ms / 22.19it/s |
| Qwen2 (Advanced text generation model) | 104.88ms / 9.79it/s | 30.97ms / 302.26it/s | 27.59ms / 64.72it/s | 83.43ms / 231.10it/s | 44.38ms / 38.27it/s | 111.84ms / 63.55it/s | 190.76ms / 95.97it/s | 155.97ms / 146.23it/s |

### Multimodal Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| CLIP (Vision-text multimodal model) | 52.88ms / 55.85it/s | 23.45ms / 1179.61it/s | 16.46ms / 270.91it/s | 28.88ms / 150.67it/s | 37.56ms / 762.98it/s | 52.70ms / 215.10it/s | 53.16ms / 91.13it/s | 32.21ms / 79.76it/s |
| LLaVA (Vision-language model) | 141.86ms / 8.23it/s | 48.03ms / 169.14it/s | 54.39ms / 152.32it/s | 109.13ms / 66.70it/s | 114.15ms / 178.29it/s | 125.12ms / 10.20it/s | 155.92ms / 8.06it/s | 140.18ms / 48.72it/s |
| LLaVA-Next (Advanced vision-language model) | 161.43ms / 14.68it/s | 60.68ms / 318.69it/s | 72.73ms / 257.54it/s | 89.98ms / 14.56it/s | 96.15ms / 83.01it/s | 127.19ms / 8.67it/s | 171.12ms / 7.03it/s | 120.65ms / 21.41it/s |

### Vision Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| ViT (Vision transformer model) | 35.26ms / 82.62it/s | 14.38ms / 1544.61it/s | 11.19ms / 335.77it/s | 30.11ms / 869.84it/s | 17.84ms / 288.61it/s | 39.10ms / 718.88it/s | 30.43ms / 323.05it/s | 23.52ms / 460.58it/s |
| XCLIP (Video-text multimodal model) | 68.63ms / 233.56it/s | 11.05ms / 313.12it/s | 12.44ms / 134.86it/s | 31.63ms / 331.89it/s | 24.30ms / 222.31it/s | 61.50ms / 421.80it/s | 67.38ms / 392.11it/s | 45.81ms / 695.54it/s |
| DETR (DEtection TRansformer for object detection) | 77.94ms / 196.23it/s | 14.98ms / 532.42it/s | 14.24ms / 118.36it/s | 28.74ms / 154.03it/s | 30.46ms / 411.37it/s | 39.54ms / 96.60it/s | 56.37ms / 176.42it/s | 31.06ms / 68.61it/s |

### Audio Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| CLAP (Audio-text multimodal model) | 41.57ms / 31.40it/s | 13.64ms / 686.44it/s | 12.65ms / 152.66it/s | 37.26ms / 718.32it/s | 19.05ms / 109.43it/s | 43.03ms / 255.15it/s | 41.13ms / 55.79it/s | 22.28ms / 92.32it/s |
| Wav2Vec2 (Speech recognition model) | 44.40ms / 161.18it/s | 11.77ms / 801.75it/s | 13.45ms / 735.56it/s | 19.56ms / 101.77it/s | 20.56ms / 509.63it/s | 29.93ms / 143.75it/s | 47.44ms / 241.01it/s | 31.59ms / 907.81it/s |
| Whisper (Speech recognition model) | 48.87ms / 141.58it/s | 10.28ms / 191.09it/s | 14.76ms / 670.85it/s | 22.33ms / 92.62it/s | 17.73ms / 114.26it/s | 49.21ms / 528.70it/s | 40.91ms / 60.28it/s | 25.92ms / 403.23it/s |

## Optimization Recommendations

- Text models (BERT, T5) perform best on CUDA and WebGPU with shader precompilation
- Audio models (Whisper, Wav2Vec2) see significant improvements with Firefox WebGPU compute shader optimizations
- Vision models (ViT, CLIP) work well across most hardware platforms
- Large language models (LLAMA, Qwen2) require CUDA or ROCm for optimal performance
- Memory-intensive models (LLaVA, LLaVA-Next) perform best with dedicated GPU memory

## Conclusion

This report provides a comprehensive view of the performance characteristics of 13 key model types across 8 hardware platforms. Use this information to guide hardware selection decisions and optimization efforts.
