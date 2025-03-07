
# ⚠️ SIMULATION WARNING ⚠️

**This report contains data from simulated hardware that may not reflect actual performance.**

The following hardware platforms were simulated:
- WEBGPU 
- WEBNN
- ROCM
- MPS
- QNN
- OPENVINO

Simulated results should be treated as approximations and not used for critical performance decisions without validation on actual hardware.

---


# ⚠️ WARNING: POTENTIALLY MISLEADING DATA ⚠️

**This report may contain simulated benchmark results that are presented as real hardware data.**

Issue: May contain simulation results presented as real data

*Marked as problematic by cleanup_stale_reports.py on 2025-03-06 19:14:42*

---

# Comprehensive Benchmark Timing Report

Generated: 2025-03-06 18:42:34

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
| BERT (Text embedding model) | 25.78ms / 46.93it/s | 8.12ms / 218.74it/s | 9.12ms / 185.68it/s | 14.42ms / 116.37it/s | 12.47ms / 152.33it/s | 17.58ms / 103.31it/s | 20.03ms / 83.96it/s | 14.70ms / 128.40it/s |
| T5 (Text-to-text generation model) | 30.54ms / 34.13it/s | 10.35ms / 172.04it/s | 12.05ms / 149.16it/s | 18.99ms / 93.94it/s | 16.90ms / 120.91it/s | 24.04ms / 83.15it/s | 23.62ms / 72.37it/s | 19.35ms / 107.62it/s |
| LLAMA (Large language model) | 91.71ms / 12.72it/s | 17.63ms / 89.93it/s | 21.39ms / 76.45it/s | 40.65ms / 31.85it/s | 40.09ms / 40.56it/s | 80.89ms / 20.29it/s | 101.92ms / 16.14it/s | 73.61ms / 20.99it/s |
| Qwen2 (Advanced text generation model) | 90.26ms / 9.28it/s | 20.70ms / 73.80it/s | 25.01ms / 66.78it/s | 50.13ms / 27.34it/s | 40.99ms / 39.65it/s | 84.88ms / 15.58it/s | 110.79ms / 12.50it/s | 80.11ms / 18.11it/s |

### Multimodal Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| CLIP (Vision-text multimodal model) | 32.18ms / 37.58it/s | 7.83ms / 186.24it/s | 9.39ms / 165.09it/s | 18.97ms / 96.19it/s | 15.17ms / 138.35it/s | 25.59ms / 76.01it/s | 25.45ms / 69.05it/s | 17.81ms / 107.30it/s |
| LLaVA (Vision-language model) | 105.02ms / 9.00it/s | 25.70ms / 63.25it/s | 29.14ms / 52.51it/s | 53.30ms / 24.85it/s | 48.51ms / 34.38it/s | 92.36ms / 14.50it/s | 117.20ms / 11.02it/s | 95.58ms / 15.14it/s |
| LLaVA-Next (Advanced vision-language model) | 112.85ms / 6.90it/s | 28.38ms / 59.67it/s | 31.13ms / 51.83it/s | 56.80ms / 21.13it/s | 47.30ms / 29.78it/s | 97.15ms / 11.75it/s | 128.57ms / 9.22it/s | 106.90ms / 14.04it/s |

### Vision Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| ViT (Vision transformer model) | 29.52ms / 43.33it/s | 7.86ms / 195.07it/s | 9.21ms / 168.26it/s | 16.92ms / 108.26it/s | 13.19ms / 139.64it/s | 22.89ms / 87.85it/s | 24.43ms / 87.24it/s | 15.82ms / 108.94it/s |
| XCLIP (Video-text multimodal model) | 38.61ms / 26.78it/s | 9.03ms / 153.45it/s | 10.89ms / 140.62it/s | 19.90ms / 88.42it/s | 20.06ms / 116.13it/s | 31.01ms / 56.18it/s | 36.33ms / 49.20it/s | 24.00ms / 85.99it/s |
| DETR (DEtection TRansformer for object detection) | 41.41ms / 23.21it/s | 11.85ms / 124.97it/s | 13.12ms / 117.60it/s | 24.62ms / 77.70it/s | 23.06ms / 104.27it/s | 36.01ms / 44.76it/s | 38.72ms / 42.96it/s | 28.96ms / 69.18it/s |

### Audio Models

| Model | cpu | cuda | rocm | mps | openvino | qnn | webnn | webgpu |
|-------|----|-----|-----|----|---------|----|------|-------|
| CLAP (Audio-text multimodal model) | 29.57ms / 38.70it/s | 8.43ms / 175.83it/s | 9.45ms / 176.60it/s | 17.53ms / 110.01it/s | 15.28ms / 140.59it/s | 24.11ms / 72.80it/s | 33.46ms / 58.51it/s | 16.85ms / 137.57it/s |
| Wav2Vec2 (Speech recognition model) | 31.58ms / 34.93it/s | 9.00ms / 188.00it/s | 10.20ms / 162.45it/s | 18.76ms / 102.41it/s | 15.92ms / 128.35it/s | 26.48ms / 71.90it/s | 33.50ms / 53.41it/s | 17.16ms / 121.66it/s |
| Whisper (Speech recognition model) | 34.32ms / 32.20it/s | 9.51ms / 169.96it/s | 10.86ms / 154.83it/s | 20.34ms / 90.65it/s | 17.82ms / 135.12it/s | 27.23ms / 63.75it/s | 38.57ms / 47.22it/s | 20.36ms / 124.94it/s |

## Optimization Recommendations

- Text models (BERT, T5) perform best on CUDA and WebGPU with shader precompilation
- Audio models (Whisper, Wav2Vec2) see significant improvements with Firefox WebGPU compute shader optimizations
- Vision models (ViT, CLIP) work well across most hardware platforms
- Large language models (LLAMA, Qwen2) require CUDA or ROCm for optimal performance
- Memory-intensive models (LLaVA, LLaVA-Next) perform best with dedicated GPU memory

## Conclusion

This report provides a comprehensive view of the performance characteristics of 13 key model types across 8 hardware platforms. Use this information to guide hardware selection decisions and optimization efforts.
