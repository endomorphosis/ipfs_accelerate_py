# Targeted WebNN and WebGPU Quantization Test Results

Date: Fri Mar  7 01:32:00 AM PST 2025

## Results Table

| Model | Platform | Bits | Mixed Precision | Status | Simulation | Time (ms) | Memory (MB) |
|-------|----------|------|----------------|--------|------------|-----------|-------------|
| BERT | webgpu | 16bit | No | ✅ Success | Yes | 42.28952077156278 | 486.1031705175609 |
| BERT | webgpu | 4bit | No | ✅ Success | Yes | 37.568491845208314 | 220.37226724275473 |
| BERT | webgpu | 4bit | Yes | ✅ Success | Yes | 46.7239666249183 | 468.7053318288196 |
| BERT | webgpu | 8bit | No | ✅ Success | Yes | 33.58244330564387 | 385.91822713273405 |
| BERT | webnn | 16bit | No | ✅ Success | Yes | 42.68968677198467 | 442.51912289683185 |
| BERT | webnn | 8bit | No | ✅ Success | Yes | 36.238656656865906 | 345.481807053426 |
| CLIP | webgpu | 16bit | No | ✅ Success | Yes | 42.59907787850818 | 413.79948552732935 |
| CLIP | webgpu | 8bit | No | ✅ Success | Yes | 32.18206166091214 | 211.55094891086404 |
| ViT | webgpu | 16bit | No | ✅ Success | Yes | 33.487909509721526 | 361.82717914022624 |
| ViT | webgpu | 4bit | Yes | ✅ Success | Yes | 49.408674052350676 | 209.33478697287373 |
| ViT | webgpu | 8bit | No | ✅ Success | Yes | 49.88449971243112 | 293.5321681922883 |
| Whisper | webgpu | 16bit | No | ✅ Success | Yes | 48.661611015030395 | 325.40432914794354 |
| Whisper | webgpu | 8bit | No | ✅ Success | Yes | 39.0244878081259 | 406.71765378693794 |
| Whisper | webgpu | firefox | No | ❌ Failed | No | N/A | N/A |

## Summary

- Total tests: 14
- Successful tests: 13 (92.85%)
- Failed tests: 1 (7.14%)
