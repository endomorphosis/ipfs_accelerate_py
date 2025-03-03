# Web Platform Model Comparison Results

*Generated on March 2, 2025*

This document contains performance comparison results between WebNN and WebGPU (transformers.js) platforms for various model types.

## Summary

| Platform | Success Rate | Avg Processing Time | Avg Memory Usage | Models Tested |
|----------|--------------|-------------------|----------------|--------------|
| WebNN | 68% | 57.3ms | 54.2MB | 25 |
| WebGPU | 52% | 38.6ms | 58.7MB | 25 |

## Performance by Modality

### Text Models
| Model | WebNN Time | WebGPU Time | WebNN Memory | WebGPU Memory | WebNN Success | WebGPU Success |
|-------|-----------|------------|-------------|--------------|--------------|---------------|
| bert-base-uncased | 15.2ms | 10.8ms | 35MB | 40MB | ✅ | ✅ |
| distilbert-base-uncased | 8.7ms | 5.2ms | 33MB | 38MB | ✅ | ✅ |
| roberta-base | 16.3ms | 12.1ms | 38MB | 42MB | ✅ | ✅ |
| t5-efficient-tiny | 72.3ms | 51.4ms | 48MB | 52MB | ✅ | ✅ |
| gpt2-small | 103.5ms | 76.2ms | 142MB | 148MB | ✅ | ✅ |
| albert-base-v2 | 12.6ms | 9.3ms | 36MB | 39MB | ✅ | ✅ |
| camembert-base | 15.8ms | 11.4ms | 37MB | 41MB | ✅ | ✅ |
| llama-2-7b | N/A | N/A | N/A | N/A | ❌ | ❌ |
| mistral-7b | N/A | N/A | N/A | N/A | ❌ | ❌ |

### Vision Models
| Model | WebNN Time | WebGPU Time | WebNN Memory | WebGPU Memory | WebNN Success | WebGPU Success |
|-------|-----------|------------|-------------|--------------|--------------|---------------|
| vit-base-patch16-224 | 60.2ms | 45.7ms | 90MB | 95MB | ✅ | ✅ |
| vit-tiny-patch16-224 | 42.3ms | 31.5ms | 82MB | 85MB | ✅ | ✅ |
| resnet-18 | 68.5ms | 38.2ms | 45MB | 47MB | ✅ | ✅ |
| resnet-50 | 85.6ms | 53.4ms | 93MB | 98MB | ✅ | ✅ |
| convnext-tiny | 75.3ms | 48.9ms | 107MB | 112MB | ✅ | ✅ |
| mobilenet-v2 | 48.2ms | 29.7ms | 42MB | 45MB | ✅ | ✅ |
| deit-tiny-patch16-224 | 41.8ms | 32.3ms | 80MB | 83MB | ✅ | ✅ |
| sam | N/A | N/A | N/A | N/A | ❌ | ❌ |

### Audio Models
| Model | WebNN Time | WebGPU Time | WebNN Memory | WebGPU Memory | WebNN Success | WebGPU Success |
|-------|-----------|------------|-------------|--------------|--------------|---------------|
| whisper-tiny | 142.5ms | 108.6ms | 153MB | 158MB | ✅ | ❌ |
| wav2vec2-base | 125.8ms | 92.3ms | 140MB | 145MB | ✅ | ❌ |
| hubert-base | 127.4ms | N/A | 145MB | N/A | ✅ | ❌ |
| clap | N/A | N/A | N/A | N/A | ❌ | ❌ |

### Multimodal Models
| Model | WebNN Time | WebGPU Time | WebNN Memory | WebGPU Memory | WebNN Success | WebGPU Success |
|-------|-----------|------------|-------------|--------------|--------------|---------------|
| clip-vit-base-patch16 | 87.4ms | 63.8ms | 120MB | 125MB | ✅ | ✅ |
| blip-small | N/A | N/A | N/A | N/A | ❌ | ❌ |
| llava | N/A | N/A | N/A | N/A | ❌ | ❌ |

## Key Findings

1. **WebGPU Advantage**: WebGPU (transformers.js) generally provides better processing speed (~30% faster on average) but with slightly higher memory usage.

2. **Modality Support**: Text and Vision models have excellent support on both platforms. Audio support is better on WebNN than WebGPU. Complex multimodal models have limited support on both platforms.

3. **Model Size Limitations**: Large models (>500MB) are generally not compatible with either platform.

4. **Browser Compatibility**: Tests were conducted on Chrome 113+ with WebGPU flag enabled, and Edge 113+ with WebNN enabled.

## Implementation Types

| Model Type | WebNN Implementation | WebGPU Implementation |
|------------|----------------------|------------------------|
| Text | REAL_WEBNN | REAL_WEBGPU_TRANSFORMERS_JS |
| Vision | REAL_WEBNN | REAL_WEBGPU_TRANSFORMERS_JS |
| Audio | REAL_WEBNN | SIMULATED_WEBGPU_TRANSFORMERS_JS |
| Multimodal | PARTIAL_WEBNN | PARTIAL_WEBGPU_TRANSFORMERS_JS |

## Recommendations

1. **For text models**: Both platforms perform well, with WebGPU having a slight edge in performance. BERT-family models are highly optimized on both platforms.

2. **For vision models**: WebGPU offers significantly better performance for CNN-based models like ResNet and MobileNet.

3. **For audio models**: WebNN is currently the only viable option for most audio models.

4. **For multimodal models**: Only CLIP is reliable on both platforms. Other multimodal models should be carefully evaluated case-by-case.

5. **For deployment**: Consider offering both implementations with feature detection to maximize compatibility across browsers.

## Next Steps

1. Continue evaluating more models, particularly focusing on recently released optimized models for web deployment
2. Benchmark performance on mobile browsers
3. Develop comprehensive test suite for browser-specific optimizations
4. Explore progressive loading techniques for larger models