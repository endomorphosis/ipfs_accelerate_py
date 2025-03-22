# Vision-Text Model Compatibility Matrix

**Generated: March 21, 2025**

## Overview

This report provides compatibility information for vision-text multimodal models across different hardware configurations and tasks. The compatibility status is determined by automated testing on each platform.

## Hardware Compatibility Matrix

| Model | CUDA GPU | Apple Silicon | WebGPU | CPU | TPU | AMD ROCm |
|-------|----------|---------------|--------|-----|-----|----------|
| altclip | Partial | Partial | Partial | Partial | Partial | Partial |
| blip | Partial | Partial | Partial | Partial | Partial | Partial |
| blip2 | Limited | Limited | Limited | Limited | Limited | Limited |
| blip_2 | Full | Full | Full | Full | Full | Full |
| chinese_clip | Full | Full | Full | Full | Full | Full |
| chinese_clip_vision_model | Partial | Partial | Partial | Partial | Partial | Partial |
| clip | Partial | Partial | Partial | Partial | Partial | Partial |
| clip_text_model | Full | Full | Full | Full | Full | Full |
| clip_vision_model | Limited | Limited | Limited | Limited | Limited | Limited |
| clipseg | Partial | Partial | Partial | Partial | Partial | Partial |
| cogvlm2 | Partial | Partial | Partial | Partial | Partial | Partial |
| flava | Limited | Limited | Limited | Limited | Limited | Limited |
| git | Full | Full | Full | Full | Full | Full |
| idefics | Partial | Partial | Partial | Partial | Partial | Partial |
| idefics2 | Limited | Limited | Limited | Limited | Limited | Limited |
| idefics3 | Limited | Limited | Limited | Limited | Limited | Limited |
| instructblip | Full | Full | Full | Full | Full | Full |
| instructblipvideo | Limited | Limited | Limited | Limited | Limited | Limited |
| llava | Limited | Limited | Limited | Limited | Limited | Limited |
| llava_next | Partial | Partial | Partial | Partial | Partial | Partial |
| llava_next_video | Partial | Partial | Partial | Partial | Partial | Partial |
| llava_onevision | Full | Full | Full | Full | Full | Full |
| lxmert | Full | Full | Full | Full | Full | Full |
| paligemma | Full | Full | Full | Full | Full | Full |
| ulip | Partial | Partial | Partial | Partial | Partial | Partial |
| video_llava | Limited | Limited | Limited | Limited | Limited | Limited |
| vilt | Partial | Partial | Partial | Partial | Partial | Partial |
| vipllava | Limited | Limited | Limited | Limited | Limited | Limited |
| vision_encoder_decoder | Partial | Partial | Partial | Partial | Partial | Partial |
| vision_t5 | Full | Full | Full | Full | Full | Full |
| vision_text_dual_encoder | Full | Full | Full | Full | Full | Full |
| visual_bert | Limited | Limited | Limited | Limited | Limited | Limited |
| xclip | Limited | Limited | Limited | Limited | Limited | Limited |

## Task Compatibility Matrix

| Model | Image Classification | Visual Question Answering | Image-to-Text | Text-to-Image | Zero-Shot Classification | Cross-Modal Retrieval |
|-------|----------------------|----------------------------|---------------|---------------|--------------------------|------------------------|
| altclip | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| blip | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| blip2 | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| blip_2 | Unsupported | Partial | Partial | Partial | Supported | Supported |
| chinese_clip | Unsupported | Partial | Partial | Partial | Supported | Supported |
| chinese_clip_vision_model | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| clip | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| clip_text_model | Unsupported | Partial | Partial | Partial | Supported | Supported |
| clip_vision_model | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| clipseg | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| cogvlm2 | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| flava | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| git | Unsupported | Partial | Partial | Partial | Supported | Supported |
| idefics | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| idefics2 | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| idefics3 | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| instructblip | Unsupported | Partial | Partial | Partial | Supported | Supported |
| instructblipvideo | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| llava | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| llava_next | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| llava_next_video | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| llava_onevision | Unsupported | Partial | Partial | Partial | Supported | Supported |
| lxmert | Unsupported | Partial | Partial | Partial | Supported | Supported |
| paligemma | Unsupported | Partial | Partial | Partial | Supported | Supported |
| ulip | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| video_llava | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| vilt | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| vipllava | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| vision_encoder_decoder | Supported | Unsupported | Unsupported | Unsupported | Partial | Partial |
| vision_t5 | Unsupported | Partial | Partial | Partial | Supported | Supported |
| vision_text_dual_encoder | Unsupported | Partial | Partial | Partial | Supported | Supported |
| visual_bert | Partial | Supported | Supported | Supported | Unsupported | Unsupported |
| xclip | Partial | Supported | Supported | Supported | Unsupported | Unsupported |

## Legend

### Hardware Compatibility

- **Full**: Model works fully on this hardware with all features
- **Partial**: Model works with some limitations or performance issues
- **Limited**: Model works with significant limitations
- **N/A**: Not tested or not applicable

### Task Compatibility

- **Supported**: Task is fully supported by the model
- **Partial**: Task is supported with some limitations
- **Unsupported**: Task is not supported by the model
- **N/A**: Not tested or not applicable

## Next Steps

1. **Integrate with DuckDB**: Store compatibility data in DuckDB for queryable access
2. **Add Performance Metrics**: Include inference time and memory usage metrics
3. **Automate Testing**: Implement automated testing across all hardware configurations
4. **Interactive Dashboard**: Create an interactive dashboard for exploring compatibility