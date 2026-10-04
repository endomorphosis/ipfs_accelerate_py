# Medium Priority Models Implementation Plan

**Date:** March 22, 2025

## Overview

With the completion of all 32 critical priority models, we now shift our focus to implementing the remaining medium-priority models. This document outlines the plan for achieving 100% coverage of medium-priority models by May 1, 2025, as specified in the roadmap.

## Current Status

- **Total Models Tracked:** 198
- **Implemented Models:** 147 (74.2%)
- **Missing Models:** 51 (25.8%)
- **Medium-Priority Models Implemented:** ~77 of 139 (55.4%)
- **Medium-Priority Models Remaining:** ~62 of 139 (44.6%)

## Implementation Approach

We will maintain the successful pattern established in the previous implementations:

1. **Batch Processing**: Implement models in batches of 10, grouped by architecture type
2. **Template-Based Generation**: Use the appropriate template for each model architecture
3. **Automated Testing**: Verify all generated test files through both syntax and runtime checks
4. **Documentation Updates**: Update the roadmap and create reports after each batch completion

## Batched Implementation Plan

### Batch 1: Decoder-Only Models (10 models)

High-impact decoder-only models to implement first:

```json
{
  "decoder_only": [
    {
      "name": "codegen",
      "template": "gpt2",
      "task": "text-generation"
    },
    {
      "name": "command_r",
      "template": "gpt2",
      "task": "text-generation",
      "original_name": "command-r"
    },
    {
      "name": "gemma2",
      "template": "gemma",
      "task": "text-generation"
    },
    {
      "name": "gemma3",
      "template": "gemma",
      "task": "text-generation"
    },
    {
      "name": "llama_3",
      "template": "llama",
      "task": "text-generation",
      "original_name": "llama-3"
    },
    {
      "name": "mamba",
      "template": "gpt2",
      "task": "text-generation"
    },
    {
      "name": "mistral_next",
      "template": "mistral",
      "task": "text-generation",
      "original_name": "mistral-next"
    },
    {
      "name": "nemotron",
      "template": "gpt2",
      "task": "text-generation"
    },
    {
      "name": "olmo",
      "template": "gpt2",
      "task": "text-generation"
    },
    {
      "name": "olmoe",
      "template": "gpt2",
      "task": "text-generation"
    }
  ]
}
```

### Batch 2: Encoder-Decoder and Encoder-Only Models (10 models)

```json
{
  "encoder_decoder": [
    {
      "name": "m2m_100",
      "template": "t5",
      "task": "translation",
      "original_name": "m2m-100"
    },
    {
      "name": "seamless_m4t",
      "template": "t5",
      "task": "translation",
      "original_name": "seamless-m4t"
    },
    {
      "name": "switch_transformers",
      "template": "t5",
      "task": "text2text-generation",
      "original_name": "switch-transformers"
    },
    {
      "name": "umt5",
      "template": "t5",
      "task": "text2text-generation"
    }
  ],
  "encoder_only": [
    {
      "name": "convbert",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "data2vec_text",
      "template": "bert",
      "task": "fill-mask",
      "original_name": "data2vec-text"
    },
    {
      "name": "deberta_v2",
      "template": "deberta",
      "task": "fill-mask",
      "original_name": "deberta-v2"
    },
    {
      "name": "esm",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "flaubert",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "ibert",
      "template": "bert",
      "task": "fill-mask"
    }
  ]
}
```

### Batch 3: Vision and Vision-Text Models (10 models)

```json
{
  "vision": [
    {
      "name": "beit3",
      "template": "vit",
      "task": "image-classification"
    },
    {
      "name": "conditional_detr",
      "template": "detr",
      "task": "object-detection",
      "original_name": "conditional-detr"
    },
    {
      "name": "convnextv2",
      "template": "convnext",
      "task": "image-classification"
    },
    {
      "name": "cvt",
      "template": "vit",
      "task": "image-classification"
    },
    {
      "name": "depth_anything",
      "template": "vit",
      "task": "depth-estimation",
      "original_name": "depth-anything"
    },
    {
      "name": "dinat",
      "template": "vit",
      "task": "image-classification"
    },
    {
      "name": "dino",
      "template": "vit",
      "task": "image-classification"
    }
  ],
  "vision_text": [
    {
      "name": "instructblip",
      "template": "blip",
      "task": "image-to-text"
    },
    {
      "name": "vision_encoder_decoder",
      "template": "vision_text_dual_encoder",
      "task": "image-to-text",
      "original_name": "vision-encoder-decoder"
    },
    {
      "name": "xclip",
      "template": "clip",
      "task": "video-classification"
    }
  ]
}
```

### Batch 4: Multimodal and Speech Models (10 models)

```json
{
  "multimodal": [
    {
      "name": "idefics2",
      "template": "idefics",
      "task": "image-to-text"
    },
    {
      "name": "idefics3",
      "template": "idefics",
      "task": "image-to-text"
    },
    {
      "name": "llava_next_video",
      "template": "llava",
      "task": "video-to-text",
      "original_name": "llava-next-video"
    },
    {
      "name": "mllama",
      "template": "llava",
      "task": "image-to-text"
    },
    {
      "name": "qwen2_vl",
      "template": "llava",
      "task": "image-to-text",
      "original_name": "qwen2-vl"
    },
    {
      "name": "qwen3_vl",
      "template": "llava",
      "task": "image-to-text",
      "original_name": "qwen3-vl"
    },
    {
      "name": "siglip",
      "template": "clip",
      "task": "image-classification"
    }
  ],
  "speech": [
    {
      "name": "speech_to_text",
      "template": "whisper",
      "task": "automatic-speech-recognition",
      "original_name": "speech-to-text"
    },
    {
      "name": "speech_to_text_2",
      "template": "whisper",
      "task": "automatic-speech-recognition",
      "original_name": "speech-to-text-2"
    },
    {
      "name": "wav2vec2_conformer",
      "template": "wav2vec2",
      "task": "automatic-speech-recognition",
      "original_name": "wav2vec2-conformer"
    }
  ]
}
```

### Batch 5: Remaining Decoder-Only and Encoder-Only Models (10 models)

```json
{
  "decoder_only": [
    {
      "name": "openai_gpt",
      "template": "gpt2",
      "task": "text-generation",
      "original_name": "openai-gpt"
    },
    {
      "name": "persimmon",
      "template": "gpt2",
      "task": "text-generation"
    },
    {
      "name": "phi3",
      "template": "phi",
      "task": "text-generation"
    },
    {
      "name": "phi4",
      "template": "phi",
      "task": "text-generation"
    },
    {
      "name": "recurrent_gemma",
      "template": "gemma",
      "task": "text-generation",
      "original_name": "recurrent-gemma"
    }
  ],
  "encoder_only": [
    {
      "name": "megatron_bert",
      "template": "bert",
      "task": "fill-mask",
      "original_name": "megatron-bert"
    },
    {
      "name": "mobilebert",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "mra",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "nezha",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "nystromformer",
      "template": "bert",
      "task": "fill-mask"
    }
  ]
}
```

### Batch 6: Final Medium-Priority Models (12 models)

```json
{
  "decoder_only": [
    {
      "name": "rwkv",
      "template": "gpt2",
      "task": "text-generation"
    },
    {
      "name": "stablelm",
      "template": "gpt2",
      "task": "text-generation"
    },
    {
      "name": "starcoder2",
      "template": "gpt2",
      "task": "text-generation"
    }
  ],
  "encoder_only": [
    {
      "name": "splinter",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "xlm",
      "template": "bert",
      "task": "fill-mask"
    },
    {
      "name": "xlm_roberta_xl",
      "template": "xlm_roberta",
      "task": "fill-mask",
      "original_name": "xlm-roberta-xl"
    },
    {
      "name": "xmod",
      "template": "bert",
      "task": "fill-mask"
    }
  ],
  "vision": [
    {
      "name": "imagegpt",
      "template": "vit",
      "task": "image-classification"
    },
    {
      "name": "mobilenet_v1",
      "template": "vit",
      "task": "image-classification",
      "original_name": "mobilenet-v1"
    },
    {
      "name": "swinv2",
      "template": "swin",
      "task": "image-classification"
    },
    {
      "name": "van",
      "template": "vit",
      "task": "image-classification"
    },
    {
      "name": "vitdet",
      "template": "vit",
      "task": "object-detection"
    }
  ]
}
```

## Implementation Workflow

For each batch:

1. **Create JSON Configuration**:
   - Define model details including name, template, task, and original_name (for hyphenated names)
   - Save to `batch_X_models.json` where X is the batch number

2. **Generate Test Files**:
   ```bash
   python generate_batch_models.py --batch-file batch_X_models.json --output-dir fixed_tests
   ```

3. **Verify Syntax**:
   ```bash
   python verify_syntax.py --directory fixed_tests
   ```

4. **Run Basic Tests**:
   ```bash
   python run_basic_tests.py --batch-file batch_X_models.json --cpu-only
   ```

5. **Update Documentation**:
   - Generate a report for the batch implementation
   - Update the roadmap with the new implementation status
   - Update coverage statistics

## Timeline

| Batch | Target Completion | Models |
|-------|------------------|---------|
| Batch 1 | April 5, 2025 | 10 decoder-only models |
| Batch 2 | April 10, 2025 | 10 encoder-decoder and encoder-only models |
| Batch 3 | April 15, 2025 | 10 vision and vision-text models |
| Batch 4 | April 20, 2025 | 10 multimodal and speech models |
| Batch 5 | April 25, 2025 | 10 remaining decoder-only and encoder-only models |
| Batch 6 | May 1, 2025 | 12 final medium-priority models |

## Success Criteria

The implementation of medium-priority models will be considered complete when:

1. All 62 remaining medium-priority models have test files
2. All test files pass syntax validation
3. At least 25% of the medium-priority models pass tests with real weights
4. The roadmap is updated with implementation status
5. All hyphenated names are properly handled
6. Medium-priority model coverage reaches 100%

## Conclusion

This plan outlines a systematic approach to implementing the remaining medium-priority models by May 1, 2025. By following this structured batched implementation, we will make steady progress toward our goal of 100% test coverage for all HuggingFace models.
