# Model Test Coverage Report

Generated on: 2025-03-22

## Summary

Total test files: 309
Total model classes with from_pretrained() tested: 309 (100%)

### From_pretrained() Testing Implementation Methods

| Method | Count | Percentage |
|--------|-------|------------|
| Explicit test_from_pretrained method | 226 | 73.1% |
| Alternative test method names | 38 | 12.3% |
| Direct calls to from_pretrained | 15 | 4.9% |
| Pipeline API usage (implicit calls) | 30 | 9.7% |
| **Total** | **309** | **100%** |

## Coverage by Architecture Type

### encoder-only (35 models)

- `albert`
- `audio`
- `bert`
- `camembert`
- `canine`
- `deberta`
- `deberta_v2`
- `decoder_only`
- `distilbert`
- `electra`
- `encoder_decoder`
- `encoder_only`
- `ernie`
- `funnel`
- `gptj`
- `hubert`
- `layoutlm`
- `multimodal`
- `rembert`
- `roberta`
- `roberta`
- `roformer`
- `seamless_m4t`
- `speech_to_text`
- `speech_to_text_2`
- `squeezebert`
- `tapas`
- `transfo-xl`
- `trocr_base`
- `trocr_large`
- `vision`
- `wav2vec2_bert`
- `xlm_roberta`
- `xlm_roberta`
- `xlnet`

### decoder-only (16 models)

- `bloom`
- `ctrl`
- `falcon`
- `gemma`
- `gpt2`
- `gpt_j`
- `gpt_neo`
- `gpt_neox`
- `llama`
- `llama`
- `mistral`
- `mixtral`
- `mpt`
- `opt`
- `paligemma`
- `phi`

### encoder-decoder (15 models)

- `bart`
- `bigbird`
- `blenderbot`
- `flan_t5`
- `led`
- `longformer`
- `longt5`
- `marian`
- `mbart`
- `mt5`
- `pegasus`
- `prophetnet`
- `reformer`
- `speecht5`
- `t5`

### vision (21 models)

- `beit`
- `bit`
- `convnext`
- `deit`
- `detr`
- `dinov2`
- `donut`
- `dpt`
- `efficientnet`
- `levit`
- `mask2former`
- `mlp-mixer`
- `mobilevit`
- `poolformer`
- `regnet`
- `resnet`
- `sam`
- `segformer`
- `swin`
- `vit`
- `yolos`

### vision-text (11 models)

- `align`
- `blip`
- `blip_2`
- `chinese_clip`
- `clip`
- `florence`
- `pix2struct`
- `vilt`
- `vinvl`
- `vision_encoder_decoder`
- `vision_text_dual_encoder`

### speech (10 models)

- `bark`
- `clap`
- `data2vec_audio`
- `encodec`
- `musicgen`
- `sew`
- `unispeech`
- `wav2vec2`
- `wavlm`
- `whisper`

### multimodal (11 models)

- `data2vec`
- `data2vec_text`
- `data2vec_vision`
- `flamingo`
- `flava`
- `git`
- `idefics`
- `imagebind`
- `llava`
- `usm`
- `video_llava`

---

This report will be automatically updated by running:
```bash
python create_coverage_tool.py --update-report
```