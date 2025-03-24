# HuggingFace Model Coverage Report

**Date:** 2025-03-22 02:27:12

## Summary

- **Total models tracked:** 155
- **Implemented models:** 0 (0.0%)
- **Missing models:** 155 (100.0%)

## Coverage by Architecture

### Decoder-only Models

- **Total decoder-only models:** 34
- **Implemented:** 0 (0.0%)
- **Missing:** 34

### Encoder-decoder Models

- **Total encoder-decoder models:** 15
- **Implemented:** 0 (0.0%)
- **Missing:** 15

### Encoder-only Models

- **Total encoder-only models:** 31
- **Implemented:** 0 (0.0%)
- **Missing:** 31

### Multimodal Models

- **Total multimodal models:** 20
- **Implemented:** 0 (0.0%)
- **Missing:** 20

### Speech Models

- **Total speech models:** 12
- **Implemented:** 0 (0.0%)
- **Missing:** 12

### Unknown Models

- **Total unknown models:** 0
- **Implemented:** 0 (0.0%)
- **Missing:** 0

### Vision Models

- **Total vision models:** 32
- **Implemented:** 0 (0.0%)
- **Missing:** 32

### Vision-text Models

- **Total vision-text models:** 11
- **Implemented:** 0 (0.0%)
- **Missing:** 11

## Coverage by Priority

### Critical Priority Models

- **Total critical models:** 35
- **Implemented:** 0 (0.0%)
- **Missing:** 35

### High Priority Models

- **Total high models:** 40
- **Implemented:** 0 (0.0%)
- **Missing:** 40

### Medium Priority Models

- **Total medium models:** 80
- **Implemented:** 0 (0.0%)
- **Missing:** 80

## Implementation Roadmap

### Critical Priority Models

These models should be implemented first due to their importance and widespread use:

- `bloom` (decoder-only)
- `falcon` (decoder-only)
- `gpt-j` (decoder-only)
- `gpt2` (decoder-only)
- `llama` (decoder-only)
- `mistral` (decoder-only)
- `opt` (decoder-only)
- `phi` (decoder-only)
- `bart` (encoder-decoder)
- `flan-t5` (encoder-decoder)
- `mbart` (encoder-decoder)
- `pegasus` (encoder-decoder)
- `t5` (encoder-decoder)
- `albert` (encoder-only)
- `bert` (encoder-only)
- `deberta` (encoder-only)
- `distilbert` (encoder-only)
- `roberta` (encoder-only)
- `xlm-roberta` (encoder-only)
- `llava` (multimodal)
- `paligemma` (multimodal)
- `pix2struct` (multimodal)
- `hubert` (speech)
- `speecht5` (speech)
- `wav2vec2` (speech)
- `whisper` (speech)
- `beit` (vision)
- `convnext` (vision)
- `deit` (vision)
- `resnet` (vision)
- `swin` (vision)
- `vit` (vision)
- `blip` (vision-text)
- `clip` (vision-text)
- `vision-text-dual-encoder` (vision-text)

### High Priority Models

These models should be implemented next:

- `codellama` (decoder-only)
- `gemma` (decoder-only)
- `gpt-neo` (decoder-only)
- `gpt-neox` (decoder-only)
- `mixtral` (decoder-only)
- `mpt` (decoder-only)
- `qwen2` (decoder-only)
- `qwen3` (decoder-only)
- `led` (encoder-decoder)
- `longt5` (encoder-decoder)
- `marian` (encoder-decoder)
- `mt5` (encoder-decoder)
- `mt5` (encoder-decoder)
- `pegasus-x` (encoder-decoder)
- `camembert` (encoder-only)
- `electra` (encoder-only)
- `ernie` (encoder-only)
- `luke` (encoder-only)
- `mpnet` (encoder-only)
- `roformer` (encoder-only)
- `flava` (multimodal)
- `fuyu` (multimodal)
- `idefics` (multimodal)
- `kosmos-2` (multimodal)
- `llava-next` (multimodal)
- `video-llava` (multimodal)
- `bark` (speech)
- `encodec` (speech)
- `musicgen` (speech)
- `wavlm` (speech)
- `dinov2` (vision)
- `efficientnet` (vision)
- `mobilenet-v2` (vision)
- `poolformer` (vision)
- `regnet` (vision)
- `segformer` (vision)
- `blip-2` (vision-text)
- `chinese-clip` (vision-text)
- `clipseg` (vision-text)
- `git` (vision-text)

### Medium Priority Models

These models can be implemented after critical and high priority models:

#### Decoder-only Models

- `codegen`
- `command-r`
- `gemma2`
- `gemma3`
- `llama-3`
- `mamba`
- `mistral-next`
- `nemotron`
- `olmo`
- `olmoe`
- `openai-gpt`
- `persimmon`
- `phi3`
- `phi4`
- `recurrent-gemma`
- `rwkv`
- `stablelm`
- `starcoder2`

#### Encoder-decoder Models

- `m2m-100`
- `seamless-m4t`
- `switch-transformers`
- `umt5`

#### Encoder-only Models

- `convbert`
- `data2vec-text`
- `deberta-v2`
- `esm`
- `flaubert`
- `funnel`
- `ibert`
- `layoutlm`
- `megatron-bert`
- `mobilebert`
- `mra`
- `nezha`
- `nystromformer`
- `splinter`
- `squeezebert`
- `xlm`
- `xlm-roberta-xl`
- `xlnet`
- `xmod`

#### Multimodal Models

- `blip`
- `clip`
- `git`
- `idefics2`
- `idefics3`
- `imagebind`
- `llava-next-video`
- `mllama`
- `qwen2-vl`
- `qwen3-vl`
- `siglip`

#### Speech Models

- `speech-to-text`
- `speech-to-text-2`
- `unispeech`
- `wav2vec2-conformer`

#### Vision Models

- `beit3`
- `bit`
- `conditional-detr`
- `convnextv2`
- `cvt`
- `depth-anything`
- `detr`
- `dinat`
- `dino`
- `dpt`
- `imagegpt`
- `levit`
- `mask2former`
- `mlp-mixer`
- `mobilenet-v1`
- `mobilevit`
- `swinv2`
- `van`
- `vitdet`
- `yolos`

#### Vision-text Models

- `instructblip`
- `pix2struct`
- `vision-encoder-decoder`
- `xclip`

## Implemented Models

These models already have test implementations: