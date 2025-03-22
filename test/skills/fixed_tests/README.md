# Fixed Tests for HuggingFace Models

This directory contains test files that have been regenerated with fixes for:

1. Hyphenated model names (e.g. "gpt-j" → "gpt_j")
2. Capitalization issues in class names (e.g. "GPTJForCausalLM" vs "GptjForCausalLM")
3. Syntax errors like unterminated string literals
4. Indentation issues

The test files in this directory are generated using the updated test generator
that handles hyphenated model names correctly. The generator now:

1. Automatically converts hyphenated model names to valid Python identifiers
2. Ensures proper capitalization patterns for class names
3. Validates that generated files have valid Python syntax
4. Fixes common syntax errors like unterminated string literals

## Example Models with Hyphenated Names

- gpt-j → test_hf_gpt_j.py
- gpt-neo → test_hf_gpt_neo.py
- xlm-roberta → test_hf_xlm_roberta.py

## Running the Tests

Tests can be run individually with:

```bash
python fixed_tests/test_hf_gpt_j.py --list-models
python fixed_tests/test_hf_xlm_roberta.py --list-models
```

To run all tests:

```bash
cd fixed_tests
for test in test_hf_*.py; do python $test --list-models; done
```

## Validation

All test files in this directory have been validated to ensure:

1. Valid Python syntax
2. Proper indentation
3. Correct class naming patterns
4. Valid Python identifiers for hyphenated model names

## Coverage Status

| Architecture Type | Models Covered |
|------------------|----------------|
| decoder-only | bloom, falcon, gemma, gpt-j, gpt-neo, gpt-neox, gpt2, llama, mistral, mixtral, mpt, opt, phi |
| encoder-decoder | bart, flan-t5, led, mbart, mt5, pegasus, prophetnet, t5 |
| encoder-only | albert, bert, bigbird, canine, deberta, deberta-v2, distilbert, electra, ernie, layoutlm, rembert, roberta, roformer, xlm-roberta |
| multimodal | blip, blip-2, clip, flava, git, idefics, imagebind, llava, paligemma, video-llava |
| speech | bark, clap, encodec, hubert, musicgen, sew, unispeech, wav2vec2, whisper |
| unknown | audio, chinese-clip, data2vec-audio, data2vec-text, data2vec-vision, decoder-only, gptj, multimodal, speech-to-text-2, trocr-base, trocr-large, vision, vision-encoder-decoder, vision-text-dual-encoder, wav2vec2-bert |
| vision | beit, convnext, convnextv2, deit, detr, dinov2, mask2former, resnet, sam, segformer, swin, vit, yolos |
| vision-text | clipseg, xclip |

## Recently Added Models

### Phase 3 Models (March 26 - April 5, 2025)

These models represent the architecture expansion phase of our coverage roadmap:

- **bigbird** (encoder-only): Big Bird model with block sparse attention (March 22, 2025)
- **bark** (speech): Bark text-to-audio model (March 22, 2025)
- **canine** (encoder-only): CANINE character-level transformer for multilingual NLP (March 22, 2025)
- **roformer** (encoder-only): RoFormer rotary position embedding transformer for NLP (March 22, 2025)
- **beit** (vision): BEiT vision transformer models for image classification (March 21, 2025)
- **clap** (speech): Contrastive Language-Audio Pretraining model (March 21, 2025)
- **clipseg** (vision-text): CLIPSeg segmentation model (March 22, 2025)
- **convnextv2** (vision): ConvNeXtV2 vision model (March 22, 2025)
- **detr** (vision): Detection Transformer models for object detection (March 21, 2025)
- **dinov2** (vision): DINOv2 self-supervised vision models (March 21, 2025)
- **encodec** (speech): EnCodec audio codec model (March 20, 2025)
- **ernie** (encoder-only): ERNIE mask language model (March 21, 2025)
- **gemma** (decoder-only): Gemma lightweight language models (March 20, 2025)
- **git** (multimodal): GenerativeImage2Text multimodal model (March 21, 2025)
- **gpt-j** (decoder-only): GPT-J autoregressive language model (March 22, 2025)
- **gpt-neo** (decoder-only): GPT-Neo autoregressive language models (March 21, 2025)
- **imagebind** (multimodal): ImageBind multimodal binding model (March 21, 2025)
- **led** (encoder-decoder): Longformer Encoder-Decoder for long text (March 20, 2025)
- **musicgen** (speech): MusicGen music generation model (March 20, 2025)
- **paligemma** (multimodal): PaliGemma multimodal model (March 21, 2025)
- **pegasus** (encoder-decoder): PEGASUS text summarization model (March 20, 2025)
- **prophetnet** (encoder-decoder): ProphetNet sequence-to-sequence model (March 21, 2025)
- **segformer** (vision): SegFormer segmentation model (March 21, 2025)
- **sew** (speech): Squeezed and Efficient Wav2Vec model (March 20, 2025)
- **speech-to-text** (speech): Speech-to-Text model for ASR (March 22, 2025)
- **xclip** (vision-text): XClip video classification model (March 22, 2025)
- **xlm-roberta** (encoder-only): XLM-RoBERTa multilingual model (March 22, 2025)

### Phase 2 Models (March 20-25, 2025)

Previously added high-priority models:

- **deberta** (encoder-only): DeBERTa masked language models
- **deberta-v2** (encoder-only): DeBERTa-v2 masked language models
- **ernie** (encoder-only): ERNIE masked language models
- **falcon** (decoder-only): Falcon autoregressive language models
- **flan-t5** (encoder-decoder): Flan-T5 instruction-tuned models
- **flava** (multimodal): FLAVA multimodal model
- **idefics** (multimodal): IDEFICS multimodal model
- **mask2former** (vision): Mask2Former segmentation model
- **mistral** (decoder-only): Mistral autoregressive language models
- **mixtral** (decoder-only): Mixtral mixture of experts models
- **mpt** (decoder-only): MPT decoder-only model
- **phi** (decoder-only): Phi small language models
- **rembert** (encoder-only): RemBERT encoder-only model
- **resnet** (vision): ResNet vision model
- **sam** (vision): Segment Anything Model vision model
- **yolos** (vision): YOLOS object detection model
