# HuggingFace Model Test Validation Report

**Date:** 2025-03-22 01:04:28

## Summary

- **Total files:** 114
- **Syntax valid:** 114 (100.0%)
- **Structure valid:** 114 (100.0%)
- **Pipeline valid:** 114 (100.0%)
- **Task appropriate:** 114 (100.0%)
- **Pipeline missing:** 0 (0.0%)

## Results by Architecture

### Decoder-only (9 files)

- **Syntax valid:** 9 (100.0%)
- **Structure valid:** 9 (100.0%)
- **Pipeline valid:** 9 (100.0%)
- **Task appropriate:** 9 (100.0%)
- **Pipeline missing:** 0 (0.0%)

### Encoder-decoder (9 files)

- **Syntax valid:** 9 (100.0%)
- **Structure valid:** 9 (100.0%)
- **Pipeline valid:** 9 (100.0%)
- **Task appropriate:** 9 (100.0%)
- **Pipeline missing:** 0 (0.0%)

### Encoder-only (13 files)

- **Syntax valid:** 13 (100.0%)
- **Structure valid:** 13 (100.0%)
- **Pipeline valid:** 13 (100.0%)
- **Task appropriate:** 13 (100.0%)
- **Pipeline missing:** 0 (0.0%)

### Multimodal (6 files)

- **Syntax valid:** 6 (100.0%)
- **Structure valid:** 6 (100.0%)
- **Pipeline valid:** 6 (100.0%)
- **Task appropriate:** 6 (100.0%)
- **Pipeline missing:** 0 (0.0%)

### Speech (2 files)

- **Syntax valid:** 2 (100.0%)
- **Structure valid:** 2 (100.0%)
- **Pipeline valid:** 2 (100.0%)
- **Task appropriate:** 2 (100.0%)
- **Pipeline missing:** 0 (0.0%)

### Unknown (61 files)

- **Syntax valid:** 61 (100.0%)
- **Structure valid:** 61 (100.0%)
- **Pipeline valid:** 61 (100.0%)
- **Task appropriate:** 61 (100.0%)
- **Pipeline missing:** 0 (0.0%)

### Vision (10 files)

- **Syntax valid:** 10 (100.0%)
- **Structure valid:** 10 (100.0%)
- **Pipeline valid:** 10 (100.0%)
- **Task appropriate:** 10 (100.0%)
- **Pipeline missing:** 0 (0.0%)

### Vision-text (4 files)

- **Syntax valid:** 4 (100.0%)
- **Structure valid:** 4 (100.0%)
- **Pipeline valid:** 4 (100.0%)
- **Task appropriate:** 4 (100.0%)
- **Pipeline missing:** 0 (0.0%)

## Detailed Results

### Valid Files

These files passed all validation checks:

- `test_hf_albert.py` - encoder-only - Task: fill-mask
- `test_hf_align.py` - unknown - Task: fill-mask
- `test_hf_audio.py` - unknown - Task: fill-mask
- `test_hf_bart.py` - encoder-decoder - Task: text2text-generation
- `test_hf_beit.py` - vision - Task: image-classification
- `test_hf_bert.py` - encoder-only - Task: fill-mask
- `test_hf_bigbird.py` - unknown - Task: fill-mask
- `test_hf_bit.py` - unknown - Task: fill-mask
- `test_hf_blenderbot.py` - unknown - Task: fill-mask
- `test_hf_blip.py` - vision-text - Task: image-to-text
- `test_hf_blip_2.py` - vision-text - Task: image-to-text
- `test_hf_bloom.py` - decoder-only - Task: text-generation
- `test_hf_camembert.py` - encoder-only - Task: fill-mask
- `test_hf_canine.py` - unknown - Task: fill-mask
- `test_hf_chinese_clip.py` - vision-text - Task: zero-shot-image-classification
- `test_hf_clap.py` - unknown - Task: fill-mask
- `test_hf_clip.py` - vision-text - Task: zero-shot-image-classification
- `test_hf_convnext.py` - vision - Task: image-classification
- `test_hf_ctrl.py` - unknown - Task: fill-mask
- `test_hf_data2vec.py` - unknown - Task: fill-mask
- `test_hf_data2vec_audio.py` - unknown - Task: fill-mask
- `test_hf_data2vec_text.py` - unknown - Task: fill-mask
- `test_hf_data2vec_vision.py` - unknown - Task: fill-mask
- `test_hf_deberta.py` - encoder-only - Task: fill-mask
- `test_hf_deberta_v2.py` - encoder-only - Task: fill-mask
- `test_hf_decoder_only.py` - unknown - Task: fill-mask
- `test_hf_deit.py` - vision - Task: image-classification
- `test_hf_detr.py` - unknown - Task: object-detection
- `test_hf_dinov2.py` - vision - Task: image-classification
- `test_hf_distilbert.py` - encoder-only - Task: fill-mask
- `test_hf_donut.py` - unknown - Task: fill-mask
- `test_hf_dpt.py` - unknown - Task: fill-mask
- `test_hf_efficientnet.py` - unknown - Task: fill-mask
- `test_hf_electra.py` - encoder-only - Task: fill-mask
- `test_hf_encodec.py` - unknown - Task: fill-mask
- `test_hf_encoder_decoder.py` - unknown - Task: fill-mask
- `test_hf_encoder_only.py` - unknown - Task: fill-mask
- `test_hf_ernie.py` - unknown - Task: fill-mask
- `test_hf_falcon.py` - decoder-only - Task: text-generation
- `test_hf_flamingo.py` - unknown - Task: fill-mask
- `test_hf_flan_t5.py` - encoder-decoder - Task: text2text-generation
- `test_hf_flava.py` - unknown - Task: fill-mask
- `test_hf_florence.py` - unknown - Task: fill-mask
- `test_hf_funnel.py` - unknown - Task: fill-mask
- `test_hf_gemma.py` - unknown - Task: fill-mask
- `test_hf_git.py` - multimodal - Task: image-to-text
- `test_hf_gpt2.py` - decoder-only - Task: text-generation
- `test_hf_gpt_j.py` - decoder-only - Task: text-generation
- `test_hf_gpt_neo.py` - unknown - Task: fill-mask
- `test_hf_gpt_neox.py` - unknown - Task: fill-mask
- `test_hf_gptj.py` - unknown - Task: fill-mask
- `test_hf_hubert.py` - encoder-only - Task: fill-mask
- `test_hf_idefics.py` - unknown - Task: fill-mask
- `test_hf_imagebind.py` - unknown - Task: fill-mask
- `test_hf_layoutlm.py` - unknown - Task: fill-mask
- `test_hf_led.py` - encoder-decoder - Task: text2text-generation
- `test_hf_levit.py` - vision - Task: image-classification
- `test_hf_llama.py` - decoder-only - Task: text-generation
- `test_hf_llava.py` - multimodal - Task: image-to-text
- `test_hf_longformer.py` - unknown - Task: fill-mask
- `test_hf_marian.py` - encoder-decoder - Task: text2text-generation
- `test_hf_mask2former.py` - unknown - Task: image-segmentation
- `test_hf_mbart.py` - encoder-decoder - Task: text2text-generation
- `test_hf_mistral.py` - decoder-only - Task: text-generation
- `test_hf_mixtral.py` - decoder-only - Task: text-generation
- `test_hf_mlp-mixer.py` - unknown - Task: fill-mask
- `test_hf_mobilevit.py` - vision - Task: image-classification
- `test_hf_mpt.py` - decoder-only - Task: text-generation
- `test_hf_mt5.py` - encoder-decoder - Task: text2text-generation
- `test_hf_multimodal.py` - unknown - Task: fill-mask
- `test_hf_musicgen.py` - unknown - Task: fill-mask
- `test_hf_opt.py` - unknown - Task: fill-mask
- `test_hf_paligemma.py` - multimodal - Task: image-to-text
- `test_hf_pegasus.py` - encoder-decoder - Task: text2text-generation
- `test_hf_phi.py` - decoder-only - Task: text-generation
- `test_hf_pix2struct.py` - multimodal - Task: image-to-text
- `test_hf_poolformer.py` - vision - Task: image-classification
- `test_hf_prophetnet.py` - unknown - Task: fill-mask
- `test_hf_reformer.py` - unknown - Task: fill-mask
- `test_hf_regnet.py` - unknown - Task: fill-mask
- `test_hf_rembert.py` - encoder-only - Task: fill-mask
- `test_hf_resnet.py` - vision - Task: image-classification
- `test_hf_roberta.py` - encoder-only - Task: fill-mask
- `test_hf_roformer.py` - unknown - Task: fill-mask
- `test_hf_sam.py` - unknown - Task: image-segmentation
- `test_hf_seamless_m4t.py` - unknown - Task: fill-mask
- `test_hf_segformer.py` - unknown - Task: image-segmentation
- `test_hf_sew.py` - unknown - Task: fill-mask
- `test_hf_speech_to_text.py` - unknown - Task: fill-mask
- `test_hf_speech_to_text_2.py` - unknown - Task: fill-mask
- `test_hf_speecht5.py` - encoder-decoder - Task: automatic-speech-recognition
- `test_hf_squeezebert.py` - encoder-only - Task: fill-mask
- `test_hf_swin.py` - vision - Task: image-classification
- `test_hf_t5.py` - encoder-decoder - Task: text2text-generation
- `test_hf_tapas.py` - unknown - Task: fill-mask
- `test_hf_transfo-xl.py` - unknown - Task: fill-mask
- `test_hf_trocr_base.py` - unknown - Task: fill-mask
- `test_hf_trocr_large.py` - unknown - Task: fill-mask
- `test_hf_unispeech.py` - unknown - Task: fill-mask
- `test_hf_usm.py` - unknown - Task: fill-mask
- `test_hf_video_llava.py` - multimodal - Task: image-to-text
- `test_hf_vilt.py` - unknown - Task: fill-mask
- `test_hf_vinvl.py` - unknown - Task: fill-mask
- `test_hf_vision.py` - unknown - Task: fill-mask
- `test_hf_vision_encoder_decoder.py` - unknown - Task: fill-mask
- `test_hf_vision_text_dual_encoder.py` - multimodal - Task: zero-shot-image-classification
- `test_hf_vit.py` - vision - Task: image-classification
- `test_hf_wav2vec2.py` - speech - Task: automatic-speech-recognition
- `test_hf_wav2vec2_bert.py` - encoder-only - Task: fill-mask
- `test_hf_wavlm.py` - unknown - Task: fill-mask
- `test_hf_whisper.py` - speech - Task: automatic-speech-recognition
- `test_hf_xlm_roberta.py` - encoder-only - Task: fill-mask
- `test_hf_xlnet.py` - unknown - Task: fill-mask
- `test_hf_yolos.py` - unknown - Task: fill-mask