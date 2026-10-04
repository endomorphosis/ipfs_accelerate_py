# Test File Validation Report

Generated: 2025-03-23 00:34:55

## Summary

- Directory: `/home/barberb/ipfs_accelerate_py/test/refactored_test_suite/models/multimodal`
- Pattern: `test_*.py`
- Total files: 8
- Valid files: 5 (62.5%)
- Invalid files: 3
- Files with warnings: 0

## Invalid Files

### test_hf_clip.py

- Test class: `TestCLIPModels`
- Model ID: `openai/clip-vit-base-patch32`

**Errors:**

- Missing required methods: test_model_loading

### test_hf_llava.py

- Test class: `TestLLaVAModels`
- Model ID: `llava-hf/llava-1.5-7b-hf`

**Errors:**

- Missing required methods: test_model_loading

### test_hf_xclip.py

- Test class: `TestXCLIPModels`
- Model ID: `microsoft/xclip-base-patch32`

**Errors:**

- Missing required methods: test_model_loading


## Valid Files

### Other Models

- `test_blip_image_captioning_base.py`: TestBlipImageCaptioningBase (Model: Salesforce/blip-image-captioning-base)
- `test_blip_vqa_base.py`: TestBlipVqaBase (Model: Salesforce/blip-vqa-base)
- `test_clip_vit_base_patch32.py`: TestClipVitBasePatch32 (Model: openai/clip-vit-base-patch32)
- `test_clip_vit_large_patch14.py`: TestClipVitLargePatch14 (Model: openai/clip-vit-large-patch14)
- `test_flava_full.py`: TestFlavaFull (Model: facebook/flava-full)

