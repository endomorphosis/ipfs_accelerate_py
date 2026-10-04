# Test File Validation Report

Generated: 2025-03-23 00:51:30

## Summary

- Directory: `refactored_test_suite`
- Pattern: `test_*.py`
- Total files: 41
- Valid files: 30 (73.2%)
- Invalid files: 11
- Files with warnings: 10

## Invalid Files

### api/test_claude_api.py

- Test class: `TestClaudeAPI`
- Model ID: `None`

**Errors:**

- Test class TestClaudeAPI does not inherit from ModelTest. Base classes: APITest

### api/test_model_api.py

- Test class: `TestModelAPI`
- Model ID: `bert-base-uncased`

**Errors:**

- Test class TestModelAPI does not inherit from ModelTest. Base classes: APITest

### browser/test_ipfs_accelerate_with_cross_browser.py

- Test class: `TestIPFSAcceleratedBrowserSharding`
- Model ID: `None`

**Errors:**

- Test class TestIPFSAcceleratedBrowserSharding does not inherit from ModelTest. Base classes: BrowserTest

### models/text/test_bert_qualcomm.py

- Test class: `TestBertQualcomm`
- Model ID: `None`

**Errors:**

- Test class TestBertQualcomm does not inherit from ModelTest. Base classes: HardwareTest

### models/vision/test_vit-base-patch16-224.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- Syntax error on line 187: unexpected indent

### test_utils.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- No test class found (class starting with 'Test')

### tests/models/text/test_bert-base-uncased.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- Syntax error on line 85: expected 'except' or 'finally' block

### tests/models/text/test_bert_fixed.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- Syntax error on line 231: expected 'except' or 'finally' block

### tests/models/text/test_bert_simple.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- Syntax error on line 27: expected 'except' or 'finally' block

### tests/unit/test_hf_t5.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- Syntax error on line 78: expected 'except' or 'finally' block

### tests/unit/test_whisper-tiny.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- Syntax error on line 117: expected 'except' or 'finally' block


## Files with Warnings

### api/test_claude_api.py

- Test class: `TestClaudeAPI`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### browser/test_ipfs_accelerate_with_cross_browser.py

- Test class: `TestIPFSAcceleratedBrowserSharding`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### hardware/webgpu/test_ipfs_accelerate_webnn_webgpu.py

- Test class: `TestIPFSAccelerateWebNNWebGPU`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### models/other/test_groq_models.py

- Test class: `TestGroqModels`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### models/other/test_single_model_hardware.py

- Test class: `TestSingleModelHardware`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### models/text/test_bert_base.py

- Test class: `TestBertBaseModel`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### models/text/test_bert_qualcomm.py

- Test class: `TestBertQualcomm`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### models/text/test_ollama_backoff.py

- Test class: `TestOllamaBackoff`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### models/text/test_ollama_backoff_comprehensive.py

- Test class: `TestOllamaBackoffComprehensive`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### models/text/test_ollama_mock.py

- Test class: `TestOllamaMock`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method


## Valid Files

### Audio Models

- `models/audio/test_hf_clap.py`: TestClapModels (Model: laion/clap-htsat-unfused)
- `models/audio/test_hf_wav2vec2.py`: TestWav2Vec2Models (Model: facebook/wav2vec2-base-960h)
- `models/audio/test_hf_whisper.py`: TestWhisperModels (Model: openai/whisper-tiny)
- `models/audio/test_wav2vec2_base_960h.py`: TestSpeechModel (Model: facebook/wav2vec2-base-960h)
- `models/audio/test_whisper_tiny.py`: TestSpeechModel (Model: openai/whisper-tiny)

### Multimodal Models

- `models/multimodal/test_blip_image_captioning_base.py`: TestBlipImageCaptioningBase (Model: Salesforce/blip-image-captioning-base)
- `models/multimodal/test_blip_vqa_base.py`: TestBlipVqaBase (Model: Salesforce/blip-vqa-base)
- `models/multimodal/test_clip_vit_base_patch32.py`: TestClipVitBasePatch32 (Model: openai/clip-vit-base-patch32)
- `models/multimodal/test_clip_vit_large_patch14.py`: TestClipVitLargePatch14 (Model: openai/clip-vit-large-patch14)
- `models/multimodal/test_flava_full.py`: TestFlavaFull (Model: facebook/flava-full)
- `models/multimodal/test_hf_clip.py`: TestCLIPModels (Model: openai/clip-vit-base-patch32)
- `models/multimodal/test_hf_llava.py`: TestLLaVAModels (Model: llava-hf/llava-1.5-7b-hf)
- `models/multimodal/test_hf_xclip.py`: TestXCLIPModels (Model: microsoft/xclip-base-patch32)

### Other Models

- `api/test_api_backend.py`: TestAPIBackend (Model: bert-base-uncased)
- `hardware/webgpu/test_ipfs_accelerate_webnn_webgpu.py`: TestIPFSAccelerateWebNNWebGPU
- `hardware/webgpu/test_webgpu_detection.py`: TestWebGPUDetection (Model: bert-base-uncased)
- `models/other/test_groq_models.py`: TestGroqModels
- `models/other/test_single_model_hardware.py`: TestSingleModelHardware

### Text Models

- `models/text/test_bert_base.py`: TestBertBaseModel
- `models/text/test_bert_base_uncased.py`: TestBertModel (Model: bert-base-uncased)
- `models/text/test_gpt2.py`: TestGptModel (Model: gpt2)
- `models/text/test_hf_qwen2.py`: TestQwen2Models (Model: Qwen/Qwen2-7B-Instruct)
- `models/text/test_hf_t5.py`: TestT5Models (Model: t5-small)
- `models/text/test_llama.py`: TestLlamaModel (Model: facebook/opt-125m)
- `models/text/test_ollama_backoff.py`: TestOllamaBackoff
- `models/text/test_ollama_backoff_comprehensive.py`: TestOllamaBackoffComprehensive
- `models/text/test_ollama_mock.py`: TestOllamaMock
- `models/text/test_roberta_base.py`: TestBertModel (Model: roberta-base)

### Vision Models

- `models/vision/test_hf_detr.py`: TestDETRModels (Model: facebook/detr-resnet-50)
- `models/vision/test_vit_base_patch16_224.py`: TestVitModel (Model: google/vit-base-patch16-224)

