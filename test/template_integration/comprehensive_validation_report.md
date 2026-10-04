# Test File Validation Report

Generated: 2025-03-23 00:44:49

## Summary

- Directory: `../refactored_test_suite`
- Pattern: `test_*.py`
- Total files: 41
- Valid files: 11 (26.8%)
- Invalid files: 30
- Files with warnings: 16

## Invalid Files

### api/test_api_backend.py

- Test class: `TestAPIBackend`
- Model ID: `None`

**Errors:**

- Test class TestAPIBackend does not inherit from ModelTest. Base classes: APITest
- Missing required methods: test_model_loading

### api/test_claude_api.py

- Test class: `TestClaudeAPI`
- Model ID: `None`

**Errors:**

- Test class TestClaudeAPI does not inherit from ModelTest. Base classes: APITest
- Missing required methods: test_model_loading

### api/test_model_api.py

- Test class: `TestModelAPI`
- Model ID: `bert-base-uncased`

**Errors:**

- Test class TestModelAPI does not inherit from ModelTest. Base classes: APITest
- Missing required methods: test_model_loading

### browser/test_ipfs_accelerate_with_cross_browser.py

- Test class: `TestIPFSAcceleratedBrowserSharding`
- Model ID: `None`

**Errors:**

- Test class TestIPFSAcceleratedBrowserSharding does not inherit from ModelTest. Base classes: BrowserTest
- Missing required methods: test_model_loading

### hardware/webgpu/test_ipfs_accelerate_webnn_webgpu.py

- Test class: `TestIPFSAccelerateWebNNWebGPU`
- Model ID: `None`

**Errors:**

- Test class TestIPFSAccelerateWebNNWebGPU does not inherit from ModelTest. Base classes: HardwareTest
- Missing required methods: test_model_loading

### hardware/webgpu/test_webgpu_detection.py

- Test class: `TestWebGPUDetection`
- Model ID: `None`

**Errors:**

- Test class TestWebGPUDetection does not inherit from ModelTest. Base classes: HardwareTest
- Missing required methods: test_model_loading

### models/audio/test_hf_clap.py

- Test class: `TestClapModels`
- Model ID: `laion/clap-htsat-unfused`

**Errors:**

- Missing required methods: test_model_loading

### models/audio/test_hf_wav2vec2.py

- Test class: `TestWav2Vec2Models`
- Model ID: `facebook/wav2vec2-base-960h`

**Errors:**

- Missing required methods: test_model_loading

### models/audio/test_hf_whisper.py

- Test class: `TestWhisperModels`
- Model ID: `openai/whisper-tiny`

**Errors:**

- Missing required methods: test_model_loading

### models/multimodal/test_hf_clip.py

- Test class: `TestCLIPModels`
- Model ID: `openai/clip-vit-base-patch32`

**Errors:**

- Missing required methods: test_model_loading

### models/multimodal/test_hf_llava.py

- Test class: `TestLLaVAModels`
- Model ID: `llava-hf/llava-1.5-7b-hf`

**Errors:**

- Missing required methods: test_model_loading

### models/multimodal/test_hf_xclip.py

- Test class: `TestXCLIPModels`
- Model ID: `microsoft/xclip-base-patch32`

**Errors:**

- Missing required methods: test_model_loading

### models/other/test_groq_models.py

- Test class: `TestGroqModels`
- Model ID: `None`

**Errors:**

- Missing required methods: test_model_loading

### models/other/test_single_model_hardware.py

- Test class: `TestSingleModelHardware`
- Model ID: `None`

**Errors:**

- Test class TestSingleModelHardware does not inherit from ModelTest. Base classes: HardwareTest
- Missing required methods: test_model_loading

### models/text/test_bert_base.py

- Test class: `TestBertBaseModel`
- Model ID: `None`

**Errors:**

- Missing required methods: test_model_loading

### models/text/test_bert_qualcomm.py

- Test class: `TestBertQualcomm`
- Model ID: `None`

**Errors:**

- Test class TestBertQualcomm does not inherit from ModelTest. Base classes: HardwareTest
- Missing required methods: test_model_loading

### models/text/test_hf_qwen2.py

- Test class: `TestQwen2Models`
- Model ID: `Qwen/Qwen2-7B-Instruct`

**Errors:**

- Missing required methods: test_model_loading

### models/text/test_hf_t5.py

- Test class: `TestT5Models`
- Model ID: `t5-small`

**Errors:**

- Missing required methods: test_model_loading

### models/text/test_llama.py

- Test class: `TestLlamaModel`
- Model ID: `facebook/opt-125m`

**Errors:**

- Missing required methods: test_model_loading

### models/text/test_ollama_backoff.py

- Test class: `TestOllamaBackoff`
- Model ID: `None`

**Errors:**

- Missing required methods: test_model_loading

### models/text/test_ollama_backoff_comprehensive.py

- Test class: `TestOllamaBackoffComprehensive`
- Model ID: `None`

**Errors:**

- Missing required methods: test_model_loading

### models/text/test_ollama_mock.py

- Test class: `TestOllamaMock`
- Model ID: `None`

**Errors:**

- Missing required methods: test_model_loading

### models/vision/test_hf_detr.py

- Test class: `TestDETRModels`
- Model ID: `facebook/detr-resnet-50`

**Errors:**

- Missing required methods: test_model_loading

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

- Test class: `TestBertBaseUncased`
- Model ID: `None`

**Errors:**

- Test class TestBertBaseUncased does not inherit from ModelTest. Base classes: 
- Missing required methods: setUp, test_model_loading

### tests/models/text/test_bert_fixed.py

- Test class: `TestBertBaseUncased`
- Model ID: `None`

**Errors:**

- Test class TestBertBaseUncased does not inherit from ModelTest. Base classes: 
- Missing required methods: setUp, test_model_loading

### tests/models/text/test_bert_simple.py

- Test class: `None`
- Model ID: `None`

**Errors:**

- No test class found (class starting with 'Test')

### tests/unit/test_hf_t5.py

- Test class: `TestT5Models`
- Model ID: `None`

**Errors:**

- Test class TestT5Models does not inherit from ModelTest. Base classes: 
- Missing required methods: setUp, test_model_loading

### tests/unit/test_whisper-tiny.py

- Test class: `TestWhisperTiny`
- Model ID: `None`

**Errors:**

- Test class TestWhisperTiny does not inherit from ModelTest. Base classes: 
- Missing required methods: setUp, test_model_loading


## Files with Warnings

### api/test_api_backend.py

- Test class: `TestAPIBackend`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

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

### hardware/webgpu/test_webgpu_detection.py

- Test class: `TestWebGPUDetection`
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

### tests/models/text/test_bert-base-uncased.py

- Test class: `TestBertBaseUncased`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### tests/models/text/test_bert_fixed.py

- Test class: `TestBertBaseUncased`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### tests/unit/test_hf_t5.py

- Test class: `TestT5Models`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method

### tests/unit/test_whisper-tiny.py

- Test class: `TestWhisperTiny`
- Model ID: `None`

**Warnings:**

- self.model_id assignment not found in setUp method


## Valid Files

### Audio Models

- `models/audio/test_wav2vec2_base_960h.py`: TestSpeechModel (Model: facebook/wav2vec2-base-960h)
- `models/audio/test_whisper_tiny.py`: TestSpeechModel (Model: openai/whisper-tiny)

### Multimodal Models

- `models/multimodal/test_blip_image_captioning_base.py`: TestBlipImageCaptioningBase (Model: Salesforce/blip-image-captioning-base)
- `models/multimodal/test_blip_vqa_base.py`: TestBlipVqaBase (Model: Salesforce/blip-vqa-base)
- `models/multimodal/test_clip_vit_base_patch32.py`: TestClipVitBasePatch32 (Model: openai/clip-vit-base-patch32)
- `models/multimodal/test_clip_vit_large_patch14.py`: TestClipVitLargePatch14 (Model: openai/clip-vit-large-patch14)
- `models/multimodal/test_flava_full.py`: TestFlavaFull (Model: facebook/flava-full)

### Text Models

- `models/text/test_bert_base_uncased.py`: TestBertModel (Model: bert-base-uncased)
- `models/text/test_gpt2.py`: TestGptModel (Model: gpt2)
- `models/text/test_roberta_base.py`: TestBertModel (Model: roberta-base)

### Vision Models

- `models/vision/test_vit_base_patch16_224.py`: TestVitModel (Model: google/vit-base-patch16-224)

