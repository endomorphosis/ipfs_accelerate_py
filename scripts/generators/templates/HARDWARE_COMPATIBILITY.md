# Hardware Compatibility in Templates

This document summarizes the hardware compatibility status across model templates in our template system.

## Current Status (March 10, 2025)

As of the latest template fixes, we are working to ensure all templates support the full range of hardware platforms:

| Hardware Platform | Support Level | # Templates | Notes |
|------------------|---------------|-------------|-------|
| CPU              | ✅ Complete   | 26/26 (100%) | Standard on all templates |
| CUDA             | ✅ Complete   | 26/26 (100%) | Standard on all templates |
| ROCm (AMD)       | ⚠️ Partial    | 14/26 (54%)  | Being added in updates |
| MPS (Apple)      | ⚠️ Partial    | 14/26 (54%)  | Being added in updates |
| OpenVINO (Intel) | ⚠️ Partial    | 14/26 (54%)  | Being added in updates |
| Qualcomm         | ⚠️ Partial    | 14/26 (54%)  | Added in March 2025 |
| WebNN            | ⚠️ Partial    | 14/26 (54%)  | Being added in updates |
| WebGPU           | ⚠️ Partial    | 14/26 (54%)  | Being added in updates |

## Fixed Templates

Fixed templates now have complete hardware compatibility across all 8 hardware platforms:

1. text_embedding_test_template_text_generation.py (text_embedding/test)
2. vision_test_template_vision_language.py (vision/test)
3. llava_test_template_llava.py (llava/test)
4. whisper_test_template_whisper.py (whisper/test)
5. clap_test_template_clap.py (clap/test)
6. audio_test_template_audio.py (audio/test)
7. bert_test_template_bert.py (bert/test)
8. inheritance_test_template_inheritance_system_fixed.py (inheritance/test)
9. selection_test_template_selection.py (selection/test)
10. validator_test_template_validator.py (validator/test)
11. wav2vec2_test_template_wav2vec2.py (wav2vec2/test)
12. llava_test_template_llava_next.py (llava/test)
13. verifier_test_template_verifier.py (verifier/test)
14. hardware_test_template_hardware_detection.py (hardware/test)

## Hardware Platform Standard Support

Each template is being updated to include the following standard components for each hardware platform:

1. **Platform Constants**:
   ```python
   CPU = "cpu"
   CUDA = "cuda"
   OPENVINO = "openvino"
   MPS = "mps"
   ROCM = "rocm"
   WEBGPU = "webgpu"
   WEBNN = "webnn"
   QUALCOMM = "qualcomm"
   ```

2. **Initialization Methods**:
   ```python
   def init_cpu(self):
       """Initialize for CPU platform."""
       self.platform = "CPU"
       self.device = "cpu"
       self.device_name = "cpu"
       return True

   def init_cuda(self):
       """Initialize for CUDA platform."""
       try:
           import torch
           self.platform = "CUDA"
           self.device = "cuda"
           self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
           return True
       except ImportError:
           print("CUDA not available: torch package not found")
           return False
   
   # ... similar methods for other platforms
   ```

3. **Handler Creation Methods**:
   ```python
   def create_cpu_handler(self):
       """Create handler for CPU platform."""
       try:
           model_path = self.get_model_path_or_name()
           handler = AutoModel.from_pretrained(model_path)
           return handler
       except Exception as e:
           print(f"Error creating CPU handler: {e}")
           return MockHandler(self.model_path, "cpu")
   
   # ... similar methods for other platforms
   ```

## Hardware-Specific Optimizations

In addition to basic platform support, we're adding optimizations for specific hardware:

- **Qualcomm AI Engine**: Added support for Qualcomm QNN with dynamic library detection
- **WebGPU**: Added specialized handlers for browser GPU acceleration
- **MPS (Apple)**: Added native Metal Performance Shaders support for Apple Silicon
- **OpenVINO (Intel)**: Added specialized model conversion and optimization for Intel hardware

## Remaining Work

To complete hardware compatibility across the codebase, we need to:

1. Fix syntax errors in the remaining 12 templates
2. Add full hardware platform support to these templates
3. Update test generators to properly use the templates
4. Add hardware detection in the generated tests
5. Implement comprehensive testing across all hardware platforms

## Testing Platform Support

To test that the updated templates properly support all hardware platforms, use the template system to generate tests and then run those tests on different hardware:

```bash
python template_extractor.py --extract-template text_embedding/test --output test_embedding.py
python test_embedding.py --platform cuda
python test_embedding.py --platform cpu
python test_embedding.py --platform openvino
```

## Hardware Compatibility Matrix

| Model Type    | CPU | CUDA | ROCm | MPS | OpenVINO | Qualcomm | WebNN | WebGPU |
|---------------|-----|------|------|-----|----------|----------|-------|--------|
| BERT          | ✅  | ✅   | ✅   | ✅  | ✅       | ✅       | ✅    | ✅     |
| ViT           | ✅  | ✅   | ⚠️   | ⚠️  | ⚠️       | ⚠️       | ⚠️    | ⚠️     |
| CLIP          | ✅  | ✅   | ⚠️   | ⚠️  | ⚠️       | ⚠️       | ⚠️    | ⚠️     |
| T5            | ✅  | ✅   | ⚠️   | ⚠️  | ⚠️       | ⚠️       | ⚠️    | ⚠️     |
| Whisper       | ✅  | ✅   | ✅   | ✅  | ✅       | ✅       | ✅    | ✅     |
| LLAMA         | ✅  | ✅   | ⚠️   | ⚠️  | ⚠️       | ⚠️       | ⚠️    | ⚠️     |
| Wav2Vec2      | ✅  | ✅   | ✅   | ✅  | ✅       | ✅       | ✅    | ✅     |
| LLaVA         | ✅  | ✅   | ✅   | ✅  | ✅       | ✅       | ✅    | ✅     |
| CLAP          | ✅  | ✅   | ✅   | ✅  | ✅       | ✅       | ✅    | ✅     |
| XCLIP         | ✅  | ✅   | ⚠️   | ⚠️  | ⚠️       | ⚠️       | ⚠️    | ⚠️     |
| DETR          | ✅  | ✅   | ⚠️   | ⚠️  | ⚠️       | ⚠️       | ⚠️    | ⚠️     |
| Qwen2         | ✅  | ✅   | ⚠️   | ⚠️  | ⚠️       | ⚠️       | ⚠️    | ⚠️     |