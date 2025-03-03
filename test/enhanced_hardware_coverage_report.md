# High Priority Models Hardware Coverage Report

Generated: 2025-03-02 22:36:25

## Summary

- Processed: 13/13 high priority models
- Missing test files: 0/13 models

## Hardware Coverage Matrix

| Model | CPU | CUDA | OpenVINO | MPS | ROCm | WebNN | WebGPU | Status | Files |
|-------|-----|------|----------|-----|------|-------|--------|--------|-------|
| bert | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| t5 | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| llama | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| clip | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| vit | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| clap | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| whisper | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| wav2vec2 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| llava | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| llava_next | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| xclip | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| qwen2 | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |
| detr | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | Partial | 1 file |

## Detailed Results

### bert

**File:** test_hf_bert.py

Updates made:
- Added openvino imports
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added openvino text implementation
- Added rocm text implementation
- Added webnn text implementation
- Added webgpu text implementation

### t5

**File:** test_hf_t5.py

Updates made:
- Added openvino imports
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added openvino vision implementation
- Added rocm vision implementation
- Added webnn vision implementation
- Added webgpu vision implementation

### llama

**File:** test_hf_llama.py

Updates made:
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added rocm audio implementation
- Added webnn audio implementation
- Added webgpu audio implementation

### clip

**File:** test_hf_clip.py

Updates made:
- Added openvino imports
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added openvino vision implementation
- Added rocm vision implementation
- Added webnn vision implementation
- Added webgpu vision implementation

### vit

**File:** test_hf_vit.py

Updates made:
- Added openvino imports
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added openvino vision implementation
- Added rocm vision implementation
- Added webnn vision implementation
- Added webgpu vision implementation

### clap

**File:** test_hf_clap.py

Updates made:
- Added rocm vision implementation

### whisper

**File:** test_hf_whisper.py

Updates made:
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added rocm audio implementation
- Added webnn audio implementation
- Added webgpu audio implementation

### wav2vec2

**File:** test_hf_wav2vec2.py

Updates made:
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added rocm audio implementation
- Added webnn audio implementation
- Added webgpu audio implementation

### llava

**File:** test_hf_llava.py

Updates made:
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added rocm audio implementation
- Added webnn audio implementation
- Added webgpu audio implementation

### llava_next

**File:** test_hf_llava_next.py

Updates made:
- Added rocm vision implementation
- Added webnn vision implementation
- Added webgpu vision implementation

### xclip

**File:** test_hf_xclip.py

Updates made:
- Added openvino imports
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added openvino vision implementation
- Added rocm vision implementation
- Added webnn vision implementation
- Added webgpu vision implementation

### qwen2

**File:** test_hf_qwen2.py

Updates made:
- Added openvino imports
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added openvino text implementation
- Added rocm text implementation
- Added webnn text implementation
- Added webgpu text implementation

### detr

**File:** test_hf_detr.py

Updates made:
- Added rocm imports
- Added webnn imports
- Added webgpu imports
- Added rocm audio implementation
- Added webnn audio implementation
- Added webgpu audio implementation

