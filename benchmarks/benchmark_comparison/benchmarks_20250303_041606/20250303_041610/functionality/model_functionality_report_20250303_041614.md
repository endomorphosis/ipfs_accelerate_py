# Model Functionality Verification Report

Generated: 2025-03-03 04:16:14

## Summary

- **Total Tests**: 13
- **Successful Tests**: 0
- **Failed Tests**: 13
- **Models Tested**: 13
- **Hardware Platforms**: cpu
- **Overall Success Rate**: 0.0%

## Results by Model and Hardware

| Model | cpu | Overall |
|-------|---------|--------|
| bert | ❌ | 0.0% |
| clap | ❌ | 0.0% |
| clip | ❌ | 0.0% |
| detr | ❌ | 0.0% |
| llama | ❌ | 0.0% |
| llava | ❌ | 0.0% |
| llava_next | ❌ | 0.0% |
| qwen2 | ❌ | 0.0% |
| t5 | ❌ | 0.0% |
| vit | ❌ | 0.0% |
| wav2vec2 | ❌ | 0.0% |
| whisper | ❌ | 0.0% |
| xclip | ❌ | 0.0% |

## Detailed Results

### detr on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_detr.py", line 277
    def init_rocm(self, model_name=None, device="hip"):
                                                       ^
IndentationError: unindent does not match any outer indentation level

```

### clip on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_clip.py", line 528
    """Test the model using OpenVINO integration."""
    ^
IndentationError: expected an indented block after function definition on line 527

```

### bert on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_bert.py", line 562
    """Test the model using OpenVINO integration."""
    ^
IndentationError: expected an indented block after function definition on line 561

```

### clap on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_clap.py", line 1012
    def enhanced_cuda_handler(audio_input=None, text=None):
SyntaxError: expected 'except' or 'finally' block

```

### llava on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_llava.py", line 257
    def init_rocm(self, model_name=None, device="hip"):
IndentationError: unexpected indent

```

### llama on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_llama.py", line 277
    def init_rocm(self, model_name=None, device="hip"):
                                                       ^
IndentationError: unindent does not match any outer indentation level

```

### qwen2 on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_qwen2.py", line 518
    """Test the model using OpenVINO integration."""
    ^
IndentationError: expected an indented block after function definition on line 517

```

### llava_next on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_llava_next.py", line 1002
    def init_webgpu(self, model_name=None):
IndentationError: unexpected indent

```

### t5 on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_t5.py", line 500
    """Test the model using OpenVINO integration."""
    ^
IndentationError: expected an indented block after function definition on line 499

```

### vit on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_vit.py", line 505
    """Test the model using OpenVINO integration."""
    ^
IndentationError: expected an indented block after function definition on line 504

```

### wav2vec2 on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_wav2vec2.py", line 277
    def init_rocm(self, model_name=None, device="hip"):
                                                       ^
IndentationError: unindent does not match any outer indentation level

```

### whisper on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_whisper.py", line 283
    def init_rocm(self, model_name=None, device="hip"):
                                                       ^
IndentationError: unindent does not match any outer indentation level

```

### xclip on cpu ❌

**Error Output**:
```
  File "/home/barberb/ipfs_accelerate_py/test/skills/test_hf_xclip.py", line 164
    X-CLIP_MODELS_REGISTRY = {
    ^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: cannot assign to expression here. Maybe you meant '==' instead of '='?

```


## Next Steps

1. Investigate and fix the failing tests
2. Re-run the verification to confirm fixes
3. Run performance benchmarks on successfully verified models
