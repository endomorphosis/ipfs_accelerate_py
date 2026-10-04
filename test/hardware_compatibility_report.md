# Hardware Compatibility Report (JSON Database)

Generated: 2025-03-10 00:59:33

Total templates analyzed: 26

## Overall Platform Support

| Hardware Platform | Templates | Percentage |
|-------------------|-----------|------------|
| cuda | 25 | 96.2% |
| cpu | 22 | 84.6% |
| mps | 22 | 84.6% |
| rocm | 3 | 11.5% |
| openvino | 22 | 84.6% |
| qualcomm | 13 | 50.0% |
| webnn | 3 | 11.5% |
| webgpu | 3 | 11.5% |

## Compatibility Matrix by Model Type

| Model Type | Count | cuda | cpu | mps | rocm | openvino | qualcomm | webnn | webgpu |
|------------|-------|---|---|---|---|---|---|---|---|
| video | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| vision | 2 | ✅ 100.0% | ✅ 100.0% | ⚠️ 50.0% | ❌ 0.0% | ✅ 100.0% | ⚠️ 50.0% | ❌ 0.0% | ❌ 0.0% |
| text_embedding | 2 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ⚠️ 50.0% | ❌ 0.0% | ❌ 0.0% |
| llava | 2 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| whisper | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| cpu | 1 | ❌ 0.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| clap | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| audio | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| llama | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| t5 | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| xclip | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| clip | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| bert | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| test | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| inheritance | 1 | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| selection | 1 | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% |
| validator | 1 | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% |
| wav2vec2 | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| detr | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |
| qwen2 | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| verifier | 1 | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% | ❌ 0.0% |
| hardware | 1 | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% |
| vit | 1 | ✅ 100.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ✅ 100.0% | ✅ 100.0% | ❌ 0.0% | ❌ 0.0% |

## Highly Compatible Model-Hardware Pairs

These combinations have >75% compatibility:

### video
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%

### vision
- cuda: 100.0%
- cpu: 100.0%
- openvino: 100.0%

### text_embedding
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%

### llava
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%

### whisper
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### cpu
- cpu: 100.0%

### clap
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### audio
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### llama
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%

### t5
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%

### xclip
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%

### clip
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### bert
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### test
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### inheritance
- cuda: 100.0%
- cpu: 100.0%

### selection
- cuda: 100.0%
- mps: 100.0%
- rocm: 100.0%
- webnn: 100.0%
- webgpu: 100.0%

### validator
- cuda: 100.0%
- mps: 100.0%
- rocm: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%
- webnn: 100.0%
- webgpu: 100.0%

### wav2vec2
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### detr
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%

### qwen2
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%

### verifier
- cuda: 100.0%

### hardware
- cuda: 100.0%
- mps: 100.0%
- rocm: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%
- webnn: 100.0%
- webgpu: 100.0%

### vit
- cuda: 100.0%
- cpu: 100.0%
- mps: 100.0%
- openvino: 100.0%
- qualcomm: 100.0%


## Improvement Opportunities

These combinations have <25% compatibility and need improvement:

### video
- rocm: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### vision
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### text_embedding
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### llava
- rocm: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### whisper
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### cpu
- cuda: 0.0%
- mps: 0.0%
- rocm: 0.0%
- openvino: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### clap
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### audio
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### llama
- rocm: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### t5
- rocm: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### xclip
- rocm: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### clip
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### bert
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### test
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### inheritance
- mps: 0.0%
- rocm: 0.0%
- openvino: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### selection
- cpu: 0.0%
- openvino: 0.0%
- qualcomm: 0.0%

### validator
- cpu: 0.0%

### wav2vec2
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### detr
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### qwen2
- rocm: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### verifier
- cpu: 0.0%
- mps: 0.0%
- rocm: 0.0%
- openvino: 0.0%
- qualcomm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

### hardware
- cpu: 0.0%

### vit
- rocm: 0.0%
- webnn: 0.0%
- webgpu: 0.0%

