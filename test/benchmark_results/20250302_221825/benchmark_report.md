# Model Benchmark Report

Generated: 2025-03-02 22:19:00

## Summary

### Hardware Platforms

| Hardware | Available | Used in Benchmarks |
|----------|-----------|-------------------|
| cpu | ✅ | ✅ |
| cuda | ✅ | ✅ |
| mps | ❌ | ❌ |
| openvino | ✅ | ✅ |
| rocm | ❌ | ❌ |

### Models Tested

| Model Key | Full Name | Family | Size | Modality |
|-----------|-----------|--------|------|----------|
| bert | prajjwal1/bert-tiny | embedding | tiny | text |
| t5 | google/t5-efficient-tiny | text_generation | tiny | text |
| vit | facebook/deit-tiny-patch16-224 | vision | tiny | vision |
| whisper | openai/whisper-tiny | audio | tiny | audio |
| clip | openai/clip-vit-base-patch32 | multimodal | base | multimodal |

## Functionality Verification Results

### Success Rates by Hardware

| Hardware | Success Rate | Models Tested | Successful | Failed |
|----------|--------------|---------------|------------|--------|
| cpu | 0.0% | 5 | 0 | 5 |
| cuda | 0.0% | 5 | 0 | 5 |
| openvino | 0.0% | 5 | 0 | 5 |

### Results by Model and Hardware

| Model | cpu | cuda | openvino |
|-------|---|---|---|
| bert | ✅ | ✅ | ✅ |
| t5 | ✅ | ✅ | ✅ |
| vit | ✅ | ✅ | ✅ |
| whisper | ✅ | ✅ | ✅ |
| clip | ✅ | ✅ | ✅ |

### Failed Tests

#### cpu

- **vit**: Unknown error
- **bert**: Unknown error
- **t5**: Unknown error
- **whisper**: Unknown error
- **clip**: Unknown error

#### cuda

- **vit**: Unknown error
- **bert**: Unknown error
- **t5**: Unknown error
- **whisper**: Unknown error
- **clip**: Unknown error

#### openvino

- **vit**: Unknown error
- **bert**: Unknown error
- **t5**: Unknown error
- **whisper**: Unknown error
- **clip**: Unknown error

## Performance Benchmark Results

## Hardware Compatibility Matrix

### Model Family Compatibility

| Model Family | cpu | cuda | openvino |
|--------------|---|---|---|
| multimodal | ❌ | ❌ | ❌ |
| embedding | ❌ | ❌ | ❌ |
| audio | ❌ | ❌ | ❌ |
| text_generation | ❌ | ❌ | ❌ |
| vision | ❌ | ❌ | ❌ |

### Hardware-Specific Issues

No specific issues identified.

## Recommendations

### Hardware Selection


### Model Selection


### Performance Optimization

- Consider using smaller batch sizes for latency-sensitive applications
- Increase batch sizes for throughput-oriented workloads
- Enable model caching when running multiple inferences with the same model

### Compatibility Issues

- multimodal models are not compatible with: cpu, cuda, openvino
- embedding models are not compatible with: cpu, cuda, openvino
- audio models are not compatible with: cpu, cuda, openvino
- text_generation models are not compatible with: cpu, cuda, openvino
- vision models are not compatible with: cpu, cuda, openvino

## Next Steps

1. Investigate and fix the failing tests
2. Re-run the verification to confirm fixes
3. Run performance benchmarks on successfully verified models

---

Generated by Model Benchmark Runner on 2025-03-02 22:19:00
