# Hardware Compatibility Report
Date: 2025-03-03 04:17:02

## Summary

- Critical Errors: 0
- Errors: 1
- Warnings: 0
- Informational: 1

## Components Checked

- ✅ ResourcePool
- ✅ HardwareDetection
- ✅ ModelFamilyClassifier
- ✅ HardwareModelIntegration

## Models Tested

- bert-base-uncased
- facebook/bart-base
- gpt2
- openai/whisper-tiny
- t5-small
- vit-base-patch16-224

## Hardware Compatibility Matrix

| Model Family | CUDA | ROCM | MPS | OPENVINO | WEBNN | WEBGPU | CPU |
|--------------|------|------|-----|----------|-------|--------|-----|
| Embedding | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Text Generation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vision | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Audio | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multimodal | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Legend:
- ✅ Compatible - No issues detected
- ⚠️ Partially Compatible - Some issues may occur
- ❌ Incompatible - Critical issues prevent operation

## Error Issues (1)

### UNKNOWN: detection_exception

- **Component**: hardware_detection
- **Model**: None
- **Message**: 'HardwareDetector' object has no attribute 'detect_hardware_with_comprehensive_checks'

**Recommendations**:

- Check hardware compatibility and system requirements

<details>
<summary>Error Details</summary>

```
Traceback (most recent call last):
  File "/home/barberb/ipfs_accelerate_py/test/hardware_compatibility_reporter.py", line 161, in collect_hardware_detection_errors
    hw_info = detector.detect_hardware_with_comprehensive_checks()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'HardwareDetector' object has no attribute 'detect_hardware_with_comprehensive_checks'

```
</details>


## Info Issues (1)

### ALL: low_memory_mode

- **Component**: resource_pool
- **Model**: None
- **Message**: System is operating in low memory mode

**Recommendations**:

- Check hardware compatibility and system requirements
