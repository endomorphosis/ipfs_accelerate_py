# WebNN/WebGPU Sample Model Compatibility Report

## Results Summary

| Model | Type | WebGPU 8-bit | WebGPU 4-bit | WebNN 8-bit | WebNN 4-bit |
|-------|------|-------------|-------------|------------|------------|
| bert-base-uncased | text | ✅ | ✅ | ✅ | ✅ |
| whisper-tiny | audio | ✅ | ✅ | ✅ | ✅ |
| vit-base-patch16-224 | vision | ✅ | ✅ | ✅ | ✅ |

## Simulation Detection

R = Real hardware, S = Simulation

| Model | WebGPU 8-bit | WebGPU 4-bit | WebNN 8-bit | WebNN 4-bit |
|-------|-------------|-------------|------------|------------|
| bert-base-uncased | S | S | S | S |
| whisper-tiny | S | S | S | S |
| vit-base-patch16-224 | S | S | S | S |
