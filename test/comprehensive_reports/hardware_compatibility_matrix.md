# Hardware Compatibility Matrix

Error generating matrix: Binder Error: Table "cpc" does not have a column named "model_id"

Candidate bindings: : "model_name", "model_family"

LINE 18:     models m ON cpc.model_id = m.model_id
                         ^

Using information from CLAUDE.md instead.

## Model Family-Based Compatibility Chart

| Model Family | CPU | CUDA | ROCm | MPS | OpenVINO | QNN | WebNN | WebGPU | Notes |
|--------------|-----|------|------|-----|----------|-----|---------|-------|--------|
| Embedding | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Fully supported on all hardware |
| Text Generation | ✅ Medium | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | Memory requirements critical |
| Vision | ✅ Medium | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | Full cross-platform support |
| Audio | ✅ Medium | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ⚠️ Limited | ⚠️ Limited | CUDA preferred, Web simulation added |
| Multimodal | ⚠️ Limited | ✅ High | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | CUDA for production, others are limited |
