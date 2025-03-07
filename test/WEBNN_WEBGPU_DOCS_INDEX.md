# WebNN and WebGPU Documentation Index

Last updated: 2025-03-07 01:30:00

## Current Documentation

### Comprehensive Guides

- [WEBNN_WEBGPU_QUANTIZATION_GUIDE.md](WEBNN_WEBGPU_QUANTIZATION_GUIDE.md): **NEW** Complete guide to WebNN and WebGPU quantization
- [WEBNN_WEBGPU_QUANTIZATION_MARCH2025_UPDATE.md](WEBNN_WEBGPU_QUANTIZATION_MARCH2025_UPDATE.md): March 2025 enhancements with experimental precision
- [WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md](WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md): Technical summary of quantization capabilities

### Command Reference

- [WEBNN_WEBGPU_QUANTIZATION_README.md](WEBNN_WEBGPU_QUANTIZATION_README.md): Quick reference for command-line usage

## Archived Documentation

The following documentation has been archived and replaced with the comprehensive real implementation testing guides above:

- [WEBGPU WEBNN QUANTIZATION SUMMARY](archived_md_files/WEBGPU_WEBNN_QUANTIZATION_SUMMARY_20250307_012440.md): Archived on 2025-03-07
- [WEBNN WEBGPU QUANTIZATION README](archived_md_files/WEBNN_WEBGPU_QUANTIZATION_README_20250307_012440.md): Archived on 2025-03-07
- [WEB PLATFORM INTEGRATION GUIDE UPDATED](archived_md_files/WEB_PLATFORM_INTEGRATION_GUIDE_UPDATED_20250307_012440.md): Archived on 2025-03-07
- [WEB PLATFORM OPTIMIZATION GUIDE JUNE2025](archived_md_files/WEB_PLATFORM_OPTIMIZATION_GUIDE_JUNE2025_20250307_012440.md): Archived on 2025-03-07
- [WEB PLATFORM SUPPORT COMPLETED](archived_md_files/WEB_PLATFORM_SUPPORT_COMPLETED_20250307_012440.md): Archived on 2025-03-07

## Implementation Status

### Browser Support Status


| Browser | WebNN Support | WebGPU Support | Best Use Case |
|---------|--------------|----------------|---------------|
| Chrome | ⚠️ Limited | ✅ Good | General WebGPU |
| Edge | ✅ Excellent | ✅ Good | WebNN acceleration |
| Firefox | ❌ Poor | ✅ Excellent | Audio models with WebGPU |
| Safari | ⚠️ Limited | ⚠️ Limited | Metal API integration |

### Precision Support


| Precision | WebNN | WebGPU | Memory Reduction | Use Case |
|-----------|-------|--------|------------------|----------|
| 2-bit | ❌ Not Supported | ✅ Supported | ~87.5% | Ultra memory constrained |
| 3-bit | ❌ Not Supported | ✅ Supported | ~81.25% | Very memory constrained |
| 4-bit | ⚠️ Experimental | ✅ Supported | ~75% | Memory constrained |
| 8-bit | ✅ Supported | ✅ Supported | ~50% | General purpose |
| 16-bit | ✅ Supported | ✅ Supported | ~0% | High accuracy |
| 32-bit | ✅ Supported | ✅ Supported | 0% | Maximum accuracy |
