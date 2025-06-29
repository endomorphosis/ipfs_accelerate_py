# Code Analysis Summary

## Overview

- **Total Files Analyzed**: 18
- **Total Classes**: 14
- **Total Functions**: 10

## Most Complex Files

| File | Complexity |
|------|------------|
| analyze_ast.py | 72 |
| utils/hardware_detection.py | 64 |
| __main__.py | 39 |
| backends/onnx_to_openvino.py | 39 |
| backends/pytorch_to_onnx.py | 37 |
| backends/onnx_to_webgpu.py | 35 |
| utils/verification.py | 30 |
| backends/onnx_to_webnn.py | 26 |
| utils/file_management.py | 19 |
| utils/logging_utils.py | 18 |

## Most Common Imports

| Module | Usage Count |
|--------|-------------|
| os | 13 |
| logging | 12 |
| typing.Dict | 12 |
| typing.Any | 12 |
| typing.List | 11 |
| typing.Optional | 11 |
| typing.Tuple | 10 |
| onnx | 9 |
| torch | 8 |
| json | 6 |
| tempfile | 6 |
| shutil | 5 |
| subprocess | 4 |
| platform | 4 |
| core.converter.ModelConverter | 4 |
| core.converter.ConversionResult | 4 |
| core.registry.register_converter | 4 |
| sys | 3 |
| openvino.runtime.Core | 3 |
| openvino.inference_engine.IECore | 3 |

## Refactoring Recommendations

1. Create unified ModelConverter base class for all format converters
2. Implement centralized hardware detection in HardwareDetector
3. Standardize model file verification with ModelVerifier
4. Create registry pattern for converter discoverability
5. Unify logging and error handling across converters
6. Implement caching system for converted models