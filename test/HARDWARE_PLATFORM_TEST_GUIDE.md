# Hardware Platform Test Coverage Guide (Updated March 2025)

This guide provides a comprehensive approach to ensuring complete test coverage for all key model types across all supported hardware platforms. It accompanies the Phase 15 implementation plan in CLAUDE.md and the tools developed to execute this plan.

**Important Note:** This guide should be used in conjunction with the [Hardware Model Validation Guide](./HARDWARE_MODEL_VALIDATION_GUIDE.md) which explains how to track and report testing results.

## Goals

1. Achieve complete test coverage for all 13 key HuggingFace model classes across all hardware platforms
2. Create comprehensive benchmarks for all model-hardware combinations
3. Document hardware-specific optimizations and workarounds
4. Establish a standardized testing methodology for ongoing verification

## Key HuggingFace Model Classes

| Model Class | Primary Use Case | Representative Models |
|-------------|------------------|----------------------|
| BERT | Embedding | bert-base-uncased, prajjwal1/bert-tiny |
| T5 | Text Generation | t5-small, google/t5-efficient-tiny | 
| LLAMA | Text Generation | facebook/opt-125m |
| CLIP | Vision-Text | openai/clip-vit-base-patch32 |
| ViT | Vision | google/vit-base-patch16-224 |
| CLAP | Audio-Text | laion/clap-htsat-unfused |
| Whisper | Audio | openai/whisper-tiny |
| Wav2Vec2 | Audio | facebook/wav2vec2-base |
| LLaVA | Multimodal | llava-hf/llava-1.5-7b-hf |
| LLaVA-Next | Multimodal | llava-hf/llava-v1.6-34b-hf |
| XCLIP | Video | microsoft/xclip-base-patch32 |
| Qwen2/3 | Text Generation | Qwen/Qwen2-7B-Instruct, Qwen/Qwen2-VL-Chat |
| DETR | Vision | facebook/detr-resnet-50 |

## Supported Hardware Platforms

| Platform | Description | CLI Flag |
|----------|-------------|----------|
| CPU | Generic CPU implementation | `--device cpu` |
| CUDA | NVIDIA GPU implementation | `--device cuda` |
| ROCm | AMD GPU implementation | `--device rocm` |
| MPS | Apple Silicon GPU implementation | `--device mps` |
| OpenVINO | Intel hardware acceleration | `--device openvino` |
| WebNN | Web Neural Network API | `--web-platform webnn` |
| WebGPU | Web GPU API | `--web-platform webgpu` |

## Testing Tools

Three primary tools have been developed for hardware platform testing:

1. **test_comprehensive_hardware_coverage.py** - Core testing framework for hardware coverage
2. **run_comprehensive_hardware_tests.sh** - Bash script for executing test batches
3. **benchmark_hardware_models.py** - Performance benchmarking tool

### Using the Test Coverage Tool

```bash
# Generate a compatibility report
python test/test_comprehensive_hardware_coverage.py --report

# Run tests for all mock implementations (Phase 1)
python test/test_comprehensive_hardware_coverage.py --phase 1

# Test a specific model across all hardware platforms
python test/test_comprehensive_hardware_coverage.py --model bert

# Test a specific hardware platform across all models
python test/test_comprehensive_hardware_coverage.py --hardware cuda
```

### Using the Benchmark Tool

```bash
# Benchmark a specific model across all hardware platforms
python test/benchmark_hardware_models.py --model bert

# Benchmark all models on a specific hardware platform
python test/benchmark_hardware_models.py --hardware cuda

# Benchmark all models of a specific category
python test/benchmark_hardware_models.py --category vision

# Quick benchmark with reduced test parameters
python test/benchmark_hardware_models.py --all --quick

# Specify custom batch sizes
python test/benchmark_hardware_models.py --model t5 --batch-sizes 1,2,4,8,16
```

## Implementation Phases

The hardware coverage completion plan is divided into five phases:

### Phase 1: Fix Mock Implementations (Priority: High)

Several model-hardware combinations currently use mock implementations. These need to be replaced with real implementations:

- T5 on OpenVINO
- CLAP on OpenVINO
- Wav2Vec2 on OpenVINO
- LLaVA on OpenVINO
- Whisper on WebNN/WebGPU
- Qwen2/3 on AMD, MPS, and OpenVINO

Implementation steps:
1. Run `python test/test_comprehensive_hardware_coverage.py --phase 1` to identify all mock implementations
2. For each mock implementation, create a proper test implementation
3. Update the implementation status in the model-hardware compatibility matrix

### Phase 2: Expand Multimodal Support (Priority: Medium)

Multimodal models (particularly LLaVA and LLaVA-Next) have limited hardware support. This phase aims to:

1. Investigate feasibility of LLaVA support on:
   - AMD ROCm platform
   - Apple MPS platform
   - OpenVINO platform
2. Implement dedicated tests for multimodal models on supported hardware
3. Create fallback mechanisms for hardware platforms with limited capabilities

### Phase 3: Web Platform Extension (Priority: Medium)

Currently, WebNN and WebGPU support is limited to text and vision models. This phase will:

1. Improve WebNN/WebGPU support:
   - Implement non-simulation Whisper tests
   - Add web platform tests for XCLIP
   - Add web platform tests for DETR
2. Create audio model tests specifically for web platforms
3. Document browser-specific compatibility requirements

### Phase 4: Comprehensive Benchmarking (Priority: Medium)

Standardize performance metrics across all model-hardware combinations:

1. Standardize benchmarking methodology across all platforms
2. Create central database of benchmarking results
3. Implement automated comparison reporting
4. Add memory usage profiling for each model-hardware combination

### Phase 5: Edge Case Handling (Priority: Low)

Improve reliability for edge cases and document workarounds:

1. Identify and document hardware-specific edge cases
2. Create specialized tests for edge conditions
3. Implement automatic fallback mechanisms
4. Document workarounds for inherent hardware limitations

## Best Practices

When implementing tests for specific hardware platforms:

1. **Resource Management**
   - Release resources properly after tests
   - Close models and free memory explicitly
   - Use context managers when available
   - Set appropriate batch sizes based on hardware capabilities

2. **Hardware-Specific Optimizations**
   - CUDA: Use mixed precision when appropriate
   - OpenVINO: Use specialized optimizations
   - WebNN/WebGPU: Minimize model size and complexity
   - ROCm: Test with multiple batch sizes for performance

3. **Error Handling**
   - Implement graceful degradation for unsupported operations
   - Provide clear error messages for hardware-specific failures
   - Include fallback paths for critical functionality

4. **Documentation**
   - Document hardware-specific limitations
   - Note performance characteristics in benchmark reports
   - Include minimum hardware requirements for each model

## Reporting

The testing framework includes a comprehensive validation and reporting system implemented in `model_hardware_validation_tracker.py`. This system:

1. Maintains a centralized database of all test results
2. Tracks implementation status for each model-hardware combination
3. Records performance metrics and hardware requirements
4. Documents known issues and their workarounds
5. Generates detailed reports and visualizations

### Validation Database Structure

The validation system uses a JSON database with the following structure:
```json
{
  "metadata": {
    "created": "timestamp",
    "last_updated": "timestamp",
    "version": "1.0",
    "update_count": 0
  },
  "models": {
    "bert": {
      "name": "BERT",
      "models": ["bert-base-uncased", "bert-tiny"],
      "category": "embedding",
      "last_updated": "timestamp"
    },
    // Other models...
  },
  "hardware": {
    "cuda": {
      "name": "CUDA",
      "flag": "--device cuda",
      "last_updated": "timestamp"
    },
    // Other hardware platforms...
  },
  "validation_results": {
    "bert": {
      "cuda": {
        "status": "pass",
        "implementation_type": "real",
        "last_test_date": "timestamp",
        "test_history": [
          {"date": "timestamp", "status": "pass", "implementation_type": "real"}
        ],
        "known_issues": [],
        "performance": {
          "timestamp": {
            "throughput": 125.5,
            "latency_ms": 7.9,
            "memory_usage_mb": 350.2,
            "batch_size": 1
          }
        },
        "requirements": {
          "memory": 500,
          "cpu": "2+ cores",
          "compute_capability": "CUDA 11.0+",
          "vram": "600 MB VRAM"
        },
        "notes": "All tests passed successfully"
      },
      // Other hardware platforms...
    },
    // Other models...
  }
}
```

### Where Results Are Saved

When running comprehensive tests, results should be saved to:

- `validation_reports/` - For validation reports
- `validation_visualizations/` - For validation visualizations
- `hardware_compatibility_reports/` - For compatibility reports
- `benchmark_results/` - For performance benchmarking data

### Report Contents

The reports include:
- Implementation status for each model-hardware combination
- Performance metrics for successful implementations
- Identified issues and potential solutions for failures
- Recommendations for hardware-specific optimizations
- Visual representations of test coverage and performance

### Generating Reports

To generate a comprehensive validation report:
```bash
python test/model_hardware_validation_tracker.py --generate-report
```

To create visualizations of validation status:
```bash
python test/model_hardware_validation_tracker.py --visualize
```