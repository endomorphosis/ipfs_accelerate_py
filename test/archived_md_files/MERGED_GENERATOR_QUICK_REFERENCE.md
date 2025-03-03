# Merged Test Generator Quick Reference (Updated)

## Key Commands

| Command | Description |
|---------|-------------|
| `--help` | Show help |
| `--list-families` | List all model families in registry |
| `--generate <model>` | Generate test file for a specific model |
| `--all` | Generate tests for all model families |
| `--batch-generate <models>` | Generate tests for comma-separated list of models |
| `--generate-missing` | Generate tests for missing model implementations |
| `--export-registry` | Export model registry to parquet format |
| `--web-platforms` | Generate tests with WebNN and WebGPU support |
| `--detailed-performance` | Enable detailed performance metrics collection |

## Common Workflows

### Find and Generate Missing Tests

```bash
# 1. List missing test implementations
python merged_test_generator.py --generate-missing --list-only

# 2. Generate high priority tests
python merged_test_generator.py --generate-missing --high-priority-only

# 3. Generate tests for specific category
python merged_test_generator.py --generate-missing --category vision

# 4. Generate tests with web platform support
python merged_test_generator.py --generate bert --web-platforms
```

### Export and Analyze Model Registry

```bash
# 1. Export registry to parquet
python merged_test_generator.py --export-registry

# 2. Analyze with pandas
import pandas as pd
df = pd.read_parquet("model_registry.parquet")
print(df.groupby("category").count())

# 3. Export with DuckDB (alternative)
python merged_test_generator.py --export-registry --use-duckdb
```

### Focused Testing

```bash
# Generate tests for language models only
python merged_test_generator.py --generate-missing --category language --limit 5

# Generate tests for specific models
python merged_test_generator.py --batch-generate bert,gpt2,t5,vit,clip

# Generate tests with web platform support
python merged_test_generator.py --batch-generate bert,vit --web-platforms
```

## Hardware Platform Support

| Platform | Detection Method | Environment Variable | Web Export |
|----------|------------------|---------------------|------------|
| CPU | Always available | N/A | N/A |
| CUDA | `torch.cuda.is_available()` | `CUDA_VISIBLE_DEVICES` | N/A |
| OpenVINO | `import openvino` | N/A | N/A |
| MPS | `torch.backends.mps.is_available()` | N/A | N/A |
| ROCm | `torch.cuda` (with AMD GPU) | `HIP_VISIBLE_DEVICES` | N/A |
| Qualcomm | `import qai_hub` | N/A | N/A |
| WebNN | ONNX-based model export | N/A | Yes (ONNX → WebNN) |
| WebGPU | transformers.js integration | N/A | Yes (ONNX → WebGPU) |

## Web Platform Compatibility

The generator now automatically analyzes models for web compatibility:

```bash
# Generate a test with full web platform support
python merged_test_generator.py --generate bert --web-platforms

# Test web platform exports
python test_hf_bert.py --platform webnn 
python test_hf_bert.py --platform webgpu

# Run comprehensive tests including all web platforms
python test_hf_bert.py --platform all
```

## Performance Metrics

Generated tests now collect detailed performance metrics:

1. **Inference Time**: Milliseconds per inference
2. **Memory Usage**: Peak and allocated memory
3. **Throughput**: Items processed per second
4. **Batch Performance**: Scaling efficiency with batch size
5. **Web Platform Metrics**: Browser-specific performance

## Input/Output

### Inputs

- **Model Registry**: Dictionary mapping model families to configurations
- **Test Templates**: Specialized templates for different model categories
- **Pipeline Tasks**: Tasks supported by each model type
- **Model Categories**: language, vision, audio, multimodal, specialized
- **Web Compatibility**: Automatic detection of web-compatible models

### Outputs

- **Test Files**: Python files ready to test specific models
- **Parquet File**: Structured registry data in parquet format 
- **Status Reports**: Information about generated tests
- **Performance Reports**: Detailed timing and resource metrics
- **Web Export Files**: ONNX models and browser-ready code

## Model Categories

| Category | Example Tasks | Web Compatibility |
|----------|---------------|------------------|
| language | text-generation, fill-mask, summarization | Medium-High |
| vision | image-classification, object-detection | Medium |
| audio | automatic-speech-recognition, audio-classification | Low |
| multimodal | image-to-text, visual-question-answering | Very Low |
| specialized | protein-folding, time-series-prediction | Model-dependent |

## Troubleshooting Web Exports

If you encounter issues with web platform testing:

1. Ensure ONNX and ONNX Runtime are installed: `pip install onnx onnxruntime`
2. For WebNN testing, use: `pip install onnx-web`
3. For transformers.js testing, ensure Node.js is available
4. Use `--skip-web-tests` to skip problematic web platform tests
5. Check browser compatibility in the generated test output