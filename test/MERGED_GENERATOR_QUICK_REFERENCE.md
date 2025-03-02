# Merged Test Generator Quick Reference

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

## Common Workflows

### Find and Generate Missing Tests

```bash
# 1. List missing test implementations
python merged_test_generator.py --generate-missing --list-only

# 2. Generate high priority tests
python merged_test_generator.py --generate-missing --high-priority-only

# 3. Generate tests for specific category
python merged_test_generator.py --generate-missing --category vision
```

### Export and Analyze Model Registry

```bash
# 1. Export registry to parquet
python merged_test_generator.py --export-registry

# 2. Analyze with pandas
import pandas as pd
df = pd.read_parquet("model_registry.parquet")
print(df.groupby("category").count())
```

### Focused Testing

```bash
# Generate tests for language models only
python merged_test_generator.py --generate-missing --category language --limit 5

# Generate tests for specific models
python merged_test_generator.py --batch-generate bert,gpt2,t5,vit,clip
```

## Hardware Support

| Platform | Detection Method | Environment Variable |
|----------|------------------|---------------------|
| CPU | Always available | N/A |
| CUDA | `torch.cuda.is_available()` | `CUDA_VISIBLE_DEVICES` |
| OpenVINO | `import openvino` | N/A |
| MPS | `torch.mps.is_available()` | N/A |

## Input/Output

### Inputs

- **Model Registry**: Dictionary mapping model families to configurations
- **Test Templates**: Specialized templates for different model categories
- **Pipeline Tasks**: Tasks supported by each model type
- **Model Categories**: language, vision, audio, multimodal, specialized

### Outputs

- **Test Files**: Python files ready to test specific models
- **Parquet File**: Structured registry data in parquet format 
- **Status Reports**: Information about generated tests

## Model Categories

| Category | Example Tasks |
|----------|---------------|
| language | text-generation, fill-mask, summarization |
| vision | image-classification, object-detection |
| audio | automatic-speech-recognition, audio-classification |
| multimodal | image-to-text, visual-question-answering |
| specialized | protein-folding, time-series-prediction |