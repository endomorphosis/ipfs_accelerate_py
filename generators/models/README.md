# Models Directory

This directory contains model implementation files for various AI model types supported by the IPFS Accelerate framework.

## Directory Structure

- **skills/**: Contains skill implementation files for specific models
- **advanced/**: Contains advanced model implementations with specialized features

## Model Implementations

The model files in this directory provide Python implementations for working with various AI models:

- BERT and text embedding models
- Vision models (ViT, CLIP, etc.)
- Audio models (Whisper, Wav2Vec2, CLAP)
- Multimodal models (LLaVA, CLIP, etc.)

## Usage

To use these model implementations, import them directly:

```python
from generators.models.skill_hf_bert import skill_hf_bert

# Initialize the model
model = skill_hf_bert()

# Use the model
result = model.infer("Sample text")
```

These model implementations are hardware-aware and can utilize various hardware platforms (CPU, CUDA, ROCm, MPS, OpenVINO, QNN, WebNN, WebGPU).