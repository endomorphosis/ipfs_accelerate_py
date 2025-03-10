# Generated Skillsets

This directory contains Python skill implementation files that have been generated from templates for various AI models.

## Skill Implementations

These skill files provide ready-to-use implementations for:

1. Model initialization with configurable parameters
2. Hardware acceleration support
3. Inference methods for various input types
4. Error handling and cross-platform compatibility

## Usage

You can use these skill implementations directly:

```python
from generators.generated_skillsets.skill_hf_bert import skill_hf_bert

# Initialize skill
skill = skill_hf_bert()

# Run inference
result = skill.infer("Sample text")
```

## Skill Organization

Skills are organized by model type:

- Text models (BERT, T5, LLAMA, etc.)
- Vision models (ViT, CLIP, etc.)
- Audio models (Whisper, Wav2Vec2, CLAP, etc.)
- Multimodal models (LLaVA, XCLIP, etc.)

## Hardware Support

These skills support multiple hardware platforms:

- CPU
- CUDA (NVIDIA)
- ROCm (AMD)
- MPS (Apple Silicon)
- OpenVINO (Intel)
- Qualcomm AI Engine
- WebNN (Browser)
- WebGPU (Browser)

Each skill detects available hardware and uses the optimal platform automatically.