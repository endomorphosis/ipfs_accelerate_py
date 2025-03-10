# Generated Tests

This directory contains Python test files that have been generated from templates for testing various AI models.

## Test Types

These test files verify:

1. Model loading and initialization
2. Inference with sample inputs
3. Cross-platform hardware compatibility
4. Integration with various hardware backends

## Running Tests

You can run these tests directly with Python:

```bash
# Run a specific test
python test_hf_bert_base_uncased.py

# Run all tests
python -m unittest discover -p "test_*.py"
```

## Test Organization

Tests are organized by model type:

- Text embedding models (BERT, etc.)
- Text generation models (T5, LLAMA, etc.)
- Vision models (ViT, CLIP, etc.)
- Audio models (Whisper, Wav2Vec2, etc.)
- Multimodal models (LLaVA, etc.)

## Hardware Support

These tests are designed to work across multiple hardware platforms:

- CPU
- CUDA (NVIDIA)
- ROCm (AMD)
- MPS (Apple Silicon)
- OpenVINO (Intel)
- Qualcomm AI Engine
- WebNN (Browser)
- WebGPU (Browser)