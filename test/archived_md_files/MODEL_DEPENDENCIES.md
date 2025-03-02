# Hugging Face Model Dependencies

This file tracks external dependencies and remote code requirements for different model types.
Models are categorized by their dependency needs and any special installation requirements.

## Models with Remote Code Requirements

The following models require `trust_remote_code=True` when loading with Transformers:

| Model | Reason for Remote Code | Special Notes |
|-------|------------------------|---------------|
| `Salesforce/blip-image-captioning-base` | Unknown reason | None |
| `facebook/sam-vit-base` | Unknown reason | None |
| `google/gemma-2b` | Quantized model often requires special handlers | Model may require special quantization libraries |
| `llava-hf/llava-1.5-7b-hf` | Unknown reason | None |
| `meta-llama/Llama-2-7b-hf` | LLM architecture 'meta-llama/Llama-2-7b-hf' often requires remote code | None |
| `mistralai/Mistral-7B-v0.1` | LLM architecture 'mistralai/Mistral-7B-v0.1' often requires remote code | None |

## Dependency Matrix

This table shows which dependencies are required by each model:

| Model | `Pillow` | `accelerate` | `accelerate` | `accelerate` | `accelerate` | `datasets` | `einops` | `evaluate` | `jiwer` | `librosa` | `matplotlib` | `numpy` | `opencv-python` | `pdf2image` | `pytesseract` | `regex` | `safetensors` | `scipy` | `sentencepiece` | `soundfile` | `timm` | `timm` | `tokenizers` | `tokenizers` | `tokenizers` | `torchvision` |
|-------|--------|------------|------------|------------|------------|----------|--------|----------|-------|---------|------------|-------|---------------|-----------|-------------|-------|-------------|-------|---------------|-----------|------|------|------------|------------|------------|-------------|
| `Salesforce/blip-image-captioning-base` | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |
| `bert-base-uncased` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |  |  |  | ✅ 0.11.0 | ✅ 0.11.0 | ✅ 0.11.0 |  |
| `distilroberta-base` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |  |  |  | ✅ 0.11.0 | ✅ 0.11.0 | ✅ 0.11.0 |  |
| `facebook/detr-resnet-50` |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |  |  |  |  | ✅ |  |  | ✅ | ✅ |  |  |  |  |
| `facebook/sam-vit-base` | ✅ |  |  |  |  |  |  |  |  |  | ✅ |  | ✅ |  |  |  |  |  |  |  | ✅ 0.9.2 | ✅ 0.9.2 |  |  |  |  |
| `facebook/wav2vec2-base-960h` |  |  |  |  |  | ✅ 2.14.0 |  |  | ✅ | ✅ |  |  |  |  |  |  |  |  |  | ✅ |  |  |  |  |  |  |
| `google/gemma-2b` |  | ✅ 0.21.0 | ✅ 0.21.0 | ✅ 0.21.0 | ✅ 0.21.0 |  |  |  |  |  |  |  |  |  |  |  | ✅ 0.3.2 |  | ✅ |  |  |  |  |  |  |  |
| `google/vit-base-patch16-224` | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ 0.9.2 | ✅ 0.9.2 |  |  |  |  |
| `gpt2` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |  |  |  |  |  |  |  |  |  |  |
| `llava-hf/llava-1.5-7b-hf` | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |
| `meta-llama/Llama-2-7b-hf` |  | ✅ 0.20.3 | ✅ 0.20.3 | ✅ 0.20.3 | ✅ 0.20.3 |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |  |  |  | ✅ 0.13.3 | ✅ 0.13.3 | ✅ 0.13.3 |  |
| `microsoft/layoutlm-base-uncased` | ✅ |  |  |  |  |  |  |  |  |  |  |  | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |
| `mistralai/Mistral-7B-v0.1` |  | ✅ 0.18.0 | ✅ 0.18.0 | ✅ 0.18.0 | ✅ 0.18.0 |  | ✅ |  |  |  |  |  |  |  |  |  | ✅ 0.3.2 |  |  |  |  |  |  |  |  |  |
| `openai/whisper-tiny` |  |  |  |  |  |  |  | ✅ | ✅ | ✅ |  | ✅ 1.20.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `t5-small` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✅ |  |  |  | ✅ | ✅ | ✅ |  |

## Installation Guide

### Common Dependency Groups

#### Group 1: 2 models

Models in this group:
- `bert-base-uncased`
- `distilroberta-base`

Required dependencies:
- `sentencepiece`
- `tokenizers>=0.11.0`

```bash
pip install "sentencepiece" "tokenizers>=0.11.0"
```

#### Group 2: 1 models

Models in this group:
- `Salesforce/blip-image-captioning-base` (requires remote code)

Required dependencies:
- `Pillow`
- `torchvision`

```bash
pip install "Pillow" "torchvision"
```

#### Group 3: 1 models

Models in this group:
- `facebook/detr-resnet-50`

Required dependencies:
- `opencv-python`
- `scipy`
- `timm`

```bash
pip install "opencv-python" "scipy" "timm"
```

#### Group 4: 1 models

Models in this group:
- `facebook/sam-vit-base` (requires remote code)

Required dependencies:
- `Pillow`
- `matplotlib`
- `opencv-python`
- `timm>=0.9.2`

```bash
pip install "Pillow" "matplotlib" "opencv-python" "timm>=0.9.2"
```

#### Group 5: 1 models

Models in this group:
- `facebook/wav2vec2-base-960h`

Required dependencies:
- `datasets>=2.14.0`
- `jiwer`
- `librosa`
- `soundfile`

```bash
pip install "datasets>=2.14.0" "jiwer" "librosa" "soundfile"
```

#### Group 6: 1 models

Models in this group:
- `google/gemma-2b` (requires remote code)

Required dependencies:
- `accelerate>=0.21.0`
- `safetensors>=0.3.2`
- `sentencepiece`

```bash
pip install "accelerate>=0.21.0" "safetensors>=0.3.2" "sentencepiece"
```

#### Group 7: 1 models

Models in this group:
- `google/vit-base-patch16-224`

Required dependencies:
- `Pillow`
- `timm>=0.9.2`

```bash
pip install "Pillow" "timm>=0.9.2"
```

#### Group 8: 1 models

Models in this group:
- `gpt2`

Required dependencies:
- `regex`

```bash
pip install "regex"
```

#### Group 9: 1 models

Models in this group:
- `llava-hf/llava-1.5-7b-hf` (requires remote code)

Required dependencies:
- `Pillow`
- `accelerate`
- `matplotlib`
- `torchvision`

```bash
pip install "Pillow" "accelerate" "matplotlib" "torchvision"
```

#### Group 10: 1 models

Models in this group:
- `meta-llama/Llama-2-7b-hf` (requires remote code)

Required dependencies:
- `accelerate>=0.20.3`
- `sentencepiece`
- `tokenizers>=0.13.3`

```bash
pip install "accelerate>=0.20.3" "sentencepiece" "tokenizers>=0.13.3"
```

#### Group 11: 1 models

Models in this group:
- `microsoft/layoutlm-base-uncased`

Required dependencies:
- `Pillow`
- `opencv-python`
- `pdf2image`
- `pytesseract`

```bash
pip install "Pillow" "opencv-python" "pdf2image" "pytesseract"
```

#### Group 12: 1 models

Models in this group:
- `mistralai/Mistral-7B-v0.1` (requires remote code)

Required dependencies:
- `accelerate>=0.18.0`
- `einops`
- `safetensors>=0.3.2`

```bash
pip install "accelerate>=0.18.0" "einops" "safetensors>=0.3.2"
```

#### Group 13: 1 models

Models in this group:
- `openai/whisper-tiny`

Required dependencies:
- `evaluate`
- `jiwer`
- `librosa`
- `numpy>=1.20.0`

```bash
pip install "evaluate" "jiwer" "librosa" "numpy>=1.20.0"
```

#### Group 14: 1 models

Models in this group:
- `t5-small`

Required dependencies:
- `sentencepiece`
- `tokenizers`

```bash
pip install "sentencepiece" "tokenizers"
```

