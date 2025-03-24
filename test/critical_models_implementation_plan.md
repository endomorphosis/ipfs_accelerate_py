# Implementation Plan for Remaining Critical Priority Models

Based on our analysis, we need to implement the following critical priority models that are currently missing:

1. **GPT-J (decoder-only)**
2. **Flan-T5 (encoder-decoder)**
3. **XLM-RoBERTa (encoder-only)**
4. **Vision-Text-Dual-Encoder (vision-text)**

## Implementation Approach

We'll use the established pattern for test generation with special handling for hyphenated names:

1. Create proper model definitions in a JSON structure
2. Use the test_toolkit.py to generate the tests
3. Verify syntax and functionality of the generated tests
4. Update the roadmap with the implementation progress

## Step-by-Step Process

### 1. Create Model Definitions

Create a file `critical_models.json` with the following content:

```json
{
  "decoder_only": [
    {
      "name": "gpt_j",
      "architecture": "decoder_only",
      "template": "gpt2",
      "default_model": "EleutherAI/gpt-j-6B",
      "task": "text-generation"
    }
  ],
  "encoder_decoder": [
    {
      "name": "flan_t5",
      "architecture": "encoder_decoder",
      "template": "t5",
      "default_model": "google/flan-t5-base",
      "task": "text2text-generation",
      "original_name": "flan-t5"
    }
  ],
  "encoder_only": [
    {
      "name": "xlm_roberta",
      "architecture": "encoder_only",
      "template": "bert",
      "default_model": "xlm-roberta-base",
      "task": "fill-mask",
      "original_name": "xlm-roberta"
    }
  ],
  "vision_text": [
    {
      "name": "vision_text_dual_encoder",
      "architecture": "vision_text",
      "template": "clip",
      "default_model": "clip-vit-base-patch32",
      "task": "image-classification",
      "original_name": "vision-text-dual-encoder"
    }
  ]
}
```

### 2. Generate and Verify the Tests

For each model, run:

```bash
python test_toolkit.py generate gpt_j --template gpt2
python test_toolkit.py generate flan_t5 --template t5
python test_toolkit.py generate xlm_roberta --template bert
python test_toolkit.py generate vision_text_dual_encoder --template clip
```

Verify syntax for all generated files:

```bash
python test_toolkit.py verify
```

### 3. Testing Generated Models

For each model, verify functionality:

```bash
python test_toolkit.py test gpt_j --cpu-only
python test_toolkit.py test flan_t5 --cpu-only
python test_toolkit.py test xlm_roberta --cpu-only
python test_toolkit.py test vision_text_dual_encoder --cpu-only
```

### 4. Update Roadmap Documentation

After successful implementation, update the roadmap:

```bash
# Generate updated coverage report
python test_toolkit.py coverage

# Update the roadmap with implementation status
python update_roadmap.py
```

## Timeline

Complete all four critical priority models by March 23, 2025.

