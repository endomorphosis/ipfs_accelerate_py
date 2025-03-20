# Fixed HuggingFace Test Files

This directory contains fixed versions of HuggingFace test files with proper indentation and architecture-specific implementations.

## Structure

- `*.py` - Fixed test files with proper indentation
- `*.py.fixed` - Alternative fixed versions (if available)
- `collected_results/` - Test results in JSON format

## Core Model Tests

The following core model tests have been fixed:

1. `test_hf_bert.py` - Encoder-only architecture
2. `test_hf_gpt2.py` - Decoder-only architecture
3. `test_hf_t5.py` - Encoder-decoder architecture
4. `test_hf_vit.py` - Vision architecture

## Architecture Types

Each test file implements architecture-specific handling:

- **Encoder-Only** (BERT):
  - Bidirectional attention
  - Mask token handling
  - Encoder-only model initialization

- **Decoder-Only** (GPT-2):
  - Autoregressive behavior
  - Padding token handling
  - Decoder-only initialization

- **Encoder-Decoder** (T5):
  - Separate encoder and decoder
  - Decoder input initialization
  - Translation task handling

- **Vision** (ViT):
  - Image tensor shape handling
  - Vision-specific preprocessing
  - Classification task implementation

## Running Tests

Run a specific test:

```bash
python test_hf_model.py
```

Run with a specific model:

```bash
python test_hf_model.py --model "model-id"
```

Test with all hardware backends:

```bash
python test_hf_model.py --all-hardware
```

Save results to JSON:

```bash
python test_hf_model.py --save
```

## Collected Results

The `collected_results/` directory contains JSON files with test results:

- Performance metrics (load time, inference time)
- Hardware information
- Success/failure status
- Error information (if any)

## Notes

These fixed test files maintain the core functionality of the original tests while addressing indentation issues and implementing architecture-specific handling to ensure proper Python syntax and execution.