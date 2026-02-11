# Phase 3 Setup and Execution Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install Phase 3 testing requirements
pip install -r requirements-phase3.txt

# Or install minimal requirements
pip install pytest pytest-cov transformers torch
```

### 2. Run Priority Model Tests

```bash
# Test all priority models (CPU only, fast check)
python scripts/test_priority_models.py

# Test with baseline establishment
python scripts/test_priority_models.py --update-baselines
```

### 3. View Results

Results are printed to stdout or saved to a file with `--output`.

---

## Detailed Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB+ recommended)
- 10GB+ disk space for models

### Hardware-Specific Setup

#### CPU Only
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-phase3.txt
```

#### NVIDIA GPU (CUDA 11.8)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-phase3.txt
```

#### NVIDIA GPU (CUDA 12.1)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-phase3.txt
```

#### Apple Silicon (M1/M2/M3)
```bash
pip install torch torchvision torchaudio
pip install -r requirements-phase3.txt
```

### Verify Installation

```bash
# Check PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check pytest
pytest --version
```

---

## Running Tests

### Basic Commands

```bash
# Test all 10 priority models
python scripts/test_priority_models.py

# Test specific models
python scripts/test_priority_models.py --models GPT-2 BERT

# Test on specific hardware
python scripts/test_priority_models.py --hardware cuda

# Update baselines
python scripts/test_priority_models.py --update-baselines

# Save report to file
python scripts/test_priority_models.py --output reports/phase3_$(date +%Y%m%d_%H%M%S).txt
```

### Advanced Usage

```bash
# Test everything and establish baselines
python scripts/test_priority_models.py --hardware all --update-baselines --output reports/phase3_complete.txt

# Test only text models
python scripts/test_priority_models.py --models GPT-2 BERT T5 BART

# Test only vision models
python scripts/test_priority_models.py --models ViT ResNet CLIP

# Test only audio models
python scripts/test_priority_models.py --models Whisper Wav2Vec2
```

### Using pytest Directly

```bash
# Run individual model test
pytest test/improved/test_hf_gpt2_improved.py --run-model-tests -v

# Run with coverage
pytest test/improved/test_hf_bert_improved.py --run-model-tests --cov=ipfs_accelerate_py -v

# Run all priority model tests
pytest test/improved/test_hf_{gpt2,clip,llama,whisper,t5,vit,bert,resnet,wav2vec2,bart}_improved.py --run-model-tests -v
```

---

## Expected Output

### Successful Run

```
🚀 Phase 3: Testing Priority Models
================================================================================

🔍 Detecting hardware...
   CPU: Available
   PyTorch: 2.1.0
   CUDA: 1 device(s)

📋 Testing 10 priority model(s)...

[1/10] Testing GPT-2... ✅ PASS
[2/10] Testing CLIP... ✅ PASS
[3/10] Testing LLaMA... ✅ PASS
[4/10] Testing Whisper... ✅ PASS
[5/10] Testing T5... ✅ PASS
[6/10] Testing ViT... ✅ PASS
[7/10] Testing BERT... ✅ PASS
[8/10] Testing ResNet... ✅ PASS
[9/10] Testing Wav2Vec2... ✅ PASS
[10/10] Testing BART... ✅ PASS

================================================================================
PHASE 3 PRIORITY MODELS TEST REPORT
================================================================================
Date: 2026-02-02 02:30:00

HARDWARE INFORMATION
--------------------------------------------------------------------------------
CPU: Available
PyTorch: 2.1.0
CUDA: Available (1 device(s))
  Device: NVIDIA Tesla T4

TEST RESULTS
--------------------------------------------------------------------------------
✅ GPT-2: PASS
✅ CLIP: PASS
✅ LLaMA: PASS
✅ Whisper: PASS
✅ T5: PASS
✅ ViT: PASS
✅ BERT: PASS
✅ ResNet: PASS
✅ Wav2Vec2: PASS
✅ BART: PASS

Summary: 10 passed, 0 failed, 0 skipped, 0 errors

PERFORMANCE BASELINES
--------------------------------------------------------------------------------
Total baselines stored: 20
  GPT2: 2 baseline(s)
  CLIP: 2 baseline(s)
  ...

================================================================================
```

### With Failures

```
[3/10] Testing LLaMA... ❌ FAIL
   Reason: Model file not found
```

### With Skips

```
[8/10] Testing ResNet... ⏭️ SKIP
   Reason: Test file not found
```

---

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch
```

### Issue: "No module named 'transformers'"

**Solution:**
```bash
pip install transformers
```

### Issue: "No module named 'pytest'"

**Solution:**
```bash
pip install pytest
```

### Issue: CUDA out of memory

**Solution:**
```bash
# Test on CPU instead
python scripts/test_priority_models.py --hardware cpu
```

### Issue: Test timeout

**Solution:**
- Models may be too large for available hardware
- Try smaller variants or test on CPU only
- Increase timeout in `scripts/test_priority_models.py` (line with `timeout=300`)

### Issue: Model download fails

**Solution:**
```bash
# Pre-download models manually
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

### Issue: Permission denied

**Solution:**
```bash
chmod +x scripts/test_priority_models.py
```

---

## Performance Baselines

### What Gets Measured

For each model and hardware combination:
- **Inference time:** Mean, median, min, max, std deviation
- **Memory usage:** Allocated, reserved, peak
- **Device type:** CPU, CUDA, MPS, etc.
- **Timestamp:** When baseline was established

### Viewing Baselines

```bash
# View all baselines
cat test/.performance_baselines.json | jq '.'

# Count baselines
cat test/.performance_baselines.json | jq 'keys | length'

# View specific model
cat test/.performance_baselines.json | jq '.GPT2_cpu'
```

### Updating Baselines

```bash
# Update all baselines
python scripts/test_priority_models.py --update-baselines

# Update specific models
python scripts/test_priority_models.py --models GPT-2 BERT --update-baselines

# Update for specific hardware
python scripts/test_priority_models.py --hardware cuda --update-baselines
```

---

## Integration with Development Workflow

### Daily Development

```bash
# Quick check of framework (skips model tests)
pytest

# Test specific model you're working on
pytest test/improved/test_hf_gpt2_improved.py --run-model-tests -v
```

### Before Committing

```bash
# Run priority model tests
python scripts/test_priority_models.py

# Check for regressions
pytest --run-model-tests -k "gpt2 or bert"
```

### CI/CD Pipeline

```bash
# In CI, run priority models as smoke test
python scripts/test_priority_models.py --hardware cpu --output ci_report.txt

# Exit with error code if any tests fail
# (script already does this automatically)
```

---

## Next Steps

After completing Phase 3:

1. **Review the report:** Check for any failures or issues
2. **Document results:** Update `docs/PHASE3_PRIORITY_MODELS.md` with actual results
3. **Proceed to Phase 4:** Bulk testing of remaining models
4. **Enable CI/CD:** Add priority model tests to CI pipeline
5. **Monitor baselines:** Track performance over time

---

## Getting Help

### Common Commands

```bash
# Help for test script
python scripts/test_priority_models.py --help

# List available improved tests
ls test/improved/*_improved.py | wc -l

# Check pytest configuration
pytest --collect-only test/improved/test_hf_bert_improved.py

# Validate baselines file
python3 -c "import json; print(json.load(open('test/.performance_baselines.json')))" 2>&1 | head -20
```

### Documentation

- `docs/development_history/PHASE3_PRIORITY_MODELS.md` - This guide
- `docs/guides/testing/MODEL_TEST_GATING_GUIDE.md` - Test usage
- `docs/summaries/HF_MODEL_TESTING_REVIEW.md` - Architecture
- `test/improved/README.md` - Test structure

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-02  
**Status:** Ready to execute
