# Phase 3: Priority Models Testing & Baseline Establishment

## Overview

Phase 3 focuses on testing the top 10 priority HuggingFace models on available hardware platforms and establishing performance baselines. This is the critical validation phase before bulk deployment.

## Status: ✅ READY TO EXECUTE

All infrastructure is in place. We now need to run the tests and establish baselines.

---

## Priority Models (Top 10)

### 1. GPT-2 (Text Generation)
- **Model ID:** `gpt2`
- **Category:** Text
- **Use Case:** General text generation
- **Test File:** `test_hf_gpt2_improved.py`
- **Status:** ✅ Converted

### 2. CLIP (Multimodal)
- **Model ID:** `openai/clip-vit-base-patch32`
- **Category:** Multimodal (Vision + Text)
- **Use Case:** Image-text matching, zero-shot classification
- **Test File:** `test_hf_clip_improved.py`
- **Status:** ✅ Converted

### 3. LLaMA (Large Language Model)
- **Model ID:** `meta-llama/Llama-2-7b-hf`
- **Category:** Text
- **Use Case:** Advanced text generation, instruction following
- **Test File:** `test_hf_llama_improved.py`
- **Status:** ✅ Converted

### 4. Whisper (Audio Transcription)
- **Model ID:** `openai/whisper-tiny`
- **Category:** Audio
- **Use Case:** Speech-to-text transcription
- **Test File:** `test_hf_whisper_improved.py`
- **Status:** ✅ Converted

### 5. T5 (Text-to-Text)
- **Model ID:** `t5-small`
- **Category:** Text
- **Use Case:** Translation, summarization, Q&A
- **Test File:** `test_hf_t5_improved.py`
- **Status:** ✅ Converted

### 6. ViT (Vision Transformer)
- **Model ID:** `google/vit-base-patch16-224`
- **Category:** Vision
- **Use Case:** Image classification
- **Test File:** `test_hf_vit_improved.py`
- **Status:** ✅ Converted

### 7. BERT (Text Encoding)
- **Model ID:** `bert-base-uncased`
- **Category:** Text
- **Use Case:** Text classification, NER, Q&A
- **Test File:** `test_hf_bert_improved.py`
- **Status:** ✅ Converted

### 8. ResNet (Computer Vision)
- **Model ID:** `microsoft/resnet-50`
- **Category:** Vision
- **Use Case:** Image classification, feature extraction
- **Test File:** `test_hf_resnet_improved.py`
- **Status:** ✅ Converted

### 9. Wav2Vec2 (Audio Encoding)
- **Model ID:** `facebook/wav2vec2-base`
- **Category:** Audio
- **Use Case:** Audio feature extraction, ASR
- **Test File:** `test_hf_wav2vec2_improved.py`
- **Status:** ✅ Converted

### 10. BART (Seq2Seq)
- **Model ID:** `facebook/bart-base`
- **Category:** Text
- **Use Case:** Summarization, translation
- **Test File:** `test_hf_bart_improved.py`
- **Status:** ✅ Converted

---

## Hardware Platforms

### Supported Platforms

1. **CPU** - Always available
   - Testing: ✅ Enabled
   - Baseline: Planned

2. **CUDA (NVIDIA GPU)** - If available
   - Testing: ✅ Enabled
   - Baseline: Planned
   - Requirements: CUDA-compatible GPU, drivers

3. **MPS (Apple Silicon)** - If available
   - Testing: ✅ Enabled
   - Baseline: Planned
   - Requirements: Apple M1/M2/M3 chip

4. **ROCm (AMD GPU)** - If available
   - Testing: ⚠️ Partial support
   - Baseline: Future
   - Requirements: AMD GPU, ROCm drivers

5. **OpenVINO (Intel)** - If available
   - Testing: ⚠️ Partial support
   - Baseline: Future
   - Requirements: OpenVINO toolkit

### Hardware Detection

The test script automatically detects:
- PyTorch availability and version
- CUDA availability and device count
- MPS (Apple Silicon) availability
- TensorFlow availability (optional)

---

## Testing Script Usage

### Basic Usage

```bash
# Test all priority models on CPU
python scripts/test_priority_models.py

# Test specific models
python scripts/test_priority_models.py --models GPT-2 BERT CLIP

# Test on CUDA (if available)
python scripts/test_priority_models.py --hardware cuda

# Update performance baselines
python scripts/test_priority_models.py --update-baselines

# Save report to file
python scripts/test_priority_models.py --output reports/phase3_results.txt
```

### Advanced Usage

```bash
# Test all models on all hardware and establish baselines
python scripts/test_priority_models.py --hardware all --update-baselines

# Test specific models on CUDA with baseline update
python scripts/test_priority_models.py --models GPT-2 BERT --hardware cuda --update-baselines

# Generate comprehensive report
python scripts/test_priority_models.py --output reports/phase3_complete_$(date +%Y%m%d).txt
```

---

## Baseline Establishment

### What Are Performance Baselines?

Performance baselines are reference measurements that capture:
- **Inference time** (mean, median, min, max, std dev)
- **Memory usage** (allocated, reserved, peak)
- **Device type** (CPU, CUDA, MPS, etc.)
- **Timestamp** (when baseline was established)

### Why Establish Baselines?

1. **Regression Detection:** Catch performance degradation early
2. **Optimization Tracking:** Measure improvements over time
3. **Hardware Comparison:** Compare performance across platforms
4. **CI/CD Integration:** Automated performance testing

### Baseline Storage

Baselines are stored in `test/.performance_baselines.json`:

```json
{
  "GPT2_cpu": {
    "inference_time_mean": 0.0234,
    "inference_time_median": 0.0231,
    "inference_time_min": 0.0220,
    "inference_time_max": 0.0250,
    "inference_time_std": 0.0005,
    "memory_allocated_mb": 456.78,
    "memory_reserved_mb": 512.00,
    "memory_peak_mb": 478.90,
    "device": "cpu",
    "timestamp": "2026-02-02T02:00:00",
    "pytorch_version": "2.1.0"
  }
}
```

### Establishing Baselines

**Step 1:** Run tests with baseline update flag:
```bash
python scripts/test_priority_models.py --update-baselines
```

**Step 2:** Verify baselines were created:
```bash
ls -lh test/.performance_baselines.json
cat test/.performance_baselines.json | jq '.' | head -20
```

**Step 3:** Run tests again to verify regression detection:
```bash
python scripts/test_priority_models.py
```

**Step 4:** Review any performance warnings in output

---

## Expected Results

### Test Execution Time

| Hardware | Expected Duration | Models Tested |
|----------|------------------|---------------|
| CPU Only | 5-10 minutes | All 10 |
| CPU + CUDA | 8-15 minutes | All 10 × 2 |
| CPU + MPS | 8-15 minutes | All 10 × 2 |

### Success Criteria

✅ **Phase 3 Complete When:**
1. All 10 priority models pass tests on CPU
2. Tests pass on at least 1 accelerator (if available)
3. Performance baselines established for all models
4. Comprehensive test report generated
5. No critical failures detected

### Possible Outcomes

**PASS** ✅
- Model loads successfully
- Inference completes without errors
- Output shapes are correct
- No NaN/Inf values in outputs

**FAIL** ❌
- Model fails to load
- Inference throws exception
- Output shapes incorrect
- NaN/Inf values detected

**SKIP** ⏭️
- Test file not found
- Dependencies missing
- Hardware not available

**TIMEOUT** ⏱️
- Test takes longer than 5 minutes
- May indicate performance issue

**ERROR** ⚠️
- Unexpected error occurred
- Script or pytest issue

---

## Hardware Compatibility Matrix

| Model | CPU | CUDA | MPS | ROCm | OpenVINO |
|-------|-----|------|-----|------|----------|
| GPT-2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| CLIP | ✅ | ✅ | ✅ | ✅ | ✅ |
| LLaMA | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Whisper | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| T5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| ViT | ✅ | ✅ | ✅ | ✅ | ✅ |
| BERT | ✅ | ✅ | ✅ | ✅ | ✅ |
| ResNet | ✅ | ✅ | ✅ | ✅ | ✅ |
| Wav2Vec2 | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| BART | ✅ | ✅ | ✅ | ✅ | ✅ |

Legend: ✅ Full support | ⚠️ Partial support | ❌ Not supported

---

## Report Format

The test script generates a comprehensive report:

```
================================================================================
PHASE 3 PRIORITY MODELS TEST REPORT
================================================================================
Date: 2026-02-02 02:00:00

HARDWARE INFORMATION
--------------------------------------------------------------------------------
CPU: Available
PyTorch: 2.1.0
CUDA: Available (1 device(s))
  Device: NVIDIA Tesla T4
MPS (Apple Silicon): Not available

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
  BART: 2 baseline(s)
  BERT: 2 baseline(s)
  CLIP: 2 baseline(s)
  GPT2: 2 baseline(s)
  LLaMA: 2 baseline(s)
  ResNet: 2 baseline(s)
  T5: 2 baseline(s)
  ViT: 2 baseline(s)
  Wav2Vec2: 2 baseline(s)
  Whisper: 2 baseline(s)

================================================================================
```

---

## Troubleshooting

### Common Issues

**Issue:** PyTorch not installed
```
WARNING: PyTorch not installed, cannot detect GPU hardware
```
**Solution:** Install PyTorch:
```bash
pip install torch torchvision torchaudio
```

**Issue:** CUDA not available
```
CUDA: Not available
```
**Solution:** 
1. Check GPU with `nvidia-smi`
2. Install CUDA-compatible PyTorch
3. Verify CUDA drivers

**Issue:** Test timeout
```
⏱️ TIMEOUT - Test timed out after 5 minutes
```
**Solution:**
1. Model may be too large for hardware
2. Try smaller model variant
3. Increase timeout in script

**Issue:** Model not found
```
⏭️ SKIP - Test file not found
```
**Solution:**
1. Verify test file exists: `ls test/improved/test_hf_*_improved.py`
2. Run conversion script if needed
3. Check model name spelling

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Phase 3 Priority Models

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-priority-models:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Test priority models
        run: |
          python scripts/test_priority_models.py --output reports/phase3.txt
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: phase3-report
          path: reports/phase3.txt
```

---

## Next Steps After Phase 3

Once Phase 3 is complete:

1. **Review Results:** Analyze test reports and baseline data
2. **Document Issues:** Log any failures or performance concerns
3. **Proceed to Phase 4:** Bulk conversion of remaining tests
4. **Enable CI/CD:** Integrate priority model tests into CI pipeline
5. **Monitor Baselines:** Track performance trends over time

---

## Phase 3 Checklist

### Pre-Flight
- [x] Test infrastructure complete
- [x] Priority models converted
- [x] Testing script created
- [x] Documentation written

### Execution
- [ ] Detect available hardware
- [ ] Run tests on CPU
- [ ] Run tests on accelerators (if available)
- [ ] Establish performance baselines
- [ ] Generate test report

### Validation
- [ ] All 10 models pass on CPU
- [ ] At least 8/10 models pass on accelerators
- [ ] Baselines stored correctly
- [ ] No critical failures
- [ ] Report is comprehensive

### Documentation
- [ ] Update Phase 3 status
- [ ] Document hardware compatibility
- [ ] Record baseline values
- [ ] Note any issues or limitations

---

## Success Metrics

**Target Goals:**
- ✅ 100% of priority models have tests
- ✅ 90%+ pass rate on CPU
- ✅ 80%+ pass rate on accelerators
- ✅ Baselines established for all passing tests
- ✅ Complete documentation

**Key Performance Indicators:**
- Test execution time < 15 minutes
- Zero critical failures
- Performance within expected ranges
- Clear path to Phase 4

---

## References

- [HF Model Testing Review](../summaries/HF_MODEL_TESTING_REVIEW.md)
- [Testing Improvement Summary](../summaries/HF_TESTING_IMPROVEMENT_SUMMARY.md)
- [Model Test Gating Guide](../guides/testing/MODEL_TEST_GATING_GUIDE.md)
- [Conversion Complete Summary](../summaries/CONVERSION_COMPLETE_SUMMARY.md)

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-02  
**Status:** Ready for execution
