# Mock Detection System for HuggingFace Model Tests

## Overview

The Mock Detection System provides clear visibility into whether HuggingFace model tests are running real inference with actual models or using mock objects for CI/CD testing. This transparency helps users and developers understand test results and differentiate between performance characteristics of real models versus mock implementations.

## Key Features

- **Visual Indicators**:
  - 🚀 Indicates REAL INFERENCE with actual models
  - 🔷 Indicates MOCK OBJECTS for CI/CD testing only

- **Dependency Tracking**:
  - `transformers`: Detection of the transformers library
  - `torch`: Detection of PyTorch
  - `tokenizers`: Detection of the tokenizers library
  - `sentencepiece`: Detection of the sentencepiece library

- **Metadata Enrichment**:
  - Test results are enriched with dependency status metadata
  - Clear indication of whether real inference or mocks were used
  - Full dependency reporting in both output and JSON results

- **Environment Variable Control**:
  - Force mocking of specific dependencies using environment variables
  - Simulate missing dependencies without modifying installed packages
  - Easily test different scenarios during development and in CI/CD

## Implementation

The system is implemented across all test files and templates:

1. **Environment Variable Control**:
   ```python
   # Check if we should mock specific dependencies
   MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
   MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
   MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
   MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'
   ```

2. **Dependency Checking**:
   ```python
   try:
       if MOCK_TORCH:
           raise ImportError("Mocked torch import failure")
       import torch
       HAS_TORCH = True
   except ImportError:
       torch = MagicMock()
       HAS_TORCH = False
       logger.warning("torch not available, using mock")
   ```

3. **Mock Detection Logic**:
   ```python
   using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
   using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
   ```

4. **Visual Indicators**:
   ```python
   if using_real_inference and not using_mocks:
       print(f"🚀 Using REAL INFERENCE with actual models")
   else:
       print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
       print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
   ```

5. **Metadata Enrichment**:
   ```python
   "metadata": {
       # ... existing fields ...
       "has_transformers": HAS_TRANSFORMERS,
       "has_torch": HAS_TORCH,
       "has_tokenizers": HAS_TOKENIZERS, 
       "has_sentencepiece": HAS_SENTENCEPIECE,
       "using_real_inference": using_real_inference,
       "using_mocks": using_mocks,
       "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
   }
   ```

## Verification

The system has been verified to work correctly across different dependency scenarios using the `verify_mock_detection.sh` script, which tests:

1. Full dependency environment (transformers, torch, tokenizers, sentencepiece)
2. Missing transformers
3. Missing torch
4. Missing tokenizers
5. Missing sentencepiece
6. Minimal environment (only core Python dependencies)

In all cases, the system correctly identifies and reports real inference vs. mock objects usage.

## Usage

### Running Tests with Mock Detection

The mock detection system is automatically integrated into all test files. When running tests, you'll see:

```
TEST RESULTS SUMMARY
====================

🚀 Using REAL INFERENCE with actual models

Model: bert-base-uncased
Device: cuda:0
```

or

```
TEST RESULTS SUMMARY
====================

🔷 Using MOCK OBJECTS for CI/CD testing only
   Dependencies: transformers=True, torch=False, tokenizers=True, sentencepiece=True

Model: bert-base-uncased
Device: cuda:0
```

### Forcing Mock Mode

You can force the use of mocks by setting environment variables:

```bash
# Mock PyTorch to test without GPU
MOCK_TORCH=true python test_hf_bert.py

# Mock transformers library
MOCK_TRANSFORMERS=true python test_hf_bert.py

# Mock multiple dependencies
MOCK_TORCH=true MOCK_TRANSFORMERS=true python test_hf_bert.py

# Mock all major dependencies for CI/CD testing
MOCK_TORCH=true MOCK_TRANSFORMERS=true MOCK_TOKENIZERS=true MOCK_SENTENCEPIECE=true python test_hf_bert.py
```

This is especially useful for:
- Testing in CI/CD environments without installing large dependencies
- Verifying your code handles missing dependencies gracefully
- Testing different scenarios without changing your environment

## Architecture-Specific Templates

The mock detection system is implemented in all architecture-specific templates:

1. `encoder_only_template.py` - For BERT, RoBERTa, etc.
2. `decoder_only_template.py` - For GPT-2, LLaMA, etc.
3. `encoder_decoder_template.py` - For T5, BART, etc.
4. `vision_template.py` - For ViT, Swin, etc.
5. `vision_text_template.py` - For CLIP, BLIP, etc.
6. `speech_template.py` - For Whisper, Wav2Vec2, etc.
7. `multimodal_template.py` - For LLaVA, etc.

## Benefits

- **Transparency**: Clear indication of when mocks are being used
- **Debugging Aid**: Helps identify missing dependencies
- **CI/CD Integration**: Prevents misinterpretation of mock results as real performance
- **Result Context**: Provides context for interpretation of test results
- **Metadata**: Enriches test results with detailed dependency information
- **Testing Flexibility**: Ability to force mock mode without uninstalling packages
- **Environment Simulation**: Simulate different dependency scenarios for comprehensive testing

## Implementation Verification

To verify the mock detection system is working correctly in your environment, run:

```bash
cd /path/to/repo/test
bash skills/verify_mock_detection.sh
```

This will test the system with various dependency scenarios and generate a summary report.

## Custom Testing

You can also create custom test configurations:

```bash
# Test with only torch missing
MOCK_TORCH=true python test_hf_bert.py

# Test with specific mock configurations
cd /path/to/repo/test
for model in bert gpt2 t5 vit; do
  echo "Testing $model with mocked torch"
  MOCK_TORCH=true python skills/fixed_tests/test_hf_$model.py
done
```

This provides flexibility for different testing scenarios without modifying your Python environment.