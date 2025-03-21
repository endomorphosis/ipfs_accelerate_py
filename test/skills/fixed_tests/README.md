# Fixed Tests for HuggingFace Models

This directory contains test files that have been regenerated with fixes for:

1. Hyphenated model names (e.g. "gpt-j" → "gpt_j")
2. Capitalization issues in class names (e.g. "GPTJForCausalLM" vs "GptjForCausalLM")
3. Syntax errors like unterminated string literals
4. Indentation issues
5. Consistent mock detection across all test files

The test files in this directory are generated using the updated test generator
that handles hyphenated model names correctly. The generator now:

1. Automatically converts hyphenated model names to valid Python identifiers
2. Ensures proper capitalization patterns for class names
3. Validates that generated files have valid Python syntax
4. Fixes common syntax errors like unterminated string literals
5. Adds proper mock detection for CI/CD environments

## Example Models with Hyphenated Names

- chinese-clip → test_hf_chinese_clip.py
- data2vec-audio → test_hf_data2vec_audio.py
- data2vec-text → test_hf_data2vec_text.py
- data2vec-vision → test_hf_data2vec_vision.py
- gpt-j → test_hf_gpt_j.py
- gpt-neo → test_hf_gpt_neo.py
- gpt-neox → test_hf_gpt_neox.py
- speech-to-text → test_hf_speech_to_text.py
- speech-to-text-2 → test_hf_speech_to_text_2.py
- trocr-base → test_hf_trocr_base.py

## Running the Tests

Tests can be run individually with:

```bash
python fixed_tests/test_hf_gpt_j.py --list-models
python fixed_tests/test_hf_xlm_roberta.py --list-models
```

To run all tests:

```bash
cd fixed_tests
for test in test_hf_*.py; do python $test --list-models; done
```

## Validation

All test files in this directory have been validated to ensure:

1. Valid Python syntax
2. Proper indentation
3. Correct class naming patterns
4. Valid Python identifiers for hyphenated model names
5. Consistent mock object detection for CI/CD environments
