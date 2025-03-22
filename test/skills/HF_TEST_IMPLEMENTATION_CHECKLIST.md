# HuggingFace Test Implementation Checklist

This document outlines the steps and requirements for implementing test files for all HuggingFace models, with a focus on correctly handling hyphenated model names.

> **HIGH PRIORITY OBJECTIVE:** Achieving 100% test coverage for all 300+ HuggingFace model classes with validated end-to-end testing is a high priority target. The current coverage of 57.6% (114/198 tracked models) must be improved to ensure comprehensive compatibility for all model architectures.

## Current Coverage Status

- **Total Models Tracked:** 198
- **Implemented Models:** 114 (57.6%)
- **Missing Models:** 84 (42.4%)
- **End-to-End Validated Models:** Limited subset with known compatibility

> **Target:** 100% test coverage for all 300+ HuggingFace model classes with validated end-to-end testing

## Implementation Priorities

### Tier 1: Critical Models (Highest Priority)
- [x] BERT (encoder-only) ✓
- [x] GPT-2 (decoder-only) ✓
- [x] T5 (encoder-decoder) ✓
- [x] BART (encoder-decoder) ✓
- [x] RoBERTa (encoder-only) ✓
- [x] GPT-J (decoder-only) ✓
- [x] GPT-Neo (decoder-only) ✓
- [x] DistilBERT (encoder-only) ✓
- [x] Electra (encoder-only) ✓
- [x] XLM-RoBERTa (encoder-only) ✓
- [x] Swin (vision) ✓
- [x] ViT (vision) ✓
- [x] CLIP (vision-text) ✓
- [x] Llama (decoder-only) ✓
- [x] Wav2Vec2 (speech) ✓

### Tier 2: High Priority Models
- [x] OPT (decoder-only) ✓
- [x] BLOOM (decoder-only) ✓
- [x] DeiT (vision) ✓
- [x] BLIP (vision-text) ✓
- [x] Whisper (speech) ✓
- [x] MT5 (encoder-decoder) ✓
- [x] MBART (encoder-decoder) ✓
- [x] LED (encoder-decoder) ✓
- [x] DETR (vision) ✓
- [x] Gemma (decoder-only) ✓
- [ ] CodeLLama (decoder-only) ❌
- [ ] Qwen2 (decoder-only) ❌
- [ ] Qwen3 (decoder-only) ❌
- [ ] LongT5 (encoder-decoder) ❌
- [ ] Pegasus-X (encoder-decoder) ❌
- [ ] Luke (encoder-only) ❌
- [ ] MPNet (encoder-only) ❌
- [ ] Fuyu (multimodal) ❌
- [ ] Kosmos-2 (multimodal) ❌
- [ ] LLaVA-Next (multimodal) ❌
- [ ] Video-LLaVA (multimodal) ❌
- [ ] Bark (speech) ❌
- [ ] MobileNet-v2 (vision) ❌
- [ ] BLIP-2 (vision-text) ❌
- [ ] ChineseCLIP (vision-text) ❌
- [ ] CLIPSeg (vision-text) ❌

### Tier 3: Medium Priority Models (Subset of 62 remaining models)
- [x] Albert (encoder-only) ✓
- [x] Pegasus (encoder-decoder) ✓
- [x] ConvNeXT (vision) ✓
- [x] BEiT (vision) ✓
- [x] ProphetNet (encoder-decoder) ✓
- [x] RemBERT (encoder-only) ✓
- [x] ERNIE (encoder-only) ✓
- [x] GPT-NeoX (decoder-only) ✓
- [x] DinoV2 (vision) ✓
- [x] SegFormer (vision) ✓
- [ ] CodeGen (decoder-only) ❌
- [ ] Command-R (decoder-only) ❌
- [ ] Mamba (decoder-only) ❌
- [ ] Mistral-Next (decoder-only) ❌
- [ ] Phi3 (decoder-only) ❌
- [ ] M2M-100 (encoder-decoder) ❌
- [ ] Seamless-M4T (encoder-decoder) ❌
- [ ] Data2Vec-Text (encoder-only) ❌
- [ ] MobileNet-V1 (vision) ❌
- [ ] Instructblip (vision-text) ❌
- [ ] Vision-Encoder-Decoder (vision-text) ❌

> **Note:** For the complete list of all 84 missing models, refer to the most recent coverage report in `/reports/missing_models.md`

## Implementation Steps

### 1. Set Up Test Infrastructure
- [x] Create architecture-specific templates
- [x] Implement model discovery using transformers introspection
- [x] Implement hyphenated name handling
- [x] Create test file generator with proper template selection

### 2. Implement Validation and Consistency Checks
- [x] Create syntax validation script
- [x] Implement consistency checking for hyphenated model names
- [x] Create test coverage report generator

### 3. Generate Test Files for All Models
- [x] Generate test files for Tier 1 models
- [x] Generate test files for Tier 2 models
- [x] Generate test files for Tier 3 models

### 4. Test and Verify
- [x] Run syntax validation on all generated files
- [x] Fix any inconsistencies in hyphenated model names
- [x] Verify that all files can be imported properly

### 5. Documentation
- [x] Create implementation summary
- [x] Document architecture-specific templates
- [x] Document hyphenated model handling

## Hyphenated Model Handling

Special care is required for models with hyphenated names, which need conversion to valid Python identifiers:

| Original Name | Python Identifier | Class Name Format |
|---------------|-------------------|-------------------|
| gpt-j         | gpt_j             | GPTJ              |
| gpt-neo       | gpt_neo           | GPTNeo            |
| gpt-neox      | gpt_neox          | GPTNeoX           |
| xlm-roberta   | xlm_roberta       | XLMRoBERTa        |

## Gotchas and Special Cases

1. **Class Name Capitalization**: Different models have different capitalization patterns for their class names:
   - BERT → BertForMaskedLM
   - GPT-2 → GPT2LMHeadModel
   - XLM-RoBERTa → XLMRoBERTaForMaskedLM

2. **Registry Naming**: Registry variables must be valid Python identifiers:
   ```python
   # Incorrect
   GPT-J_MODELS_REGISTRY = { ... }
   
   # Correct
   GPT_J_MODELS_REGISTRY = { ... }
   ```

3. **Imports and References**: Class references must not contain hyphens:
   ```python
   # Incorrect
   model_class = transformers.GPT-JForCausalLM
   
   # Correct
   model_class = transformers.GPTJForCausalLM
   ```

4. **Test Class Names**: Class declarations must not contain hyphens:
   ```python
   # Incorrect
   class TestGPT-JModels:
   
   # Correct
   class TestGPTJModels:
   ```

## Transformers Documentation Reference

The official Transformers documentation is an essential reference for implementing accurate model tests:

- **Documentation Build Folder**: `/home/barberb/ipfs_accelerate_py/test/doc-builder/build`
- **API Reference**: `/home/barberb/ipfs_accelerate_py/test/doc-builder/build/api`
- **Model Documentation**: `/home/barberb/ipfs_accelerate_py/test/doc-builder/build/main_classes/model`

## Implementation Best Practices

### 1. Modify Generators, Not Generated Files

> **CRITICAL:** Always modify the test generators rather than the generated test files. Directly editing generated files will lead to inconsistencies and code debt when generators are run again.

- ✅ Modify `test_generator_fixed.py` to improve template for all models
- ✅ Update template files in the `templates/` directory 
- ❌ Directly edit individual test files in `fixed_tests/` directory
- ❌ Create one-off fixes that bypass the generator infrastructure

### 2. Integrate Fixes Into Core Generators

- ✅ Push fixes upstream into `templates` and `generators`
- ✅ Ensure hyphenated model name handling is in the core generator
- ✅ Update model detection and architecture mapping in the generator
- ❌ Use separate scripts for batch fixing that don't update generators

### 3. Testing Strategy

- ✅ Implement tests through the standard generation process
- ✅ Validate all test files through the validation pipeline
- ✅ Run end-to-end testing on a representative sample
- ✅ Document any model-specific considerations in the generator

## Scripts and Utilities

1. **find_models.py**: Discover model classes and their architectures
2. **fix_indentation_and_apply_template.py**: Create test files with proper templates
3. **generate_hyphenated_tests.py**: Generate test files for hyphenated models
4. **check_test_consistency.py**: Verify test file consistency
5. **create_coverage_report.py**: Generate test coverage report
6. **test_generator_fixed.py**: Core test file generator (modify this, not the outputs)
7. **validate_model_tests.py**: Validate test files for syntax, structure, and configuration

## Running the Tests

```bash
# Run a specific test
python fixed_tests/test_hf_gpt_j.py --list-models

# Run all tests
cd fixed_tests
for test in test_hf_*.py; do python $test --list-models; done
```